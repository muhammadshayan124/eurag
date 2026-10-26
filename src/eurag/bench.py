"""Run every retriever x chunking over the question set and keep per-question metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from eurag import corpus, questions, retrieve


def run(data_dir: Path, cache_dir: Path, log=print) -> dict:
    arts = corpus.load(data_dir)
    qs = questions.load()
    problems = questions.verify(qs, arts)
    if problems:
        raise ValueError("question labels are inconsistent with the corpus:\n" + "\n".join(problems))
    amb = questions.ambiguity(qs, arts)
    out: dict = {"n_questions": len(qs), "ambiguous_ids": sorted(amb), "configs": {}, "latency_ms": {}}
    for mode in ("article", "paragraph"):
        chunks = corpus.chunk(arts, mode)
        bm25 = retrieve.BM25(chunks)
        dense = retrieve.Dense(chunks, cache_dir)
        retrievers = {"bm25": bm25, "dense": dense, "hybrid": retrieve.Hybrid(bm25, dense)}
        act_of = np.array([c.article.split(":")[0] for c in chunks])
        for name, r in retrievers.items():
            per_q, times, tops, filtered = [], [], [], []
            for q in qs:
                t0 = time.perf_counter()
                scores = r.scores(q.question)
                ranking = retrieve.article_ranking(scores, chunks)
                oracle = np.where(act_of == q.gold[0].split(":")[0], scores, -np.inf)  # as if the user had said which act they mean
                filtered.append(retrieve.metrics_for(retrieve.article_ranking(oracle, chunks), q.gold))
                times.append((time.perf_counter() - t0) * 1000)
                per_q.append(retrieve.metrics_for(ranking, q.gold))
                tops.append(ranking[:5])
            key = f"{name}/{mode}"
            out["configs"][key] = {m: [p[m] for p in per_q] for m in per_q[0]}
            out.setdefault("top5", {})[key] = tops
            out.setdefault("act_filtered", {})[key] = {m: [p[m] for p in filtered] for m in filtered[0]}
            out["latency_ms"][key] = {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}
            log(f"{key}: hit@5 {np.mean(out['configs'][key]['hit@5']):.3f}  mrr@10 {np.mean(out['configs'][key]['mrr@10']):.3f}")
    out["question_ids"] = [q.id for q in qs]
    return out


def save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")


def run_stores(data_dir: Path, cache_dir: Path, scale: int = 100_000, k: int = 10, log=print) -> dict:
    """Compare pgvector and Qdrant (exact and HNSW) on the real paragraph vectors, then on a synthetic scale-up."""
    from eurag import stores

    arts = corpus.load(data_dir)
    chunks = corpus.chunk(arts, "paragraph")
    dense = retrieve.Dense(chunks, cache_dir)
    queries = np.stack([dense.embed_query(q.question) for q in questions.load()])
    out: dict = {"k": k, "sizes": {}}
    sets = {"real": dense.vectors, f"synthetic_{scale}": stores.scale_up(dense.vectors, scale)}
    for label, vecs in sets.items():
        rows = []
        for fn, hnsw in ((stores.bench_pgvector, False), (stores.bench_pgvector, True), (stores.bench_qdrant, False), (stores.bench_qdrant, True)):
            r = fn(vecs, queries, k, hnsw)
            rows.append(r.__dict__)
            log(f"{label} n={len(vecs)} {r.name}: build {r.build_s:.1f}s p50 {r.p50_ms:.2f}ms p95 {r.p95_ms:.2f}ms overlap {r.overlap:.3f}")
        out["sizes"][label] = {"n": int(len(vecs)), "results": rows}
    return out

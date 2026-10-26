"""Markdown tables for the README, generated from results/bench.json and results/stores.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eurag import corpus, questions
from eurag.retrieve import bootstrap_ci, paired_bootstrap

LABEL = {"bm25": "BM25", "dense": "Dense (bge-small)", "hybrid": "Hybrid (RRF)"}


def _name(key: str) -> str:
    r, mode = key.split("/")
    return f"{LABEL[r]}, {mode} chunks"


def main_table(b: dict) -> str:
    rows = ["| retriever | hit@1 | hit@5 | recall@5 | MRR@10 | NDCG@10 (95% CI) | query latency p50 |", "|---|---|---|---|---|---|---|"]
    for key, m in b["configs"].items():
        n = bootstrap_ci(np.array(m["ndcg@10"]))
        rows.append(f"| {_name(key)} | {np.mean(m['hit@1']):.3f} | {np.mean(m['hit@5']):.3f} | {np.mean(m['recall@5']):.3f} | "
                    f"{np.mean(m['mrr@10']):.3f} | {n[0]:.3f} ({n[1]:.3f}-{n[2]:.3f}) | {b['latency_ms'][key]['p50']:.1f} ms |")
    return "\n".join(rows)


def significance(b: dict, baseline: str = "bm25/article") -> str:
    base = np.array(b["configs"][baseline]["ndcg@10"])
    rows = []
    for key, m in b["configs"].items():
        if key != baseline:
            rows.append((key, *paired_bootstrap(np.array(m["ndcg@10"]), base)))
    order = np.argsort([r[4] for r in rows])
    holm, run = {}, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(rows) - rank) * rows[i][4]))
        holm[rows[i][0]] = run
    lines = [f"Paired bootstrap over the {b['n_questions']} questions, NDCG@10 difference versus **{_name(baseline)}**; Holm-adjusted across {len(rows)} comparisons.", "",
             "| retriever | difference | 95% CI | adjusted p |", "|---|---|---|---|"]
    for key, d, lo, hi, _p in rows:
        lines.append(f"| {_name(key)} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {holm[key]:.3f} |")
    return "\n".join(lines)


def pairs(b: dict) -> str:
    """The comparisons the prose draws, tested directly (not inferred from overlapping intervals)."""
    spec = [("dense/paragraph", "dense/article", "paragraph vs article chunks, dense"),
            ("dense/paragraph", "hybrid/paragraph", "dense alone vs hybrid, paragraph chunks"),
            ("dense/article", "hybrid/article", "dense alone vs hybrid, article chunks")]
    rows = [(label, *paired_bootstrap(np.array(b["configs"][x]["ndcg@10"]), np.array(b["configs"][y]["ndcg@10"]))) for x, y, label in spec]
    lines = ["Direct paired comparisons (NDCG@10 difference, first minus second; unadjusted p; these three were picked after seeing the main table, so treat them as exploratory):", "",
             "| comparison | difference | 95% CI | p |", "|---|---|---|---|"]
    for label, d, lo, hi, p in rows:
        lines.append(f"| {label} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.2f} |")
    return "\n".join(lines)


def unambiguous(b: dict) -> str:
    amb = set(b["ambiguous_ids"])
    keep = np.array([q not in amb for q in b["question_ids"]])
    lines = [f"{int((~keep).sum())} of {len(keep)} questions have an evidence phrase that also appears in another article of the same act, so a "
             f"retriever can surface a defensible article that is scored as a miss. Restricting to the {int(keep.sum())} unambiguous questions:", "",
             "| retriever | hit@5 (all) | hit@5 (unambiguous) | NDCG@10 (all) | NDCG@10 (unambiguous) |", "|---|---|---|---|---|"]
    for key, m in b["configs"].items():
        h, n = np.array(m["hit@5"]), np.array(m["ndcg@10"])
        lines.append(f"| {_name(key)} | {h.mean():.3f} | {h[keep].mean():.3f} | {n.mean():.3f} | {n[keep].mean():.3f} |")
    return "\n".join(lines)


def act_filter(b: dict) -> str:
    lines = ["Same retrievers, but candidates restricted to the act the question is about (an oracle: it assumes the user or an upstream router names the act).", "",
             "| retriever | hit@5 | hit@5 with act filter | NDCG@10 | NDCG@10 with act filter |", "|---|---|---|---|---|"]
    for key, m in b["configs"].items():
        f = b["act_filtered"][key]
        lines.append(f"| {_name(key)} | {np.mean(m['hit@5']):.3f} | {np.mean(f['hit@5']):.3f} | {np.mean(m['ndcg@10']):.3f} | {np.mean(f['ndcg@10']):.3f} |")
    return "\n".join(lines)


def by_act(b: dict, keys=("bm25/paragraph", "dense/paragraph")) -> str:
    acts = [q.split("-")[0].upper() for q in b["question_ids"]]
    order = ["AI", "GDPR", "DSA", "DATA", "NIS2", "CRA"]
    lines = ["| act | questions | " + " | ".join(f"hit@5 {_name(k)}" for k in keys) + " |", "|---|---|" + "---|" * len(keys)]
    for a in order:
        idx = np.array([x == a for x in acts])
        lines.append(f"| {a} | {int(idx.sum())} | " + " | ".join(f"{np.array(b['configs'][k]['hit@5'])[idx].mean():.2f}" for k in keys) + " |")
    return "\n".join(lines)


def failures(b: dict, data_dir: Path, key: str | None = None, limit: int = 12) -> str:
    key = key or max(b["configs"], key=lambda k: np.mean(b["configs"][k]["ndcg@10"]))
    qs = {q.id: q for q in questions.load()}
    titles = {a.key: a.heading for a in corpus.load(data_dir)}
    lines = [f"Misses at top-5 for the best configuration ({_name(key)}):", ""]
    n = 0
    for qid, hit, top in zip(b["question_ids"], b["configs"][key]["hit@5"], b["top5"][key]):
        if hit == 0 and n < limit:
            q = qs[qid]
            lines.append(f"- **{qid}** \"{q.question}\" -> wanted {', '.join(q.gold)}; got {', '.join(top[:3])} ({titles[top[0]].split(': ', 1)[1]})")
            n += 1
    total = sum(1 for h in b["configs"][key]["hit@5"] if h == 0)
    lines.append(f"\n({total} misses in total; {n} shown.)")
    return "\n".join(lines)


def stores_table(s: dict) -> str:
    out = []
    for label, d in s["sizes"].items():
        kind = "real paragraph vectors" if label == "real" else "synthetic scale-up (real vectors + noise)"
        out += [f"**{d['n']:,} vectors, {kind}**", "", "| store | index build | p50 | p95 | top-10 overlap with exact |", "|---|---|---|---|---|"]
        for r in d["results"]:
            out.append(f"| {r['name']} | {r['build_s']:.1f} s | {r['p50_ms']:.1f} ms | {r['p95_ms']:.1f} ms | {r['overlap']:.3f} |")
        out.append("")
    return "\n".join(out)


def build(results: Path, readme: Path, data_dir: Path) -> None:
    b = json.loads((results / "bench.json").read_text(encoding="utf-8"))
    s = json.loads((results / "stores.json").read_text(encoding="utf-8"))
    sections = {
        "main-table": main_table(b), "significance": significance(b), "pairs": pairs(b), "unambiguous": unambiguous(b), "by-act": by_act(b), "act-filter": act_filter(b),
        "failures": failures(b, data_dir), "stores": stores_table(s),
    }
    text = readme.read_text(encoding="utf-8")
    for name, body in sections.items():
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        i, j = text.find(start), text.find(end)
        if i == -1 or j < i:
            raise ValueError(f"README is missing the {name} markers")
        text = text[: i + len(start)] + "\n" + body.strip("\n") + "\n" + text[j:]
    readme.write_text(text, encoding="utf-8")

"""Retrievers (BM25, dense, hybrid) and the article-level metrics.

Every retriever maps a query to a score per chunk. `article_ranking` collapses chunk scores to a
ranking of distinct articles (best chunk wins), which is what the metrics are computed on, so
article-level and paragraph-level chunkings are scored on the same footing.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from eurag.corpus import Chunk

STOP = frozenset(
    "a an and are as at be been being by can could do does for from has have how if in into is it its may must of on or shall should "
    "than that the their them then there these they this those to under was were what when where which who whom why will with would".split()
)
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # the model card's retrieval convention
RRF_K = 60


def stem(w: str) -> str:
    """A deliberately crude suffix stripper, identical for queries and documents. It is not a linguistic
    stemmer; it exists so that 'fine', 'fines' and 'fined', or 'provide' and 'providing', meet in one term."""
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if suf in ("ies", "ied"):
                return base + "y"
            if suf == "es" and not base.endswith(("s", "x", "z", "ch", "sh")):
                base = w[:-1]  # "fines" -> "fine": the 'e' belongs to the stem
            elif suf == "s" and base.endswith("s"):
                return w  # "access", "business"
            return base[:-1] if base.endswith("e") and len(base) > 3 else base
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        docs = [tokenize(c.text) for c in chunks]
        self.n = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.lengths.mean())
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                s[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return s


class Dense:
    name = "dense"

    def __init__(self, chunks: list[Chunk], cache_dir: Path | None = None, model: str = DENSE_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model)
        key = hashlib.sha1((model + "\n".join(c.id + c.text for c in chunks)).encode()).hexdigest()[:16]
        cache = cache_dir / f"emb_{key}.npy" if cache_dir else None
        if cache and cache.exists():
            self.vectors = np.load(cache)
        else:
            self.vectors = np.asarray(list(self.model.embed([c.text for c in chunks], batch_size=32)), dtype=np.float32)
            if cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.vectors)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def embed_query(self, query: str) -> np.ndarray:
        v = np.asarray(next(iter(self.model.embed([QUERY_PREFIX + query]))), dtype=np.float32)
        return v / np.linalg.norm(v)

    def scores(self, query: str) -> np.ndarray:
        return self.vectors @ self.embed_query(query)


def rrf(rankings: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Reciprocal-rank fusion of several score vectors into one score vector (Cormack et al., 2009)."""
    out = np.zeros(len(rankings[0]), dtype=np.float32)
    for s in rankings:
        ranks = np.empty(len(s), dtype=np.int64)
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + ranks)
    return out


class Hybrid:
    name = "hybrid"

    def __init__(self, bm25: BM25, dense: Dense):
        self.bm25, self.dense = bm25, dense

    def scores(self, query: str) -> np.ndarray:
        return rrf([self.bm25.scores(query), self.dense.scores(query)])


def article_ranking(chunk_scores: np.ndarray, chunks: list[Chunk], depth: int = 20) -> list[str]:
    """Distinct articles ordered by their best chunk score."""
    best: dict[str, float] = {}
    for i in np.argsort(-chunk_scores, kind="stable")[: max(depth * 8, 200)]:
        a = chunks[i].article
        if a not in best:
            best[a] = float(chunk_scores[i])
    return list(best)[:depth]


# -- metrics ------------------------------------------------------------------------------------

def metrics_for(ranking: list[str], gold: tuple[str, ...]) -> dict[str, float]:
    g = set(gold)
    out: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        top = ranking[:k]
        out[f"hit@{k}"] = float(any(a in g for a in top))
        out[f"recall@{k}"] = len(g & set(top)) / len(g)
    rr = next((1.0 / (r + 1) for r, a in enumerate(ranking[:10]) if a in g), 0.0)
    dcg = sum(1.0 / math.log2(r + 2) for r, a in enumerate(ranking[:10]) if a in g)
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(len(g), 10)))
    out["mrr@10"], out["ndcg@10"] = rr, dcg / ideal
    return out


def bootstrap_ci(values: np.ndarray, reps: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(reps)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, reps: int = 5000, seed: int = 0) -> tuple[float, float, float, float]:
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(min(1.0, max(p, 1.0 / reps)))

"""Vector stores: pgvector (PostgreSQL) and Qdrant, each exact and with an HNSW index.

All four are given the same normalised vectors and the same queries. What is compared:
  * overlap with the exact top-k computed in NumPy (is the index losing results?)
  * query latency (p50/p95 over repeated passes, query embedding excluded)
  * index build time
At the corpus's real size (a few thousand chunks) every option is exact in practice and every query
takes about a millisecond, so a second experiment scales the collection with perturbed copies of the
real vectors. That one is synthetic by construction: it measures the engines, not retrieval quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

PG_DSN = "postgresql://postgres:postgres@127.0.0.1:5544/postgres"  # throwaway local container, see docker-compose.yml
QDRANT_URL = "http://127.0.0.1:6334"


def exact_topk(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(queries @ vectors.T), axis=1)[:, :k]


@dataclass
class Result:
    name: str
    build_s: float
    p50_ms: float
    p95_ms: float
    overlap: float  # mean |store top-k  intersect  exact top-k| / k


def _timed_queries(search, queries: np.ndarray, k: int, passes: int) -> tuple[list[list[int]], list[float]]:
    found, times = [], []
    for p in range(passes):
        for q in queries:
            t0 = time.perf_counter()
            ids = search(q, k)
            times.append((time.perf_counter() - t0) * 1000)
            if p == 0:
                found.append(ids)
    return found, times


def _overlap(found: list[list[int]], truth: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(f[:k]) & set(t[:k].tolist())) / k for f, t in zip(found, truth)]))


def bench_qdrant(vectors: np.ndarray, queries: np.ndarray, k: int, hnsw: bool, passes: int = 5, url: str | None = QDRANT_URL) -> Result:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=url) if url else QdrantClient(":memory:")
    name = "bench"
    client.recreate_collection(name, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
                               hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100))
    t0 = time.perf_counter()
    step = 1000
    for s in range(0, len(vectors), step):
        client.upsert(name, points=[models.PointStruct(id=i, vector=vectors[i].tolist()) for i in range(s, min(s + step, len(vectors)))], wait=True)
    if hnsw and url:  # wait until the index is built so latency is measured against it
        while client.get_collection(name).status != models.CollectionStatus.GREEN:
            time.sleep(0.2)
    build = time.perf_counter() - t0
    params = models.SearchParams(exact=not hnsw, hnsw_ef=64 if hnsw else None)

    def search(q, kk):
        return [p.id for p in client.query_points(name, query=q.tolist(), limit=kk, search_params=params).points]

    found, times = _timed_queries(search, queries, k, passes)
    truth = exact_topk(vectors, queries, k)
    client.delete_collection(name)
    return Result(f"qdrant-{'hnsw' if hnsw else 'exact'}", build, float(np.percentile(times, 50)), float(np.percentile(times, 95)), _overlap(found, truth, k))


def bench_pgvector(vectors: np.ndarray, queries: np.ndarray, k: int, hnsw: bool, passes: int = 5, dsn: str = PG_DSN) -> Result:
    import psycopg

    dim = vectors.shape[1]
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("DROP TABLE IF EXISTS bench")
        conn.execute(f"CREATE TABLE bench (id int PRIMARY KEY, emb vector({dim}))")
        t0 = time.perf_counter()
        with conn.cursor() as cur, cur.copy("COPY bench (id, emb) FROM STDIN") as copy:
            for i, v in enumerate(vectors):
                copy.write_row((i, "[" + ",".join(f"{x:.7f}" for x in v) + "]"))
        if hnsw:
            conn.execute("CREATE INDEX ON bench USING hnsw (emb vector_cosine_ops) WITH (m = 16, ef_construction = 100)")
            conn.execute("SET hnsw.ef_search = 64")
        else:
            conn.execute("SET enable_indexscan = off")
        conn.execute("ANALYZE bench")
        build = time.perf_counter() - t0

        def search(q, kk):
            vec = "[" + ",".join(f"{x:.7f}" for x in q) + "]"
            return [r[0] for r in conn.execute("SELECT id FROM bench ORDER BY emb <=> %s::vector LIMIT %s", (vec, kk)).fetchall()]

        found, times = _timed_queries(search, queries, k, passes)
        conn.execute("DROP TABLE bench")
    truth = exact_topk(vectors, queries, k)
    return Result(f"pgvector-{'hnsw' if hnsw else 'exact'}", build, float(np.percentile(times, 50)), float(np.percentile(times, 95)), _overlap(found, truth, k))


def scale_up(vectors: np.ndarray, n: int, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """n synthetic vectors: real chunk embeddings plus Gaussian noise, renormalised."""
    rng = np.random.default_rng(seed)
    base = vectors[rng.integers(0, len(vectors), n)]
    out = base + rng.normal(0, noise, base.shape).astype(np.float32)
    return out / np.linalg.norm(out, axis=1, keepdims=True)

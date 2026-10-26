# eurag

Which retrieval setup finds the right article of EU digital law, and does the choice of vector
database matter? A small, checked benchmark over six acts: the AI Act, GDPR, Digital Services Act,
Data Act, NIS2 and the Cyber Resilience Act (472 articles).

This is the **retrieval** half of a RAG system, measured on its own. There is no generation step and
no faithfulness score here; that needs a language model and is a separate piece of work.

## Corpus and questions

- **Corpus:** official Official-Journal text from the EU Publications Office (CELLAR), parsed into
  articles and paragraphs. The parser asserts the exact article count of each act (113 / 99 / 93 / 50 / 46 / 71).
  Fetched by `eurag fetch`, not committed.
- **Questions:** 67 plain-language questions written by hand, each labelled with the article that
  answers it. Every label carries an *evidence phrase*, and `eurag verify` (also run by the test suite)
  fails if the phrase does not literally occur in the gold article, so a wrong article number cannot
  quietly lower every score. Questions paraphrase the law rather than copy its headings, because
  copying headings turns the task into keyword matching.
- **Metric level:** relevance is judged at the **article** level. Paragraph-chunk rankings are
  de-duplicated to articles first, so both chunkings are scored on the same footing.

## Results

<!-- main-table:start -->
| retriever | hit@1 | hit@5 | recall@5 | MRR@10 | NDCG@10 (95% CI) | query latency p50 |
|---|---|---|---|---|---|---|
| BM25, article chunks | 0.418 | 0.597 | 0.597 | 0.498 | 0.536 (0.430-0.640) | 1.0 ms |
| Dense (bge-small), article chunks | 0.597 | 0.806 | 0.806 | 0.687 | 0.733 (0.646-0.814) | 23.9 ms |
| Hybrid (RRF), article chunks | 0.507 | 0.791 | 0.791 | 0.621 | 0.684 (0.598-0.766) | 23.1 ms |
| BM25, paragraph chunks | 0.313 | 0.657 | 0.657 | 0.456 | 0.514 (0.420-0.612) | 2.6 ms |
| Dense (bge-small), paragraph chunks | 0.552 | 0.851 | 0.851 | 0.687 | 0.745 (0.667-0.820) | 27.0 ms |
| Hybrid (RRF), paragraph chunks | 0.537 | 0.761 | 0.761 | 0.633 | 0.691 (0.603-0.776) | 34.7 ms |
<!-- main-table:end -->

<!-- significance:start -->
Paired bootstrap over the 67 questions, NDCG@10 difference versus **BM25, article chunks**; Holm-adjusted across 5 comparisons.

| retriever | difference | 95% CI | adjusted p |
|---|---|---|---|
| Dense (bge-small), article chunks | +0.197 | [+0.093, +0.299] | 0.001 |
| Hybrid (RRF), article chunks | +0.147 | [+0.080, +0.214] | 0.001 |
| BM25, paragraph chunks | -0.023 | [-0.070, +0.024] | 0.351 |
| Dense (bge-small), paragraph chunks | +0.209 | [+0.111, +0.308] | 0.001 |
| Hybrid (RRF), paragraph chunks | +0.155 | [+0.095, +0.218] | 0.001 |
<!-- significance:end -->

<!-- pairs:start -->
Direct paired comparisons (NDCG@10 difference, first minus second; unadjusted p; these three were picked after seeing the main table, so treat them as exploratory):

| comparison | difference | 95% CI | p |
|---|---|---|---|
| paragraph vs article chunks, dense | +0.012 | [-0.050, +0.079] | 0.71 |
| dense alone vs hybrid, paragraph chunks | +0.054 | [-0.014, +0.128] | 0.12 |
| dense alone vs hybrid, article chunks | +0.049 | [-0.017, +0.120] | 0.15 |
<!-- pairs:end -->

### By act

<!-- by-act:start -->
| act | questions | hit@5 BM25, paragraph chunks | hit@5 Dense (bge-small), paragraph chunks |
|---|---|---|---|
| AI | 20 | 0.70 | 0.90 |
| GDPR | 20 | 0.45 | 0.70 |
| DSA | 11 | 0.73 | 1.00 |
| DATA | 6 | 0.50 | 0.83 |
| NIS2 | 5 | 1.00 | 1.00 |
| CRA | 5 | 1.00 | 0.80 |
<!-- by-act:end -->

### What a metadata filter would buy

<!-- act-filter:start -->
Same retrievers, but candidates restricted to the act the question is about (an oracle: it assumes the user or an upstream router names the act).

| retriever | hit@5 | hit@5 with act filter | NDCG@10 | NDCG@10 with act filter |
|---|---|---|---|---|
| BM25, article chunks | 0.597 | 0.672 | 0.536 | 0.592 |
| Dense (bge-small), article chunks | 0.806 | 0.866 | 0.733 | 0.787 |
| Hybrid (RRF), article chunks | 0.791 | 0.866 | 0.684 | 0.760 |
| BM25, paragraph chunks | 0.657 | 0.716 | 0.514 | 0.626 |
| Dense (bge-small), paragraph chunks | 0.851 | 0.925 | 0.745 | 0.817 |
| Hybrid (RRF), paragraph chunks | 0.761 | 0.896 | 0.691 | 0.768 |
<!-- act-filter:end -->

### How much of the "miss" rate is the labels?

<!-- unambiguous:start -->
31 of 67 questions have an evidence phrase that also appears in another article of the same act, so a retriever can surface a defensible article that is scored as a miss. Restricting to the 36 unambiguous questions:

| retriever | hit@5 (all) | hit@5 (unambiguous) | NDCG@10 (all) | NDCG@10 (unambiguous) |
|---|---|---|---|---|
| BM25, article chunks | 0.597 | 0.528 | 0.536 | 0.483 |
| Dense (bge-small), article chunks | 0.806 | 0.806 | 0.733 | 0.688 |
| Hybrid (RRF), article chunks | 0.791 | 0.833 | 0.684 | 0.671 |
| BM25, paragraph chunks | 0.657 | 0.583 | 0.514 | 0.466 |
| Dense (bge-small), paragraph chunks | 0.851 | 0.833 | 0.745 | 0.696 |
| Hybrid (RRF), paragraph chunks | 0.761 | 0.778 | 0.691 | 0.646 |
<!-- unambiguous:end -->

### Where the best retriever fails

<!-- failures:start -->
Misses at top-5 for the best configuration (Dense (bge-small), paragraph chunks):

- **ai-10** "What documentation must the makers of large foundation models hand to the regulator?" -> wanted AI:53; got AI:91, AI:11, CRA:31 (Power to request documentation and information)
- **ai-14** "Which procedure proves a high-risk system meets the rules before it can be sold?" -> wanted AI:43; got AI:9, AI:80, AI:16 (Risk management system)
- **gdpr-04** "On what legal grounds is it permitted to handle personal data at all?" -> wanted GDPR:6; got DATA:4, GDPR:10, DATA:5 (The rights and obligations of users and data holders with regard to access, use and making available product data and related service data)
- **gdpr-09** "Can I receive my data in a reusable electronic form and take it to another service?" -> wanted GDPR:20; got DATA:30, GDPR:15, DATA:3 (Technical aspects of switching)
- **gdpr-10** "What can I ask a company to tell me about the data it holds about me?" -> wanted GDPR:15; got GDPR:13, DATA:3, DATA:18 (Information to be provided where personal data are collected from the data subject)
- **gdpr-12** "What must a company make sure of before hiring a service provider to handle personal data for it?" -> wanted GDPR:28; got GDPR:25, GDPR:13, DATA:18 (Data protection by design and by default)
- **gdpr-17** "Does the law reach a company outside Europe that sells to people in the EU?" -> wanted GDPR:3; got GDPR:48, DSA:30, GDPR:49 (Transfers or disclosures not authorised by Union law)
- **gdpr-20** "Can I tell a company to stop using my data for advertising mail?" -> wanted GDPR:21; got DSA:26, DATA:4, DATA:15 (Advertising on online platforms)
- **data-04** "Can a user instruct the manufacturer to pass their device data on to another company?" -> wanted DATA:5; got DATA:3, CRA:13, DATA:4 (Obligation to make product data and related service data accessible to the user)
- **cra-03** "Which document must the maker draw up to state that the product meets the cybersecurity requirements?" -> wanted CRA:28; got CRA:6, CRA:31, CRA:13 (Requirements for products with digital elements)

(10 misses in total; 10 shown.)
<!-- failures:end -->

## What the numbers say

- **Dense retrieval beats BM25 by a wide margin on these questions**: about +0.20 NDCG@10 (adjusted p = 0.001),
  and +0.23 at paragraph level in a direct comparison. The questions paraphrase the law ("hand to the regulator" for
  "technical documentation of the model"), so lexical overlap is the wrong signal. This is a property of the
  question style: the gap would shrink for queries that quote the statute's own wording.
- **Chunking size did not matter measurably.** Paragraph chunks are ahead on hit@5 (0.85 vs 0.81) and on NDCG@10
  (+0.012), but the interval around that difference is [-0.05, +0.08]: with 67 questions this is unresolved, not
  zero. The model's 512-token window truncates long articles, which is the mechanism that would favour paragraphs; the data does not confirm it.
- **Hybrid fusion did not help.** Equal-weight reciprocal-rank fusion scored below dense alone (0.691 vs 0.745 at paragraph
  level) because it blends in a much weaker ranker. The difference is not statistically resolved (p = 0.12-0.15), so the
  defensible reading is "no gain here", not "fusion hurts".
- **A large part of the remaining error is cross-act confusion.** Several misses retrieve a Data Act or DSA article for a
  GDPR question (see the failure list). Restricting candidates to the right act lifts dense/paragraph hit@5 from 0.85 to 0.93.
  The act filter is an oracle here; in a product it would come from a router, a UI scope or a classifier, and that component
  can fail too.
- **The labels cost something.** About half of the questions have an evidence phrase that also appears in other articles of
  the same act, so some "misses" are defensible answers. On the unambiguous subset the ranking of retrievers is unchanged.
- **Latency is dominated by embedding the query** (about 22-25 ms on CPU) rather than by search; BM25 answers in 1-3 ms.

## pgvector vs Qdrant

Same vectors, same queries, both run in Docker on one laptop; query embedding is excluded. Overlap
is against the exact top-10 computed in NumPy.

<!-- stores:start -->
**2,766 vectors, real paragraph vectors**

| store | index build | p50 | p95 | top-10 overlap with exact |
|---|---|---|---|---|
| pgvector-exact | 1.9 s | 6.4 ms | 18.7 ms | 1.000 |
| pgvector-hnsw | 4.4 s | 3.8 ms | 9.0 ms | 0.993 |
| qdrant-exact | 2.8 s | 10.3 ms | 29.1 ms | 1.000 |
| qdrant-hnsw | 2.5 s | 10.0 ms | 22.6 ms | 1.000 |

**100,000 vectors, synthetic scale-up (real vectors + noise)**

| store | index build | p50 | p95 | top-10 overlap with exact |
|---|---|---|---|---|
| pgvector-exact | 71.7 s | 61.0 ms | 142.5 ms | 1.000 |
| pgvector-hnsw | 542.7 s | 13.4 ms | 43.8 ms | 0.836 |
| qdrant-exact | 138.2 s | 47.0 ms | 98.2 ms | 1.000 |
| qdrant-hnsw | 191.9 s | 10.7 ms | 36.0 ms | 0.931 |
<!-- stores:end -->

**Reading the store results.** At the corpus's real size (2.8k vectors) all four configurations return the right results
(overlap 0.99-1.00) and the choice does not matter: the single-digit-millisecond differences are driver and container overhead.
The scale-up shows what indexes are for: at 100k vectors, HNSW answers in about 11-13 ms versus 47-61 ms for exact search,
at the cost of recall (0.84 pgvector, 0.93 Qdrant at `ef`=64) and a much longer build (9 minutes for pgvector's HNSW here versus
3 for Qdrant's). That recall figure is on deliberately hard synthetic data and with untuned parameters; it should be read as
"approximate indexes need their recall measured", not as a ranking of the products. For a corpus this size, the deciding
factors are operational (you already run Postgres; you want filtering and snapshots built in), not performance.

## Limitations

- **67 questions by one author.** The confidence intervals are wide, and the questions share the
  author's phrasing habits. They are not a sample of what real users ask.
- **Short gold lists.** Several questions have more than one defensible article; only the labelled
  one counts, so every retriever's accuracy is understated. The ambiguity table above sizes that.
- **One embedding model** (`BAAI/bge-small-en-v1.5`, 384 dimensions, via ONNX on CPU), used as-is.
  Chunks longer than the model's 512-token window are truncated, which penalises article-level
  chunks of long articles and is part of the chunking comparison, not a bug.
- **BM25 uses a crude suffix stemmer** and no query expansion. It is a fair classical baseline, not a tuned one.
- **Hybrid uses plain reciprocal-rank fusion with equal weights.** When one ranker is much weaker, equal-weight
  fusion can fall below the stronger ranker alone, which is what happens here; a weighted or learned fusion might not.
- **No reranker and no generation.** The numbers say how often the right article is retrieved, not how good an answer built on it would be.
- **The store comparison is a laptop microbenchmark** through Docker Desktop: latencies include driver and
  container overhead, p95 is noisy, and the 100k-vector set is synthetic (real vectors plus noise), so it
  measures the engines, not retrieval quality.

## Run it

```bash
pip install -e ".[dev]"
pytest                          # parsing, metrics, label verification, in-memory Qdrant
eurag fetch                     # download the six acts
eurag verify                    # check all question labels against the text
eurag bench                     # retrievers x chunkings (downloads the embedding model once)
docker compose up -d
eurag stores                    # pgvector vs Qdrant
eurag report                    # refresh the tables in this README
```

Source text: European Union, https://eur-lex.europa.eu/ , reused under Commission Decision 2011/833/EU. MIT licence (code).

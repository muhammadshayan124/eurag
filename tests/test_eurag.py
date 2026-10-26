from __future__ import annotations

import numpy as np
import pytest

from eurag import corpus, questions, retrieve, stores

XHTML = """<html><body>
<div class="eli-subdivision" id="art_1"><p class="oj-ti-art">Article\xa01</p>
  <div class="eli-title"><p class="oj-sti-art">Subject matter</p></div>
  <div id="001.001"><p class="oj-normal">1.\xa0This law protects <b>people</b>.</p></div>
  <div id="001.002"><p class="oj-normal">2.\xa0It also protects data.</p></div></div>
<div class="eli-subdivision" id="art_2"><p class="oj-ti-art">Article 2</p>
  <div class="eli-title"><p class="oj-sti-art">Scope</p></div>
  <p class="oj-normal">This applies to everyone.</p></div>
<div class="eli-subdivision" id="art_3a"><p class="oj-ti-art">Article 3a</p>
  <div class="eli-title"><p class="oj-sti-art">Extra</p></div>
  <div id="003.001"><p class="oj-normal">Extra text.</p></div></div>
</body></html>"""


def arts():
    return corpus.parse_act("AI", XHTML)


def test_parse_numbers_titles_paragraphs_and_unnumbered_bodies():
    a = arts()
    assert [x.number for x in a] == ["1", "2", "3a"]
    assert a[0].title == "Subject matter" and len(a[0].paragraphs) == 2
    assert "people" in a[0].paragraphs[0] and "\xa0" not in a[0].paragraphs[0]
    assert a[1].paragraphs == ("This applies to everyone.",)


def test_chunking_modes():
    a = arts()
    assert [c.id for c in corpus.chunk(a, "article")] == ["AI:1", "AI:2", "AI:3a"]
    para = corpus.chunk(a, "paragraph")
    assert [c.id for c in para] == ["AI:1#1", "AI:1#2", "AI:2", "AI:3a"]
    assert all(c.text.startswith("AI Act, Article") for c in para)  # every chunk carries its article heading
    with pytest.raises(ValueError):
        corpus.chunk(a, "sentence")


def test_bm25_ranks_the_matching_chunk_first_and_is_query_order_free():
    chunks = [corpus.Chunk("a", "A", "penalties and fines for infringements"), corpus.Chunk("b", "B", "the definition of an artificial intelligence system"),
              corpus.Chunk("c", "C", "transparency obligations for chatbots")]
    bm = retrieve.BM25(chunks)
    assert int(np.argmax(bm.scores("what fines apply to infringement"))) == 0
    assert int(np.argmax(bm.scores("chatbot transparency"))) == 2
    assert np.allclose(bm.scores("fines infringements"), bm.scores("infringements fines"))
    assert bm.scores("zzzzunknown").sum() == 0


def test_tokenizer_drops_stopwords_and_stems_consistently():
    assert retrieve.tokenize("The providers shall notify authorities") == ["provider", "notify", "authority"]
    for group in (("fine", "fines", "fined"), ("provide", "provides", "provided", "providing"), ("infringement", "infringements")):
        assert len({retrieve.stem(w) for w in group}) == 1, group
    assert retrieve.stem("access") == "access"
    assert "the" not in retrieve.tokenize("the and of")


def test_rrf_rewards_agreement_between_rankers():
    a = np.array([0.9, 0.5, 0.1, 0.0])
    b = np.array([0.8, 0.1, 0.7, 0.0])
    fused = retrieve.rrf([a, b])
    assert int(np.argmax(fused)) == 0
    assert fused[0] > fused[3]


def test_article_ranking_dedupes_to_best_chunk():
    chunks = [corpus.Chunk("A#1", "A", ""), corpus.Chunk("A#2", "A", ""), corpus.Chunk("B", "B", "")]
    ranking = retrieve.article_ranking(np.array([0.2, 0.9, 0.5]), chunks, depth=5)
    assert ranking == ["A", "B"]


def test_metrics_hand_checked():
    m = retrieve.metrics_for(["X", "G", "Y"], ("G",))
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["mrr@10"] == pytest.approx(0.5)
    assert m["ndcg@10"] == pytest.approx(1 / np.log2(3))
    two = retrieve.metrics_for(["G1", "X", "G2"], ("G1", "G2"))
    assert two["recall@1"] == 0.5 and two["recall@3"] == 1.0
    assert retrieve.metrics_for(["X"], ("G",))["mrr@10"] == 0.0


def test_paired_bootstrap_on_identical_scores_is_null():
    x = np.random.default_rng(0).random(60)
    d, lo, hi, p = retrieve.paired_bootstrap(x, x.copy())
    assert (d, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)


def test_question_verification_flags_wrong_labels():
    a = arts()
    good = [questions.Question("q1", "?", ("AI:1",), "protects data")]
    assert questions.verify(good, a) == []
    bad = [questions.Question("q2", "?", ("AI:2",), "protects data"), questions.Question("q3", "?", ("AI:99",), "x"),
           questions.Question("q1", "?", ("AI:1",), "protects data"), questions.Question("q1", "?", ("AI:1",), "protects data")]
    msgs = questions.verify(bad, a)
    assert any("not found in AI:2" in m for m in msgs)
    assert any("does not exist" in m for m in msgs)
    assert any("duplicate id" in m for m in msgs)


def test_shipped_question_file_is_well_formed():
    qs = questions.load()
    assert len(qs) >= 60 and len({q.id for q in qs}) == len(qs)
    assert all(q.gold and q.evidence and q.question.endswith("?") for q in qs)


def test_qdrant_in_memory_matches_exact_search():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(300, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[:10] + 0.01
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    res = stores.bench_qdrant(v, q, k=5, hnsw=False, passes=1, url=None)
    assert res.overlap == pytest.approx(1.0)


def test_scale_up_is_normalised_and_deterministic():
    v = np.random.default_rng(1).normal(size=(50, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    a, b = stores.scale_up(v, 200), stores.scale_up(v, 200)
    assert a.shape == (200, 8) and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5) and np.array_equal(a, b)

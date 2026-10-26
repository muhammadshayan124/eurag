"""The EU digital-regulation corpus: download from the Publications Office, parse, chunk.

Source: the Publications Office of the EU (CELLAR) serves the Official Journal text as XHTML by
CELEX number. EU legislation may be reused with attribution (Commission Decision 2011/833/EU), but
the corpus is fetched rather than committed so the repo stays small and always matches the source.

Two chunkings are compared in the benchmark:
  article    one chunk per article, prefixed with its number and title
  paragraph  one chunk per numbered paragraph, prefixed the same way (articles with no numbered
             paragraphs stay whole)
Relevance is always judged at the ARTICLE level: a paragraph chunk is a hit if its article is a
gold article, and the ranking is de-duplicated to articles before metrics are computed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from lxml import etree

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CELLAR = "http://publications.europa.eu/resource/celex/{celex}"

ACTS = {
    "AI": ("32024R1689", "AI Act"),
    "GDPR": ("32016R0679", "GDPR"),
    "DSA": ("32022R2065", "Digital Services Act"),
    "DATA": ("32023R2854", "Data Act"),
    "NIS2": ("32022L2555", "NIS2 Directive"),
    "CRA": ("32024R2847", "Cyber Resilience Act"),
}
EXPECTED_ARTICLES = {"AI": 113, "GDPR": 99, "DSA": 93, "DATA": 50, "NIS2": 46, "CRA": 71}  # sanity check on parsing


def fetch(directory: Path = DATA_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for key, (celex, _) in ACTS.items():
        r = httpx.get(CELLAR.format(celex=celex), headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
                      timeout=120, follow_redirects=True)
        r.raise_for_status()
        if "xhtml" not in r.headers.get("content-type", ""):
            raise ValueError(f"{key}: unexpected content type {r.headers.get('content-type')}")
        path = directory / f"{key}.xhtml"
        path.write_bytes(r.content)
        out.append(path)
    return out


@dataclass(frozen=True)
class Article:
    act: str
    number: str
    title: str
    paragraphs: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.act}:{self.number}"

    @property
    def heading(self) -> str:
        return f"{ACTS[self.act][1]}, Article {self.number}: {self.title}"

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


@dataclass(frozen=True)
class Chunk:
    id: str
    article: str  # Article.key
    text: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def parse_act(act: str, source: bytes | str | Path) -> list[Article]:
    raw = Path(source).read_bytes() if isinstance(source, Path) else source
    root = etree.fromstring(raw if isinstance(raw, bytes) else raw.encode("utf-8"), etree.HTMLParser())
    articles = []
    for div in root.xpath('//div[@class="eli-subdivision" and starts-with(@id,"art_")]'):
        label = _clean("".join(div.xpath('.//p[@class="oj-ti-art"]//text()')))
        m = re.search(r"(\d+[a-z]?)\s*$", label)
        if not m:
            continue
        title = _clean("".join(div.xpath('.//p[@class="oj-sti-art"]//text()')))
        paras = [_clean("".join(p.itertext())) for p in div.xpath('./div[re:test(@id, "^\\d{3}\\.\\d{3}$")]', namespaces={"re": "http://exslt.org/regular-expressions"})]
        if not paras:  # unnumbered body: everything after the heading
            body = [_clean("".join(p.itertext())) for p in div.xpath('.//p[@class="oj-normal"]')]
            paras = [b for b in body if b]
        paras = [p for p in paras if p]
        articles.append(Article(act, m.group(1), title, tuple(paras)))
    return articles


def load(directory: Path = DATA_DIR, acts: tuple[str, ...] | None = None) -> list[Article]:
    out: list[Article] = []
    for key in acts or tuple(ACTS):
        arts = parse_act(key, directory / f"{key}.xhtml")
        if len(arts) != EXPECTED_ARTICLES[key]:
            raise ValueError(f"{key}: parsed {len(arts)} articles, expected {EXPECTED_ARTICLES[key]}")
        out += arts
    return out


def chunk(articles: list[Article], mode: str) -> list[Chunk]:
    if mode not in ("article", "paragraph"):
        raise ValueError("mode must be 'article' or 'paragraph'")
    out = []
    for a in articles:
        if mode == "article" or len(a.paragraphs) <= 1:
            out.append(Chunk(f"{a.key}", a.key, f"{a.heading}\n{a.text}"))
        else:
            out += [Chunk(f"{a.key}#{i + 1}", a.key, f"{a.heading}\n{p}") for i, p in enumerate(a.paragraphs)]
    return out

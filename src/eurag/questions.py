"""The hand-labelled question set.

Each question names its gold article(s) and an `evidence` phrase that must literally occur in the
gold article's text. `verify` checks that mechanically, so a wrong article number or a typo in the
label cannot silently depress every retriever's score. `ambiguity` reports how many *other*
articles contain the same evidence phrase (a label that another article also satisfies is a
label with a hole in it, and is flagged rather than hidden).

Limits, stated plainly: the questions were written by the repo author, not by independent users,
so they share the author's phrasing habits; there are 70 of them; and gold lists are short, so a
retriever that surfaces a different but genuinely relevant article is scored as a miss.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from eurag.corpus import Article

QUESTIONS = Path(__file__).resolve().parents[2] / "questions.jsonl"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold: tuple[str, ...]
    evidence: str


def load(path: Path = QUESTIONS) -> list[Question]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Question(d["id"], d["question"], tuple(d["gold"]), d["evidence"]))
    return out


def norm(s: str) -> str:
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"').replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).lower()


def verify(questions: list[Question], articles: list[Article]) -> list[str]:
    """Problems with the labels; empty means every gold article contains its evidence phrase."""
    by_key = {a.key: a for a in articles}
    problems = []
    seen = set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        for g in q.gold:
            if g not in by_key:
                problems.append(f"{q.id}: gold article {g} does not exist")
            elif norm(q.evidence) not in norm(by_key[g].heading + " " + by_key[g].text):
                problems.append(f"{q.id}: evidence {q.evidence!r} not found in {g}")
    return problems


def ambiguity(questions: list[Question], articles: list[Article]) -> dict[str, list[str]]:
    """For each question, other articles in the SAME act whose text also contains the evidence phrase."""
    out = {}
    for q in questions:
        act = q.gold[0].split(":")[0]
        phrase = norm(q.evidence)
        others = [a.key for a in articles if a.act == act and a.key not in q.gold and phrase in norm(a.text)]
        if others:
            out[q.id] = others
    return out

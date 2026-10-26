from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from eurag import bench, corpus, questions

app = typer.Typer(help="Retrieval benchmark over EU digital regulation.", no_args_is_help=True)
console = Console(highlight=False)
DATA, CACHE, RESULTS = corpus.DATA_DIR, corpus.DATA_DIR / "cache", Path("results")


@app.command()
def fetch() -> None:
    """Download the six acts from the Publications Office (CELLAR)."""
    for p in corpus.fetch():
        console.print(f"saved {p}")


@app.command()
def verify() -> None:
    """Check every question label against the corpus text and report ambiguous evidence phrases."""
    arts = corpus.load()
    qs = questions.load()
    problems = questions.verify(qs, arts)
    for p in problems:
        console.print(f"PROBLEM {p}")
    amb = questions.ambiguity(qs, arts)
    console.print(f"{len(qs)} questions, {len(problems)} problems, {len(amb)} with an evidence phrase that also appears in another article")
    raise typer.Exit(1 if problems else 0)


@app.command("bench")
def bench_cmd(out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS) -> None:
    """Retrievers x chunkings over the question set."""
    bench.save(bench.run(DATA, CACHE, log=console.print), out / "bench.json")


@app.command("stores")
def stores_cmd(
    out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS,
    scale: Annotated[int, typer.Option(help="Size of the synthetic scale-up")] = 100_000,
) -> None:
    """pgvector vs Qdrant (needs `docker compose up -d`)."""
    result = bench.run_stores(DATA, CACHE, scale=scale, log=console.print)
    (out / "stores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


@app.command("report")
def report_cmd(
    out: Annotated[Path, typer.Option(help="Results directory")] = RESULTS,
    readme: Annotated[Path, typer.Option(help="README with <!-- name:start/end --> markers")] = Path("README.md"),
) -> None:
    """Fill the README tables from the saved results."""
    from eurag import report

    report.build(out, readme, DATA)
    console.print(f"updated {readme}")


if __name__ == "__main__":
    app()

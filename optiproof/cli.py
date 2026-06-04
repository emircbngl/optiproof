"""Command-line surface: optimize / profile / verify.

`profile` and `verify` work without any LLM — they let a user trust the harness
(hotspot location, correctness + benchmark gates) independently of candidate
generation, and they're a clean manual on-ramp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .adapters.base import AdapterRegistry
from .llm.null_provider import NullProvider
from .models import Candidate, OptimizeKind, OptimizeRequest, SandboxBackend
from .orchestrator import optimize as run_optimize
from .reporting.json_report import to_json
from .reporting.terminal import render

app = typer.Typer(
    add_completion=False,
    help="OptiProof — like code review, but it optimizes code and proves the speedup.",
)
console = Console()


def _require_selector(target: str) -> None:
    if "::" not in target:
        console.print("[red]target must be FILE::FUNCTION, e.g. mymod.py::solve[/red]")
        raise typer.Exit(2)


def _sandbox(value: str) -> SandboxBackend:
    try:
        return SandboxBackend(value)
    except ValueError:
        console.print(f"[red]invalid --sandbox {value!r}; choose: local, docker[/red]")
        raise typer.Exit(2)


@app.command()
def optimize(
    target: str = typer.Argument(..., help="FILE::FUNCTION to optimize"),
    candidates: int = typer.Option(5, "--candidates", "-n", help="candidates per round"),
    max_rounds: int = typer.Option(3, "--max-rounds", help="max generate→prove rounds"),
    threshold: float = typer.Option(1.10, "--threshold", help="min median speedup to accept"),
    satisfied_at: float = typer.Option(3.0, "--satisfied-at", help="early-stop speedup"),
    sandbox: str = typer.Option("local", "--sandbox", help="local | docker"),
    provider: str = typer.Option("anthropic", "--provider", help="anthropic | null"),
    model: Optional[str] = typer.Option(None, "--model"),
    seed: int = typer.Option(1234, "--seed"),
    toolchain_image: Optional[str] = typer.Option(None, "--toolchain-image", help="Docker image for --sandbox docker"),
    workload_size: int = typer.Option(1200, "--workload-size", help="benchmark workload size"),
    report: str = typer.Option("md", "--report", help="md | json"),
    apply: bool = typer.Option(False, "--apply", help="write the winning diff back to the file"),
):
    """Search for a verified-faster rewrite of FUNCTION and prove it."""
    _require_selector(target)
    path = Path(target.split("::", 1)[0])
    if sandbox == "local":
        console.print("[yellow]note: --sandbox local is not isolated from the host; use docker for untrusted code[/yellow]")
    req = OptimizeRequest(
        path=path, selector=target, candidates_per_round=candidates, max_rounds=max_rounds,
        threshold=threshold, satisfied_at=satisfied_at, sandbox=_sandbox(sandbox),
        provider=provider, model=model, seed=seed,
        toolchain_image=toolchain_image, workload_size=workload_size,
    )
    result = run_optimize(req)
    if report == "json":
        console.print(to_json(result))
    else:
        console.print(render(result))
    if apply and result.improved and result.best:
        from .patch import apply_candidate

        adapter = AdapterRegistry.detect(path)
        t = adapter.locate_target(path, target)
        original = path.read_text()
        path.write_text(apply_candidate(original, t, result.best))
        console.print(f"[green]applied winning diff to {path}[/green]")


@app.command()
def profile(target: str = typer.Argument(..., help="FILE::FUNCTION (or FILE to list functions)")):
    """Locate the target (no LLM). Automatic hotspot ranking is Phase 2."""
    if "::" in target:
        path = Path(target.split("::", 1)[0])
        adapter = AdapterRegistry.detect(path)
        t = adapter.locate_target(path, target)
        console.print(
            f"{t.file.name}::{t.symbol}  lines {t.start_line}-{t.end_line}  params={t.param_types or '(none)'}"
        )
    else:
        import ast

        path = Path(target)
        tree = ast.parse(path.read_text())
        funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        console.print(f"{path.name}: functions = {funcs}")
    console.print("[dim]MVP: explicit targets; dynamic hotspot ranking is Phase 2.[/dim]")


@app.command()
def verify(
    target: str = typer.Argument(..., help="FILE::FUNCTION (the original)"),
    candidate_file: str = typer.Argument(..., help="file holding the replacement function source"),
    prelude: Optional[str] = typer.Option(None, "--prelude", help="module-level code to add (e.g. imports)"),
    sandbox: str = typer.Option("local", "--sandbox"),
    toolchain_image: Optional[str] = typer.Option(None, "--toolchain-image", help="Docker image for --sandbox docker"),
    threshold: float = typer.Option(1.10, "--threshold"),
    report: str = typer.Option("md", "--report"),
):
    """Run only the correctness + benchmark gates on a human-supplied replacement."""
    _require_selector(target)
    path = Path(target.split("::", 1)[0])
    adapter = AdapterRegistry.detect(path)
    t = adapter.locate_target(path, target)
    cand = Candidate(
        id="human-candidate", kind=OptimizeKind.REWRITE, title="human-supplied",
        new_source=Path(candidate_file).read_text(), module_prelude=prelude,
    )
    req = OptimizeRequest(
        path=path, selector=target, sandbox=_sandbox(sandbox), threshold=threshold, max_rounds=1,
        toolchain_image=toolchain_image,
    )
    result = run_optimize(req, provider=NullProvider(by_symbol={t.symbol: [cand]}))
    console.print(to_json(result) if report == "json" else render(result))


if __name__ == "__main__":
    app()

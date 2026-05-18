"""CLI tool for Noodly — query and manage the Company Brain."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from noodly.config import get_settings

console = Console()


def _build_ledger(settings=None):
    """Build a FactLedger using the configured backend."""
    from noodly.scoring.ledger import FactLedger

    if settings is None:
        settings = get_settings()
    if settings.use_graphiti_backend:
        from noodly.graph.brain import Brain
        from noodly.storage.graphiti_backend import GraphitiBackend

        brain = Brain(settings)
        return FactLedger(backend=GraphitiBackend(brain)), brain
    return FactLedger(backend=settings.brain_dir / "ledger.json"), None


def _run(coro):
    """Run an async coroutine from sync Click commands."""
    return asyncio.run(coro)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Noodly — Open-source Company Brain."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@cli.command()
def init() -> None:
    """Initialize the brain (set up graph indices)."""
    from noodly.pipeline import Pipeline

    settings = get_settings()

    async def _init():
        pipeline = Pipeline(settings)
        await pipeline.initialize()
        await pipeline.close()

    _run(_init())
    console.print("[green]Brain initialized.[/green]")


@cli.command()
@click.option(
    "--watch-dir", type=click.Path(exists=True), default=None,
    help="Override watch directory",
)
def ingest(watch_dir: str | None) -> None:
    """Scan the inbox folder and ingest new files."""
    from noodly.pipeline import Pipeline

    settings = get_settings()
    if watch_dir:
        settings.watch_dir = Path(watch_dir)

    async def _ingest():
        pipeline = Pipeline(settings)
        await pipeline.initialize()
        stats = await pipeline.run()
        await pipeline.close()
        return stats

    stats = _run(_ingest())
    console.print(
        f"[green]Ingested:[/green] {stats['artifacts']} files "
        f"→ {stats['claims']} claims → {stats['projected']} projected"
    )


@cli.command()
@click.argument("text")
@click.option("--title", "-t", default="Manual input", help="Title for the ingested text")
@click.option("--author", "-a", default="", help="Author of the text")
def add(text: str, title: str, author: str) -> None:
    """Add raw text directly to the brain."""
    from noodly.pipeline import Pipeline

    settings = get_settings()

    async def _add():
        pipeline = Pipeline(settings)
        await pipeline.initialize()
        stats = await pipeline.ingest_text(title=title, body=text, author=author)
        await pipeline.close()
        return stats

    stats = _run(_add())
    console.print(
        f"[green]Added:[/green] {stats['claims']} claims extracted "
        f"→ {stats['projected']} files projected"
    )


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Maximum results")
def search(query: str, limit: int) -> None:
    """Search the brain for entities and facts."""
    from noodly.graph.brain import Brain

    settings = get_settings()

    async def _search():
        brain = Brain(settings)
        nodes = await brain.search_nodes(query, limit=limit)
        facts = await brain.search_facts(query, limit=limit)
        await brain.close()
        return nodes, facts

    nodes, facts = _run(_search())

    if nodes:
        table = Table(title="Entities")
        table.add_column("Name", style="cyan")
        table.add_column("Summary")
        for n in nodes:
            table.add_row(n.get("name", ""), n.get("summary", ""))
        console.print(table)

    if facts:
        table = Table(title="Facts")
        table.add_column("Fact", style="green")
        table.add_column("Created")
        table.add_column("Expired")
        for f in facts:
            table.add_row(
                f.get("fact", f.get("name", "")),
                f.get("created_at", ""),
                f.get("expired_at", ""),
            )
        console.print(table)

    if not nodes and not facts:
        console.print("[yellow]No results found.[/yellow]")


@cli.command()
@click.option("--status", "-s", default="", help="Filter by claim status")
@click.option("--limit", "-n", default=20, help="Maximum results")
def claims(status: str, limit: int) -> None:
    """List claims from the fact ledger."""
    from noodly.models.claims import ClaimStatus

    settings = get_settings()
    ledger, brain = _build_ledger(settings)

    claim_status = None
    if status:
        try:
            claim_status = ClaimStatus(status)
        except ValueError:
            console.print(f"[red]Unknown status: {status}[/red]")
            return

    async def _claims():
        if ledger.is_async_backend:
            await ledger.load_async()

        results = ledger.list_claims(status=claim_status, limit=limit)

        if not results:
            console.print("[yellow]No claims found.[/yellow]")
        else:
            table = Table(title=f"Claims ({len(results)})")
            table.add_column("Score", style="bold", justify="right")
            table.add_column("Status", style="cyan")
            table.add_column("Class")
            table.add_column("Claim", style="green")
            table.add_column("Evidence")

            for c in results:
                table.add_row(
                    f"{c.truth_score:.2f}",
                    c.status.value,
                    c.knowledge_class.value,
                    c.natural_language[:80],
                    str(len(c.evidence)),
                )

            console.print(table)

        if brain:
            await brain.close()

    if ledger.is_async_backend:
        _run(_claims())
    else:
        results = ledger.list_claims(status=claim_status, limit=limit)
        if not results:
            console.print("[yellow]No claims found.[/yellow]")
            return
        table = Table(title=f"Claims ({len(results)})")
        table.add_column("Score", style="bold", justify="right")
        table.add_column("Status", style="cyan")
        table.add_column("Class")
        table.add_column("Claim", style="green")
        table.add_column("Evidence")
        for c in results:
            table.add_row(
                f"{c.truth_score:.2f}",
                c.status.value,
                c.knowledge_class.value,
                c.natural_language[:80],
                str(len(c.evidence)),
            )
        console.print(table)


@cli.command()
def stats() -> None:
    """Show brain statistics."""
    settings = get_settings()
    ledger, brain = _build_ledger(settings)

    def _print_stats(all_claims):
        status_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        for c in all_claims:
            status_counts[c.status.value] = status_counts.get(c.status.value, 0) + 1
            class_counts[c.knowledge_class.value] = class_counts.get(c.knowledge_class.value, 0) + 1

        console.print("\n[bold]Noodly Brain Stats[/bold]")
        console.print(f"Total claims: {len(all_claims)}")

        if all_claims:
            avg_score = sum(c.truth_score for c in all_claims) / len(all_claims)
            console.print(f"Average truth score: {avg_score:.3f}")

        if status_counts:
            console.print("\n[bold]By Status:[/bold]")
            for s, count in sorted(status_counts.items()):
                console.print(f"  {s}: {count}")

        if class_counts:
            console.print("\n[bold]By Knowledge Class:[/bold]")
            for k, count in sorted(class_counts.items()):
                console.print(f"  {k}: {count}")

    if ledger.is_async_backend:
        async def _stats():
            await ledger.load_async()
            _print_stats(ledger.list_claims(limit=10000))
            if brain:
                await brain.close()

        _run(_stats())
    else:
        _print_stats(ledger.list_claims(limit=10000))


@cli.command()
def project() -> None:
    """Project the current brain state to Markdown files."""
    from noodly.projection.markdown import MarkdownProjector

    settings = get_settings()
    ledger, brain = _build_ledger(settings)
    projector = MarkdownProjector(settings.brain_dir)

    if ledger.is_async_backend:
        async def _project():
            await ledger.load_async()
            all_claims = ledger.list_claims(limit=10000)
            written = projector.project(all_claims)
            msg = f"[green]Projected {written} Markdown files to {settings.brain_dir}[/green]"
            console.print(msg)
            if brain:
                await brain.close()

        _run(_project())
    else:
        all_claims = ledger.list_claims(limit=10000)
        written = projector.project(all_claims)
        console.print(f"[green]Projected {written} Markdown files to {settings.brain_dir}[/green]")


@cli.command()
def migrate() -> None:
    """Migrate claims from JSON ledger to Graphiti backend."""
    from noodly.graph.brain import Brain
    from noodly.scoring.ledger import FactLedger
    from noodly.storage.graphiti_backend import GraphitiBackend

    settings = get_settings()
    json_path = settings.brain_dir / "ledger.json"

    if not json_path.exists():
        console.print(f"[red]No JSON ledger found at {json_path}[/red]")
        return

    json_ledger = FactLedger(backend=json_path)
    all_claims = json_ledger.list_claims(limit=100000)

    if not all_claims:
        console.print("[yellow]No claims to migrate.[/yellow]")
        return

    console.print(f"Found {len(all_claims)} claims to migrate.")

    brain = Brain(settings)
    graphiti_backend = GraphitiBackend(brain)

    async def _migrate():
        await brain.initialize()
        for i, claim in enumerate(all_claims, 1):
            await graphiti_backend.save_claim_async(claim)
            if i % 10 == 0:
                console.print(f"  Migrated {i}/{len(all_claims)}...")
        await brain.close()

    _run(_migrate())
    console.print(
        f"[green]Migrated {len(all_claims)} claims to Graphiti backend.[/green]\n"
        "Set NOODLY_USE_GRAPHITI_BACKEND=true to use the new backend."
    )


@cli.command()
def serve() -> None:
    """Start the MCP server (for AI agent integration)."""
    from noodly.server.mcp_server import run_server

    console.print("[bold]Starting Noodly MCP server...[/bold]")
    run_server()

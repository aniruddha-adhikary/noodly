"""CLI tool for Noodly — query and manage the Company Brain."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from noodly.config import get_settings

console = Console()


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
@click.option("--as-of", default="", help="Bi-temporal: valid-time filter (ISO date)")
def claims(status: str, limit: int, as_of: str) -> None:
    """List claims from the fact ledger."""
    from noodly.models.claims import ClaimStatus
    from noodly.scoring.authority import AuthorityRegistry
    from noodly.scoring.ledger import FactLedger

    settings = get_settings()
    authority = AuthorityRegistry(settings.brain_dir / "authority.json")
    ledger = FactLedger(settings.brain_dir / "ledger.json", authority_registry=authority)

    claim_status = None
    if status:
        try:
            claim_status = ClaimStatus(status)
        except ValueError:
            console.print(f"[red]Unknown status: {status}[/red]")
            return

    as_of_valid = None
    if as_of:
        try:
            as_of_valid = datetime.fromisoformat(as_of)
            if as_of_valid.tzinfo is None:
                as_of_valid = as_of_valid.replace(tzinfo=timezone.utc)
        except ValueError:
            console.print(f"[red]Invalid date format: {as_of}[/red]")
            return

    results = ledger.list_claims(status=claim_status, limit=limit, as_of_valid=as_of_valid)

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
    from noodly.scoring.ledger import FactLedger

    settings = get_settings()
    ledger = FactLedger(settings.brain_dir / "ledger.json")

    all_claims = ledger.list_claims(limit=10000)

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


@cli.command()
def project() -> None:
    """Project the current brain state to Markdown files."""
    from noodly.projection.markdown import MarkdownProjector
    from noodly.scoring.ledger import FactLedger

    settings = get_settings()
    ledger = FactLedger(settings.brain_dir / "ledger.json")
    projector = MarkdownProjector(settings.brain_dir)

    all_claims = ledger.list_claims(limit=10000)
    written = projector.project(all_claims)
    console.print(f"[green]Projected {written} Markdown files to {settings.brain_dir}[/green]")


@cli.command()
def serve() -> None:
    """Start the MCP server (for AI agent integration)."""
    from noodly.server.mcp_server import run_server

    console.print("[bold]Starting Noodly MCP server...[/bold]")
    run_server()


@cli.group()
def authority() -> None:
    """Manage source authority weights."""


@authority.command(name="list")
def authority_list() -> None:
    """List all registered source authority weights."""
    from noodly.scoring.authority import AuthorityRegistry

    settings = get_settings()
    registry = AuthorityRegistry(settings.brain_dir / "authority.json")
    sources = registry.list_sources()

    if not sources:
        console.print("[yellow]No source authorities configured. Default weight is 0.5.[/yellow]")
        return

    table = Table(title="Source Authority Weights")
    table.add_column("Source", style="cyan")
    table.add_column("Weight", justify="right")

    for source, weight in sources.items():
        table.add_row(source, f"{weight:.2f}")

    console.print(table)


@authority.command(name="set")
@click.argument("source")
@click.argument("weight", type=float)
def authority_set(source: str, weight: float) -> None:
    """Set authority weight for a source (0.0-1.0)."""
    from noodly.scoring.authority import AuthorityRegistry

    settings = get_settings()
    registry = AuthorityRegistry(settings.brain_dir / "authority.json")
    registry.set(source, weight)
    console.print(f"[green]Set {source} = {registry.get(source):.2f}[/green]")


@authority.command(name="remove")
@click.argument("source")
def authority_remove(source: str) -> None:
    """Remove a source from the authority registry."""
    from noodly.scoring.authority import AuthorityRegistry

    settings = get_settings()
    registry = AuthorityRegistry(settings.brain_dir / "authority.json")
    if registry.remove(source):
        console.print(f"[green]Removed {source}[/green]")
    else:
        console.print(f"[yellow]{source} not found in registry[/yellow]")


@cli.command()
@click.option("--limit", "-n", default=30, help="Maximum events to show")
@click.option("--source", "-s", default="", help="Filter by source URI")
def changelog(limit: int, source: str) -> None:
    """Show the change log (append-only event history)."""
    from noodly.tracking.changelog import ChangeLog

    settings = get_settings()
    log = ChangeLog(settings.brain_dir / "changelog.json")

    if source:
        events = log.for_source(source)
    else:
        events = log.recent(limit=limit)

    if not events:
        console.print("[yellow]No change events found.[/yellow]")
        return

    table = Table(title=f"Change Log ({len(events)} events)")
    table.add_column("Time", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Entity/Source", style="green")
    table.add_column("Agent")

    for event in events:
        table.add_row(
            event.timestamp.strftime("%Y-%m-%d %H:%M"),
            event.change_type.value,
            event.entity_id or event.source_uri[:50],
            event.agent or "-",
        )

    console.print(table)


@cli.group()
def conflicts() -> None:
    """Manage conflict detection and resolution."""


@conflicts.command(name="list")
@click.option("--limit", "-n", default=20, help="Maximum results")
def conflicts_list(limit: int) -> None:
    """List detected conflicts and their resolution status."""
    from noodly.resolution.audit import ResolutionAudit

    settings = get_settings()
    audit = ResolutionAudit(settings.brain_dir / "resolutions.json")
    resolutions = audit.list_resolutions(limit=limit)

    if not resolutions:
        console.print("[yellow]No conflict resolutions recorded.[/yellow]")
        return

    table = Table(title=f"Conflict Resolutions ({len(resolutions)})")
    table.add_column("ID", style="dim")
    table.add_column("Strategy", style="cyan")
    table.add_column("Resolved By")
    table.add_column("Confidence", justify="right")
    table.add_column("Winner", style="green")
    table.add_column("Date", style="dim")

    for r in resolutions:
        table.add_row(
            str(r.conflict_id)[:8],
            r.strategy_used,
            r.resolved_by,
            f"{r.confidence:.3f}",
            str(r.winner_id)[:8] if r.winner_id else "pending",
            r.resolved_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


@conflicts.command(name="detect")
def conflicts_detect() -> None:
    """Detect conflicts among existing claims."""
    from noodly.resolution.detector import ConflictDetector
    from noodly.scoring.ledger import FactLedger

    settings = get_settings()
    ledger = FactLedger(settings.brain_dir / "ledger.json")
    detector = ConflictDetector(
        similarity_threshold=settings.conflict_similarity_threshold,
    )

    all_claims = ledger.list_claims(limit=10000)
    conflicts_found = detector.detect_within(all_claims)

    if not conflicts_found:
        console.print("[green]No conflicts detected.[/green]")
        return

    table = Table(title=f"Detected Conflicts ({len(conflicts_found)})")
    table.add_column("Subject", style="cyan")
    table.add_column("Predicate")
    table.add_column("Value A", style="green")
    table.add_column("Value B", style="red")
    table.add_column("Score Delta", justify="right")

    for c in conflicts_found:
        table.add_row(
            c.claim_a.subject[:30],
            c.claim_a.predicate[:30],
            c.claim_a.object[:30],
            c.claim_b.object[:30],
            f"{c.score_delta:.3f}",
        )

    console.print(table)


@conflicts.command(name="resolve")
@click.option("--auto-threshold", "-t", type=float, default=None,
              help="Override auto-resolve threshold")
@click.option("--strategy", "-s", type=click.Choice(
    ["authority_wins", "recency_wins", "majority_wins", "higher_score"]),
    default=None, help="Override resolution strategy")
def conflicts_resolve(auto_threshold: float | None, strategy: str | None) -> None:
    """Detect and resolve conflicts among existing claims."""
    from noodly.resolution.audit import ResolutionAudit
    from noodly.resolution.detector import ConflictDetector
    from noodly.resolution.resolver import ConflictResolver
    from noodly.resolution.strategies import AutoResolveStrategy
    from noodly.scoring.authority import AuthorityRegistry
    from noodly.scoring.ledger import FactLedger
    from noodly.tracking.changelog import ChangeLog

    settings = get_settings()
    authority = AuthorityRegistry(settings.brain_dir / "authority.json")
    ledger = FactLedger(settings.brain_dir / "ledger.json", authority_registry=authority)
    audit = ResolutionAudit(settings.brain_dir / "resolutions.json")
    changelog = ChangeLog(settings.brain_dir / "changelog.json")

    threshold = auto_threshold if auto_threshold is not None else settings.auto_resolve_threshold
    strat_str = strategy or settings.resolve_strategy
    try:
        strat = AutoResolveStrategy(strat_str)
    except ValueError:
        strat = AutoResolveStrategy.AUTHORITY_WINS

    detector = ConflictDetector(
        similarity_threshold=settings.conflict_similarity_threshold,
    )
    resolver = ConflictResolver(
        ledger=ledger,
        audit=audit,
        changelog=changelog,
        auto_threshold=threshold,
        strategy=strat,
    )

    all_claims = ledger.list_claims(limit=10000)
    conflicts_found = detector.detect_within(all_claims)

    if not conflicts_found:
        console.print("[green]No conflicts detected.[/green]")
        return

    console.print(f"Found {len(conflicts_found)} conflicts. Resolving...")

    async def _resolve():
        return await resolver.resolve_batch(conflicts_found)

    resolutions = _run(_resolve())

    auto = sum(1 for r in resolutions if r.winner_id is not None)
    manual = sum(1 for r in resolutions if r.winner_id is None)
    console.print(
        f"[green]Resolved:[/green] {auto} auto-resolved, {manual} pending manual review"
    )


@cli.group()
def dispatch() -> None:
    """Event dispatch system management."""


@dispatch.command(name="stats")
def dispatch_stats() -> None:
    """Show event dispatch and audit statistics."""
    import json

    settings = get_settings()
    audit_path = settings.brain_dir / "audit.jsonl"
    resolutions_path = settings.brain_dir / "resolutions.json"

    console.print("\n[bold]Event Dispatch Stats[/bold]")

    if audit_path.exists():
        with open(audit_path) as f:
            line_count = sum(1 for _ in f)
        console.print(f"  Audit log entries: {line_count}")
    else:
        console.print("  Audit log: not started")

    if resolutions_path.exists():
        try:
            data = json.loads(resolutions_path.read_text())
            total = len(data)
            pending = sum(1 for r in data if r.get("winner_id") is None)
            console.print(f"  Total resolutions: {total}")
            console.print(f"  Pending manual: {pending}")
        except (json.JSONDecodeError, Exception):
            console.print("  Resolutions file: corrupt")
    else:
        console.print("  Resolutions: none")


@cli.group()
def cache() -> None:
    """Manage caches (parse, extraction, agent decisions)."""


@cache.command(name="stats")
def cache_stats() -> None:
    """Show cache statistics."""
    from noodly.caching.manager import CacheManager

    settings = get_settings()
    mgr = CacheManager(settings.brain_dir / ".cache")

    content_count = len(list((settings.brain_dir / ".cache" / "content").glob("*.md")))
    extraction_count = len(list((settings.brain_dir / ".cache" / "extractions").glob("*.json")))

    console.print("\n[bold]Cache Stats[/bold]")
    console.print(f"  Parse cache entries: {content_count}")
    console.print(f"  Extraction cache entries: {extraction_count}")
    console.print(f"  Agent decision merges: {mgr.decisions.merge_count}")


@cache.command(name="clear")
@click.option("--level", type=click.Choice(["parse", "extraction", "all"]), default="all")
def cache_clear(level: str) -> None:
    """Clear cached data."""
    import shutil

    settings = get_settings()
    cache_dir = settings.brain_dir / ".cache"

    if level in ("parse", "all"):
        content_dir = cache_dir / "content"
        if content_dir.exists():
            shutil.rmtree(content_dir)
            content_dir.mkdir(parents=True, exist_ok=True)
            console.print("[green]Parse cache cleared.[/green]")

    if level in ("extraction", "all"):
        from noodly.caching.manager import CacheManager

        mgr = CacheManager(cache_dir)
        removed = mgr.extraction.invalidate_all()
        console.print(f"[green]Extraction cache cleared ({removed} entries).[/green]")

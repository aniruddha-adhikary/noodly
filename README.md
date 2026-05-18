# Noodly 🍜

Open-source Company Brain — ingest knowledge from multiple sources, build a temporal context graph with provenance tracking, and expose it to AI agents via MCP and CLI.

## What it does

Noodly watches your files, extracts structured claims using LLMs, scores them for truthfulness, and projects everything into a queryable knowledge graph + human-readable Markdown tree.

```
Files/Messages → Source Artifacts → LLM Extraction → Claims → Scored Truth → Markdown + MCP
```

### The 3-Layer Evidence Model

1. **Raw Artifacts** — immutable records of ingested content (files, messages, docs)
2. **Claims** — normalized assertions extracted by LLM with source anchoring (subject → predicate → object)
3. **Scored Truth** — claims promoted through a truth pipeline with confidence × authority × recency scoring

### Truth Scoring

Every claim carries a composite truth score:

```
truth_score = confidence × authority × recency × corroboration × conflict_penalty
```

Claims decay based on their knowledge class:
- **stable** (0.1%/day) — legal names, product names, repo ownership
- **process** (0.5%/day) — workflows, procedures, onboarding steps
- **tacit** (2%/day) — informal know-how, shortcuts, workarounds
- **stateful** (5%/day) — active incidents, current owners, open deals

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for FalkorDB)
- OpenAI API key

### Setup

```bash
# Clone
git clone https://github.com/aniruddha-adhikary/noodly.git
cd noodly

# Start FalkorDB
docker compose up -d

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Usage

```bash
# Initialize the brain (creates graph indices)
noodly init

# Drop files into the inbox folder, then ingest
mkdir -p inbox
echo "The billing service is owned by Jane Smith. It uses Python 3.12 and FastAPI." > inbox/billing-notes.md
noodly ingest

# Or add text directly
noodly add "PortNet integration is managed by the platform team"

# Search the brain
noodly search "who owns billing"

# List extracted claims
noodly claims

# View brain stats
noodly stats

# Project to Markdown files
noodly project

# Start the MCP server (for Claude, Cursor, Devin)
noodly serve
```

### MCP Integration

Add to your Claude/Cursor MCP config:

```json
{
  "mcpServers": {
    "noodly": {
      "command": "noodly",
      "args": ["serve"]
    }
  }
}
```

Available MCP tools:
- `search` — hybrid semantic + keyword search over entities and facts
- `search_claims` — search extracted claims in the fact ledger
- `get_entity` — look up a specific entity by name
- `list_claims` — list claims filtered by status
- `list_recent_episodes` — show recently ingested episodes
- `brain_stats` — summary statistics

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Connectors │────▶│   Pipeline   │────▶│   Outputs   │
│             │     │              │     │             │
│ Local FS    │     │ 1. Ingest    │     │ Markdown    │
│ (Slack)*    │     │ 2. Extract   │     │ MCP Server  │
│ (Email)*    │     │ 3. Score     │     │ CLI         │
│ (Notion)*   │     │ 4. Project   │     │             │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────┴───────┐
                    │   Storage    │
                    │              │
                    │ FalkorDB    │
                    │ (Graphiti)   │
                    │              │
                    │ Fact Ledger  │
                    │ (JSON/v1)    │
                    └──────────────┘

* = planned for Phase 2
```

## Project Structure

```
src/noodly/
├── models/          # Pydantic models (SourceArtifact, Claim, ClaimEvidence)
├── graph/           # Graphiti + FalkorDB brain wrapper
├── connectors/      # Data source connectors (local filesystem)
├── extraction/      # LLM-powered claim extraction (OpenAI)
├── scoring/         # Claim scoring engine + fact ledger
├── projection/      # Graph → Markdown filesystem projection
├── server/          # MCP server for AI agent integration
├── pipeline.py      # End-to-end orchestrator
├── cli.py           # Click CLI tool
└── config.py        # Settings (pydantic-settings)
```

## Development

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/noodly/

# Test
pytest
```

## Roadmap

- [x] **Phase 1** — Core loop (ingest → extract → score → project)
  - [x] Local filesystem connector
  - [x] OpenAI claim extraction with source anchoring
  - [x] Fact ledger with truth scoring + decay
  - [x] Markdown projection with frontmatter
  - [x] MCP server
  - [x] CLI tool
- [ ] **Phase 2** — Truth maintenance
  - [ ] ATMS-inspired support sets
  - [ ] Source authority registry
  - [ ] Conflict detection + retraction propagation
  - [ ] Bi-temporal querying (point-in-time truth)
- [ ] **Phase 3** — Connectors
  - [ ] Slack
  - [ ] Google Drive / Notion
  - [ ] Email (IMAP)
  - [ ] GitHub (issues, PRs, discussions)
  - [ ] Meeting transcripts
- [ ] **Phase 4** — Scale
  - [ ] PostgreSQL-backed ledger
  - [ ] Async pipeline with task queue
  - [ ] Web dashboard
  - [ ] Multi-tenant support

## License

MIT — Copyright (c) 2026 Aniruddha Adhikary

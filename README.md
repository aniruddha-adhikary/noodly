# Noodly

Open-source Company Brain — ingest knowledge from multiple sources, build a temporal knowledge graph with provenance tracking, and expose scored truth to AI agents via MCP and CLI.

## What It Does

Noodly watches your files, extracts structured claims using LLMs, scores them for truthfulness with decay and authority weighting, deduplicates semantically, resolves conflicts, and projects everything into a queryable knowledge graph + human-readable Markdown tree that can be published to GitLab.

```
Documents → Parse → Extract Claims → Score → Deduplicate → Promote → Project → Publish
   │          │          │              │          │            │          │         │
   ▼          ▼          ▼              ▼          ▼            ▼          ▼         ▼
 PDF/DOCX  MarkItDown  GPT-4o-mini  Truth     Semantic     Candidate  Markdown   GitLab
 XLSX/PPTX  /Docling   extraction   scoring   similarity   → Corr.   entities   repo
 Markdown              + aliases    + decay   (0.92 cos)   → Canon.  + topics
```

## Key Features

### Three-Layer Evidence Model

1. **Raw Artifacts** — immutable records of ingested content (files, messages, docs)
2. **Claims** — normalized subject-predicate-object triples extracted by LLM with source anchoring
3. **Scored Truth** — claims promoted through a truth pipeline with composite scoring

### Truth Scoring

Every claim carries a composite truth score:

```
truth_score = confidence x authority x recency x corroboration x conflict_penalty
```

Claims decay based on their knowledge class:

| Class | Decay Rate | Examples |
|-------|-----------|----------|
| **stable** | 0.1%/day | Legal names, product names, repo ownership |
| **process** | 0.5%/day | Workflows, procedures, onboarding steps |
| **tacit** | 2%/day | Informal know-how, shortcuts, workarounds |
| **stateful** | 5%/day | Active incidents, current owners, open deals |

### Claim Lifecycle

Claims progress through a promotion ladder based on evidence:

```
candidate → unverified → corroborated → owner_confirmed → canonical
                                                              │
                                                    superseded / rejected
```

Promotion is automatic based on configurable thresholds:
- **Score-based**: `truth_score >= 0.15` promotes candidate to unverified
- **Authority-based**: `source_authority >= 0.8` promotes candidate to unverified
- **Corroboration**: `2+` independent sources promotes to corroborated

### Semantic Deduplication

Two-pass dedup at ingestion time:
1. **Exact fingerprint** — `subject|predicate|object` hash match
2. **Semantic similarity** — cosine similarity of embeddings (threshold: 0.92)

When a semantic match is found, evidence is merged into the existing claim rather than creating a duplicate.

### Conflict Resolution

Configurable strategies for handling contradictory claims:
- `authority_wins` — higher authority source prevails
- `recency_wins` — most recent claim wins
- `majority_wins` — most-supported claim wins
- `higher_score` — highest truth score wins

### Multi-Format Document Parsing

| Format | Parser | Notes |
|--------|--------|-------|
| PDF | MarkItDown (default) or Docling | Direct PDF ingestion, no pre-conversion |
| DOCX | MarkItDown | Microsoft Word documents |
| XLSX | MarkItDown | Excel spreadsheets with table extraction |
| PPTX | MarkItDown | PowerPoint presentations |
| Markdown | Native | Direct text processing |
| HTML | MarkItDown | Web page content |

### Knowledge Graph (Graphiti + FalkorDB)

Entity and relationship extraction powered by [Graphiti](https://github.com/getzep/graphiti) with [FalkorDB](https://www.falkordb.com/) as the graph database backend. Supports:
- Hybrid search (semantic + BM25 keyword)
- Temporal awareness (bi-temporal querying)
- Configurable embeddings (default: `text-embedding-3-large`, 3072 dims)

### GitLab Publishing

Publish your projected brain as Markdown files to any GitLab instance (including enterprise/self-hosted):
- Full sync or incremental updates
- Entities, topics, sources, and index pages
- Configurable target branch and knowledge path

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

# For PDF support (recommended)
pip install "markitdown[pdf]"

# Configure
cp .env.example .env
# Edit .env and set NOODLY_OPENAI_API_KEY
```

### Optional Extras

```bash
# Docling parser (alternative PDF/document parser with OCR)
pip install -e ".[docling]"

# PostgreSQL storage backend
pip install -e ".[postgresql]"
```

### Basic Usage

```bash
# Initialize the brain (creates graph indices)
noodly init

# Drop files into the inbox folder, then ingest
mkdir -p inbox
cp /path/to/documents/*.pdf inbox/
noodly ingest

# Or add text directly
noodly add "The billing service is owned by Jane Smith. It uses Python 3.12 and FastAPI."

# Search the brain
noodly search "who owns billing"

# List extracted claims
noodly claims

# View brain stats
noodly stats

# Project to Markdown files
noodly project

# Publish to GitLab
noodly publish

# Start the MCP server (for Claude, Cursor, Devin)
noodly serve
```

## CLI Reference

### Core Commands

| Command | Description |
|---------|-------------|
| `noodly init` | Initialize the brain (set up graph indices) |
| `noodly ingest [--watch-dir DIR]` | Scan inbox and ingest new/changed files |
| `noodly add TEXT [-t TITLE] [-a AUTHOR]` | Add raw text directly to the brain |
| `noodly search QUERY [-n LIMIT]` | Hybrid semantic + keyword search |
| `noodly claims [-s STATUS] [-n LIMIT] [--as-of DATE]` | List claims with optional bi-temporal filtering |
| `noodly stats` | Show brain statistics |
| `noodly project [--full]` | Project brain state to Markdown files |
| `noodly publish [-b BRANCH] [-m MSG] [--dry-run]` | Publish projected brain to GitLab |
| `noodly serve` | Start the MCP server |

### Authority Management

| Command | Description |
|---------|-------------|
| `noodly authority list` | List all source authority weights |
| `noodly authority set SOURCE WEIGHT [--topic TOPIC]` | Set authority weight (0.0-1.0), optionally per-topic |
| `noodly authority remove SOURCE [--topic TOPIC]` | Remove a source from the registry |

### Claim Promotion

| Command | Description |
|---------|-------------|
| `noodly promote` | Auto-promote all eligible claims |
| `noodly promote-claim ID STATUS` | Manually set a claim's status |
| `noodly embedding-stats` | Show embedding coverage statistics |
| `noodly embed-claims [--batch-size N]` | Backfill embeddings for claims missing them |

### Conflict Resolution

| Command | Description |
|---------|-------------|
| `noodly conflicts list` | List detected conflicts and resolution status |
| `noodly conflicts detect` | Detect conflicts among existing claims |
| `noodly conflicts resolve [-t THRESHOLD] [-s STRATEGY]` | Auto-resolve conflicts |

### GitLab Integration

| Command | Description |
|---------|-------------|
| `noodly gitlab sync [-m MSG]` | Full sync: render all claims and push to GitLab |
| `noodly gitlab push [-s SUBJECTS] [-m MSG]` | Incremental push: update only changed entities |
| `noodly gitlab diff` | Preview what would change on next sync (dry run) |

### Cache Management

| Command | Description |
|---------|-------------|
| `noodly cache stats` | Show parse and extraction cache statistics |
| `noodly cache clear [--level parse\|extraction\|all]` | Clear cached data |

### Other

| Command | Description |
|---------|-------------|
| `noodly changelog [-n LIMIT] [-s SOURCE]` | Show the change log (event history) |
| `noodly dispatch stats` | Show event dispatch and audit statistics |

## MCP Integration

Add to your Claude/Cursor/Devin MCP config:

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

## Configuration

All settings are configurable via environment variables with the `NOODLY_` prefix. Managed by [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_OPENAI_API_KEY` | *(required)* | OpenAI API key for extraction and embeddings |
| `NOODLY_OPENAI_MODEL` | `gpt-4o-mini` | LLM model for claim extraction |
| `NOODLY_BRAIN_DIR` | `./brain` | Directory for brain data (ledger, projections, cache) |
| `NOODLY_WATCH_DIR` | `./inbox` | Directory to scan for new documents |

### FalkorDB

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_FALKORDB_HOST` | `localhost` | FalkorDB host |
| `NOODLY_FALKORDB_PORT` | `6379` | FalkorDB port |
| `NOODLY_FALKORDB_DATABASE` | `default` | FalkorDB database name |

### Embeddings & Semantic Dedup

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `NOODLY_EMBEDDING_DIM` | `3072` | Embedding dimensions |
| `NOODLY_ENABLE_INGESTION_EMBEDDINGS` | `true` | Embed claims at ingestion time |
| `NOODLY_SEMANTIC_DEDUP_THRESHOLD` | `0.92` | Cosine similarity threshold for dedup |
| `NOODLY_ENABLE_SEMANTIC_DEDUP` | `false` | Enable post-hoc semantic dedup pass |

### Claim Promotion

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_PROMOTE_THRESHOLD` | `0.15` | Truth score threshold for candidate to unverified |
| `NOODLY_HIGH_AUTHORITY_THRESHOLD` | `0.8` | Source authority for auto-unverified |
| `NOODLY_CORROBORATION_COUNT` | `2` | Independent sources needed for corroborated |

### Conflict Resolution

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_ENABLE_CONFLICT_RESOLUTION` | `false` | Enable conflict detection and resolution |
| `NOODLY_RESOLVE_STRATEGY` | `authority_wins` | Resolution strategy |
| `NOODLY_AUTO_RESOLVE_THRESHOLD` | `0.3` | Confidence threshold for auto-resolution |
| `NOODLY_CONFLICT_SIMILARITY_THRESHOLD` | `0.8` | Similarity threshold for conflict detection |

### Parallel LLM Dispatch

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_LLM_MAX_CONCURRENT` | `8` | Maximum concurrent LLM API calls |
| `NOODLY_LLM_RATE_LIMIT_RPM` | `500` | Rate limit (requests per minute) |
| `NOODLY_LLM_RETRY_MAX` | `3` | Maximum retries on failure |
| `NOODLY_LLM_REQUEST_TIMEOUT` | `30.0` | Request timeout in seconds |

### Topic Clustering

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_ENABLE_TOPIC_CLUSTERING` | `true` | Enable LLM-based topic classification |
| `NOODLY_TOPIC_MODEL` | `gpt-4o-mini` | Model for topic classification |
| `NOODLY_AUTHORITY_TOPIC_INFERENCE` | `llm` | Topic inference mode (`llm` or `keyword`) |

### Emission Planning

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_EMISSION_MODE` | `incremental` | Projection mode (`incremental` or `full`) |

### Document Parsing

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_EXTRACTION_MODE` | `auto` | Parser selection (`auto`, `markitdown`, `docling`, `multi`) |
| `NOODLY_ENABLE_DOCLING` | `false` | Enable Docling parser |
| `NOODLY_CHUNK_SIZE` | `6000` | Document chunk size for extraction |

### Agentic Features

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_ENABLE_QA_AGENT` | `false` | Enable extraction QA agent |
| `NOODLY_ENABLE_GRAPH_AGENT` | `false` | Enable graph population agent |
| `NOODLY_QA_CHANGE_THRESHOLD` | `0.05` | Change threshold for QA agent review |

### GitLab Integration

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_GITLAB_URL` | `https://gitlab.com` | GitLab instance URL (supports enterprise/self-hosted) |
| `NOODLY_GITLAB_TOKEN` | *(empty)* | GitLab personal access token (`api` scope) |
| `NOODLY_GITLAB_PROJECT_ID` | *(empty)* | Numeric GitLab project ID |
| `NOODLY_GITLAB_TARGET_BRANCH` | `main` | Target branch for commits |
| `NOODLY_GITLAB_KNOWLEDGE_PATH` | `knowledge` | Path prefix for knowledge files in the repo |
| `NOODLY_ENABLE_GITLAB_PROJECTION` | `false` | Auto-sync to GitLab after ingest |
| `NOODLY_ENABLE_GITLAB_HANDLER` | `false` | Enable GitLab event handler |

### Storage Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `NOODLY_STORAGE_BACKEND` | `json` | Storage backend (`json` or `postgresql`) |
| `NOODLY_POSTGRESQL_DSN` | *(empty)* | PostgreSQL connection string |

## Architecture

```
┌──────────────────┐     ┌───────────────────────────┐     ┌──────────────────┐
│   Connectors     │     │        Pipeline            │     │     Outputs      │
│                  │     │                            │     │                  │
│ Local Filesystem │────▶│ 1. Parse (MarkItDown)      │────▶│ Markdown tree    │
│                  │     │ 2. Extract (GPT-4o-mini)   │     │   entities/      │
│                  │     │ 3. Embed (text-emb-3-lg)   │     │   topics/        │
│                  │     │ 4. Dedup (semantic + exact) │     │   sources/       │
│                  │     │ 5. Score (truth pipeline)   │     │   index.md       │
│                  │     │ 6. Promote (auto-promote)   │     │                  │
│                  │     │ 7. Classify (topics)        │     │ MCP Server       │
│                  │     │ 8. Project (incremental)    │     │ CLI              │
│                  │     │ 9. Publish (GitLab)         │     │ GitLab repo      │
│                  │     └────────────┬────────────────┘     └──────────────────┘
│                  │                  │
│                  │     ┌────────────┴────────────────┐
│                  │     │         Storage              │
│                  │     │                              │
│                  │     │ FalkorDB (Graphiti)           │
│                  │     │   Entities + Facts + Episodes │
│                  │     │   Vector search (3072-dim)    │
│                  │     │                              │
│                  │     │ Fact Ledger (JSON / PostgreSQL)│
│                  │     │   Claims + Evidence + Scores  │
│                  │     │   Embeddings + Fingerprints   │
│                  │     │                              │
│                  │     │ Authority Registry (JSON)     │
│                  │     │   Per-source, per-topic       │
│                  │     │   weights                     │
│                  │     └──────────────────────────────┘
└──────────────────┘
```

## Project Structure

```
src/noodly/
├── models/          # Pydantic models (SourceArtifact, Claim, ClaimEvidence)
├── graph/           # Graphiti + FalkorDB brain wrapper
├── connectors/      # Data source connectors (local filesystem)
├── parsing/         # Multi-format document parsing (MarkItDown, Docling)
├── extraction/      # LLM-powered claim extraction with parallel dispatch
├── scoring/         # Truth scoring, authority registry, semantic dedup, topic classification
├── resolution/      # Conflict detection, resolution strategies, audit trail
├── projection/      # Markdown projection with emission planning, GitLab projection
├── dispatch/        # Event dispatch system, GitLab handler
├── tracking/        # Change tracking and changelog
├── caching/         # Parse and extraction caching
├── agents/          # Agentic extraction QA and graph population
├── storage/         # Storage backends (JSON, PostgreSQL)
├── server/          # MCP server for AI agent integration
├── pipeline.py      # End-to-end orchestrator
├── cli.py           # Click CLI (30+ commands)
└── config.py        # Settings via pydantic-settings (NOODLY_* env vars)
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

CI runs automatically on push via GitHub Actions (`ruff check` + `pytest` on Python 3.11/3.12).

## Roadmap

- [x] **Phase 1** — Core loop (ingest → extract → score → project)
- [x] **Phase 2** — Truth maintenance, authority registry, bi-temporal, persistent hashes, claim dedup
- [x] **Phase 3** — Multi-format parsing, diff-aware change tracking, caching, agentic extraction
- [x] **Phase 4** — Semantic dedup, conflict resolution, event dispatch, GitLab MR handler, PostgreSQL backend
- [x] **Phase 5** — Semantic claim merging, LLM prompt improvements, Docling integration
- [x] **Phase 6** — CI/CD, GitLab knowledge projection, Docling OCR fix
- [x] **Phase 7** — Parallel LLM dispatch, document consolidation, topic-aware authority, emission planning
- [x] **Phase 8** — Ingestion-time semantic dedup, claim promotion pipeline, graph node embeddings
- [ ] **Phase 9+** — Additional connectors (Slack, Google Drive, Notion, Email, GitHub), web dashboard, multi-tenant

## Built With

- [Graphiti](https://github.com/getzep/graphiti) — temporal knowledge graph engine
- [FalkorDB](https://www.falkordb.com/) — graph database
- [OpenAI](https://openai.com/) — LLM extraction and embeddings
- [MarkItDown](https://github.com/microsoft/markitdown) — document-to-Markdown conversion
- [Docling](https://github.com/DS4SD/docling) — advanced document parsing with OCR (optional)
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — configuration management
- [Click](https://click.palletsprojects.com/) + [Rich](https://rich.readthedocs.io/) — CLI framework

## License

MIT — Copyright (c) 2026 Aniruddha Adhikary

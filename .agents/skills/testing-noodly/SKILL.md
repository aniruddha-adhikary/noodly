---
name: testing-noodly
description: E2E testing for noodly pipeline (ingest, extract, score, project). Use when verifying pipeline, agent, caching, or CLI changes.
---

# Testing Noodly E2E

## Devin Secrets Needed
- `NOODLY_OPENAI_API_KEY` — required for claim extraction and agent tests

## Environment Setup

1. Start FalkorDB:
   ```bash
   cd /home/ubuntu/repos/noodly && docker compose up -d
   ```
2. Verify container is running:
   ```bash
   docker compose ps  # should show falkordb healthy
   ```
3. Clean previous state (if needed):
   ```bash
   rm -rf brain/ inbox/ && mkdir -p inbox
   ```
4. Initialize brain:
   ```bash
   noodly init
   ```

## Core Pipeline Testing

1. **Create test documents** in `inbox/` — use .md, .txt, .csv files with varied content (company policies, meeting notes, data tables)
2. **Initial ingest**: `noodly ingest` — verify artifacts, claims, and projected file counts are >0
3. **Stats check**: `noodly stats` — verify claim distribution across knowledge classes (stable, stateful, process)
4. **Search**: `noodly search "<term>"` — verify semantic search returns relevant results
5. **Claims**: `noodly claims --limit 20` — verify claims have scores, status, and evidence counts

## Cache Testing

- `noodly cache stats` — shows parse, extraction, and agent decision cache counts
- `noodly cache clear --level parse` — clears only parse cache
- `noodly cache clear --level extraction` — clears only extraction cache
- `noodly cache clear` — clears all caches
- After re-ingest of a modified file, pipeline should report `N cached` chunks (unchanged sections served from extraction cache)

## Change Tracking Testing

1. **Unchanged re-ingest**: run `noodly ingest` without modifying files — should process 0 files (hash dedup)
2. **Modified file**: edit a file, run `noodly ingest` — should detect and process only the changed file
3. **Changelog**: `noodly changelog -n 10` — shows recent events (document_added, document_modified, claim_added)
4. **Source filter**: `noodly changelog --source <path>` — filters events to a specific source file

## Agent Testing

Agents are OFF by default. Enable via environment variables:

### QA Agent
```bash
NOODLY_ENABLE_QA_AGENT=true noodly ingest
```
- Create a document with intentionally malformed tables (column count mismatches, missing headers)
- QA agent should detect issues and report them with severity ratings
- Direct test: use `ExtractionQAAgent.review(markdown, content_diff)` to test without full pipeline

### Graph Agent
```bash
NOODLY_ENABLE_GRAPH_AGENT=true noodly ingest
```
- Ingest documents with overlapping entities across files
- Graph agent should discover relationships and gaps
- Check `noodly changelog` for events with `agent="graph_population_agent"`

## File Format Support

Supported: .md, .txt, .csv, .html, .pdf, .docx, .xlsx, .pptx (via MarkItDown)
Not supported: .exe, .tar.gz, .dll, .bin (correctly rejected by connector)

## Common Issues

- **FalkorDB connection noise**: `Connection closed by server` logs during init are cosmetic — retry is automatic
- **Graphiti edge warnings**: `Target entity not found in nodes for edge relation` is from Graphiti internals, not noodly code
- **Slow first ingest**: Initial ingest with OpenAI API calls can take 2-5 minutes for 4+ files (Graphiti episode creation is the bottleneck)
- **Decision cache at 0**: Normal if test documents don't have entity name aliases — the merge cache only populates when the graph agent identifies alias pairs

## Unit Tests

```bash
pytest  # runs all 109 tests
pytest tests/test_parsing.py  # parsing + chunking
pytest tests/test_tracking.py  # content diff + claim diff + changelog
pytest tests/test_caching.py  # cache manager + layers
pytest tests/test_agents.py  # QA + graph agents
```

## Lint

```bash
ruff check src/
ruff format --check src/
```

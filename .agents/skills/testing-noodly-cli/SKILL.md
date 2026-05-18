---
name: testing-noodly-cli
description: Test noodly CLI commands and storage backends (JSONBackend/GraphitiBackend). Use when verifying CLI, ledger, or storage changes.
---

# Testing Noodly CLI & Storage Backends

## Devin Secrets Needed

- `NOODLY_OPENAI_API_KEY` — required for any command that initializes Brain (search, add, ingest, migrate). Can use `fake` for JSON-backend-only testing.
- FalkorDB must be running (`docker compose up -d`) for GraphitiBackend testing.

## Environment Setup

1. Install the project: `pip install -e ".[dev]"`
2. For GraphitiBackend testing: `docker compose up -d` (starts FalkorDB on ports 6379/3000)
3. Set env vars:
   - `NOODLY_BRAIN_DIR` — directory for ledger.json and projected Markdown
   - `NOODLY_OPENAI_API_KEY` — OpenAI key (or `fake` for JSON-only testing)
   - `NOODLY_USE_GRAPHITI_BACKEND=true` — to test GraphitiBackend path

## Testing JSON Backend Path (no external deps)

These CLI commands work with just `NOODLY_BRAIN_DIR` and `NOODLY_OPENAI_API_KEY=fake`:

```bash
# Create a test brain dir
mkdir -p /tmp/test_brain
export NOODLY_BRAIN_DIR=/tmp/test_brain
export NOODLY_OPENAI_API_KEY=fake

# Populate test data via Python script
python3 -c "
from pathlib import Path
from noodly.models.claims import Claim, ClaimEvidence, ClaimStatus, KnowledgeClass
from noodly.storage.json_backend import JSONBackend

backend = JSONBackend(Path('$NOODLY_BRAIN_DIR/ledger.json'))
ev = ClaimEvidence(artifact_id='12345678-1234-1234-1234-123456789abc', supports=True)
c = Claim(subject='Test', predicate='is', object='working', natural_language='Test is working.', evidence=[ev])
backend.save_claim(c)
print(f'Saved claim {c.id}')
"

# Test CLI commands
noodly claims                    # Should show claims table
noodly claims --status candidate # Should filter by status
noodly claims --status bogus     # Should show "Unknown status" error
noodly stats                     # Should show totals and breakdowns
noodly project                   # Should project Markdown files
noodly migrate                   # Without FalkorDB: shows "Found N claims" then ConnectionRefusedError
```

## Testing GraphitiBackend Path (requires FalkorDB + OpenAI)

```bash
docker compose up -d
export NOODLY_USE_GRAPHITI_BACKEND=true
export NOODLY_OPENAI_API_KEY=<real-key>

# First populate JSON ledger, then migrate
noodly migrate

# Verify claims transferred
noodly claims
noodly stats
```

## Unit Tests

```bash
pytest tests/ -v  # All tests should pass (37+)
```

## Known Issues

- `noodly migrate` crashes with a full Python traceback (`ConnectionRefusedError`) if FalkorDB is not running. This is Brain constructor behavior — it does not gracefully handle missing FalkorDB.
- The `graphiti_core` package module paths may change between versions. If you see `ModuleNotFoundError`, check the installed package structure (e.g., `graphiti_core.embedder` vs `graphiti_core.embedder_client`).
- `NOODLY_OPENAI_API_KEY` must be set even for JSON-backend commands because `get_settings()` validates it. Use `fake` as a placeholder.

## Lint

```bash
ruff check src/ tests/
ruff format --check src/
```

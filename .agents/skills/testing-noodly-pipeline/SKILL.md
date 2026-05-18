---
name: testing-noodly-pipeline
description: End-to-end testing procedures for the noodly Company Brain pipeline. Use when verifying ingestion, claim extraction, semantic dedup, auto-promotion, or quality scoring changes.
---

# Testing the Noodly Pipeline E2E

## Devin Secrets Needed
- `NOODLY_OPENAI_API_KEY` — OpenAI API key for LLM extraction and embeddings
- `NOODLY_GITLAB_TOKEN` — GitLab PAT for GitLab MR handler testing (optional)
- `NOODLY_GITLAB_URL` — GitLab base URL (optional)
- `NOODLY_GITLAB_PROJECT_ID` — GitLab project ID (optional)

## Prerequisites
1. FalkorDB must be running: `docker compose up -d` (port 6379 Redis, port 3000 UI)
2. Install deps: `pip install -e ".[dev]"` and `pip install "markitdown[pdf]"`
3. Source env: `source .env` (must contain `NOODLY_OPENAI_API_KEY`)

## Test Environment Setup
Use a separate brain/inbox directory to avoid polluting the main data:
```bash
mkdir -p /home/ubuntu/test-brain /home/ubuntu/test-inbox
# Copy test documents into test-inbox
cp inbox/rfc-http1.md inbox/rfc-http2.md inbox/customs-circular-06-2025-origin.md /home/ubuntu/test-inbox/
```

## Running the Pipeline Programmatically
```python
import asyncio, os
from noodly.config import Settings
from noodly.pipeline import Pipeline

async def run():
    s = Settings(
        openai_api_key=os.environ['NOODLY_OPENAI_API_KEY'],
        brain_dir='/home/ubuntu/test-brain',
        watch_dir='/home/ubuntu/test-inbox',
        falkordb_host='localhost',
        enable_semantic_dedup=True,
        embedding_model='text-embedding-3-large',
    )
    p = Pipeline(s)
    await p.initialize()
    stats = await p.run()
    await p.close()
    print(f'Stats: {stats}')

asyncio.run(run())
```

## Key Test Scenarios

### 1. Source Artifact Provenance
After ingestion, inspect `ledger.json` — every `ClaimEvidence` should have a non-empty `source_artifact` field matching the source filename.

### 2. Temporal Bounds
Check `valid_from` population rate. With the enhanced prompt, expect >50% of claims to have `valid_from`. Look for supersession claims (`predicate` contains "supersedes", "obsoletes", "updates").

### 3. Entity Aliases
Test with documents containing parenthetical abbreviations like "Singapore Land Authority (SLA)". Expect claims with `predicate == "is alias of"`.

### 4. Semantic Dedup Merge
Create a near-duplicate document (reworded version of an existing doc), then re-ingest. Check pipeline stats for `semantic_merged > 0`. Verify merged claims have 2+ evidence entries with different `source_artifact` values.

### 5. Auto-Promotion
After semantic dedup merge, claims with 2+ independent sources should be promoted to `corroborated`. Inspect `status` field in `ledger.json`.

### 6. Quality Score
Test `ParsedDocument.quality_score` with different content types. Expected ordering: well-structured markdown > plain text > garbled > binary garbage > empty.

## Common Issues
- **Semantic dedup threshold (0.92) might be too high** — if near-duplicate claims aren't merging, consider lowering `NOODLY_SEMANTIC_DEDUP_THRESHOLD` to 0.85-0.88.
- **Entity aliases might not appear** if the LLM doesn't detect abbreviation patterns. This is expected — test with explicit parenthetical abbreviations.
- **Auto-promotion requires going through `add_claim()` or explicit `_auto_promote()` call** — if you see merged claims stuck at `candidate`, check whether the merge path calls `_auto_promote()`.
- **FalkorDB connection errors** are cosmetic log noise during testing — claims are stored in the JSON ledger regardless.
- **Extraction cache** — if re-ingesting the same document, the extraction cache may serve stale results. Clear with `noodly cache clear` or delete the cache directory.

## Verification Commands
```python
import json
data = json.loads(open('/path/to/ledger.json').read())
print(f'Total claims: {len(data)}')
print(f'With source_artifact: {sum(1 for c in data for ev in c["evidence"] if ev.get("source_artifact"))}')
print(f'With valid_from: {sum(1 for c in data if c.get("valid_from"))}')
print(f'Corroborated: {sum(1 for c in data if c.get("status") == "corroborated")}')
print(f'Supersession: {sum(1 for c in data if any(w in c.get("predicate","").lower() for w in ["supersede","obsolete","update"]))}')
```

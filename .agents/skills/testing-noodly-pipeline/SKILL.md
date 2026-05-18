---
name: testing-noodly-pipeline
description: End-to-end testing procedures for the noodly Company Brain pipeline. Use when verifying ingestion, claim extraction, semantic dedup, auto-promotion, quality scoring, parallel dispatch, document consolidation, topic classification, authority config, or emission planning changes.
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

### 7. Parallel LLM Dispatch
Run `noodly -v ingest` and check logs for:
- `Dispatcher: submitting X jobs (max_concurrent=N)` — confirms dispatcher was used
- `Dispatcher complete: X jobs, Y succeeded, avg Zms` — confirms parallel execution
If these messages are absent, the pipeline may have fallen back to sequential processing.

### 8. Document Consolidation (Projection)
Run `noodly project --full` and verify:
- `entities/` dir — one .md file per subject with claims grouped by topic
- `topics/` dir — one .md file per LLM-classified topic
- `sources/` dir — one .md file per source document
- `index.md` — dashboard with Summary, Entities, Topics, Sources sections
- **No** `claims/` directory (old format should not be generated)

### 9. File-to-File Cross-Linking
In entity pages, verify:
- Claims grouped under topic headings with `../topics/*.md` links
- "## Related Entities" section with links to other entity files
- "## Source Documents" section with `../sources/*.md` links
- Source provenance quoted in blockquotes

In topic pages: links back to `../entities/*.md`
In source pages: numbered claim list with `→ [entity](../entities/*.md)` links

### 10. Emission Planner & Manifest
- After `noodly project --full`, check `_manifest.json` exists with:
  - `version: 1`, `files` dict, `claim_to_files` mapping, `last_emission` timestamp
  - Each file entry: `content_hash` (16-char hex), `claim_ids` (list), `last_written` (ISO date)
- **Note:** Incremental mode may still re-write all files because `last_updated` timestamps in frontmatter change the content hash. This is a known limitation — not a bug in the planner.

### 11. Topic-Aware Authority
Test the full CLI lifecycle:
```bash
noodly authority set customs.gov.sg 0.7                    # flat weight
noodly authority set customs.gov.sg 0.95 --topic trade     # topic-specific
noodly authority list                                       # should show (default) + trade rows
noodly authority remove customs.gov.sg --topic trade        # remove topic only
```
Verify `authority.json` on disk supports mixed format: `{"source": {"_default": 0.7, "trade": 0.95}}` for topic-aware, `{"source": 0.8}` for flat.

### 12. Topic Classifier
Run with `NOODLY_ENABLE_TOPIC_CLUSTERING=true` and verify:
- `topic_cache.json` created with claim_id → topic list mappings
- Topic names are lowercase-hyphenated (no spaces, no uppercase)
- `topics/` dir has matching .md files

### 13. Config Env Vars (Phase 7)
Verify these settings respond to env var overrides:
```
NOODLY_LLM_MAX_CONCURRENT (default: 8)
NOODLY_LLM_RATE_LIMIT_RPM (default: 500)
NOODLY_LLM_RETRY_MAX (default: 3)
NOODLY_LLM_REQUEST_TIMEOUT (default: 30.0)
NOODLY_EMISSION_MODE (default: incremental)
NOODLY_ENABLE_TOPIC_CLUSTERING (default: true)
NOODLY_TOPIC_MODEL (default: gpt-4o-mini)
NOODLY_AUTHORITY_TOPIC_INFERENCE (default: llm)
```

## Common Issues
- **Semantic dedup threshold (0.92) might be too high** — if near-duplicate claims aren't merging, consider lowering `NOODLY_SEMANTIC_DEDUP_THRESHOLD` to 0.85-0.88.
- **Entity aliases might not appear** if the LLM doesn't detect abbreviation patterns. This is expected — test with explicit parenthetical abbreviations.
- **Auto-promotion requires going through `add_claim()` or explicit `_auto_promote()` call** — if you see merged claims stuck at `candidate`, check whether the merge path calls `_auto_promote()`.
- **FalkorDB connection errors** are cosmetic log noise during testing — claims are stored in the JSON ledger regardless.
- **Extraction cache** — if re-ingesting the same document, the extraction cache may serve stale results. Clear with `noodly cache clear` or delete the cache directory.
- **CLI display cosmetic** — `authority set --topic` and `authority remove --topic` may show generic output (e.g., `Set source = weight`) instead of topic-specific output. The underlying behavior is correct.
- **Incremental emission re-writes all files** — because frontmatter timestamps change content hashes. This is a known trade-off, not a planner bug.

## Verification Commands
```python
import json
data = json.loads(open('/path/to/ledger.json').read())
print(f'Total claims: {len(data)}')
print(f'With source_artifact: {sum(1 for c in data for ev in c["evidence"] if ev.get("source_artifact"))}')
print(f'With valid_from: {sum(1 for c in data if c.get("valid_from"))}')
print(f'Corroborated: {sum(1 for c in data if c.get("status") == "corroborated")}')
print(f'Supersession: {sum(1 for c in data if any(w in c.get("predicate","").lower() for w in ["supersede","obsolete","update"]))}')

# Phase 7 manifest verification
m = json.loads(open('/path/to/_manifest.json').read())
print(f'Manifest files: {len(m["files"])}')
print(f'Claim-to-file mappings: {len(m.get("claim_to_files", {}))}')

# Topic cache verification
tc = json.loads(open('/path/to/topic_cache.json').read())
topics = set(t for ts in tc.values() for t in ts)
print(f'Topics: {sorted(topics)}')
bad = [t for t in topics if t != t.lower() or " " in t]
print(f'Format violations: {bad or "NONE"}')
```

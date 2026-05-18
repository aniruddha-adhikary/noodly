---
name: noodly-testing
description: End-to-end testing procedures for noodly knowledge brain — covers infrastructure setup, document ingestion, provenance verification, cross-reference checks, and QnA validation.
---

# Noodly Testing Skill

## Prerequisites

1. **FalkorDB**: `docker compose up -d` (Redis on 6379, UI on 3000)
2. **OpenAI API key**: Set `NOODLY_OPENAI_API_KEY` in environment
3. **PDF support**: `pip install "markitdown[pdf]"` for direct PDF ingestion
4. **Install dev deps**: `pip install -e ".[dev]"`

## Unit Tests

```bash
pytest  # 194 tests covering models, authority, ledger, connector, projector, pipeline, parsing, tracking, caching, agents, Phase 4 features
```

## E2E Test Procedure

### 1. Initialize Brain
```bash
noodly init
```

### 2. Place Documents in inbox/
Supported formats: .md, .txt, .pdf, .docx, .xlsx, .pptx, .csv, .html

### 3. Ingest Documents
```bash
noodly ingest
```
Expect: claims extracted per chunk, projected files created in brain/

### 4. Verify Stats
```bash
noodly stats
```
Check: total claims, knowledge class distribution (stable, stateful, process, tacit)

### 5. Search / QnA
```bash
noodly search "your query here"
```
Verify: entities returned with summaries, facts with created/expired timestamps

### 6. Re-ingest (Idempotency)
```bash
noodly ingest
```
Expect: 0 files processed (hash-based dedup via .hashes.json)

### 7. Verify Projected Files
```bash
ls brain/entities/  # Entity pages with claims and source spans
ls brain/claims/    # Claims organized by knowledge class
cat brain/index.md  # Index of all entities
```

## Phase 4 Feature Tests

### Conflict Detection
```bash
NOODLY_ENABLE_CONFLICT_RESOLUTION=true noodly conflicts detect
noodly conflicts list
```

### Event Dispatch
```bash
NOODLY_ENABLE_EVENT_DISPATCH=true noodly dispatch stats
```

### GitLab Integration
Requires: `NOODLY_GITLAB_TOKEN`, `NOODLY_GITLAB_URL`, `NOODLY_GITLAB_PROJECT_ID`

## Advanced Document Testing

### Document Types to Test
1. **Scientific papers** (arXiv PDFs) — test cross-references (e.g., BERT cites Transformer)
2. **Government circulars** (customs.gov.sg) — test update chains (Notice updates Circular)
3. **RFCs** — test supersession chains (RFC 2616 → 7230-7235 → 9110-9112)
4. **PRDs** — test version supersession (v1.0 → v2.0)

### Provenance Verification
Check brain/ledger.json:
- Every claim should have `evidence[].artifact_id` linking to source document
- Every claim should have `evidence[].source_span` with exact text excerpt
- 12 unique artifact IDs = 12 documents ingested

### Cross-Reference Verification
Search for entities that span multiple documents:
- "Transformer" should appear in both Attention paper and BERT paper claims
- "Circular No: 06/2021" should link to both the original circular and Notice 07/2025
- RFC entities should show obsolescence relationships

## Clean State for Re-testing
```bash
rm -rf brain/ inbox/.hashes.json
noodly init
```

## Lint
```bash
ruff check src/
ruff format --check src/
```

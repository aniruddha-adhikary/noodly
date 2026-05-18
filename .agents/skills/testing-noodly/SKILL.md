---
name: testing-noodly
description: Test noodly's knowledge pipeline end-to-end — ingestion, extraction, projection, GitLab sync, and CI. Use when verifying pipeline changes, projection features, or parser fixes.
---

# Testing Noodly

## Devin Secrets Needed
- `NOODLY_OPENAI_API_KEY` — OpenAI API key for LLM extraction
- `NOODLY_GITLAB_TOKEN` — GitLab personal access token with `api` scope
- `NOODLY_GITLAB_URL` — GitLab base URL (e.g., `https://gitlab.com/aniruddha-adhikary/noodle-test`)
- `NOODLY_GITLAB_PROJECT_ID` — GitLab project numeric ID (e.g., `82292054`)

## Environment Setup

1. Start FalkorDB: `docker compose up -d` (runs on port 6379/Redis + 3000/UI)
2. Install dev deps: `pip install -e ".[dev]"`
3. Create test directories:
   ```bash
   mkdir -p /home/ubuntu/test-brain /home/ubuntu/test-inbox
   ```
4. Place test documents in `/home/ubuntu/test-inbox/` (use real docs from arXiv, customs.gov.sg, etc.)

## Running Tests

### Unit Tests
```bash
pytest tests/ -x -q  # All tests (270+)
ruff check src/ tests/  # Lint
```

### Pipeline E2E
```bash
# Basic ingest
NOODLY_BRAIN_DIR=/home/ubuntu/test-brain NOODLY_WATCH_DIR=/home/ubuntu/test-inbox noodly ingest

# With GitLab projection auto-sync
NOODLY_BRAIN_DIR=/home/ubuntu/test-brain NOODLY_WATCH_DIR=/home/ubuntu/test-inbox NOODLY_ENABLE_GITLAB_PROJECTION=true noodly ingest
```

### GitLab Projection
```bash
# Dry run — see what files would be synced
noodly gitlab diff

# Full sync — push all claims to GitLab
noodly gitlab sync -m "commit message"

# Incremental push — only specific subjects
noodly gitlab push -s "HTTP/2" -m "commit message"

# Idempotency check — re-run sync, expect 0 changes
noodly gitlab sync -m "should be no-op"
```

### Verifying GitLab API
Use the GitLab API to verify files were created:
```bash
curl -H "PRIVATE-TOKEN: $NOODLY_GITLAB_TOKEN" \
  "https://gitlab.com/api/v4/projects/$NOODLY_GITLAB_PROJECT_ID/repository/tree?path=knowledge&recursive=true"
```

## Key Testing Patterns

### GitLab Projection Tests
1. **Full sync**: Verify all entity pages + claim files + index.md are created
2. **Idempotency**: Re-running sync with no changes should produce 0 commits
3. **Incremental**: Pushing for a specific subject should only process that subject's files
4. **Content detection**: Changes to volatile frontmatter (`last_updated`, `last_projected`) should NOT trigger updates
5. **Modification detection**: Adding a new claim for a subject should trigger 1 create + 2 updates (entity page + index)

### Docling OCR
- Verify source code: `DocumentConverter(**converter_kwargs)` (not bare `DocumentConverter()`)
- Check imports: `PdfPipelineOptions`, `StandardPdfPipeline` (Docling v2 API)
- OCR disabled path: no `pipeline_options` in kwargs

### CI Workflow
- Check `.github/workflows/ci.yml` exists
- Verify lint job runs `ruff check`
- Verify test job runs `pytest` on Python 3.11 + 3.12 matrix
- Use `git_pr_checks` to verify CI passes on PRs

## Gotchas
- **Ledger JSON format**: `ledger.json` is a flat list of claim dicts. All UUID fields (e.g., `artifact_id`, `id`) must be valid UUIDs — manually editing with non-UUID strings will cause `FactLedger._load()` to fail silently and start fresh.
- **FalkorDB in CI**: Pipeline tests must mock `Brain` at the module level (`patch("noodly.pipeline.Brain")`) because `Brain.__init__` eagerly connects to Redis via `FalkorDriver`.
- **GitLab API auth**: Uses `PRIVATE-TOKEN` header (not Bearer). The `GitLabClient` handles this.
- **Content comparison**: The `_content_changed()` function strips `last_updated` and `last_projected` lines before comparing — this prevents no-op commits from timestamp-only changes.
- **Env vars**: All use `NOODLY_` prefix. Key feature flags: `NOODLY_ENABLE_GITLAB_PROJECTION`, `NOODLY_ENABLE_SEMANTIC_DEDUP`, `NOODLY_ENABLE_CONFLICT_RESOLUTION`, `NOODLY_ENABLE_EVENT_DISPATCH`.
- **Real documents preferred**: Use actual PDFs from arXiv, government circulars from customs.gov.sg, etc. rather than synthetic markdown. Ingest PDFs directly to test the full parsing pipeline.

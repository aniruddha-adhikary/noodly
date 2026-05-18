"""Parallel LLM job dispatcher — concurrent OpenAI API calls with rate limiting.

Wraps ``ClaimExtractor.extract()`` calls in an async job queue with:
- Configurable concurrency (``max_concurrent``)
- Token-bucket rate limiting (``rate_limit_rpm``)
- Exponential backoff on transient errors
- Per-job timeout
- Observability (latency, token usage, error counts)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from uuid import uuid4

from noodly.extraction.extractor import ClaimExtractor
from noodly.models.artifacts import SourceArtifact
from noodly.models.claims import Claim

logger = logging.getLogger(__name__)


@dataclass
class LLMJob:
    """A single extraction job to dispatch."""

    artifact: SourceArtifact
    source_filename: str = ""
    chunk_index: int = 0
    priority: int = 0
    id: str = field(default_factory=lambda: str(uuid4())[:8])


@dataclass
class LLMJobResult:
    """Result of a dispatched job."""

    job_id: str
    claims: list[Claim] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0
    tokens_used: int = 0


@dataclass
class DispatchStats:
    """Aggregate stats for a batch dispatch."""

    total_jobs: int = 0
    succeeded: int = 0
    failed: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0

    @property
    def avg_latency_ms(self) -> int:
        if self.succeeded == 0:
            return 0
        return self.total_latency_ms // self.succeeded

    @property
    def summary(self) -> str:
        parts = [f"{self.total_jobs} jobs"]
        if self.succeeded:
            parts.append(f"{self.succeeded} succeeded")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.succeeded:
            parts.append(f"avg {self.avg_latency_ms}ms")
        if self.total_tokens:
            parts.append(f"{self.total_tokens} tokens")
        return ", ".join(parts)


class _TokenBucket:
    """Simple token-bucket rate limiter for requests per minute."""

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._interval = 60.0 / max(rpm, 1)
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()


class LLMJobDispatcher:
    """Dispatches extraction jobs concurrently with rate limiting and retries.

    Usage::

        dispatcher = LLMJobDispatcher(extractor, max_concurrent=8)
        jobs = [LLMJob(artifact=a, source_filename=f) for a, f in items]
        results = await dispatcher.submit_batch(jobs)
    """

    def __init__(
        self,
        extractor: ClaimExtractor,
        max_concurrent: int = 8,
        rate_limit_rpm: int = 500,
        retry_max: int = 3,
        retry_backoff: float = 1.0,
        request_timeout: float = 30.0,
    ) -> None:
        self._extractor = extractor
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = _TokenBucket(rate_limit_rpm)
        self._retry_max = retry_max
        self._retry_backoff = retry_backoff
        self._request_timeout = request_timeout

    async def submit_batch(self, jobs: list[LLMJob]) -> list[LLMJobResult]:
        """Submit a batch of jobs and wait for all to complete.

        Jobs run concurrently up to ``max_concurrent``. Results are returned
        in the same order as the input jobs.
        """
        if not jobs:
            return []

        logger.info(
            "Dispatcher: submitting %d jobs (max_concurrent=%d)",
            len(jobs),
            self._max_concurrent,
        )

        tasks = [self._execute_job(job) for job in jobs]
        results = await asyncio.gather(*tasks)

        stats = DispatchStats(total_jobs=len(jobs))
        for r in results:
            if r.error is None:
                stats.succeeded += 1
                stats.total_latency_ms += r.latency_ms
                stats.total_tokens += r.tokens_used
            else:
                stats.failed += 1

        logger.info("Dispatcher complete: %s", stats.summary)
        return list(results)

    async def _execute_job(self, job: LLMJob) -> LLMJobResult:
        """Execute a single job with semaphore, rate limiting, and retries."""
        async with self._semaphore:
            for attempt in range(self._retry_max + 1):
                await self._rate_limiter.acquire()
                start = time.monotonic()

                try:
                    claims = await asyncio.wait_for(
                        self._extractor.extract(
                            job.artifact,
                            source_filename=job.source_filename,
                        ),
                        timeout=self._request_timeout,
                    )
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    return LLMJobResult(
                        job_id=job.id,
                        claims=claims,
                        latency_ms=elapsed_ms,
                    )
                except asyncio.TimeoutError:
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    if attempt < self._retry_max:
                        backoff = self._retry_backoff * (2**attempt)
                        logger.warning(
                            "Job %s timed out (attempt %d/%d), retrying in %.1fs",
                            job.id,
                            attempt + 1,
                            self._retry_max + 1,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                    else:
                        total = self._retry_max + 1
                        logger.error("Job %s timed out after %d attempts", job.id, total)
                        return LLMJobResult(
                            job_id=job.id,
                            error=f"Timeout after {self._retry_max + 1} attempts",
                            latency_ms=elapsed_ms,
                        )
                except Exception as exc:
                    elapsed_ms = int((time.monotonic() - start) * 1000)
                    error_str = str(exc)

                    # Retry on transient errors (rate limit, server errors)
                    is_transient = any(
                        indicator in error_str.lower()
                        for indicator in ("429", "rate_limit", "500", "502", "503", "timeout")
                    )

                    if is_transient and attempt < self._retry_max:
                        backoff = self._retry_backoff * (2**attempt)
                        logger.warning(
                            "Job %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            job.id,
                            attempt + 1,
                            self._retry_max + 1,
                            error_str[:100],
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                    else:
                        logger.error(
                            "Job %s failed permanently after %d attempts: %s",
                            job.id,
                            attempt + 1,
                            error_str[:200],
                        )
                        return LLMJobResult(
                            job_id=job.id,
                            error=error_str[:500],
                            latency_ms=elapsed_ms,
                        )

            # Should not reach here, but safety net
            return LLMJobResult(job_id=job.id, error="Exhausted retries")

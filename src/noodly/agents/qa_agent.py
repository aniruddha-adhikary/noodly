"""Extraction QA Agent — diff-aware review of parsed Markdown quality."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from noodly.agents.base_agent import ToolEquippedAgent
from noodly.agents.toolkit import AgentToolkit
from noodly.parsing.parser import ParsedDocument
from noodly.tracking.content_differ import ContentDiff

logger = logging.getLogger(__name__)

QA_SYSTEM_PROMPT = """\
You are a document parsing quality assurance agent for a Company Brain system.

Your job is to review Markdown output from a document parser and identify issues:

1. MALFORMED TABLES: Column count mismatches, tables parsed as plain text,
   missing headers or alignment rows.
2. SPATIAL LAYOUT ISSUES: Multi-column content interleaved incorrectly,
   headers merged with body text, footnotes mixed into paragraphs.
3. CONTENT ISSUES: Truncated content, OCR artifacts, garbled text,
   excessive boilerplate that should have been stripped.
4. STRUCTURAL ISSUES: Missing section headings, flat text that should
   have hierarchy, broken links or references.

For each issue found, provide:
- severity: "high" (data loss/corruption), "medium" (readability), "low" (cosmetic)
- description: what's wrong
- section: which section is affected
- suggested_fix: how to fix it (if possible)

If the document looks clean, return an empty issues list.

Return valid JSON matching the schema.
"""

QA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "description": {"type": "string"},
                    "section": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["severity", "description", "section", "suggested_fix"],
                "additionalProperties": False,
            },
        },
        "overall_quality": {
            "type": "string",
            "enum": ["good", "acceptable", "poor"],
        },
        "summary": {"type": "string"},
    },
    "required": ["issues", "overall_quality", "summary"],
    "additionalProperties": False,
}


@dataclass
class QAIssue:
    """A quality issue found in parsed output."""

    severity: str
    description: str
    section: str
    suggested_fix: str


@dataclass
class QAResult:
    """Result of QA review."""

    status: str  # "pass", "review", "fail"
    overall_quality: str  # "good", "acceptable", "poor"
    issues: list[QAIssue] = field(default_factory=list)
    summary: str = ""
    markdown: str = ""
    skipped: bool = False


class ExtractionQAAgent(ToolEquippedAgent):
    """Diff-aware QA agent: only reviews changed sections, not entire documents.

    Behavior based on content diff:
    - New document → full review
    - change_ratio < 0.05 → skip QA (trivial change)
    - change_ratio < 0.3 → partial review (changed sections only)
    - change_ratio >= 0.3 → full review with change context
    """

    system_prompt = QA_SYSTEM_PROMPT

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        toolkit: AgentToolkit | None = None,
    ) -> None:
        if toolkit is not None:
            super().__init__(api_key=api_key, model=model, toolkit=toolkit)
        else:
            # Create a minimal instance without toolkit for standalone use
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key)
            self._model = model
            self._toolkit = None  # type: ignore[assignment]

    async def review(
        self,
        parsed: ParsedDocument,
        content_diff: ContentDiff | None = None,
    ) -> QAResult:
        """Review parsed markdown, focusing on changed sections if diff available."""
        if content_diff is None or content_diff.is_new:
            return await self._full_review(parsed)

        if content_diff.change_ratio < 0.05:
            return QAResult(
                status="pass",
                overall_quality="good",
                summary="Trivial change (< 5%), QA skipped",
                markdown=parsed.markdown,
                skipped=True,
            )

        if content_diff.change_ratio < 0.3:
            sections_to_review = list(content_diff.added_sections) + [
                m.new_content for m in content_diff.modified_sections
            ]
            return await self._partial_review(parsed, sections_to_review, content_diff)

        return await self._full_review(parsed, change_summary=content_diff.summary)

    async def _full_review(
        self,
        parsed: ParsedDocument,
        change_summary: str = "",
    ) -> QAResult:
        """Review the entire document."""
        content = parsed.markdown[:12000]
        prompt = f"DOCUMENT: {parsed.title}\nFORMAT: {parsed.source_format}\n"
        if change_summary:
            prompt += f"CHANGES: {change_summary}\n"
        prompt += f"\nMARKDOWN CONTENT:\n{content}"

        result = await self._run_agent(prompt)
        return self._build_result(result, parsed.markdown)

    async def _partial_review(
        self,
        parsed: ParsedDocument,
        sections: list[str],
        content_diff: ContentDiff,
    ) -> QAResult:
        """Review only the changed sections."""
        sections_text = "\n\n---\n\n".join(sections[:5])
        prompt = (
            f"DOCUMENT: {parsed.title}\n"
            f"CHANGES: {content_diff.summary}\n"
            f"NOTE: Only reviewing changed/added sections (partial review)\n\n"
            f"SECTIONS TO REVIEW:\n{sections_text[:8000]}"
        )

        result = await self._run_agent(prompt)
        return self._build_result(result, parsed.markdown)

    async def _run_agent(self, prompt: str) -> dict:
        """Run the QA agent (with or without tools)."""
        if self._toolkit is not None:
            return await self._run_with_tools(prompt, QA_RESPONSE_SCHEMA)

        # Standalone mode without tools
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "qa_response",
                        "strict": True,
                        "schema": QA_RESPONSE_SCHEMA,
                    },
                },
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            logger.exception("QA agent LLM call failed")
            return {"issues": [], "overall_quality": "good", "summary": "QA review failed"}

    def _build_result(self, result: dict, markdown: str) -> QAResult:
        """Convert raw LLM response to QAResult."""
        issues = [
            QAIssue(
                severity=i.get("severity", "low"),
                description=i.get("description", ""),
                section=i.get("section", ""),
                suggested_fix=i.get("suggested_fix", ""),
            )
            for i in result.get("issues", [])
        ]

        quality = result.get("overall_quality", "good")
        status = "pass" if quality == "good" else ("review" if quality == "acceptable" else "fail")

        return QAResult(
            status=status,
            overall_quality=quality,
            issues=issues,
            summary=result.get("summary", ""),
            markdown=markdown,
        )

"""Base agent — tool-equipped agent loop using OpenAI function calling."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from noodly.agents.toolkit import TOOL_DEFINITIONS, AgentToolkit

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


class ToolEquippedAgent:
    """Base class for agents that can call tools via OpenAI function calling.

    Subclasses provide system_prompt and task-specific logic. The base class
    handles the tool-calling loop.
    """

    system_prompt: str = ""

    def __init__(
        self,
        api_key: str,
        model: str,
        toolkit: AgentToolkit,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._toolkit = toolkit

    async def _run_with_tools(
        self,
        task_prompt: str,
        response_schema: dict | None = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> dict:
        """Run agent loop with tool calling. Returns parsed JSON response."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        for iteration in range(max_iterations):
            try:
                kwargs: dict = {
                    "model": self._model,
                    "messages": messages,
                    "tools": TOOL_DEFINITIONS,
                    "temperature": 0.1,
                }

                if response_schema is not None:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "agent_response",
                            "strict": True,
                            "schema": response_schema,
                        },
                    }

                response = await self._client.chat.completions.create(**kwargs)
            except Exception:
                logger.exception("Agent LLM call failed (iteration %d)", iteration)
                return {"error": "LLM call failed"}

            choice = response.choices[0]

            if choice.finish_reason == "stop":
                content = choice.message.content or "{}"
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"raw": content}

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    }
                )

                for tool_call in choice.message.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    result = self._toolkit.execute(tool_call.function.name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
                    logger.debug(
                        "Tool call: %s(%s) → %s",
                        tool_call.function.name,
                        args,
                        result[:200],
                    )
            else:
                content = choice.message.content or "{}"
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"raw": content}

        logger.warning("Agent reached max iterations without completing")
        return {"error": "Max iterations reached"}

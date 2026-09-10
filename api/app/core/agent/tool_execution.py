"""单次工具调用的共享执行契约。"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import replace

from langchain_core.tools import BaseTool

from app.core.agent.tool_contract import (
    TOOL_CACHEABLE_METADATA_KEY,
    TOOL_READ_ONLY_METADATA_KEY,
    ToolCall,
    ToolExecutionError,
    ToolOutcome,
)
from app.core.agent.tracing import get_tracer


def format_tool_content(observation: object) -> str:
    """把普通工具或 MCP 返回值规范化为供模型消费的文本。"""
    if isinstance(observation, str):
        text = observation.strip()
        if text and text[0] in "[{(" and text[-1] in "]})":
            try:
                parsed = ast.literal_eval(text)
                if not isinstance(parsed, str):
                    return format_tool_content(parsed)
            except (ValueError, SyntaxError):
                pass
        return text

    if isinstance(observation, list):
        parts: list[str] = []
        for item in observation:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            try:
                parts.append(json.dumps(item, ensure_ascii=False, indent=2))
            except (TypeError, ValueError):
                parts.append(str(item))
        return "\n\n".join(part.strip() for part in parts if part)

    if isinstance(observation, dict):
        text = observation.get("text")
        if isinstance(text, str):
            return text
        try:
            return json.dumps(observation, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(observation)

    text = getattr(observation, "text", None)
    if isinstance(text, str):
        return text
    return str(observation)


class ToolExecutor:
    """统一单次执行、结果规范化、异常、缓存资格与 trace 元数据。"""

    def __init__(
        self,
        tools: list[BaseTool],
        *,
        stats_holder: dict[str, dict] | None = None,
        tracer=None,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._stats_holder = stats_holder if stats_holder is not None else {}
        self._tracer = tracer if tracer is not None else get_tracer()
        self._cache: dict[str, ToolOutcome] = {}

    @staticmethod
    def _policy(tool: BaseTool) -> tuple[bool, bool]:
        metadata = tool.metadata or {}
        read_only = metadata.get(TOOL_READ_ONLY_METADATA_KEY) is True
        cacheable = read_only and metadata.get(TOOL_CACHEABLE_METADATA_KEY) is True
        return read_only, cacheable

    @staticmethod
    def _cache_key(call: ToolCall) -> str | None:
        try:
            args = json.dumps(
                call.validated_args,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return None
        return f"{call.tool_key}:{args}"

    async def execute(self, call: ToolCall, *, attempt: int = 1) -> ToolOutcome:
        started = time.monotonic()
        tool = self._tools.get(call.tool_key)
        read_only = False
        cacheable = False
        if tool is not None:
            read_only, cacheable = self._policy(tool)
        cache_key = self._cache_key(call) if cacheable else None
        query = call.validated_args.get("query", "")

        async with self._tracer.span(
            f"工具:{call.tool_key}",
            span_type="tool_call",
            attributes={
                "comet.tool.name": call.tool_key,
                "comet.tool.call_id": call.call_id,
                "comet.tool.query": str(query)[:200],
            },
        ) as span:
            cached = self._cache.get(cache_key) if cache_key is not None else None
            if cached is not None:
                outcome = replace(cached, attempt=attempt, latency_ms=0, cached=True, stats={})
            elif tool is None:
                outcome = ToolOutcome(
                    status="error",
                    content=f"未知工具：{call.tool_key}",
                    error_code="unknown_tool",
                    attempt=attempt,
                    latency_ms=int((time.monotonic() - started) * 1000),
                )
            else:
                try:
                    raw = await tool.ainvoke(call.validated_args)
                except ToolExecutionError as exc:
                    outcome = ToolOutcome(
                        status="error",
                        content=str(exc),
                        error_code=exc.error_code,
                        retryable=exc.retryable,
                        artifact_ref=exc.artifact_ref,
                        attempt=attempt,
                    )
                except TimeoutError as exc:
                    detail = str(exc).strip()
                    outcome = ToolOutcome(
                        status="error",
                        content=f"工具执行超时{f'：{detail}' if detail else ''}",
                        error_code="tool_timeout",
                        retryable=read_only,
                        attempt=attempt,
                    )
                except Exception as exc:  # noqa: BLE001 - 工具异常统一转 outcome
                    outcome = ToolOutcome(
                        status="error",
                        content=f"工具执行失败：{exc}",
                        error_code="tool_execution_failed",
                        attempt=attempt,
                    )
                else:
                    if isinstance(raw, ToolOutcome):
                        outcome = replace(raw, attempt=attempt, cached=False)
                    else:
                        outcome = ToolOutcome(
                            status="success",
                            content=format_tool_content(raw),
                            attempt=attempt,
                        )

                stats = self._stats_holder.pop(call.tool_key, {})
                outcome = replace(
                    outcome,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    stats=stats,
                )
                if cache_key is not None and outcome.status == "success":
                    self._cache[cache_key] = outcome

            span.set_payload("status", outcome.status)
            span.set_payload("cached", outcome.cached)
            span.set_payload("attempt", outcome.attempt)
            span.set_payload("latency_ms", outcome.latency_ms)
            span.set_payload("output_chars", len(outcome.content))
            if outcome.content:
                span.set_payload("output_preview", outcome.content[:600])
            if outcome.error_code:
                span.set_payload("error_code", outcome.error_code)
            if query:
                span.set_payload("tool_query", str(query)[:300])
            if outcome.status == "error":
                span.mark_error(outcome.content)
            return outcome


__all__ = [
    "TOOL_CACHEABLE_METADATA_KEY",
    "TOOL_READ_ONLY_METADATA_KEY",
    "ToolCall",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolOutcome",
    "format_tool_content",
]

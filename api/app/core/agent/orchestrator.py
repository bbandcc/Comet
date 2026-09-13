"""Agent 编排：方案B 双路径工具循环，产出统一事件流。

- 强模型（支持 function calling）：bind_tools + 流式工具循环，原生决定调用哪个工具。
- 弱模型：ToolOrchestrator（prompt 模拟 ReAct），解析 Action/Action Input 手动调工具。

两条路径都产出统一事件 dict：
  {"type": "tool_start", "call_id", "tool", "args", "query"} /
  {"type": "tool_result", "call_id", "tool", "args", "status", "text", ...} /
  {"type": "token", "text"} /
  {"type": "final", "status", "stop_reason", "text", "partial_answer"}
引用由工具执行时写入外部传入的 citations 列表，编排结束后由调用方读取。

工具统计（命中数 / 实体数 / 网页数 等）由各工具写入 ctx.stats_holder[tool_key]，
本编排器在产 tool_result 事件时读取并附在事件上，前端 chip 副文动态绑定。
"""

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import nullcontext
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.core.agent.agent_contract import (
    DEFAULT_MAX_MODEL_REQUESTS,
    AgentRunLimits,
    AgentTerminal,
    tool_result_event,
    tool_start_event,
)
from app.core.agent.prompt_renderer import render_agent_prompt
from app.core.agent.tool_execution import (
    ToolCallValidationError,
    ToolExecutor,
    ToolOutcome,
)
from app.core.agent.tracing import get_tracer
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_TOOL_ITERATIONS = DEFAULT_MAX_MODEL_REQUESTS
_ChunkT = TypeVar("_ChunkT")


def _validation_outcome(exc: ToolCallValidationError) -> ToolOutcome:
    return ToolOutcome(status="error", content=str(exc), error_code=exc.error_code)


def _partial_answer(parts: list[str], current: str = "") -> str:
    return "\n".join(part for part in [*parts, current] if part).strip()


def _deadline_terminal(partial_answer: str) -> dict:
    return AgentTerminal(
        "budget_exhausted",
        "total_deadline_exceeded",
        partial_answer=partial_answer,
    ).as_event()


def _tool_schema_text(tool: BaseTool) -> str:
    schema = tool.args_schema
    if schema is None:
        schema = tool.get_input_schema()
    if not isinstance(schema, dict):
        schema = schema.model_json_schema()
    return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))


async def _stream_with_active_deadline(
    stream: AsyncIterator[_ChunkT],
    timeout_seconds: float,
    clock: Callable[[], float],
) -> AsyncGenerator[_ChunkT, None]:
    """限制模型实际等待时间；下游消费 yield 的时间不计入预算。"""
    remaining = timeout_seconds
    iterator = aiter(stream)
    while True:
        if remaining <= 0:
            raise TimeoutError
        started = clock()
        try:
            async with asyncio.timeout(remaining):
                chunk = await anext(iterator)
        except StopAsyncIteration:
            return
        remaining -= max(0.0, clock() - started)
        if remaining <= 0:
            raise TimeoutError
        yield chunk


async def run_function_calling(
    model: ChatOpenAI,
    tools: list[BaseTool],
    messages: list,
    stats_holder: dict[str, dict] | None = None,
    limits: AgentRunLimits | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncGenerator[dict, None]:
    """强模型路径：原生 function calling 流式工具循环。"""
    try:
        model_with_tools = model.bind_tools(tools) if tools else model
    except Exception as exc:  # noqa: BLE001 - provider schema 不兼容属于不可恢复编排错误
        logger.warning("FC 工具绑定失败: %s", exc)
        yield AgentTerminal("failed", "tool_binding_failed").as_event()
        return
    stats_holder = stats_holder if stats_holder is not None else {}
    executor = ToolExecutor(tools, stats_holder=stats_holder)
    limits = limits or AgentRunLimits()
    parameter_corrections = 0
    deadline = clock() + limits.total_deadline_seconds
    partial_parts: list[str] = []
    # 取真实 model_name 用于成本核算(LangChain ChatOpenAI 的 model_name 字段)
    chat_model_name = getattr(model, "model_name", None) or getattr(model, "model", "chat")
    tracer = get_tracer()

    for iteration in range(limits.max_model_requests):
        remaining = deadline - clock()
        if remaining <= 0:
            yield _deadline_terminal(_partial_answer(partial_parts))
            return
        # 每轮 LLM 流式调用包一个 llm_call span,流完后从 usage_metadata 抽 token
        # 抓最后一条 user/tool 消息做请求摘要
        last_msg_text = ""
        for m in reversed(messages):
            content = getattr(m, "content", None)
            if isinstance(content, str) and content:
                last_msg_text = content
                break
        gathered = None
        iter_chunks: list[str] = []
        try:
            # 有工具时先缓冲模型本轮输出，可安全用 hard deadline；无工具直答需跨 yield
            # 保持流式，避免 timeout 上下文误取消消费者自己的 SSE/Redis I/O。
            model_timeout = asyncio.timeout(remaining) if tools else nullcontext()
            async with model_timeout:
                async with tracer.llm_span(
                    f"chat:{chat_model_name} (轮 {iteration + 1})",
                    model_name=chat_model_name,
                    attributes={
                        "comet.chat.iteration": iteration + 1,
                        "comet.chat.tools_bound": len(tools),
                    },
                ) as lsp:
                    lsp.set_payload("messages_count", len(messages))
                    if last_msg_text:
                        lsp.set_payload("request_summary", last_msg_text[:600])
                    model_stream = model_with_tools.astream(messages)
                    if not tools:
                        model_stream = _stream_with_active_deadline(
                            model_stream,
                            remaining,
                            clock,
                        )
                    async for chunk in model_stream:
                        if chunk.content:
                            text = (
                                chunk.content
                                if isinstance(chunk.content, str)
                                else str(chunk.content)
                            )
                            iter_chunks.append(text)
                            if not tools:
                                yield {"type": "token", "text": text}
                        gathered = chunk if gathered is None else gathered + chunk
                    usage = getattr(gathered, "usage_metadata", None) or {}
                    in_t = int(usage.get("input_tokens", 0) or 0)
                    out_t = int(usage.get("output_tokens", 0) or 0)
                    cached = int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
                    lsp.set_tokens(
                        input=in_t,
                        output=out_t,
                        cached=cached,
                        model_name=chat_model_name,
                    )
                    tool_calls = getattr(gathered, "tool_calls", None) or []
                    invalid_tool_calls = getattr(gathered, "invalid_tool_calls", None) or []
                    pending_calls = [*tool_calls, *invalid_tool_calls]
                    lsp.set_payload("tool_calls_count", len(pending_calls))
                    iter_text = "".join(iter_chunks)
                    if iter_text:
                        lsp.set_payload("response_preview", iter_text[:600])
                    elif pending_calls:
                        lsp.set_payload(
                            "response_preview",
                            "(本轮无文字输出,触发工具:"
                            + ", ".join(tc.get("name", "?") for tc in pending_calls[:5])
                            + ")",
                        )
        except asyncio.CancelledError:
            yield AgentTerminal(
                "cancelled",
                "external_cancellation",
                partial_answer=_partial_answer(partial_parts, "".join(iter_chunks)),
            ).as_event()
            raise
        except TimeoutError:
            partial = _partial_answer(partial_parts, "".join(iter_chunks))
            if clock() >= deadline:
                yield _deadline_terminal(partial)
            else:
                yield AgentTerminal(
                    "failed", "model_request_timeout", partial_answer=partial
                ).as_event()
            return
        except Exception as exc:  # noqa: BLE001 - 编排不可恢复错误转明确终态
            logger.warning("FC 模型请求失败: %s", exc)
            yield AgentTerminal(
                "failed",
                "model_request_failed",
                partial_answer=_partial_answer(partial_parts, "".join(iter_chunks)),
            ).as_event()
            return

        if not pending_calls:
            # 无工具调用 → 已是最终回答
            final_answer = "".join(iter_chunks)
            if tools:
                for text in iter_chunks:
                    yield {"type": "token", "text": text}
            yield AgentTerminal("completed", "final_answer", final_answer=final_answer).as_event()
            return

        iter_text = "".join(iter_chunks)
        if iter_text:
            partial_parts.append(iter_text)
        # 有工具调用：执行后把结果回灌，继续循环
        messages.append(gathered)
        for tc in pending_calls:
            if clock() >= deadline:
                yield _deadline_terminal(_partial_answer(partial_parts))
                return
            name = tc.get("name", "")
            args = tc.get("args", {}) or {}
            raw_call_id = tc.get("id")
            call_id = str(raw_call_id) if raw_call_id else f"fc-{uuid.uuid4().hex}"
            try:
                call = executor.validate_call(call_id, name, args)
            except ToolCallValidationError as exc:
                outcome = _validation_outcome(exc)
                yield tool_start_event(call_id, name, args, args_validated=False)
                yield tool_result_event(
                    call_id,
                    name,
                    args,
                    outcome,
                    args_validated=False,
                )
                messages.append(
                    ToolMessage(content=outcome.content, tool_call_id=call_id, status="error")
                )
                if exc.error_code == "tool_schema_invalid":
                    yield AgentTerminal(
                        "failed",
                        "tool_schema_invalid",
                        partial_answer=_partial_answer(partial_parts),
                    ).as_event()
                    return
                parameter_corrections += 1
                if parameter_corrections >= limits.max_parameter_corrections:
                    yield AgentTerminal(
                        "budget_exhausted",
                        "parameter_correction_budget_exhausted",
                        partial_answer=_partial_answer(partial_parts),
                    ).as_event()
                    return
                continue
            yield tool_start_event(
                call.call_id,
                call.tool_key,
                call.validated_args,
                args_validated=True,
            )
            try:
                outcome = await executor.execute(
                    call,
                    timeout_seconds=min(
                        limits.tool_timeout_seconds,
                        max(0.001, deadline - clock()),
                    ),
                )
            except asyncio.CancelledError:
                yield AgentTerminal(
                    "cancelled",
                    "external_cancellation",
                    partial_answer=_partial_answer(partial_parts),
                ).as_event()
                raise
            except Exception as exc:  # noqa: BLE001 - executor 基础设施异常才终止 Agent
                logger.warning("FC 工具执行器失败: %s", exc)
                yield AgentTerminal(
                    "failed",
                    "tool_executor_failed",
                    partial_answer=_partial_answer(partial_parts),
                ).as_event()
                return
            yield tool_result_event(
                call.call_id,
                call.tool_key,
                call.validated_args,
                outcome,
                args_validated=True,
            )
            messages.append(
                ToolMessage(
                    content=outcome.content,
                    tool_call_id=call.call_id,
                    status=outcome.status,
                    artifact=outcome.artifact_ref,
                )
            )
            if clock() >= deadline:
                yield _deadline_terminal(_partial_answer(partial_parts))
                return

    # 达到最大迭代仍未收敛：用现有内容兜底
    yield AgentTerminal(
        "budget_exhausted",
        "model_request_budget_exhausted",
        partial_answer=_partial_answer(partial_parts),
    ).as_event()


_ACTION_RE = re.compile(r"Action\s*:\s*(.+)")
_ACTION_INPUT_RE = re.compile(
    r"Action\s*Input\s*:\s*(.*?)(?=\n(?:Thought|Action|Observation|Final\s*Answer)\s*:|\Z)",
    re.DOTALL,
)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.*)", re.DOTALL)


async def run_react(
    model: ChatOpenAI,
    tools: list[BaseTool],
    user_text: str,
    history: list,
    system_prompt: str,
    stats_holder: dict[str, dict] | None = None,
    limits: AgentRunLimits | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncGenerator[dict, None]:
    """弱模型路径：prompt 模拟 ReAct，手动解析并调用工具。"""
    try:
        sys = render_agent_prompt(
            "react.jinja2",
            tools=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "args_schema": _tool_schema_text(tool),
                }
                for tool in tools
            ],
            system_prompt=system_prompt,
        )
    except Exception as exc:  # noqa: BLE001 - 无法表达工具 schema 时不可继续编排
        logger.warning("ReAct 工具 schema 渲染失败: %s", exc)
        yield AgentTerminal("failed", "tool_schema_render_failed").as_event()
        return
    convo: list = [SystemMessage(content=sys), *history, HumanMessage(content=user_text)]
    stats_holder = stats_holder if stats_holder is not None else {}
    executor = ToolExecutor(tools, stats_holder=stats_holder)
    limits = limits or AgentRunLimits()
    parameter_corrections = 0
    deadline = clock() + limits.total_deadline_seconds
    # 取真实 model_name 用于 token / cost 记账
    react_model_name = getattr(model, "model_name", None) or getattr(model, "model", "chat")
    tracer = get_tracer()

    for iteration in range(limits.max_model_requests):
        remaining = deadline - clock()
        if remaining <= 0:
            yield _deadline_terminal("")
            return
        try:
            async with asyncio.timeout(remaining):
                async with tracer.llm_span(
                    f"chat(ReAct):{react_model_name} (轮 {iteration + 1})",
                    model_name=react_model_name,
                    attributes={
                        "comet.chat.iteration": iteration + 1,
                        "comet.chat.mode": "react",
                    },
                ) as lsp:
                    resp = await model.ainvoke(convo)
                    usage = getattr(resp, "usage_metadata", None) or {}
                    in_t = int(usage.get("input_tokens", 0) or 0)
                    out_t = int(usage.get("output_tokens", 0) or 0)
                    cached = int((usage.get("input_token_details") or {}).get("cache_read", 0) or 0)
                    lsp.set_tokens(
                        input=in_t,
                        output=out_t,
                        cached=cached,
                        model_name=react_model_name,
                    )
        except asyncio.CancelledError:
            yield AgentTerminal(
                "cancelled",
                "external_cancellation",
            ).as_event()
            raise
        except TimeoutError:
            if clock() >= deadline:
                yield _deadline_terminal("")
            else:
                yield AgentTerminal("failed", "model_request_timeout").as_event()
            return
        except Exception as exc:  # noqa: BLE001 - 编排不可恢复错误转明确终态
            logger.warning("ReAct 模型请求失败: %s", exc)
            yield AgentTerminal(
                "failed",
                "model_request_failed",
            ).as_event()
            return
        text = resp.content if isinstance(resp.content, str) else str(resp.content)

        final_match = _FINAL_RE.search(text)
        if final_match:
            answer = final_match.group(1).strip()
            yield {"type": "token", "text": answer}
            yield AgentTerminal("completed", "final_answer", final_answer=answer).as_event()
            return

        action_match = _ACTION_RE.search(text)
        input_match = _ACTION_INPUT_RE.search(text)
        if not action_match:
            # 没有 Action 也没有 Final，把整段当回答兜底
            yield {"type": "token", "text": text}
            yield AgentTerminal("completed", "direct_answer", final_answer=text).as_event()
            return

        if clock() >= deadline:
            yield _deadline_terminal("")
            return
        tool_name = action_match.group(1).strip().splitlines()[0].strip()
        raw_args: object = input_match.group(1).strip() if input_match else user_text
        call_id = f"react-{uuid.uuid4().hex}"
        try:
            call = executor.validate_call(
                call_id,
                tool_name,
                raw_args,
                allow_plain_query=True,
            )
        except ToolCallValidationError as exc:
            outcome = _validation_outcome(exc)
            yield tool_start_event(call_id, tool_name, raw_args, args_validated=False)
            yield tool_result_event(
                call_id,
                tool_name,
                raw_args,
                outcome,
                args_validated=False,
            )
            convo.append(AIMessage(content=text))
            convo.append(HumanMessage(content=f"Observation: {outcome.content}"))
            if exc.error_code == "tool_schema_invalid":
                yield AgentTerminal("failed", "tool_schema_invalid").as_event()
                return
            parameter_corrections += 1
            if parameter_corrections >= limits.max_parameter_corrections:
                yield AgentTerminal(
                    "budget_exhausted",
                    "parameter_correction_budget_exhausted",
                ).as_event()
                return
            continue
        yield tool_start_event(
            call.call_id,
            call.tool_key,
            call.validated_args,
            args_validated=True,
        )
        try:
            outcome = await executor.execute(
                call,
                timeout_seconds=min(
                    limits.tool_timeout_seconds,
                    max(0.001, deadline - clock()),
                ),
            )
        except asyncio.CancelledError:
            yield AgentTerminal(
                "cancelled",
                "external_cancellation",
            ).as_event()
            raise
        except Exception as exc:  # noqa: BLE001 - executor 基础设施异常才终止 Agent
            logger.warning("ReAct 工具执行器失败: %s", exc)
            yield AgentTerminal(
                "failed",
                "tool_executor_failed",
            ).as_event()
            return
        yield tool_result_event(
            call.call_id,
            call.tool_key,
            call.validated_args,
            outcome,
            args_validated=True,
        )
        # 把模型上一轮输出 + Observation 回灌
        convo.append(AIMessage(content=text))
        convo.append(HumanMessage(content=f"Observation: {outcome.content}"))
        if clock() >= deadline:
            yield _deadline_terminal("")
            return

    yield AgentTerminal(
        "budget_exhausted",
        "model_request_budget_exhausted",
    ).as_event()


__all__ = ["run_function_calling", "run_react", "MAX_TOOL_ITERATIONS"]

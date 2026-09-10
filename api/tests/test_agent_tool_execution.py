import asyncio
import unittest
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.agent.orchestrator import run_function_calling, run_react
from app.core.agent.tool_execution import (
    TOOL_CACHEABLE_METADATA_KEY,
    TOOL_READ_ONLY_METADATA_KEY,
    ToolCall,
    ToolExecutionError,
    ToolExecutor,
)
from app.core.agent.tools.base import BUILTIN_REGISTRY
from app.core.agent.tools.registry import build_enabled_tools


class _QueryInput(BaseModel):
    query: str = Field(...)


class _TwoFieldInput(BaseModel):
    query: str = Field(...)
    limit: int = Field(...)


def _fake_tool(name, coroutine, *, read_only: bool, cacheable: bool) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=f"fake {name}",
        args_schema=_QueryInput,
        metadata={
            TOOL_READ_ONLY_METADATA_KEY: read_only,
            TOOL_CACHEABLE_METADATA_KEY: cacheable,
        },
    )


class _ScriptedFunctionCallingModel:
    model_name = "scripted-fc"

    def __init__(self, responses: list[AIMessageChunk]):
        self._responses = list(responses)
        self.seen_messages: list[list] = []

    def bind_tools(self, _tools):
        return self

    async def astream(self, _messages):
        self.seen_messages.append(list(_messages))
        yield self._responses.pop(0)


class _ScriptedReactModel:
    model_name = "scripted-react"

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)

    async def ainvoke(self, _messages):
        return self._responses.pop(0)


async def _collect_events(stream) -> list[dict]:
    return [event async for event in stream]


class ToolExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_only_success_is_cached_with_full_arguments_preserved(self):
        executions = 0

        async def run(query: str) -> str:
            nonlocal executions
            executions += 1
            return f"result:{query}"

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)
        executor = ToolExecutor([tool])
        first_call = ToolCall("call-1", "reader", {"query": "same"})
        second_call = ToolCall("call-2", "reader", {"query": "same"})

        first = await executor.execute(first_call)
        second = await executor.execute(second_call)

        self.assertEqual(first_call.validated_args, {"query": "same"})
        self.assertEqual(first.status, "success")
        self.assertFalse(first.cached)
        self.assertEqual(second.status, "success")
        self.assertTrue(second.cached)
        self.assertEqual(second.content, "result:same")
        self.assertEqual(executions, 1)

    async def test_business_failure_is_error_and_never_cached(self):
        executions = 0

        async def run(query: str) -> str:
            nonlocal executions
            executions += 1
            raise ToolExecutionError(
                f"rejected:{query}",
                error_code="business_rejected",
            )

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)
        executor = ToolExecutor([tool])

        first = await executor.execute(ToolCall("call-1", "reader", {"query": "same"}))
        second = await executor.execute(ToolCall("call-2", "reader", {"query": "same"}))

        self.assertEqual(first.status, "error")
        self.assertEqual(first.error_code, "business_rejected")
        self.assertFalse(first.cached)
        self.assertEqual(second.status, "error")
        self.assertFalse(second.cached)
        self.assertEqual(executions, 2)

    async def test_unexpected_exception_is_error_and_never_cached(self):
        executions = 0

        async def run(query: str) -> str:
            nonlocal executions
            executions += 1
            raise RuntimeError(f"broken:{query}")

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)
        executor = ToolExecutor([tool])

        first = await executor.execute(ToolCall("call-1", "reader", {"query": "same"}))
        second = await executor.execute(ToolCall("call-2", "reader", {"query": "same"}))

        self.assertEqual(first.status, "error")
        self.assertEqual(first.error_code, "tool_execution_failed")
        self.assertFalse(first.cached)
        self.assertEqual(second.status, "error")
        self.assertEqual(executions, 2)

    async def test_timeout_is_error_not_cached_and_not_automatically_retried(self):
        executions = 0

        async def run(query: str) -> str:
            nonlocal executions
            executions += 1
            raise TimeoutError(f"timeout:{query}")

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)
        executor = ToolExecutor([tool])

        first = await executor.execute(ToolCall("call-1", "reader", {"query": "same"}))
        self.assertEqual(executions, 1)
        second = await executor.execute(ToolCall("call-2", "reader", {"query": "same"}))

        self.assertEqual(first.status, "error")
        self.assertEqual(first.error_code, "tool_timeout")
        self.assertTrue(first.retryable)
        self.assertFalse(first.cached)
        self.assertEqual(second.status, "error")
        self.assertEqual(executions, 2)

    async def test_empty_result_is_success(self):
        async def run(query: str) -> str:
            return ""

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)
        outcome = await ToolExecutor([tool]).execute(
            ToolCall("call-empty", "reader", {"query": "nothing"})
        )

        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.content, "")
        self.assertIsNone(outcome.error_code)

    async def test_write_tool_executes_each_identical_call(self):
        executions = 0

        async def run(query: str) -> str:
            nonlocal executions
            executions += 1
            return f"written:{query}"

        tool = _fake_tool("writer", run, read_only=False, cacheable=False)
        executor = ToolExecutor([tool])

        first = await executor.execute(ToolCall("call-1", "writer", {"query": "same"}))
        second = await executor.execute(ToolCall("call-2", "writer", {"query": "same"}))

        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success")
        self.assertFalse(first.cached)
        self.assertFalse(second.cached)
        self.assertEqual(executions, 2)

    async def test_write_timeout_is_not_marked_retryable(self):
        executions = 0

        async def run(query: str) -> str:
            nonlocal executions
            executions += 1
            raise TimeoutError(query)

        tool = _fake_tool("writer", run, read_only=False, cacheable=False)
        outcome = await ToolExecutor([tool]).execute(
            ToolCall("write-timeout", "writer", {"query": "same"})
        )

        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.error_code, "tool_timeout")
        self.assertFalse(outcome.retryable)
        self.assertEqual(executions, 1)

    async def test_unknown_tool_is_error_and_not_cached(self):
        executor = ToolExecutor([])

        first = await executor.execute(ToolCall("missing-1", "missing", {"query": "same"}))
        second = await executor.execute(ToolCall("missing-2", "missing", {"query": "same"}))

        self.assertEqual(first.status, "error")
        self.assertEqual(first.error_code, "unknown_tool")
        self.assertFalse(first.cached)
        self.assertEqual(second.status, "error")
        self.assertFalse(second.cached)

    async def test_cancellation_is_not_converted_to_tool_error(self):
        async def run(query: str) -> str:
            raise asyncio.CancelledError(query)

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)

        with self.assertRaises(asyncio.CancelledError):
            await ToolExecutor([tool]).execute(ToolCall("call-cancel", "reader", {"query": "stop"}))

    async def test_error_outcome_marks_trace_as_error(self):
        class RecordingSpan:
            def __init__(self):
                self.payload: dict[str, object] = {}
                self.error: str | None = None

            def set_payload(self, key: str, value: object) -> None:
                self.payload[key] = value

            def mark_error(self, message: str) -> None:
                self.error = message

        class RecordingTracer:
            def __init__(self):
                self.span_handle = RecordingSpan()

            @asynccontextmanager
            async def span(self, *_args, **_kwargs):
                yield self.span_handle

        async def run(query: str) -> str:
            raise ToolExecutionError(query, error_code="business_rejected")

        tracer = RecordingTracer()
        tool = _fake_tool("reader", run, read_only=True, cacheable=True)

        outcome = await ToolExecutor([tool], tracer=tracer).execute(
            ToolCall("call-error", "reader", {"query": "rejected"})
        )

        self.assertEqual(outcome.status, "error")
        self.assertEqual(tracer.span_handle.payload["status"], "error")
        self.assertEqual(tracer.span_handle.error, "rejected")


class BuiltinToolPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_builtin_cache_policy_is_explicit_and_matches_side_effects(self):
        expected = {
            "knowledge_search": (True, True),
            "memory_search": (False, False),
            "datetime": (True, False),
            "web_search": (True, True),
            "create_scheduled_task": (False, False),
        }

        actual = {
            key: (BUILTIN_REGISTRY[key].read_only, BUILTIN_REGISTRY[key].cacheable)
            for key in expected
        }

        self.assertEqual(actual, expected)

    async def test_registry_copies_builtin_policy_to_runtime_tool_metadata(self):
        overrides = {key: key == "datetime" for key in BUILTIN_REGISTRY}
        with (
            patch(
                "app.core.agent.tools.registry._enabled_map",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "app.core.agent.tools.mcp.loader.build_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            tools = await build_enabled_tools(
                object(), uuid.uuid4(), [], overrides=overrides, stats_holder={}
            )

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "datetime")
        self.assertIs(tools[0].metadata[TOOL_READ_ONLY_METADATA_KEY], True)
        self.assertIs(tools[0].metadata[TOOL_CACHEABLE_METADATA_KEY], False)


class BuiltinToolFailureTests(unittest.IsolatedAsyncioTestCase):
    async def _build_only(self, tool_key: str) -> StructuredTool:
        overrides = {key: key == tool_key for key in BUILTIN_REGISTRY}
        with (
            patch(
                "app.core.agent.tools.registry._enabled_map",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch(
                "app.core.agent.tools.mcp.loader.build_mcp_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            tools = await build_enabled_tools(
                object(), uuid.uuid4(), [], overrides=overrides, stats_holder={}
            )
        self.assertEqual(len(tools), 1)
        return tools[0]

    async def test_web_search_timeout_is_explicit_retryable_error(self):
        with patch(
            "app.core.agent.tools.builtin.web_search._get_websearch_config",
            new_callable=AsyncMock,
            return_value=("tavily", "secret"),
        ):
            tool = await self._build_only("web_search")

        search = AsyncMock(side_effect=httpx.ReadTimeout("provider timeout"))
        with patch(
            "app.core.agent.web_search.web_search",
            new=search,
        ):
            executor = ToolExecutor([tool])
            outcome = await executor.execute(ToolCall("web-1", "web_search", {"query": "latest"}))
            repeated = await executor.execute(ToolCall("web-2", "web_search", {"query": "latest"}))

        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.error_code, "web_search_timeout")
        self.assertTrue(outcome.retryable)
        self.assertFalse(outcome.cached)
        self.assertEqual(repeated.status, "error")
        self.assertEqual(search.await_count, 2)

    async def test_memory_search_same_args_executes_twice(self):
        tool = await self._build_only("memory_search")
        search = AsyncMock(return_value=[])

        with (
            patch("app.core.llm.resolver.get_client_for_type", new_callable=AsyncMock),
            patch("app.core.agent.tools.builtin.memory.search_memory", new=search),
        ):
            executor = ToolExecutor([tool])
            first = await executor.execute(ToolCall("memory-1", "memory_search", {"query": "same"}))
            second = await executor.execute(
                ToolCall("memory-2", "memory_search", {"query": "same"})
            )

        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success")
        self.assertFalse(first.cached)
        self.assertFalse(second.cached)
        self.assertEqual(search.await_count, 2)

    async def test_memory_search_timeout_is_not_retryable(self):
        tool = await self._build_only("memory_search")
        search = AsyncMock(side_effect=TimeoutError("memory timeout"))

        with (
            patch("app.core.llm.resolver.get_client_for_type", new_callable=AsyncMock),
            patch("app.core.agent.tools.builtin.memory.search_memory", new=search),
        ):
            outcome = await ToolExecutor([tool]).execute(
                ToolCall("memory-timeout", "memory_search", {"query": "same"})
            )

        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.error_code, "tool_timeout")
        self.assertFalse(outcome.retryable)
        self.assertFalse(outcome.cached)
        self.assertEqual(search.await_count, 1)

    async def test_schedule_business_failure_is_explicit_non_retryable_error(self):
        from app.core.exceptions import BizError

        tool = await self._build_only("create_scheduled_task")

        with patch(
            "app.services.agent_task_service.AgentTaskService.create",
            new_callable=AsyncMock,
            side_effect=BizError("invalid schedule"),
        ):
            outcome = await ToolExecutor([tool]).execute(
                ToolCall(
                    "schedule-1",
                    "create_scheduled_task",
                    {
                        "instruction": "track releases",
                        "trigger_type": "daily",
                        "time": "09:00",
                    },
                )
            )

        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.error_code, "scheduled_task_rejected")
        self.assertFalse(outcome.retryable)
        self.assertFalse(outcome.cached)

    async def test_schedule_success_with_same_args_executes_twice(self):
        tool = await self._build_only("create_scheduled_task")
        create = AsyncMock(return_value=SimpleNamespace(name="track", next_run_at=None))
        args = {
            "instruction": "track releases",
            "trigger_type": "daily",
            "time": "09:00",
        }

        with patch("app.services.agent_task_service.AgentTaskService.create", new=create):
            executor = ToolExecutor([tool])
            first = await executor.execute(ToolCall("schedule-1", "create_scheduled_task", args))
            second = await executor.execute(ToolCall("schedule-2", "create_scheduled_task", args))

        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success")
        self.assertFalse(first.cached)
        self.assertFalse(second.cached)
        self.assertEqual(create.await_count, 2)


class AgentOrchestratorContractTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _fc_model(
        tool_name: str,
        args: dict,
        *,
        call_ids: tuple[str, ...] = ("fc-call-1",),
    ) -> _ScriptedFunctionCallingModel:
        calls = [
            AIMessageChunk(
                content="",
                tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
            )
            for call_id in call_ids
        ]
        return _ScriptedFunctionCallingModel([*calls, AIMessageChunk(content="done")])

    @staticmethod
    def _react_model(tool_name: str) -> _ScriptedReactModel:
        return _ScriptedReactModel(
            [
                AIMessage(content=f"Action: {tool_name}\nAction Input: same"),
                AIMessage(content="Final Answer: done"),
            ]
        )

    async def _run_fc(self, model, tools: list[StructuredTool]) -> list[dict]:
        return await _collect_events(run_function_calling(model, tools, []))

    async def _run_react(self, model, tools: list[StructuredTool]) -> list[dict]:
        return await _collect_events(run_react(model, tools, "same", [], ""))

    async def test_fc_and_react_share_success_semantics(self):
        async def run(query: str) -> str:
            return f"result:{query}"

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)
        fc_events = await self._run_fc(self._fc_model("reader", {"query": "same"}), [tool])
        react_events = await self._run_react(self._react_model("reader"), [tool])
        fc_result = next(event for event in fc_events if event["type"] == "tool_result")
        react_result = next(event for event in react_events if event["type"] == "tool_result")

        self.assertEqual(fc_result["status"], "success")
        self.assertEqual(react_result["status"], "success")
        self.assertEqual(fc_result["text"], react_result["text"])
        self.assertIsNone(fc_result["error_code"])
        self.assertIsNone(react_result["error_code"])

    async def test_fc_and_react_share_failure_semantics(self):
        async def run(query: str) -> str:
            raise ToolExecutionError(
                f"rejected:{query}",
                error_code="business_rejected",
            )

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)
        fc_model = self._fc_model("reader", {"query": "same"})
        fc_events = await self._run_fc(fc_model, [tool])
        react_events = await self._run_react(self._react_model("reader"), [tool])
        fc_result = next(event for event in fc_events if event["type"] == "tool_result")
        react_result = next(event for event in react_events if event["type"] == "tool_result")

        self.assertEqual(fc_result["status"], "error")
        self.assertEqual(react_result["status"], "error")
        self.assertEqual(fc_result["error_code"], "business_rejected")
        self.assertEqual(react_result["error_code"], "business_rejected")
        self.assertFalse(fc_result["cached"])
        self.assertFalse(react_result["cached"])
        self.assertEqual(fc_model.seen_messages[1][-1].status, "error")

    async def test_tool_start_and_result_share_call_id_in_both_paths(self):
        async def run(query: str) -> str:
            return query

        tool = _fake_tool("reader", run, read_only=True, cacheable=True)
        fc_events = await self._run_fc(self._fc_model("reader", {"query": "same"}), [tool])
        react_events = await self._run_react(self._react_model("reader"), [tool])

        for events in (fc_events, react_events):
            start = next(event for event in events if event["type"] == "tool_start")
            result = next(event for event in events if event["type"] == "tool_result")
            self.assertTrue(start["call_id"])
            self.assertEqual(start["call_id"], result["call_id"])
        fc_start = next(event for event in fc_events if event["type"] == "tool_start")
        self.assertEqual(fc_start["call_id"], "fc-call-1")

    async def test_fc_event_preserves_full_structured_arguments(self):
        async def run(query: str, limit: int) -> str:
            return f"{query}:{limit}"

        tool = StructuredTool.from_function(
            coroutine=run,
            name="reader",
            description="two-field reader",
            args_schema=_TwoFieldInput,
            metadata={
                TOOL_READ_ONLY_METADATA_KEY: True,
                TOOL_CACHEABLE_METADATA_KEY: True,
            },
        )
        args = {"query": "same", "limit": 3}
        events = await self._run_fc(self._fc_model("reader", args), [tool])
        start = next(event for event in events if event["type"] == "tool_start")
        result = next(event for event in events if event["type"] == "tool_result")

        self.assertEqual(start["tool"], "reader")
        self.assertEqual(start["args"], args)
        self.assertEqual(result["args"], args)

    async def test_fc_uses_declared_cache_policy_for_read_and_write_tools(self):
        read_executions = 0
        write_executions = 0

        async def read(query: str) -> str:
            nonlocal read_executions
            read_executions += 1
            return query

        async def write(query: str) -> str:
            nonlocal write_executions
            write_executions += 1
            return query

        reader = _fake_tool("reader", read, read_only=True, cacheable=True)
        writer = _fake_tool("writer", write, read_only=False, cacheable=False)
        read_events = await self._run_fc(
            self._fc_model("reader", {"query": "same"}, call_ids=("read-1", "read-2")),
            [reader],
        )
        write_events = await self._run_fc(
            self._fc_model("writer", {"query": "same"}, call_ids=("write-1", "write-2")),
            [writer],
        )

        read_results = [event for event in read_events if event["type"] == "tool_result"]
        write_results = [event for event in write_events if event["type"] == "tool_result"]
        self.assertEqual(read_executions, 1)
        self.assertFalse(read_results[0]["cached"])
        self.assertTrue(read_results[1]["cached"])
        self.assertEqual(write_executions, 2)
        self.assertFalse(write_results[0]["cached"])
        self.assertFalse(write_results[1]["cached"])

    async def test_unknown_tool_error_retains_call_id_and_is_not_cached(self):
        events = await self._run_fc(self._fc_model("missing", {"query": "same"}), [])
        start = next(event for event in events if event["type"] == "tool_start")
        result = next(event for event in events if event["type"] == "tool_result")

        self.assertEqual(start["call_id"], "fc-call-1")
        self.assertEqual(result["call_id"], "fc-call-1")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "unknown_tool")
        self.assertFalse(result["cached"])


if __name__ == "__main__":
    unittest.main()

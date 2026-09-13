import asyncio
import unittest
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.agent.agent_contract import AgentRunLimits
from app.core.agent.orchestrator import run_function_calling, run_react
from app.core.agent.tool_execution import (
    TOOL_CACHEABLE_METADATA_KEY,
    TOOL_READ_ONLY_METADATA_KEY,
    ToolCall,
    ToolCallValidationError,
    ToolExecutor,
)
from app.core.agent.tools.registry import build_enabled_tools
from app.schemas.chat_schema import ChatStreamRequest
from app.services.chat_service import ChatService


class _TypedInput(BaseModel):
    query: str = Field(...)
    limit: int = Field(...)


def _tool(name: str, coroutine, args_schema: type[BaseModel]) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=f"fake {name}",
        args_schema=args_schema,
        metadata={
            TOOL_READ_ONLY_METADATA_KEY: False,
            TOOL_CACHEABLE_METADATA_KEY: False,
        },
    )


class _FunctionCallingModel:
    model_name = "scripted-fc"

    def __init__(self, responses: list[AIMessageChunk]):
        self.responses = list(responses)
        self.request_count = 0

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    async def astream(self, _messages):
        self.request_count += 1
        yield self.responses.pop(0)


class _ReactModel:
    model_name = "scripted-react"

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.request_count = 0
        self.seen_messages: list[list] = []

    async def ainvoke(self, messages):
        self.request_count += 1
        self.seen_messages.append(list(messages))
        return self.responses.pop(0)


def _fc_call(name: str, args: dict, call_id: str) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


async def _events(stream) -> list[dict]:
    return [event async for event in stream]


class AgentParameterContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_json_schema_is_validated_before_execution(self):
        executions = 0

        async def run(query: str, limit: int) -> str:
            nonlocal executions
            executions += 1
            return f"{query}:{limit}"

        tool = StructuredTool.from_function(
            coroutine=run,
            name="server__search",
            description="fake MCP tool",
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query", "limit"],
                "additionalProperties": False,
            },
        )
        executor = ToolExecutor([tool])

        with self.assertRaises(ToolCallValidationError):
            executor.validate_call(
                "invalid",
                "server__search",
                {"query": "same", "limit": "many"},
            )
        call = executor.validate_call(
            "valid",
            "server__search",
            {"query": "same", "limit": 3},
        )
        outcome = await executor.execute(call)

        self.assertEqual(call.validated_args, {"query": "same", "limit": 3})
        self.assertEqual(outcome.status, "success")
        self.assertEqual(executions, 1)

    async def test_react_executes_complete_multifield_json_arguments(self):
        from app.core.agent.tools.base import BUILTIN_REGISTRY

        overrides = {key: key == "create_scheduled_task" for key in BUILTIN_REGISTRY}
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
            tools = await build_enabled_tools(object(), uuid.uuid4(), [], overrides=overrides)
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        model = _ReactModel(
            [
                AIMessage(
                    content=(
                        "Action: create_scheduled_task\n"
                        "Action Input: {\n"
                        '  "instruction": "track releases",\n'
                        '  "trigger_type": "weekly",\n'
                        '  "time": "09:30",\n'
                        '  "weekday": 2,\n'
                        '  "name": "release watch"\n'
                        "}"
                    )
                ),
                AIMessage(content="Final Answer: scheduled"),
            ]
        )

        create = AsyncMock(return_value=SimpleNamespace(name="release watch", next_run_at=None))
        with patch("app.services.agent_task_service.AgentTaskService.create", new=create):
            events = await _events(run_react(model, [tool], "schedule it", [], ""))

        self.assertEqual(create.await_count, 1)
        request = create.await_args.args[1]
        self.assertEqual(request.instruction, "track releases")
        self.assertEqual(request.trigger_type, "weekly")
        self.assertEqual(request.trigger_time, "09:30")
        self.assertEqual(request.trigger_weekday, 2)
        self.assertIsNone(request.trigger_interval_hours)
        self.assertEqual(request.name, "release watch")
        start = next(event for event in events if event["type"] == "tool_start")
        self.assertTrue(start["args_validated"])
        self.assertEqual(
            start["args"],
            {
                "instruction": "track releases",
                "trigger_type": "weekly",
                "time": "09:30",
                "weekday": 2,
                "interval_hours": None,
                "name": "release watch",
            },
        )
        self.assertEqual(events[-1]["status"], "completed")

    async def test_invalid_arguments_do_not_execute_and_can_be_corrected(self):
        received: list[dict] = []

        async def search(query: str, limit: int) -> str:
            received.append({"query": query, "limit": limit})
            return "found"

        tool = _tool("typed_search", search, _TypedInput)
        model = _ReactModel(
            [
                AIMessage(
                    content='Action: typed_search\nAction Input: {"query":"same","limit":"many"}'
                ),
                AIMessage(content='Action: typed_search\nAction Input: {"query":"same","limit":3}'),
                AIMessage(content="Final Answer: corrected"),
            ]
        )

        events = await _events(run_react(model, [tool], "same", [], ""))

        results = [event for event in events if event["type"] == "tool_result"]
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["error_code"], "tool_validation_error")
        self.assertFalse(results[0]["args_validated"])
        starts = [event for event in events if event["type"] == "tool_start"]
        self.assertEqual(starts[0]["call_id"], results[0]["call_id"])
        self.assertEqual(results[1]["status"], "success")
        self.assertEqual(received, [{"query": "same", "limit": 3}])
        self.assertEqual(events[-1]["status"], "completed")

    async def test_repeated_invalid_arguments_exhaust_correction_budget(self):
        executions = 0

        async def search(query: str, limit: int) -> str:
            nonlocal executions
            executions += 1
            return f"{query}:{limit}"

        tool = _tool("typed_search", search, _TypedInput)
        model = _ReactModel(
            [
                AIMessage(content="Action: typed_search\nAction Input: {}"),
                AIMessage(content='Action: typed_search\nAction Input: {"query":"same"}'),
                AIMessage(content='Action: typed_search\nAction Input: {"query":"same","limit":3}'),
            ]
        )

        events = await _events(
            run_react(
                model,
                [tool],
                "same",
                [],
                "",
                limits=AgentRunLimits(max_parameter_corrections=2),
            )
        )

        self.assertEqual(executions, 0)
        self.assertEqual(model.request_count, 2)
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["status"], "budget_exhausted")
        self.assertEqual(events[-1]["stop_reason"], "parameter_correction_budget_exhausted")

        fc_model = _FunctionCallingModel(
            [
                _fc_call("typed_search", {}, "invalid-1"),
                _fc_call("typed_search", {"query": "same"}, "invalid-2"),
                _fc_call("typed_search", {"query": "same", "limit": 3}, "must-not-run"),
            ]
        )
        fc_events = await _events(
            run_function_calling(
                fc_model,
                [tool],
                [],
                limits=AgentRunLimits(max_parameter_corrections=2),
            )
        )

        self.assertEqual(executions, 0)
        self.assertEqual(fc_model.request_count, 2)
        self.assertEqual(fc_events[-1]["status"], "budget_exhausted")
        self.assertEqual(fc_events[-1]["stop_reason"], "parameter_correction_budget_exhausted")

    async def test_react_budget_exhaustion_does_not_expose_protocol_as_partial_answer(self):
        async def search(query: str, limit: int) -> str:
            return f"{query}:{limit}"

        protocol = "Thought: inspect\nAction: typed_search\nAction Input: {}"
        model = _ReactModel([AIMessage(content=protocol), AIMessage(content=protocol)])

        events = await _events(
            run_react(
                model,
                [_tool("typed_search", search, _TypedInput)],
                "same",
                [],
                "",
                limits=AgentRunLimits(max_parameter_corrections=2),
            )
        )

        terminal = events[-1]
        self.assertEqual(terminal["status"], "budget_exhausted")
        self.assertEqual(terminal["partial_answer"], "")
        self.assertNotIn("Thought:", terminal["partial_answer"])
        self.assertNotIn("Action:", terminal["partial_answer"])

    async def test_invalid_tool_schema_is_failed_without_parameter_correction(self):
        executions = 0

        async def search(query: str) -> str:
            nonlocal executions
            executions += 1
            return query

        tool = StructuredTool.from_function(
            coroutine=search,
            name="broken_schema",
            description="invalid schema tool",
            args_schema={
                "type": "object",
                "properties": {"query": {"type": "not-a-json-schema-type"}},
            },
        )
        react_model = _ReactModel(
            [
                AIMessage(content='Action: broken_schema\nAction Input: {"query":"same"}'),
                AIMessage(content="Final Answer: must not run"),
            ]
        )
        fc_model = _FunctionCallingModel(
            [
                _fc_call("broken_schema", {"query": "same"}, "broken-1"),
                AIMessageChunk(content="must not run"),
            ]
        )

        react_events = await _events(run_react(react_model, [tool], "same", [], ""))
        fc_events = await _events(run_function_calling(fc_model, [tool], []))

        for events in (react_events, fc_events):
            self.assertEqual(events[-1]["status"], "failed")
            self.assertEqual(events[-1]["stop_reason"], "tool_schema_invalid")
        self.assertEqual(react_model.request_count, 1)
        self.assertEqual(fc_model.request_count, 1)
        self.assertEqual(executions, 0)

    async def test_fc_and_react_use_same_schema_validation(self):
        async def search(query: str, limit: int) -> str:
            return f"{query}:{limit}"

        tool = _tool("typed_search", search, _TypedInput)
        fc_model = _FunctionCallingModel(
            [
                _fc_call("typed_search", {"query": "same", "limit": "many"}, "fc-invalid"),
                _fc_call("typed_search", {"query": "same", "limit": 3}, "fc-valid"),
                AIMessageChunk(content="done"),
            ]
        )
        react_model = _ReactModel(
            [
                AIMessage(
                    content='Action: typed_search\nAction Input: {"query":"same","limit":"many"}'
                ),
                AIMessage(content='Action: typed_search\nAction Input: {"query":"same","limit":3}'),
                AIMessage(content="Final Answer: done"),
            ]
        )

        fc_events = await _events(run_function_calling(fc_model, [tool], []))
        react_events = await _events(run_react(react_model, [tool], "same", [], ""))
        fc_results = [event for event in fc_events if event["type"] == "tool_result"]
        react_results = [event for event in react_events if event["type"] == "tool_result"]

        self.assertEqual(
            [(event["status"], event["error_code"]) for event in fc_results],
            [("error", "tool_validation_error"), ("success", None)],
        )
        self.assertEqual(
            [(event["status"], event["error_code"]) for event in react_results],
            [("error", "tool_validation_error"), ("success", None)],
        )
        self.assertEqual(fc_results[1]["args"], react_results[1]["args"])

    async def test_fc_malformed_json_arguments_are_correctable_validation_error(self):
        executions = 0

        async def search(query: str, limit: int) -> str:
            nonlocal executions
            executions += 1
            return f"{query}:{limit}"

        tool = _tool("typed_search", search, _TypedInput)
        model = _FunctionCallingModel(
            [
                AIMessageChunk(
                    content="",
                    invalid_tool_calls=[
                        {
                            "name": "typed_search",
                            "args": '{"query":"same","limit":',
                            "id": "malformed",
                            "error": "invalid JSON",
                            "type": "invalid_tool_call",
                        }
                    ],
                ),
                _fc_call("typed_search", {"query": "same", "limit": 3}, "corrected"),
                AIMessageChunk(content="done"),
            ]
        )

        events = await _events(run_function_calling(model, [tool], []))
        results = [event for event in events if event["type"] == "tool_result"]

        self.assertEqual(results[0]["call_id"], "malformed")
        self.assertEqual(results[0]["status"], "error")
        self.assertEqual(results[0]["error_code"], "tool_validation_error")
        self.assertEqual(results[1]["status"], "success")
        self.assertEqual(executions, 1)
        self.assertEqual(events[-1]["status"], "completed")


class AgentCapabilityScopeTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _named_tool(name: str) -> StructuredTool:
        async def run(query: str) -> str:
            return query

        class QueryInput(BaseModel):
            query: str = Field(...)

        return _tool(name, run, QueryInput)

    async def test_final_filter_blocks_unlisted_builtin_and_all_mcp_tools(self):
        allowed = self._named_tool("allowed_builtin")
        blocked = self._named_tool("blocked_builtin")
        mcp = self._named_tool("allowed_builtin")

        load_mcp = AsyncMock(return_value=[mcp])
        with (
            patch(
                "app.core.agent.tools.registry._build_builtin_tools",
                new_callable=AsyncMock,
                return_value=[allowed, blocked],
            ),
            patch(
                "app.core.agent.tools.mcp.loader.build_mcp_tools",
                new=load_mcp,
            ),
        ):
            tools = await build_enabled_tools(
                object(),
                uuid.uuid4(),
                [],
                allowed_tool_keys={"allowed_builtin"},
            )

        self.assertEqual(tools, [allowed])
        load_mcp.assert_not_awaited()

    async def test_no_skill_scope_preserves_enabled_builtin_and_mcp_tools(self):
        builtin = self._named_tool("builtin")
        mcp = self._named_tool("server__tool")

        with (
            patch(
                "app.core.agent.tools.registry._build_builtin_tools",
                new_callable=AsyncMock,
                return_value=[builtin],
            ),
            patch(
                "app.core.agent.tools.mcp.loader.build_mcp_tools",
                new_callable=AsyncMock,
                return_value=[mcp],
            ),
        ):
            tools = await build_enabled_tools(object(), uuid.uuid4(), [])

        self.assertEqual(tools, [builtin, mcp])

    async def test_chat_skill_scope_passes_builtin_only_authoritative_allowlist(self):
        service = ChatService(object())
        skill = SimpleNamespace(
            tool_keys=["knowledge_search", "server__mcp_tool"],
            kb_id=None,
        )
        body = ChatStreamRequest(message="question")

        with patch(
            "app.repositories.knowledge_base_repository.KnowledgeBaseRepository."
            "list_chat_enabled_ids",
            new_callable=AsyncMock,
            return_value=[],
        ):
            overrides, kb_ids, allowed = await service._tool_scope(uuid.uuid4(), body, skill)

        self.assertEqual(allowed, {"knowledge_search"})
        self.assertTrue(overrides["knowledge_search"])
        self.assertNotIn("server__mcp_tool", allowed)
        self.assertEqual(kb_ids, [])

    async def test_context_managed_registry_applies_same_final_filter(self):
        allowed = self._named_tool("allowed_builtin")
        blocked = self._named_tool("blocked_builtin")
        mcp = self._named_tool("server__tool")

        open_count = 0

        @asynccontextmanager
        async def open_tools(*_args, **_kwargs):
            nonlocal open_count
            open_count += 1
            yield [mcp]

        with (
            patch(
                "app.core.agent.tools.registry._build_builtin_tools",
                new_callable=AsyncMock,
                return_value=[allowed, blocked],
            ),
            patch("app.core.agent.tools.mcp.loader.open_mcp_tools", new=open_tools),
        ):
            from app.core.agent.tools.registry import build_enabled_tools_cm

            async with build_enabled_tools_cm(
                object(),
                uuid.uuid4(),
                [],
                allowed_tool_keys={"allowed_builtin"},
            ) as tools:
                self.assertEqual(tools, [allowed])
        self.assertEqual(open_count, 0)

    async def test_fc_and_react_see_the_same_filtered_capability_set(self):
        allowed = self._named_tool("allowed_builtin")
        blocked = self._named_tool("blocked_builtin")
        mcp = self._named_tool("server__tool")

        with (
            patch(
                "app.core.agent.tools.registry._build_builtin_tools",
                new_callable=AsyncMock,
                return_value=[allowed, blocked],
            ),
            patch(
                "app.core.agent.tools.mcp.loader.build_mcp_tools",
                new_callable=AsyncMock,
                return_value=[mcp],
            ),
        ):
            tools = await build_enabled_tools(
                object(),
                uuid.uuid4(),
                [],
                allowed_tool_keys={"allowed_builtin"},
            )

        fc_model = _FunctionCallingModel([AIMessageChunk(content="done")])
        react_model = _ReactModel([AIMessage(content="Final Answer: done")])
        await _events(run_function_calling(fc_model, tools, []))
        await _events(run_react(react_model, tools, "same", [], ""))

        react_prompt = react_model.seen_messages[0][0].content
        self.assertEqual(fc_model.bound_tool_names, ["allowed_builtin"])
        self.assertIn("- allowed_builtin：", react_prompt)
        self.assertNotIn("blocked_builtin", react_prompt)
        self.assertNotIn("server__tool", react_prompt)


class AgentTerminationAndBudgetTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _query_tool(name: str, coroutine, *, read_only: bool = True) -> StructuredTool:
        class QueryInput(BaseModel):
            query: str = Field(...)

        tool = _tool(name, coroutine, QueryInput)
        tool.metadata[TOOL_READ_ONLY_METADATA_KEY] = read_only
        return tool

    async def test_normal_final_answer_is_completed(self):
        fc_events = await _events(
            run_function_calling(
                _FunctionCallingModel([AIMessageChunk(content="fc answer")]),
                [],
                [],
            )
        )
        react_events = await _events(
            run_react(
                _ReactModel([AIMessage(content="Final Answer: react answer")]),
                [],
                "question",
                [],
                "",
            )
        )

        self.assertEqual(
            (fc_events[-1]["status"], fc_events[-1]["stop_reason"], fc_events[-1]["text"]),
            ("completed", "final_answer", "fc answer"),
        )
        self.assertEqual(
            (
                react_events[-1]["status"],
                react_events[-1]["stop_reason"],
                react_events[-1]["text"],
            ),
            ("completed", "final_answer", "react answer"),
        )

    async def test_model_request_budget_exhaustion_stops_new_requests(self):
        executions = 0

        async def search(query: str) -> str:
            nonlocal executions
            executions += 1
            return query

        tool = self._query_tool("reader", search)
        model = _FunctionCallingModel(
            [
                _fc_call("reader", {"query": "one"}, "call-1"),
                _fc_call("reader", {"query": "two"}, "call-2"),
                AIMessageChunk(content="must not run"),
            ]
        )

        events = await _events(
            run_function_calling(
                model,
                [tool],
                [],
                limits=AgentRunLimits(max_model_requests=2),
            )
        )

        self.assertEqual(model.request_count, 2)
        self.assertEqual(executions, 2)
        self.assertEqual(events[-1]["status"], "budget_exhausted")
        self.assertEqual(events[-1]["stop_reason"], "model_request_budget_exhausted")

    async def test_function_calling_intermediate_text_is_not_completed_answer(self):
        async def search(query: str) -> str:
            return query

        tool = self._query_tool("reader", search)
        model = _FunctionCallingModel(
            [
                AIMessageChunk(
                    content="准备查询……",
                    tool_calls=[
                        {
                            "name": "reader",
                            "args": {"query": "same"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessageChunk(content="真正答案"),
            ]
        )

        events = await _events(run_function_calling(model, [tool], []))

        streamed = "".join(event["text"] for event in events if event["type"] == "token")
        self.assertEqual(streamed, "真正答案")
        self.assertEqual(events[-1]["text"], "真正答案")
        self.assertNotIn("准备查询", events[-1]["text"])

    async def test_model_failure_has_failed_terminal_and_partial_answer(self):
        class FailingModel(_FunctionCallingModel):
            async def astream(self, _messages):
                self.request_count += 1
                yield AIMessageChunk(content="draft")
                raise RuntimeError("provider unavailable")

        events = await _events(run_function_calling(FailingModel([]), [], []))

        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(events[-1]["stop_reason"], "model_request_failed")
        self.assertEqual(events[-1]["text"], "")
        self.assertEqual(events[-1]["partial_answer"], "draft")

    async def test_unrecoverable_tool_binding_error_is_failed(self):
        async def search(query: str) -> str:
            return query

        class BindingFailureModel(_FunctionCallingModel):
            def bind_tools(self, _tools):
                raise RuntimeError("invalid provider schema")

        events = await _events(
            run_function_calling(
                BindingFailureModel([]),
                [self._query_tool("reader", search)],
                [],
            )
        )

        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(events[-1]["stop_reason"], "tool_binding_failed")

    async def test_cancellation_emits_cancelled_then_remains_control_flow(self):
        class CancelledModel(_ReactModel):
            async def ainvoke(self, _messages):
                self.request_count += 1
                raise asyncio.CancelledError

        stream = run_react(CancelledModel([]), [], "question", [], "")

        terminal = await anext(stream)
        self.assertEqual(terminal["status"], "cancelled")
        self.assertEqual(terminal["stop_reason"], "external_cancellation")
        with self.assertRaises(asyncio.CancelledError):
            await anext(stream)

    async def test_expired_deadline_prevents_starting_tool_or_new_model_request(self):
        class Clock:
            now = 0.0

            def __call__(self) -> float:
                return self.now

        clock = Clock()
        executions = 0

        async def search(query: str) -> str:
            nonlocal executions
            executions += 1
            return query

        class DeadlineModel(_FunctionCallingModel):
            async def astream(self, messages):
                async for chunk in super().astream(messages):
                    clock.now = 2.0
                    yield chunk

        tool = self._query_tool("reader", search)
        model = DeadlineModel(
            [
                _fc_call("reader", {"query": "must not run"}, "call-1"),
                AIMessageChunk(content="must not run"),
            ]
        )

        events = await _events(
            run_function_calling(
                model,
                [tool],
                [],
                limits=AgentRunLimits(total_deadline_seconds=1),
                clock=clock,
            )
        )

        self.assertEqual(model.request_count, 1)
        self.assertEqual(executions, 0)
        self.assertEqual(events[-1]["status"], "budget_exhausted")
        self.assertEqual(events[-1]["stop_reason"], "total_deadline_exceeded")

    async def test_no_tool_stream_is_stopped_by_remaining_deadline(self):
        class Clock:
            now = 0.0

            def __call__(self) -> float:
                return self.now

        clock = Clock()

        class TimedModel(_FunctionCallingModel):
            async def astream(self, _messages):
                self.request_count += 1
                clock.now += 0.6
                yield AIMessageChunk(content="first")
                clock.now += 0.6
                yield AIMessageChunk(content="too late")

        events = await _events(
            run_function_calling(
                TimedModel([]),
                [],
                [],
                limits=AgentRunLimits(total_deadline_seconds=1),
                clock=clock,
            )
        )

        self.assertEqual(
            [event["text"] for event in events if event["type"] == "token"],
            ["first"],
        )
        self.assertEqual(events[-1]["status"], "budget_exhausted")
        self.assertEqual(events[-1]["stop_reason"], "total_deadline_exceeded")

    async def test_no_tool_stream_deadline_excludes_downstream_yield_time(self):
        class Clock:
            now = 0.0

            def __call__(self) -> float:
                return self.now

        clock = Clock()

        class TimedModel(_FunctionCallingModel):
            async def astream(self, _messages):
                self.request_count += 1
                clock.now += 0.4
                yield AIMessageChunk(content="first")
                clock.now += 0.4
                yield AIMessageChunk(content="second")

        stream = run_function_calling(
            TimedModel([]),
            [],
            [],
            limits=AgentRunLimits(total_deadline_seconds=1),
            clock=clock,
        )
        first = await anext(stream)
        clock.now += 100
        remaining = await _events(stream)

        self.assertEqual(first, {"type": "token", "text": "first"})
        self.assertEqual(
            [event["text"] for event in remaining if event["type"] == "token"],
            ["second"],
        )
        self.assertEqual(remaining[-1]["status"], "completed")

    async def test_write_tool_timeout_is_not_retried(self):
        executions = 0

        async def write(query: str) -> str:
            nonlocal executions
            executions += 1
            raise TimeoutError(query)

        tool = self._query_tool("writer", write, read_only=False)
        model = _FunctionCallingModel(
            [
                _fc_call("writer", {"query": "once"}, "write-1"),
                AIMessageChunk(content="timeout reported"),
            ]
        )

        events = await _events(run_function_calling(model, [tool], []))
        results = [event for event in events if event["type"] == "tool_result"]

        self.assertEqual(executions, 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["error_code"], "tool_timeout")
        self.assertFalse(results[0]["retryable"])
        self.assertEqual(events[-1]["status"], "completed")

    async def test_executor_applies_explicit_tool_timeout_budget(self):
        executions = 0

        async def write(query: str) -> str:
            nonlocal executions
            executions += 1
            return query

        class ForcedTimeout:
            async def __aenter__(self):
                return self

            async def __aexit__(self, _exc_type, _exc, _tb):
                raise TimeoutError("forced timeout")

        tool = self._query_tool("writer", write, read_only=False)
        with patch(
            "app.core.agent.tool_execution.asyncio.timeout",
            return_value=ForcedTimeout(),
        ) as timeout:
            outcome = await ToolExecutor([tool]).execute(
                ToolCall("write-timeout", "writer", {"query": "once"}),
                timeout_seconds=7.5,
            )

        timeout.assert_called_once_with(7.5)
        self.assertEqual(executions, 1)
        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.error_code, "tool_timeout")
        self.assertFalse(outcome.retryable)


if __name__ == "__main__":
    unittest.main()

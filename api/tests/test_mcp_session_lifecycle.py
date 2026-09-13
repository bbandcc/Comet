import asyncio
import sys
import time
import unittest
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anyio
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.core.agent.research.models import SOURCE_MCP, Source
from app.core.agent.research.retriever import gather_mcp_sources
from app.core.agent.tools.mcp.loader import open_mcp_tools
from app.core.agent.tools.registry import build_enabled_tools_cm


class _EchoInput(BaseModel):
    value: str


def _servers(count: int, modes: list[str] | None = None) -> list[SimpleNamespace]:
    modes = modes or ["success"] * count
    return [
        SimpleNamespace(
            id=uuid.uuid4(),
            name=f"server_{index}",
            mode=modes[index],
        )
        for index in range(count)
    ]


class _LifecycleFixture:
    """Owner-sensitive local session boundary backed by a real AnyIO CancelScope."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.active: set[str] = set()
        self.enter_tasks: dict[str, asyncio.Task] = {}
        self.exit_tasks: dict[str, asyncio.Task] = {}
        self.owner_mismatches: list[str] = []
        self.load_started: set[str] = set()
        self.entered = asyncio.Event()
        self._never = asyncio.Event()

    @asynccontextmanager
    async def session(self, server: SimpleNamespace):
        key = str(server.id)
        owner = asyncio.current_task()
        if owner is None:
            raise AssertionError("session must run inside an asyncio task")
        self.opened.append(key)
        self.active.add(key)
        self.enter_tasks[key] = owner
        self.entered.set()
        try:
            with anyio.CancelScope():
                if server.mode == "enter_fail":
                    raise RuntimeError(f"failed:{server.name}")
                if server.mode == "timeout":
                    await self._never.wait()
                yield SimpleNamespace(server=server)
        except BaseException:
            self._record_exit(key, owner)
            raise
        else:
            self._record_exit(key, owner)
        finally:
            self.active.discard(key)

    def _record_exit(self, key: str, owner: asyncio.Task) -> None:
        current = asyncio.current_task()
        if current is None:
            raise AssertionError("session exit must run inside an asyncio task")
        self.exit_tasks[key] = current
        if current is owner:
            self.closed.append(key)
        else:
            self.owner_mismatches.append(key)

    async def load_tools(self, session: SimpleNamespace) -> list[StructuredTool]:
        server = session.server
        key = str(server.id)
        self.load_started.add(key)
        if server.mode == "load_fail":
            raise RuntimeError(f"failed:{server.name}")
        if server.mode == "load_timeout":
            await self._never.wait()

        async def echo(value: str) -> str:
            return f"{server.name}:{value}"

        return [
            StructuredTool.from_function(
                coroutine=echo,
                name="echo",
                description="local lifecycle fixture",
                args_schema=_EchoInput,
            )
        ]

    def client_type(self):
        fixture = self

        class Client:
            def __init__(self, connections):
                self.server = next(iter(connections.values()))["server"]

            def session(self, _server_name: str):
                return fixture.session(self.server)

        return Client


@contextmanager
def _patched_loader(servers: list[SimpleNamespace], fixture: _LifecycleFixture):
    with (
        patch(
            "app.core.agent.tools.mcp.loader.MCPServerRepository.list_by_user",
            new_callable=AsyncMock,
            return_value=servers,
        ),
        patch(
            "app.core.agent.tools.mcp.loader.build_connection",
            side_effect=lambda server: {"server": server},
        ),
        patch(
            "app.core.agent.tools.mcp.loader.MultiServerMCPClient",
            new=fixture.client_type(),
        ),
        patch(
            "app.core.agent.tools.mcp.loader.load_mcp_tools",
            new=fixture.load_tools,
        ),
    ):
        yield


class MCPSessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_server_enters_and_exits_in_owner_task(self):
        servers = _servers(1)
        fixture = _LifecycleFixture()
        owner = asyncio.current_task()

        with _patched_loader(servers, fixture):
            async with open_mcp_tools(object(), uuid.uuid4()) as tools:
                self.assertEqual(len(tools), 1)
                self.assertEqual(await tools[0].ainvoke({"value": "ok"}), "server_0:ok")
                self.assertEqual(fixture.active, {str(servers[0].id)})

        self.assertEqual(fixture.owner_mismatches, [])
        self.assertEqual(fixture.closed, [str(servers[0].id)])
        self.assertIs(fixture.enter_tasks[str(servers[0].id)], owner)
        self.assertIs(fixture.exit_tasks[str(servers[0].id)], owner)
        self.assertEqual(fixture.active, set())

    async def test_two_servers_skip_failed_initialization_and_close_both(self):
        servers = _servers(2, ["enter_fail", "success"])
        fixture = _LifecycleFixture()

        with _patched_loader(servers, fixture):
            async with open_mcp_tools(object(), uuid.uuid4()) as tools:
                self.assertEqual([tool.name for tool in tools], ["server_1__echo"])
                self.assertEqual(await tools[0].ainvoke({"value": "ok"}), "server_1:ok")

        self.assertEqual(fixture.owner_mismatches, [])
        self.assertCountEqual(fixture.closed, [str(server.id) for server in servers])
        self.assertEqual(fixture.active, set())

    async def test_five_servers_skip_failure_and_timeout_without_leaks(self):
        servers = _servers(
            5,
            ["success", "load_fail", "timeout", "success", "success"],
        )
        fixture = _LifecycleFixture()

        with (
            _patched_loader(servers, fixture),
            patch("app.core.agent.tools.mcp.loader._MCP_LOAD_TIMEOUT", 0.01),
        ):
            async with open_mcp_tools(object(), uuid.uuid4()) as tools:
                self.assertEqual(
                    [tool.name for tool in tools],
                    ["server_0__echo", "server_3__echo", "server_4__echo"],
                )

        self.assertEqual(fixture.owner_mismatches, [])
        self.assertCountEqual(fixture.closed, [str(server.id) for server in servers])
        self.assertEqual(fixture.active, set())

    async def test_load_timeout_closes_entered_session_and_continues(self):
        servers = _servers(2, ["load_timeout", "success"])
        fixture = _LifecycleFixture()
        owner = asyncio.current_task()
        timeout_key = str(servers[0].id)

        with (
            _patched_loader(servers, fixture),
            patch("app.core.agent.tools.mcp.loader._MCP_LOAD_TIMEOUT", 0.01),
        ):
            async with open_mcp_tools(object(), uuid.uuid4()) as tools:
                self.assertEqual([tool.name for tool in tools], ["server_1__echo"])
                self.assertEqual(await tools[0].ainvoke({"value": "ok"}), "server_1:ok")

        self.assertIn(timeout_key, fixture.load_started)
        self.assertIs(fixture.enter_tasks[timeout_key], owner)
        self.assertIs(fixture.exit_tasks[timeout_key], owner)
        self.assertEqual(fixture.owner_mismatches, [])
        self.assertCountEqual(fixture.closed, [str(server.id) for server in servers])
        self.assertEqual(fixture.active, set())

    async def test_external_cancellation_closes_owner_session_and_propagates(self):
        servers = _servers(1)
        fixture = _LifecycleFixture()
        entered = asyncio.Event()
        hold = asyncio.Event()

        async def use_tools() -> None:
            async with open_mcp_tools(object(), uuid.uuid4()) as tools:
                self.assertEqual(len(tools), 1)
                entered.set()
                await hold.wait()

        with _patched_loader(servers, fixture):
            owner = asyncio.create_task(use_tools())
            await entered.wait()
            owner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner

        self.assertTrue(owner.done())
        self.assertEqual(fixture.owner_mismatches, [])
        self.assertEqual(fixture.closed, [str(servers[0].id)])
        self.assertEqual(fixture.active, set())

    async def test_cancellation_during_initialization_closes_in_owner_task(self):
        servers = _servers(1, ["timeout"])
        fixture = _LifecycleFixture()

        async def open_tools() -> None:
            async with open_mcp_tools(object(), uuid.uuid4()):
                self.fail("cancelled initialization must not yield tools")

        with (
            _patched_loader(servers, fixture),
            patch("app.core.agent.tools.mcp.loader._MCP_LOAD_TIMEOUT", 60),
        ):
            owner = asyncio.create_task(open_tools())
            await fixture.entered.wait()
            owner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner

        self.assertTrue(owner.done())
        self.assertEqual(fixture.owner_mismatches, [])
        self.assertEqual(fixture.closed, [str(servers[0].id)])
        self.assertEqual(fixture.active, set())

    async def test_registry_context_uses_owner_safe_mcp_sessions(self):
        servers = _servers(1)
        fixture = _LifecycleFixture()

        with (
            _patched_loader(servers, fixture),
            patch(
                "app.core.agent.tools.registry._build_builtin_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            async with build_enabled_tools_cm(object(), uuid.uuid4(), []) as tools:
                self.assertEqual(len(tools), 1)
                self.assertEqual(await tools[0].ainvoke({"value": "ok"}), "server_0:ok")

        self.assertEqual(fixture.owner_mismatches, [])
        self.assertEqual(fixture.active, set())

    async def test_research_path_uses_owner_safe_mcp_sessions(self):
        servers = _servers(1)
        fixture = _LifecycleFixture()
        expected = [Source(index=0, type=SOURCE_MCP, title="fixture", content="data")]

        with (
            _patched_loader(servers, fixture),
            patch("app.core.agent.research.retriever.settings.research_mcp_enabled", True),
            patch(
                "app.core.agent.research.retriever._run_mcp_loop",
                new_callable=AsyncMock,
                return_value=expected,
            ) as run_loop,
        ):
            sources = await gather_mcp_sources(
                object(),
                uuid.uuid4(),
                "topic",
                object(),
                supports_fc=True,
            )

        self.assertEqual(sources, expected)
        self.assertEqual(len(run_loop.await_args.args[1]), 1)
        self.assertEqual(fixture.owner_mismatches, [])
        self.assertEqual(fixture.active, set())

    async def test_real_stdio_sdk_sessions_are_usable_and_owner_safe(self):
        fixture_path = Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
        warnings: list[str] = []
        connect_timings: dict[int, float] = {}
        first_result_timings: dict[int, float] = {}

        def capture_warning(message, *args) -> None:
            warnings.append(message % args if args else str(message))

        for count in (1, 2, 5):
            servers = _servers(count)

            def capture_info(message, *args) -> None:
                if message.startswith("MCP 持久会话串行打开完成"):
                    connect_timings[count] = args[-1]

            with (
                patch(
                    "app.core.agent.tools.mcp.loader.MCPServerRepository.list_by_user",
                    new_callable=AsyncMock,
                    return_value=servers,
                ),
                patch(
                    "app.core.agent.tools.mcp.loader.build_connection",
                    return_value={
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(fixture_path)],
                    },
                ),
                patch(
                    "app.core.agent.tools.mcp.loader.logger.warning",
                    new=capture_warning,
                ),
                patch("app.core.agent.tools.mcp.loader.logger.info", new=capture_info),
            ):
                started = time.monotonic()
                async with open_mcp_tools(object(), uuid.uuid4()) as tools:
                    self.assertEqual(len(tools), count)
                    result = await tools[0].ainvoke({"value": "sdk"})
                    first_result_timings[count] = time.monotonic() - started
                    self.assertIn("echo:sdk", str(result))
                    for tool in tools[1:]:
                        result = await tool.ainvoke({"value": "sdk"})
                        self.assertIn("echo:sdk", str(result))

        self.assertEqual(set(connect_timings), {1, 2, 5})
        self.assertEqual(set(first_result_timings), {1, 2, 5})
        print(
            "local MCP SDK elapsed: "
            + ", ".join(
                f"{count}=connect {connect_timings[count]:.2f}s, "
                f"first result {first_result_timings[count]:.2f}s"
                for count in (1, 2, 5)
            )
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()

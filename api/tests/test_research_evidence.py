import asyncio
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessageChunk
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.dialects import postgresql

from app.core.agent.agent_contract import MAX_TOOL_RESULT_PREVIEW
from app.core.agent.loop.rubric.research import RESEARCH_RUBRIC
from app.core.agent.loop.verifier.llm_verifier import (
    _JUDGE_EVIDENCE_TOTAL_CHARS,
    _render_evidence_context,
    _render_research_prompt,
)
from app.core.agent.orchestrator import run_function_calling
from app.core.agent.research.distiller import _distill_one
from app.core.agent.research.evidence_store import (
    delete_external_evidence,
    prepare_evidence_sources,
)
from app.core.agent.research.models import SOURCE_KB, SOURCE_MCP, SOURCE_WEB, Source
from app.core.agent.research.engine import _build_verifier_artifact, _public_loop_event
from app.core.agent.research.retriever import (
    _run_mcp_loop,
    deduplicate_sources,
    gather_kb_sources,
    gather_web_sources,
)
from app.core.agent.tool_contract import ToolOutcome
from app.repositories.research_report_repository import ResearchReportRepository
from app.core.rag.search import _resolve_parent_content
from app.services.agent_task_service import AgentTaskService
from app.services.report_share_service import ReportShareService
from app.services.research_service import ResearchService, _public_report_event


class _QueryArgs(BaseModel):
    query: str


class _ScriptedFCModel:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def bind_tools(self, _tools):
        return self

    async def astream(self, _messages):
        for chunk in self._chunks.pop(0):
            yield chunk


def _query_tool(name, coroutine):
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description="fixture",
        args_schema=_QueryArgs,
    )


def _fc_tool_chunk(name: str, call_id: str) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_calls=[{"name": name, "args": {"query": "q"}, "id": call_id}],
    )


class ResearchEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_tool_preview_is_bounded_but_internal_outcome_keeps_tail(self):
        tail = "TAIL-FACT-RETAINED"
        full = "x" * (MAX_TOOL_RESULT_PREVIEW + 50) + tail

        async def tool(query: str) -> ToolOutcome:
            return ToolOutcome(status="success", content=full, artifact_ref="artifact://one")

        lc_tool = StructuredTool.from_function(
            coroutine=tool,
            name="fixture",
            description="fixture",
            args_schema=_QueryArgs,
        )
        model = _ScriptedFCModel(
            [
                [
                    AIMessageChunk(
                        content="",
                        tool_calls=[{"name": "fixture", "args": {"query": "q"}, "id": "call-1"}],
                    )
                ],
                [AIMessageChunk(content="done")],
            ]
        )
        captured = []

        async def capture(call, outcome):
            captured.append((call, outcome))

        events = [
            event
            async for event in run_function_calling(model, [lc_tool], [], outcome_handler=capture)
        ]
        public = next(event for event in events if event["type"] == "tool_result")

        self.assertNotIn(tail, public["text"])
        self.assertLessEqual(len(public["text"]), MAX_TOOL_RESULT_PREVIEW + 3)
        self.assertEqual(captured[0][0].call_id, "call-1")
        self.assertIn(tail, captured[0][1].content)
        self.assertEqual(captured[0][1].artifact_ref, "artifact://one")

        research_model = _ScriptedFCModel(
            [
                [
                    AIMessageChunk(
                        content="",
                        tool_calls=[{"name": "fixture", "args": {"query": "q"}, "id": "call-2"}],
                    )
                ],
                [AIMessageChunk(content="done")],
            ]
        )
        sources = await _run_mcp_loop(research_model, [lc_tool], "topic")
        self.assertEqual(sources[0].call_id, "call-2")
        self.assertEqual(sources[0].artifact_ref, "artifact://one")
        self.assertIn(tail, sources[0].content)
        verifier_prompt = _render_research_prompt(
            topic="topic",
            rubric=RESEARCH_RUBRIC,
            artifact={"markdown": "report", "sources": [sources[0].as_verifier_evidence()]},
        )
        self.assertIn(tail, verifier_prompt)

    async def test_web_sources_express_full_and_fallback_without_truncating_storage(self):
        results = [
            {"url": "https://Example.com/a#fragment", "title": "full", "snippet": "s1"},
            {"url": "https://example.com/b", "title": "fallback", "snippet": "fallback"},
        ]

        async def fetch(url):
            if url.endswith("/b"):
                raise RuntimeError("blocked")
            return "Fetched", "f" * 20

        with (
            patch(
                "app.core.agent.web_search.web_search_structured",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch("app.core.rag.web_crawler.fetch_url_content", side_effect=fetch),
            patch(
                "app.core.agent.research.retriever.settings.research_source_quality_filter", False
            ),
            patch("app.core.agent.research.retriever.settings.research_source_truncate_chars", 10),
        ):
            sources = await gather_web_sources("p", "k", ["q"])

        by_title = {source.title: source for source in sources}
        self.assertEqual(by_title["Fetched"].fetch_status, "full")
        self.assertFalse(by_title["Fetched"].truncated)
        self.assertEqual(by_title["Fetched"].content, "f" * 20)
        self.assertEqual(by_title["fallback"].fetch_status, "fallback")
        self.assertFalse(by_title["fallback"].truncated)

    async def test_kb_evidence_keeps_real_hit_and_parent_content_locations(self):
        hits = [
            {
                "kb_id": "kb-1",
                "source_id": "source-1",
                "source_type": "document",
                "doc_name": "Doc",
                "chunk_id": "child-7",
                "content_chunk_id": "parent-2",
                "content": "evidence",
            }
        ]
        with (
            patch(
                "app.repositories.knowledge_base_repository.KnowledgeBaseRepository.get_by_name",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch("app.core.rag.search.hybrid_search", new_callable=AsyncMock, return_value=hits),
        ):
            sources = await gather_kb_sources(object(), object(), ["q"], None)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].type, SOURCE_KB)
        self.assertEqual(sources[0].locations[0]["source_id"], "source-1")
        self.assertEqual(sources[0].locations[0]["source_type"], "document")
        self.assertEqual(sources[0].locations[0]["retrieval_hit_chunk_id"], "child-7")
        self.assertEqual(sources[0].locations[0]["content_chunk_id"], "parent-2")

    def test_same_origin_deduplicates_across_content_versions_and_three_rounds(self):
        rounds = [
            Source(
                0,
                SOURCE_WEB,
                "A",
                "version one",
                "HTTPS://Example.com/a/?b=2&a=1#x",
                retrieved_at="2026-09-15T01:00:00+00:00",
            ),
            Source(
                0,
                SOURCE_WEB,
                "A2",
                "version two",
                "https://example.com/a?a=1&b=2",
                retrieved_at="2026-09-15T02:00:00+00:00",
            ),
            Source(
                0,
                SOURCE_WEB,
                "A3",
                "version three",
                "https://example.com/a/?a=1&b=2#y",
                retrieved_at="2026-09-15T03:00:00+00:00",
            ),
        ]
        self.assertEqual(len({source.stable_id for source in rounds}), 1)
        self.assertEqual(len({source.content_hash for source in rounds}), 3)
        merged = []
        for source in rounds:
            merged, _ = deduplicate_sources(merged, [source])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].index, 1)
        self.assertTrue(merged[0].stable_id)
        self.assertIn("version three", merged[0].content)
        self.assertEqual(merged[0].retrieved_at, "2026-09-15T03:00:00+00:00")

    def test_web_full_evidence_never_downgrades_to_fallback(self):
        full = Source(
            1,
            SOURCE_WEB,
            "full",
            "complete body",
            "https://example.com/a",
            fetch_status="full",
        )
        fallback = Source(
            0,
            SOURCE_WEB,
            "fallback",
            "search snippet",
            "https://example.com/a",
            fetch_status="fallback",
            truncated=True,
        )
        stable_id = full.stable_id
        content_hash = full.content_hash

        merged, _ = deduplicate_sources([full], [fallback])

        self.assertEqual(merged[0].stable_id, stable_id)
        self.assertEqual(merged[0].index, 1)
        self.assertEqual(merged[0].fetch_status, "full")
        self.assertEqual(merged[0].content, "complete body")
        self.assertEqual(merged[0].content_hash, content_hash)
        self.assertFalse(merged[0].truncated)

    def test_rejected_web_fallback_keeps_full_version_retrieval_time(self):
        full = Source(
            1,
            SOURCE_WEB,
            "full",
            "complete body",
            "https://example.com/a",
            fetch_status="full",
            retrieved_at="2026-09-15T01:00:00+00:00",
        )
        fallback = Source(
            0,
            SOURCE_WEB,
            "fallback",
            "search snippet",
            "https://example.com/a",
            fetch_status="fallback",
            retrieved_at="2026-09-15T02:00:00+00:00",
        )

        merged, _ = deduplicate_sources([full], [fallback])

        self.assertEqual(merged[0].retrieved_at, "2026-09-15T01:00:00+00:00")

    def test_web_fallback_upgrades_to_full_without_changing_identity(self):
        fallback = Source(
            1,
            SOURCE_WEB,
            "fallback",
            "search snippet",
            "https://example.com/a",
            fetch_status="fallback",
            retrieved_at="2026-09-15T01:00:00+00:00",
        )
        full = Source(
            0,
            SOURCE_WEB,
            "full",
            "complete body",
            "https://example.com/a",
            fetch_status="full",
            retrieved_at="2026-09-15T02:00:00+00:00",
        )
        stable_id = fallback.stable_id

        merged, _ = deduplicate_sources([fallback], [full])

        self.assertEqual(merged[0].stable_id, stable_id)
        self.assertEqual(merged[0].index, 1)
        self.assertEqual(merged[0].fetch_status, "full")
        self.assertEqual(merged[0].content, "complete body")
        self.assertEqual(merged[0].retrieved_at, "2026-09-15T02:00:00+00:00")

    async def test_parent_resolution_reports_actual_content_chunk(self):
        parent_es = SimpleNamespace(
            search=AsyncMock(
                return_value={"hits": {"hits": [{"_source": {"content": "parent body"}}]}}
            )
        )
        missing_es = SimpleNamespace(search=AsyncMock(return_value={"hits": {"hits": []}}))
        child = {"parent_id": "parent-2", "content": "child body"}

        self.assertEqual(
            await _resolve_parent_content(parent_es, "user", "child-7", child),
            ("parent body", "parent-2"),
        )
        self.assertEqual(
            await _resolve_parent_content(missing_es, "user", "child-7", child),
            ("child body", "child-7"),
        )

    def test_kb_same_source_merges_new_chunk_provenance_and_evidence(self):
        first = Source(
            0,
            SOURCE_KB,
            "Doc",
            "parent one",
            origin_ref="kb:kb-1:source:source-1",
            locations=[{"retrieval_hit_chunk_id": "child-1", "content_chunk_id": "parent-1"}],
        )
        second = Source(
            0,
            SOURCE_KB,
            "Doc",
            "parent two",
            origin_ref="kb:kb-1:source:source-1",
            locations=[{"retrieval_hit_chunk_id": "child-2", "content_chunk_id": "parent-2"}],
        )
        merged, added = deduplicate_sources([first], [second])

        self.assertEqual(len(merged), 1)
        self.assertEqual(added, [merged[0]])
        self.assertIn("parent one", merged[0].content)
        self.assertIn("parent two", merged[0].content)
        self.assertEqual(len(merged[0].locations), 2)

    def test_mcp_origin_uses_canonical_args_hash_not_raw_args(self):
        origin_ref = Source.mcp_origin_ref("tool", {"secret": "raw-value"})
        source = Source(0, SOURCE_MCP, "tool", "content", origin_ref=origin_ref)

        self.assertNotIn("raw-value", source.origin_ref)
        self.assertNotIn("raw-value", str(source.as_verifier_evidence()))

        same_call = Source(
            0,
            SOURCE_MCP,
            "tool",
            "new content",
            origin_ref=Source.mcp_origin_ref("tool", {"b": 2, "a": 1}),
        )
        reordered_call = Source(
            0,
            SOURCE_MCP,
            "tool",
            "newer content",
            origin_ref=Source.mcp_origin_ref("tool", {"a": 1, "b": 2}),
        )
        merged, _ = deduplicate_sources([], [same_call, reordered_call])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].index, 1)

    def test_verifier_prompt_contains_stable_identity_and_evidence_content(self):
        source = Source(1, SOURCE_WEB, "A", "TAIL FACT", "https://example.com/a")
        prompt = _render_research_prompt(
            topic="topic",
            rubric=RESEARCH_RUBRIC,
            artifact={
                "markdown": "report",
                "headings": ["h"],
                "sources": [source.as_verifier_evidence()],
            },
        )

        self.assertIn(source.stable_id, prompt)
        self.assertIn(source.content_ref, prompt)
        self.assertIn("TAIL FACT", prompt)

    def test_judge_evidence_has_deterministic_total_budget(self):
        sources = [
            Source(i + 1, SOURCE_WEB, f"S{i}", chr(65 + i) * 6000, f"https://e.test/{i}")
            for i in range(8)
        ]
        evidence = [source.as_verifier_evidence() for source in sources]
        context = _render_evidence_context(evidence)

        self.assertLessEqual(len(context), _JUDGE_EVIDENCE_TOTAL_CHARS)
        self.assertIn("truncated: true", context)
        self.assertEqual(context, _render_evidence_context(evidence))

    def test_judge_budget_prioritizes_later_sources_cited_by_report(self):
        sources = [
            Source(i + 1, SOURCE_WEB, f"S{i + 1}", str(i + 1) * 6000, f"https://e/{i}")
            for i in range(8)
        ]
        prompt = _render_research_prompt(
            topic="topic",
            rubric=RESEARCH_RUBRIC,
            artifact={
                "markdown": "关键结论[来源 8]",
                "sources": [source.as_verifier_evidence() for source in sources],
            },
        )

        self.assertIn("[来源 8] S8", prompt)
        self.assertIn("8888888888", prompt)
        self.assertIn("[来源 1] omitted: true", prompt)
        self.assertIn("不得仅据此判定来源错误或不可信", prompt)

    def test_engine_artifact_preserves_citations_before_markdown_linkification(self):
        sources = [
            Source(i + 1, SOURCE_WEB, f"S{i + 1}", str(i + 1) * 6000, f"https://e/{i}")
            for i in range(8)
        ]
        artifact = _build_verifier_artifact(
            "title",
            {"tldr": "摘要", "key_points": []},
            [("结论", "关键事实[来源 8]")],
            sources,
        )

        self.assertNotIn("[来源 8]", artifact["markdown"])
        self.assertEqual(artifact["cited_source_indices"], [8])
        prompt = _render_research_prompt(
            topic="topic",
            rubric=RESEARCH_RUBRIC,
            artifact=artifact,
        )
        self.assertIn("[来源 8] S8", prompt)
        self.assertIn("8888888888", prompt)
        self.assertIn("[来源 1] omitted: true", prompt)

    def test_3500_char_evidence_is_full_for_judge_but_not_marked_truncated(self):
        source = Source(1, SOURCE_WEB, "A", "x" * 3500, "https://example.com/a")
        context = _render_evidence_context([source.as_verifier_evidence()])

        self.assertIn("x" * 3500, context)
        self.assertIn("truncated: false", context)
        self.assertFalse(source.as_persisted_evidence()["truncated"])

    async def test_distiller_keeps_existing_3000_char_working_limit(self):
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content='{"learnings": [{"text": "ok", "relevance": 1}]}'
                )
            )
        )
        source = Source(1, SOURCE_WEB, "A", "x" * 3500, "https://example.com/a")
        with patch(
            "app.core.agent.research.distiller.settings.research_source_truncate_chars", 3000
        ):
            await _distill_one(model, "topic", ["h"], source, "2026-09-15")

        prompt = model.ainvoke.await_args.args[0]
        self.assertIn("x" * 3000, prompt)
        self.assertNotIn("x" * 3001, prompt)

    async def test_report_persists_full_evidence_separately_from_public_sources(self):
        report = SimpleNamespace(user_id="user")
        repo = SimpleNamespace(
            get_by_id=AsyncMock(return_value=report),
            save=AsyncMock(),
        )
        public = [{"index": 1, "title": "A", "url": None}]
        evidence = [{"stable_id": "sid", "content": "private full text"}]
        service = ResearchService(object())

        with patch(
            "app.services.research_service.ResearchReportRepository",
            return_value=repo,
        ):
            await service._finish(object(), object(), "title", "md", None, public, evidence)

        self.assertEqual(report.sources, public)
        self.assertEqual(report.evidence_sources[0]["content"], "private full text")
        self.assertFalse(report.evidence_sources[0]["storage_truncated"])
        self.assertNotIn("content", report.sources[0])

    async def test_outcome_handler_failure_becomes_failed_terminal(self):
        async def tool(query: str) -> str:
            return query

        async def broken_handler(_call, _outcome):
            raise RuntimeError("sink unavailable")

        model = _ScriptedFCModel(
            [[_fc_tool_chunk("fixture", "call-failed")], [AIMessageChunk(content="unused")]]
        )
        events = [
            event
            async for event in run_function_calling(
                model,
                [_query_tool("fixture", tool)],
                [],
                outcome_handler=broken_handler,
            )
        ]

        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(events[-1]["stop_reason"], "outcome_handler_failed")

    async def test_outcome_handler_cancellation_emits_terminal_then_reraises(self):
        async def tool(query: str) -> str:
            return query

        async def cancelled_handler(_call, _outcome):
            raise asyncio.CancelledError

        model = _ScriptedFCModel([[_fc_tool_chunk("fixture", "call-cancelled")]])
        stream = run_function_calling(
            model,
            [_query_tool("fixture", tool)],
            [],
            outcome_handler=cancelled_handler,
        )

        start = await anext(stream)
        terminal = await anext(stream)
        self.assertEqual(start["type"], "tool_start")
        self.assertEqual(terminal["status"], "cancelled")
        self.assertEqual(terminal["stop_reason"], "external_cancellation")
        with self.assertRaises(asyncio.CancelledError):
            await anext(stream)

    def test_loop_event_does_not_expose_evidence_content(self):
        event = {
            "type": "loop_finished",
            "final_artifact": {
                "markdown": "report",
                "artifact_version": "internal-version",
                "cited_source_indices": [1],
                "claims": [
                    {
                        "claim_id": "internal-claim",
                        "section_id": "section:1",
                        "claim_text": "private claim",
                    }
                ],
                "sources": [
                    {
                        "index": 1,
                        "type": "mcp",
                        "title": "tool",
                        "url": None,
                        "stable_id": "sid",
                        "content": "private full text",
                    }
                ],
            },
        }
        public = _public_loop_event(event)

        self.assertEqual(public["final_artifact"]["markdown"], "report")
        self.assertNotIn("content", public["final_artifact"]["sources"][0])
        self.assertNotIn("stable_id", public["final_artifact"]["sources"][0])
        self.assertNotIn("claims", public["final_artifact"])
        self.assertNotIn("artifact_version", public["final_artifact"])
        self.assertNotIn("internal-claim", str(public))

    def test_report_detail_does_not_expose_full_evidence(self):
        report = SimpleNamespace(
            id="r",
            topic="topic",
            title="title",
            status="done",
            report_md="md",
            outline=None,
            sources=[{"title": "public"}],
            evidence_sources=[{"content": "private full"}],
            error_msg=None,
            created_at=None,
        )
        detail = ResearchService.to_detail(report)

        self.assertNotIn("evidence_sources", detail)
        self.assertNotIn("private full", str(detail))

    def test_report_sse_does_not_expose_internal_evidence(self):
        public = _public_report_event(
            {
                "type": "report",
                "sources": [{"title": "public"}],
                "_evidence_sources": [{"content": "private full"}],
            }
        )

        self.assertNotIn("_evidence_sources", public)
        self.assertNotIn("private full", str(public))

    async def test_share_and_task_history_do_not_expose_full_evidence(self):
        share = SimpleNamespace(
            is_active=True,
            expire_at=None,
            view_count=0,
            title="title",
            content_md="md",
            sources=[{"title": "public"}],
            evidence_sources=[{"content": "private full"}],
            created_at=None,
        )
        share_service = ReportShareService(object())
        share_service.repo = SimpleNamespace(
            get_by_token=AsyncMock(return_value=share), save=AsyncMock(return_value=share)
        )
        public_share = await share_service.get_public("token")

        report = SimpleNamespace(
            id="report",
            title="title",
            topic="topic",
            status="done",
            error_msg=None,
            created_at=None,
            evidence_sources=[{"content": "private full"}],
        )

        class _LoopResult:
            def scalars(self):
                return self

            def all(self):
                return []

        session = SimpleNamespace(execute=AsyncMock(return_value=_LoopResult()))
        task_service = AgentTaskService(session)
        task_service._get_or_404 = AsyncMock(return_value=object())
        with patch(
            "app.repositories.research_report_repository.ResearchReportRepository.list_by_task",
            new_callable=AsyncMock,
            return_value=[report],
        ):
            history = await task_service.list_runs("user", "task")

        self.assertNotIn("private full", str(public_share))
        self.assertNotIn("private full", str(history))
        self.assertNotIn("evidence_sources", history[0])


class ResearchEvidenceRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_cleanup_is_best_effort_and_continues(self):
        storage = SimpleNamespace(
            delete=AsyncMock(side_effect=[RuntimeError("delete failed"), None])
        )
        with patch("app.core.agent.research.evidence_store.get_storage", return_value=storage):
            await delete_external_evidence(
                ["storage:user/report/one.txt", "storage:user/report/two.txt"]
            )

        self.assertEqual(storage.delete.await_count, 2)

    async def test_hot_report_queries_defer_large_evidence_column(self):
        statements = []

        class _Result:
            def scalar_one_or_none(self):
                return None

            def scalars(self):
                return self

            def all(self):
                return []

        session = SimpleNamespace(
            execute=AsyncMock(side_effect=lambda stmt: statements.append(stmt) or _Result()),
            scalar=AsyncMock(return_value=0),
        )
        repo = ResearchReportRepository(session)
        await repo.get("user", "report")
        await repo.get_by_id("report")
        await repo.list_paged("user", 1, 10)
        await repo.list_by_task("user", "task")

        for statement in statements:
            sql = str(statement.compile(dialect=postgresql.dialect()))
            self.assertNotIn("evidence_sources", sql)

    async def test_content_ref_resolver_reads_inline_and_external_evidence(self):
        inline_hash = hashlib.sha256(b"inline full").hexdigest()
        external_hash = hashlib.sha256(b"external full").hexdigest()
        inline_evidence = {
            "content_ref": f"evidence:sid:{inline_hash}",
            "content_hash": inline_hash,
            "content": "inline full",
        }
        external_ref = f"evidence:external:{external_hash}"
        repo = SimpleNamespace(
            get_evidence_sources=AsyncMock(
                return_value=[
                    inline_evidence,
                    {
                        "content_ref": external_ref,
                        "content_hash": external_hash,
                        "content": None,
                        "storage_ref": "storage:user/research-evidence/report/sid-hash.txt",
                    },
                ]
            )
        )
        service = ResearchService(object())
        service.repo = repo
        with patch(
            "app.services.research_service.read_external_evidence",
            new_callable=AsyncMock,
            return_value="external full",
        ):
            inline = await service.resolve_evidence_content(
                "user", "report", f"evidence:sid:{inline_hash}"
            )
            external = await service.resolve_evidence_content("user", "report", external_ref)

        self.assertEqual(inline, "inline full")
        self.assertEqual(external, "external full")

    async def test_content_ref_resolver_rejects_hash_mismatch(self):
        repo = SimpleNamespace(
            get_evidence_sources=AsyncMock(
                return_value=[
                    {
                        "content_ref": "evidence:sid:wrong",
                        "content_hash": "0" * 64,
                        "content": "tampered",
                    }
                ]
            )
        )
        service = ResearchService(object())
        service.repo = repo

        self.assertIsNone(
            await service.resolve_evidence_content("user", "report", "evidence:sid:wrong")
        )

        repo.get_evidence_sources.return_value = [
            {
                "content_ref": "evidence:sid:external-wrong",
                "content_hash": "0" * 64,
                "storage_ref": "storage:user/report/evidence.txt",
            }
        ]
        with patch(
            "app.services.research_service.read_external_evidence",
            new_callable=AsyncMock,
            return_value="tampered external",
        ):
            self.assertIsNone(
                await service.resolve_evidence_content(
                    "user", "report", "evidence:sid:external-wrong"
                )
            )

    async def test_delete_report_cleans_external_evidence(self):
        report = SimpleNamespace(id="report")
        service = ResearchService(object())
        service._get_or_404 = AsyncMock(return_value=report)
        service.repo = SimpleNamespace(
            get_evidence_sources=AsyncMock(
                return_value=[{"storage_ref": "storage:user/report/evidence.txt"}]
            ),
            delete=AsyncMock(),
        )
        with patch(
            "app.services.research_service.delete_external_evidence",
            new_callable=AsyncMock,
        ) as cleanup:
            await service.delete("user", "report")

        service.repo.delete.assert_awaited_once_with(report)
        cleanup.assert_awaited_once_with(["storage:user/report/evidence.txt"])

    async def test_report_save_failure_cleans_new_external_evidence(self):
        report = SimpleNamespace(user_id="user")
        repo = SimpleNamespace(
            get_by_id=AsyncMock(return_value=report),
            save=AsyncMock(side_effect=RuntimeError("commit failed")),
        )
        service = ResearchService(object())
        evidence = [
            {
                "stable_id": "sid",
                "content_hash": "hash",
                "content": "oversized",
            }
        ]
        with (
            patch("app.services.research_service.ResearchReportRepository", return_value=repo),
            patch("app.core.agent.research.evidence_store.MAX_INLINE_EVIDENCE_CHARS", 0),
            patch(
                "app.services.research_service.delete_external_evidence",
                new_callable=AsyncMock,
            ) as cleanup,
            patch(
                "app.core.agent.research.evidence_store.get_storage",
                return_value=SimpleNamespace(
                    exists=AsyncMock(return_value=False),
                    save=AsyncMock(return_value="key"),
                ),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "commit failed"):
                await service._finish(object(), "report", "title", "md", None, [], evidence)

        cleanup.assert_awaited_once()
        self.assertEqual(len(cleanup.await_args.args[0]), 1)

    async def test_oversized_evidence_uses_storage_without_silent_truncation(self):
        storage = SimpleNamespace(
            exists=AsyncMock(return_value=False),
            save=AsyncMock(return_value="key"),
        )
        evidence = [
            {
                "stable_id": "sid",
                "content_hash": "hash",
                "content_ref": "evidence:sid:hash",
                "content": "complete oversized evidence",
                "truncated": False,
            }
        ]
        with (
            patch("app.core.agent.research.evidence_store.MAX_INLINE_EVIDENCE_CHARS", 5),
            patch("app.core.agent.research.evidence_store.get_storage", return_value=storage),
        ):
            prepared = await prepare_evidence_sources("user", "report", evidence)

        storage.save.assert_awaited_once()
        self.assertIsNone(prepared[0]["content"])
        self.assertEqual(prepared[0]["content_ref"], "evidence:sid:hash")
        self.assertTrue(prepared[0]["storage_ref"].startswith("storage:"))
        self.assertFalse(prepared[0]["storage_truncated"])


if __name__ == "__main__":
    unittest.main()

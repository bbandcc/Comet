import json
import unittest
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.core.agent.loop.controller import LoopController, RepairCallbackArgs
from app.core.agent.loop.models import (
    DECISION_JUDGE_ERROR,
    DECISION_RETRY_PATCH,
    QUALITY_FAILED,
    QUALITY_JUDGE_ERROR,
    QUALITY_PASSED,
    QUALITY_SKIPPED,
    QUALITY_UNAVAILABLE,
    IterationOutcome,
    VerifyScore,
)
from app.core.agent.loop.rubric.research import RESEARCH_RUBRIC
from app.core.agent.loop.store import LoopStore
from app.core.agent.loop.verifier import (
    JudgeError,
    SameModelVerifier,
    VerifierUnavailableError,
    build_verifier,
)
from app.core.agent.research.engine import run_research
from app.services.agent_task_service import AgentTaskService
from app.services.dashboard_service import DashboardService
from app.services.research_service import ResearchService
from app.tasks.agent_task import _check_loop_passed


def _raw_scores(value: float) -> dict[str, float]:
    return {dim.key: value for dim in RESEARCH_RUBRIC.dims}


def _judge_json(raw_scores) -> str:
    return json.dumps(
        {
            "raw_scores": raw_scores,
            "feedback": {
                "summary": "fixture",
                "issues": [],
                "missing_coverage": [],
                "wrong_citations": [],
                "weak_chapters": [],
            },
        },
        allow_nan=True,
    )


def _judge_json_with_feedback(feedback) -> str:
    return json.dumps(
        {"raw_scores": _raw_scores(4.0), "feedback": feedback},
        allow_nan=True,
    )


class _ScriptedModel:
    def __init__(self, *, content: str = "", error: Exception | None = None):
        self.content = content
        self.error = error

    async def ainvoke(self, _messages):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)


class VerifierOutputContractTests(unittest.IsolatedAsyncioTestCase):
    async def _verify(self, content: str) -> VerifyScore:
        verifier = SameModelVerifier(_ScriptedModel(content=content))
        return await verifier.verify(topic="topic", artifact={}, rubric=RESEARCH_RUBRIC)

    async def test_timeout_and_model_call_failure_are_judge_errors(self):
        for error in (TimeoutError("provider timeout"), RuntimeError("provider failed")):
            with self.subTest(error=type(error).__name__):
                verifier = SameModelVerifier(_ScriptedModel(error=error))
                with self.assertRaises(JudgeError):
                    await verifier.verify(topic="topic", artifact={}, rubric=RESEARCH_RUBRIC)

    async def test_invalid_outputs_are_judge_errors(self):
        missing = _raw_scores(4.0)
        missing.pop(RESEARCH_RUBRIC.dims[0].key)
        invalid_cases = {
            "non_json": "not json",
            "top_level_not_object": "[]",
            "raw_scores_not_object": _judge_json([1, 2, 3]),
            "missing_dimension": _judge_json(missing),
            "numeric_string": _judge_json({**_raw_scores(4.0), "coverage": "4"}),
            "boolean": _judge_json({**_raw_scores(4.0), "coverage": True}),
            "nan": _judge_json({**_raw_scores(4.0), "coverage": float("nan")}),
            "infinity": _judge_json({**_raw_scores(4.0), "coverage": float("inf")}),
            "overflow": _judge_json({**_raw_scores(4.0), "coverage": 10**400}),
            "negative": _judge_json({**_raw_scores(4.0), "coverage": -0.1}),
            "over_raw_max": _judge_json(
                {**_raw_scores(4.0), "coverage": RESEARCH_RUBRIC.raw_max + 0.1}
            ),
            "feedback_not_object": json.dumps({"raw_scores": _raw_scores(4.0), "feedback": []}),
        }

        for name, content in invalid_cases.items():
            with self.subTest(name=name), self.assertRaises(JudgeError):
                await self._verify(content)

    async def test_feedback_must_match_the_prompt_schema(self):
        valid = {
            "summary": "fixture",
            "issues": [{"dim": "coverage", "detail": "missing section"}],
            "missing_coverage": ["subtopic"],
            "wrong_citations": [3],
            "weak_chapters": ["Conclusion"],
        }
        invalid_feedback = {
            "missing_summary": {k: v for k, v in valid.items() if k != "summary"},
            "summary_not_string": {**valid, "summary": 1},
            "issues_not_list": {**valid, "issues": {}},
            "issue_not_object": {**valid, "issues": ["bad"]},
            "issue_missing_dim": {**valid, "issues": [{"detail": "bad"}]},
            "issue_missing_detail": {**valid, "issues": [{"dim": "coverage"}]},
            "issue_dim_not_string": {**valid, "issues": [{"dim": 1, "detail": "bad"}]},
            "issue_detail_not_string": {**valid, "issues": [{"dim": "coverage", "detail": 1}]},
            "missing_coverage_not_list": {**valid, "missing_coverage": "bad"},
            "missing_coverage_item_not_string": {**valid, "missing_coverage": [1]},
            "wrong_citations_not_list": {**valid, "wrong_citations": 3},
            "wrong_citation_not_integer": {**valid, "wrong_citations": ["3"]},
            "wrong_citation_boolean": {**valid, "wrong_citations": [True]},
            "weak_chapters_not_list": {**valid, "weak_chapters": "bad"},
            "weak_chapter_not_string": {**valid, "weak_chapters": [1]},
        }

        missing_feedback = json.dumps({"raw_scores": _raw_scores(4.0)})
        with self.assertRaises(JudgeError):
            await self._verify(missing_feedback)
        for name, feedback in invalid_feedback.items():
            with self.subTest(name=name), self.assertRaises(JudgeError):
                await self._verify(_judge_json_with_feedback(feedback))

        score = await self._verify(_judge_json_with_feedback(valid))
        self.assertEqual(score.feedback, valid)

    async def test_valid_pass_and_failed_quality_scores_remain_normal_scores(self):
        passing = await self._verify(_judge_json(_raw_scores(4.0)))
        failing = await self._verify(_judge_json(_raw_scores(1.0)))

        self.assertEqual(passing.raw_scores, _raw_scores(4.0))
        self.assertEqual(passing.total, 0.8)
        self.assertEqual(failing.raw_scores, _raw_scores(1.0))
        self.assertEqual(failing.total, 0.2)


class CrossVerifierAvailabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_cross_config_is_unavailable(self):
        with patch(
            "app.core.agent.loop.verifier.llm_verifier." "ModelConfigRepository.list_by_user",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with self.assertRaises(VerifierUnavailableError):
                await build_verifier(
                    object(),
                    uuid.uuid4(),
                    kind="cross",
                    generator_model=object(),
                )

    async def test_cross_config_query_failure_is_unavailable(self):
        session = SimpleNamespace(rollback=AsyncMock())
        with patch(
            "app.core.agent.loop.verifier.llm_verifier." "ModelConfigRepository.list_by_user",
            new_callable=AsyncMock,
            side_effect=RuntimeError("database unavailable"),
        ):
            with self.assertRaises(VerifierUnavailableError):
                await build_verifier(
                    session,
                    uuid.uuid4(),
                    kind="cross",
                    generator_model=object(),
                )
        session.rollback.assert_awaited_once_with()

    async def test_cross_model_build_failure_is_unavailable(self):
        config = SimpleNamespace(is_default=True, model_name="judge")
        with (
            patch(
                "app.core.agent.loop.verifier.llm_verifier." "ModelConfigRepository.list_by_user",
                new_callable=AsyncMock,
                return_value=[config],
            ),
            patch(
                "app.core.agent.loop.verifier.llm_verifier.build_chat_model",
                side_effect=RuntimeError("invalid credentials"),
            ),
        ):
            with self.assertRaises(VerifierUnavailableError):
                await build_verifier(
                    object(),
                    uuid.uuid4(),
                    kind="cross",
                    generator_model=object(),
                )


class _FakeStore:
    def __init__(self) -> None:
        self.run_id = uuid.uuid4()
        self.created: dict | None = None
        self.iterations = []
        self.finished: dict | None = None

    async def create_run(self, **kwargs):
        self.created = kwargs
        return SimpleNamespace(id=self.run_id)

    async def record_iteration(self, _run_id, outcome) -> None:
        self.iterations.append(outcome)

    async def finish_run(self, _run_id, **kwargs) -> None:
        self.finished = kwargs


class LoopStoreIterationCountTests(unittest.IsolatedAsyncioTestCase):
    async def test_record_iteration_updates_run_to_persisted_iteration_number(self):
        run = SimpleNamespace(iterations=0)
        session = SimpleNamespace(
            add=Mock(),
            get=AsyncMock(return_value=run),
            commit=AsyncMock(),
            rollback=AsyncMock(),
        )
        score = VerifyScore(raw_scores=_raw_scores(4.0), total=0.8)

        await LoopStore(session).record_iteration(
            uuid.uuid4(),
            IterationOutcome(iteration_no=1, score=score, decision="execution_error"),
        )

        self.assertEqual(run.iterations, 1)
        persisted = session.add.call_args.args[0]
        self.assertEqual(persisted.iteration_no, 1)
        self.assertEqual(persisted.scores, {"raw": score.raw_scores, "total": score.total})
        self.assertEqual(persisted.decision, "execution_error")
        session.commit.assert_awaited_once_with()
        session.rollback.assert_not_awaited()


class _FakeVerifier:
    kind = "same"
    model_name = "fixture-judge"

    def __init__(self, score: VerifyScore | None = None, error: Exception | None = None):
        self.score = score
        self.error = error

    async def verify(self, **_kwargs) -> VerifyScore:
        if self.error is not None:
            raise self.error
        if self.score is None:
            raise AssertionError("test verifier needs a score or error")
        return self.score


class _SequenceVerifier(_FakeVerifier):
    def __init__(self, results: list[VerifyScore | Exception]):
        self.results = list(results)

    async def verify(self, **_kwargs) -> VerifyScore:
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _PolicyMustNotRun:
    def __init__(self) -> None:
        self.decide = Mock(
            side_effect=AssertionError("invalid judge outcomes must not reach repair policy")
        )


class _PolicyRaises:
    def decide(self, **_kwargs):
        raise RuntimeError("policy failed")


class _PlanRaises:
    def plan(self, **_kwargs):
        raise RuntimeError("repair plan failed")


class _PlanRaisesPolicy:
    def decide(self, **_kwargs):
        return DECISION_RETRY_PATCH, _PlanRaises()


class LoopControllerQualityStatusTests(unittest.IsolatedAsyncioTestCase):
    def _controller(self, *, max_iterations: int = 1, policy=None):
        controller = LoopController(
            session=object(),
            user_id=uuid.uuid4(),
            task_type="research",
            task_id=uuid.uuid4(),
            max_iterations=max_iterations,
            policy=policy,
        )
        store = _FakeStore()
        controller.store = store
        return controller, store

    async def _run(self, controller: LoopController, verifier=None, **kwargs):
        mocked_builder = AsyncMock(return_value=verifier)
        if isinstance(verifier, Exception):
            mocked_builder = AsyncMock(side_effect=verifier)
        with patch("app.core.agent.loop.controller.build_verifier", new=mocked_builder):
            events = [
                event
                async for event in controller.run(
                    topic="topic",
                    initial_artifact={"title": "report", "markdown": "body"},
                    verifier_kind="cross",
                    generator_model=object(),
                    repair_ctx=kwargs.get("repair_ctx"),
                    enabled=kwargs.get("enabled", True),
                )
            ]
        return events, mocked_builder

    async def test_judge_error_does_not_call_policy_or_repair(self):
        controller, store = self._controller(policy=_PolicyMustNotRun())
        patch_callback = AsyncMock()
        rewrite_callback = AsyncMock()

        events, _ = await self._run(
            controller,
            _FakeVerifier(error=JudgeError("invalid output")),
            repair_ctx=RepairCallbackArgs(
                patch_callback=patch_callback,
                rewrite_callback=rewrite_callback,
            ),
        )

        self.assertFalse(any(event["type"].startswith("loop_repair") for event in events))
        self.assertEqual(events[-1]["quality_status"], QUALITY_JUDGE_ERROR)
        self.assertEqual(store.finished["quality_status"], QUALITY_JUDGE_ERROR)
        self.assertIsNone(store.finished["final_score"])
        self.assertEqual(len(store.iterations), 1)
        self.assertIsNone(store.iterations[0].score)
        self.assertEqual(store.iterations[0].decision, DECISION_JUDGE_ERROR)
        controller.policy.decide.assert_not_called()
        patch_callback.assert_not_awaited()
        rewrite_callback.assert_not_awaited()

    async def test_invalid_feedback_does_not_reach_policy_or_repair(self):
        controller, store = self._controller(policy=_PolicyMustNotRun())
        verifier = SameModelVerifier(
            _ScriptedModel(
                content=_judge_json_with_feedback(
                    {
                        "summary": "fixture",
                        "issues": [],
                        "missing_coverage": [1],
                        "wrong_citations": [],
                        "weak_chapters": [],
                    }
                )
            )
        )

        events, _ = await self._run(controller, verifier)

        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(events[-1]["quality_status"], QUALITY_JUDGE_ERROR)
        self.assertEqual(store.finished["quality_status"], QUALITY_JUDGE_ERROR)
        controller.policy.decide.assert_not_called()

    async def test_controller_error_before_judge_has_no_quality_conclusion(self):
        controller, store = self._controller()
        with patch(
            "app.core.agent.loop.controller.get_tracer", side_effect=RuntimeError("trace failed")
        ):
            events, _ = await self._run(controller, _FakeVerifier(score=VerifyScore()))

        self.assertEqual(events[-1]["status"], "failed")
        self.assertIsNone(events[-1]["quality_status"])
        self.assertEqual(events[-1]["iterations"], 0)
        self.assertEqual(store.iterations, [])
        self.assertIsNone(store.finished["quality_status"])

    async def test_policy_and_repair_plan_errors_preserve_valid_quality_conclusion(self):
        passing = VerifyScore(raw_scores=_raw_scores(4.0), total=0.8)
        repairable_raw = _raw_scores(4.0)
        repairable_raw["coverage"] = 2.0
        repairable = VerifyScore(
            raw_scores=repairable_raw,
            total=RESEARCH_RUBRIC.weighted_total(repairable_raw),
        )
        cases = (
            (_PolicyRaises(), passing, QUALITY_PASSED, "execution_error"),
            (_PlanRaisesPolicy(), repairable, QUALITY_FAILED, DECISION_RETRY_PATCH),
        )

        for policy, score, expected_quality, expected_decision in cases:
            with self.subTest(policy=type(policy).__name__):
                controller, store = self._controller(max_iterations=2, policy=policy)
                events, _ = await self._run(controller, _FakeVerifier(score=score))

                self.assertEqual(events[-1]["status"], "failed")
                self.assertEqual(events[-1]["quality_status"], expected_quality)
                self.assertEqual(store.finished["quality_status"], expected_quality)
                self.assertEqual(store.finished["final_score"], score.total)
                self.assertNotEqual(events[-1]["quality_status"], QUALITY_JUDGE_ERROR)
                self.assertEqual(events[-1]["iterations"], 1)
                self.assertEqual(len(store.iterations), 1)
                self.assertEqual(store.iterations[0].score, score)
                self.assertEqual(store.iterations[0].decision, expected_decision)

    async def test_tracing_error_after_valid_score_preserves_iteration(self):
        score = VerifyScore(raw_scores=_raw_scores(4.0), total=0.8)
        controller, store = self._controller()

        class PayloadFailTracer:
            @asynccontextmanager
            async def span(self, *_args, **_kwargs):
                yield SimpleNamespace(
                    set_iteration_id=Mock(),
                    set_payload=Mock(side_effect=RuntimeError("trace payload failed")),
                )

        with patch("app.core.agent.loop.controller.get_tracer", return_value=PayloadFailTracer()):
            events, _ = await self._run(controller, _FakeVerifier(score=score))

        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(events[-1]["quality_status"], QUALITY_PASSED)
        self.assertEqual(events[-1]["iterations"], 1)
        self.assertEqual(len(store.iterations), 1)
        self.assertEqual(store.iterations[0].score, score)
        self.assertEqual(store.iterations[0].decision, "execution_error")

    async def test_later_judge_error_preserves_prior_valid_quality_conclusion(self):
        controller, store = self._controller(max_iterations=2)
        repairable_raw = _raw_scores(4.0)
        repairable_raw["coverage"] = 2.0
        repairable = VerifyScore(
            raw_scores=repairable_raw,
            total=RESEARCH_RUBRIC.weighted_total(repairable_raw),
            feedback={"missing_coverage": ["missing subtopic"]},
        )
        patch_callback = AsyncMock(return_value={"title": "report", "markdown": "repaired"})

        events, _ = await self._run(
            controller,
            _SequenceVerifier([repairable, JudgeError("second judge failed")]),
            repair_ctx=RepairCallbackArgs(patch_callback=patch_callback),
        )

        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(events[-1]["quality_status"], QUALITY_FAILED)
        self.assertEqual(store.finished["quality_status"], QUALITY_FAILED)
        self.assertEqual(store.finished["final_score"], repairable.total)
        self.assertEqual(len(store.iterations), 2)
        self.assertEqual(store.iterations[0].score, repairable)
        self.assertIsNone(store.iterations[1].score)

    async def test_cross_unavailable_finishes_without_verify_or_repair(self):
        controller, store = self._controller(policy=_PolicyMustNotRun())

        events, _ = await self._run(controller, VerifierUnavailableError("cross config missing"))

        self.assertEqual(events[-1]["quality_status"], QUALITY_UNAVAILABLE)
        self.assertEqual(store.finished["quality_status"], QUALITY_UNAVAILABLE)
        self.assertEqual(store.iterations, [])
        self.assertIsNone(store.finished["final_score"])

    async def test_disabled_loop_is_skipped_without_building_judge(self):
        controller, store = self._controller(policy=_PolicyMustNotRun())

        events, builder = await self._run(controller, enabled=False)

        builder.assert_not_awaited()
        self.assertEqual(events[-1]["quality_status"], QUALITY_SKIPPED)
        self.assertEqual(store.finished["quality_status"], QUALITY_SKIPPED)
        self.assertEqual(store.iterations, [])
        self.assertEqual(events[-1]["final_artifact"]["markdown"], "body")

    async def test_valid_scores_map_to_passed_and_failed_quality(self):
        cases = (
            (4.0, QUALITY_PASSED),
            (1.0, QUALITY_FAILED),
        )
        for value, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                controller, store = self._controller(max_iterations=1)
                score = VerifyScore(
                    raw_scores=_raw_scores(value),
                    total=RESEARCH_RUBRIC.weighted_total(_raw_scores(value)),
                )

                events, _ = await self._run(controller, _FakeVerifier(score=score))

                self.assertEqual(events[-1]["quality_status"], expected_status)
                self.assertEqual(store.finished["quality_status"], expected_status)

    async def test_valid_failed_quality_can_use_existing_repair_policy(self):
        controller, store = self._controller(max_iterations=2)
        failing_raw = _raw_scores(4.0)
        failing_raw["coverage"] = 2.0
        failing = VerifyScore(
            raw_scores=failing_raw,
            total=RESEARCH_RUBRIC.weighted_total(failing_raw),
            feedback={"missing_coverage": ["missing subtopic"]},
        )
        passing_raw = _raw_scores(4.0)
        passing = VerifyScore(
            raw_scores=passing_raw,
            total=RESEARCH_RUBRIC.weighted_total(passing_raw),
        )
        patch_callback = AsyncMock(return_value={"title": "report", "markdown": "repaired body"})

        events, _ = await self._run(
            controller,
            _SequenceVerifier([failing, passing]),
            repair_ctx=RepairCallbackArgs(patch_callback=patch_callback),
        )

        patch_callback.assert_awaited_once_with(["missing subtopic"])
        self.assertTrue(any(event["type"] == "loop_repair_start" for event in events))
        self.assertEqual(store.finished["quality_status"], QUALITY_PASSED)


class ResearchDisabledLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_loop_records_skipped_and_still_delivers_report(self):
        plan = SimpleNamespace(
            title="Fixture report",
            sections=[SimpleNamespace(heading="Section", points=[])],
            queries=[],
        )
        curated = [SimpleNamespace(heading="Section", thesis="", learning_ids=[])]
        store = _FakeStore()
        builder = AsyncMock(side_effect=AssertionError("disabled loop must not build judge"))

        async def write_section(*_args, **_kwargs):
            yield "Report body"

        with (
            patch("app.core.agent.research.engine.settings.loop_enabled", False),
            patch("app.core.agent.research.engine.settings.tracing_enabled", False),
            patch(
                "app.core.llm.chat_model.build_default_chat_model",
                new_callable=AsyncMock,
                return_value=(object(), SimpleNamespace(model_name="generator")),
            ),
            patch("app.core.llm.chat_model.supports_function_call", return_value=True),
            patch(
                "app.core.agent.research.engine.get_websearch_config",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.core.agent.research.engine.make_plan",
                new_callable=AsyncMock,
                return_value=plan,
            ),
            patch(
                "app.core.agent.research.engine.gather_kb_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.core.agent.research.engine.gather_mcp_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.core.agent.research.engine.distill_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.core.agent.research.engine.curate_outline",
                new_callable=AsyncMock,
                return_value=curated,
            ),
            patch(
                "app.core.agent.research.engine.write_section_stream",
                new=write_section,
            ),
            patch(
                "app.core.agent.research.engine.summarize",
                new_callable=AsyncMock,
                return_value={"tldr": "Summary", "key_points": []},
            ),
            patch("app.core.agent.loop.controller.LoopStore", return_value=store),
            patch("app.core.agent.loop.controller.build_verifier", new=builder),
        ):
            events = [
                event
                async for event in run_research(
                    object(),
                    uuid.uuid4(),
                    "topic",
                    report_id=uuid.uuid4(),
                )
            ]

        builder.assert_not_awaited()
        finished = next(event for event in events if event["type"] == "loop_finished")
        report = next(event for event in events if event["type"] == "report")
        self.assertEqual(finished["quality_status"], QUALITY_SKIPPED)
        self.assertEqual(store.finished["quality_status"], QUALITY_SKIPPED)
        self.assertIn("Report body", report["markdown"])


class LoopApiQualityStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_loop_detail_exposes_quality_status_separately(self):
        service = ResearchService(object())
        service._get_or_404 = AsyncMock()
        run = SimpleNamespace(
            id=uuid.uuid4(),
            task_type="research",
            status="failed",
            quality_status=QUALITY_JUDGE_ERROR,
            iterations=1,
            final_score=None,
            pass_threshold=0.7,
            max_iterations=2,
            rubric_name="research",
            generator_model="generator",
            verifier_model="judge",
            verifier_kind="cross",
            note="invalid output",
            started_at=None,
            finished_at=None,
        )

        with (
            patch(
                "app.core.agent.loop.store.LoopStore.find_latest_by_task",
                new_callable=AsyncMock,
                return_value=run,
            ),
            patch(
                "app.core.agent.loop.store.LoopStore.list_iterations",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            detail = await service.get_loop_detail(uuid.uuid4(), uuid.uuid4())

        self.assertEqual(detail["status"], "failed")
        self.assertEqual(detail["quality_status"], QUALITY_JUDGE_ERROR)

    async def test_loop_health_uses_only_explicit_and_valid_quality_populations(self):
        runs = [
            SimpleNamespace(
                quality_status=quality_status,
                iterations=iterations,
                final_score=score,
                verifier_kind=verifier_kind,
            )
            for quality_status, iterations, score, verifier_kind in (
                (None, 1, 0.99, "same"),
                (QUALITY_PASSED, 1, 0.8, "same"),
                (QUALITY_FAILED, 2, 0.6, "cross"),
                (QUALITY_JUDGE_ERROR, 1, None, "same"),
                (QUALITY_UNAVAILABLE, 0, None, "cross"),
                (QUALITY_SKIPPED, 0, None, "cross"),
            )
        ]
        run_rows = SimpleNamespace(all=lambda: runs)
        legitimate_failure = {"raw": {**_raw_scores(4.0), "coverage": 1.0}, "total": 0.7}
        fake_failure = {"raw": _raw_scores(0.0), "total": 0.0}
        iteration_rows = SimpleNamespace(
            all=lambda: [
                (legitimate_failure, QUALITY_FAILED),
                (fake_failure, QUALITY_JUDGE_ERROR),
                (fake_failure, None),
                (fake_failure, QUALITY_SKIPPED),
                (fake_failure, QUALITY_UNAVAILABLE),
            ]
        )
        session = SimpleNamespace(execute=AsyncMock(side_effect=[run_rows, iteration_rows]))

        result = await DashboardService(session).loop_health(uuid.uuid4())

        self.assertEqual(result["total"], 5)
        self.assertEqual(result["judged_total"], 2)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["failed_quality"], 1)
        self.assertEqual(result["judge_error"], 1)
        self.assertEqual(result["unavailable"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["pass_rate"], 0.5)
        self.assertEqual(result["one_shot_pass_rate"], 0.5)
        self.assertEqual(result["avg_final_score"], 0.7)
        self.assertEqual(
            result["failure_dims"], [{"dim": "coverage", "label": "覆盖度", "count": 1}]
        )
        self.assertEqual(result["verifier_kinds"], {"same": 2, "cross": 1})


class AgentTaskVerifiedCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_keeps_legacy_status_while_quality_status_is_separate(self):
        report_id = uuid.uuid4()
        report = SimpleNamespace(
            id=report_id,
            title="report",
            topic="topic",
            status="done",
            error_msg=None,
            created_at=None,
        )
        loop_run = SimpleNamespace(
            task_id=report_id,
            status="failed",
            quality_status=QUALITY_JUDGE_ERROR,
            final_score=None,
        )
        scalar_rows = SimpleNamespace(all=lambda: [loop_run])
        session = SimpleNamespace(
            execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: scalar_rows))
        )
        service = AgentTaskService(session)
        service._get_or_404 = AsyncMock()

        with patch(
            "app.repositories.research_report_repository.ResearchReportRepository.list_by_task",
            new_callable=AsyncMock,
            return_value=[report],
        ):
            runs = await service.list_runs(uuid.uuid4(), uuid.uuid4())

        self.assertEqual(runs[0]["verified"], "failed")
        self.assertEqual(runs[0]["quality_status"], QUALITY_JUDGE_ERROR)


@asynccontextmanager
async def _session_context():
    yield object()


class ScheduledPushQualityGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_loop_run_fails_closed(self):
        with (
            patch("app.tasks.agent_task.settings.loop_enabled", True),
            patch(
                "app.core.agent.loop.store.LoopStore.find_latest_by_task",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            self.assertFalse(await _check_loop_passed(_session_context, uuid.uuid4()))

    async def test_loop_query_failure_fails_closed(self):
        with (
            patch("app.tasks.agent_task.settings.loop_enabled", True),
            patch(
                "app.core.agent.loop.store.LoopStore.find_latest_by_task",
                new_callable=AsyncMock,
                side_effect=RuntimeError("database unavailable"),
            ),
        ):
            self.assertFalse(await _check_loop_passed(_session_context, uuid.uuid4()))

    async def test_only_explicit_quality_pass_is_accepted(self):
        for quality_status in (
            QUALITY_PASSED,
            QUALITY_FAILED,
            QUALITY_JUDGE_ERROR,
            QUALITY_UNAVAILABLE,
            QUALITY_SKIPPED,
            None,
        ):
            with (
                self.subTest(quality_status=quality_status),
                patch("app.tasks.agent_task.settings.loop_enabled", True),
                patch(
                    "app.core.agent.loop.store.LoopStore.find_latest_by_task",
                    new_callable=AsyncMock,
                    return_value=SimpleNamespace(quality_status=quality_status),
                ),
            ):
                accepted = await _check_loop_passed(_session_context, uuid.uuid4())
                self.assertEqual(accepted, quality_status == QUALITY_PASSED)

    async def test_disabled_quality_gate_keeps_existing_delivery_behavior(self):
        with patch("app.tasks.agent_task.settings.loop_enabled", False):
            self.assertTrue(await _check_loop_passed(_session_context, uuid.uuid4()))


if __name__ == "__main__":
    unittest.main()

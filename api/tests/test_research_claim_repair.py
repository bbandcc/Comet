import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.core.agent.loop.controller import LoopController, RepairCallbackArgs
from app.core.agent.loop.models import (
    DECISION_EXCEED,
    DECISION_PASS,
    DECISION_RETRY_PATCH,
    QUALITY_FAILED,
    QUALITY_JUDGE_ERROR,
    QUALITY_PASSED,
    ClaimRepairTarget,
    RepairAction,
    VerifyScore,
)
from app.core.agent.loop.policy import Policy
from app.core.agent.loop.repair.patch_repair import PatchRepair
from app.core.agent.loop.rubric.research import RESEARCH_RUBRIC
from app.core.agent.loop.verifier import JudgeError, SameModelVerifier
from app.core.agent.research.claim_repair import (
    ACTION_INSUFFICIENT,
    ACTION_REPLACE,
    ClaimReplacementDecision,
    decide_claim_replacement,
)
from app.core.agent.research.claims import apply_claim_replacements, build_claim_contract
from app.core.agent.research.engine import run_research
from app.core.agent.research.models import SOURCE_WEB, Learning, Source
from app.core.agent.research.writer import summarize
from app.services.research_service import ResearchService


def _scores(value: float = 4.0) -> dict[str, float]:
    return {dim.key: value for dim in RESEARCH_RUBRIC.dims}


def _feedback(claim_verdicts: list[dict]) -> dict:
    return {
        "summary": "fixture",
        "issues": [],
        "missing_coverage": [],
        "wrong_citations": [],
        "weak_chapters": [],
        "claim_verdicts": claim_verdicts,
    }


class ClaimContractTests(unittest.IsolatedAsyncioTestCase):
    def _artifact(self, citation_index: int = 1) -> dict:
        source = Source(
            1,
            SOURCE_WEB,
            "Evidence",
            "The audited value is 10.",
            "https://example.com/evidence",
        )
        version, claims = build_claim_contract(
            {"tldr": "", "key_points": []},
            [("Results", f"The audited value is 20 [来源 {citation_index}].")],
            [source],
        )
        return {
            "title": "Report",
            "markdown": "report",
            "sources": [source.as_verifier_evidence()],
            "headings": ["Results"],
            "artifact_version": version,
            "claims": claims,
            "cited_source_indices": [citation_index],
        }

    def test_claim_map_binds_raw_claim_to_current_evidence_version(self):
        artifact = self._artifact()
        claim = artifact["claims"][0]
        evidence = artifact["sources"][0]

        self.assertEqual(claim["section_id"], "section:1")
        self.assertTrue(claim["claim_id"])
        self.assertIn("value is 20", claim["claim_text"])
        self.assertEqual(claim["artifact_version"], artifact["artifact_version"])
        self.assertEqual(
            claim["cited_sources"],
            [
                {
                    "index": 1,
                    "content_ref": evidence["content_ref"],
                    "content_hash": evidence["content_hash"],
                }
            ],
        )

    async def test_forged_source_index_fails_before_judge_invocation(self):
        model = SimpleNamespace(ainvoke=AsyncMock())
        verifier = SameModelVerifier(model)

        with self.assertRaises(JudgeError):
            await verifier.verify(
                topic="topic",
                artifact=self._artifact(citation_index=99),
                rubric=RESEARCH_RUBRIC,
            )

        model.ainvoke.assert_not_awaited()

    async def test_insufficient_claim_forces_patch_even_when_rubric_passes(self):
        artifact = self._artifact()
        claim = artifact["claims"][0]
        verdict = {
            "claim_id": claim["claim_id"],
            "status": "insufficient",
            "evidence_refs": [claim["cited_sources"][0]["content_ref"]],
            "reason": "The evidence says 10, not 20.",
        }
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps(
                        {
                            "raw_scores": _scores(),
                            "feedback": _feedback([verdict]),
                        }
                    )
                )
            )
        )
        score = await SameModelVerifier(model).verify(
            topic="topic", artifact=artifact, rubric=RESEARCH_RUBRIC
        )

        decision, executor = Policy().decide(
            score=score,
            rubric=RESEARCH_RUBRIC,
            iteration_no=1,
            max_iterations=2,
        )

        self.assertEqual(score.feedback["claim_verdicts"][0]["status"], "insufficient")
        self.assertEqual(decision, DECISION_RETRY_PATCH)
        self.assertEqual(executor.kind, "patch")
        judge_prompt = model.ainvoke.await_args.args[0][1]["content"]
        self.assertIn(claim["claim_id"], judge_prompt)
        self.assertIn("The audited value is 10", judge_prompt)

    async def test_stale_claim_ref_or_evidence_hash_fails_before_judge(self):
        for mutation in ("claim_ref", "source_hash"):
            with self.subTest(mutation=mutation):
                artifact = self._artifact()
                if mutation == "claim_ref":
                    artifact["claims"][0]["cited_sources"][0]["content_ref"] = "evidence:stale"
                else:
                    artifact["sources"][0]["content_hash"] = "0" * 64
                model = SimpleNamespace(ainvoke=AsyncMock())

                with self.assertRaises(JudgeError):
                    await SameModelVerifier(model).verify(
                        topic="topic", artifact=artifact, rubric=RESEARCH_RUBRIC
                    )

                model.ainvoke.assert_not_awaited()

    async def test_invalid_claim_verdict_schema_id_and_evidence_ref_are_judge_errors(self):
        artifact = self._artifact()
        claim = artifact["claims"][0]
        valid = {
            "claim_id": claim["claim_id"],
            "status": "supported",
            "evidence_refs": [claim["cited_sources"][0]["content_ref"]],
            "reason": "Evidence directly states the value.",
        }
        invalid_verdicts = {
            "unknown_claim": [{**valid, "claim_id": "unknown"}],
            "unknown_ref": [{**valid, "evidence_refs": ["evidence:unknown"]}],
            "bad_status": [{**valid, "status": "maybe"}],
            "missing_reason": [{key: value for key, value in valid.items() if key != "reason"}],
            "duplicate": [valid, valid],
            "missing_claim": [],
        }

        for name, verdicts in invalid_verdicts.items():
            with self.subTest(name=name):
                model = SimpleNamespace(
                    ainvoke=AsyncMock(
                        return_value=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "raw_scores": _scores(),
                                    "feedback": _feedback(verdicts),
                                }
                            )
                        )
                    )
                )
                with self.assertRaises(JudgeError):
                    await SameModelVerifier(model).verify(
                        topic="topic", artifact=artifact, rubric=RESEARCH_RUBRIC
                    )

    async def test_supported_verdict_cannot_borrow_another_claims_evidence(self):
        sources = [
            Source(1, SOURCE_WEB, "Claim one", "Evidence for claim one.", "https://one"),
            Source(2, SOURCE_WEB, "Claim two", "Evidence for claim two.", "https://two"),
        ]
        version, claims = build_claim_contract(
            {},
            [
                ("One", "Claim one is true [来源 1]."),
                ("Two", "Claim two is true [来源 2]."),
            ],
            sources,
        )
        artifact = {
            "title": "Report",
            "markdown": "report",
            "sources": [source.as_verifier_evidence() for source in sources],
            "headings": ["One", "Two"],
            "artifact_version": version,
            "claims": claims,
            "cited_source_indices": [1, 2],
        }
        verdicts = [
            {
                "claim_id": claims[0]["claim_id"],
                "status": "supported",
                "evidence_refs": [sources[1].content_ref],
                "reason": "Borrowed from claim two.",
            },
            {
                "claim_id": claims[1]["claim_id"],
                "status": "supported",
                "evidence_refs": [sources[1].content_ref],
                "reason": "Directly supported.",
            },
        ]
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps({"raw_scores": _scores(), "feedback": _feedback(verdicts)})
                )
            )
        )

        with self.assertRaises(JudgeError):
            await SameModelVerifier(model).verify(
                topic="topic", artifact=artifact, rubric=RESEARCH_RUBRIC
            )

    async def test_omitted_evidence_cannot_form_supported_verdict(self):
        sources = [
            Source(
                index,
                SOURCE_WEB,
                f"Evidence {index} " + "T" * 200,
                f"Fact {index}.",
                f"https://example.com/{index}/" + "u" * 700,
            )
            for index in range(1, 21)
        ]
        citations = " ".join(f"[来源 {index}]" for index in range(1, 21))
        version, claims = build_claim_contract(
            {}, [("Results", f"The combined result is 20 {citations}.")], sources
        )
        artifact = {
            "title": "Report",
            "markdown": "report",
            "sources": [source.as_verifier_evidence() for source in sources],
            "headings": ["Results"],
            "artifact_version": version,
            "claims": claims,
            "cited_source_indices": list(range(1, 21)),
        }
        verdict = {
            "claim_id": claims[0]["claim_id"],
            "status": "supported",
            "evidence_refs": [sources[0].content_ref],
            "reason": "Claims support from omitted evidence.",
        }
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps({"raw_scores": _scores(), "feedback": _feedback([verdict])})
                )
            )
        )

        with self.assertRaises(JudgeError):
            await SameModelVerifier(model).verify(
                topic="topic", artifact=artifact, rubric=RESEARCH_RUBRIC
            )

        prompt = model.ainvoke.await_args.args[0][1]["content"]
        self.assertIn("[来源 1] omitted: true", prompt)

    async def test_uncited_factual_claim_cannot_pass_with_empty_claim_map(self):
        source = Source(1, SOURCE_WEB, "Evidence", "The audited value is 10.", "https://one")
        version, claims = build_claim_contract(
            {}, [("Results", "The audited value is 20.")], [source]
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["cited_sources"], [])
        artifact = {
            "title": "Report",
            "markdown": "report",
            "sources": [source.as_verifier_evidence()],
            "headings": ["Results"],
            "artifact_version": version,
            "claims": claims,
            "cited_source_indices": [],
        }
        verdict = {
            "claim_id": claims[0]["claim_id"],
            "status": "insufficient",
            "evidence_refs": [],
            "reason": "The claim has no cited evidence.",
        }
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps({"raw_scores": _scores(), "feedback": _feedback([verdict])})
                )
            )
        )

        score = await SameModelVerifier(model).verify(
            topic="topic", artifact=artifact, rubric=RESEARCH_RUBRIC
        )
        decision, _ = Policy().decide(
            score=score,
            rubric=RESEARCH_RUBRIC,
            iteration_no=1,
            max_iterations=2,
        )

        self.assertEqual(decision, DECISION_RETRY_PATCH)

    def test_uncited_candidate_rule_keeps_facts_and_excludes_opinions(self):
        _, claims = build_claim_contract(
            {},
            [
                (
                    "Assessment",
                    "2026年收入增长42%。公司成立于2010年。"
                    "建议是优先控制成本。我的判断是暂缓扩张。",
                )
            ],
            [],
        )

        claim_texts = [claim["claim_text"] for claim in claims]
        self.assertEqual(
            claim_texts,
            ["2026年收入增长42%。", "公司成立于2010年。"],
        )

    def test_markdown_prefixes_do_not_change_factual_candidate_semantics(self):
        _, claims = build_claim_contract(
            {},
            [
                (
                    "Assessment",
                    "1. 建议是优先控制成本。\n"
                    "2) 我的判断是暂缓扩张。\n"
                    "- 公司成立于2010年。\n"
                    "1. 2026年收入增长42%。",
                )
            ],
            [],
        )

        self.assertEqual(
            [claim["claim_text"] for claim in claims],
            ["- 公司成立于2010年。", "1. 2026年收入增长42%。"],
        )


class ClaimReplacementDecisionTests(unittest.IsolatedAsyncioTestCase):
    _artifact = ClaimContractTests._artifact

    async def test_related_but_non_answering_learning_can_be_rejected_as_insufficient(self):
        source = Source(
            3,
            SOURCE_WEB,
            "Company profile",
            "Revenue is not reported. The company was founded in 2010.",
            "https://example.com/profile",
        )
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps(
                        {
                            "action": "insufficient",
                            "replacement": "",
                            "source_index": None,
                        }
                    )
                )
            )
        )

        decision = await decide_claim_replacement(
            model,
            claim_text="The company's 2026 revenue was $20 million.",
            reason="The cited source does not report revenue.",
            sources=[source],
            learnings=[Learning(text="The company was founded in 2010.", source_index=3)],
        )

        self.assertEqual(decision.action, ACTION_INSUFFICIENT)
        self.assertEqual(decision.replacement, "")
        prompt = model.ainvoke.await_args.args[0]
        self.assertIn("不能回答原断言", prompt)
        self.assertIn("founded in 2010", prompt)

    async def test_invalid_or_foreign_source_decision_falls_back_to_insufficient(self):
        source = Source(3, SOURCE_WEB, "Revenue", "Revenue was 10.", "https://example.com")
        invalid_outputs = [
            "not-json",
            json.dumps(
                {
                    "action": "replace",
                    "replacement": "Revenue was 10.",
                    "source_index": 99,
                }
            ),
            json.dumps(
                {
                    "action": "replace",
                    "replacement": "Revenue was 10 [来源 3].",
                    "source_index": 3,
                }
            ),
        ]

        for output in invalid_outputs:
            with self.subTest(output=output):
                model = SimpleNamespace(
                    ainvoke=AsyncMock(return_value=SimpleNamespace(content=output))
                )
                decision = await decide_claim_replacement(
                    model,
                    claim_text="Revenue was 20.",
                    reason="Incorrect value.",
                    sources=[source],
                    learnings=[Learning(text="Revenue was 10.", source_index=3)],
                )
                self.assertEqual(decision.action, ACTION_INSUFFICIENT)

    async def test_valid_replacement_must_use_target_source(self):
        source = Source(3, SOURCE_WEB, "Revenue", "Revenue was 10.", "https://example.com")
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps(
                        {
                            "action": "replace",
                            "replacement": "Revenue was 10.",
                            "source_index": 3,
                        }
                    )
                )
            )
        )

        decision = await decide_claim_replacement(
            model,
            claim_text="Revenue was 20.",
            reason="Incorrect value.",
            sources=[source],
            learnings=[Learning(text="Revenue was 10.", source_index=3)],
        )

        self.assertEqual(decision.action, ACTION_REPLACE)
        self.assertEqual(decision.source_index, 3)

    async def test_summarize_prompt_requires_factual_citations(self):
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps(
                        {
                            "tldr": "Revenue was 10 [来源 2].",
                            "key_points": ["Revenue was corrected [来源 2]."],
                        }
                    )
                )
            )
        )

        result = await summarize(model, "Report", "Revenue was 10 [来源 2].")

        self.assertEqual(result["tldr"], "Revenue was 10 [来源 2].")
        prompt = model.ainvoke.await_args.args[0]
        self.assertIn("保留正文已有的 `[来源 N]`", prompt)

    def test_unresolved_claim_at_iteration_limit_cannot_pass(self):
        artifact = self._artifact()
        claim = artifact["claims"][0]
        score = VerifyScore(
            raw_scores=_scores(),
            total=0.8,
            feedback=_feedback(
                [
                    {
                        "claim_id": claim["claim_id"],
                        "status": "contradicted",
                        "evidence_refs": [claim["cited_sources"][0]["content_ref"]],
                        "reason": "Contradicted by evidence.",
                    }
                ]
            ),
        )

        decision, executor = Policy().decide(
            score=score,
            rubric=RESEARCH_RUBRIC,
            iteration_no=2,
            max_iterations=2,
        )

        self.assertEqual(decision, DECISION_EXCEED)
        self.assertIsNone(executor)

    def test_supported_claim_with_passing_rubric_can_pass(self):
        artifact = self._artifact()
        claim = artifact["claims"][0]
        score = VerifyScore(
            raw_scores=_scores(),
            total=0.8,
            feedback=_feedback(
                [
                    {
                        "claim_id": claim["claim_id"],
                        "status": "supported",
                        "evidence_refs": [claim["cited_sources"][0]["content_ref"]],
                        "reason": "Directly supported.",
                    }
                ]
            ),
        )

        decision, executor = Policy().decide(
            score=score,
            rubric=RESEARCH_RUBRIC,
            iteration_no=1,
            max_iterations=2,
        )

        self.assertEqual(decision, DECISION_PASS)
        self.assertIsNone(executor)

    async def test_invalid_claim_judge_output_becomes_judge_error_without_repair(self):
        artifact = self._artifact()
        model = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps(
                        {
                            "raw_scores": _scores(),
                            "feedback": _feedback(
                                [
                                    {
                                        "claim_id": "fabricated",
                                        "status": "supported",
                                        "evidence_refs": [artifact["sources"][0]["content_ref"]],
                                        "reason": "invalid id",
                                    }
                                ]
                            ),
                        }
                    )
                )
            )
        )

        class _Store:
            def __init__(self):
                self.iterations = []
                self.finished = None

            async def create_run(self, **_kwargs):
                return SimpleNamespace(id=uuid.uuid4())

            async def record_iteration(self, _run_id, outcome):
                self.iterations.append(outcome)

            async def finish_run(self, _run_id, **kwargs):
                self.finished = kwargs

        controller = LoopController(
            session=object(),
            user_id=uuid.uuid4(),
            task_type="research",
            max_iterations=2,
        )
        store = _Store()
        controller.store = store
        patch_callback = AsyncMock()
        with patch(
            "app.core.agent.loop.controller.build_verifier",
            new_callable=AsyncMock,
            return_value=SameModelVerifier(model),
        ):
            events = [
                event
                async for event in controller.run(
                    topic="topic",
                    initial_artifact=artifact,
                    verifier_kind="same",
                    generator_model=object(),
                    repair_ctx=RepairCallbackArgs(patch_callback=patch_callback),
                )
            ]

        self.assertEqual(events[-1]["quality_status"], QUALITY_JUDGE_ERROR)
        self.assertEqual(store.finished["quality_status"], QUALITY_JUDGE_ERROR)
        self.assertFalse(any(event["type"].startswith("loop_repair") for event in events))
        patch_callback.assert_not_awaited()

    async def test_controller_marks_unresolved_round_failed_then_verifies_new_artifact(self):
        initial = self._artifact()
        replacement_source = Source(
            2,
            SOURCE_WEB,
            "Replacement",
            "The audited value is 10.",
            "https://example.com/replacement",
        )
        replacement_version, replacement_claims = build_claim_contract(
            {"tldr": ""},
            [("Results", "The audited value is 10 [来源 2].")],
            [replacement_source],
        )
        replacement = {
            **initial,
            "artifact_version": replacement_version,
            "claims": replacement_claims,
            "sources": [replacement_source.as_verifier_evidence()],
        }
        old_claim = initial["claims"][0]
        new_claim = replacement["claims"][0]
        scores = [
            VerifyScore(
                raw_scores=_scores(),
                total=0.8,
                feedback=_feedback(
                    [
                        {
                            "claim_id": old_claim["claim_id"],
                            "status": "insufficient",
                            "evidence_refs": [old_claim["cited_sources"][0]["content_ref"]],
                            "reason": "The source does not support 20.",
                        }
                    ]
                ),
            ),
            VerifyScore(
                raw_scores=_scores(),
                total=0.8,
                feedback=_feedback(
                    [
                        {
                            "claim_id": new_claim["claim_id"],
                            "status": "supported",
                            "evidence_refs": [new_claim["cited_sources"][0]["content_ref"]],
                            "reason": "The replacement evidence supports 10.",
                        }
                    ]
                ),
            ),
        ]

        class _Verifier:
            kind = "same"
            model_name = "fixture"

            def __init__(self):
                self.artifacts = []

            async def verify(self, *, topic, artifact, rubric):  # noqa: ARG002
                self.artifacts.append(artifact)
                return scores[len(self.artifacts) - 1]

        class _Store:
            def __init__(self):
                self.iterations = []
                self.finished = None

            async def create_run(self, **_kwargs):
                return SimpleNamespace(id=uuid.uuid4())

            async def record_iteration(self, _run_id, outcome):
                self.iterations.append(outcome)

            async def finish_run(self, _run_id, **kwargs):
                self.finished = kwargs

        verifier = _Verifier()
        store = _Store()
        controller = LoopController(
            session=object(),
            user_id=uuid.uuid4(),
            task_type="research",
            max_iterations=2,
        )
        controller.store = store
        patch_callback = AsyncMock(return_value=replacement)
        with patch(
            "app.core.agent.loop.controller.build_verifier",
            new_callable=AsyncMock,
            return_value=verifier,
        ):
            events = [
                event
                async for event in controller.run(
                    topic="topic",
                    initial_artifact=initial,
                    verifier_kind="same",
                    generator_model=object(),
                    repair_ctx=RepairCallbackArgs(patch_callback=patch_callback),
                )
            ]

        verify_events = [event for event in events if event["type"] == "loop_verify_done"]
        self.assertEqual(verify_events[0]["quality_status"], QUALITY_FAILED)
        self.assertEqual(verify_events[1]["quality_status"], QUALITY_PASSED)
        self.assertEqual(events[-1]["quality_status"], QUALITY_PASSED)
        self.assertEqual(store.finished["quality_status"], QUALITY_PASSED)
        self.assertEqual(len(store.iterations), 2)
        self.assertEqual(verifier.artifacts, [initial, replacement])
        patch_callback.assert_awaited_once()


class TargetedPatchContractTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _artifact() -> dict:
        source = Source(1, SOURCE_WEB, "Evidence", "value is 10", "https://example.com/e")
        version, claims = build_claim_contract(
            {}, [("Results", "The audited value is 20 [来源 1].")], [source]
        )
        return {
            "artifact_version": version,
            "claims": claims,
            "sources": [source.as_verifier_evidence()],
        }

    def test_patch_plan_targets_claim_and_queries_its_text_and_reason(self):
        artifact = self._artifact()
        claim = artifact["claims"][0]
        score = VerifyScore(
            raw_scores=_scores(),
            total=0.8,
            feedback=_feedback(
                [
                    {
                        "claim_id": claim["claim_id"],
                        "status": "contradicted",
                        "evidence_refs": [claim["cited_sources"][0]["content_ref"]],
                        "reason": "The evidence states 10.",
                    }
                ]
            ),
        )

        action = PatchRepair().plan(score=score, artifact=artifact)

        self.assertEqual(action.artifact_version, artifact["artifact_version"])
        self.assertEqual(action.target_claims[0].claim_id, claim["claim_id"])
        self.assertEqual(action.target_claims[0].claim_text, claim["claim_text"])
        self.assertEqual(action.target_claims[0].locator_start, claim["locator_start"])
        self.assertEqual(action.target_claims[0].locator_end, claim["locator_end"])
        self.assertEqual(action.target_claims[0].repair_query, action.patch_queries[0])
        self.assertIn("audited value is 20", action.patch_queries[0])
        self.assertIn("evidence states 10", action.patch_queries[0])
        self.assertNotIn("核对来源", action.patch_queries[0])

    async def test_stale_artifact_version_does_not_invoke_patch_callback(self):
        callback = AsyncMock()
        artifact = self._artifact()
        action = RepairAction(
            kind="patch",
            artifact_version="stale-version",
            patch_queries=["query"],
        )

        result = await PatchRepair().execute(
            action=action,
            artifact=artifact,
            ctx={"patch_callback": callback},
        )

        self.assertIs(result, artifact)
        callback.assert_not_awaited()

    def test_targeted_replacement_removes_old_claim_and_wrong_citation(self):
        summary = {"tldr": "Summary", "key_points": []}
        claim_text = "The audited value is 20 [来源 1]."
        sections = [("Results", f"{claim_text} Other text.")]
        target = ClaimRepairTarget(
            claim_id="claim-1",
            section_id="section:1",
            claim_text=claim_text,
            locator_start=0,
            locator_end=len(claim_text),
            repair_query="audited value",
            status="contradicted",
            evidence_refs=["evidence:old"],
            reason="The evidence says 10.",
        )

        new_summary, new_sections, changed, body_changed = apply_claim_replacements(
            summary,
            sections,
            [target],
            {"claim-1": "The audited value is 10 [来源 2]."},
        )

        self.assertTrue(changed)
        self.assertTrue(body_changed)
        self.assertEqual(new_summary, summary)
        self.assertNotIn("value is 20", new_sections[0][1])
        self.assertNotIn("[来源 1]", new_sections[0][1])
        self.assertIn("value is 10 [来源 2]", new_sections[0][1])

    def test_targeted_replacement_downgrades_when_new_evidence_is_insufficient(self):
        claim_text = "Unverified claim [来源 1]."
        target = ClaimRepairTarget(
            claim_id="claim-1",
            section_id="section:1",
            claim_text=claim_text,
            locator_start=0,
            locator_end=len(claim_text),
            repair_query="unverified claim",
            status="insufficient",
            evidence_refs=["evidence:old"],
            reason="No supporting evidence.",
        )

        _, sections, changed, body_changed = apply_claim_replacements(
            {},
            [("Results", claim_text)],
            [target],
            {"claim-1": "现有证据不足，无法确认。"},
        )

        self.assertTrue(changed)
        self.assertTrue(body_changed)
        self.assertEqual(sections[0][1], "现有证据不足，无法确认。")
        self.assertNotIn("[来源 1]", sections[0][1])

    def test_duplicate_claim_text_repairs_only_the_targeted_locator(self):
        source = Source(1, SOURCE_WEB, "Evidence", "Value is 10.", "https://one")
        repeated = "  Value is 20 [来源 1]. Value is 20 [来源 1]."
        _, claims = build_claim_contract({}, [("Results", repeated)], [source])
        second = claims[1]
        target = ClaimRepairTarget(
            claim_id=second["claim_id"],
            section_id=second["section_id"],
            claim_text=second["claim_text"],
            locator_start=second["locator_start"],
            locator_end=second["locator_end"],
            repair_query="second value",
            status="contradicted",
            evidence_refs=[source.content_ref],
            reason="Only the second occurrence is wrong.",
        )

        _, sections, changed, body_changed = apply_claim_replacements(
            {},
            [("Results", repeated)],
            [target],
            {target.claim_id: "Value is 10 [来源 1]."},
        )

        self.assertTrue(changed)
        self.assertTrue(body_changed)
        self.assertEqual(
            sections[0][1],
            "  Value is 20 [来源 1]. Value is 10 [来源 1].",
        )

    def test_summary_claim_uses_its_locator_without_changing_body(self):
        source = Source(1, SOURCE_WEB, "Evidence", "Value is 10.", "https://one")
        summary = {"tldr": "Value is 20 [来源 1].", "key_points": []}
        _, claims = build_claim_contract(summary, [("Results", "Narrative only.")], [source])
        claim = claims[0]
        target = ClaimRepairTarget(
            claim_id=claim["claim_id"],
            section_id=claim["section_id"],
            claim_text=claim["claim_text"],
            locator_start=claim["locator_start"],
            locator_end=claim["locator_end"],
            repair_query="summary value",
            status="contradicted",
            evidence_refs=[source.content_ref],
            reason="The value is 10.",
        )

        updated_summary, sections, changed, body_changed = apply_claim_replacements(
            summary,
            [("Results", "Narrative only.")],
            [target],
            {target.claim_id: "Value is 10 [来源 1]."},
        )

        self.assertTrue(changed)
        self.assertFalse(body_changed)
        self.assertEqual(updated_summary["tldr"], "Value is 10 [来源 1].")
        self.assertEqual(sections, [("Results", "Narrative only.")])


class ResearchPatchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        repair_source: Source | None,
        repair_learnings: list[Learning],
        target_section_ids: list[str] | None = None,
        initial_sources: list[Source] | None = None,
        written_contents: list[str] | None = None,
        repair_batches: list[list[Source]] | None = None,
        distill_batches: list[list[Learning]] | None = None,
        summarize_results: list[dict] | None = None,
        replacement_decider=None,
        repair_mode: str = "targeted",
        rewrite_content: str | None = None,
    ):
        initial_sources = initial_sources or [
            Source(
                0,
                SOURCE_WEB,
                "Old",
                "The evidence states 10, not 20.",
                "https://example.com/old",
            )
        ]
        written_contents = written_contents or ["The audited value is 20 [来源 1]."]
        target_section_ids = target_section_ids or ["section:1"]
        controller_artifacts: list[dict] = []

        class _Controller:
            def __init__(self, **_kwargs):
                pass

            async def run(self, **kwargs):
                artifact = kwargs["initial_artifact"]
                controller_artifacts.append(artifact)
                if repair_mode == "targeted":
                    claims = [
                        next(
                            item for item in artifact["claims"] if item["section_id"] == section_id
                        )
                        for section_id in target_section_ids
                    ]
                    score = VerifyScore(
                        raw_scores=_scores(),
                        total=0.8,
                        feedback=_feedback(
                            [
                                {
                                    "claim_id": claim["claim_id"],
                                    "status": "contradicted",
                                    "evidence_refs": [claim["cited_sources"][0]["content_ref"]],
                                    "reason": "The source says 10.",
                                }
                                for claim in claims
                            ]
                        ),
                    )
                    action = PatchRepair().plan(score=score, artifact=artifact)
                    updated = await PatchRepair().execute(
                        action=action,
                        artifact=artifact,
                        ctx={"patch_callback": kwargs["repair_ctx"].patch_callback},
                    )
                elif repair_mode == "coverage":
                    updated = await kwargs["repair_ctx"].patch_callback(["coverage gap"])
                elif repair_mode == "rewrite":
                    updated = await kwargs["repair_ctx"].rewrite_callback(["Results 1"])
                else:
                    raise AssertionError(f"unknown repair mode: {repair_mode}")
                controller_artifacts.append(updated)
                yield {
                    "type": "loop_finished",
                    "status": "passed",
                    "quality_status": "passed",
                    "final_artifact": updated,
                }

        section_contents = {
            f"Results {index}": content for index, content in enumerate(written_contents, start=1)
        }

        write_counts: dict[str, int] = {}

        async def write_section(*args, **_kwargs):
            heading = args[2]
            count = write_counts.get(heading, 0)
            write_counts[heading] = count + 1
            if count and rewrite_content is not None:
                yield rewrite_content
            else:
                yield section_contents[heading]

        if repair_batches is None:
            repair_batches = [[repair_source] if repair_source else []]
        if distill_batches is None:
            distill_batches = [repair_learnings] if repair_source else []
        kb_results = [initial_sources, *repair_batches]
        distill_results = [[Learning(text="old", source_index=1)], *distill_batches]
        summarize_results = summarize_results or [
            {"tldr": "Old summary [来源 1].", "key_points": []},
            {"tldr": "Regenerated summary", "key_points": []},
        ]

        async def default_replacement_decider(
            _model,
            *,
            claim_text,
            reason,
            sources,
            learnings,  # noqa: ARG001
        ):
            source_indices = {source.index for source in sources}
            learning = next(
                (item for item in learnings if item.source_index in source_indices),
                None,
            )
            if learning is None:
                return ClaimReplacementDecision(action=ACTION_INSUFFICIENT)
            return ClaimReplacementDecision(
                action=ACTION_REPLACE,
                replacement=learning.text,
                source_index=learning.source_index,
            )

        replacement_decider = replacement_decider or default_replacement_decider
        plan = SimpleNamespace(
            title="Report",
            sections=[SimpleNamespace(heading=heading, points=[]) for heading in section_contents],
            queries=["initial"],
        )
        curated = [
            SimpleNamespace(heading=heading, thesis="", learning_ids=[1])
            for heading in section_contents
        ]
        with (
            patch("app.core.agent.research.engine.settings.research_reflection_rounds", 0),
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
                side_effect=kb_results,
            ),
            patch(
                "app.core.agent.research.engine.gather_mcp_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "app.core.agent.research.engine.distill_sources",
                new_callable=AsyncMock,
                side_effect=distill_results,
            ),
            patch(
                "app.core.agent.research.engine.curate_outline",
                new_callable=AsyncMock,
                return_value=curated,
            ),
            patch("app.core.agent.research.engine.write_section_stream", new=write_section),
            patch(
                "app.core.agent.research.engine.summarize",
                new_callable=AsyncMock,
                side_effect=summarize_results,
            ) as summarize_mock,
            patch(
                "app.core.agent.research.engine.decide_claim_replacement",
                new_callable=AsyncMock,
                side_effect=replacement_decider,
            ),
            patch("app.core.agent.research.engine.LoopController", _Controller),
        ):
            events = [
                event
                async for event in run_research(
                    object(), uuid.uuid4(), "topic", report_id=uuid.uuid4()
                )
            ]
        report = next(event for event in events if event["type"] == "report")
        return controller_artifacts, report, summarize_mock

    async def test_patch_replaces_wrong_claim_and_next_verifier_gets_new_artifact(self):
        new_source = Source(
            0,
            SOURCE_WEB,
            "New",
            "The audited value is 10.",
            "https://example.com/new",
        )
        artifacts, report, summarize_mock = await self._run(
            repair_source=new_source,
            repair_learnings=[Learning(text="The audited value is 10.", source_index=2)],
        )

        self.assertEqual(len(artifacts), 2)
        self.assertNotEqual(artifacts[0]["artifact_version"], artifacts[1]["artifact_version"])
        self.assertNotIn("value is 20", artifacts[1]["markdown"])
        self.assertIn("value is 10", artifacts[1]["markdown"])
        self.assertIn("Regenerated summary", artifacts[1]["markdown"])
        self.assertNotIn("value is 20", report["markdown"])
        self.assertEqual(summarize_mock.await_count, 2)

    async def test_patch_without_new_evidence_downgrades_claim_to_uncertainty(self):
        artifacts, report, summarize_mock = await self._run(
            repair_source=None,
            repair_learnings=[],
        )

        self.assertIn("现有证据不足，无法确认", artifacts[1]["markdown"])
        self.assertNotIn("value is 20", artifacts[1]["markdown"])
        self.assertIn("现有证据不足，无法确认", report["markdown"])
        self.assertEqual(summarize_mock.await_count, 2)

    async def test_summary_target_is_applied_without_regenerating_unchanged_body(self):
        new_source = Source(
            0,
            SOURCE_WEB,
            "New",
            "The audited value is 10.",
            "https://example.com/new",
        )
        artifacts, report, summarize_mock = await self._run(
            repair_source=new_source,
            repair_learnings=[Learning(text="Corrected summary", source_index=2)],
            target_section_ids=["summary:tldr"],
        )

        self.assertIn("Corrected summary", artifacts[1]["markdown"])
        self.assertIn("New", artifacts[1]["markdown"])
        self.assertIn("Corrected summary", report["markdown"])
        self.assertEqual(summarize_mock.await_count, 1)

    async def test_each_target_uses_only_learning_from_its_own_retrieval(self):
        initial_sources = [
            Source(0, SOURCE_WEB, "Old A", "Old A evidence.", "https://old/a"),
            Source(0, SOURCE_WEB, "Old B", "Old B evidence.", "https://old/b"),
        ]
        source_a = Source(0, SOURCE_WEB, "New A", "Correct A.", "https://new/a")
        source_b = Source(0, SOURCE_WEB, "New B", "Correct B.", "https://new/b")

        artifacts, report, _ = await self._run(
            repair_source=None,
            repair_learnings=[],
            target_section_ids=["section:1", "section:2"],
            initial_sources=initial_sources,
            written_contents=[
                "Claim A is wrong [来源 1].",
                "Claim B is wrong [来源 2].",
            ],
            repair_batches=[[source_a], [source_b]],
            distill_batches=[
                [
                    Learning(text="Wrong B replacement", source_index=4, relevance=1.0),
                    Learning(text="Correct A replacement", source_index=3, relevance=0.5),
                ],
                [
                    Learning(text="Wrong A replacement", source_index=3, relevance=1.0),
                    Learning(text="Correct B replacement", source_index=4, relevance=0.5),
                ],
            ],
        )

        self.assertIn("Correct A replacement", artifacts[1]["markdown"])
        self.assertIn("Correct B replacement", artifacts[1]["markdown"])
        self.assertIn("New A", artifacts[1]["markdown"])
        self.assertIn("New B", artifacts[1]["markdown"])
        self.assertNotIn("Wrong A replacement", artifacts[1]["markdown"])
        self.assertNotIn("Wrong B replacement", artifacts[1]["markdown"])
        self.assertNotIn("Claim A is wrong", report["markdown"])
        self.assertNotIn("Claim B is wrong", report["markdown"])

    async def test_nonanswering_target_learning_downgrades_instead_of_replacing_claim(self):
        source = Source(
            0,
            SOURCE_WEB,
            "Company profile",
            "Revenue is unavailable. The company was founded in 2010.",
            "https://example.com/profile",
        )

        async def reject_nonanswering(*_args, **_kwargs):
            return ClaimReplacementDecision(action=ACTION_INSUFFICIENT)

        artifacts, _, _ = await self._run(
            repair_source=source,
            repair_learnings=[Learning(text="The company was founded in 2010.", source_index=2)],
            replacement_decider=reject_nonanswering,
        )

        self.assertIn("现有证据不足，无法确认", artifacts[1]["markdown"])
        self.assertNotIn("founded in 2010", artifacts[1]["markdown"])

    async def test_body_and_summary_targets_rebuild_cited_summary_from_new_body(self):
        new_source = Source(
            0,
            SOURCE_WEB,
            "Corrected evidence",
            "The audited value is 10.",
            "https://example.com/corrected",
        )
        artifacts, report, summarize_mock = await self._run(
            repair_source=None,
            repair_learnings=[],
            target_section_ids=["summary:tldr", "section:1"],
            repair_batches=[[new_source], [new_source]],
            distill_batches=[
                [Learning(text="The audited value is 10.", source_index=2)],
                [Learning(text="The audited value is 10.", source_index=2)],
            ],
            summarize_results=[
                {"tldr": "Old summary [来源 1].", "key_points": []},
                {
                    "tldr": "The audited value is 10 [来源 2].",
                    "key_points": ["The corrected value is 10 [来源 2]."],
                },
            ],
        )

        final_artifact = artifacts[1]
        summary_claims = [
            claim
            for claim in final_artifact["claims"]
            if claim["section_id"].startswith("summary:")
        ]
        self.assertEqual(summarize_mock.await_count, 2)
        self.assertTrue(summary_claims)
        self.assertTrue(all(claim["cited_sources"][0]["index"] == 2 for claim in summary_claims))
        self.assertNotIn("Old summary", final_artifact["markdown"])
        self.assertNotIn("value is 20", final_artifact["markdown"])
        self.assertIn("audited value is 10", report["markdown"])

    async def test_legacy_coverage_patch_refreshes_summary_after_body_change(self):
        new_source = Source(
            0,
            SOURCE_WEB,
            "Coverage evidence",
            "The missing market segment grew by 15%.",
            "https://example.com/coverage",
        )
        artifacts, report, summarize_mock = await self._run(
            repair_source=new_source,
            repair_learnings=[
                Learning(text="The missing market segment grew by 15%.", source_index=2)
            ],
            repair_mode="coverage",
            summarize_results=[
                {"tldr": "Old summary [来源 1].", "key_points": []},
                {
                    "tldr": "The market segment grew by 15% [来源 2].",
                    "key_points": [],
                },
            ],
        )

        self.assertEqual(summarize_mock.await_count, 2)
        self.assertIn("补充信息", artifacts[1]["markdown"])
        self.assertIn("market segment grew by 15%", artifacts[1]["markdown"])
        self.assertIn("market segment grew by 15%", report["markdown"])

    async def test_chapter_rewrite_refreshes_summary_after_body_change(self):
        artifacts, report, summarize_mock = await self._run(
            repair_source=None,
            repair_learnings=[],
            repair_mode="rewrite",
            rewrite_content="The rewritten audited value is 10 [来源 1].",
            summarize_results=[
                {"tldr": "Old summary [来源 1].", "key_points": []},
                {
                    "tldr": "The rewritten audited value is 10 [来源 1].",
                    "key_points": [],
                },
            ],
        )

        self.assertEqual(summarize_mock.await_count, 2)
        self.assertNotIn("value is 20", artifacts[1]["markdown"])
        self.assertIn("rewritten audited value is 10", artifacts[1]["markdown"])
        self.assertIn("rewritten audited value is 10", report["markdown"])

    async def test_unchanged_chapter_rewrite_does_not_refresh_summary(self):
        artifacts, _, summarize_mock = await self._run(
            repair_source=None,
            repair_learnings=[],
            repair_mode="rewrite",
        )

        self.assertEqual(summarize_mock.await_count, 1)
        self.assertEqual(artifacts[0]["artifact_version"], artifacts[1]["artifact_version"])


class ClaimPrivacyTests(unittest.IsolatedAsyncioTestCase):
    async def test_loop_detail_does_not_expose_claim_ids_or_evidence_refs(self):
        service = ResearchService(object())
        service._get_or_404 = AsyncMock()
        run = SimpleNamespace(
            id=uuid.uuid4(),
            task_type="research",
            status="failed",
            quality_status="failed_quality",
            iterations=1,
            final_score=0.6,
            pass_threshold=0.7,
            max_iterations=2,
            rubric_name="research",
            generator_model="generator",
            verifier_model="judge",
            verifier_kind="cross",
            note=None,
            started_at=None,
            finished_at=None,
        )
        iteration = SimpleNamespace(
            iteration_no=1,
            scores={"total": 0.6},
            feedback={
                "summary": "needs repair",
                "claim_verdicts": [
                    {
                        "claim_id": "private-claim-id",
                        "evidence_refs": ["private-content-ref"],
                    }
                ],
            },
            decision="retry_patch",
            repair_action={
                "kind": "patch",
                "artifact_version": "private-version",
                "target_claims": [{"claim_id": "private-claim-id"}],
                "patch_queries": ["query"],
            },
            duration_ms=1,
            artifact_snapshot={},
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
                return_value=[iteration],
            ),
        ):
            detail = await service.get_loop_detail(uuid.uuid4(), uuid.uuid4())

        self.assertNotIn("private-claim-id", str(detail))
        self.assertNotIn("private-content-ref", str(detail))
        self.assertNotIn("private-version", str(detail))
        self.assertEqual(detail["iterations_detail"][0]["feedback"]["summary"], "needs repair")
        self.assertEqual(
            detail["iterations_detail"][0]["repair_action"]["patch_queries"], ["query"]
        )


if __name__ == "__main__":
    unittest.main()

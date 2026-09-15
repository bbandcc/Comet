"""LLM-as-judge Verifier:同模型 self-critique 基线 + 独立配置 Judge 两套实现。

两套并存,跑 A/B 实验:
- SameModelVerifier (kind="same"): 同 chat 模型新开 session,critic 角色 prompt
- CrossModelVerifier (kind="cross"): 用独立配置的 verifier 模型

`cross` 只表示使用独立 verifier 配置，不推断或保证模型 family 不同。
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import uuid
from typing import Any

from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.agent.loop.models import RubricDef, VerifyScore
from app.core.agent.loop.verifier.base import (
    JudgeError,
    Verifier,
    VerifierUnavailableError,
)
from app.core.agent.loop.verifier.prompt_renderer import render_verifier_prompt
from app.core.exceptions import BizError
from app.core.llm.chat_model import build_chat_model
from app.core.logging import get_logger
from app.models.model_config_model import ModelConfig
from app.repositories.model_config_repository import ModelConfigRepository

logger = get_logger(__name__)

_JUDGE_EVIDENCE_TOTAL_CHARS = 12_000
_JUDGE_EVIDENCE_SOURCE_CHARS = 4_000


# ── 共用工具 ──


def _critic_role() -> str:
    return render_verifier_prompt("critic_role.jinja2")


def _evidence_metadata(source: dict[str, Any]) -> str:
    return (
        f"[来源 {source.get('index', '')}] {str(source.get('title') or '')[:200]}\n"
        f"url: {str(source.get('url') or '')[:500]}\n"
        f"stable_id: {str(source.get('stable_id') or '')[:128]}\n"
        f"origin_ref: {str(source.get('origin_ref') or '')[:300]}\n"
        f"content_ref: {str(source.get('content_ref') or '')[:300]}\n"
        f"fetch_status: {str(source.get('fetch_status') or '')[:32]}\n"
    )


def _render_source_block(source: dict[str, Any], budget: int) -> tuple[str | None, bool]:
    metadata = _evidence_metadata(source)
    fixed = metadata + "truncated: false\nevidence:\n"
    if budget <= len(fixed):
        return None, False
    content = str(source.get("content") or "")
    excerpt = content[: min(_JUDGE_EVIDENCE_SOURCE_CHARS, budget - len(fixed))]
    truncated = bool(source.get("truncated")) or len(excerpt) < len(content)
    block = metadata + f"truncated: {str(truncated).lower()}\nevidence:\n" + excerpt
    if len(block) > budget:
        return None, False
    return block, bool(excerpt)


def _render_evidence_context_with_refs(
    sources: list[dict[str, Any]], cited_indices: set[int] | None = None
) -> tuple[str, set[str]]:
    """渲染 Judge context，并返回本轮实际包含正文 excerpt 的 content refs。"""
    visible_refs: set[str] = set()

    def record_visible(source: dict[str, Any], has_excerpt: bool) -> None:
        content_ref = source.get("content_ref")
        if has_excerpt and isinstance(content_ref, str):
            visible_refs.add(content_ref)

    cited_indices = cited_indices or set()
    cited = [source for source in sources if source.get("index") in cited_indices]
    uncited = [source for source in sources if source.get("index") not in cited_indices]
    if cited:
        omitted = [f"[来源 {source.get('index', '')}] omitted: true" for source in uncited]
        omitted_text = "\n\n".join(omitted)
        cited_budget = _JUDGE_EVIDENCE_TOTAL_CHARS - len(omitted_text)
        if omitted_text:
            cited_budget -= 2
        blocks: list[str] = []
        remaining = cited_budget
        for position, source in enumerate(cited):
            separator_size = 2 if blocks else 0
            remaining -= separator_size
            source_budget = remaining // (len(cited) - position)
            block, has_excerpt = _render_source_block(source, source_budget)
            if block is None:
                block = f"[来源 {source.get('index', '')}] omitted: true"
            else:
                record_visible(source, has_excerpt)
            blocks.append(block)
            remaining -= len(block)
        if omitted_text:
            blocks.append(omitted_text)
        return "\n\n".join(blocks)[:_JUDGE_EVIDENCE_TOTAL_CHARS], visible_refs

    blocks: list[str] = []
    remaining = _JUDGE_EVIDENCE_TOTAL_CHARS
    for position, source in enumerate(uncited):
        separator = "\n\n" if blocks else ""
        remaining -= len(separator)
        if remaining <= 0:
            break
        block, has_excerpt = _render_source_block(source, remaining)
        if block is None:
            omitted = "\n\n".join(
                f"[来源 {item.get('index', '')}] omitted: true" for item in uncited[position:]
            )
            if len(omitted) <= remaining:
                blocks.append(omitted)
            break
        blocks.append(block)
        record_visible(source, has_excerpt)
        remaining -= len(block)
    return "\n\n".join(blocks), visible_refs


def _render_evidence_context(
    sources: list[dict[str, Any]], cited_indices: set[int] | None = None
) -> str:
    """兼容调用方：只返回有界 Judge evidence 文本。"""
    return _render_evidence_context_with_refs(sources, cited_indices)[0]


def _artifact_evidence_context(artifact: dict[str, Any]) -> tuple[str, set[str]]:
    structured_citations = artifact.get("cited_source_indices")
    if isinstance(structured_citations, list):
        cited_indices = {
            value
            for value in structured_citations
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
    else:
        cited_indices = {
            int(value) for value in re.findall(r"\[来源\s*(\d+)\]", artifact.get("markdown") or "")
        }
    return _render_evidence_context_with_refs(artifact.get("sources") or [], cited_indices)


def _render_research_prompt(*, topic: str, rubric: RubricDef, artifact: dict[str, Any]) -> str:
    """渲染研究报告的 verifier prompt(给单条 user message,system 走 critic_role)。"""
    evidence_context, _ = _artifact_evidence_context(artifact)
    return render_verifier_prompt(
        "verify_research.jinja2",
        topic=topic,
        rubric_max=int(rubric.raw_max),
        dims=[
            {"key": d.key, "label": d.label, "weight": d.weight, "desc": d.desc}
            for d in rubric.dims
        ],
        headings=artifact.get("headings") or [],
        artifact_markdown=(artifact.get("markdown") or "").strip(),
        claims=artifact.get("claims") or [],
        evidence_context=evidence_context,
    )


def _validate_claim_artifact(artifact: dict[str, Any]) -> None:
    if "claims" not in artifact:
        return
    claims = artifact.get("claims")
    version = artifact.get("artifact_version")
    sources = artifact.get("sources")
    if not isinstance(claims, list) or not isinstance(version, str) or not version:
        raise JudgeError("Verifier artifact claim contract 非法")
    if not isinstance(sources, list) or any(not isinstance(source, dict) for source in sources):
        raise JudgeError("Verifier artifact sources 非法")

    sources_by_index: dict[int, dict[str, Any]] = {}
    for source in sources:
        index = source.get("index")
        content = source.get("content")
        content_hash = source.get("content_hash")
        content_ref = source.get("content_ref")
        stable_id = source.get("stable_id")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(content, str)
            or not isinstance(content_hash, str)
            or not isinstance(content_ref, str)
            or not isinstance(stable_id, str)
        ):
            raise JudgeError("Verifier artifact evidence version 非法")
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_hash != content_hash or content_ref != f"evidence:{stable_id}:{content_hash}":
            raise JudgeError("Verifier artifact evidence hash/ref 不匹配")
        sources_by_index[index] = source

    seen_claim_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise JudgeError("Verifier artifact claim 必须是 object")
        claim_id = claim.get("claim_id")
        if (
            not isinstance(claim_id, str)
            or not claim_id
            or claim_id in seen_claim_ids
            or not isinstance(claim.get("section_id"), str)
            or not isinstance(claim.get("claim_text"), str)
            or isinstance(claim.get("locator_start"), bool)
            or not isinstance(claim.get("locator_start"), int)
            or claim["locator_start"] < 0
            or isinstance(claim.get("locator_end"), bool)
            or not isinstance(claim.get("locator_end"), int)
            or claim["locator_end"] <= claim["locator_start"]
            or claim.get("artifact_version") != version
            or not isinstance(claim.get("cited_sources"), list)
        ):
            raise JudgeError("Verifier artifact claim 定位契约非法")
        seen_claim_ids.add(claim_id)
        for cited in claim["cited_sources"]:
            if not isinstance(cited, dict):
                raise JudgeError("Verifier artifact cited source 必须是 object")
            source = sources_by_index.get(cited.get("index"))
            if source is None:
                raise JudgeError("断言引用号不属于当前 artifact")
            if (
                cited.get("content_ref") != source["content_ref"]
                or cited.get("content_hash") != source["content_hash"]
            ):
                raise JudgeError("断言 evidence ref/hash 不是当前版本")


def _validate_claim_verdicts(
    feedback: dict[str, Any],
    artifact: dict[str, Any],
    visible_evidence_refs: set[str],
) -> None:
    if "claims" not in artifact:
        return
    verdicts = feedback.get("claim_verdicts")
    if not isinstance(verdicts, list):
        raise JudgeError("Judge feedback.claim_verdicts 必须是 array")
    claims = {claim["claim_id"]: claim for claim in artifact.get("claims") or []}
    seen: set[str] = set()
    required_fields = {"claim_id", "status", "evidence_refs", "reason"}
    for index, verdict in enumerate(verdicts):
        if not isinstance(verdict, dict) or set(verdict) != required_fields:
            raise JudgeError(f"Judge claim_verdicts[{index}] schema 非法")
        claim_id = verdict.get("claim_id")
        status = verdict.get("status")
        evidence_refs = verdict.get("evidence_refs")
        reason = verdict.get("reason")
        if not isinstance(claim_id, str) or claim_id not in claims or claim_id in seen:
            raise JudgeError(f"Judge claim_verdicts[{index}].claim_id 非法")
        if status not in {"supported", "contradicted", "insufficient"}:
            raise JudgeError(f"Judge claim_verdicts[{index}].status 非法")
        claim_refs = {
            cited.get("content_ref")
            for cited in claims[claim_id].get("cited_sources") or []
            if isinstance(cited, dict) and isinstance(cited.get("content_ref"), str)
        }
        if not isinstance(evidence_refs, list) or any(
            not isinstance(ref, str) or ref not in claim_refs or ref not in visible_evidence_refs
            for ref in evidence_refs
        ):
            raise JudgeError(f"Judge claim_verdicts[{index}].evidence_refs 非法")
        if status != "insufficient" and not evidence_refs:
            raise JudgeError(f"Judge claim_verdicts[{index}].evidence_refs 非法")
        if not isinstance(reason, str) or not reason.strip():
            raise JudgeError(f"Judge claim_verdicts[{index}].reason 非法")
        seen.add(claim_id)
    if seen != set(claims):
        raise JudgeError("Judge claim_verdicts 未覆盖全部当前 claims")


def _parse_verify_response(
    text: str,
    rubric: RubricDef,
    artifact: dict[str, Any] | None = None,
    visible_evidence_refs: set[str] | None = None,
) -> VerifyScore:
    """严格解析 Judge JSON；非法输出不得形成质量分。"""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        raise JudgeError("Judge 输出不是合法 JSON") from e
    if not isinstance(data, dict):
        raise JudgeError("Judge 输出必须是 JSON object")

    src = data.get("raw_scores")
    if not isinstance(src, dict):
        raise JudgeError("Judge raw_scores 必须是 object")

    expected_keys = {dim.key for dim in rubric.dims}
    missing_keys = expected_keys - set(src)
    if missing_keys:
        raise JudgeError(f"Judge raw_scores 缺少维度: {', '.join(sorted(missing_keys))}")

    raw_scores: dict[str, float] = {}
    for dim in rubric.dims:
        v = src.get(dim.key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise JudgeError(f"Judge 维度 {dim.key} 必须是 JSON number")
        try:
            value = float(v)
        except OverflowError as e:
            raise JudgeError(f"Judge 维度 {dim.key} 必须是有限数字") from e
        if not math.isfinite(value):
            raise JudgeError(f"Judge 维度 {dim.key} 必须是有限数字")
        if value < 0 or value > rubric.raw_max:
            raise JudgeError(f"Judge 维度 {dim.key} 超出范围 0..{rubric.raw_max:g}")
        raw_scores[dim.key] = value

    feedback = data.get("feedback")
    if not isinstance(feedback, dict):
        raise JudgeError("Judge feedback 必须是 object")
    required_feedback = {
        "summary",
        "issues",
        "missing_coverage",
        "wrong_citations",
        "weak_chapters",
    }
    missing_feedback = required_feedback - set(feedback)
    if missing_feedback:
        raise JudgeError(f"Judge feedback 缺少字段: {', '.join(sorted(missing_feedback))}")
    if not isinstance(feedback["summary"], str):
        raise JudgeError("Judge feedback.summary 必须是 string")

    issues = feedback["issues"]
    if not isinstance(issues, list):
        raise JudgeError("Judge feedback.issues 必须是 array")
    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            raise JudgeError(f"Judge feedback.issues[{index}] 必须是 object")
        if not isinstance(issue.get("dim"), str) or not isinstance(issue.get("detail"), str):
            raise JudgeError(f"Judge feedback.issues[{index}] 必须包含 string dim/detail")

    for key in ("missing_coverage", "weak_chapters"):
        values = feedback[key]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise JudgeError(f"Judge feedback.{key} 必须是 string array")

    wrong_citations = feedback["wrong_citations"]
    if not isinstance(wrong_citations, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in wrong_citations
    ):
        raise JudgeError("Judge feedback.wrong_citations 必须是 integer array")
    _validate_claim_verdicts(feedback, artifact or {}, visible_evidence_refs or set())
    total = rubric.weighted_total(raw_scores)
    if not math.isfinite(total):
        raise JudgeError("Judge 加权分不是有限数字")
    return VerifyScore(raw_scores=raw_scores, total=total, feedback=feedback)


async def _invoke_critic(model: ChatOpenAI, system: str, user: str) -> str:
    """以 critic 角色调用 LLM(独立 session,messages 数组直接构造,不带历史)。"""
    try:
        resp = await model.ainvoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:  # noqa: BLE001
        logger.warning("Verifier LLM 调用失败: %s", e)
        raise JudgeError(f"Verifier LLM 调用失败: {type(e).__name__}") from e


# ── 同模型 self-critique 基线 ──


class SameModelVerifier(Verifier):
    """同模型 self-critique:用 generator 同款 ChatOpenAI 实例,但新开 session(messages 独立)。

    存在偏置风险(模型可能倾向认可自己生成的风格),在 A/B 实验中作为基线对照。
    """

    kind = "same"

    def __init__(self, model: ChatOpenAI, model_name: str = ""):
        self.model = model
        self.model_name = model_name or getattr(model, "model_name", "") or ""

    async def verify(
        self, *, topic: str, artifact: dict[str, Any], rubric: RubricDef
    ) -> VerifyScore:
        _validate_claim_artifact(artifact)
        system = _critic_role()
        user = _render_research_prompt(topic=topic, rubric=rubric, artifact=artifact)
        _, visible_refs = _artifact_evidence_context(artifact)
        text = await _invoke_critic(self.model, system, user)
        return _parse_verify_response(text, rubric, artifact, visible_refs)


# ── 独立配置 Verifier ──


class CrossModelVerifier(Verifier):
    """使用用户单独配置的 verifier 模型(model_configs.type='verifier')。

    没配 verifier 类型模型时,build_verifier() 会显式报告 unavailable。
    """

    kind = "cross"

    def __init__(self, model: ChatOpenAI, model_name: str = ""):
        self.model = model
        self.model_name = model_name or getattr(model, "model_name", "") or ""

    async def verify(
        self, *, topic: str, artifact: dict[str, Any], rubric: RubricDef
    ) -> VerifyScore:
        _validate_claim_artifact(artifact)
        system = _critic_role()
        user = _render_research_prompt(topic=topic, rubric=rubric, artifact=artifact)
        _, visible_refs = _artifact_evidence_context(artifact)
        text = await _invoke_critic(self.model, system, user)
        return _parse_verify_response(text, rubric, artifact, visible_refs)


# ── 工厂:按用户配置和 kind 选 verifier ──


async def _get_verifier_config(session: AsyncSession, user_id: uuid.UUID) -> ModelConfig | None:
    """取用户的 verifier 类型模型配置(默认那条);未配置返回 None。"""
    configs = await ModelConfigRepository(session).list_by_user(user_id, "verifier")
    if not configs:
        return None
    return next((c for c in configs if c.is_default), configs[0])


async def build_verifier(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    kind: str,
    generator_model: ChatOpenAI,
    generator_model_name: str = "",
) -> Verifier:
    """工厂方法:按 kind 构建 verifier。

    kind="same":  直接复用 generator_model(同模型新开 session)
    kind="cross": 用用户配置的 verifier 类型模型;配置/构建不可用时显式抛错

    Args:
        session: DB session(查 verifier 模型配置)
        user_id: 当前用户
        kind: "same" / "cross"
        generator_model: 当前 generator 用的 ChatOpenAI(same 模式直接复用)
        generator_model_name: generator 模型名(落库 audit)
    """
    if kind == "same":
        return SameModelVerifier(generator_model, model_name=generator_model_name)

    if kind == "cross":
        try:
            cfg = await _get_verifier_config(session, user_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("拉取 verifier 模型配置失败: %s", e)
            try:
                await session.rollback()
            except Exception as rollback_error:  # noqa: BLE001
                logger.warning("verifier 配置查询失败后 rollback 失败: %s", rollback_error)
            raise VerifierUnavailableError("verifier 配置查询失败") from e
        if cfg is None:
            raise VerifierUnavailableError("未配置 cross verifier 模型")
        try:
            model = build_chat_model(cfg, temperature=0.0, streaming=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("构建独立配置 verifier 失败: %s", e)
            raise VerifierUnavailableError("cross verifier 模型构建失败") from e
        return CrossModelVerifier(model, model_name=cfg.model_name)

    raise BizError(f"未知 verifier kind: {kind}", code=2020)

"""LLM-as-judge Verifier:同模型 self-critique 基线 + 独立配置 Judge 两套实现。

两套并存,跑 A/B 实验:
- SameModelVerifier (kind="same"): 同 chat 模型新开 session,critic 角色 prompt
- CrossModelVerifier (kind="cross"): 用独立配置的 verifier 模型

`cross` 只表示使用独立 verifier 配置，不推断或保证模型 family 不同。
"""

from __future__ import annotations

import json
import math
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


# ── 共用工具 ──


def _critic_role() -> str:
    return render_verifier_prompt("critic_role.jinja2")


def _render_research_prompt(*, topic: str, rubric: RubricDef, artifact: dict[str, Any]) -> str:
    """渲染研究报告的 verifier prompt(给单条 user message,system 走 critic_role)。"""
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
        sources=artifact.get("sources") or [],
    )


def _parse_verify_response(text: str, rubric: RubricDef) -> VerifyScore:
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
        system = _critic_role()
        user = _render_research_prompt(topic=topic, rubric=rubric, artifact=artifact)
        text = await _invoke_critic(self.model, system, user)
        return _parse_verify_response(text, rubric)


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
        system = _critic_role()
        user = _render_research_prompt(topic=topic, rubric=rubric, artifact=artifact)
        text = await _invoke_critic(self.model, system, user)
        return _parse_verify_response(text, rubric)


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

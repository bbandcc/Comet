"""PatchRepair:有界补丁修复。

适用问题:覆盖度 / 引用对齐 / 时效性单点缺漏(verifier feedback 里 missing_coverage / wrong_citations 有内容)。

思路:
1. 从 claim verdict 生成目标专属查询；普通覆盖缺漏仍兼容补充查询
2. 通过上层 callback 执行检索，并按 artifact version + locator 精确替换目标断言
3. 无目标专属证据时降级为不确定表述，不拿其它 claim 的 Learning 代替

loop 模块本身不依赖 research engine,通过 callback 解耦。
"""

from __future__ import annotations

from typing import Any

from app.core.agent.loop.models import ClaimRepairTarget, RepairAction, VerifyScore
from app.core.agent.loop.repair.base import RepairExecutor
from app.core.logging import get_logger

logger = get_logger(__name__)


class PatchRepair(RepairExecutor):
    """有界补丁修复(默认每轮最多处理 3 个目标或补搜查询)。"""

    kind = "patch"

    def __init__(self, max_queries: int = 3):
        self.max_queries = max_queries

    def plan(self, *, score: VerifyScore, artifact: dict[str, Any]) -> RepairAction:
        """从 verifier feedback 抽取「缺什么」→ 生成补搜子查询。"""
        fb = score.feedback or {}
        queries: list[str] = []
        claims_by_id = {
            claim.get("claim_id"): claim
            for claim in artifact.get("claims") or []
            if isinstance(claim, dict)
        }
        targets: list[ClaimRepairTarget] = []

        for verdict in fb.get("claim_verdicts") or []:
            if len(targets) >= self.max_queries:
                break
            if not isinstance(verdict, dict) or verdict.get("status") not in {
                "contradicted",
                "insufficient",
            }:
                continue
            claim = claims_by_id.get(verdict.get("claim_id"))
            if claim is None:
                continue
            repair_query = f"{claim['claim_text']} {verdict['reason']}".strip()
            target = ClaimRepairTarget(
                claim_id=claim["claim_id"],
                section_id=claim["section_id"],
                claim_text=claim["claim_text"],
                locator_start=claim["locator_start"],
                locator_end=claim["locator_end"],
                repair_query=repair_query,
                status=verdict["status"],
                evidence_refs=list(verdict.get("evidence_refs") or []),
                reason=verdict["reason"],
            )
            targets.append(target)
            queries.append(repair_query)

        # 1) 直接利用 missing_coverage(verifier 已经写成自然语言子问题)
        for item in fb.get("missing_coverage") or []:
            if isinstance(item, str) and item.strip():
                queries.append(item.strip())

        # 2) 利用 issues 里维度为 coverage / faithfulness / timeliness 的具体问题
        for it in fb.get("issues") or []:
            if not isinstance(it, dict):
                continue
            dim = (it.get("dim") or "").lower()
            detail = (it.get("detail") or "").strip()
            if not detail:
                continue
            if dim in {"coverage", "faithfulness", "timeliness"}:
                queries.append(detail)

        # 去重 + 截断
        seen: set[str] = set()
        deduped: list[str] = []
        for q in queries:
            key = q[:120]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(q)
        deduped = deduped[: self.max_queries]

        rationale = (
            f"verifier 指出 {len(targets)} 个待修复断言、"
            f"{len(fb.get('missing_coverage') or [])} 处覆盖缺漏，"
            f"选 {len(deduped)} 个最关键的检索问题。"
        )
        return RepairAction(
            kind=self.kind,
            patch_queries=deduped,
            artifact_version=artifact.get("artifact_version") if targets else None,
            target_claims=targets,
            rationale=rationale,
        )

    async def execute(
        self,
        *,
        action: RepairAction,
        artifact: dict[str, Any],
        ctx: dict[str, Any],
    ) -> dict[str, Any]:
        """调上层注入的 callback 完成定向断言修复或 legacy 覆盖补搜。"""
        callback = ctx.get("patch_callback")
        if action.artifact_version and action.artifact_version != artifact.get("artifact_version"):
            logger.warning("PatchRepair.execute: artifact version 已过期,拒绝修改")
            return artifact
        if callback is None or not action.patch_queries:
            logger.warning("PatchRepair.execute: 未提供 patch_callback 或无子查询,返回原 artifact")
            return artifact
        try:
            callback_arg = action if action.target_claims else action.patch_queries
            new_artifact = await callback(callback_arg)
            if not isinstance(new_artifact, dict):
                logger.warning("patch_callback 未返回 dict,沿用旧 artifact")
                return artifact
            return new_artifact
        except Exception as e:  # noqa: BLE001
            logger.warning("PatchRepair 执行失败,沿用旧 artifact: %s", e)
            return artifact

"""目标断言 replacement 的窄决策契约。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from app.core.agent.research.models import Learning, Source
from app.core.agent.research.prompt_renderer import render_research_prompt
from app.core.agent.tracing import push_llm_usage
from app.core.logging import get_logger

logger = get_logger(__name__)

ACTION_REPLACE = "replace"
ACTION_INSUFFICIENT = "insufficient"
_SOURCE_EXCERPT_CHARS = 2000
_MAX_SOURCES = 3
_MAX_LEARNINGS = 8
_CITATION_RE = re.compile(r"\[来源\s*\d+\]")


@dataclass(frozen=True)
class ClaimReplacementDecision:
    action: str
    replacement: str = ""
    source_index: int | None = None


def _insufficient() -> ClaimReplacementDecision:
    return ClaimReplacementDecision(action=ACTION_INSUFFICIENT)


async def decide_claim_replacement(
    model: ChatOpenAI,
    *,
    claim_text: str,
    reason: str,
    sources: list[Source],
    learnings: list[Learning],
) -> ClaimReplacementDecision:
    """严格决定是否有目标专属证据可替换原 claim；任何不确定性均 fail closed。"""
    visible_sources = sources[:_MAX_SOURCES]
    allowed_indices = {source.index for source in visible_sources}
    visible_learnings = [
        learning for learning in learnings if learning.source_index in allowed_indices
    ][:_MAX_LEARNINGS]
    if not visible_sources or not visible_learnings:
        return _insufficient()
    learning_indices = {learning.source_index for learning in visible_learnings}
    prompt = render_research_prompt(
        "repair_claim.jinja2",
        claim_text=claim_text,
        reason=reason,
        sources=[
            {
                "index": source.index,
                "title": source.title,
                "content": source.content[:_SOURCE_EXCERPT_CHARS],
            }
            for source in visible_sources
        ],
        learnings=[
            {"source_index": learning.source_index, "text": learning.text[:1000]}
            for learning in visible_learnings
        ],
    )
    try:
        response = await model.ainvoke(prompt)
        push_llm_usage(response, model)
        text = response.content if isinstance(response.content, str) else str(response.content)
        data = json.loads(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("claim replacement decision 失败,降级为证据不足: %s", e)
        return _insufficient()

    if not isinstance(data, dict) or set(data) != {"action", "replacement", "source_index"}:
        return _insufficient()
    action = data.get("action")
    replacement = data.get("replacement")
    source_index = data.get("source_index")
    if action == ACTION_INSUFFICIENT:
        return _insufficient()
    if (
        action != ACTION_REPLACE
        or not isinstance(replacement, str)
        or not replacement.strip()
        or _CITATION_RE.search(replacement)
        or isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or source_index not in allowed_indices
        or source_index not in learning_indices
    ):
        return _insufficient()
    return ClaimReplacementDecision(
        action=ACTION_REPLACE,
        replacement=replacement.strip(),
        source_index=source_index,
    )


__all__ = [
    "ACTION_INSUFFICIENT",
    "ACTION_REPLACE",
    "ClaimReplacementDecision",
    "decide_claim_replacement",
]

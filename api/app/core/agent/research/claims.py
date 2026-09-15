"""Research 内部断言定位契约；不进入公共 SSE/API。"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.agent.loop.models import ClaimRepairTarget
    from app.core.agent.research.models import Source

_CITATION_RE = re.compile(r"\[来源\s*(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])|(?<!\d\.)(?<=[.!?])\s+|\n+")
_MARKDOWN_PREFIX_RE = re.compile(r"^(?:(?:[-*+>])|(?:\d+[.)]))(?:\s+|$)")
_FACTUAL_RE = re.compile(
    r"\d|(?:达到|包含|位于|发布于|成立于|总部位于|注册于|上市于|已上市|状态为|评级为|属于|增长|下降)|"
    r"\b(?:contains?|reported|increased|decreased|founded|headquartered|located|listed|owned|valued)\b",
    re.IGNORECASE,
)
_SUBJECTIVE_RE = re.compile(r"^(?:建议|我的判断|我认为|我们认为|应当|应该|可考虑|可以考虑)")


def _classification_text(segment: str) -> str:
    """仅为候选分类去掉 Markdown 前缀，保留原文定位与 claim 文本。"""
    text = segment
    while True:
        stripped = _MARKDOWN_PREFIX_RE.sub("", text, count=1).lstrip()
        if stripped == text:
            return text
        text = stripped


def _claim_segments(text: str) -> list[tuple[str, int, int]]:
    """返回需证据候选及其在原 block 内的精确 span。"""
    claims: list[tuple[str, int, int]] = []
    cursor = 0
    for raw_segment in _SENTENCE_SPLIT_RE.split(text or ""):
        segment = raw_segment.strip()
        if not segment:
            cursor += len(raw_segment)
            continue
        start = text.find(segment, cursor)
        if start < 0:
            continue
        end = start + len(segment)
        cursor = end
        classification_text = _classification_text(segment)
        if _CITATION_RE.search(classification_text) or (
            _FACTUAL_RE.search(classification_text)
            and not _SUBJECTIVE_RE.search(classification_text)
        ):
            claims.append((segment, start, end))
    return claims


def build_claim_contract(
    summary: dict,
    sections: list[tuple[str, str]],
    sources: list[Source],
) -> tuple[str, list[dict]]:
    """从 linkify 前原文构造 artifact 版本与可定位 claim map。"""
    source_versions = [
        {
            "index": source.index,
            "stable_id": source.stable_id,
            "content_ref": source.content_ref,
            "content_hash": source.content_hash,
        }
        for source in sources
    ]
    raw_payload = {
        "summary": summary,
        "sections": sections,
        "source_versions": source_versions,
    }
    canonical = json.dumps(
        raw_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )
    artifact_version = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    sources_by_index = {source.index: source for source in sources}

    blocks: list[tuple[str, str]] = [("summary:tldr", str(summary.get("tldr") or ""))]
    blocks.extend(
        (f"summary:key_point:{index}", str(text))
        for index, text in enumerate(summary.get("key_points") or [], start=1)
    )
    blocks.extend(
        (f"section:{index}", content) for index, (_heading, content) in enumerate(sections, start=1)
    )

    claims: list[dict] = []
    for section_id, text in blocks:
        for ordinal, (claim_text, locator_start, locator_end) in enumerate(
            _claim_segments(text), start=1
        ):
            claim_identity = (
                f"{section_id}\0{ordinal}\0{locator_start}\0{locator_end}\0{claim_text}"
            )
            claim_id = hashlib.sha256(claim_identity.encode("utf-8")).hexdigest()
            cited_sources = []
            for index in dict.fromkeys(int(value) for value in _CITATION_RE.findall(claim_text)):
                source = sources_by_index.get(index)
                cited_sources.append(
                    {
                        "index": index,
                        "content_ref": source.content_ref if source else None,
                        "content_hash": source.content_hash if source else None,
                    }
                )
            claims.append(
                {
                    "section_id": section_id,
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "locator_start": locator_start,
                    "locator_end": locator_end,
                    "cited_sources": cited_sources,
                    "artifact_version": artifact_version,
                }
            )
    return artifact_version, claims


def apply_claim_replacements(
    summary: dict,
    sections: list[tuple[str, str]],
    targets: list[ClaimRepairTarget],
    replacements: dict[str, str],
) -> tuple[dict, list[tuple[str, str]], bool, bool]:
    """按 claim 定位替换原始断言；精确文本不匹配时不做猜测性修改。"""
    updated_summary = dict(summary)
    updated_summary["key_points"] = list(summary.get("key_points") or [])
    updated_sections = list(sections)
    changed = False
    body_changed = False

    for target in sorted(targets, key=lambda item: item.locator_start, reverse=True):
        replacement = replacements.get(target.claim_id)
        if replacement is None:
            continue
        section_id = target.section_id
        if section_id == "summary:tldr":
            current = str(updated_summary.get("tldr") or "")
            if current[target.locator_start : target.locator_end] == target.claim_text:
                updated_summary["tldr"] = (
                    current[: target.locator_start] + replacement + current[target.locator_end :]
                )
                changed = True
            continue
        if section_id.startswith("summary:key_point:"):
            try:
                index = int(section_id.rsplit(":", 1)[1]) - 1
            except ValueError:
                continue
            points = updated_summary["key_points"]
            current = str(points[index]) if 0 <= index < len(points) else ""
            if current[target.locator_start : target.locator_end] == target.claim_text:
                points[index] = (
                    current[: target.locator_start] + replacement + current[target.locator_end :]
                )
                changed = True
            continue
        if section_id.startswith("section:"):
            try:
                index = int(section_id.split(":", 1)[1]) - 1
            except ValueError:
                continue
            if 0 <= index < len(updated_sections):
                heading, content = updated_sections[index]
                if content[target.locator_start : target.locator_end] == target.claim_text:
                    updated_sections[index] = (
                        heading,
                        (
                            content[: target.locator_start]
                            + replacement
                            + content[target.locator_end :]
                        ),
                    )
                    changed = True
                    body_changed = True

    return updated_summary, updated_sections, changed, body_changed


__all__ = ["apply_claim_replacements", "build_claim_contract"]

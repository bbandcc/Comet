"""研究证据持久化边界：小正文内联，大正文转已有对象存储。"""

import uuid
from collections.abc import Iterable

from app.core.logging import get_logger
from app.core.storage import build_file_key, get_storage

MAX_INLINE_EVIDENCE_CHARS = 100_000
logger = get_logger(__name__)


async def prepare_evidence_sources(
    user_id: uuid.UUID,
    report_id: uuid.UUID,
    evidence_sources: list[dict],
    *,
    created_storage_refs: list[str] | None = None,
) -> list[dict]:
    """限制 JSONB 热行大小；超限正文显式写入对象存储，不静默截断。"""
    prepared: list[dict] = []
    inline_chars = 0
    storage = None
    for source in evidence_sources:
        item = dict(source)
        content = str(item.get("content") or "")
        item["storage_truncated"] = False
        if inline_chars + len(content) <= MAX_INLINE_EVIDENCE_CHARS:
            inline_chars += len(content)
        elif content:
            storage = storage or get_storage()
            file_id = f"{item.get('stable_id', 'source')}-{item.get('content_hash', 'version')}"
            file_key = build_file_key(
                str(user_id),
                "research-evidence",
                f"{report_id}/{file_id}",
                ".txt",
            )
            existed = await storage.exists(file_key)
            await storage.save(file_key, content.encode("utf-8"))
            item["content"] = None
            item["storage_ref"] = f"storage:{file_key}"
            if created_storage_refs is not None and not existed:
                created_storage_refs.append(item["storage_ref"])
        prepared.append(item)
    return prepared


async def read_external_evidence(storage_ref: str) -> str:
    if not storage_ref.startswith("storage:"):
        raise ValueError("不是外部证据引用")
    payload = await get_storage().get(storage_ref.removeprefix("storage:"))
    return payload.decode("utf-8")


async def delete_external_evidence(storage_refs: Iterable[str]) -> None:
    """Best-effort 删除证据对象；清理失败不覆盖主业务结果。"""
    storage = None
    for storage_ref in dict.fromkeys(storage_refs):
        if not storage_ref.startswith("storage:"):
            continue
        try:
            storage = storage or get_storage()
            await storage.delete(storage_ref.removeprefix("storage:"))
        except Exception as exc:  # noqa: BLE001 - 对象清理是补偿动作
            logger.warning("删除外部研究证据失败: ref=%s err=%s", storage_ref, exc)
            continue


__all__ = [
    "MAX_INLINE_EVIDENCE_CHARS",
    "delete_external_evidence",
    "prepare_evidence_sources",
    "read_external_evidence",
]

"""定时/主动任务 Celery 任务：

- heartbeat：每分钟扫 agent_tasks 找到期的，一个事务内推进 next_run_at（防重复触发）后派发执行。
- run_agent_task：跑深度研究引擎产出报告（research_reports，task_id 关联），回写任务状态。
  - 单任务并发护栏：Redis owner-aware 锁（SET NX + TTL + 原子释放，防重叠跑与误删锁）。
  - 整体硬超时：asyncio.wait_for（跨平台，Windows 上 celery time_limit 不生效）。

队列：heartbeat → beat（轻量，不可被堵）；run → research（重活，独立队列）。
与其他 beat 任务一致：任务级独立引擎（NullPool）+ 独立事件循环（asyncio.run）。
"""

import asyncio
import uuid
from datetime import datetime

from redis import asyncio as aioredis
from redis.exceptions import LockNotOwnedError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.models  # noqa: F401  确保 ORM 模型注册
from app.celery_app import celery_app
from app.config import settings
from app.core.agent.research.engine import run_research
from app.core.logging import get_logger
from app.db.postgres import create_task_engine
from app.models.agent_task_model import (
    TASK_RUN_DONE,
    TASK_RUN_FAILED,
    TASK_RUN_RUNNING,
)
from app.models.research_report_model import (
    RESEARCH_STATUS_DONE,
    RESEARCH_STATUS_FAILED,
    RESEARCH_STATUS_PENDING,
    ResearchReport,
)
from app.repositories.agent_task_repository import AgentTaskRepository
from app.repositories.research_report_repository import ResearchReportRepository
from app.services.agent_task_service import TZ, compute_next_run

logger = get_logger(__name__)


# ── 每分钟心跳：原子认领到期任务 → 推进 next_run_at → 派发 ──


async def _heartbeat() -> int:
    engine_db = create_task_engine()
    sm = async_sessionmaker(engine_db, expire_on_commit=False, class_=AsyncSession)
    dispatched_ids: list[str] = []
    try:
        async with sm() as session:
            repo = AgentTaskRepository(session)
            # FOR UPDATE SKIP LOCKED 认领 + 同事务推进 next_run_at + 一次提交，
            # 避免逐行提交后尾部任务在另一心跳里被重复触发。
            due = await repo.list_due()
            for task in due:
                task.next_run_at = compute_next_run(task)
                dispatched_ids.append(str(task.id))
            if due:
                await session.commit()
    finally:
        await engine_db.dispose()

    # 提交（next_run_at 已推进）之后再派发，避免「派发了却没推进」
    for tid in dispatched_ids:
        run_agent_task_task.delay(tid)
    if dispatched_ids:
        logger.info("定时任务心跳：派发 %d 个到期任务", len(dispatched_ids))
    return len(dispatched_ids)


@celery_app.task(name="app.tasks.agent_task.heartbeat")
def heartbeat_task() -> int:
    """每分钟调度心跳的 Celery 任务入口。"""
    return asyncio.run(_heartbeat())


# ── 执行一次研究任务（Redis 锁 + 硬超时）──


async def _run_task(task_id: str) -> None:
    tid = uuid.UUID(task_id)
    attempt_id = uuid.uuid4().hex
    lock_key = f"agent_task:lock:{task_id}"
    lock_ttl = settings.research_task_timeout + 120  # 略大于超时，自动过期防死锁
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    lock = redis.lock(lock_key, timeout=lock_ttl, blocking=False)
    acquired = False
    try:
        try:
            acquired = await lock.acquire()
        except Exception as e:
            logger.error(
                "获取定时任务锁失败，停止执行: id=%s attempt=%s "
                "status=acquire_failed err_type=%s err=%s",
                tid,
                attempt_id,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise
        if not acquired:
            logger.info(
                "定时任务已在运行中，跳过本次触发: id=%s attempt=%s status=skipped_busy",
                tid,
                attempt_id,
            )
            return
        logger.info("定时任务锁获取成功: id=%s attempt=%s status=acquired", tid, attempt_id)
        await _do_run(tid)
    finally:
        if acquired:
            try:
                # redis-py 会用本次 acquire 的 token 执行 Lua compare-delete。
                await lock.release()
                logger.info(
                    "定时任务锁释放成功: id=%s attempt=%s status=released",
                    tid,
                    attempt_id,
                )
            except LockNotOwnedError:
                logger.warning(
                    "定时任务锁所有权已丢失，跳过释放: id=%s attempt=%s status=lock_lost",
                    tid,
                    attempt_id,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "释放定时任务锁失败，等待 TTL 回收: id=%s attempt=%s "
                    "status=release_failed err_type=%s err=%s",
                    tid,
                    attempt_id,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
        try:
            await redis.aclose()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "关闭定时任务 Redis 客户端失败: id=%s attempt=%s err_type=%s err=%s",
                tid,
                attempt_id,
                type(e).__name__,
                e,
            )


async def _do_run(tid: uuid.UUID) -> None:
    engine_db = create_task_engine()
    sm = async_sessionmaker(engine_db, expire_on_commit=False, class_=AsyncSession)
    try:
        # 1) 建报告行 + 标任务运行中（健康事务）
        async with sm() as session:
            task = await AgentTaskRepository(session).get_by_id(tid)
            if not task:
                logger.warning("定时任务执行：任务不存在 %s", tid)
                return
            user_id = task.user_id
            instruction = task.instruction
            notify_enabled = task.notify_enabled
            kb_ids = [str(k) for k in (task.kb_ids or [])] or None
            report = await ResearchReportRepository(session).create(
                ResearchReport(
                    user_id=user_id,
                    topic=instruction,
                    status=RESEARCH_STATUS_PENDING,
                    task_id=task.id,
                )
            )
            report_id = report.id
            task.last_status = TASK_RUN_RUNNING
            await AgentTaskRepository(session).save(task)

        # 2) 跑研究（独立 session + 整体硬超时）
        ok = await _execute_research(sm, report_id, user_id, instruction, kb_ids)

        # 3) 回写任务最近运行状态/时间（全新 session，避免受研究阶段事务影响）
        async with sm() as session:
            task = await AgentTaskRepository(session).get_by_id(tid)
            if task:
                task.last_run_at = datetime.now(TZ)
                task.last_status = TASK_RUN_DONE if ok else TASK_RUN_FAILED
                await AgentTaskRepository(session).save(task)
        logger.info("定时任务执行完成: id=%s ok=%s", tid, ok)

        # 4) 成功且任务开启推送 → 检查 Verifier Loop 通过状态;不合格则不推送(避免低质量内容打扰)
        if ok and notify_enabled:
            verified = await _check_loop_passed(sm, report_id)
            if verified:
                await _notify_user(sm, user_id, report_id, instruction)
            else:
                logger.info(
                    "定时任务 Verifier Loop 未通过,跳过手机推送: report=%s task=%s",
                    report_id,
                    tid,
                )
    finally:
        await engine_db.dispose()


async def _execute_research(
    sm: async_sessionmaker,
    report_id: uuid.UUID,
    user_id: uuid.UUID,
    topic: str,
    kb_ids: list[str] | None,
) -> bool:
    """消费研究引擎事件并落库（无 SSE/bus，后台直跑），带整体硬超时。"""
    holder: dict = {
        "md": None,
        "sources": [],
        "evidence_sources": [],
        "title": None,
        "outline": None,
        "partial": "",
    }

    async def _collect(session: AsyncSession) -> None:
        async for ev in run_research(session, user_id, topic, kb_ids, report_id=report_id):
            etype = ev.get("type")
            if etype == "plan":
                holder["outline"] = {
                    "title": ev.get("title", ""),
                    "sections": ev.get("sections", []),
                    "queries": ev.get("queries", []),
                }
                holder["title"] = ev.get("title")
            elif etype == "section_start":
                holder["partial"] += f"\n\n## {ev.get('heading', '')}\n\n"
            elif etype == "token":
                holder["partial"] += ev.get("text", "")
            elif etype == "report":
                holder["md"] = ev.get("markdown", "")
                holder["sources"] = ev.get("sources", [])
                holder["evidence_sources"] = ev.get("_evidence_sources", [])
                holder["title"] = ev.get("title", holder["title"])
            elif etype == "error":
                raise RuntimeError(ev.get("message", "研究失败"))

    try:
        async with sm() as session:
            await asyncio.wait_for(_collect(session), timeout=settings.research_task_timeout)
            if holder["md"] is None:
                raise RuntimeError("研究未产出报告")
            # 正常完成：session 健康，直接落 done
            repo = ResearchReportRepository(session)
            report = await repo.get_by_id(report_id)
            if report:
                report.title = (holder["title"] or topic)[:255]
                report.report_md = holder["md"]
                report.sources = holder["sources"]
                from app.core.agent.research.evidence_store import (
                    delete_external_evidence,
                    prepare_evidence_sources,
                )

                created_storage_refs: list[str] = []
                try:
                    report.evidence_sources = await prepare_evidence_sources(
                        user_id,
                        report_id,
                        holder["evidence_sources"],
                        created_storage_refs=created_storage_refs,
                    )
                    report.outline = holder["outline"]
                    report.status = RESEARCH_STATUS_DONE
                    report.error_msg = None
                    await repo.save(report)
                except BaseException:
                    await delete_external_evidence(created_storage_refs)
                    raise
        return True
    except (TimeoutError, asyncio.TimeoutError):
        msg = f"研究超时（超过 {settings.research_task_timeout} 秒）"
        logger.error("定时研究超时: report=%s", report_id)
    except Exception as e:
        msg = str(e)
        logger.error("定时研究执行失败: report=%s err=%s", report_id, e, exc_info=True)
    # 失败落库：用全新 session（超时取消可能令原 session 处于脏/中断状态）
    await _mark_failed(sm, report_id, msg, holder)
    return False


async def _mark_failed(
    sm: async_sessionmaker, report_id: uuid.UUID, msg: str, holder: dict
) -> None:
    try:
        async with sm() as session:
            repo = ResearchReportRepository(session)
            report = await repo.get_by_id(report_id)
            if report:
                report.status = RESEARCH_STATUS_FAILED
                report.error_msg = (msg or "研究失败")[:2000]
                if holder["partial"].strip():
                    report.report_md = holder["partial"]
                report.outline = holder["outline"]
                await repo.save(report)
    except Exception as e:  # noqa: BLE001
        logger.warning("研究失败落库出错（忽略）: report=%s err=%s", report_id, e)


@celery_app.task(name="app.tasks.agent_task.run")
def run_agent_task_task(task_id: str) -> str:
    """执行一次定时研究任务的 Celery 入口。"""
    asyncio.run(_run_task(task_id))
    return task_id


# ── 完成后推送通知 ──


def _extract_summary(md: str, limit: int = 360) -> str:
    """从报告 Markdown 提取简报：TL;DR 引用块 + 核心要点前几条，截断。"""
    if not md:
        return ""
    lines = md.splitlines()
    tldr: list[str] = []
    points: list[str] = []
    in_points = False
    for line in lines:
        s = line.strip()
        if s.startswith(">"):
            tldr.append(s.lstrip("> ").strip())
        elif s.startswith("##") and ("核心要点" in s or "要点" in s):
            in_points = True
        elif s.startswith("##"):
            in_points = False
        elif in_points and (s.startswith("- ") or s.startswith("* ")):
            points.append(s[2:].strip())
    parts: list[str] = []
    if tldr:
        parts.append(" ".join(tldr))
    if points:
        parts.append("\n".join(f"· {p}" for p in points[:5]))
    text = "\n\n".join(parts).strip()
    if not text:
        # 兜底：取正文前若干字（去标题/角标）
        text = " ".join(s for s in (ln.strip() for ln in lines) if s and not s.startswith("#"))
    return text[:limit] + ("…" if len(text) > limit else "")


async def _notify_user(
    sm: async_sessionmaker, user_id: uuid.UUID, report_id: uuid.UUID, topic: str
) -> None:
    """把完成的报告 TL;DR 推到用户的消息渠道。整步降级，绝不影响任务。"""
    try:
        from app.config import settings
        from app.services.notify_service import NotifyService

        async with sm() as session:
            report = await ResearchReportRepository(session).get_by_id(report_id)
            if not report or not report.report_md:
                return
            title = report.title or topic[:40] or "研究报告"
            summary = _extract_summary(report.report_md)
            link = f"{settings.notify_site_url.rstrip('/')}/research?report={report_id}"
            content = f"{summary}\n\n📄 查看完整报告：{link}"
            sent = await NotifyService(session).push_to_user(user_id, f"🔬 {title}", content)
            if sent:
                logger.info("定时任务推送完成: report=%s 渠道数=%d", report_id, sent)
    except Exception as e:  # noqa: BLE001
        logger.warning("定时任务推送失败（忽略）: report=%s err=%s", report_id, e)


async def _check_loop_passed(sm: async_sessionmaker, report_id: uuid.UUID) -> bool:
    """检查 research engine 已跑过的 Verifier Loop 通过状态。

    V0.0.5 ② 设计:engine 内已经接 LoopController(task_type=research, task_id=report_id),
    定时任务**不重复跑 verify**,只读结果决定要不要推送。

    返回 True 当且仅当:
    - Loop 关闭(settings.loop_enabled=False) → 不调用 Judge、记录 skipped，报告与推送继续交付
    - 强质量门禁开启时，找到 LoopRun 且 quality_status=passed
    其他情况均视为未通过；缺 run 与查询错误必须 fail-closed。
    """
    if not settings.loop_enabled:
        return True
    try:
        from app.core.agent.loop.store import LoopStore
        from app.core.agent.loop.models import QUALITY_PASSED

        async with sm() as session:
            run = await LoopStore(session).find_latest_by_task(
                task_type="research", task_id=report_id
            )
            if run is None:
                logger.warning("Verifier Loop 未找到对应 run,拒绝推送: report=%s", report_id)
                return False
            return run.quality_status == QUALITY_PASSED
    except Exception as e:  # noqa: BLE001
        logger.warning("查 LoopRun 失败,拒绝推送: report=%s err=%s", report_id, e)
        return False

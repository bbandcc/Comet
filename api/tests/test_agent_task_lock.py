import asyncio
import os
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from redis import asyncio as aioredis
from redis.exceptions import LockNotOwnedError

from app.config import settings
from app.tasks.agent_task import _run_task


TEST_REDIS_URL: str | None = os.getenv("COMET_TEST_REDIS_URL")


class AgentTaskLockTests(unittest.IsolatedAsyncioTestCase):
    def _redis_with_lock(
        self,
        *,
        acquired: bool = True,
        acquire_error: Exception | None = None,
        release_error: Exception | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        lock = MagicMock()
        if acquire_error is None:
            lock.acquire = AsyncMock(return_value=acquired)
        else:
            lock.acquire = AsyncMock(side_effect=acquire_error)
        lock.release = AsyncMock(side_effect=release_error)

        redis = MagicMock()
        redis.lock.return_value = lock
        redis.aclose = AsyncMock()
        return redis, lock

    async def test_acquired_lock_runs_research_and_releases_lease(self) -> None:
        task_id = str(uuid.uuid4())
        redis, lock = self._redis_with_lock()

        with (
            patch("app.tasks.agent_task.aioredis.from_url", return_value=redis),
            patch("app.tasks.agent_task._do_run", new_callable=AsyncMock) as do_run,
        ):
            await _run_task(task_id)

        redis.lock.assert_called_once_with(
            f"agent_task:lock:{task_id}",
            timeout=settings.research_task_timeout + 120,
            blocking=False,
        )
        lock.acquire.assert_awaited_once_with()
        do_run.assert_awaited_once_with(uuid.UUID(task_id))
        lock.release.assert_awaited_once_with()
        redis.aclose.assert_awaited_once_with()

    async def test_busy_lock_skips_without_running_or_releasing(self) -> None:
        task_id = str(uuid.uuid4())
        redis, lock = self._redis_with_lock(acquired=False)

        with (
            patch("app.tasks.agent_task.aioredis.from_url", return_value=redis),
            patch("app.tasks.agent_task._do_run", new_callable=AsyncMock) as do_run,
        ):
            await _run_task(task_id)

        do_run.assert_not_awaited()
        lock.release.assert_not_awaited()
        redis.aclose.assert_awaited_once_with()

    async def test_lock_acquisition_error_blocks_research(self) -> None:
        task_id = str(uuid.uuid4())
        redis, lock = self._redis_with_lock(acquire_error=ConnectionError("redis unavailable"))

        with (
            patch("app.tasks.agent_task.aioredis.from_url", return_value=redis),
            patch("app.tasks.agent_task._do_run", new_callable=AsyncMock) as do_run,
            patch("app.tasks.agent_task.logger") as logger,
            self.assertRaisesRegex(ConnectionError, "redis unavailable"),
        ):
            await _run_task(task_id)

        do_run.assert_not_awaited()
        lock.release.assert_not_awaited()
        redis.aclose.assert_awaited_once_with()
        self.assertIn("status=acquire_failed", logger.error.call_args.args[0])

    async def test_research_error_still_releases_owned_lease(self) -> None:
        task_id = str(uuid.uuid4())
        redis, lock = self._redis_with_lock()

        with (
            patch("app.tasks.agent_task.aioredis.from_url", return_value=redis),
            patch(
                "app.tasks.agent_task._do_run",
                new_callable=AsyncMock,
                side_effect=RuntimeError("research failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "research failed"),
        ):
            await _run_task(task_id)

        lock.release.assert_awaited_once_with()
        redis.aclose.assert_awaited_once_with()

    async def test_lost_ownership_does_not_delete_new_owner_or_fail_completed_run(self) -> None:
        task_id = str(uuid.uuid4())
        redis, lock = self._redis_with_lock(release_error=LockNotOwnedError("lease replaced"))

        with (
            patch("app.tasks.agent_task.aioredis.from_url", return_value=redis),
            patch("app.tasks.agent_task._do_run", new_callable=AsyncMock) as do_run,
            patch("app.tasks.agent_task.logger") as logger,
        ):
            await _run_task(task_id)

        do_run.assert_awaited_once()
        lock.release.assert_awaited_once_with()
        redis.aclose.assert_awaited_once_with()
        self.assertIn("status=lock_lost", logger.warning.call_args.args[0])

    async def test_release_error_is_observable_and_does_not_fail_completed_run(self) -> None:
        task_id = str(uuid.uuid4())
        redis, lock = self._redis_with_lock(release_error=RuntimeError("release unavailable"))

        with (
            patch("app.tasks.agent_task.aioredis.from_url", return_value=redis),
            patch("app.tasks.agent_task._do_run", new_callable=AsyncMock) as do_run,
            patch("app.tasks.agent_task.logger") as logger,
        ):
            await _run_task(task_id)

        do_run.assert_awaited_once()
        self.assertIn("status=release_failed", logger.warning.call_args.args[0])
        redis.aclose.assert_awaited_once_with()

    async def test_cancellation_is_not_swallowed_and_still_releases_owned_lease(self) -> None:
        task_id = str(uuid.uuid4())
        redis, lock = self._redis_with_lock()

        with (
            patch("app.tasks.agent_task.aioredis.from_url", return_value=redis),
            patch(
                "app.tasks.agent_task._do_run",
                new_callable=AsyncMock,
                side_effect=asyncio.CancelledError(),
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await _run_task(task_id)

        lock.release.assert_awaited_once_with()
        redis.aclose.assert_awaited_once_with()

    async def test_different_tasks_use_independent_lock_keys(self) -> None:
        first_id = str(uuid.uuid4())
        second_id = str(uuid.uuid4())
        first_redis, _ = self._redis_with_lock()
        second_redis, _ = self._redis_with_lock()

        with (
            patch(
                "app.tasks.agent_task.aioredis.from_url",
                side_effect=[first_redis, second_redis],
            ),
            patch("app.tasks.agent_task._do_run", new_callable=AsyncMock),
        ):
            await _run_task(first_id)
            await _run_task(second_id)

        self.assertEqual(first_redis.lock.call_args.args[0], f"agent_task:lock:{first_id}")
        self.assertEqual(second_redis.lock.call_args.args[0], f"agent_task:lock:{second_id}")


@unittest.skipUnless(TEST_REDIS_URL, "需要设置 COMET_TEST_REDIS_URL 才运行 Redis 集成测试")
class RedisLockOwnershipIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.first_redis = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
        self.second_redis = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
        self.lock_key = f"comet:test:agent-task-lock:{uuid.uuid4()}"

    async def asyncTearDown(self) -> None:
        await self.first_redis.delete(self.lock_key)
        await self.first_redis.aclose()
        await self.second_redis.aclose()

    async def test_old_owner_cannot_release_reacquired_lock(self) -> None:
        first_lock = self.first_redis.lock(self.lock_key, timeout=30, blocking=False)
        second_lock = self.second_redis.lock(self.lock_key, timeout=30, blocking=False)

        self.assertTrue(await first_lock.acquire())
        first_token = await self.first_redis.get(self.lock_key)
        self.assertIsNotNone(first_token)
        self.assertGreater(await self.first_redis.ttl(self.lock_key), 0)
        self.assertFalse(await second_lock.acquire())

        # 确定性模拟首个租约过期，再由另一个 worker 获取同名锁。
        await self.first_redis.delete(self.lock_key)
        self.assertTrue(await second_lock.acquire())
        second_token = await self.second_redis.get(self.lock_key)
        self.assertIsNotNone(second_token)
        self.assertNotEqual(second_token, first_token)
        self.assertGreater(await self.second_redis.ttl(self.lock_key), 0)

        with self.assertRaises(LockNotOwnedError):
            await first_lock.release()

        self.assertEqual(await self.first_redis.get(self.lock_key), second_token)
        self.assertTrue(await second_lock.owned())
        await second_lock.release()


if __name__ == "__main__":
    unittest.main()

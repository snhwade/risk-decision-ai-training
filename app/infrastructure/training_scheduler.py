"""后台定时调度：按 cron 自动触发 AI 训练。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from app.domain.training_schedule import TrainingScheduleService

logger = logging.getLogger("ai_training.scheduler")


class TrainingScheduler:
    """每分钟检查一次 enabled 计划，命中 cron 则触发训练。"""

    def __init__(
        self,
        schedule_service_factory: Callable[[], TrainingScheduleService],
        *,
        poll_seconds: int = 60,
    ) -> None:
        self._factory = schedule_service_factory
        self._poll_seconds = max(15, poll_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ai-training-scheduler", daemon=True)
        self._thread.start()
        logger.info("AI 训练调度器已启动，轮询间隔 %ss", self._poll_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self._poll_seconds + 5)
            self._thread = None
        logger.info("AI 训练调度器已停止")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                service = self._factory()
                outcomes = service.tick()
                for sid, outcome in outcomes:
                    logger.info("定时训练计划 id=%s 触发结果=%s", sid, outcome)
            except Exception as exc:  # noqa: BLE001
                logger.exception("定时训练调度异常: %s", exc)
            self._stop.wait(self._poll_seconds)

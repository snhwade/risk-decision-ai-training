"""AI 训练定时计划领域逻辑。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol

from app.domain.training_job import TrainingJob, TrainingJobRepository, TrainingService

CRON_PATTERN = re.compile(
    r"^(\S+\s+\S+\s+\S+\s+\S+\s+\S+|\S+\s+\S+\s+\S+\s+\S+)$"
)


class ScheduleRunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class TrainingSchedule:
    id: int | None
    name: str
    enabled: bool
    cron_expression: str
    window_days: int
    last_triggered_at: datetime | None = None
    last_job_id: str | None = None
    last_run_status: str | None = None
    last_fail_reason: str | None = None


class TrainingScheduleRepository(Protocol):
    def save(self, schedule: TrainingSchedule) -> TrainingSchedule:
        ...

    def find_by_id(self, schedule_id: int) -> TrainingSchedule | None:
        ...

    def delete(self, schedule_id: int) -> bool:
        ...

    def list_all(self) -> list[TrainingSchedule]:
        ...

    def list_enabled(self) -> list[TrainingSchedule]:
        ...

    def update_run_result(
        self,
        schedule_id: int,
        *,
        triggered_at: datetime,
        job_id: str | None,
        status: str,
        fail_reason: str | None = None,
    ) -> None:
        ...


def validate_cron_expression(expr: str) -> str:
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Cron 表达式不能为空")
    if not CRON_PATTERN.match(expr):
        raise ValueError("Cron 表达式格式无效，应为 5 段式，如 0 2 * * *")
    try:
        from croniter import croniter

        croniter(expr)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Cron 表达式无效：{exc}") from exc
    return expr


def validate_window_days(days: int) -> int:
    if days < 1 or days > 365:
        raise ValueError("数据窗口天数须在 1–365 之间")
    return days


def compute_data_window(window_days: int, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """计算滑动窗口 [data_from, data_to]（UTC，data_to 为当前时刻）。"""
    end = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    start = end - timedelta(days=window_days)
    return start, end


def cron_matches_minute(cron_expression: str, moment: datetime) -> bool:
    """判断 moment 所在分钟是否为 cron 触发点。"""
    from croniter import croniter

    minute_start = moment.replace(second=0, microsecond=0)
    itr = croniter(cron_expression, minute_start - timedelta(minutes=1))
    next_fire = itr.get_next(datetime)
    return next_fire == minute_start


class TrainingScheduleService:
    def __init__(
        self,
        schedule_repo: TrainingScheduleRepository,
        job_repo: TrainingJobRepository,
        training_service: TrainingService,
        *,
        now_provider=None,
    ) -> None:
        self._schedules = schedule_repo
        self._jobs = job_repo
        self._training = training_service
        self._now = now_provider or (lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    def create(
        self,
        name: str,
        cron_expression: str,
        window_days: int,
        *,
        enabled: bool = True,
    ) -> TrainingSchedule:
        name = (name or "").strip()
        if not name:
            raise ValueError("计划名称不能为空")
        cron_expression = validate_cron_expression(cron_expression)
        window_days = validate_window_days(window_days)
        schedule = TrainingSchedule(
            id=None,
            name=name,
            enabled=enabled,
            cron_expression=cron_expression,
            window_days=window_days,
        )
        return self._schedules.save(schedule)

    def update(
        self,
        schedule_id: int,
        *,
        name: str | None = None,
        cron_expression: str | None = None,
        window_days: int | None = None,
        enabled: bool | None = None,
    ) -> TrainingSchedule:
        existing = self._schedules.find_by_id(schedule_id)
        if existing is None:
            raise ValueError(f"训练计划不存在: {schedule_id}")
        updated = TrainingSchedule(
            id=existing.id,
            name=(name or existing.name).strip(),
            enabled=existing.enabled if enabled is None else enabled,
            cron_expression=validate_cron_expression(cron_expression)
            if cron_expression is not None
            else existing.cron_expression,
            window_days=validate_window_days(window_days)
            if window_days is not None
            else existing.window_days,
            last_triggered_at=existing.last_triggered_at,
            last_job_id=existing.last_job_id,
            last_run_status=existing.last_run_status,
            last_fail_reason=existing.last_fail_reason,
        )
        if not updated.name:
            raise ValueError("计划名称不能为空")
        return self._schedules.save(updated)

    def delete(self, schedule_id: int) -> None:
        if not self._schedules.delete(schedule_id):
            raise ValueError(f"训练计划不存在: {schedule_id}")

    def list_all(self) -> list[TrainingSchedule]:
        return self._schedules.list_all()

    def run_now(self, schedule_id: int) -> TrainingJob | ScheduleRunStatus:
        schedule = self._schedules.find_by_id(schedule_id)
        if schedule is None:
            raise ValueError(f"训练计划不存在: {schedule_id}")
        return self._execute_schedule(schedule, manual=True)

    def tick(self, moment: datetime | None = None) -> list[tuple[int, str]]:
        """调度器每分钟调用：对匹配的 enabled 计划触发训练。返回 [(schedule_id, outcome)]."""
        now = moment or self._now()
        outcomes: list[tuple[int, str]] = []
        for schedule in self._schedules.list_enabled():
            if schedule.id is None:
                continue
            if schedule.last_triggered_at and schedule.last_triggered_at.replace(
                second=0, microsecond=0
            ) == now.replace(second=0, microsecond=0):
                continue
            if not cron_matches_minute(schedule.cron_expression, now):
                continue
            result = self._execute_schedule(schedule, manual=False)
            outcome = result.status.value if isinstance(result, TrainingJob) else str(result.value)
            outcomes.append((schedule.id, outcome))
        return outcomes

    def _execute_schedule(self, schedule: TrainingSchedule, *, manual: bool):
        now = self._now()
        if schedule.id is None:
            raise ValueError("计划 ID 无效")

        if self._jobs.has_running_job():
            reason = "已有训练任务 RUNNING，跳过本次触发"
            self._schedules.update_run_result(
                schedule.id,
                triggered_at=now,
                job_id=None,
                status=ScheduleRunStatus.SKIPPED.value,
                fail_reason=reason,
            )
            return ScheduleRunStatus.SKIPPED

        data_from, data_to = compute_data_window(schedule.window_days, now=now)
        job = self._training.execute(data_from, data_to)
        status = (
            ScheduleRunStatus.SUCCESS.value
            if job.status.value == "SUCCESS"
            else ScheduleRunStatus.FAILED.value
        )
        self._schedules.update_run_result(
            schedule.id,
            triggered_at=now,
            job_id=job.job_id,
            status=status,
            fail_reason=job.fail_reason if status == ScheduleRunStatus.FAILED.value else None,
        )
        return job

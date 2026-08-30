"""AI 训练定时计划控制器（与 Web 框架解耦）。"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.adapter.schemas import (
    CreateTrainingScheduleRequest,
    UpdateTrainingScheduleRequest,
    training_job_to_dict,
    training_schedule_to_dict,
    validation_error_body,
    validation_error_body_from_message,
)
from app.domain.training_job import TrainingJob
from app.domain.training_schedule import ScheduleRunStatus, TrainingScheduleService


class TrainingScheduleController:
    def __init__(self, service: TrainingScheduleService) -> None:
        self._service = service

    def list_schedules(self) -> tuple[int, dict]:
        schedules = self._service.list_all()
        return 200, {"data": [training_schedule_to_dict(s) for s in schedules]}

    def create(self, body: Any) -> tuple[int, dict]:
        try:
            request = CreateTrainingScheduleRequest.model_validate(body)
        except ValidationError as exc:
            return 400, validation_error_body(exc)
        try:
            schedule = self._service.create(
                request.name,
                request.cron_expression,
                request.window_days,
                enabled=request.enabled,
            )
        except ValueError as exc:
            return 400, validation_error_body_from_message(str(exc), field="schedule")
        return 201, training_schedule_to_dict(schedule)

    def update(self, schedule_id: int, body: Any) -> tuple[int, dict]:
        try:
            request = UpdateTrainingScheduleRequest.model_validate(body)
        except ValidationError as exc:
            return 400, validation_error_body(exc)
        try:
            schedule = self._service.update(
                schedule_id,
                name=request.name,
                cron_expression=request.cron_expression,
                window_days=request.window_days,
                enabled=request.enabled,
            )
        except ValueError as exc:
            msg = str(exc)
            if "不存在" in msg:
                return 404, {"code": "NOT_FOUND", "message": msg}
            return 400, validation_error_body_from_message(msg, field="schedule")
        return 200, training_schedule_to_dict(schedule)

    def delete(self, schedule_id: int) -> tuple[int, dict]:
        try:
            self._service.delete(schedule_id)
        except ValueError as exc:
            return 404, {"code": "NOT_FOUND", "message": str(exc)}
        return 200, {"deleted": True, "id": schedule_id}

    def run_now(self, schedule_id: int) -> tuple[int, dict]:
        try:
            result = self._service.run_now(schedule_id)
        except ValueError as exc:
            return 404, {"code": "NOT_FOUND", "message": str(exc)}
        if isinstance(result, TrainingJob):
            return 200, {
                "outcome": "TRIGGERED",
                "job": training_job_to_dict(result),
            }
        return 200, {
            "outcome": ScheduleRunStatus.SKIPPED.value,
            "reason": "已有训练任务 RUNNING，跳过本次触发",
        }

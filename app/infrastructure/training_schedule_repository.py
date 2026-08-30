"""AI 训练定时计划持久化。"""

from __future__ import annotations

from typing import Callable

from app.domain.training_schedule import TrainingSchedule, TrainingScheduleRepository


class InMemoryTrainingScheduleRepository:
    def __init__(self) -> None:
        self._rows: dict[int, TrainingSchedule] = {}
        self._next_id = 1

    def save(self, schedule: TrainingSchedule) -> TrainingSchedule:
        if schedule.id is None:
            sid = self._next_id
            self._next_id += 1
            stored = TrainingSchedule(
                id=sid,
                name=schedule.name,
                enabled=schedule.enabled,
                cron_expression=schedule.cron_expression,
                window_days=schedule.window_days,
                last_triggered_at=schedule.last_triggered_at,
                last_job_id=schedule.last_job_id,
                last_run_status=schedule.last_run_status,
                last_fail_reason=schedule.last_fail_reason,
            )
        else:
            stored = schedule
        self._rows[stored.id] = stored
        return stored

    def find_by_id(self, schedule_id: int) -> TrainingSchedule | None:
        return self._rows.get(schedule_id)

    def delete(self, schedule_id: int) -> bool:
        return self._rows.pop(schedule_id, None) is not None

    def list_all(self) -> list[TrainingSchedule]:
        return sorted(self._rows.values(), key=lambda s: s.id or 0, reverse=True)

    def list_enabled(self) -> list[TrainingSchedule]:
        return [s for s in self.list_all() if s.enabled]

    def update_run_result(
        self,
        schedule_id: int,
        *,
        triggered_at,
        job_id: str | None,
        status: str,
        fail_reason: str | None = None,
    ) -> None:
        row = self._rows.get(schedule_id)
        if row is None:
            return
        self._rows[schedule_id] = TrainingSchedule(
            id=row.id,
            name=row.name,
            enabled=row.enabled,
            cron_expression=row.cron_expression,
            window_days=row.window_days,
            last_triggered_at=triggered_at,
            last_job_id=job_id,
            last_run_status=status,
            last_fail_reason=fail_reason,
        )


class SqlAlchemyTrainingScheduleRepository:
    def __init__(self, engine_factory: Callable[[], object]) -> None:
        self._engine_factory = engine_factory

    def save(self, schedule: TrainingSchedule) -> TrainingSchedule:
        from sqlalchemy import text

        engine = self._engine_factory()
        if schedule.id is None:
            stmt = text(
                "INSERT INTO ai_training_schedule "
                "(name, enabled, cron_expression, window_days) "
                "VALUES (:name, :enabled, :cron, :window_days)"
            )
            params = {
                "name": schedule.name,
                "enabled": 1 if schedule.enabled else 0,
                "cron": schedule.cron_expression,
                "window_days": schedule.window_days,
            }
            with engine.begin() as conn:  # type: ignore[attr-defined]
                result = conn.execute(stmt, params)
                new_id = result.lastrowid
            return TrainingSchedule(
                id=int(new_id),
                name=schedule.name,
                enabled=schedule.enabled,
                cron_expression=schedule.cron_expression,
                window_days=schedule.window_days,
            )

        stmt = text(
            "UPDATE ai_training_schedule SET "
            "name=:name, enabled=:enabled, cron_expression=:cron, window_days=:window_days "
            "WHERE id=:id"
        )
        params = {
            "id": schedule.id,
            "name": schedule.name,
            "enabled": 1 if schedule.enabled else 0,
            "cron": schedule.cron_expression,
            "window_days": schedule.window_days,
        }
        with engine.begin() as conn:  # type: ignore[attr-defined]
            conn.execute(stmt, params)
        return schedule

    def find_by_id(self, schedule_id: int) -> TrainingSchedule | None:
        from sqlalchemy import text

        engine = self._engine_factory()
        stmt = text("SELECT * FROM ai_training_schedule WHERE id=:id")
        with engine.connect() as conn:  # type: ignore[attr-defined]
            row = conn.execute(stmt, {"id": schedule_id}).mappings().first()
        return _row_to_schedule(row) if row else None

    def delete(self, schedule_id: int) -> bool:
        from sqlalchemy import text

        engine = self._engine_factory()
        stmt = text("DELETE FROM ai_training_schedule WHERE id=:id")
        with engine.begin() as conn:  # type: ignore[attr-defined]
            result = conn.execute(stmt, {"id": schedule_id})
            return result.rowcount > 0

    def list_all(self) -> list[TrainingSchedule]:
        from sqlalchemy import text

        engine = self._engine_factory()
        stmt = text("SELECT * FROM ai_training_schedule ORDER BY id DESC")
        with engine.connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(stmt).mappings().all()
        return [_row_to_schedule(r) for r in rows]

    def list_enabled(self) -> list[TrainingSchedule]:
        return [s for s in self.list_all() if s.enabled]

    def update_run_result(
        self,
        schedule_id: int,
        *,
        triggered_at,
        job_id: str | None,
        status: str,
        fail_reason: str | None = None,
    ) -> None:
        from sqlalchemy import text

        engine = self._engine_factory()
        stmt = text(
            "UPDATE ai_training_schedule SET "
            "last_triggered_at=:triggered_at, last_job_id=:job_id, "
            "last_run_status=:status, last_fail_reason=:fail_reason "
            "WHERE id=:id"
        )
        params = {
            "id": schedule_id,
            "triggered_at": triggered_at,
            "job_id": job_id,
            "status": status,
            "fail_reason": fail_reason,
        }
        with engine.begin() as conn:  # type: ignore[attr-defined]
            conn.execute(stmt, params)


def _row_to_schedule(row) -> TrainingSchedule:
    return TrainingSchedule(
        id=int(row["id"]),
        name=row["name"],
        enabled=bool(row["enabled"]),
        cron_expression=row["cron_expression"],
        window_days=int(row["window_days"]),
        last_triggered_at=row.get("last_triggered_at"),
        last_job_id=row.get("last_job_id"),
        last_run_status=row.get("last_run_status"),
        last_fail_reason=row.get("last_fail_reason"),
    )

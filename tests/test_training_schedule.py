"""AI 训练定时计划领域与 REST 测试。"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapter.schedule_controller import TrainingScheduleController  # noqa: E402
from app.domain.training_job import (  # noqa: E402
    TrainingJob,
    TrainingJobStatus,
    TrainingResult,
    TrainingService,
)
from app.domain.training_schedule import (  # noqa: E402
    ScheduleRunStatus,
    TrainingScheduleService,
    compute_data_window,
    cron_matches_minute,
)
from app.infrastructure.training_job_repository import InMemoryTrainingJobRepository  # noqa: E402
from app.infrastructure.training_schedule_repository import (  # noqa: E402
    InMemoryTrainingScheduleRepository,
)


class _ListSampleSource:
    def __init__(self, count: int) -> None:
        self._samples = [{"merchant_id": f"m{i}"} for i in range(count)]

    def read_range(self, data_from, data_to):
        return self._samples


class _StubTrainer:
    def train(self, samples):
        return TrainingResult(model_version="v-sched-1", metrics={"n": len(samples)})


class _RecordingAlarm:
    def alarm(self, title: str, detail: str) -> None:
        pass


def _build_schedule_service(*, sample_count: int = 20, min_samples: int = 5):
    job_repo = InMemoryTrainingJobRepository()
    schedule_repo = InMemoryTrainingScheduleRepository()
    training = TrainingService(
        sample_source=_ListSampleSource(sample_count),
        trainer=_StubTrainer(),
        repository=job_repo,
        alarm=_RecordingAlarm(),
        min_training_samples=min_samples,
    )
    schedule = TrainingScheduleService(
        schedule_repo=schedule_repo,
        job_repo=job_repo,
        training_service=training,
        now_provider=lambda: datetime(2026, 7, 4, 2, 0, 0),
    )
    return schedule, schedule_repo, job_repo


class TrainingScheduleDomainTest(unittest.TestCase):
    def test_compute_data_window(self):
        now = datetime(2026, 7, 4, 12, 0, 0)
        start, end = compute_data_window(30, now=now)
        self.assertEqual(end, now)
        self.assertEqual((end - start).days, 30)

    def test_cron_matches_minute(self):
        moment = datetime(2026, 7, 4, 2, 0, 0)
        self.assertTrue(cron_matches_minute("0 2 * * *", moment))
        self.assertFalse(cron_matches_minute("0 3 * * *", moment))

    def test_tick_triggers_training_on_cron_match(self):
        service, schedule_repo, job_repo = _build_schedule_service(sample_count=20)
        created = service.create("daily", "0 2 * * *", 7)
        outcomes = service.tick(datetime(2026, 7, 4, 2, 0, 0))
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0][1], TrainingJobStatus.SUCCESS.value)
        stored = schedule_repo.find_by_id(created.id)
        assert stored is not None
        self.assertEqual(stored.last_run_status, ScheduleRunStatus.SUCCESS.value)
        self.assertEqual(len(job_repo.list_all()), 1)

    def test_tick_skips_when_running_job_exists(self):
        service, schedule_repo, _job_repo = _build_schedule_service(sample_count=20)
        created = service.create("daily", "0 2 * * *", 7)
        running = TrainingJob(
            job_id="running-1",
            data_from=datetime(2026, 6, 1),
            data_to=datetime(2026, 7, 1),
            status=TrainingJobStatus.RUNNING,
            started_at=datetime(2026, 7, 4, 1, 0, 0),
        )
        _job_repo.save(running)
        outcomes = service.tick(datetime(2026, 7, 4, 2, 0, 0))
        self.assertEqual(outcomes[0][1], ScheduleRunStatus.SKIPPED.value)
        stored = schedule_repo.find_by_id(created.id)
        assert stored is not None
        self.assertEqual(stored.last_run_status, ScheduleRunStatus.SKIPPED.value)


class TrainingScheduleControllerTest(unittest.TestCase):
    def test_crud_and_run_now(self):
        service, _, _ = _build_schedule_service(sample_count=20)
        controller = TrainingScheduleController(service)

        status, body = controller.create(
            {"name": "nightly", "cronExpression": "0 2 * * *", "windowDays": 14}
        )
        self.assertEqual(status, 201)
        schedule_id = body["id"]

        status, listed = controller.list_schedules()
        self.assertEqual(status, 200)
        self.assertEqual(len(listed["data"]), 1)

        status, updated = controller.update(
            schedule_id, {"enabled": False, "windowDays": 21}
        )
        self.assertEqual(status, 200)
        self.assertFalse(updated["enabled"])
        self.assertEqual(updated["windowDays"], 21)

        status, run = controller.run_now(schedule_id)
        self.assertEqual(status, 200)
        self.assertEqual(run["outcome"], "TRIGGERED")
        self.assertEqual(run["job"]["status"], TrainingJobStatus.SUCCESS.value)

        status, deleted = controller.delete(schedule_id)
        self.assertEqual(status, 200)
        self.assertTrue(deleted["deleted"])

    def test_invalid_cron_returns_400(self):
        service, _, _ = _build_schedule_service()
        controller = TrainingScheduleController(service)
        status, body = controller.create(
            {"name": "bad", "cronExpression": "invalid cron", "windowDays": 7}
        )
        self.assertEqual(status, 400)
        self.assertIn("schedule", body["fields"])


if __name__ == "__main__":
    unittest.main()

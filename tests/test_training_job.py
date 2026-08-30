"""训练任务与样本/超时控制单元测试（R13.1/13.3/13.6/13.7）。

使用标准库 unittest + 内存替身，无需真实数据库/训练算法：
    python -m unittest discover -s tests
（亦兼容 pytest 运行。）

覆盖：
- 样本不足拒绝（R13.6），且不进入训练、不告警
- 样本充足训练成功，记录数据范围/模型版本/评估指标（R13.3）
- 训练超时终止、记录失败原因并告警、不写指标（R13.7）
- 训练异常终止、记录失败原因并告警、不写指标（R13.7）
- 最小样本量/最长训练时长取值范围边界与越界（R13.6/13.7）
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from datetime import datetime

# 确保可导入 app 包（tests 与 app 同级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.domain.training_job import (  # noqa: E402
    MAX_SECONDS_DEFAULT,
    MIN_SAMPLES_DEFAULT,
    TrainingFailureKind,
    TrainingJobStatus,
    TrainingResult,
    TrainingService,
    validate_max_training_seconds,
    validate_min_training_samples,
)
from app.infrastructure.training_job_repository import (  # noqa: E402
    InMemoryTrainingJobRepository,
)


# ---- 测试替身 ----------------------------------------------------------------


class _ListSampleSource:
    """以预置样本数量构造样本来源，无视时间范围返回固定样本列表。"""

    def __init__(self, count: int) -> None:
        self._samples = [{"i": i} for i in range(count)]

    def read_range(self, data_from, data_to):
        return self._samples


class _StubTrainer:
    """成功训练替身：返回固定模型版本与评估指标，并记录是否被调用。"""

    def __init__(self) -> None:
        self.called = False

    def train(self, samples):
        self.called = True
        return TrainingResult(model_version="v-20240101", metrics={"auc": 0.91, "n": len(samples)})


class _ExplodingTrainer:
    """异常训练替身：训练时抛出异常。"""

    def train(self, samples):
        raise RuntimeError("特征矩阵异常")


class _SleepingTrainer:
    """耗时训练替身：训练时阻塞，用于触发超时。"""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def train(self, samples):
        time.sleep(self._seconds)
        return TrainingResult(model_version="v-slow", metrics={})


class _RecordingAlarm:
    """记录告警调用的替身。"""

    def __init__(self) -> None:
        self.alarms: list[tuple[str, str]] = []

    def alarm(self, title: str, detail: str) -> None:
        self.alarms.append((title, detail))


_FROM = datetime(2024, 1, 1)
_TO = datetime(2024, 1, 31)


def _build_service(*, sample_count, trainer=None, min_samples=10, max_seconds=MAX_SECONDS_DEFAULT):
    repo = InMemoryTrainingJobRepository()
    alarm = _RecordingAlarm()
    service = TrainingService(
        sample_source=_ListSampleSource(sample_count),
        trainer=trainer or _StubTrainer(),
        repository=repo,
        alarm=alarm,
        min_training_samples=min_samples,
        max_training_seconds=max_seconds,
    )
    return service, repo, alarm


class TrainingServiceTest(unittest.TestCase):
    def test_insufficient_samples_rejected_without_training_or_alarm(self):
        # R13.6：样本量 < 最小阈值，拒绝、记录原因、不进入训练、不告警
        trainer = _StubTrainer()
        service, repo, alarm = _build_service(sample_count=5, trainer=trainer, min_samples=10)
        job = service.execute(_FROM, _TO)

        self.assertEqual(job.status, TrainingJobStatus.FAILED)
        self.assertEqual(job.failure_kind, TrainingFailureKind.INSUFFICIENT_SAMPLES)
        self.assertIn("训练样本不足", job.fail_reason)
        self.assertFalse(trainer.called, "样本不足时不应进入训练")
        self.assertEqual(alarm.alarms, [], "样本不足是业务拒绝，不应触发告警")
        # 任务记录已持久化，供任务列表/前端展示
        self.assertIsNotNone(repo.find(job.job_id))

    def test_sufficient_samples_success_records_range_version_metrics(self):
        # R13.3：成功记录数据范围/模型版本/评估指标
        service, repo, alarm = _build_service(sample_count=20, min_samples=10)
        job = service.execute(_FROM, _TO)

        self.assertEqual(job.status, TrainingJobStatus.SUCCESS)
        self.assertEqual(job.data_from, _FROM)
        self.assertEqual(job.data_to, _TO)
        self.assertEqual(job.model_version, "v-20240101")
        self.assertEqual(job.metrics["auc"], 0.91)
        self.assertIsNone(job.fail_reason)
        self.assertEqual(alarm.alarms, [])

    def test_sample_count_equal_threshold_is_sufficient(self):
        # 边界：样本量恰好等于阈值视为充足（< 阈值才拒绝）
        service, _repo, _alarm = _build_service(sample_count=10, min_samples=10)
        job = service.execute(_FROM, _TO)
        self.assertEqual(job.status, TrainingJobStatus.SUCCESS)

    def test_training_exception_terminates_records_and_alarms(self):
        # R13.7：训练异常终止、记录失败原因并告警、不写指标
        service, _repo, alarm = _build_service(
            sample_count=20, trainer=_ExplodingTrainer(), min_samples=10
        )
        job = service.execute(_FROM, _TO)

        self.assertEqual(job.status, TrainingJobStatus.FAILED)
        self.assertEqual(job.failure_kind, TrainingFailureKind.EXCEPTION)
        self.assertIn("训练异常", job.fail_reason)
        self.assertIsNone(job.model_version, "失败任务不应记录模型版本")
        self.assertEqual(len(alarm.alarms), 1)

    def test_training_timeout_terminates_records_and_alarms(self):
        # R13.7：超过最长训练时长则终止、记录失败原因并告警、不写指标
        # 配置取值范围最小为 60 秒，为避免测试缓慢，注入将超时封顶为极小值的保护器，
        # 训练替身会阻塞 2 秒，必然触发超时。
        repo = InMemoryTrainingJobRepository()
        alarm = _RecordingAlarm()
        service = TrainingService(
            sample_source=_ListSampleSource(20),
            trainer=_SleepingTrainer(seconds=2.0),
            repository=repo,
            alarm=alarm,
            min_training_samples=10,
            max_training_seconds=60,
            timeout_guard=_ShortTimeoutGuard(0.2),
        )
        job = service.execute(_FROM, _TO)

        self.assertEqual(job.status, TrainingJobStatus.FAILED)
        self.assertEqual(job.failure_kind, TrainingFailureKind.TIMEOUT)
        self.assertIn("超时", job.fail_reason)
        self.assertIsNone(job.model_version)
        self.assertEqual(len(alarm.alarms), 1)


class _ShortTimeoutGuard:
    """将超时秒数封顶为很小值的保护器，便于快速触发超时（仅测试用）。"""

    def __init__(self, cap_seconds: float) -> None:
        from app.infrastructure.timeout_guard import ThreadTimeoutGuard

        self._cap = cap_seconds
        self._delegate = ThreadTimeoutGuard()

    def run(self, func, timeout_seconds):
        return self._delegate.run(func, min(self._cap, timeout_seconds))


class ConfigRangeValidationTest(unittest.TestCase):
    def test_min_samples_bounds(self):
        self.assertEqual(validate_min_training_samples(1), 1)
        self.assertEqual(validate_min_training_samples(1_000_000), 1_000_000)
        self.assertEqual(validate_min_training_samples(MIN_SAMPLES_DEFAULT), 1000)
        with self.assertRaises(ValueError):
            validate_min_training_samples(0)
        with self.assertRaises(ValueError):
            validate_min_training_samples(1_000_001)

    def test_max_seconds_bounds(self):
        self.assertEqual(validate_max_training_seconds(60), 60)
        self.assertEqual(validate_max_training_seconds(86_400), 86_400)
        self.assertEqual(validate_max_training_seconds(MAX_SECONDS_DEFAULT), 3600)
        with self.assertRaises(ValueError):
            validate_max_training_seconds(59)
        with self.assertRaises(ValueError):
            validate_max_training_seconds(86_401)

    def test_service_construction_rejects_out_of_range_config(self):
        with self.assertRaises(ValueError):
            _build_service(sample_count=1, min_samples=0)


# ---- 任务 17.3：交易对手指标提取与写入集成（R13.2/13.5/13.8）------------------


class _StaticExtractor:
    """返回预置指标列表的提取器替身。"""

    def __init__(self, metrics) -> None:
        self._metrics = metrics

    def extract(self, samples):
        return self._metrics


class _ExplodingExtractor:
    """提取时抛异常的替身（验证提取失败不影响训练成功）。"""

    def extract(self, samples):
        raise RuntimeError("关系图构建失败")


class _CollectingWriter:
    """记录写入指标的替身。"""

    def __init__(self) -> None:
        self.written = []

    def write(self, metric) -> None:
        self.written.append(metric)


class _AlwaysFailingWriter:
    """始终写入失败的替身。"""

    def write(self, metric) -> None:
        raise RuntimeError("指标存储不可用")


def _sample_metrics():
    from app.domain.counterparty import CounterpartyMetric

    return [
        CounterpartyMetric("ai_cp_degree", "A", 2.0, 1_700_000_000),
        CounterpartyMetric("ai_cp_txn_count", "A", 3.0, 1_700_000_000),
    ]


class TrainingServiceIndicatorWriteTest(unittest.TestCase):
    def _build(self, *, extractor=None, writer=None, max_retries=3):
        repo = InMemoryTrainingJobRepository()
        alarm = _RecordingAlarm()
        service = TrainingService(
            sample_source=_ListSampleSource(20),
            trainer=_StubTrainer(),
            repository=repo,
            alarm=alarm,
            min_training_samples=10,
            counterparty_extractor=extractor,
            indicator_writer=writer,
            indicator_write_max_retries=max_retries,
        )
        return service, repo, alarm

    def test_success_extracts_and_writes_indicators(self):
        # R13.2：训练成功后提取并写入交易对手关系指标
        writer = _CollectingWriter()
        service, _repo, alarm = self._build(
            extractor=_StaticExtractor(_sample_metrics()), writer=writer
        )
        job = service.execute(_FROM, _TO)

        self.assertEqual(job.status, TrainingJobStatus.SUCCESS)
        self.assertEqual(job.indicator_metrics_written, 2)
        self.assertEqual(job.indicator_write_failed, 0)
        self.assertEqual(len(writer.written), 2)
        self.assertEqual(alarm.alarms, [], "全部写入成功不应告警")

    def test_success_without_ai_ports_writes_nothing(self):
        # R13.4：未启用 AI 指标写入端口时，训练仍成功且不写指标、不告警
        service, _repo, alarm = self._build(extractor=None, writer=None)
        job = service.execute(_FROM, _TO)
        self.assertEqual(job.status, TrainingJobStatus.SUCCESS)
        self.assertEqual(job.indicator_metrics_written, 0)
        self.assertEqual(alarm.alarms, [])

    def test_write_exhausts_retries_records_and_alarms_but_job_success(self):
        # R13.8：写入耗尽重试仍失败 → 记录失败 + 告警，但训练任务仍记为成功
        service, _repo, alarm = self._build(
            extractor=_StaticExtractor(_sample_metrics()),
            writer=_AlwaysFailingWriter(),
            max_retries=3,
        )
        job = service.execute(_FROM, _TO)

        self.assertEqual(job.status, TrainingJobStatus.SUCCESS, "写入失败不应影响训练成功")
        self.assertEqual(job.indicator_metrics_written, 0)
        self.assertEqual(job.indicator_write_failed, 2)
        self.assertEqual(len(alarm.alarms), 1)
        self.assertIn("写入失败", alarm.alarms[0][0])

    def test_extraction_failure_records_and_alarms_but_job_success(self):
        # 提取异常属增强链路异常：告警但不影响训练成功
        service, _repo, alarm = self._build(
            extractor=_ExplodingExtractor(), writer=_CollectingWriter()
        )
        job = service.execute(_FROM, _TO)
        self.assertEqual(job.status, TrainingJobStatus.SUCCESS)
        self.assertEqual(job.indicator_metrics_written, 0)
        self.assertEqual(len(alarm.alarms), 1)

    def test_out_of_range_write_retries_rejected_at_construction(self):
        # R13.8：写入重试次数取值范围 1..10，越界构造即拒绝
        with self.assertRaises(ValueError):
            self._build(max_retries=11)


if __name__ == "__main__":
    unittest.main()

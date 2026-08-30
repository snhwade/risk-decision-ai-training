"""训练任务执行与样本/超时控制（R13.1 / R13.3 / R13.6 / R13.7）。

本模块承载 AI 训练任务的核心业务编排，遵循设计文档「AI 指标写入链路时序图」：

    读取历史数据 → 样本量校验 → 训练(带超时保护) → 记录数据范围/模型版本/评估指标

业务约束：
- R13.6：可用训练样本量 < 最小训练样本量（取值范围 1..1,000,000，默认 1000）时，
  拒绝该次训练并返回「训练样本不足」错误信息（不触发告警、不进入训练）。
- R13.7：训练发生异常或超过最长训练时长（取值范围 60..86400 秒，默认 3600）时，
  终止该次训练、记录失败原因并触发告警，且不写入任何交易对手关系指标。
- R13.3：训练成功时记录所用数据时间范围、模型版本与评估指标。
- R13.2/R13.5/R13.8（任务 17.3 追加）：训练成功后基于交易对手关系图提取交易对手关系指标
  并写入指标存储；写入失败最多重试可配置次数（1..10，默认 3），仍失败记录原因并告警，
  但不影响训练任务本身记为成功，也不影响核心功能。写入后的指标可被规则引用。

DDD 分层：本模块属 domain 层，仅依赖抽象端口（Protocol），不直接依赖具体技术实现：
- SampleSource：按时间范围读取训练样本（由基础设施 OrderReader 实现）
- ModelTrainer：实际模型训练算法（可插拔）
- TimeoutGuard：超时保护执行器
- TrainingJobRepository：训练任务持久化（对应 ai_training_job 表）
- AlarmNotifier：失败告警通道
- CounterpartyGraphExtractor：交易对手关系图指标提取（任务 17.3 追加，可选注入）
- IndicatorWriter：交易对手关系指标写入指标存储（任务 17.3 追加，可选注入）

> 注：训练成功后的「交易对手关系指标提取与写入指标存储」由任务 17.3 在成功路径上追加
>     （见下方 execute() 成功分支：CounterpartyGraphExtractor + IndicatorWriter 两个新增端口）。
>     该追加为旁路增强，写入失败不影响训练任务本身记为成功（R13.8）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Protocol, Sequence, TypeVar

from app.domain.counterparty import (
    CounterpartyGraphExtractor,
    IndicatorWriter,
    IndicatorWriteOutcome,
    INDICATOR_WRITE_RETRIES_DEFAULT,
    validate_indicator_write_max_retries,
    write_metrics_with_retry,
)

T = TypeVar("T")

# ---- 可配置项取值范围（来自 R13.6 / R13.7）-------------------------------------

MIN_SAMPLES_LOWER_BOUND = 1
MIN_SAMPLES_UPPER_BOUND = 1_000_000
MIN_SAMPLES_DEFAULT = 1000

MAX_SECONDS_LOWER_BOUND = 60
MAX_SECONDS_UPPER_BOUND = 86_400
MAX_SECONDS_DEFAULT = 3600


def validate_min_training_samples(value: int) -> int:
    """校验「最小训练样本量」取值范围（R13.6：1..1,000,000）。越界抛出 ValueError。"""
    if not MIN_SAMPLES_LOWER_BOUND <= value <= MIN_SAMPLES_UPPER_BOUND:
        raise ValueError(
            f"最小训练样本量取值越界：{value}，应在 "
            f"[{MIN_SAMPLES_LOWER_BOUND}, {MIN_SAMPLES_UPPER_BOUND}] 之间"
        )
    return value


def validate_max_training_seconds(value: int) -> int:
    """校验「最长训练时长」取值范围（R13.7：60..86400 秒）。越界抛出 ValueError。"""
    if not MAX_SECONDS_LOWER_BOUND <= value <= MAX_SECONDS_UPPER_BOUND:
        raise ValueError(
            f"最长训练时长取值越界：{value} 秒，应在 "
            f"[{MAX_SECONDS_LOWER_BOUND}, {MAX_SECONDS_UPPER_BOUND}] 秒之间"
        )
    return value


# ---- 领域枚举与值对象 ----------------------------------------------------------


class TrainingJobStatus(str, Enum):
    """训练任务状态（对应 ai_training_job.status）。"""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TrainingFailureKind(str, Enum):
    """训练失败类别，便于上层（REST/前端）区分提示（R13.11）。"""

    INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"  # 样本不足被拒绝（R13.6）
    TIMEOUT = "TIMEOUT"  # 超过最长训练时长（R13.7）
    EXCEPTION = "EXCEPTION"  # 训练过程异常（R13.7）


@dataclass(frozen=True)
class TrainingResult:
    """模型训练产出：模型版本与评估指标（R13.3）。

    model：可选的已拟合模型对象（opaque）。监督式欺诈评分模型据此在成功路径产出
    ``ai_fraud_score`` 指标；占位训练器或不产出可评分模型时为 None。
    """

    model_version: str
    metrics: dict
    model: object | None = None


@dataclass
class TrainingJob:
    """训练任务记录（对应 ai_training_job 表）。"""

    job_id: str
    data_from: datetime
    data_to: datetime
    status: TrainingJobStatus
    started_at: datetime
    sample_count: int = 0
    model_version: str | None = None
    metrics: dict | None = None
    failure_kind: TrainingFailureKind | None = None
    fail_reason: str | None = None
    finished_at: datetime | None = None
    # 交易对手关系指标写入结果（R13.2/13.8，任务 17.3 追加）：
    # 成功路径写入后填充；写入失败不改变 status（仍为 SUCCESS）。
    indicator_metrics_written: int = 0
    indicator_write_failed: int = 0


# ---- 抽象端口（Protocol）------------------------------------------------------


class SampleSource(Protocol):
    """训练样本来源：按数据时间范围读取样本（OrderReader 结构上即满足该端口）。"""

    def read_range(self, data_from: datetime, data_to: datetime) -> Sequence[object]:
        ...


class ModelTrainer(Protocol):
    """模型训练算法端口：基于样本训练并返回模型版本与评估指标。"""

    def train(self, samples: Sequence[object]) -> TrainingResult:
        ...


class TimeoutGuard(Protocol):
    """超时保护执行器：在指定秒数内执行 func，超时抛出内置 TimeoutError。"""

    def run(self, func: Callable[[], T], timeout_seconds: float) -> T:
        ...


class TrainingJobRepository(Protocol):
    """训练任务持久化端口（对应 ai_training_job 表）。"""

    def save(self, job: TrainingJob) -> None:
        ...

    def list_all(self) -> "list[TrainingJob]":
        """返回全部训练任务（按开始时间倒序），供任务列表查询（R13.10）。"""
        ...

    def query(
        self,
        *,
        job_id: str | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple["list[TrainingJob]", int]:
        """分页查询训练任务（按 started_at 降序）。"""
        ...

    def has_running_job(self) -> bool:
        """是否存在 RUNNING 状态任务（定时触发防重叠）。"""
        ...


class AlarmNotifier(Protocol):
    """告警通道端口。"""

    def alarm(self, title: str, detail: str) -> None:
        ...


class FraudScorerPort(Protocol):
    """欺诈评分指标产出端口：用训练产出的模型为各商户产出 ``ai_fraud_score`` 指标。

    具体实现见 `app.domain.fraud_model.FraudScorer`；在训练成功路径注入，产出的指标
    与交易对手关系指标一并经指标写入链路写入指标存储（R13.2/13.5）。
    """

    def score(self, model: object, samples: Sequence[object]) -> Sequence[object]:
        ...


class ModelStorePort(Protocol):
    """模型存储端口（S12.2）：训练成功后将模型按版本落盘，支持多次评分与回滚。

    具体实现见 `app.infrastructure.model_repository.FileModelRepository`（joblib 落盘）。
    """

    def save(
        self, model_kind: str, version: str, model: object, metrics: dict
    ) -> object:
        ...


# ---- 训练任务应用/领域服务 ----------------------------------------------------


class TrainingService:
    """训练任务执行编排，落实样本量与超时控制（R13.1/13.3/13.6/13.7）。

    依赖通过构造函数注入，便于单元测试以内存替身替换数据库/告警/训练算法。
    """

    def __init__(
        self,
        *,
        sample_source: SampleSource,
        trainer: ModelTrainer,
        repository: TrainingJobRepository,
        alarm: AlarmNotifier,
        min_training_samples: int = MIN_SAMPLES_DEFAULT,
        max_training_seconds: int = MAX_SECONDS_DEFAULT,
        timeout_guard: TimeoutGuard | None = None,
        now_provider: Callable[[], datetime] | None = None,
        counterparty_extractor: CounterpartyGraphExtractor | None = None,
        indicator_writer: IndicatorWriter | None = None,
        indicator_write_max_retries: int = INDICATOR_WRITE_RETRIES_DEFAULT,
        fraud_scorer: "FraudScorerPort | None" = None,
        anomaly_scorer: "FraudScorerPort | None" = None,
        model_store: "ModelStorePort | None" = None,
    ) -> None:
        # 配置取值范围校验（越界即拒绝，避免误配置）
        self._min_training_samples = validate_min_training_samples(min_training_samples)
        self._max_training_seconds = validate_max_training_seconds(max_training_seconds)
        # R13.8：指标写入最大重试次数取值范围校验（1..10）
        self._indicator_write_max_retries = validate_indicator_write_max_retries(
            indicator_write_max_retries
        )
        self._sample_source = sample_source
        self._trainer = trainer
        self._repository = repository
        self._alarm = alarm
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        # 交易对手关系指标提取与写入端口（任务 17.3）：未注入时成功路径跳过指标写入，
        # 训练仍正常完成（R13.4：AI 增强能力可禁用，不影响核心功能）。
        self._counterparty_extractor = counterparty_extractor
        self._indicator_writer = indicator_writer
        # 监督式欺诈评分指标产出器（AI 增强）：注入后在训练成功路径用已训练模型产出
        # ai_fraud_score 指标，经同一指标写入链路写入指标存储。未注入则跳过。
        self._fraud_scorer = fraud_scorer
        # 无监督异常检测指标产出器（AI 增强）：注入后在训练成功路径产出 ai_anomaly_score
        # 指标（无监督，自行训练异常模型），经同一写入链路写入。未注入则跳过。
        self._anomaly_scorer = anomaly_scorer
        # 模型存储端口（S12.2）：注入后在训练成功路径将模型按版本落盘，支持训练一次多次
        # 评分与版本回滚；落盘失败仅记录+告警，不影响训练成功与核心功能。未注入则跳过。
        self._model_store = model_store
        if timeout_guard is None:
            # 延迟导入默认实现，保持 domain 层不在模块级强依赖 infrastructure
            from app.infrastructure.timeout_guard import ThreadTimeoutGuard

            timeout_guard = ThreadTimeoutGuard()
        self._timeout_guard = timeout_guard

    def execute(self, data_from: datetime, data_to: datetime) -> TrainingJob:
        """执行一次训练任务，返回最终任务记录（不以异常表达业务结果）。

        流程：读取样本 → 样本量校验(R13.6) → 训练(超时保护,R13.7) → 记录结果(R13.3)。
        无论成功或失败，任务记录均持久化，供任务列表查询与前端展示（R13.10/13.11）。
        """
        # 读取该范围历史交易数据（data_from 晚于 data_to 时 read_range 会抛 ValueError）
        samples = self._sample_source.read_range(data_from, data_to)
        sample_count = len(samples)

        job = TrainingJob(
            job_id=uuid.uuid4().hex,
            data_from=data_from,
            data_to=data_to,
            status=TrainingJobStatus.RUNNING,
            started_at=self._now_provider(),
            sample_count=sample_count,
        )

        # R13.6：样本不足则拒绝并返回错误信息（不进入训练、不告警）
        if sample_count < self._min_training_samples:
            job.status = TrainingJobStatus.FAILED
            job.failure_kind = TrainingFailureKind.INSUFFICIENT_SAMPLES
            job.fail_reason = (
                f"训练样本不足：实际 {sample_count} 条 < 最小训练样本量 "
                f"{self._min_training_samples} 条"
            )
            job.finished_at = self._now_provider()
            self._repository.save(job)
            return job

        # 训练，使用超时保护（R13.7：超过最长训练时长则终止）
        try:
            result = self._timeout_guard.run(
                lambda: self._trainer.train(samples), self._max_training_seconds
            )
        except TimeoutError:
            # R13.7：超时终止、记录失败原因并告警，不写入任何交易对手关系指标
            job.status = TrainingJobStatus.FAILED
            job.failure_kind = TrainingFailureKind.TIMEOUT
            job.fail_reason = (
                f"训练超时：超过最长训练时长 {self._max_training_seconds} 秒"
            )
            job.finished_at = self._now_provider()
            self._repository.save(job)
            self._alarm.alarm("AI 训练超时", job.fail_reason)
            return job
        except Exception as exc:  # noqa: BLE001 - 训练算法可能抛出任意异常
            # R13.7：训练异常终止、记录失败原因并告警，不写入任何交易对手关系指标
            job.status = TrainingJobStatus.FAILED
            job.failure_kind = TrainingFailureKind.EXCEPTION
            job.fail_reason = f"训练异常：{exc}"
            job.finished_at = self._now_provider()
            self._repository.save(job)
            self._alarm.alarm("AI 训练异常", job.fail_reason)
            return job

        # R13.3：成功，记录数据范围（已在 job 中）/模型版本/评估指标
        job.status = TrainingJobStatus.SUCCESS
        job.model_version = result.model_version
        job.metrics = dict(result.metrics)
        job.finished_at = self._now_provider()
        # S12.2：训练成功后将模型按版本落盘，支持训练一次多次评分与版本回滚。
        # 落盘为旁路增强，失败仅记录+告警，不改变训练成功状态、不影响核心功能。
        self._persist_model(job, result.model)
        # R13.2/13.5/13.8（任务 17.3）：训练成功后提取交易对手关系指标并写入指标存储。
        # 写入为旁路增强，失败仅记录+告警，不改变训练成功状态、不影响核心功能。
        self._extract_and_write_indicators(job, samples, result.model)
        self._repository.save(job)
        return job

    def list_jobs(
        self,
        *,
        job_id: str | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple["list[TrainingJob]", int]:
        """分页列出训练任务（按开始时间倒序，R13.10）。"""
        if hasattr(self._repository, "query"):
            return self._repository.query(
                job_id=job_id,
                status=status,
                start_time=start_time,
                end_time=end_time,
                page=page,
                page_size=page_size,
            )
        jobs = self._repository.list_all()
        filtered = _filter_jobs(jobs, job_id, status, start_time, end_time)
        total = len(filtered)
        start_idx = max(0, (page - 1) * page_size)
        end_idx = start_idx + page_size
        return filtered[start_idx:end_idx], total

    def _persist_model(self, job: TrainingJob, model: object) -> None:
        """训练成功后将模型按版本落盘（S12.2）。

        - 未注入模型存储或无可持久化模型时跳过（能力可禁用）。
        - 落盘失败仅记录+告警，不抛异常、不改变训练成功状态（不影响核心功能）。
        - model_kind 固定为 ``fraud``（监督欺诈评分模型）；版本号取 job.model_version。
        """
        if self._model_store is None or model is None:
            return
        try:
            self._model_store.save(
                "fraud", job.model_version or job.job_id, model, job.metrics or {}
            )
        except Exception as exc:  # noqa: BLE001 - 落盘实现可能抛任意异常
            self._alarm.alarm(
                "AI 模型持久化失败",
                f"任务 {job.job_id} 训练成功但模型落盘失败：{exc}",
            )

    def _extract_and_write_indicators(
        self, job: TrainingJob, samples: Sequence[object], model: object = None
    ) -> None:
        """训练成功后提取增强指标并写入指标存储（R13.2/13.5/13.8）。

        汇集两类增强指标后经同一写入链路写入：
        - 交易对手关系指标（度数/笔数/金额合计），由 CounterpartyGraphExtractor 提取；
        - 监督式欺诈评分指标 ``ai_fraud_score``，由 FraudScorer 用已训练模型产出。

        - 两类增强能力均可独立禁用（端口未注入则跳过，R13.4）。
        - 提取/评分过程异常或写入耗尽重试仍失败：记录失败原因并告警，但不抛出异常、
          不改变训练成功状态（R13.8：不影响核心功能）。
        - 写入后的指标可被规则引用（R13.5，由指标引用名规范保证）。
        """
        if self._indicator_writer is None:
            # 指标写入未启用：训练正常完成，不写任何指标
            return

        metrics: list = []

        # 交易对手关系指标（任务 17.3）
        if self._counterparty_extractor is not None:
            try:
                metrics.extend(self._counterparty_extractor.extract(samples))
            except Exception as exc:  # noqa: BLE001 - 提取实现可能抛出任意异常
                self._alarm.alarm(
                    "AI 交易对手指标提取失败",
                    f"任务 {job.job_id} 训练成功但交易对手关系指标提取失败：{exc}",
                )

        # 监督式欺诈评分指标 ai_fraud_score（AI 增强）
        if self._fraud_scorer is not None and model is not None:
            try:
                metrics.extend(self._fraud_scorer.score(model, samples))
            except Exception as exc:  # noqa: BLE001 - 评分实现可能抛出任意异常
                self._alarm.alarm(
                    "AI 欺诈评分指标产出失败",
                    f"任务 {job.job_id} 训练成功但 ai_fraud_score 指标产出失败：{exc}",
                )

        # 无监督异常检测指标 ai_anomaly_score（AI 增强，无监督，不依赖监督模型）
        if self._anomaly_scorer is not None:
            try:
                metrics.extend(self._anomaly_scorer.score(model, samples))
            except Exception as exc:  # noqa: BLE001 - 评分实现可能抛出任意异常
                self._alarm.alarm(
                    "AI 异常检测指标产出失败",
                    f"任务 {job.job_id} 训练成功但 ai_anomaly_score 指标产出失败：{exc}",
                )

        if not metrics:
            return

        outcome: IndicatorWriteOutcome = write_metrics_with_retry(
            self._indicator_writer,
            metrics,
            max_attempts=self._indicator_write_max_retries,
        )
        job.indicator_metrics_written = outcome.succeeded
        job.indicator_write_failed = len(outcome.failures)

        if outcome.has_failure:
            # R13.8：重试耗尽仍失败，记录失败原因并告警，不影响训练成功与核心功能
            sample_reason = outcome.failures[0][1]
            detail = (
                f"任务 {job.job_id} 训练成功，但 {len(outcome.failures)}/{outcome.attempted} "
                f"条增强指标写入指标存储失败（每条最多重试 "
                f"{self._indicator_write_max_retries} 次）。示例失败原因：{sample_reason}"
            )
            self._alarm.alarm("AI 增强指标写入失败", detail)


def _filter_jobs(
    jobs: "list[TrainingJob]",
    job_id: str | None,
    status: str | None,
    start_time: datetime | None,
    end_time: datetime | None,
) -> "list[TrainingJob]":
    """内存过滤 + 按 started_at 降序。"""
    result: list[TrainingJob] = []
    for job in jobs:
        if job_id and job_id not in (job.job_id or ""):
            continue
        if status and job.status.value != status:
            continue
        if start_time and job.started_at < start_time:
            continue
        if end_time and job.started_at > end_time:
            continue
        result.append(job)
    return sorted(result, key=lambda j: j.started_at, reverse=True)

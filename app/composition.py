"""组合根（Composition Root）：装配 AI 训练服务的依赖（R13）。

集中创建并连接领域服务与基础设施实现，供 FastAPI 入口（main.py）使用：
- 样本来源：基于 SQLAlchemy 的历史订单行来源（按数据时间范围读取 MySQL）。
- 训练算法：默认占位训练器（可替换为基于 scikit-learn 的实现）。
- 仓储：基于 SQLAlchemy 写入/读取 ai_training_job 表。
- 告警：基于日志的告警通道。
- 交易对手指标提取器 + 指标写入器：训练成功后写入指标存储（旁路增强）。

所有外部资源（MySQL engine、httpx 写入）均延迟创建，避免在无依赖环境导入即失败。
依赖项可通过参数覆盖，便于测试或本地以内存替身替换。
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Sequence

from app.config import Settings, get_settings
from app.domain.training_job import (
    ModelTrainer,
    TrainingResult,
    TrainingService,
)


class _PlaceholderTrainer:
    """占位训练器：返回基于样本量的确定性模型版本与评估指标。

    生产环境可替换为基于 scikit-learn 的实际训练实现；此处保证服务在缺少模型代码时
    仍可端到端跑通「提交→记录」链路（评估指标取样本量等可观测值）。
    """

    def train(self, samples: Sequence[object]) -> TrainingResult:
        n = len(samples)
        version = "model-" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return TrainingResult(model_version=version, metrics={"sampleCount": n})


@lru_cache(maxsize=1)
def _mysql_engine_factory():
    """构建并缓存 MySQL engine（延迟导入 SQLAlchemy，避免无依赖环境导入失败）。"""
    from sqlalchemy import create_engine

    settings = get_settings()
    engine = create_engine(settings.mysql_url, pool_pre_ping=True)
    return engine


def build_training_service(
    settings: Settings | None = None,
    *,
    trainer: ModelTrainer | None = None,
    repository=None,
) -> TrainingService:
    """装配生产用 TrainingService（连接 MySQL + 指标存储）。

    在无数据库/网络的环境下，本函数不会立即建立连接（engine 延迟创建），但实际执行训练
    时才会触发连接。单元测试不应调用本函数，而应直接以内存替身构造 TrainingService。

    默认训练器为监督式欺诈评分模型（FraudModelTrainer）：以历史订单 final_decision 为弱标签
    训练欺诈二分类，训练成功后为每个商户产出 ai_fraud_score 指标写入指标存储，供规则引用。
    """
    settings = settings or get_settings()

    from app.domain.anomaly_model import AnomalyScorer
    from app.domain.fraud_model import FraudModelTrainer, FraudScorer
    from app.infrastructure.alarm import LoggingAlarmNotifier
    from app.infrastructure.anomaly_detector import build_anomaly_detector
    from app.infrastructure.counterparty_graph import (
        CompositeCounterpartyExtractor,
        GraphAnalyticsExtractor,
        NetworkxCounterpartyGraphExtractor,
    )
    from app.infrastructure.fraud_classifier import build_fraud_classifier
    from app.infrastructure.indicator_writer import HttpxIndicatorWriter
    from app.infrastructure.model_repository import FileModelRepository
    from app.infrastructure.order_reader import OrderReader, sqlalchemy_row_source
    from app.infrastructure.training_job_repository import (
        SqlAlchemyTrainingJobRepository,
    )

    sample_source = OrderReader(sqlalchemy_row_source(_mysql_engine_factory))
    repository = repository or SqlAlchemyTrainingJobRepository(_mysql_engine_factory)
    indicator_writer = HttpxIndicatorWriter(settings.indicator_store_base_url)
    # 交易对手指标 = 基础关系指标（度数/笔数/金额）+ 团伙与中心度指标（团伙规模/疑似团伙/PageRank）
    counterparty_extractor = CompositeCounterpartyExtractor(
        NetworkxCounterpartyGraphExtractor(),
        GraphAnalyticsExtractor(),
    )

    return TrainingService(
        sample_source=sample_source,
        # 默认监督式欺诈评分训练器；scikit-learn 可用时内部用 GBDT，否则纯 Python 逻辑回归
        trainer=trainer or FraudModelTrainer(model_factory=build_fraud_classifier),
        repository=repository,
        alarm=LoggingAlarmNotifier(),
        min_training_samples=settings.min_training_samples,
        max_training_seconds=settings.max_training_seconds,
        counterparty_extractor=counterparty_extractor,
        indicator_writer=indicator_writer,
        indicator_write_max_retries=settings.indicator_write_max_retries,
        # 训练成功后产出 ai_fraud_score 商户级欺诈概率指标
        fraud_scorer=FraudScorer(),
        # 训练成功后产出 ai_anomaly_score 商户级无监督异常分指标（孤立森林，可回退）
        anomaly_scorer=AnomalyScorer(detector_factory=build_anomaly_detector),
        # 训练成功后将模型按版本落盘（joblib），支持训练一次多次评分与版本回滚（S12.2）
        model_store=FileModelRepository(settings.model_store_dir),
    )


def build_schedule_service(
    settings: Settings | None = None,
    *,
    training_service: TrainingService | None = None,
    schedule_repository=None,
    job_repository=None,
):
    """装配 TrainingScheduleService（与 TrainingService 共享 job 仓储）。"""
    from app.domain.training_schedule import TrainingScheduleService
    from app.infrastructure.training_schedule_repository import (
        SqlAlchemyTrainingScheduleRepository,
    )
    from app.infrastructure.training_job_repository import SqlAlchemyTrainingJobRepository

    settings = settings or get_settings()
    job_repo = job_repository
    training = training_service
    if training is None:
        job_repo = job_repo or SqlAlchemyTrainingJobRepository(_mysql_engine_factory)
        training = build_training_service(settings, repository=job_repo)
    elif job_repo is None:
        job_repo = training._repository  # noqa: SLF001 - composition root 共享仓储

    schedule_repo = schedule_repository or SqlAlchemyTrainingScheduleRepository(
        _mysql_engine_factory
    )
    return TrainingScheduleService(
        schedule_repo=schedule_repo,
        job_repo=job_repo,
        training_service=training,
    )


def build_scoring_service(settings: Settings | None = None) -> "OnlineScoringService":
    """装配在线评分服务（与训练共用模型落盘目录）。"""
    from app.domain.online_scoring import OnlineScoringService
    from app.infrastructure.model_repository import FileModelRepository

    settings = settings or get_settings()
    return OnlineScoringService(FileModelRepository(settings.model_store_dir))


def build_model_management_service(
    settings: Settings | None = None,
    scoring_service: "OnlineScoringService | None" = None,
) -> "ModelManagementService":
    """装配模型管理（版本列表 / 启用当前版本 / 评分可用性探测）。"""
    from app.domain.model_management import ModelManagementService
    from app.domain.online_scoring import OnlineScoringService
    from app.infrastructure.model_repository import FileModelRepository

    settings = settings or get_settings()
    store = FileModelRepository(settings.model_store_dir)
    scoring = scoring_service or OnlineScoringService(store)
    return ModelManagementService(store, scoring)


"""无监督异常检测模型（AI 增强 S11++）。

监督式欺诈评分（fraud_model）依赖历史 ``final_decision`` 标签，适合识别「已知」欺诈模式；
但上线初期标签稀缺、且新型欺诈无历史标签可学。本模块提供**无监督异常检测**作为互补：
不依赖任何标签，仅从订单特征分布中识别「离群」交易主体，产出商户级异常分
``ai_anomaly_score``（0..1，越高越异常），供规则引用（如 ``ai_anomaly_score > 0.9``
触发人工复核）。典型用途：冷启动、未知欺诈模式、与监督分互为补充。

模型：
- scikit-learn `IsolationForest`（孤立森林）—— 表格型异常检测的工业常用无监督模型；
  环境未安装 scikit-learn 时回退到内置纯 Python 实现（基于稳健 z-score 的离群度），
  保证最小环境也能端到端跑通（与既有回退策略一致）。
- 输出统一归一化到 0..1：孤立森林用 `decision_function` 经单调变换映射；纯 Python 回退
  用各特征稳健 z-score 的最大绝对值经 logistic 压缩。二者均为确定性。

复用既有特征工程：直接采用 `fraud_model.extract_features`，保证有/无监督两类模型「看到」
同一套特征，便于解释与对齐。

DDD 分层：本模块属 domain 层，仅依赖标准库与领域内既有组件（CounterpartyMetric /
validate_ref_name / fraud_model 特征抽取）。具体孤立森林实现位于基础设施层并在组合根注入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Sequence

from app.domain.counterparty import CounterpartyMetric, validate_ref_name
from app.domain.fraud_model import extract_features, _vectorize, current_day_slice_ts

# 异常分指标默认引用名（满足 [A-Za-z0-9_] 规范，可被规则引用，R13.5）
ANOMALY_SCORE_REF_NAME = "ai_anomaly_score"


# ---- 异常检测器抽象与纯 Python 回退实现 ---------------------------------------


class AnomalyDetector(Protocol):
    """无监督异常检测器端口：在特征矩阵上拟合并给出单样本异常分（0..1，越高越异常）。"""

    def fit(self, x_matrix: list[list[float]]) -> None:
        ...

    def anomaly_score_one(self, x_row: list[float]) -> float:
        ...


class RobustZScoreDetector:
    """纯 Python 异常检测回退：基于各特征稳健 z-score（中位数 + MAD）的最大偏离度。

    对每个特征计算中位数与 MAD（绝对中位差），单样本异常度取各特征稳健 z-score 绝对值的
    最大值，再经 logistic 压缩到 0..1。无第三方依赖、确定性，作为孤立森林不可用时的回退。
    """

    # MAD 到标准差的一致性常数（正态分布下 1.4826 * MAD ≈ std）
    _MAD_SCALE = 1.4826

    def __init__(self, *, steepness: float = 1.0) -> None:
        self._steepness = steepness
        self._median: list[float] = []
        self._mad: list[float] = []

    def fit(self, x_matrix: list[list[float]]) -> None:
        if not x_matrix:
            raise ValueError("特征矩阵为空，无法拟合异常检测模型")
        dim = len(x_matrix[0])
        self._median = [0.0] * dim
        self._mad = [0.0] * dim
        for j in range(dim):
            col = sorted(row[j] for row in x_matrix)
            med = _median_sorted(col)
            abs_dev = sorted(abs(v - med) for v in col)
            mad = _median_sorted(abs_dev)
            self._median[j] = med
            # MAD=0（该列几乎恒定）时置一个极小正数避免除零；此列对异常度贡献趋于 0
            self._mad[j] = mad * self._MAD_SCALE if mad > 1e-12 else 0.0

    def anomaly_score_one(self, x_row: list[float]) -> float:
        if not self._median:
            return 0.0
        max_z = 0.0
        for j in range(len(self._median)):
            scale = self._mad[j]
            if scale <= 0.0:
                continue
            z = abs(x_row[j] - self._median[j]) / scale
            if z > max_z:
                max_z = z
        # logistic 压缩：z=0→0.0，z 越大→趋近 1（减 0.5 再翻倍使 z=0 时得 0）
        return 2.0 * (_sigmoid(self._steepness * max_z) - 0.5)


def _median_sorted(values: list[float]) -> float:
    """对已排序列表求中位数。"""
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def default_detector_factory() -> AnomalyDetector:
    """默认异常检测器工厂：返回纯 Python 稳健 z-score 实现（无第三方依赖）。"""
    return RobustZScoreDetector()


# ---- 训练结果值对象 ------------------------------------------------------------


@dataclass
class FittedAnomalyModel:
    """拟合完成的异常检测模型 + 特征列对齐信息。"""

    detector: AnomalyDetector
    feature_columns: list[str]


@dataclass
class AnomalyTrainOutcome:
    """一次异常检测模型训练的产出：模型 + 统计指标。"""

    model: FittedAnomalyModel
    metrics: dict = field(default_factory=dict)


# ---- 训练与评分 ----------------------------------------------------------------


def train_anomaly_model(
    samples: Sequence[object],
    *,
    detector_factory=default_detector_factory,
) -> AnomalyTrainOutcome:
    """在订单样本上拟合无监督异常检测模型（不依赖标签）。

    抽取全部样本特征（无需标签）→ 对齐固定特征列 → 拟合检测器 → 统计训练集异常分分布。
    样本为空时抛出 ValueError，由上层记为训练失败（R13.7）。
    """
    feature_rows = [extract_features(record) for record in samples]
    feature_rows = [f for f in feature_rows if f]
    if not feature_rows:
        raise ValueError("无可用样本特征，无法训练异常检测模型")

    feature_columns = sorted({name for feats in feature_rows for name in feats})
    x_matrix = [_vectorize(feats, feature_columns) for feats in feature_rows]

    detector = detector_factory()
    detector.fit(x_matrix)

    scores = [detector.anomaly_score_one(row) for row in x_matrix]
    metrics = {
        "anomalySampleCount": len(x_matrix),
        "anomalyFeatureCount": len(feature_columns),
        "anomalyScoreMax": round(max(scores), 6),
        "anomalyScoreMean": round(sum(scores) / len(scores), 6),
    }
    return AnomalyTrainOutcome(
        model=FittedAnomalyModel(detector=detector, feature_columns=feature_columns),
        metrics=metrics,
    )


def score_merchant_anomaly(
    model: FittedAnomalyModel,
    samples: Sequence[object],
    *,
    slice_ts: int,
    ref_name: str = ANOMALY_SCORE_REF_NAME,
) -> list[CounterpartyMetric]:
    """用已训练异常检测模型为每个商户产出异常分指标 ``ai_anomaly_score``。

    将同一商户多笔订单特征按列求均值得到商户级向量，再计算异常分（0..1）。无 merchant_id
    的样本跳过。结果按 dimension_key 升序，保证确定性与幂等。
    """
    ref_name = validate_ref_name(ref_name)

    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for record in samples:
        merchant = _merchant_of(record)
        if merchant is None:
            continue
        feats = extract_features(record)
        bucket = sums.setdefault(merchant, {})
        for name, value in feats.items():
            bucket[name] = bucket.get(name, 0.0) + value
        counts[merchant] = counts.get(merchant, 0) + 1

    metrics: list[CounterpartyMetric] = []
    for merchant, bucket in sums.items():
        count = counts[merchant]
        mean_feats = {name: total / count for name, total in bucket.items()}
        row = _vectorize(mean_feats, model.feature_columns)
        score = model.detector.anomaly_score_one(row)
        metrics.append(CounterpartyMetric(ref_name, merchant, round(score, 6), slice_ts))

    metrics.sort(key=lambda m: m.dimension_key)
    return metrics


def _merchant_of(record: object) -> str | None:
    if isinstance(record, dict):
        value = record.get("merchant_id")
    else:
        value = getattr(record, "merchant_id", None)
    if value is None or str(value).strip() == "":
        return None
    return str(value)


# ---- 与 TrainingService 集成的评分组件 ----------------------------------------


class AnomalyScorer:
    """异常分指标产出器：训练成功路径用已训练异常模型为各商户产出 ``ai_anomaly_score``。

    与欺诈评分/交易对手指标并行，由 TrainingService 在成功路径合并写入指标存储。
    本评分器自身负责训练异常模型（无监督，无需标签），故只需注入样本即可。
    """

    def __init__(
        self,
        *,
        ref_name: str = ANOMALY_SCORE_REF_NAME,
        detector_factory=default_detector_factory,
        now_provider=None,
    ) -> None:
        self._ref_name = ref_name
        self._detector_factory = detector_factory
        self._now_provider = now_provider

    def score(self, model: object, samples: Sequence[object]) -> Sequence[CounterpartyMetric]:
        # 异常检测无监督，不依赖传入的监督模型；就地训练后评分
        outcome = train_anomaly_model(samples, detector_factory=self._detector_factory)
        slice_ts = current_day_slice_ts(self._now_provider)
        return score_merchant_anomaly(
            outcome.model, samples, slice_ts=slice_ts, ref_name=self._ref_name
        )

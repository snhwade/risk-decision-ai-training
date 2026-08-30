"""监督式欺诈评分模型（AI 增强，R13 旁路增强扩展）。

本模块在既有「训练任务 + 交易对手关系图」之外，新增一类**监督式欺诈评分模型**：
以历史订单（risk_order）的 ``final_decision`` 为弱标签（REJECT=欺诈正样本、PASS=负样本），
从订单上下文（context JSON）抽取数值特征训练二分类模型，训练成功后为每个商户产出
``ai_fraud_score`` 欺诈概率指标（0..1），经既有指标写入链路写入指标存储，供规则引擎引用
（例如 ``ai_fraud_score > 0.8`` 触发 REVIEW/REJECT）。

设计取舍（与既有 AI 旁路增强保持一致）：
- **零决策链路侵入**：模型产出的是「指标」，复用 `IndicatorWriter` + 重试链路写入，
  规则像引用任何其它指标一样引用 ``ai_fraud_score``，无需改动事中决策编排。
- **无第三方依赖也能跑通**：核心训练/评分逻辑为纯 Python，提供内置的逻辑回归回退实现
  （`LogisticRegressionFallback`）；当环境安装了 scikit-learn 时，基础设施层用 GBDT 实现
  替换以获得更优精度。二者通过 `FraudClassifier` 抽象统一，互不影响领域逻辑。
- **确定性**：相同样本、相同实现产出相同模型与评分（回退实现用固定迭代与初值），
  便于业务集成测试断言。

DDD 分层：本模块属 domain 层，仅依赖标准库与领域内既有值对象（CounterpartyMetric），
不直接依赖 scikit-learn / pandas。具体 GBDT 实现位于基础设施层并在组合根注入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, Sequence

from app.domain.counterparty import CounterpartyMetric, validate_ref_name

# ---- 常量 ----------------------------------------------------------------------

# 欺诈评分指标默认引用名（满足 [A-Za-z0-9_] 规范，可被规则引用，R13.5）
FRAUD_SCORE_REF_NAME = "ai_fraud_score"

# 标签映射：最终决策 → 二分类标签（None 表示该样本不参与监督训练，但仍可被评分）
_LABEL_POSITIVE = "REJECT"  # 欺诈正样本
_LABEL_NEGATIVE = "PASS"  # 正常负样本
# REVIEW / None / 其它：标签不明确，排除出训练集（仍参与评分）


# ---- 特征抽取（OrderRecord → 数值特征向量）------------------------------------


def extract_features(record: object) -> dict[str, float]:
    """从单条订单记录抽取数值特征字典（缺失字段安全降级为 0）。

    特征来源：
    - 订单上下文 context（交易金额等业务字段）；
    - 受理时间 event_time（小时、星期，刻画时间模式）。

    返回的特征名集合在不同记录间可能不同（上下文字段不定），由 `train_fraud_model`
    在训练集上取并集后对齐为固定列。
    """
    features: dict[str, float] = {}

    context = _get_attr(record, "context")
    amount = _coerce_float(_lookup(context, "amount", "txn_amount", "trade_amount"))
    features["amount"] = amount
    features["log_amount"] = math.log1p(amount) if amount > 0 else 0.0

    # 上下文中的其它数值字段统一以 ctx_ 前缀纳入特征（布尔按 0/1，数值原样）
    if isinstance(context, dict):
        for key, value in context.items():
            if key in ("amount", "txn_amount", "trade_amount"):
                continue
            num = _coerce_optional_float(value)
            if num is not None:
                features[f"ctx_{key}"] = num

    event_time = _get_attr(record, "event_time")
    if isinstance(event_time, datetime):
        features["hour"] = float(event_time.hour)
        features["dow"] = float(event_time.weekday())

    return features


def derive_label(final_decision: object) -> int | None:
    """由最终决策派生二分类标签：REJECT→1（欺诈），PASS→0（正常），其它→None（不训练）。"""
    if not isinstance(final_decision, str):
        return None
    decision = final_decision.strip().upper()
    if decision == _LABEL_POSITIVE:
        return 1
    if decision == _LABEL_NEGATIVE:
        return 0
    return None


# ---- 分类器抽象与纯 Python 回退实现 -------------------------------------------


class FraudClassifier(Protocol):
    """欺诈二分类器端口：在数值特征矩阵上拟合并给出单样本欺诈概率。

    具体实现可为内置逻辑回归回退（无依赖）或基于 scikit-learn 的 GBDT（基础设施层）。
    """

    def fit(self, x_matrix: list[list[float]], y: list[int]) -> None:
        ...

    def predict_proba_one(self, x_row: list[float]) -> float:
        ...

    def feature_importances(self) -> list[float]:
        """返回与特征列同序的重要度权重（非负，便于归一化）。用于可解释性。"""
        ...


class LogisticRegressionFallback:
    """纯 Python 逻辑回归（带 z-score 标准化 + 批量梯度下降），无第三方依赖。

    作为 scikit-learn 不可用时的回退实现，保证 AI 评分链路在最小环境也能端到端跑通。
    采用固定学习率/迭代次数与零初值，保证确定性（相同输入产出相同模型）。
    """

    def __init__(self, *, learning_rate: float = 0.1, iterations: int = 500,
                 balance_classes: bool = False) -> None:
        self._lr = learning_rate
        self._iters = iterations
        self._balance_classes = balance_classes
        self._weights: list[float] = []
        self._bias: float = 0.0
        self._mean: list[float] = []
        self._std: list[float] = []

    def fit(self, x_matrix: list[list[float]], y: list[int]) -> None:
        n = len(x_matrix)
        if n == 0:
            raise ValueError("训练特征矩阵为空，无法拟合欺诈评分模型")
        dim = len(x_matrix[0])
        # 计算每列均值与标准差用于标准化（std=0 时置 1 避免除零）
        self._mean = [0.0] * dim
        self._std = [0.0] * dim
        for j in range(dim):
            col = [row[j] for row in x_matrix]
            mean = sum(col) / n
            var = sum((v - mean) ** 2 for v in col) / n
            self._mean[j] = mean
            self._std[j] = math.sqrt(var) if var > 1e-12 else 1.0

        # 类别不平衡处理（S12.4）：按类频率反比为每个样本计算权重，使少数类影响更大
        weights_per_sample = self._class_weights(y) if self._balance_classes else None

        scaled = [self._scale(row) for row in x_matrix]
        self._weights = [0.0] * dim
        self._bias = 0.0
        weight_sum = float(sum(weights_per_sample)) if weights_per_sample else float(n)
        for _ in range(self._iters):
            grad_w = [0.0] * dim
            grad_b = 0.0
            for idx, (row, label) in enumerate(zip(scaled, y)):
                pred = self._sigmoid(self._dot(self._weights, row) + self._bias)
                w = weights_per_sample[idx] if weights_per_sample else 1.0
                error = (pred - label) * w
                for j in range(dim):
                    grad_w[j] += error * row[j]
                grad_b += error
            for j in range(dim):
                self._weights[j] -= self._lr * grad_w[j] / weight_sum
            self._bias -= self._lr * grad_b / weight_sum

    @staticmethod
    def _class_weights(y: list[int]) -> list[float]:
        """按类频率反比计算样本权重：w_i = n / (n_classes * n_yi)。"""
        n = len(y)
        counts: dict[int, int] = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        n_classes = len(counts) or 1
        return [n / (n_classes * counts[label]) for label in y]

    def predict_proba_one(self, x_row: list[float]) -> float:
        if not self._weights:
            return 0.0
        scaled = self._scale(x_row)
        return self._sigmoid(self._dot(self._weights, scaled) + self._bias)

    def feature_importances(self) -> list[float]:
        """以标准化特征上的权重绝对值作为重要度（特征已 z-score 标准化，量纲可比）。"""
        return [abs(w) for w in self._weights]

    def _scale(self, row: list[float]) -> list[float]:
        return [(row[j] - self._mean[j]) / self._std[j] for j in range(len(row))]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)


def default_model_factory() -> FraudClassifier:
    """默认分类器工厂：返回纯 Python 逻辑回归回退实现（无第三方依赖）。"""
    return LogisticRegressionFallback()


# ---- 训练结果值对象 ------------------------------------------------------------


@dataclass
class FittedFraudModel:
    """训练完成的欺诈评分模型 + 特征列对齐信息。

    - classifier：已拟合的分类器。
    - feature_columns：训练时确定的固定特征列顺序，评分时据此对齐缺失列补 0。
    - feature_baseline：训练集各特征均值（S12.5 单笔解释的基线行；缺失时按全 0 处理）。
    """

    classifier: FraudClassifier
    feature_columns: list[str]
    feature_baseline: list[float] = field(default_factory=list)


@dataclass
class FraudTrainOutcome:
    """一次欺诈评分模型训练的产出：模型 + 评估指标。"""

    model: FittedFraudModel
    metrics: dict = field(default_factory=dict)


# ---- 训练与评估 ----------------------------------------------------------------


def train_fraud_model(
    samples: Sequence[object],
    *,
    model_factory=default_model_factory,
    cv_folds: int = 5,
) -> FraudTrainOutcome:
    """在历史订单样本上训练监督式欺诈评分模型并**留出/交叉验证**评估（R13.3 真实评估指标）。

    流程：
    1. 抽取带标签样本（REJECT/PASS）→ 对齐固定特征列。
    2. **分层 K 折交叉验证**评估泛化能力（默认 5 折）：每折在其余折上训练、在该折上评估，
       汇总各折的样本外（out-of-sample）预测计算 AUC/KS/准确率——避免在训练集上自评的
       过拟合乐观偏差。样本过少不足以分层 K 折时，自动降级为单次分层留出（hold-out）评估；
       仍不足则标记 evalMethod=in_sample 并退回训练集自评（仅作兜底，附 warning）。
    3. **在全部带标签样本上重新拟合最终模型**（用于产出 ai_fraud_score），最大化数据利用。

    当带标签样本缺失或仅含单一类别（无法学习决策边界）时抛出 ValueError，由上层
    `TrainingService` 记为训练失败并告警（R13.7），不写入任何指标。
    """
    labeled: list[tuple[dict[str, float], int]] = []
    for record in samples:
        label = derive_label(_get_attr(record, "final_decision"))
        if label is None:
            continue
        labeled.append((extract_features(record), label))

    if not labeled:
        raise ValueError("无可用带标签样本（需 final_decision 为 REJECT 或 PASS），无法训练欺诈评分模型")

    positives = sum(1 for _f, y in labeled if y == 1)
    negatives = len(labeled) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"正负样本不均衡，无法训练：正样本 {positives} 条、负样本 {negatives} 条（两者均需 ≥ 1）"
        )

    feature_columns = sorted({name for feats, _y in labeled for name in feats})
    x_matrix = [_vectorize(feats, feature_columns) for feats, _y in labeled]
    y = [label for _f, label in labeled]

    # 样本外评估（分层 K 折 CV / 留出 / 兜底自评）
    metrics = _cross_validated_metrics(x_matrix, y, model_factory, cv_folds=cv_folds)
    metrics.update(
        {
            "sampleCount": len(samples),
            "labeledCount": len(labeled),
            "positiveCount": positives,
            "negativeCount": negatives,
            "featureCount": len(feature_columns),
        }
    )

    # 在全部带标签样本上重新拟合最终模型（用于评分与特征重要度）
    classifier = model_factory()
    classifier.fit(x_matrix, y)

    # 训练集各特征均值（S12.5 单笔解释基线行）
    n = len(x_matrix)
    dim = len(feature_columns)
    feature_baseline = [sum(row[j] for row in x_matrix) / n for j in range(dim)] if n else [0.0] * dim

    # 可解释性（合规：为何拒绝）：输出 Top 特征重要度，落评估指标供审计/前端展示
    importances = _safe_feature_importances(classifier, feature_columns)
    if importances:
        metrics["featureImportances"] = importances
        metrics["topFeatures"] = [name for name, _w in _top_importances(importances, k=5)]

    return FraudTrainOutcome(
        model=FittedFraudModel(
            classifier=classifier,
            feature_columns=feature_columns,
            feature_baseline=feature_baseline,
        ),
        metrics=metrics,
    )


def explain_fraud_prediction(
    model: FittedFraudModel,
    features: dict[str, float],
    *,
    top_k: int = 5,
) -> dict:
    """对单条样本的欺诈评分给出**特征级贡献度解释**（S12.5，合规：为何拒绝这一笔）。

    - 线性模型（逻辑回归回退）：精确线性 SHAP —— 贡献 = 标准化权重 ×（标准化特征值）。
    - 通用模型（GBDT）：逐特征扰动到基线（训练均值），用分数变化作为局部贡献近似。
    二者统一返回结构，所有计算确定性，便于审计与前端「决策依据」展示。

    返回：{ score, baseScore, method, contributions:[{feature,value,contribution}...]（|贡献|降序 top_k） }
    """
    columns = model.feature_columns
    row = _vectorize(features, columns)
    classifier = model.classifier
    score = classifier.predict_proba_one(row)

    if isinstance(classifier, LogisticRegressionFallback):
        base_row = list(classifier._mean) if classifier._mean else [0.0] * len(columns)
        base_score = classifier.predict_proba_one(base_row)
        scaled = classifier._scale(row)
        contributions = [
            {
                "feature": col,
                "value": round(row[j], 6),
                "contribution": round(classifier._weights[j] * scaled[j], 6),
            }
            for j, col in enumerate(columns)
        ]
        method = "linear_shap"
    else:
        base_row = _baseline_row(model)
        base_score = classifier.predict_proba_one(base_row)
        contributions = []
        for j, col in enumerate(columns):
            perturbed = list(row)
            perturbed[j] = base_row[j]
            delta = score - classifier.predict_proba_one(perturbed)
            contributions.append(
                {"feature": col, "value": round(row[j], 6), "contribution": round(delta, 6)}
            )
        method = "perturbation"

    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return {
        "score": round(score, 6),
        "baseScore": round(base_score, 6),
        "method": method,
        "contributions": contributions[:top_k],
    }


def _baseline_row(model: FittedFraudModel) -> list[float]:
    """单笔解释用的基线行：优先用训练时记录的特征均值，缺失则全 0。"""
    baseline = getattr(model, "feature_baseline", None)
    if isinstance(baseline, list) and len(baseline) == len(model.feature_columns):
        return list(baseline)
    return [0.0] * len(model.feature_columns)


def score_merchant_fraud(
    model: FittedFraudModel,
    samples: Sequence[object],
    *,
    slice_ts: int,
    ref_name: str = FRAUD_SCORE_REF_NAME,
) -> list[CounterpartyMetric]:
    """用已训练模型为每个商户产出欺诈概率指标 ``ai_fraud_score``（R13.2 写入指标存储）。

    将同一商户的多笔订单特征按列求均值得到商户级特征向量，再预测欺诈概率（0..1）。
    无 merchant_id 的样本无法形成可被规则引用的维度键，跳过。结果按 dimension_key 升序，
    保证确定性与幂等（支撑重复训练写入一致语义）。
    """
    ref_name = validate_ref_name(ref_name)

    # 按商户聚合特征（按列累加后求均值）
    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for record in samples:
        merchant = _get_attr(record, "merchant_id")
        if merchant is None or str(merchant).strip() == "":
            continue
        merchant = str(merchant)
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
        score = model.classifier.predict_proba_one(row)
        metrics.append(CounterpartyMetric(ref_name, merchant, round(score, 6), slice_ts))

    metrics.sort(key=lambda m: m.dimension_key)
    return metrics


def current_day_slice_ts(now_provider=None) -> int:
    """返回当前 UTC 自然日 0 点的 Unix 秒切片戳（与交易对手指标切片对齐）。"""
    now = (now_provider or (lambda: datetime.now(timezone.utc)))()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    day_start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(day_start.timestamp())


# ---- 评估指标（纯 Python，无 sklearn 依赖）------------------------------------


def _cross_validated_metrics(
    x_matrix: list[list[float]],
    y: list[int],
    model_factory,
    *,
    cv_folds: int,
) -> dict:
    """分层 K 折交叉验证评估（样本外），返回含 evalMethod 的指标字典。

    - 每个类别至少有 cv_folds 个样本时做分层 K 折：逐折训练+预测，汇总样本外预测算 AUC/KS/准确率。
    - 样本不足以 K 折但可二分时降级为分层留出（约 70/30），评估测试集（样本外）。
    - 两者都不足时退回训练集自评（in_sample，附 warning）作为兜底。
    所有划分均确定性（按类别取模分桶），保证相同输入产出相同评估。
    """
    pos = sum(y)
    neg = len(y) - pos
    effective_folds = min(cv_folds, pos, neg)

    if effective_folds >= 2:
        oos_scores, oos_y = _kfold_oos_predictions(x_matrix, y, effective_folds, model_factory)
        metrics = _evaluate(oos_scores, oos_y)
        metrics["evalMethod"] = "cv"
        metrics["cvFolds"] = effective_folds
        return metrics

    # 降级：分层留出（每类至少 1 训练 + 1 测试时可用）
    if pos >= 2 and neg >= 2:
        tr_idx, te_idx = _stratified_holdout_indices(y, test_ratio=0.3)
        if tr_idx and te_idx:
            clf = model_factory()
            clf.fit([x_matrix[i] for i in tr_idx], [y[i] for i in tr_idx])
            scores = [clf.predict_proba_one(x_matrix[i]) for i in te_idx]
            metrics = _evaluate(scores, [y[i] for i in te_idx])
            metrics["evalMethod"] = "holdout"
            metrics["holdoutTestSize"] = len(te_idx)
            return metrics

    # 兜底：样本太少，训练集自评（带过拟合偏差，明确标注）
    clf = model_factory()
    clf.fit(x_matrix, y)
    scores = [clf.predict_proba_one(row) for row in x_matrix]
    metrics = _evaluate(scores, y)
    metrics["evalMethod"] = "in_sample"
    metrics["warning"] = "样本过少，评估退回训练集自评，AUC/KS 含过拟合乐观偏差，仅供参考"
    return metrics


def _stratified_folds(y: list[int], folds: int) -> list[list[int]]:
    """按类别分层将样本索引分到 folds 个桶（确定性：同类别样本按出现序取模分桶）。"""
    buckets: list[list[int]] = [[] for _ in range(folds)]
    per_class_counter: dict[int, int] = {}
    for idx, label in enumerate(y):
        c = per_class_counter.get(label, 0)
        buckets[c % folds].append(idx)
        per_class_counter[label] = c + 1
    return buckets


def _kfold_oos_predictions(
    x_matrix: list[list[float]], y: list[int], folds: int, model_factory
) -> tuple[list[float], list[int]]:
    """分层 K 折，返回所有样本的样本外预测分与对应标签（顺序与折一致）。"""
    fold_buckets = _stratified_folds(y, folds)
    oos_scores: list[float] = []
    oos_y: list[int] = []
    for f in range(folds):
        test_idx = fold_buckets[f]
        train_idx = [i for bf in range(folds) if bf != f for i in fold_buckets[bf]]
        if not test_idx or not train_idx:
            continue
        # 训练集需含两类，否则跳过该折（极端不平衡的兜底）
        train_y = [y[i] for i in train_idx]
        if sum(train_y) == 0 or sum(train_y) == len(train_y):
            continue
        clf = model_factory()
        clf.fit([x_matrix[i] for i in train_idx], train_y)
        for i in test_idx:
            oos_scores.append(clf.predict_proba_one(x_matrix[i]))
            oos_y.append(y[i])
    return oos_scores, oos_y


def _stratified_holdout_indices(
    y: list[int], *, test_ratio: float
) -> tuple[list[int], list[int]]:
    """分层留出：每类按比例切出测试集（确定性，每类至少留 1 训 1 测）。"""
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(y):
        by_class.setdefault(label, []).append(idx)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for _label, idxs in by_class.items():
        n = len(idxs)
        n_test = max(1, min(n - 1, round(n * test_ratio)))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
    return train_idx, test_idx


def _evaluate(scores: list[float], y: list[int]) -> dict:
    """计算 AUC（Mann-Whitney U）、KS 与 0.5 阈值准确率。"""
    auc = _auc(scores, y)
    ks = _ks(scores, y)
    correct = sum(1 for s, label in zip(scores, y) if (1 if s >= 0.5 else 0) == label)
    accuracy = correct / len(y) if y else 0.0
    return {
        "auc": round(auc, 6),
        "ks": round(ks, 6),
        "accuracy": round(accuracy, 6),
    }


def _auc(scores: list[float], y: list[int]) -> float:
    """以秩和法（Mann-Whitney U）计算 AUC，处理并列分数取平均秩。"""
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 秩从 1 开始
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _ks(scores: list[float], y: list[int]) -> float:
    """KS 统计量：正/负样本累计分布的最大差异。"""
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return 0.0
    paired = sorted(zip(scores, y), key=lambda p: p[0])
    cum_pos = 0
    cum_neg = 0
    ks = 0.0
    for _score, label in paired:
        if label == 1:
            cum_pos += 1
        else:
            cum_neg += 1
        ks = max(ks, abs(cum_pos / pos - cum_neg / neg))
    return ks


# ---- 内部工具 ------------------------------------------------------------------


def _vectorize(features: dict[str, float], columns: list[str]) -> list[float]:
    """按固定列顺序将特征字典转为向量，缺失列补 0。"""
    return [float(features.get(col, 0.0)) for col in columns]


def _safe_feature_importances(
    classifier: object, feature_columns: list[str]
) -> dict[str, float]:
    """提取并归一化分类器特征重要度，返回 {特征名: 归一化重要度}（按重要度降序）。

    - 分类器未实现 feature_importances 或长度不匹配时返回空字典（可解释性为尽力而为，
      不影响训练成功）。
    - 归一化为总和 1（全 0 时按列均分），便于跨模型/跨次比较与前端展示。
    """
    getter = getattr(classifier, "feature_importances", None)
    if not callable(getter):
        return {}
    try:
        weights = list(getter())
    except Exception:  # noqa: BLE001 - 可解释性失败不应影响训练
        return {}
    if len(weights) != len(feature_columns) or not feature_columns:
        return {}

    weights = [abs(float(w)) for w in weights]
    total = sum(weights)
    if total <= 0.0:
        uniform = 1.0 / len(feature_columns)
        normalized = {name: round(uniform, 6) for name in feature_columns}
    else:
        normalized = {
            name: round(w / total, 6) for name, w in zip(feature_columns, weights)
        }
    # 按重要度降序返回（dict 保序）
    return dict(sorted(normalized.items(), key=lambda kv: kv[1], reverse=True))


def _top_importances(
    importances: dict[str, float], *, k: int
) -> list[tuple[str, float]]:
    """返回重要度最高的前 k 个 (特征名, 重要度)。"""
    return list(importances.items())[:k]


def _get_attr(record: object, name: str):
    """从 dataclass/对象属性或 dict 取值，缺失返回 None。"""
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _lookup(context: object, *names: str):
    """从上下文字典按候选键取首个非空值。"""
    if not isinstance(context, dict):
        return None
    for name in names:
        if name in context and context[name] is not None:
            return context[name]
    return None


def _coerce_float(value: object) -> float:
    """尽力转 float，失败返回 0.0。"""
    result = _coerce_optional_float(value)
    return result if result is not None else 0.0


def _coerce_optional_float(value: object) -> float | None:
    """尽力转 float：bool→0/1，数值/数字字符串→float，否则 None。"""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


# ---- 与 TrainingService 集成的适配组件 ----------------------------------------


class FraudModelTrainer:
    """监督式欺诈评分模型训练器（满足 `app.domain.training_job.ModelTrainer` 端口）。

    在训练样本上拟合欺诈二分类模型，返回 `TrainingResult`：
    - model_version：以训练时刻派生的可读版本号；
    - metrics：真实评估指标（AUC/KS/准确率 + 样本统计），落库供前端展示（R13.3/13.11）；
    - model：已拟合模型（opaque），供成功路径产出 ``ai_fraud_score`` 指标。

    训练数据不足以学习（无带标签样本或单一类别）时抛出异常，由 `TrainingService` 据 R13.7
    记为训练失败并告警，不写入任何指标。
    """

    def __init__(self, *, model_factory=default_model_factory, now_provider=None) -> None:
        self._model_factory = model_factory
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def train(self, samples):
        # 延迟导入避免与 training_job 形成模块级循环依赖
        from app.domain.training_job import TrainingResult

        outcome = train_fraud_model(samples, model_factory=self._model_factory)
        version = "fraud-" + self._now_provider().strftime("%Y%m%d%H%M%S")
        return TrainingResult(
            model_version=version,
            metrics=outcome.metrics,
            model=outcome.model,
        )


class FraudScorer:
    """欺诈评分指标产出器：用已训练模型为每个商户产出 ``ai_fraud_score`` 指标。

    供 `TrainingService` 在训练成功路径注入；输入为训练产出的 `FittedFraudModel` 与样本，
    输出可经既有指标写入链路写入指标存储的指标列表（R13.2/13.5）。
    """

    def __init__(self, *, ref_name: str = FRAUD_SCORE_REF_NAME, now_provider=None) -> None:
        self._ref_name = ref_name
        self._now_provider = now_provider

    def score(self, model: object, samples: Sequence[object]) -> Sequence[CounterpartyMetric]:
        if not isinstance(model, FittedFraudModel):
            return []
        slice_ts = current_day_slice_ts(self._now_provider)
        return score_merchant_fraud(
            model, samples, slice_ts=slice_ts, ref_name=self._ref_name
        )

"""欺诈评分分类器的基础设施实现（GBDT，AI 增强）。

优先使用 scikit-learn 的梯度提升树（`GradientBoostingClassifier`）拟合欺诈二分类，
这是表格型风控数据上精度与可解释性兼顾的工业常用模型；当运行环境未安装 scikit-learn
时，回退到领域层纯 Python 逻辑回归实现（`LogisticRegressionFallback`），保证 AI 评分
链路在最小环境也能端到端跑通（与 networkx 提取器的回退策略一致）。

scikit-learn 来自 PyPI 公共 registry（已在 requirements.txt 声明），无私有源依赖。
"""

from __future__ import annotations

from app.domain.fraud_model import FraudClassifier, LogisticRegressionFallback


class SklearnGbdtClassifier:
    """基于 scikit-learn GradientBoostingClassifier 的欺诈分类器（FraudClassifier 端口实现）。

    固定 random_state 保证可复现（确定性），便于业务集成测试断言。单类别等退化场景由
    上层 `train_fraud_model` 提前拦截，此处不再处理。

    类别不平衡处理（S12.4）：风控欺诈样本天然为少数类，GBDT 无内置 class_weight，
    故按「类频率反比」计算 sample_weight 喂入 fit，等效提升少数类（欺诈）权重，避免模型
    一边倒地预测多数类。可通过 ``balance_classes=False`` 关闭。
    """

    def __init__(self, *, random_state: int = 42, balance_classes: bool = True) -> None:
        from sklearn.ensemble import GradientBoostingClassifier  # 延迟导入

        self._model = GradientBoostingClassifier(random_state=random_state)
        self._positive_index: int = 1
        self._balance_classes = balance_classes

    def fit(self, x_matrix: list[list[float]], y: list[int]) -> None:
        sample_weight = _balanced_sample_weight(y) if self._balance_classes else None
        self._model.fit(x_matrix, y, sample_weight=sample_weight)
        # 记录正类（标签 1）在 classes_ 中的列索引，供 predict_proba 取概率
        classes = list(self._model.classes_)
        self._positive_index = classes.index(1) if 1 in classes else len(classes) - 1

    def predict_proba_one(self, x_row: list[float]) -> float:
        proba = self._model.predict_proba([x_row])[0]
        return float(proba[self._positive_index])

    def feature_importances(self) -> list[float]:
        """GBDT 自带的特征重要度（基于分裂增益），与特征列同序。"""
        return [float(v) for v in self._model.feature_importances_]


def _balanced_sample_weight(y: list[int]) -> list[float]:
    """按类频率反比计算样本权重（sklearn 'balanced' 等效）：w_i = n / (n_classes * n_yi)。

    使少数类（欺诈）样本获得更高权重，缓解类别不平衡。所有类缺失时退回全 1。
    """
    n = len(y)
    counts: dict[int, int] = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    n_classes = len(counts) or 1
    return [n / (n_classes * counts[label]) for label in y]


def build_fraud_classifier(*, balance_classes: bool = True) -> FraudClassifier:
    """欺诈分类器工厂：scikit-learn 可用时返回 GBDT，否则回退纯 Python 逻辑回归。

    供组合根作为 `model_factory` 注入训练流程；返回的对象满足 `FraudClassifier` 端口。
    ``balance_classes`` 控制是否启用类别不平衡处理（S12.4，默认开）。
    """
    try:
        import sklearn  # noqa: F401 - 仅探测是否可用

        return SklearnGbdtClassifier(balance_classes=balance_classes)
    except ImportError:
        return LogisticRegressionFallback(balance_classes=balance_classes)

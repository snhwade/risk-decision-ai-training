"""无监督异常检测器的基础设施实现（孤立森林，AI 增强 S11++）。

优先使用 scikit-learn 的 `IsolationForest`（孤立森林）——表格型无监督异常检测的工业常用
模型，对高维、非线性离群点鲁棒；当运行环境未安装 scikit-learn 时，回退到领域层纯 Python
稳健 z-score 实现（`RobustZScoreDetector`），保证最小环境也能端到端跑通（与既有回退一致）。

孤立森林的 `decision_function` 越小越异常；本实现将其经单调变换归一化到 0..1（越大越异常），
与领域层异常分语义对齐。固定 random_state 保证可复现（确定性）。

scikit-learn 来自 PyPI 公共 registry（已在 requirements.txt 声明），无私有源依赖。
"""

from __future__ import annotations

from app.domain.anomaly_model import AnomalyDetector, RobustZScoreDetector


class SklearnIsolationForestDetector:
    """基于 scikit-learn IsolationForest 的异常检测器（AnomalyDetector 端口实现）。

    将 `decision_function` 输出（越小越异常，通常在 [-0.5, 0.5] 附近）经
    ``score = clip(0.5 - decision_function, 0, 1)`` 映射为「越大越异常」的 0..1 分值，
    保持单调，便于规则按阈值引用（如 ``ai_anomaly_score > 0.7``）。
    """

    def __init__(self, *, random_state: int = 42, contamination="auto") -> None:
        from sklearn.ensemble import IsolationForest  # 延迟导入

        self._model = IsolationForest(
            random_state=random_state, contamination=contamination
        )

    def fit(self, x_matrix: list[list[float]]) -> None:
        self._model.fit(x_matrix)

    def anomaly_score_one(self, x_row: list[float]) -> float:
        # decision_function 越小越异常；0.5 - df 使其变为越大越异常，再裁剪到 [0, 1]
        df = float(self._model.decision_function([x_row])[0])
        score = 0.5 - df
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score


def build_anomaly_detector(contamination="auto") -> AnomalyDetector:
    """异常检测器工厂：scikit-learn 可用时返回孤立森林，否则回退纯 Python 稳健 z-score。

    供组合根作为 `detector_factory` 注入 AnomalyScorer；返回对象满足 `AnomalyDetector` 端口。
    ``contamination``（S12.4 可调参）："auto" 为默认；传入 (0, 0.5] 浮点显式指定预期异常占比，
    影响孤立森林判定阈值；回退实现不使用该参数（基于稳健 z-score）。
    """
    try:
        import sklearn  # noqa: F401 - 仅探测是否可用

        return SklearnIsolationForestDetector(contamination=contamination)
    except ImportError:
        return RobustZScoreDetector()

"""在线评分领域服务（enhancement-plan T2）。

为决策流 MODEL 节点提供 ``POST /api/v1/ai/score``：从模型仓库加载 fraud/anomaly 当前版本，
对请求 features 打分。无模型时返回 available=false（不抛异常），由引擎按节点降级处理。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.domain.anomaly_model import FittedAnomalyModel
from app.domain.fraud_model import FittedFraudModel, extract_features, _vectorize
from app.domain.model_store import ModelStore


@dataclass(frozen=True)
class ScoreResult:
    available: bool
    score: float | None
    label: str | None
    reason: str | None
    model_kind: str | None
    model_version: str | None

    def to_api(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "score": self.score,
            "label": self.label,
            "reason": self.reason,
            "modelKind": self.model_kind,
            "modelVersion": self.model_version,
        }


class OnlineScoringService:
    """基于 ModelStore 的在线评分。"""

    def __init__(self, model_store: ModelStore) -> None:
        self._store = model_store

    def score(self, model_ref: str | None, features: dict | None) -> ScoreResult:
        kind, version = resolve_model_ref(model_ref)
        if kind is None:
            return ScoreResult(
                available=False,
                score=None,
                label=None,
                reason=f"无法解析 modelRef={model_ref!r}，期望 fraud / anomaly 或 kind@version",
                model_kind=None,
                model_version=None,
            )

        model, used_version = self._load(kind, version)
        if model is None:
            return ScoreResult(
                available=False,
                score=None,
                label=None,
                reason=(
                    f"模型不可用：未找到 {kind}"
                    + (f"@{version}" if version else " 当前版本")
                    + "（请先完成训练任务）"
                ),
                model_kind=kind,
                model_version=version,
            )

        feats = features if isinstance(features, dict) else {}
        try:
            value, label = _score_model(kind, model, feats)
        except Exception as ex:  # noqa: BLE001
            return ScoreResult(
                available=False,
                score=None,
                label=None,
                reason=f"评分失败: {ex}",
                model_kind=kind,
                model_version=used_version,
            )

        return ScoreResult(
            available=True,
            score=round(float(value), 6),
            label=label,
            reason=None,
            model_kind=kind,
            model_version=used_version,
        )

    def _load(self, kind: str, version: str | None) -> tuple[object | None, str | None]:
        if version:
            model = self._store.load_version(kind, version)
            return (model, version) if model is not None else (None, None)

        model = self._store.load_latest(kind)
        if model is None:
            return None, None
        current = None
        if hasattr(self._store, "current_version"):
            current = self._store.current_version(kind)  # type: ignore[attr-defined]
        elif hasattr(self._store, "_read_manifest"):
            current = self._store._read_manifest(kind).get("current")  # noqa: SLF001
        if not current:
            versions = self._store.list_versions(kind)
            current = versions[0].version if versions else None
        return model, current


def resolve_model_ref(model_ref: str | None) -> tuple[str | None, str | None]:
    """解析 modelRef → (model_kind, optional version)。"""
    if model_ref is None or str(model_ref).strip() == "":
        return "fraud", None
    raw = str(model_ref).strip()
    version = None
    if "@" in raw:
        left, right = raw.split("@", 1)
        raw, version = left.strip(), right.strip() or None
    key = raw.lower().replace("-", "_")
    aliases = {
        "fraud": "fraud",
        "ai_fraud_score": "fraud",
        "fraud_score": "fraud",
        "fittedfraudmodel": "fraud",
        "anomaly": "anomaly",
        "ai_anomaly_score": "anomaly",
        "anomaly_score": "anomaly",
        "fittedanomalymodel": "anomaly",
    }
    if key in aliases:
        return aliases[key], version
    if raw.replace("_", "").isalnum():
        return raw, version
    return None, None


def _score_model(kind: str, model: object, features: dict) -> tuple[float, str | None]:
    record = _FeatureBag(features)
    feat_dict = extract_features(record)
    for k, v in features.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            feat_dict.setdefault(str(k), float(v))

    if kind == "fraud":
        if not isinstance(model, FittedFraudModel):
            raise TypeError(f"期望 FittedFraudModel，实际 {type(model).__name__}")
        row = _vectorize(feat_dict, model.feature_columns)
        score = model.classifier.predict_proba_one(row)
        label = "HIGH" if score >= 0.8 else ("MID" if score >= 0.5 else "LOW")
        return score, label

    if kind == "anomaly":
        if not isinstance(model, FittedAnomalyModel):
            raise TypeError(f"期望 FittedAnomalyModel，实际 {type(model).__name__}")
        row = _vectorize(feat_dict, model.feature_columns)
        score = model.detector.anomaly_score_one(row)
        label = "OUTLIER" if score >= 0.8 else "NORMAL"
        return score, label

    raise ValueError(f"不支持的 model_kind={kind}")


class _FeatureBag:
    """将请求 features 适配为 extract_features 可读的订单形对象。"""

    def __init__(self, features: dict) -> None:
        self.context = dict(features)
        self.merchant_id = features.get("merchantId") or features.get("merchant_id")
        self.final_decision = features.get("finalDecision") or features.get("final_decision")
        et = features.get("eventTime") or features.get("event_time")
        if isinstance(et, datetime):
            self.event_time = et
        elif isinstance(et, (int, float)):
            self.event_time = datetime.fromtimestamp(
                float(et) / (1000 if et > 1e12 else 1), tz=timezone.utc
            )
        else:
            self.event_time = None

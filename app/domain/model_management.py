"""模型管理应用服务：列出版本、对比指标、切换当前生效版本、探测在线评分可用性。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.domain.model_store import ModelStore, ModelVersionInfo
from app.domain.online_scoring import OnlineScoringService


class KindListingStore(Protocol):
    """扩展 ModelStore：可枚举已落盘的 model_kind，并维护描述。"""

    def list_kinds(self) -> list[str]:
        ...

    def current_version(self, model_kind: str) -> str | None:
        ...

    def list_versions(self, model_kind: str) -> list[ModelVersionInfo]:
        ...

    def set_current(self, model_kind: str, version: str) -> None:
        ...

    def get_kind_description(self, model_kind: str) -> str | None:
        ...

    def set_kind_description(self, model_kind: str, description: str | None) -> None:
        ...

    def set_version_description(
        self, model_kind: str, version: str, description: str | None
    ) -> None:
        ...


@dataclass(frozen=True)
class ModelKindSummary:
    model_kind: str
    current_version: str | None
    scoring_available: bool
    scoring_reason: str | None
    description: str | None
    versions: list[dict[str, Any]]

    def to_api(self) -> dict[str, Any]:
        return {
            "modelKind": self.model_kind,
            "currentVersion": self.current_version,
            "scoringAvailable": self.scoring_available,
            "scoringReason": self.scoring_reason,
            "description": self.description,
            "versions": self.versions,
        }


# 控制台默认关注的模型类别（即使尚未落盘也会展示空状态）
_DEFAULT_KINDS = ("fraud", "anomaly")


class ModelManagementService:
    def __init__(
        self,
        model_store: ModelStore | KindListingStore,
        scoring: OnlineScoringService | None = None,
    ) -> None:
        self._store = model_store
        self._scoring = scoring

    def list_models(self) -> list[ModelKindSummary]:
        kinds = list(_DEFAULT_KINDS)
        list_kinds = getattr(self._store, "list_kinds", None)
        if callable(list_kinds):
            for kind in list_kinds():
                if kind not in kinds:
                    kinds.append(kind)
        return [self._summarize(kind) for kind in kinds]

    def get_model(self, model_kind: str) -> ModelKindSummary | None:
        kind = (model_kind or "").strip()
        if not kind:
            return None
        return self._summarize(kind)

    def activate(self, model_kind: str, version: str) -> ModelKindSummary:
        kind = (model_kind or "").strip()
        ver = (version or "").strip()
        if not kind or not ver:
            raise ValueError("modelKind 与 version 均不能为空")
        self._store.set_current(kind, ver)
        return self._summarize(kind)

    def update_kind_description(self, model_kind: str, description: str | None) -> ModelKindSummary:
        kind = (model_kind or "").strip()
        if not kind:
            raise ValueError("modelKind 不能为空")
        setter = getattr(self._store, "set_kind_description", None)
        if not callable(setter):
            raise ValueError("当前模型存储不支持类别描述")
        setter(kind, description)
        return self._summarize(kind)

    def update_version_description(
        self, model_kind: str, version: str, description: str | None
    ) -> ModelKindSummary:
        kind = (model_kind or "").strip()
        ver = (version or "").strip()
        if not kind or not ver:
            raise ValueError("modelKind 与 version 均不能为空")
        setter = getattr(self._store, "set_version_description", None)
        if not callable(setter):
            raise ValueError("当前模型存储不支持版本描述")
        setter(kind, ver, description)
        return self._summarize(kind)

    def _summarize(self, model_kind: str) -> ModelKindSummary:
        current = None
        current_fn = getattr(self._store, "current_version", None)
        if callable(current_fn):
            current = current_fn(model_kind)
        kind_desc = None
        get_desc = getattr(self._store, "get_kind_description", None)
        if callable(get_desc):
            kind_desc = get_desc(model_kind)
        versions_raw = self._store.list_versions(model_kind)
        versions = [
            {
                "version": info.version,
                "createdAtTs": info.created_at_ts,
                "metrics": info.metrics or {},
                "description": info.description,
                "current": info.version == current,
            }
            for info in versions_raw
        ]
        available = False
        reason: str | None = "尚未探测"
        if self._scoring is not None:
            probe = self._scoring.score(model_kind, {})
            available = bool(probe.available)
            reason = None if available else probe.reason
        elif not current:
            reason = "尚无已保存模型"
        else:
            available = True
            reason = None
        return ModelKindSummary(
            model_kind=model_kind,
            current_version=current,
            scoring_available=available,
            scoring_reason=reason,
            description=kind_desc,
            versions=versions,
        )

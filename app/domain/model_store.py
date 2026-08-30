"""模型持久化与版本管理端口（S12.2）。

S11 的训练产物（`FittedFraudModel` 等）此前仅在内存中用一次即丢弃，无法「训练一次、
后续多次评分」，也无法版本回滚。本模块定义模型存储抽象端口 `ModelStore`，让训练成功后
的模型可落盘、按版本检索、列出历史版本与回滚，具体落盘实现（joblib）位于基础设施层。

DDD 分层：本模块属 domain 层，仅定义抽象端口与版本元数据值对象，不依赖 joblib/文件系统。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ModelVersionInfo:
    """已保存模型的版本元数据（供版本列表/回滚展示）。

    - model_kind：模型类别（如 ``fraud`` / ``anomaly``），用于隔离不同模型的版本空间。
    - version：版本标识（通常为训练时刻派生的可读版本号，如 ``fraud-20260603...``）。
    - created_at_ts：保存时刻 Unix 秒。
    - metrics：保存时一并记录的评估指标快照（AUC/KS 等），便于版本间对比选优。
    - description：版本备注/用途说明，便于多人协作时理解该版本作用。
    """

    model_kind: str
    version: str
    created_at_ts: int
    metrics: dict = field(default_factory=dict)
    description: str | None = None


class ModelStore(Protocol):
    """模型存储端口：持久化训练模型、按版本/最新检索、列出版本、设定当前版本（回滚）。

    约定：
    - ``save`` 持久化模型对象 + 元数据，并默认将其设为该 model_kind 的当前版本。
    - ``load_latest`` 返回当前版本的模型对象；无任何版本时返回 None。
    - ``load_version`` 按版本号返回模型对象；不存在返回 None。
    - ``list_versions`` 按保存时间倒序返回版本元数据列表。
    - ``set_current`` 将指定版本设为当前版本（回滚/切换）；版本不存在抛 ValueError。
    实现需保证写入原子性与跨进程可见（落盘实现以文件 + 元数据清单达成）。
    """

    def save(
        self, model_kind: str, version: str, model: object, metrics: dict
    ) -> ModelVersionInfo:
        ...

    def load_latest(self, model_kind: str) -> object | None:
        ...

    def load_version(self, model_kind: str, version: str) -> object | None:
        ...

    def list_versions(self, model_kind: str) -> list[ModelVersionInfo]:
        ...

    def set_current(self, model_kind: str, version: str) -> None:
        ...

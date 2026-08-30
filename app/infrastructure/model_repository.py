"""模型持久化与版本管理的基础设施实现（joblib 落盘，S12.2）。

将训练成功后的模型对象（如 `FittedFraudModel`）以 joblib 序列化落盘到模型存储目录，
按 `model_kind` 隔离版本空间，每个版本一个文件 + 一份 JSON 清单记录版本元数据与当前版本。
支持：保存（默认置为当前版本）、按版本/最新加载、版本列表、设定当前版本（回滚）。

序列化优先用 joblib（scikit-learn 模型的推荐序列化方式，PyPI 公共依赖）；环境缺失 joblib
时回退到标准库 pickle，保证最小环境也能持久化（与既有回退策略一致）。

目录结构：
    {model_store_dir}/{model_kind}/
        manifest.json            # { current: 版本号, versions: [ {version, createdAtTs, metrics, file} ] }
        {version}.joblib         # 各版本模型文件

DDD 分层：本模块属 infrastructure 层，实现 domain 层 `ModelStore` 端口。
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from app.domain.model_store import ModelVersionInfo

_MANIFEST_NAME = "manifest.json"


def _dump(obj: object, path: str) -> None:
    """序列化对象到 path：优先 joblib，缺失时回退 pickle。"""
    try:
        import joblib  # 延迟导入

        joblib.dump(obj, path)
    except ImportError:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(obj, fh)


def _load(path: str) -> object:
    """从 path 反序列化对象：优先 joblib，缺失时回退 pickle。"""
    try:
        import joblib  # 延迟导入

        return joblib.load(path)
    except ImportError:
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)


class FileModelRepository:
    """基于文件系统 + joblib 的模型仓储（ModelStore 端口实现）。"""

    def __init__(self, base_dir: str) -> None:
        self._base_dir = base_dir

    # ---- 公共 API（ModelStore 端口）------------------------------------------

    def save(
        self, model_kind: str, version: str, model: object, metrics: dict
    ) -> ModelVersionInfo:
        return self.save_version(model_kind, version, model, metrics, promote_to_current=None)

    def save_version(
        self,
        model_kind: str,
        version: str,
        model: object,
        metrics: dict,
        promote_to_current: bool | None = None,
    ) -> ModelVersionInfo:
        kind_dir = self._kind_dir(model_kind)
        os.makedirs(kind_dir, exist_ok=True)

        file_name = f"{_safe(version)}.joblib"
        _dump(model, os.path.join(kind_dir, file_name))

        created_ts = int(time.time())
        manifest = self._read_manifest(model_kind)
        # 同版本号覆盖既有条目（保留原 description）
        existing = next((v for v in manifest.get("versions", []) if v.get("version") == version), None)
        versions = [v for v in manifest.get("versions", []) if v.get("version") != version]
        # 单调递增序号：保证同秒内多次保存仍可按保存顺序稳定排序（避免秒级精度并列）
        next_seq = max((int(v.get("seq", 0)) for v in versions), default=0) + 1
        entry = {
            "version": version,
            "createdAtTs": created_ts,
            "seq": next_seq,
            "metrics": _json_safe_metrics(metrics),
            "file": file_name,
        }
        if existing and existing.get("description"):
            entry["description"] = existing.get("description")
        versions.append(entry)
        manifest["versions"] = versions

        # IM2：默认不自动设为 current；首个版本仍晋升，避免完全无 current
        promote = promote_to_current
        if promote is None:
            try:
                from app.config import get_settings

                promote = bool(get_settings().auto_promote_on_save)
            except Exception:  # noqa: BLE001
                promote = False
        current = manifest.get("current")
        if promote or not current:
            manifest["current"] = version
        self._write_manifest(model_kind, manifest)

        return ModelVersionInfo(
            model_kind=model_kind,
            version=version,
            created_at_ts=created_ts,
            metrics=dict(metrics),
            description=(existing or {}).get("description") if existing else None,
        )

    def load_latest(self, model_kind: str) -> object | None:
        current = self.current_version(model_kind)
        if not current:
            return None
        return self.load_version(model_kind, current)

    def current_version(self, model_kind: str) -> str | None:
        """返回该 model_kind 的当前版本号（无则 None）。"""
        manifest = self._read_manifest(model_kind)
        current = manifest.get("current")
        return str(current) if current else None

    def load_version(self, model_kind: str, version: str) -> object | None:
        manifest = self._read_manifest(model_kind)
        entry = next(
            (v for v in manifest.get("versions", []) if v.get("version") == version), None
        )
        if entry is None:
            return None
        path = os.path.join(self._kind_dir(model_kind), entry["file"])
        if not os.path.exists(path):
            return None
        return _load(path)

    def list_versions(self, model_kind: str) -> list[ModelVersionInfo]:
        manifest = self._read_manifest(model_kind)
        raw = manifest.get("versions", [])
        infos = [
            (
                int(v.get("seq", 0)),
                ModelVersionInfo(
                    model_kind=model_kind,
                    version=v["version"],
                    created_at_ts=int(v.get("createdAtTs", 0)),
                    metrics=v.get("metrics", {}) or {},
                    description=_optional_str(v.get("description")),
                ),
            )
            for v in raw
        ]
        # 按保存序号倒序（最新在前），序号保证同秒保存也稳定有序
        infos.sort(key=lambda pair: pair[0], reverse=True)
        return [info for _seq, info in infos]

    def set_current(self, model_kind: str, version: str) -> None:
        manifest = self._read_manifest(model_kind)
        exists = any(v.get("version") == version for v in manifest.get("versions", []))
        if not exists:
            raise ValueError(f"模型版本不存在，无法切换/回滚：{model_kind}/{version}")
        manifest["current"] = version
        self._write_manifest(model_kind, manifest)

    def list_kinds(self) -> list[str]:
        """扫描模型存储目录，返回已有 model_kind 列表（有序）。"""
        if not os.path.isdir(self._base_dir):
            return []
        kinds: list[str] = []
        for name in sorted(os.listdir(self._base_dir)):
            kind_path = os.path.join(self._base_dir, name)
            if not os.path.isdir(kind_path):
                continue
            if os.path.isfile(os.path.join(kind_path, _MANIFEST_NAME)):
                kinds.append(name)
        return kinds

    def get_kind_description(self, model_kind: str) -> str | None:
        manifest = self._read_manifest(model_kind)
        return _optional_str(manifest.get("description"))

    def set_kind_description(self, model_kind: str, description: str | None) -> None:
        manifest = self._read_manifest(model_kind)
        text = _normalize_description(description)
        if text is None:
            manifest.pop("description", None)
        else:
            manifest["description"] = text
        self._write_manifest(model_kind, manifest)

    def set_version_description(
        self, model_kind: str, version: str, description: str | None
    ) -> None:
        manifest = self._read_manifest(model_kind)
        found = False
        text = _normalize_description(description)
        for entry in manifest.get("versions", []):
            if entry.get("version") == version:
                found = True
                if text is None:
                    entry.pop("description", None)
                else:
                    entry["description"] = text
                break
        if not found:
            raise ValueError(f"模型版本不存在：{model_kind}/{version}")
        self._write_manifest(model_kind, manifest)

    # ---- 内部工具 ------------------------------------------------------------

    def _kind_dir(self, model_kind: str) -> str:
        return os.path.join(self._base_dir, _safe(model_kind))

    def _manifest_path(self, model_kind: str) -> str:
        return os.path.join(self._kind_dir(model_kind), _MANIFEST_NAME)

    def _read_manifest(self, model_kind: str) -> dict:
        path = self._manifest_path(model_kind)
        if not os.path.exists(path):
            return {"current": None, "versions": []}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data.setdefault("current", None)
                data.setdefault("versions", [])
                return data
        except Exception:  # noqa: BLE001 - 清单损坏时按空清单处理，避免阻断
            pass
        return {"current": None, "versions": []}

    def _write_manifest(self, model_kind: str, manifest: dict) -> None:
        kind_dir = self._kind_dir(model_kind)
        os.makedirs(kind_dir, exist_ok=True)
        path = self._manifest_path(model_kind)
        # 原子写：先写临时文件再替换
        fd, tmp = tempfile.mkstemp(dir=kind_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def _safe(name: str) -> str:
    """将版本/类别名净化为安全文件名片段（仅保留字母数字与 _-.）。"""
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in str(name))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_description(description: str | None) -> str | None:
    if description is None:
        return None
    text = str(description).strip()
    if not text:
        return None
    if len(text) > 512:
        raise ValueError("描述不超过 512 字符")
    return text


def _json_safe_metrics(metrics: dict) -> dict:
    """只保留可 JSON 序列化的指标键值（剔除无法序列化的对象）。"""
    safe: dict = {}
    for key, value in (metrics or {}).items():
        try:
            json.dumps(value)
            safe[key] = value
        except (TypeError, ValueError):
            safe[key] = str(value)
    return safe

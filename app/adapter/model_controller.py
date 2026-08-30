"""模型管理 HTTP 控制器：列表、详情、切换当前生效版本、更新描述。"""

from __future__ import annotations

from typing import Any

from app.domain.model_management import ModelManagementService


class ModelController:
    def __init__(self, service: ModelManagementService) -> None:
        self._service = service

    def list_models(self) -> tuple[int, dict[str, Any]]:
        items = [m.to_api() for m in self._service.list_models()]
        return 200, {"data": items}

    def get_model(self, model_kind: str) -> tuple[int, dict[str, Any]]:
        summary = self._service.get_model(model_kind)
        if summary is None:
            return 404, {"code": "MODEL_KIND_INVALID", "message": "modelKind 无效"}
        return 200, summary.to_api()

    def activate(self, model_kind: str, body: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
        payload = body if isinstance(body, dict) else {}
        version = payload.get("version") or payload.get("modelVersion")
        if version is None or str(version).strip() == "":
            return 400, {
                "code": "INVALID_VERSION",
                "message": "version 不能为空",
                "fields": {"version": "required"},
            }
        try:
            summary = self._service.activate(model_kind, str(version))
        except ValueError as exc:
            return 404, {"code": "MODEL_VERSION_NOT_FOUND", "message": str(exc)}
        return 200, summary.to_api()

    def update_meta(self, model_kind: str, body: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
        """更新模型类别描述和/或指定版本备注。

        - ``{ "description": "欺诈评分，供 MODEL 节点使用" }``
        - ``{ "version": "fraud-xxx", "versionDescription": "补训提升 AUC" }``
        - 二者可同时提交。
        """
        payload = body if isinstance(body, dict) else {}
        has_kind = "description" in payload
        version = payload.get("version") or payload.get("modelVersion")
        has_version = version is not None and str(version).strip() != "" and "versionDescription" in payload
        if not has_kind and not has_version:
            return 400, {
                "code": "INVALID_META",
                "message": "请提供 description，或 version + versionDescription",
            }
        try:
            summary = None
            if has_kind:
                summary = self._service.update_kind_description(
                    model_kind, _as_optional_str(payload.get("description"))
                )
            if has_version:
                summary = self._service.update_version_description(
                    model_kind,
                    str(version),
                    _as_optional_str(payload.get("versionDescription")),
                )
            assert summary is not None
            return 200, summary.to_api()
        except ValueError as exc:
            msg = str(exc)
            code = "MODEL_VERSION_NOT_FOUND" if "不存在" in msg else "INVALID_META"
            status = 404 if code == "MODEL_VERSION_NOT_FOUND" else 400
            return status, {"code": code, "message": msg}


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

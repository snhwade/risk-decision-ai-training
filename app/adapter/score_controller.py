"""在线评分 HTTP 控制器（enhancement-plan T2）。"""

from __future__ import annotations

from typing import Any

from app.domain.online_scoring import OnlineScoringService


class ScoreController:
    def __init__(self, scoring: OnlineScoringService) -> None:
        self._scoring = scoring

    def score(self, body: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
        payload = body if isinstance(body, dict) else {}
        model_ref = payload.get("modelRef") or payload.get("model_ref")
        features = payload.get("features")
        if features is not None and not isinstance(features, dict):
            return 400, {
                "code": "INVALID_FEATURES",
                "message": "features 必须为对象",
                "fields": {"features": "must be object"},
            }
        result = self._scoring.score(
            str(model_ref) if model_ref is not None else None,
            features if isinstance(features, dict) else {},
        )
        # 始终 200：available=false 时由引擎降级，避免 4xx 被当成传输错误
        return 200, result.to_api()

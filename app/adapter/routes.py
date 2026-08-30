"""AI 训练 REST 路由（FastAPI 薄绑定层，R13.1/13.9/13.10/13.11）。

本模块把 HTTP 请求转发给协议无关的 `TrainingJobController`，自身仅做：
- 路由声明（POST/GET /api/v1/ai/training-jobs）；
- 读取原始请求体并以受控方式返回控制器产出的 (status_code, body)。

为保持 domain/adapter 在未安装 FastAPI 的环境仍可导入与测试，FastAPI 在函数内部延迟导入。
请求体解析交由控制器（基于 pydantic）完成，以便对「缺字段/格式非法/范围非法」统一产出
结构化错误体 `{ code, message, fields }`，而非 FastAPI 默认的 422 体。

> 注：本模块刻意不使用 `from __future__ import annotations`。因为 `Request` 在 `build_router`
>     内延迟导入，若注解被惰性化为字符串，新版 FastAPI 在模块全局命名空间解析 `"Request"`
>     会失败，从而把注入参数误判为必填查询参数并返回 422。保留运行期求值的真实类型注解可
>     让 FastAPI 正确识别并注入 Request。
"""

from typing import Any

from app.adapter.controller import TrainingJobController
from app.adapter.model_controller import ModelController
from app.adapter.schedule_controller import TrainingScheduleController
from app.adapter.score_controller import ScoreController

# 在模块全局导入 FastAPI 类型，使路由函数的真实类型注解可被 FastAPI 的 get_type_hints
# 在模块命名空间内正确解析（避免注入参数被误判为查询参数）。未安装 FastAPI 的环境
# （如纯 domain 单元测试）不会导入本模块，故此处的硬导入不影响那些场景。
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


def build_router(
    controller: TrainingJobController,
    schedule_controller: TrainingScheduleController | None = None,
    score_controller: ScoreController | None = None,
    model_controller: ModelController | None = None,
) -> "APIRouter":
    """构建挂载训练任务端点的 FastAPI APIRouter。

    Args:
        controller: 已注入 TrainingService 的控制器实例。
        schedule_controller: 定时训练计划控制器（可选）。
        score_controller: 在线评分控制器（enhancement T2，可选）。
        model_controller: 模型版本管理控制器（可选）。
    Returns:
        fastapi.APIRouter：可被主应用 include_router 挂载。
    """
    router = APIRouter(prefix="/api/v1/ai", tags=["ai-training"])

    @router.post("/training-jobs")
    async def submit_training_job(request: Request) -> JSONResponse:
        """提交训练任务（请求体含数据时间范围，触发 TrainingService 执行）。"""
        body = await _read_json_body(request)
        status_code, payload = controller.submit(body)
        return JSONResponse(status_code=status_code, content=payload)

    @router.get("/training-jobs")
    async def list_training_jobs(request: Request) -> JSONResponse:
        """分页列出训练任务（默认按开始时间降序，可选筛选）。"""
        status_code, payload = controller.list_jobs(dict(request.query_params))
        return JSONResponse(status_code=status_code, content=payload)

    if score_controller is not None:

        @router.post("/score")
        async def score_model(request: Request) -> JSONResponse:
            """在线评分（决策流 MODEL 节点）。"""
            body = await _read_json_body(request)
            status_code, payload = score_controller.score(body)
            return JSONResponse(status_code=status_code, content=payload)

    if model_controller is not None:

        @router.get("/models")
        async def list_models() -> JSONResponse:
            status_code, payload = model_controller.list_models()
            return JSONResponse(status_code=status_code, content=payload)

        @router.get("/models/{model_kind}")
        async def get_model(model_kind: str) -> JSONResponse:
            status_code, payload = model_controller.get_model(model_kind)
            return JSONResponse(status_code=status_code, content=payload)

        @router.put("/models/{model_kind}/current")
        async def activate_model(model_kind: str, request: Request) -> JSONResponse:
            body = await _read_json_body(request)
            status_code, payload = model_controller.activate(model_kind, body)
            return JSONResponse(status_code=status_code, content=payload)

        @router.put("/models/{model_kind}")
        async def update_model_meta(model_kind: str, request: Request) -> JSONResponse:
            body = await _read_json_body(request)
            status_code, payload = model_controller.update_meta(model_kind, body)
            return JSONResponse(status_code=status_code, content=payload)

    if schedule_controller is not None:

        @router.get("/training-schedules")
        async def list_training_schedules() -> JSONResponse:
            status_code, payload = schedule_controller.list_schedules()
            return JSONResponse(status_code=status_code, content=payload)

        @router.post("/training-schedules")
        async def create_training_schedule(request: Request) -> JSONResponse:
            body = await _read_json_body(request)
            status_code, payload = schedule_controller.create(body)
            return JSONResponse(status_code=status_code, content=payload)

        @router.put("/training-schedules/{schedule_id}")
        async def update_training_schedule(schedule_id: int, request: Request) -> JSONResponse:
            body = await _read_json_body(request)
            status_code, payload = schedule_controller.update(schedule_id, body)
            return JSONResponse(status_code=status_code, content=payload)

        @router.delete("/training-schedules/{schedule_id}")
        async def delete_training_schedule(schedule_id: int) -> JSONResponse:
            status_code, payload = schedule_controller.delete(schedule_id)
            return JSONResponse(status_code=status_code, content=payload)

        @router.post("/training-schedules/{schedule_id}/run-now")
        async def run_training_schedule_now(schedule_id: int) -> JSONResponse:
            status_code, payload = schedule_controller.run_now(schedule_id)
            return JSONResponse(status_code=status_code, content=payload)

    return router


async def _read_json_body(request: Any) -> dict:
    """读取并解析请求 JSON 体；非法/空体返回空 dict，交由控制器做字段级校验。"""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - 请求体非 JSON 时按空体处理，交由校验产出字段错误
        return {}
    return body if isinstance(body, dict) else {}

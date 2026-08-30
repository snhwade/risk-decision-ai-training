"""AI 训练服务入口（FastAPI）。

骨架由任务 1.1/17.1 建立；训练任务执行（17.2）、交易对手指标提取写入（17.3）已实现。
本文件在任务 17.4 挂载 AI 训练 REST 路由：
- POST /api/v1/ai/training-jobs：提交训练任务（请求体含数据时间范围，触发 TrainingService）。
- GET  /api/v1/ai/training-jobs：列出训练任务（状态/数据范围/模型版本/评估指标）。
- CRUD /api/v1/ai/training-schedules：定时训练计划管理；后台调度器按 cron 自动触发。

本地运行：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapter.controller import TrainingJobController
from app.adapter.routes import build_router
from app.adapter.schedule_controller import TrainingScheduleController
from app.adapter.score_controller import ScoreController
from app.adapter.model_controller import ModelController
from app.composition import (
    build_model_management_service,
    build_schedule_service,
    build_scoring_service,
    build_training_service,
)
from app.config import get_settings
from app.infrastructure.training_scheduler import TrainingScheduler

_scheduler: TrainingScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    settings = get_settings()
    if settings.scheduler_enabled:
        _scheduler = TrainingScheduler(
            build_schedule_service,
            poll_seconds=settings.scheduler_poll_seconds,
        )
        _scheduler.start()
    yield
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None


app = FastAPI(
    title="AI Training Service",
    description="风控决策平台 - AI 训练服务（旁路增强）",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """健康检查端点。"""
    settings = get_settings()
    return {
        "status": "UP",
        "service": "ai-training-service",
        "minTrainingSamples": settings.min_training_samples,
        "maxTrainingSeconds": settings.max_training_seconds,
        "schedulerEnabled": settings.scheduler_enabled,
    }


_training_service = build_training_service()
_schedule_service = build_schedule_service(training_service=_training_service)
_scoring_service = build_scoring_service()
_model_mgmt_service = build_model_management_service(scoring_service=_scoring_service)
app.include_router(
    build_router(
        TrainingJobController(_training_service),
        TrainingScheduleController(_schedule_service),
        ScoreController(_scoring_service),
        ModelController(_model_mgmt_service),
    )
)

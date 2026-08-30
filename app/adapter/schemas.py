"""AI 训练 REST 适配层的请求/响应模型与序列化（R13.9-13.11）。

使用 pydantic v2（PyPI 公共依赖）做请求体校验与结构化错误生成，输出对齐平台统一
错误体约定 `{ code, message, fields? }`（见设计文档「统一错误响应与异常体系」）。

请求/响应字段统一采用 camelCase（与平台其它 REST 服务一致，如 eventTypeCode、dimensionKey），
内部领域模型使用 snake_case，二者在本模块完成映射。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.domain.training_job import TrainingJob
from app.domain.training_schedule import TrainingSchedule

# 结构化错误码（输入校验类，对应 4xx 语义）
ERROR_CODE_VALIDATION = "VALIDATION_ERROR"


class ListTrainingJobsQuery(BaseModel):
    """训练任务列表查询：默认按 started_at 降序分页。"""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str | None = Field(default=None, alias="jobId")
    status: str | None = None
    start_time_ms: int | None = Field(default=None, alias="startTimeMs")
    end_time_ms: int | None = Field(default=None, alias="endTimeMs")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, alias="pageSize", ge=1, le=200)

    @model_validator(mode="after")
    def _validate_time_range(self) -> "ListTrainingJobsQuery":
        if (
            self.start_time_ms is not None
            and self.end_time_ms is not None
            and self.start_time_ms > self.end_time_ms
        ):
            raise ValueError("起始时间不得晚于结束时间")
        return self

    def start_time(self) -> datetime | None:
        if self.start_time_ms is None:
            return None
        return datetime.fromtimestamp(self.start_time_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)

    def end_time(self) -> datetime | None:
        if self.end_time_ms is None:
            return None
        return datetime.fromtimestamp(self.end_time_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


class SubmitTrainingJobRequest(BaseModel):
    """提交训练任务请求体：数据时间范围（必填，起始不晚于结束）。

    - 接受 camelCase 字段 ``dataFrom`` / ``dataTo``（亦兼容 snake_case，populate_by_name）。
    - 两字段均必填；缺失或类型非法由 pydantic 产出字段级错误。
    - 起始晚于结束时，模型级校验拒绝并给出可读原因（R13.1 触发训练前的入参校验）。
    """

    model_config = ConfigDict(populate_by_name=True)

    data_from: datetime = Field(alias="dataFrom", description="数据时间范围起始")
    data_to: datetime = Field(alias="dataTo", description="数据时间范围结束")

    @model_validator(mode="after")
    def _validate_range(self) -> "SubmitTrainingJobRequest":
        if self.data_from > self.data_to:
            raise ValueError("数据时间范围起始不能晚于结束")
        return self


def _camel_field_name(loc: tuple[Any, ...]) -> str:
    """将 pydantic 错误定位（loc）映射为对外暴露的 camelCase 字段名。

    - ``data_from`` / ``dataFrom`` → ``dataFrom``；``data_to`` / ``dataTo`` → ``dataTo``。
    - 模型级校验（loc 为空）归类到 ``dataRange``，便于前端在时间范围控件上定位。
    """
    if not loc:
        return "dataRange"
    head = str(loc[0])
    mapping = {
        "data_from": "dataFrom",
        "dataFrom": "dataFrom",
        "data_to": "dataTo",
        "dataTo": "dataTo",
    }
    return mapping.get(head, head)


def validation_error_body(exc: ValidationError) -> dict:
    """将 pydantic ValidationError 转换为统一结构化错误体 `{ code, message, fields }`。

    - ``fields``：字段名 → 可读错误信息，供前端将错误映射到对应表单项并保留输入。
    - ``message``：汇总信息，便于无字段定位场景展示。
    """
    fields: dict[str, str] = {}
    for err in exc.errors():
        field = _camel_field_name(tuple(err.get("loc", ())))
        msg = _humanize_error(err)
        # 同一字段保留首个错误信息即可
        fields.setdefault(field, msg)
    return {
        "code": ERROR_CODE_VALIDATION,
        "message": "请求体校验失败：" + "；".join(f"{k}: {v}" for k, v in fields.items()),
        "fields": fields,
    }


def validation_error_body_from_message(message: str, *, field: str = "dataRange") -> dict:
    """据可读消息构造结构化错误体（用于领域层抛出的 ValueError 等非 pydantic 场景）。"""
    return {
        "code": ERROR_CODE_VALIDATION,
        "message": message,
        "fields": {field: message},
    }


def _humanize_error(err: dict) -> str:
    """将 pydantic 原始错误翻译为中文可读信息。"""
    etype = err.get("type", "")
    if etype == "missing":
        return "必填字段缺失"
    if etype.startswith("datetime") or "datetime" in etype:
        return "日期时间格式无效，应为 ISO 8601（如 2024-01-01T00:00:00）"
    if etype == "value_error":
        # 模型级 ValueError：pydantic 会在 msg 前加 "Value error, "
        msg = str(err.get("msg", "取值非法"))
        return msg.replace("Value error, ", "")
    return str(err.get("msg", "字段校验失败"))


def training_job_to_dict(job: TrainingJob) -> dict:
    """将训练任务领域模型序列化为对外 JSON（R13.10/13.11）。

    包含任务状态、所用数据时间范围、模型版本与评估指标；失败任务附带失败类别与原因，
    供 Admin_Console 展示「训练样本不足/训练失败」的原因（R13.11）。
    """
    return {
        "jobId": job.job_id,
        "status": job.status.value,
        "dataFrom": _iso(job.data_from),
        "dataTo": _iso(job.data_to),
        "sampleCount": job.sample_count,
        "modelVersion": job.model_version,
        "metrics": job.metrics,
        "failureKind": job.failure_kind.value if job.failure_kind else None,
        "failReason": job.fail_reason,
        "startedAt": _iso(job.started_at),
        "finishedAt": _iso(job.finished_at),
        "indicatorMetricsWritten": job.indicator_metrics_written,
        "indicatorWriteFailed": job.indicator_write_failed,
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class CreateTrainingScheduleRequest(BaseModel):
    """创建定时训练计划。"""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1, max_length=128)
    cron_expression: str = Field(alias="cronExpression")
    window_days: int = Field(default=30, alias="windowDays", ge=1, le=365)
    enabled: bool = True


class UpdateTrainingScheduleRequest(BaseModel):
    """更新定时训练计划（字段均可选）。"""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = Field(default=None, min_length=1, max_length=128)
    cron_expression: str | None = Field(default=None, alias="cronExpression")
    window_days: int | None = Field(default=None, alias="windowDays", ge=1, le=365)
    enabled: bool | None = None


def training_schedule_to_dict(schedule: TrainingSchedule) -> dict:
    return {
        "id": schedule.id,
        "name": schedule.name,
        "enabled": schedule.enabled,
        "cronExpression": schedule.cron_expression,
        "windowDays": schedule.window_days,
        "lastTriggeredAt": _iso(schedule.last_triggered_at),
        "lastJobId": schedule.last_job_id,
        "lastRunStatus": schedule.last_run_status,
        "lastFailReason": schedule.last_fail_reason,
    }

"""AI 训练任务控制器（与 Web 框架解耦，R13.1/13.9/13.10/13.11）。

`TrainingJobController` 承载 REST 适配层的协议无关逻辑：解析/校验请求体、调用领域服务
`TrainingService`、序列化训练任务记录。刻意不依赖 FastAPI，以便在未安装 Web 框架的
环境中也能用 unittest + 内存替身完整测试路由行为；FastAPI 路由（routes.py）只是把
HTTP 请求转发给本控制器的薄封装。

返回约定：每个处理方法返回 (status_code, body) 二元组，body 为可直接 JSON 序列化的 dict。
- 提交成功：201 + 训练任务记录（含状态/数据范围/模型版本/评估指标）。
- 请求体校验失败：400 + 结构化错误体 `{ code, message, fields }`。

> 注：训练样本不足（R13.6）在领域层被视为一次「已完成、结果为 FAILED」的训练任务，
>     而非请求级校验错误，故仍返回 201 + 任务记录（status=FAILED、failureKind=
>     INSUFFICIENT_SAMPLES、failReason 含原因），由 Admin_Console 据此展示样本不足原因
>     （R13.11）。这与「请求体格式非法（400）」是不同语义。
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.adapter.schemas import (
    ListTrainingJobsQuery,
    SubmitTrainingJobRequest,
    training_job_to_dict,
    validation_error_body,
    validation_error_body_from_message,
)
from app.domain.training_job import TrainingService


class TrainingJobController:
    """训练任务控制器：依赖注入 TrainingService，便于以内存替身做单元测试。"""

    def __init__(self, service: TrainingService) -> None:
        self._service = service

    def submit(self, body: Any) -> tuple[int, dict]:
        """处理 `POST /api/v1/ai/training-jobs`：提交并执行一次训练任务。

        - 校验请求体（dataFrom/dataTo 必填，起始不晚于结束）；失败返回 400 + 结构化错误。
        - 校验通过则调用 TrainingService.execute 触发训练，返回 201 + 任务记录。
        - 训练执行本身不以异常表达业务结果（样本不足/超时/异常均记录在任务记录中返回），
          因此正常情况下返回 201。
        """
        try:
            request = SubmitTrainingJobRequest.model_validate(body)
        except ValidationError as exc:
            return 400, validation_error_body(exc)

        try:
            job = self._service.execute(request.data_from, request.data_to)
        except ValueError as exc:
            # 领域层入参校验（如样本来源对非法时间范围的兜底校验）兜底为 400
            return 400, validation_error_body_from_message(str(exc))

        return 201, training_job_to_dict(job)

    def list_jobs(self, query_params: Any = None) -> tuple[int, dict]:
        """处理 `GET /api/v1/ai/training-jobs`：分页列出训练任务（默认按开始时间降序）。

        支持筛选：jobId、status、startTimeMs/endTimeMs；分页 page/pageSize（1–200）。
        """
        params = query_params or {}
        try:
            query = ListTrainingJobsQuery.model_validate(params)
        except ValidationError as exc:
            return 400, validation_error_body(exc)
        except ValueError as exc:
            return 400, validation_error_body_from_message(str(exc), field="timeRange")

        jobs, total = self._service.list_jobs(
            job_id=query.job_id,
            status=query.status,
            start_time=query.start_time(),
            end_time=query.end_time(),
            page=query.page,
            page_size=query.page_size,
        )
        return 200, {
            "data": [training_job_to_dict(job) for job in jobs],
            "page": query.page,
            "pageSize": query.page_size,
            "total": total,
        }

"""AI 训练 REST 适配层测试（R13.1/13.9/13.10/13.11）。

使用标准库 unittest + 内存替身（内存仓储 / 桩训练器），不连真实数据库与网络：
    python -m unittest discover -s tests
（亦兼容 pytest 运行。）

测试分两层：
1) TrainingJobController（协议无关）：无需安装 FastAPI 即可运行，覆盖核心路由行为。
   - 提交成功返回 201 + 任务记录（含状态/数据范围/模型版本/评估指标）。
   - 样本不足：训练完成但结果 FAILED，返回原因信息（供前端展示，R13.11）。
   - 列表返回字段完整（R13.10）。
   - 请求体校验失败（缺字段 / 范围非法）返回 400 + 结构化错误体 { code, message, fields }。
2) FastAPI TestClient（端到端 HTTP）：仅在已安装 fastapi/httpx 时运行，否则跳过。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from datetime import datetime

# 确保可导入 app 包（tests 与 app 同级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapter.controller import TrainingJobController  # noqa: E402
from app.adapter.schemas import ERROR_CODE_VALIDATION  # noqa: E402
from app.domain.training_job import (  # noqa: E402
    TrainingFailureKind,
    TrainingJobStatus,
    TrainingResult,
    TrainingService,
)
from app.infrastructure.training_job_repository import (  # noqa: E402
    InMemoryTrainingJobRepository,
)

_FROM_ISO = "2024-01-01T00:00:00"
_TO_ISO = "2024-01-31T00:00:00"


# ---- 测试替身 ----------------------------------------------------------------


class _ListSampleSource:
    """按预置样本数量返回固定样本列表（忽略时间范围）。"""

    def __init__(self, count: int) -> None:
        self._samples = [{"merchant_id": f"m{i}", "counterparty_id": f"c{i}"} for i in range(count)]

    def read_range(self, data_from, data_to):
        return self._samples


class _StubTrainer:
    """成功训练替身：返回固定模型版本与评估指标。"""

    def train(self, samples):
        return TrainingResult(model_version="v-rest-1", metrics={"auc": 0.88, "n": len(samples)})


class _RecordingAlarm:
    def __init__(self) -> None:
        self.alarms: list[tuple[str, str]] = []

    def alarm(self, title: str, detail: str) -> None:
        self.alarms.append((title, detail))


def _build_controller(*, sample_count: int, min_samples: int = 10):
    repo = InMemoryTrainingJobRepository()
    service = TrainingService(
        sample_source=_ListSampleSource(sample_count),
        trainer=_StubTrainer(),
        repository=repo,
        alarm=_RecordingAlarm(),
        min_training_samples=min_samples,
    )
    return TrainingJobController(service), repo


# ---- 控制器层测试（无需 FastAPI）---------------------------------------------


class TrainingJobControllerTest(unittest.TestCase):
    def test_submit_success_returns_201_with_job_fields(self):
        # R13.1/13.3/13.10：提交成功，返回任务记录含状态/数据范围/模型版本/评估指标
        controller, repo = _build_controller(sample_count=20, min_samples=10)
        status, body = controller.submit({"dataFrom": _FROM_ISO, "dataTo": _TO_ISO})

        self.assertEqual(status, 201)
        self.assertEqual(body["status"], TrainingJobStatus.SUCCESS.value)
        self.assertEqual(body["dataFrom"], _FROM_ISO)
        self.assertEqual(body["dataTo"], _TO_ISO)
        self.assertEqual(body["modelVersion"], "v-rest-1")
        self.assertEqual(body["metrics"]["auc"], 0.88)
        self.assertIsNone(body["failReason"])
        # 已持久化，可被列表查询到
        self.assertEqual(len(repo.list_all()), 1)

    def test_submit_accepts_snake_case_alias(self):
        # populate_by_name：同时兼容 snake_case 入参
        controller, _repo = _build_controller(sample_count=20, min_samples=10)
        status, body = controller.submit({"data_from": _FROM_ISO, "data_to": _TO_ISO})
        self.assertEqual(status, 201)
        self.assertEqual(body["status"], TrainingJobStatus.SUCCESS.value)

    def test_submit_insufficient_samples_returns_reason(self):
        # R13.6/13.11：样本不足 → 训练完成但 FAILED，返回原因供前端展示
        controller, _repo = _build_controller(sample_count=3, min_samples=10)
        status, body = controller.submit({"dataFrom": _FROM_ISO, "dataTo": _TO_ISO})

        self.assertEqual(status, 201)
        self.assertEqual(body["status"], TrainingJobStatus.FAILED.value)
        self.assertEqual(body["failureKind"], TrainingFailureKind.INSUFFICIENT_SAMPLES.value)
        self.assertIn("训练样本不足", body["failReason"])

    def test_submit_missing_field_returns_400_structured_error(self):
        # 请求体校验：缺 dataTo → 400 + 结构化错误体，字段级定位到 dataTo
        controller, _repo = _build_controller(sample_count=20)
        status, body = controller.submit({"dataFrom": _FROM_ISO})

        self.assertEqual(status, 400)
        self.assertEqual(body["code"], ERROR_CODE_VALIDATION)
        self.assertIn("dataTo", body["fields"])

    def test_submit_invalid_range_returns_400(self):
        # 请求体校验：起始晚于结束 → 400 + 结构化错误体
        controller, _repo = _build_controller(sample_count=20)
        status, body = controller.submit({"dataFrom": _TO_ISO, "dataTo": _FROM_ISO})

        self.assertEqual(status, 400)
        self.assertEqual(body["code"], ERROR_CODE_VALIDATION)
        self.assertIn("dataRange", body["fields"])
        self.assertIn("起始不能晚于结束", body["fields"]["dataRange"])

    def test_submit_invalid_datetime_format_returns_400(self):
        # 请求体校验：日期时间格式非法 → 400，字段定位到 dataFrom
        controller, _repo = _build_controller(sample_count=20)
        status, body = controller.submit({"dataFrom": "not-a-date", "dataTo": _TO_ISO})

        self.assertEqual(status, 400)
        self.assertEqual(body["code"], ERROR_CODE_VALIDATION)
        self.assertIn("dataFrom", body["fields"])

    def test_list_returns_complete_fields(self):
        # R13.10：列表返回各任务状态/数据范围/模型版本/评估指标
        controller, _repo = _build_controller(sample_count=20, min_samples=10)
        controller.submit({"dataFrom": _FROM_ISO, "dataTo": _TO_ISO})

        status, body = controller.list_jobs({})
        self.assertEqual(status, 200)
        self.assertEqual(body["total"], 1)
        self.assertEqual(len(body["data"]), 1)
        item = body["data"][0]
        for key in ("jobId", "status", "dataFrom", "dataTo", "modelVersion", "metrics"):
            self.assertIn(key, item)
        self.assertEqual(item["status"], TrainingJobStatus.SUCCESS.value)
        self.assertEqual(item["modelVersion"], "v-rest-1")

    def test_list_empty_returns_empty_items(self):
        # 无任务时返回空列表（空态语义）
        controller, _repo = _build_controller(sample_count=20)
        status, body = controller.list_jobs({})
        self.assertEqual(status, 200)
        self.assertEqual(body["data"], [])
        self.assertEqual(body["total"], 0)

    def test_list_orders_jobs_by_started_at_desc(self):
        # 列表按开始时间倒序，最新任务在前（R13.10 便于展示）
        controller, _repo = _build_controller(sample_count=20, min_samples=10)
        controller.submit({"dataFrom": _FROM_ISO, "dataTo": _TO_ISO})
        controller.submit({"dataFrom": _FROM_ISO, "dataTo": _TO_ISO})
        status, body = controller.list_jobs({})
        self.assertEqual(status, 200)
        self.assertEqual(body["total"], 2)
        self.assertEqual(len(body["data"]), 2)
        starts = [item["startedAt"] for item in body["data"]]
        self.assertGreaterEqual(starts[0], starts[1])


# ---- FastAPI 端到端测试（仅在已安装 fastapi/httpx 时运行）---------------------

_FASTAPI_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


@unittest.skipUnless(_FASTAPI_AVAILABLE, "未安装 fastapi/httpx，跳过 TestClient 端到端测试")
class TrainingRestEndToEndTest(unittest.TestCase):
    def _build_client(self, *, sample_count: int, min_samples: int = 10):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.adapter.routes import build_router

        repo = InMemoryTrainingJobRepository()
        service = TrainingService(
            sample_source=_ListSampleSource(sample_count),
            trainer=_StubTrainer(),
            repository=repo,
            alarm=_RecordingAlarm(),
            min_training_samples=min_samples,
        )
        app = FastAPI()
        app.include_router(build_router(TrainingJobController(service)))
        return TestClient(app)

    def test_post_then_get_training_jobs(self):
        client = self._build_client(sample_count=20, min_samples=10)
        resp = client.post(
            "/api/v1/ai/training-jobs", json={"dataFrom": _FROM_ISO, "dataTo": _TO_ISO}
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["status"], TrainingJobStatus.SUCCESS.value)

        resp = client.get("/api/v1/ai/training-jobs")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["total"], 1)
        self.assertEqual(len(resp.json()["data"]), 1)

    def test_post_missing_field_returns_400(self):
        client = self._build_client(sample_count=20)
        resp = client.post("/api/v1/ai/training-jobs", json={"dataFrom": _FROM_ISO})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["code"], ERROR_CODE_VALIDATION)

    def test_post_insufficient_samples_returns_reason(self):
        client = self._build_client(sample_count=3, min_samples=10)
        resp = client.post(
            "/api/v1/ai/training-jobs", json={"dataFrom": _FROM_ISO, "dataTo": _TO_ISO}
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["status"], TrainingJobStatus.FAILED.value)
        self.assertIn("训练样本不足", body["failReason"])


if __name__ == "__main__":
    unittest.main()

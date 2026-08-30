"""训练任务持久化（对应 ai_training_job 表，R13.3/13.7/13.10/13.11）。

提供两种实现：
- InMemoryTrainingJobRepository：进程内存储，便于本地运行与单元测试（无数据库依赖）。
- SqlAlchemyTrainingJobRepository：基于 SQLAlchemy 写入 MySQL（生产实现），engine 工厂
  延迟注入，避免在无数据库的测试环境建立真实连接。

ai_training_job 表关键字段：data_from/data_to、status、model_version、metrics、fail_reason。
"""

from __future__ import annotations

import json
from typing import Callable

from app.domain.training_job import TrainingJob, TrainingJobStatus


class InMemoryTrainingJobRepository:
    """进程内训练任务仓储（线程不安全，仅用于本地/测试）。"""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}

    def save(self, job: TrainingJob) -> None:
        # 以 job_id 为键覆盖保存，反映任务从 RUNNING 到终态的演进
        self._jobs[job.job_id] = job

    def find(self, job_id: str) -> TrainingJob | None:
        return self._jobs.get(job_id)

    def list_all(self) -> list[TrainingJob]:
        # 按开始时间倒序，便于任务列表展示（R13.10）
        return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def query(
        self,
        *,
        job_id: str | None = None,
        status: str | None = None,
        start_time=None,
        end_time=None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[TrainingJob], int]:
        from app.domain.training_job import _filter_jobs

        filtered = _filter_jobs(self.list_all(), job_id, status, start_time, end_time)
        total = len(filtered)
        page = max(1, page)
        page_size = min(max(1, page_size), 200)
        start_idx = (page - 1) * page_size
        return filtered[start_idx : start_idx + page_size], total

    def has_running_job(self) -> bool:
        return any(j.status == TrainingJobStatus.RUNNING for j in self._jobs.values())


class SqlAlchemyTrainingJobRepository:
    """基于 SQLAlchemy 的训练任务仓储（写入 MySQL ai_training_job 表）。"""

    def __init__(self, engine_factory: Callable[[], object]) -> None:
        self._engine_factory = engine_factory

    def save(self, job: TrainingJob) -> None:
        from sqlalchemy import text  # 延迟导入，避免测试环境强依赖

        engine = self._engine_factory()
        # 以 job_id 作为业务主键 upsert，保证同一任务从 RUNNING 到终态只有一条记录
        stmt = text(
            "INSERT INTO ai_training_job "
            "(job_id, data_from, data_to, status, model_version, metrics, fail_reason, "
            " started_at, finished_at, sample_count) "
            "VALUES (:job_id, :data_from, :data_to, :status, :model_version, :metrics, "
            " :fail_reason, :started_at, :finished_at, :sample_count) "
            "ON DUPLICATE KEY UPDATE "
            " status=VALUES(status), model_version=VALUES(model_version), "
            " metrics=VALUES(metrics), fail_reason=VALUES(fail_reason), "
            " finished_at=VALUES(finished_at), sample_count=VALUES(sample_count)"
        )
        params = {
            "job_id": job.job_id,
            "data_from": job.data_from,
            "data_to": job.data_to,
            "status": job.status.value,
            "model_version": job.model_version,
            "metrics": json.dumps(job.metrics, ensure_ascii=False) if job.metrics else None,
            "fail_reason": job.fail_reason,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "sample_count": job.sample_count,
        }
        with engine.begin() as conn:  # type: ignore[attr-defined]
            conn.execute(stmt, params)

    def list_all(self) -> list[TrainingJob]:
        """读取全部训练任务（按开始时间倒序），供任务列表查询（R13.10）。"""
        from datetime import datetime  # 延迟导入，避免模块级强依赖

        from sqlalchemy import text  # 延迟导入，避免测试环境强依赖

        from app.domain.training_job import TrainingJobStatus

        engine = self._engine_factory()
        query = text(
            "SELECT job_id, data_from, data_to, status, model_version, metrics, "
            " fail_reason, started_at, finished_at, sample_count "
            "FROM ai_training_job ORDER BY started_at DESC"
        )
        jobs: list[TrainingJob] = []
        with engine.connect() as conn:  # type: ignore[attr-defined]
            for row in conn.execute(query).mappings():
                metrics_raw = row.get("metrics")
                jobs.append(
                    TrainingJob(
                        job_id=row["job_id"],
                        data_from=row.get("data_from") or datetime.min,
                        data_to=row.get("data_to") or datetime.min,
                        status=TrainingJobStatus(row["status"]),
                        started_at=row.get("started_at") or datetime.min,
                        sample_count=row.get("sample_count") or 0,
                        model_version=row.get("model_version"),
                        metrics=json.loads(metrics_raw) if metrics_raw else None,
                        fail_reason=row.get("fail_reason"),
                        finished_at=row.get("finished_at"),
                    )
                )
        return jobs

    def query(
        self,
        *,
        job_id: str | None = None,
        status: str | None = None,
        start_time=None,
        end_time=None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[TrainingJob], int]:
        from datetime import datetime

        from sqlalchemy import text

        from app.domain.training_job import TrainingJobStatus

        page = max(1, page)
        page_size = min(max(1, page_size), 200)
        offset = (page - 1) * page_size

        where_clauses = ["1=1"]
        params: dict = {"limit": page_size, "offset": offset}
        if job_id:
            where_clauses.append("job_id LIKE :job_id")
            params["job_id"] = f"%{job_id}%"
        if status:
            where_clauses.append("status = :status")
            params["status"] = status
        if start_time:
            where_clauses.append("started_at >= :start_time")
            params["start_time"] = start_time
        if end_time:
            where_clauses.append("started_at <= :end_time")
            params["end_time"] = end_time
        where_sql = " AND ".join(where_clauses)

        count_params = {k: v for k, v in params.items() if k not in ("limit", "offset")}

        engine = self._engine_factory()
        count_sql = text(f"SELECT COUNT(1) AS cnt FROM ai_training_job WHERE {where_sql}")
        query_sql = text(
            f"SELECT job_id, data_from, data_to, status, model_version, metrics, "
            f" fail_reason, started_at, finished_at, sample_count "
            f"FROM ai_training_job WHERE {where_sql} "
            f"ORDER BY started_at DESC LIMIT :limit OFFSET :offset"
        )

        with engine.connect() as conn:  # type: ignore[attr-defined]
            total_row = conn.execute(count_sql, count_params).scalar()
            total = int(total_row or 0)
            rows = conn.execute(query_sql, params).mappings()

            jobs: list[TrainingJob] = []
            for row in rows:
                metrics_raw = row.get("metrics")
                jobs.append(
                    TrainingJob(
                        job_id=row["job_id"],
                        data_from=row.get("data_from") or datetime.min,
                        data_to=row.get("data_to") or datetime.min,
                        status=TrainingJobStatus(row["status"]),
                        started_at=row.get("started_at") or datetime.min,
                        sample_count=row.get("sample_count") or 0,
                        model_version=row.get("model_version"),
                        metrics=json.loads(metrics_raw) if metrics_raw else None,
                        fail_reason=row.get("fail_reason"),
                        finished_at=row.get("finished_at"),
                    )
                )
        return jobs, total

    def has_running_job(self) -> bool:
        from sqlalchemy import text

        engine = self._engine_factory()
        stmt = text("SELECT COUNT(1) AS cnt FROM ai_training_job WHERE status = 'RUNNING'")
        with engine.connect() as conn:  # type: ignore[attr-defined]
            count = conn.execute(stmt).scalar()
        return int(count or 0) > 0

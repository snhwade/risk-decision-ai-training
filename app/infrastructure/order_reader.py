"""历史交易订单读取（R13.1）。

从 MySQL 的 risk_order 表按数据时间范围读取历史交易订单数据，供模型训练与
交易对手关系图构建使用。读取逻辑通过可注入的 connection-provider 解耦，
便于在无真实数据库时进行单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Protocol


@dataclass(frozen=True)
class OrderRecord:
    """训练用的订单记录（最小字段集合）。

    context：事件上下文（risk_order.context，JSON 文本），承载交易金额等业务字段，
    是监督式欺诈评分模型的特征来源（R13 AI 增强）。无上下文时为 None。
    """

    event_id: str
    merchant_id: str | None
    event_type_code: str | None
    final_decision: str | None
    event_time: datetime
    context: dict | None = None


def _parse_context(raw: object) -> dict | None:
    """将 risk_order.context（JSON 文本/字典/None）解析为字典，非法或非对象返回 None。"""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        import json

        try:
            parsed = json.loads(text)
        except Exception:  # noqa: BLE001 - 非法 JSON 视为无上下文
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class RowSource(Protocol):
    """行数据来源协议：返回可迭代的行（dict）。便于以内存数据替身做单元测试。"""

    def rows(self, data_from: datetime, data_to: datetime) -> Iterable[dict]:
        ...


class OrderReader:
    """按时间范围读取历史订单。"""

    def __init__(self, row_source: RowSource) -> None:
        self._row_source = row_source

    def read_range(self, data_from: datetime, data_to: datetime) -> list[OrderRecord]:
        if data_from > data_to:
            raise ValueError("data_from 不能晚于 data_to")
        records: list[OrderRecord] = []
        for row in self._row_source.rows(data_from, data_to):
            records.append(
                OrderRecord(
                    event_id=row["event_id"],
                    merchant_id=row.get("merchant_id"),
                    event_type_code=row.get("event_type_code"),
                    final_decision=row.get("final_decision"),
                    event_time=row["event_time"],
                    context=_parse_context(row.get("context")),
                )
            )
        return records


def sqlalchemy_row_source(engine_factory: Callable[[], object]) -> RowSource:
    """构建基于 SQLAlchemy 的行来源（生产实现，按需在运行期注入 engine 工厂）。

    说明：此处仅提供工厂封装；具体 SQL 执行在任务 17.2 的训练流程中按需启用，
    以避免在无数据库的单元测试环境中建立真实连接。
    """

    from sqlalchemy import text  # 延迟导入，避免测试环境强依赖

    class _SqlAlchemyRowSource:
        def rows(self, data_from: datetime, data_to: datetime) -> Iterable[dict]:
            engine = engine_factory()
            query = text(
                "SELECT event_id, merchant_id, event_type_code, final_decision, event_time, context "
                "FROM risk_order WHERE event_time BETWEEN :f AND :t"
            )
            with engine.connect() as conn:  # type: ignore[attr-defined]
                result = conn.execute(query, {"f": data_from, "t": data_to})
                for row in result.mappings():
                    yield dict(row)

    return _SqlAlchemyRowSource()

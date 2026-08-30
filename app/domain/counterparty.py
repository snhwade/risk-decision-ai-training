"""交易对手关系图与交易对手关系指标提取（R13.2 / R13.5 / R13.8）。

本模块承载「训练成功后」基于交易对手之间的交易关系构建关系图、并从图中提取
交易对手关系指标的领域逻辑，遵循设计文档「AI 指标写入链路时序图」：

    训练成功 → 构建交易对手关系图 → 提取交易对手关系指标 → 写入指标存储(可重试)

业务约束：
- R13.2：一次模型训练成功完成后，基于交易对手之间的交易关系提取交易对手关系指标，
  并写入指标存储（Indicator_Store）。
- R13.5：写入指标存储后，规则引擎允许规则引用该类指标（故提取出的指标引用名须满足
  指标定义引用名规范：1..64 且仅 [A-Za-z0-9_]）。
- R13.8：写入失败时最多重试可配置次数（取值范围 1..10，默认 3），仍失败后记录失败原因
  并触发告警，且不影响事件/规则/决策/指标累计等核心功能（训练任务本身仍记为成功）。

DDD 分层：本模块属 domain 层，仅依赖抽象端口（Protocol）与标准库，不直接依赖
networkx / httpx 等具体技术实现。具体实现（networkx 关系图、httpx 写指标存储）位于
基础设施层，并在组合根注入。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Protocol, Sequence

# ---- 指标写入重试次数取值范围（来自 R13.8）------------------------------------

INDICATOR_WRITE_RETRIES_LOWER_BOUND = 1
INDICATOR_WRITE_RETRIES_UPPER_BOUND = 10
INDICATOR_WRITE_RETRIES_DEFAULT = 3

# 指标引用名规范（与 indicator_definition.ref_name 一致：1..64 且仅 [A-Za-z0-9_]）
_REF_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,64}$")

# 交易对手关系指标默认引用名前缀（AI 提取的指标统一以 ai_cp 前缀，便于规则识别与引用）
DEFAULT_REF_NAME_PREFIX = "ai_cp"


def validate_indicator_write_max_retries(value: int) -> int:
    """校验「指标写入最大重试次数」取值范围（R13.8：1..10）。越界抛出 ValueError。"""
    if not INDICATOR_WRITE_RETRIES_LOWER_BOUND <= value <= INDICATOR_WRITE_RETRIES_UPPER_BOUND:
        raise ValueError(
            f"指标写入最大重试次数取值越界：{value}，应在 "
            f"[{INDICATOR_WRITE_RETRIES_LOWER_BOUND}, {INDICATOR_WRITE_RETRIES_UPPER_BOUND}] 之间"
        )
    return value


def validate_ref_name(ref_name: str) -> str:
    """校验指标引用名是否满足规则可引用规范（R13.5）。不合规抛出 ValueError。"""
    if not _REF_NAME_PATTERN.match(ref_name):
        raise ValueError(
            f"指标引用名不合规：{ref_name!r}，须为 1..64 且仅包含 [A-Za-z0-9_]"
        )
    return ref_name


# ---- 领域值对象 ----------------------------------------------------------------


@dataclass(frozen=True)
class CounterpartyTransaction:
    """交易对手关系图的一条边：一笔 source → target 的交易。

    - source/target：交易双方的标识（如商户号、交易对手标识），构成关系图的节点。
    - amount：交易金额（用于金额类关系指标，缺省 0）。
    - event_time：交易发生时间（可用于切片，缺省 None）。
    """

    source: str
    target: str
    amount: float = 0.0
    event_time: datetime | None = None


@dataclass(frozen=True)
class CounterpartyMetric:
    """从交易对手关系图提取出的、可写入指标存储的单条指标。

    字段与指标存储切片模型对齐（refName/dimensionKey/sliceTs/value）：
    - ref_name：指标引用名（规则据此引用，R13.5），如 ``ai_cp_degree``。
    - dimension_key：维度键（通常为交易对手/商户标识）。
    - value：指标值。
    - slice_ts：切片时间戳（Unix epoch 秒）。
    """

    ref_name: str
    dimension_key: str
    value: float
    slice_ts: int


# ---- 抽象端口（Protocol）------------------------------------------------------


class CounterpartyGraphExtractor(Protocol):
    """交易对手关系图指标提取端口：基于训练样本构建关系图并提取交易对手关系指标。

    具体实现（如基于 networkx 的关系图构建）位于基础设施层并在组合根注入；
    单元测试可用内存替身替换。
    """

    def extract(self, samples: Sequence[object]) -> Sequence[CounterpartyMetric]:
        ...


class IndicatorWriter(Protocol):
    """指标写入端口：将单条交易对手关系指标写入指标存储（Indicator_Store）。

    约定：写入失败时抛出异常（由上层据 R13.8 重试/告警）；成功时正常返回。
    具体实现（如基于 httpx 调用 indicator-store-service REST 接口）位于基础设施层。
    """

    def write(self, metric: CounterpartyMetric) -> None:
        ...


# ---- 交易对手关系指标提取（纯 Python 参考实现，无外部依赖，便于测试）-----------


def build_counterparty_metrics(
    transactions: Iterable[CounterpartyTransaction],
    *,
    slice_ts: int,
    ref_name_prefix: str = DEFAULT_REF_NAME_PREFIX,
) -> list[CounterpartyMetric]:
    """从交易对手交易关系中提取交易对手关系指标（R13.2）。

    将每笔交易视为关系图中的一条无向关系边，按节点（交易主体/交易对手）聚合，
    为每个节点产出三类关系指标：

    - ``{prefix}_degree``：交易对手数量（不同对手的去重计数，即节点度数）。
    - ``{prefix}_txn_count``：参与的交易笔数。
    - ``{prefix}_txn_amount``：参与的交易金额合计。

    返回结果按 (ref_name, dimension_key) 升序排序，保证确定性与幂等性
    （对同一份数据多次提取，结果一致——支撑 R13.8 重复写入的一致语义与 17.5 属性测试）。

    边的 source/target 为空白标识时跳过；自环（source==target）仅计为该节点一笔交易，
    不增加交易对手数量。
    """
    neighbors: dict[str, set[str]] = {}
    txn_count: dict[str, float] = {}
    txn_amount: dict[str, float] = {}

    def _touch(node: str) -> None:
        neighbors.setdefault(node, set())
        txn_count.setdefault(node, 0.0)
        txn_amount.setdefault(node, 0.0)

    for txn in transactions:
        source = (txn.source or "").strip()
        target = (txn.target or "").strip()
        if not source or not target:
            # 缺失任一交易主体的边无法构成关系，跳过
            continue
        amount = float(txn.amount or 0.0)

        if source == target:
            # 自环：仅记一笔交易与金额，不增加交易对手数量
            _touch(source)
            txn_count[source] += 1.0
            txn_amount[source] += amount
            continue

        _touch(source)
        _touch(target)
        neighbors[source].add(target)
        neighbors[target].add(source)
        txn_count[source] += 1.0
        txn_count[target] += 1.0
        txn_amount[source] += amount
        txn_amount[target] += amount

    degree_ref = validate_ref_name(f"{ref_name_prefix}_degree")
    count_ref = validate_ref_name(f"{ref_name_prefix}_txn_count")
    amount_ref = validate_ref_name(f"{ref_name_prefix}_txn_amount")

    metrics: list[CounterpartyMetric] = []
    for node in neighbors:
        metrics.append(CounterpartyMetric(degree_ref, node, float(len(neighbors[node])), slice_ts))
        metrics.append(CounterpartyMetric(count_ref, node, txn_count[node], slice_ts))
        metrics.append(CounterpartyMetric(amount_ref, node, txn_amount[node], slice_ts))

    metrics.sort(key=lambda m: (m.ref_name, m.dimension_key))
    return metrics


# ---- 指标写入结果（供训练任务记录与告警，R13.8）-------------------------------


@dataclass
class IndicatorWriteOutcome:
    """一次「批量写入交易对手关系指标」的结果汇总。

    - attempted：尝试写入的指标条数。
    - succeeded：成功写入的指标条数。
    - failures：写入失败（重试耗尽）的 (指标, 最后一次失败原因) 列表。
    """

    attempted: int = 0
    succeeded: int = 0
    failures: list[tuple[CounterpartyMetric, str]] = field(default_factory=list)

    @property
    def has_failure(self) -> bool:
        return bool(self.failures)


def write_metrics_with_retry(
    writer: IndicatorWriter,
    metrics: Sequence[CounterpartyMetric],
    *,
    max_attempts: int,
) -> IndicatorWriteOutcome:
    """逐条写入交易对手关系指标，单条失败最多重试 ``max_attempts`` 次（R13.8）。

    - ``max_attempts`` 为单条指标的最大尝试次数（含首次），取值范围 1..10。
    - 任一条指标在耗尽尝试后仍失败，则记入 ``failures``，但不抛出异常、继续写入其余指标
      （避免单条失败影响其它指标与核心功能）。
    - 返回写入结果汇总，由调用方据此记录失败原因并触发告警。
    """
    validate_indicator_write_max_retries(max_attempts)
    outcome = IndicatorWriteOutcome(attempted=len(metrics))

    for metric in metrics:
        last_error: str | None = None
        for _attempt in range(max_attempts):
            try:
                writer.write(metric)
                outcome.succeeded += 1
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - 写入端可能抛出任意异常
                last_error = str(exc)
        if last_error is not None:
            outcome.failures.append((metric, last_error))

    return outcome

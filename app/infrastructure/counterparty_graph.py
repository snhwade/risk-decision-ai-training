"""交易对手关系图指标提取（基础设施实现，R13.2）。

将训练样本（历史交易订单记录）映射为交易对手之间的交易关系边，构建关系图并提取
交易对手关系指标。关系图的构建优先使用开源库 networkx（PyPI 公共 registry）；
当运行环境未安装 networkx 时，回退到领域层提供的纯 Python 参考实现
（`app.domain.counterparty.build_counterparty_metrics`），二者提取结果在语义上一致。

> 注：networkx 仅用于以「图」的方式组织/校验关系结构；交易对手关系指标（节点度数=交易对手
>     数量、交易笔数、交易金额合计）本身可由两种实现等价计算，故回退不会改变指标取值。

订单记录如何映射为交易对手关系边：
- 训练样本通常为 `app.infrastructure.order_reader.OrderRecord` 或等价的 dict。
- 以 `merchant_id`（交易主体）与上下文中的交易对手标识构成一条边。由于 risk_order 表
  仅固定列含 merchant_id，交易对手标识从订单上下文中按候选键提取（counterparty/payee/
  payer/counterparty_id 等），缺失则跳过该样本。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from app.domain.counterparty import (
    CounterpartyMetric,
    CounterpartyTransaction,
    DEFAULT_REF_NAME_PREFIX,
    build_counterparty_metrics,
)
from app.domain.graph_analytics import (
    RING_SIZE_THRESHOLD_DEFAULT,
    build_graph_analytics_metrics,
)


def _day_slice_ts(now_provider) -> int:
    """返回当前 UTC 自然日 0 点的 Unix 秒切片戳（供各提取器对齐切片）。"""
    now = now_provider()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    day_start = now.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int(day_start.timestamp())

# 从订单记录中识别「交易对手标识」的候选字段名（按优先级）
_COUNTERPARTY_KEYS = (
    "counterparty_id",
    "counterparty",
    "payee",
    "payee_id",
    "payer",
    "payer_id",
)
# 从订单记录中识别「交易金额」的候选字段名
_AMOUNT_KEYS = ("amount", "txn_amount", "trade_amount")


def _get(record: object, *names: str):
    """从 dataclass/对象属性或 dict 中按候选名取值，全部缺失返回 None。

    兼容真实 `OrderRecord`：交易对手/金额等业务字段常存于 ``context`` 字典（risk_order.context），
    顶层仅有 merchant_id。故先查顶层属性/键，未命中再查 ``context`` 字典。
    """
    # 1) 顶层属性 / dict 键
    for name in names:
        if isinstance(record, dict):
            if name in record and record[name] is not None:
                return record[name]
        else:
            value = getattr(record, name, None)
            if value is not None:
                return value
    # 2) 回退到 context 字典（OrderRecord.context 或 dict["context"]）
    context = record.get("context") if isinstance(record, dict) else getattr(record, "context", None)
    if isinstance(context, dict):
        for name in names:
            if name in context and context[name] is not None:
                return context[name]
    return None


def order_records_to_transactions(
    samples: Sequence[object],
) -> list[CounterpartyTransaction]:
    """将训练样本映射为交易对手关系边（source=交易主体，target=交易对手）。

    无法识别交易主体或交易对手的样本将被跳过（不构成关系边）。
    """
    transactions: list[CounterpartyTransaction] = []
    for record in samples:
        source = _get(record, "merchant_id", "source")
        target = _get(record, *_COUNTERPARTY_KEYS)
        if source is None or target is None:
            continue
        amount = _get(record, *_AMOUNT_KEYS)
        event_time = _get(record, "event_time")
        transactions.append(
            CounterpartyTransaction(
                source=str(source),
                target=str(target),
                amount=float(amount) if amount is not None else 0.0,
                event_time=event_time if isinstance(event_time, datetime) else None,
            )
        )
    return transactions


class NetworkxCounterpartyGraphExtractor:
    """基于交易对手关系图提取交易对手关系指标的提取器（CounterpartyGraphExtractor 端口实现）。

    使用 networkx 构建无向加权多重图组织交易关系；若环境缺失 networkx，则回退到领域层
    纯 Python 参考实现，保证可在无第三方依赖的测试环境运行。
    """

    def __init__(
        self,
        *,
        ref_name_prefix: str = DEFAULT_REF_NAME_PREFIX,
        now_provider=None,
    ) -> None:
        self._ref_name_prefix = ref_name_prefix
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def extract(self, samples: Sequence[object]) -> Sequence[CounterpartyMetric]:
        transactions = order_records_to_transactions(samples)
        # 统一以提取时刻所属切片（按天，UTC）作为指标切片时间戳，保证同批次指标对齐
        slice_ts = self._current_day_slice_ts()

        try:
            return self._extract_with_networkx(transactions, slice_ts)
        except ImportError:
            # 环境未安装 networkx：回退纯 Python 参考实现（语义等价）
            return build_counterparty_metrics(
                transactions, slice_ts=slice_ts, ref_name_prefix=self._ref_name_prefix
            )

    def _current_day_slice_ts(self) -> int:
        now = self._now_provider()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        day_start = now.astimezone(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return int(day_start.timestamp())

    def _extract_with_networkx(
        self, transactions: list[CounterpartyTransaction], slice_ts: int
    ) -> list[CounterpartyMetric]:
        """使用 networkx 构建关系图并提取指标（与纯 Python 实现语义一致）。"""
        import networkx as nx  # 延迟导入，缺失时由 extract() 捕获 ImportError 回退

        from app.domain.counterparty import validate_ref_name

        graph = nx.MultiGraph()
        self_loop_count: dict[str, float] = {}
        self_loop_amount: dict[str, float] = {}

        for txn in transactions:
            source = (txn.source or "").strip()
            target = (txn.target or "").strip()
            if not source or not target:
                continue
            amount = float(txn.amount or 0.0)
            if source == target:
                # networkx 的自环会被 degree 计两次，这里单独记账以与参考实现保持一致
                graph.add_node(source)
                self_loop_count[source] = self_loop_count.get(source, 0.0) + 1.0
                self_loop_amount[source] = self_loop_amount.get(source, 0.0) + amount
                continue
            graph.add_edge(source, target, amount=amount)

        degree_ref = validate_ref_name(f"{self._ref_name_prefix}_degree")
        count_ref = validate_ref_name(f"{self._ref_name_prefix}_txn_count")
        amount_ref = validate_ref_name(f"{self._ref_name_prefix}_txn_amount")

        metrics: list[CounterpartyMetric] = []
        for node in graph.nodes:
            # 交易对手数量 = 去重邻居数（不含自身），即简单图意义下的度数
            degree = float(len(set(graph.neighbors(node)) - {node}))
            # 交易笔数 = 关联非自环边数 + 自环笔数
            txn_count = float(graph.degree(node)) + self_loop_count.get(node, 0.0)
            txn_amount = self_loop_amount.get(node, 0.0)
            for _, _, data in graph.edges(node, data=True):
                txn_amount += float(data.get("amount", 0.0))
            metrics.append(CounterpartyMetric(degree_ref, node, degree, slice_ts))
            metrics.append(CounterpartyMetric(count_ref, node, txn_count, slice_ts))
            metrics.append(CounterpartyMetric(amount_ref, node, txn_amount, slice_ts))

        metrics.sort(key=lambda m: (m.ref_name, m.dimension_key))
        return metrics


class GraphAnalyticsExtractor:
    """团伙识别与中心度指标提取器（CounterpartyGraphExtractor 端口实现，AI 增强 S11+）。

    在基础交易对手指标之外，从关系图结构提取「团伙规模/疑似团伙标志/PageRank 中心度」
    三类指标（见 `app.domain.graph_analytics`）。纯 Python 确定性实现，无第三方依赖，
    与基础提取器并行产出、互不影响，由 `CompositeCounterpartyExtractor` 合并写入。
    """

    def __init__(
        self,
        *,
        ref_name_prefix: str = DEFAULT_REF_NAME_PREFIX,
        ring_size_threshold: int = RING_SIZE_THRESHOLD_DEFAULT,
        now_provider=None,
    ) -> None:
        self._ref_name_prefix = ref_name_prefix
        self._ring_size_threshold = ring_size_threshold
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))

    def extract(self, samples: Sequence[object]) -> Sequence[CounterpartyMetric]:
        transactions = order_records_to_transactions(samples)
        slice_ts = _day_slice_ts(self._now_provider)
        return build_graph_analytics_metrics(
            transactions,
            slice_ts=slice_ts,
            ref_name_prefix=self._ref_name_prefix,
            ring_size_threshold=self._ring_size_threshold,
        )


class CompositeCounterpartyExtractor:
    """组合多个交易对手指标提取器（CounterpartyGraphExtractor 端口实现）。

    依次调用各子提取器并合并其产出，便于在「基础关系指标」之外叠加「团伙/中心度指标」，
    且保持端口契约不变（TrainingService 仅依赖单一 extractor 端口）。任一子提取器抛错
    会向上传播，由 TrainingService 的增强链路异常处理捕获并告警（R13.8）。
    """

    def __init__(self, *extractors) -> None:
        self._extractors = extractors

    def extract(self, samples: Sequence[object]) -> Sequence[CounterpartyMetric]:
        merged: list[CounterpartyMetric] = []
        for extractor in self._extractors:
            merged.extend(extractor.extract(samples))
        merged.sort(key=lambda m: (m.ref_name, m.dimension_key))
        return merged

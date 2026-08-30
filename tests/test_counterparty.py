"""交易对手关系图与指标提取写入单元测试（R13.2 / R13.5 / R13.8）。

使用标准库 unittest + 内存/替身，无需真实指标存储或 networkx：
    python -m unittest discover -s tests
（亦兼容 pytest 运行。）

覆盖：
- 交易对手关系指标提取：度数（交易对手数量）/交易笔数/交易金额合计（R13.2）
- 提取结果确定性与幂等（同数据多次提取结果一致，支撑 R13.8 重复写入语义）
- 自环与空白主体边的处理
- 指标引用名规范（R13.5）：合规/越界/非法字符
- 指标写入重试取值范围（R13.8：1..10）
- 写入失败重试到成功（在最大尝试次数内）
- 写入耗尽重试仍失败：记录失败、继续其余、不抛异常
- networkx 提取器与纯 Python 参考实现语义一致
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone

# 确保可导入 app 包（tests 与 app 同级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.domain.counterparty import (  # noqa: E402
    INDICATOR_WRITE_RETRIES_DEFAULT,
    CounterpartyMetric,
    CounterpartyTransaction,
    build_counterparty_metrics,
    validate_indicator_write_max_retries,
    validate_ref_name,
    write_metrics_with_retry,
)
from app.infrastructure.counterparty_graph import (  # noqa: E402
    NetworkxCounterpartyGraphExtractor,
    order_records_to_transactions,
)
from app.infrastructure.indicator_writer import (  # noqa: E402
    HttpxIndicatorWriter,
    IndicatorStoreWriteError,
)


_SLICE = 1_700_000_000


def _txn(source, target, amount=0.0):
    return CounterpartyTransaction(source=source, target=target, amount=amount)


def _metric_map(metrics):
    """转为 {(ref_name, dimension_key): value} 便于断言。"""
    return {(m.ref_name, m.dimension_key): m.value for m in metrics}


# ---- 测试替身 ----------------------------------------------------------------


class _RecordingWriter:
    """记录每次写入调用的内存写入器。"""

    def __init__(self) -> None:
        self.writes: list[CounterpartyMetric] = []

    def write(self, metric: CounterpartyMetric) -> None:
        self.writes.append(metric)


class _FlakyWriter:
    """前 fail_times 次抛异常，其后成功；用于验证重试到成功。"""

    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self.attempts = 0

    def write(self, metric: CounterpartyMetric) -> None:
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise RuntimeError("瞬时网络抖动")


class _AlwaysFailWriter:
    """始终写入失败的写入器，记录总尝试次数。"""

    def __init__(self) -> None:
        self.attempts = 0

    def write(self, metric: CounterpartyMetric) -> None:
        self.attempts += 1
        raise RuntimeError("指标存储不可用")


# ---- 指标提取（R13.2）--------------------------------------------------------


class BuildCounterpartyMetricsTest(unittest.TestCase):
    def test_degree_counts_distinct_counterparties(self):
        # A 与 B、C 交易；B 仅与 A；C 仅与 A
        txns = [_txn("A", "B", 10.0), _txn("A", "C", 20.0)]
        metrics = build_counterparty_metrics(txns, slice_ts=_SLICE)
        mm = _metric_map(metrics)
        self.assertEqual(mm[("ai_cp_degree", "A")], 2.0)
        self.assertEqual(mm[("ai_cp_degree", "B")], 1.0)
        self.assertEqual(mm[("ai_cp_degree", "C")], 1.0)

    def test_txn_count_and_amount_aggregated_per_node(self):
        txns = [_txn("A", "B", 10.0), _txn("A", "C", 20.0)]
        mm = _metric_map(build_counterparty_metrics(txns, slice_ts=_SLICE))
        # A 参与 2 笔，金额 30；B 参与 1 笔，金额 10；C 参与 1 笔，金额 20
        self.assertEqual(mm[("ai_cp_txn_count", "A")], 2.0)
        self.assertEqual(mm[("ai_cp_txn_amount", "A")], 30.0)
        self.assertEqual(mm[("ai_cp_txn_count", "B")], 1.0)
        self.assertEqual(mm[("ai_cp_txn_amount", "B")], 10.0)
        self.assertEqual(mm[("ai_cp_txn_count", "C")], 1.0)
        self.assertEqual(mm[("ai_cp_txn_amount", "C")], 20.0)

    def test_duplicate_counterparty_increases_count_not_degree(self):
        # A↔B 交易两笔：交易对手数量仍为 1，笔数为 2
        txns = [_txn("A", "B", 5.0), _txn("A", "B", 7.0)]
        mm = _metric_map(build_counterparty_metrics(txns, slice_ts=_SLICE))
        self.assertEqual(mm[("ai_cp_degree", "A")], 1.0)
        self.assertEqual(mm[("ai_cp_txn_count", "A")], 2.0)
        self.assertEqual(mm[("ai_cp_txn_amount", "A")], 12.0)

    def test_self_loop_counts_txn_not_degree(self):
        txns = [_txn("A", "A", 9.0)]
        mm = _metric_map(build_counterparty_metrics(txns, slice_ts=_SLICE))
        self.assertEqual(mm[("ai_cp_degree", "A")], 0.0)
        self.assertEqual(mm[("ai_cp_txn_count", "A")], 1.0)
        self.assertEqual(mm[("ai_cp_txn_amount", "A")], 9.0)

    def test_blank_endpoints_skipped(self):
        txns = [_txn("A", "  ", 1.0), _txn("", "B", 1.0), _txn("A", "B", 4.0)]
        mm = _metric_map(build_counterparty_metrics(txns, slice_ts=_SLICE))
        # 仅 A-B 有效
        self.assertEqual(mm[("ai_cp_degree", "A")], 1.0)
        self.assertEqual(mm[("ai_cp_txn_amount", "A")], 4.0)

    def test_extraction_is_deterministic_and_idempotent(self):
        txns = [_txn("A", "B", 10.0), _txn("B", "C", 20.0), _txn("A", "C", 30.0)]
        first = build_counterparty_metrics(txns, slice_ts=_SLICE)
        second = build_counterparty_metrics(list(reversed(txns)), slice_ts=_SLICE)
        # 结果与输入顺序无关，且完全一致（幂等）
        self.assertEqual(first, second)

    def test_custom_ref_name_prefix(self):
        metrics = build_counterparty_metrics(
            [_txn("A", "B", 1.0)], slice_ts=_SLICE, ref_name_prefix="cp"
        )
        refs = {m.ref_name for m in metrics}
        self.assertEqual(refs, {"cp_degree", "cp_txn_count", "cp_txn_amount"})


# ---- 指标引用名规范（R13.5）--------------------------------------------------


class RefNameValidationTest(unittest.TestCase):
    def test_valid_ref_names(self):
        self.assertEqual(validate_ref_name("ai_cp_degree"), "ai_cp_degree")
        self.assertEqual(validate_ref_name("A" * 64), "A" * 64)

    def test_invalid_ref_names_rejected(self):
        with self.assertRaises(ValueError):
            validate_ref_name("")
        with self.assertRaises(ValueError):
            validate_ref_name("A" * 65)
        with self.assertRaises(ValueError):
            validate_ref_name("非法名称")
        with self.assertRaises(ValueError):
            validate_ref_name("has space")


# ---- 写入重试取值范围（R13.8）------------------------------------------------


class RetryRangeValidationTest(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(validate_indicator_write_max_retries(1), 1)
        self.assertEqual(validate_indicator_write_max_retries(10), 10)
        self.assertEqual(
            validate_indicator_write_max_retries(INDICATOR_WRITE_RETRIES_DEFAULT), 3
        )
        with self.assertRaises(ValueError):
            validate_indicator_write_max_retries(0)
        with self.assertRaises(ValueError):
            validate_indicator_write_max_retries(11)


# ---- 写入重试与失败处理（R13.8）----------------------------------------------


class WriteMetricsWithRetryTest(unittest.TestCase):
    def _metrics(self, n=2):
        return build_counterparty_metrics(
            [_txn("A", "B", 1.0)] if n else [], slice_ts=_SLICE
        )[:n]

    def test_all_success_writes_each_once(self):
        writer = _RecordingWriter()
        metrics = self._metrics(3)
        outcome = write_metrics_with_retry(writer, metrics, max_attempts=3)
        self.assertEqual(outcome.attempted, 3)
        self.assertEqual(outcome.succeeded, 3)
        self.assertFalse(outcome.has_failure)
        self.assertEqual(len(writer.writes), 3)

    def test_retry_until_success_within_attempts(self):
        # 前两次失败、第三次成功，max_attempts=3 应最终成功
        writer = _FlakyWriter(fail_times=2)
        metrics = self._metrics(1)
        outcome = write_metrics_with_retry(writer, metrics, max_attempts=3)
        self.assertEqual(outcome.succeeded, 1)
        self.assertFalse(outcome.has_failure)
        self.assertEqual(writer.attempts, 3)

    def test_exhausts_retries_records_failure_without_raising(self):
        writer = _AlwaysFailWriter()
        metrics = self._metrics(2)
        outcome = write_metrics_with_retry(writer, metrics, max_attempts=3)
        self.assertEqual(outcome.succeeded, 0)
        self.assertTrue(outcome.has_failure)
        self.assertEqual(len(outcome.failures), 2)
        # 每条指标尝试 max_attempts 次：2 条 * 3 次 = 6
        self.assertEqual(writer.attempts, 6)
        # 失败项携带原因
        _metric, reason = outcome.failures[0]
        self.assertIn("指标存储不可用", reason)

    def test_partial_failure_continues_other_metrics(self):
        # 第一条失败、其余成功：通过自定义写入器按 dimension_key 区分
        class _SelectiveWriter:
            def __init__(self) -> None:
                self.ok: list[str] = []

            def write(self, metric: CounterpartyMetric) -> None:
                if metric.dimension_key == "A":
                    raise RuntimeError("A 写入失败")
                self.ok.append(metric.dimension_key)

        writer = _SelectiveWriter()
        metrics = build_counterparty_metrics([_txn("A", "B", 1.0)], slice_ts=_SLICE)
        outcome = write_metrics_with_retry(writer, metrics, max_attempts=2)
        # A 的三条指标失败，B 的三条成功
        self.assertEqual(len(outcome.failures), 3)
        self.assertEqual(outcome.succeeded, 3)
        self.assertTrue(all(dk == "B" for dk in writer.ok))

    def test_out_of_range_max_attempts_rejected(self):
        writer = _RecordingWriter()
        with self.assertRaises(ValueError):
            write_metrics_with_retry(writer, self._metrics(1), max_attempts=0)
        with self.assertRaises(ValueError):
            write_metrics_with_retry(writer, self._metrics(1), max_attempts=11)


# ---- 订单记录 → 关系边映射 ----------------------------------------------------


class OrderRecordMappingTest(unittest.TestCase):
    def test_maps_merchant_and_counterparty_from_dict(self):
        samples = [
            {"merchant_id": "M1", "counterparty": "C1", "amount": 100.0},
            {"merchant_id": "M1", "payee": "C2", "txn_amount": 50.0},
        ]
        txns = order_records_to_transactions(samples)
        self.assertEqual(len(txns), 2)
        self.assertEqual(txns[0].source, "M1")
        self.assertEqual(txns[0].target, "C1")
        self.assertEqual(txns[0].amount, 100.0)
        self.assertEqual(txns[1].target, "C2")
        self.assertEqual(txns[1].amount, 50.0)

    def test_skips_records_without_counterparty(self):
        samples = [{"merchant_id": "M1"}, {"merchant_id": "M1", "counterparty": "C1"}]
        txns = order_records_to_transactions(samples)
        self.assertEqual(len(txns), 1)

    def test_maps_from_object_attributes(self):
        class _Rec:
            def __init__(self, merchant_id, counterparty, amount):
                self.merchant_id = merchant_id
                self.counterparty = counterparty
                self.amount = amount

        txns = order_records_to_transactions([_Rec("M1", "C1", 7.0)])
        self.assertEqual(txns[0].source, "M1")
        self.assertEqual(txns[0].target, "C1")
        self.assertEqual(txns[0].amount, 7.0)


# ---- 提取器：networkx 实现与纯 Python 参考实现语义一致 ------------------------


class NetworkxExtractorTest(unittest.TestCase):
    def _fixed_now(self):
        # 固定切片时间，便于断言一致性
        return datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    def test_extractor_matches_reference_implementation(self):
        samples = [
            {"merchant_id": "A", "counterparty": "B", "amount": 10.0},
            {"merchant_id": "B", "counterparty": "C", "amount": 20.0},
            {"merchant_id": "A", "counterparty": "C", "amount": 30.0},
            {"merchant_id": "A", "counterparty": "A", "amount": 5.0},  # 自环
        ]
        extractor = NetworkxCounterpartyGraphExtractor(now_provider=self._fixed_now)
        produced = list(extractor.extract(samples))

        # 与领域层纯 Python 参考实现对比（用同一切片时间戳）
        slice_ts = extractor._current_day_slice_ts()
        reference = build_counterparty_metrics(
            order_records_to_transactions(samples), slice_ts=slice_ts
        )
        self.assertEqual(_metric_map(produced), _metric_map(reference))

    def test_extractor_emits_referenceable_ref_names(self):
        extractor = NetworkxCounterpartyGraphExtractor(now_provider=self._fixed_now)
        metrics = extractor.extract([{"merchant_id": "A", "counterparty": "B"}])
        for m in metrics:
            # R13.5：写入的指标引用名须可被规则引用
            self.assertEqual(validate_ref_name(m.ref_name), m.ref_name)


# ---- httpx 指标写入器（以替身 client 验证契约，不发起真实网络调用）-----------


class HttpxIndicatorWriterTest(unittest.TestCase):
    def test_post_success_no_raise(self):
        captured = {}

        class _Resp:
            status_code = 200

        class _Client:
            def post(self, url, json, timeout):
                captured["url"] = url
                captured["json"] = json
                return _Resp()

        writer = HttpxIndicatorWriter("http://store:8084", client=_Client())
        writer.write(CounterpartyMetric("ai_cp_degree", "M1", 2.0, _SLICE))
        self.assertEqual(captured["url"], "http://store:8084/api/v1/indicators/ai_cp_degree")
        self.assertEqual(captured["json"]["dimensionKey"], "M1")
        self.assertEqual(captured["json"]["value"], 2.0)
        self.assertEqual(captured["json"]["source"], "AI")

    def test_non_2xx_raises_write_error(self):
        class _Resp:
            status_code = 503

        class _Client:
            def post(self, url, json, timeout):
                return _Resp()

        writer = HttpxIndicatorWriter("http://store:8084", client=_Client())
        with self.assertRaises(IndicatorStoreWriteError):
            writer.write(CounterpartyMetric("ai_cp_degree", "M1", 2.0, _SLICE))

    def test_network_exception_raises_write_error(self):
        class _Client:
            def post(self, url, json, timeout):
                raise OSError("连接被拒绝")

        writer = HttpxIndicatorWriter("http://store:8084", client=_Client())
        with self.assertRaises(IndicatorStoreWriteError):
            writer.write(CounterpartyMetric("ai_cp_degree", "M1", 2.0, _SLICE))


if __name__ == "__main__":
    unittest.main()

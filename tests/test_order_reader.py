"""OrderReader 单元测试（R13.1）：使用内存行来源，无需真实数据库。

使用标准库 unittest，便于在未安装 pytest 的环境直接运行：
    python -m unittest discover -s tests
（亦兼容 pytest 运行。）
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime

# 确保可导入 app 包（tests 与 app 同级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.infrastructure.order_reader import OrderReader  # noqa: E402


class _InMemoryRowSource:
    def __init__(self, rows):
        self._rows = rows

    def rows(self, data_from, data_to):
        return [r for r in self._rows if data_from <= r["event_time"] <= data_to]


def _row(event_id, merchant, t):
    return {
        "event_id": event_id,
        "merchant_id": merchant,
        "event_type_code": "B2B_RECV",
        "final_decision": "PASS",
        "event_time": t,
    }


class OrderReaderTest(unittest.TestCase):
    def test_read_range_filters_by_time(self):
        rows = [
            _row("e1", "m1", datetime(2024, 1, 1, 10, 0)),
            _row("e2", "m2", datetime(2024, 1, 5, 10, 0)),
            _row("e3", "m3", datetime(2024, 2, 1, 10, 0)),
        ]
        reader = OrderReader(_InMemoryRowSource(rows))
        result = reader.read_range(datetime(2024, 1, 1), datetime(2024, 1, 31))
        self.assertEqual([r.event_id for r in result], ["e1", "e2"])

    def test_read_range_maps_fields(self):
        rows = [_row("e1", "m1", datetime(2024, 1, 1, 10, 0))]
        reader = OrderReader(_InMemoryRowSource(rows))
        result = reader.read_range(datetime(2024, 1, 1), datetime(2024, 1, 2))
        self.assertEqual(result[0].merchant_id, "m1")
        self.assertEqual(result[0].event_type_code, "B2B_RECV")

    def test_read_range_invalid_range_raises(self):
        reader = OrderReader(_InMemoryRowSource([]))
        with self.assertRaises(ValueError):
            reader.read_range(datetime(2024, 2, 1), datetime(2024, 1, 1))


if __name__ == "__main__":
    unittest.main()

"""在线评分单元测试（enhancement-plan T2）。"""

from __future__ import annotations

import tempfile
import unittest

from app.domain.fraud_model import FittedFraudModel, LogisticRegressionFallback, train_fraud_model
from app.domain.online_scoring import OnlineScoringService, resolve_model_ref
from app.infrastructure.model_repository import FileModelRepository


class _Order:
    def __init__(self, amount: float, decision: str, merchant_id: str = "M1") -> None:
        self.context = {"amount": amount}
        self.final_decision = decision
        self.merchant_id = merchant_id
        self.event_time = None


class OnlineScoringTest(unittest.TestCase):
    def test_resolve_model_ref(self):
        self.assertEqual(resolve_model_ref(None), ("fraud", None))
        self.assertEqual(resolve_model_ref("ai_fraud_score"), ("fraud", None))
        self.assertEqual(resolve_model_ref("fraud@v1"), ("fraud", "v1"))
        self.assertEqual(resolve_model_ref("anomaly"), ("anomaly", None))

    def test_unavailable_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = OnlineScoringService(FileModelRepository(tmp))
            result = svc.score("fraud", {"amount": 100})
            self.assertFalse(result.available)
            self.assertIsNone(result.score)
            self.assertIn("未找到", result.reason or "")

    def test_score_after_train_and_save(self):
        samples = [
            _Order(100, "PASS"),
            _Order(120, "PASS"),
            _Order(90000, "REJECT"),
            _Order(95000, "REJECT"),
            _Order(110, "PASS"),
            _Order(88000, "REJECT"),
        ]
        outcome = train_fraud_model(samples, model_factory=LogisticRegressionFallback)
        self.assertIsInstance(outcome.model, FittedFraudModel)

        with tempfile.TemporaryDirectory() as tmp:
            store = FileModelRepository(tmp)
            store.save("fraud", "v-test-1", outcome.model, outcome.metrics)
            svc = OnlineScoringService(store)

            high = svc.score("fraud", {"amount": 92000})
            self.assertTrue(high.available)
            self.assertIsNotNone(high.score)
            self.assertGreaterEqual(high.score, 0.0)
            self.assertLessEqual(high.score, 1.0)
            self.assertEqual(high.model_version, "v-test-1")

            body = high.to_api()
            self.assertIn("score", body)
            self.assertTrue(body["available"])


if __name__ == "__main__":
    unittest.main()

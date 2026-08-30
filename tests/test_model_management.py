"""模型管理服务与控制器单元测试。"""

from __future__ import annotations

import tempfile
import unittest

from app.adapter.model_controller import ModelController
from app.domain.fraud_model import FittedFraudModel, LogisticRegressionFallback
from app.domain.model_management import ModelManagementService
from app.domain.online_scoring import OnlineScoringService
from app.infrastructure.model_repository import FileModelRepository


class _DummyClassifier(LogisticRegressionFallback):
    pass


class ModelManagementTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = FileModelRepository(self._tmpdir.name)
        self.scoring = OnlineScoringService(self.store)
        self.service = ModelManagementService(self.store, self.scoring)
        self.controller = ModelController(self.service)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_list_empty_defaults(self):
        status, body = self.controller.list_models()
        self.assertEqual(status, 200)
        kinds = {item["modelKind"] for item in body["data"]}
        self.assertIn("fraud", kinds)
        self.assertIn("anomaly", kinds)
        fraud = next(i for i in body["data"] if i["modelKind"] == "fraud")
        self.assertFalse(fraud["scoringAvailable"])
        self.assertEqual(fraud["versions"], [])

    def test_activate_and_list(self):
        model = FittedFraudModel(
            classifier=_DummyClassifier(),
            feature_columns=["amount"],
            feature_baseline=[0.0],
        )
        self.store.save("fraud", "v1", model, {"auc": 0.8, "ks": 0.3})
        self.store.save("fraud", "v2", model, {"auc": 0.9, "ks": 0.4})

        status, body = self.controller.activate("fraud", {"version": "v1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["currentVersion"], "v1")
        self.assertTrue(any(v["version"] == "v1" and v["current"] for v in body["versions"]))

        listed = self.controller.list_models()[1]["data"]
        fraud = next(i for i in listed if i["modelKind"] == "fraud")
        self.assertEqual(fraud["currentVersion"], "v1")
        self.assertTrue(fraud["scoringAvailable"])

    def test_auto_promote_off_by_default(self):
        model = FittedFraudModel(
            classifier=_DummyClassifier(),
            feature_columns=["amount"],
            feature_baseline=[0.0],
        )
        self.store.save("fraud", "v1", model, {"auc": 0.8})
        self.store.save("fraud", "v2", model, {"auc": 0.9})
        # IM2：默认不自动晋升，current 仍为 v1
        self.assertEqual(self.store.current_version("fraud"), "v1")

    def test_auto_promote_explicit(self):
        model = FittedFraudModel(
            classifier=_DummyClassifier(),
            feature_columns=["amount"],
            feature_baseline=[0.0],
        )
        self.store.save_version("fraud", "v1", model, {"auc": 0.8}, promote_to_current=True)
        self.store.save_version("fraud", "v2", model, {"auc": 0.9}, promote_to_current=True)
        self.assertEqual(self.store.current_version("fraud"), "v2")

    def test_activate_missing_version(self):
        status, body = self.controller.activate("fraud", {"version": "nope"})
        self.assertEqual(status, 404)
        self.assertEqual(body["code"], "MODEL_VERSION_NOT_FOUND")

    def test_update_descriptions(self):
        model = FittedFraudModel(
            classifier=_DummyClassifier(),
            feature_columns=["amount"],
            feature_baseline=[0.0],
        )
        self.store.save("fraud", "v1", model, {"auc": 0.8})
        status, body = self.controller.update_meta(
            "fraud",
            {"description": "欺诈评分", "version": "v1", "versionDescription": "首版"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["description"], "欺诈评分")
        self.assertEqual(body["versions"][0]["description"], "首版")


if __name__ == "__main__":
    unittest.main()

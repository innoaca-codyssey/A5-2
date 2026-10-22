"""동일 모델의 실제 입력과 화면 출력 사이의 일치를 확인한다."""

from pathlib import Path
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from analysis_shap import explain
from finance import FEATURES
from prepare import digest

ROOT = Path(__file__).resolve().parents[1]


class ExplanationTests(unittest.TestCase):
    def test_provenance_and_additivity(self):
        metadata = json.loads((ROOT / "models/metadata.json").read_text())
        self.assertEqual(digest(ROOT / "finance_data.csv"), metadata["data_sha256"])
        self.assertEqual(
            digest(ROOT / "models/classifier.joblib"), metadata["model_sha256"]
        )
        raw = pd.read_csv(ROOT / "finance_data.csv")[FEATURES].iloc[[0, 1, 2, 3]]
        model = joblib.load(ROOT / "models/classifier.joblib")
        exp, prob, error = explain(model, raw)
        self.assertEqual(exp.values.shape, (4, 6))
        np.testing.assert_allclose(
            exp.base_values + exp.values.sum(axis=1), prob, atol=1e-6
        )
        self.assertLess(np.abs(error).max(), 1e-6)

    def test_approval_and_rejection_form(self):
        cases = json.loads((ROOT / "outputs/shap-analysis.json").read_text())["cases"]
        app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.number_input), 6)
        for case in cases:
            for widget, feature in zip(app.number_input, FEATURES):
                value = case["features"][feature]
                widget.set_value(
                    float(value) if feature == "debt_ratio" else int(value)
                )
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, f"{case['probability']:.2%}")
            self.assertEqual(app.dataframe[0].value.shape, (6, 3))
            self.assertTrue(
                any(
                    case["decision"] == "approval"
                    and "승인" in x.value
                    or case["decision"] == "rejection"
                    and "거절" in x.value
                    for x in app.markdown
                )
            )

import json, tempfile, threading, unittest, urllib.request, urllib.error
import numpy as np
from churn.data import make_data, FEATURES
from churn.drift import psi, drift_report
from churn.registry import Registry
from churn.pipeline import initial_train, monitor_and_retrain
from churn.serve import serve

class TestData(unittest.TestCase):
    def test_seeded_reproducible(self):
        a, ya = make_data(500, seed=1); b, yb = make_data(500, seed=1)
        self.assertTrue(a.equals(b) and ya.equals(yb))
    def test_columns_and_label_rate(self):
        X, y = make_data(5000); self.assertEqual(list(X.columns), FEATURES); self.assertTrue(0.1 < y.mean() < 0.9)

class TestDrift(unittest.TestCase):
    def test_psi_identical_is_near_zero(self):
        x = np.random.default_rng(0).normal(size=5000); self.assertLess(psi(x, x), 1e-6)
    def test_no_false_alarm_same_distribution(self):
        a, _ = make_data(8000, seed=1); b, _ = make_data(8000, seed=2)
        self.assertFalse(drift_report(a, b)["_any_drift"])
    def test_detects_shift(self):
        a, _ = make_data(8000, seed=1); b, _ = make_data(8000, seed=2, drift=0.8)
        self.assertTrue(drift_report(a, b)["_any_drift"])

class TestPipelineAndServe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); cls.reg = Registry(cls.d)
        X, y = make_data(6000, seed=0); cls.v, cls.met = initial_train(cls.reg, X, y)
        cls.srv, cls.state = serve(cls.d, port=0); cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()
    def _post(self, body):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}/predict", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r).read())
    def test_model_quality_floor(self): self.assertGreater(self.met["auc"], 0.60)  # oracle ceiling on this synthetic data is ~0.735
    def test_health(self):
        self.assertEqual(json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health").read())["status"], "ok")
    def test_predict_valid(self):
        X, _ = make_data(1, seed=5); out = self._post(X.iloc[0].to_dict())
        self.assertTrue(0 <= out["churn_probability"] <= 1)
    def test_predict_missing_feature_400(self):
        with self.assertRaises(urllib.error.HTTPError) as c: self._post({"tenure_months": 3})
        self.assertEqual(c.exception.code, 400)
    def test_registry_lineage(self):
        X, y = make_data(4000, seed=9, drift=0.9); out = monitor_and_retrain(self.reg, X, y)
        self.assertTrue(out["retrained"]); self.assertEqual(self.reg.meta["versions"][-1]["parent_version"], 1)

if __name__ == "__main__": unittest.main()

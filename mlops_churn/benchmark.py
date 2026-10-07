"""Runs the whole project end-to-end and writes measured results to results.json."""
import json, time, tempfile, threading, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics import roc_auc_score
from churn.data import make_data, FEATURES
from churn.drift import drift_report
from churn.registry import Registry
from churn.pipeline import initial_train, monitor_and_retrain
from churn.serve import serve

R = {}
root = tempfile.mkdtemp(); reg = Registry(root)

# 1. baseline training
X, y = make_data(20000, seed=0)
t = time.perf_counter(); v, met = initial_train(reg, X, y); R["baseline"] = {**met, "total_pipeline_seconds": round(time.perf_counter() - t, 2), "rows": 20000}

Xh, yh = make_data(20000, seed=4242)
ph = reg.load(1)[0].predict_proba(Xh[FEATURES])[:, 1]; top = np.argsort(-ph)[:len(ph)//10]
R["baseline"]["top_decile_lift"] = round(float(yh.values[top].mean() / yh.mean()), 2)
R["baseline"]["base_churn_rate_pct"] = round(100 * float(yh.mean()), 1)
R["baseline"]["auc_on_fresh_20k"] = round(roc_auc_score(yh, ph), 4)
# 2. drift detector: false-alarm rate and detection rate by drift severity (batch=5000)
fa = sum(drift_report(X, make_data(5000, seed=100 + s)[0])["_any_drift"] for s in range(50))
R["drift_false_alarm_rate"] = {"batches": 50, "alarms": fa, "rate_pct": round(100 * fa / 50, 1)}
det = {}
for lvl in (0.1, 0.2, 0.4, 0.8):
    hits = [drift_report(X, make_data(5000, seed=500 + s, drift=lvl)[0]) for s in range(30)]
    det[str(lvl)] = {"detection_rate_pct": round(100 * sum(h["_any_drift"] for h in hits) / 30, 1),
                     "avg_features_flagged": round(statistics.mean(h["_n_drifted"] for h in hits), 2)}
R["drift_detection_by_severity"] = det

# 3. end-to-end drift -> retrain -> champion/challenger gate
Xn, yn = make_data(8000, seed=999, drift=0.8)
t = time.perf_counter(); out = monitor_and_retrain(reg, Xn, yn); sec = round(time.perf_counter() - t, 2)
R["retrain_cycle"] = {"seconds_drift_check_plus_retrain": sec, "features_flagged": out["drift"]["_n_drifted"], "of": len(FEATURES),
                      "champion_auc_on_drifted_holdout": out["champion_auc_same_holdout"], "challenger_auc": out["challenger_auc"],
                      "auc_lift": round(out["challenger_auc"] - out["champion_auc_same_holdout"], 4), "promoted_version": out.get("promoted")}
# non-drift batch must NOT retrain
# no-drift check must use a fresh registry whose champion was trained on non-drifted data
fresh = Registry(tempfile.mkdtemp()); initial_train(fresh, X, y)
nd = [monitor_and_retrain(fresh, *make_data(8000, seed=2000 + s))["retrained"] for s in range(20)]
R["no_drift_batches_retrain_triggered"] = {"batches": 20, "retrains": sum(nd)}

# 4. serving load test (reset to v1 so we also exercise hot-reload)
reg2 = Registry(root); reg2.promote(1)
srv, state = serve(root, port=0); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
Xs, _ = make_data(500, seed=7); rows = [json.dumps(r).encode() for r in Xs.to_dict("records")]
def call(i):
    t = time.perf_counter()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/predict", rows[i % 500], {"Content-Type": "application/json"})
    urllib.request.urlopen(r).read(); return time.perf_counter() - t
for i in range(50): call(i)  # warmup
lat = [call(i) for i in range(2000)]
def pct(a, p): a = sorted(a); return round(a[int(len(a) * p) - 1] * 1e3, 2)
R["serving_sequential_2000_req"] = {"p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95), "p99_ms": pct(lat, .99)}
t = time.perf_counter()
with ThreadPoolExecutor(8) as ex: lat8 = list(ex.map(call, range(2000)))
wall = time.perf_counter() - t
R["serving_8_concurrent_2000_req"] = {"throughput_rps": round(2000 / wall, 1), "p50_ms": pct(lat8, .5), "p95_ms": pct(lat8, .95), "errors": state.errors}
# hot reload: promote v2 and reload with no downtime
reg3 = Registry(root); reg3.promote(2)
before = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health").read())["model_version"]
t = time.perf_counter(); urllib.request.urlopen(f"http://127.0.0.1:{port}/reload").read(); rl = (time.perf_counter() - t) * 1e3
after = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health").read())["model_version"]
R["hot_reload"] = {"from_version": before, "to_version": after, "reload_ms": round(rl, 1)}
srv.shutdown()
R["env"] = {"cpus": 1, "note": "single shared sandbox CPU; synthetic data; stdlib HTTP server"}
json.dump(R, open("results.json", "w"), indent=2); print(json.dumps(R, indent=2))

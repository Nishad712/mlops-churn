"""Orchestration steps. Plain functions; each is Prefect-ready (see flows/prefect_flow.py)."""
import pandas as pd
from sklearn.metrics import roc_auc_score
from .data import FEATURES
from .train import train, data_hash
from .drift import drift_report

def initial_train(reg, X, y):
    m, met = train(X, y)
    v = reg.register(m, met, X[FEATURES], data_hash(X, y)); reg.promote(v); return v, met

def monitor_and_retrain(reg, X_new, y_new, min_auc_gain=0.0):
    """Check drift on an incoming batch; if drifted, train a challenger and promote it only if it beats
    the champion on a held-out slice of fresh labelled data."""
    champ, cv = reg.load()
    ref = pd.read_csv(reg.root / f"reference_v{cv}.csv")
    rep = drift_report(ref, X_new[FEATURES])
    champ_auc = roc_auc_score(y_new, champ.predict_proba(X_new[FEATURES])[:, 1])
    out = {"drift": rep, "champion_version": cv, "champion_auc_on_new": round(champ_auc, 4), "retrained": False}
    if rep["_any_drift"]:
        cut = int(len(X_new) * 0.75)
        Xa, ya, Xb, yb = X_new.iloc[:cut], y_new.iloc[:cut], X_new.iloc[cut:], y_new.iloc[cut:]
        m, met = train(Xa, ya)
        ch_auc = roc_auc_score(yb, m.predict_proba(Xb[FEATURES])[:, 1])
        champ_auc_b = roc_auc_score(yb, champ.predict_proba(Xb[FEATURES])[:, 1])
        v = reg.register(m, {**met, "holdout_auc": round(ch_auc, 4)}, Xa[FEATURES], data_hash(Xa, ya), parent=cv)
        out.update(challenger_version=v, challenger_auc=round(ch_auc, 4), champion_auc_same_holdout=round(champ_auc_b, 4))
        if ch_auc > champ_auc_b + min_auc_gain:
            reg.promote(v); out["promoted"] = v
        out["retrained"] = True
    return out

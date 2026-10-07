"""Data-drift detection: Population Stability Index + two-sample KS test per feature."""
import numpy as np
from scipy.stats import ks_2samp

def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))

def drift_report(ref_df, cur_df, psi_thresh=0.2, ks_alpha=0.01):
    rep = {}
    for col in ref_df.columns:
        p = psi(ref_df[col].values, cur_df[col].values)
        ks = ks_2samp(ref_df[col].values, cur_df[col].values)
        rep[col] = {"psi": round(p, 4), "ks_p": float(ks.pvalue), "drifted": bool(p > psi_thresh and ks.pvalue < ks_alpha)}
    feats = [v for k, v in rep.items()]
    rep["_any_drift"] = any(v["drifted"] for v in feats)
    rep["_n_drifted"] = sum(v["drifted"] for v in feats)
    return rep

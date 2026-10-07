import hashlib, time
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from .data import FEATURES

def data_hash(X, y):
    return hashlib.sha256(pd.util.hash_pandas_object(X).values.tobytes() + y.values.tobytes()).hexdigest()[:12]

def train(X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    t = time.perf_counter()
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed).fit(Xtr[FEATURES], ytr)
    secs = time.perf_counter() - t
    p = m.predict_proba(Xte[FEATURES])[:, 1]
    return m, {"auc": round(roc_auc_score(yte, p), 4), "f1": round(f1_score(yte, p > 0.5), 4),
               "train_seconds": round(secs, 2), "n_train": len(Xtr)}

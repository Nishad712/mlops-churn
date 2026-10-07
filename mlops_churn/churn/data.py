"""Synthetic customer-churn data generator (seeded, with controllable drift)."""
import numpy as np
import pandas as pd

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "usage_gb", "contract_months", "late_payments"]

def make_data(n=20000, seed=0, drift=0.0):
    """drift in [0,1] shifts feature distributions and changes the label mechanism (covariate + concept drift)."""
    rng = np.random.default_rng(seed)
    tenure = rng.gamma(4, 8, n) * (1 - 0.4 * drift)
    charges = rng.normal(70 + 25 * drift, 20, n).clip(15, 200)
    calls = rng.poisson(2 + 3 * drift, n)
    usage = rng.lognormal(3 - 0.5 * drift, 0.6, n)
    contract = rng.choice([1, 12, 24], n, p=[0.5 - 0.3 * drift, 0.3, 0.2 + 0.3 * drift])
    late = rng.poisson(0.7 + 1.2 * drift, n)
    logit = (-1.0 - 0.04 * tenure + 0.015 * (charges - 70) + 0.35 * calls
             - 0.01 * usage - 0.03 * contract + 0.4 * late)
    logit = logit - 0.03 * drift * usage + 0.5 * drift * (charges > 90)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"tenure_months": tenure, "monthly_charges": charges, "support_calls": calls,
                       "usage_gb": usage, "contract_months": contract, "late_payments": late})
    return df, pd.Series(y, name="churn")

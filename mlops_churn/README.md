# Churn MLOps pipeline
Synthetic churn data -> train -> versioned registry -> drift detection (PSI + KS) -> champion/challenger retrain -> inference API with hot-reload.
Run: `python -m unittest discover -s tests -t .` | `python benchmark.py` (writes results.json)
Executed & measured: everything under `churn/`, `tests/`, `benchmark.py`.
Written but NOT executed (no Docker/K8s/Prefect/GitHub in build sandbox): `Dockerfile`, `k8s/`, `.github/workflows/`, `flows/`.

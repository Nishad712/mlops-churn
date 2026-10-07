"""Prefect wrapper (NOT executed in the build sandbox - Prefect wasn't installable there. Run it locally: pip install prefect)."""
from prefect import flow, task
from churn.registry import Registry
from churn.data import make_data
from churn.pipeline import monitor_and_retrain

@task(retries=2, retry_delay_seconds=10)
def fetch_batch(seed): return make_data(8000, seed=seed, drift=0.8)

@task
def check_and_retrain(X, y): return monitor_and_retrain(Registry("registry"), X, y)

@flow(name="churn-drift-monitor")
def drift_monitor(seed: int = 999):
    X, y = fetch_batch(seed); return check_and_retrain(X, y)

if __name__ == "__main__":
    drift_monitor.serve(name="daily-drift-check", cron="0 2 * * *")

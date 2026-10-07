"""Tiny file-based model registry: versioned artifacts + metadata + lineage + 'production' alias."""
import json, time, joblib
from pathlib import Path

class Registry:
    def __init__(self, root="registry"):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "registry.json"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {"versions": [], "production": None}

    def _save(self): self.meta_path.write_text(json.dumps(self.meta, indent=2))

    def register(self, model, metrics, reference_df, data_hash, parent=None):
        v = len(self.meta["versions"]) + 1
        joblib.dump(model, self.root / f"model_v{v}.joblib")
        reference_df.to_csv(self.root / f"reference_v{v}.csv", index=False)
        self.meta["versions"].append({"version": v, "metrics": metrics, "data_hash": data_hash,
                                      "parent_version": parent, "created": time.time()})
        self._save(); return v

    def promote(self, v): self.meta["production"] = v; self._save()
    def production(self): return self.meta["production"]
    def load(self, v=None):
        v = v or self.production(); return joblib.load(self.root / f"model_v{v}.joblib"), v

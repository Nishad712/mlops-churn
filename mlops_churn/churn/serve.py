"""Inference service (stdlib http.server): /predict, /health, /metrics, /reload (hot-swap to production version)."""
import json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pandas as pd
from .data import FEATURES
from .registry import Registry

class State:
    def __init__(self, reg_root):
        self.root = reg_root; self.lock = threading.Lock()
        self.model, self.version = Registry(reg_root).load(); self.lat = []; self.errors = 0; self.requests = 0
    def maybe_reload(self):
        reg = Registry(self.root)
        if reg.production() != self.version:
            m, v = reg.load()
            with self.lock: self.model, self.version = m, v

def make_handler(state):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass
        def _send(self, code, obj):
            b = json.dumps(obj).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path == "/health": return self._send(200, {"status": "ok", "model_version": state.version})
            if self.path == "/reload": state.maybe_reload(); return self._send(200, {"model_version": state.version})
            if self.path == "/metrics":
                l = sorted(state.lat) or [0.0]
                return self._send(200, {"requests": state.requests, "errors": state.errors,
                                        "p50_ms": l[len(l) // 2] * 1e3, "p95_ms": l[max(int(len(l) * .95) - 1, 0)] * 1e3})
            self._send(404, {"error": "not found"})
        def do_POST(self):
            t = time.perf_counter(); state.requests += 1
            try:
                n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n))
                missing = [f for f in FEATURES if f not in body]
                if missing: raise ValueError(f"missing features: {missing}")
                with state.lock: m, v = state.model, state.version
                p = float(m.predict_proba(pd.DataFrame([body])[FEATURES])[0, 1])
                self._send(200, {"churn_probability": round(p, 4), "model_version": v})
            except Exception as e:
                state.errors += 1; self._send(400, {"error": str(e)})
            state.lat.append(time.perf_counter() - t)
    return H

def serve(reg_root="registry", port=8080):
    s = State(reg_root); return ThreadingHTTPServer(("127.0.0.1", port), make_handler(s)), s

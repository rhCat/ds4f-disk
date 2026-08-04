#!/usr/bin/env python3
"""serve-demo.py -- honest structure-demo endpoint for ds4f-disk.

Runs the real engine on the real checkpoint on demand and returns the
run report + state hash as JSON. This is a STRUCTURE demo: the engine
has no attention/tokenizer/sampling, so no text is produced -- the
value is proving the disk-streaming pipeline end to end on the 304B
checkpoint from an HTTP request.

Usage (on the acer, after convert+quantize):
  python3 tools/serve-demo.py \
      --model-dir ~/ds4f-mxfp4 \
      --trunk ~/ds4f-model/trunk.bin --offsets ~/ds4f-model/trunk.offsets \
      --pool ~/ds4f-mxfp4/pool-mxfp4.bin \
      --layout-trunk ~/ds4f-model/trunk.json \
      --layout-pool ~/ds4f-mxfp4/pool-mxfp4.json \
      --port 8734

Endpoints:
  GET /            status + last run summary
  GET /run?gen=5   run the engine (gen tokens), return JSON report
Stdlib only. Concurrency: one run at a time (subprocess, serialized).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
LAST = {"status": "idle", "report": None}
ARGS = None


def run_engine(gen):
    """Run ./ds4f, parse the report lines, hash the state dump."""
    cmd = [os.path.join(ARGS.repo, "ds4f"), ARGS.model_dir,
           "--trunk", ARGS.trunk, "--offsets", ARGS.offsets,
           "--pool", ARGS.pool,
           "--layout-trunk", ARGS.layout_trunk,
           "--layout-pool", ARGS.layout_pool,
           "--preset", ARGS.preset, "--gen", str(gen),
           "--dump-state", os.path.join(ARGS.model_dir, "state-demo.bin")]
    if ARGS.no_simd:
        cmd.append("--no-simd")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    out = r.stderr + r.stdout

    report = {"exit": r.returncode, "stdout": r.stdout[-2000:],
              "stderr": r.stderr[-8000:]}
    m = re.search(r"(\d+) tokens in ([\d.]+) s, ([\d.]+) s/token", out)
    if m:
        report["tokens"] = int(m.group(1))
        report["seconds"] = float(m.group(2))
        report["s_per_token"] = float(m.group(3))
    m = re.search(r"GB read per token: ([\d.]+)", out)
    if m:
        report["gb_per_token"] = float(m.group(1))
    m = re.search(r"cache: (\d+) requests, (\d+) hits \(([\d.]+)%\), (\d+) dropped", out)
    if m:
        report.update(requests=int(m.group(1)), hits=int(m.group(2)),
                      hit_pct=float(m.group(3)), dropped=int(m.group(4)))
    m = re.search(r"router: real matvec on (\d+)/(\d+) layers", out)
    if m:
        report["real_router_layers"] = int(m.group(1))
    m = re.search(r"kernels: (simd|scalar)", out)
    if m:
        report["kernels"] = m.group(1)
    m = re.search(r"PEAK RSS: ([\d.]+) GB", out)
    if m:
        report["peak_rss_gb"] = float(m.group(1))
    m = re.search(r"moe: (\d+) matvecs, (\d+) decoded", out)
    if m:
        report["matvecs"] = int(m.group(1))
        report["decoded_elements"] = int(m.group(2))

    sp = os.path.join(ARGS.model_dir, "state-demo.bin")
    if os.path.exists(sp) and r.returncode == 0:
        h = hashlib.sha256()
        with open(sp, "rb") as f:
            h.update(f.read())
        report["state_sha256"] = h.hexdigest()
        report["state_bytes"] = os.path.getsize(sp)
    return report


class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/status":
            self._json({"app": "ds4f-disk structure demo",
                        "model": "DeepSeek-V4-Flash-class (304B)",
                        "last": LAST})
            return
        m = re.match(r"^/run\?gen=(\d+)$", self.path)
        if not m:
            self._json({"error": "try /run?gen=5 or /status"}, 404)
            return
        gen = max(1, min(int(m.group(1)), 20))
        if not LOCK.acquire(blocking=False):
            self._json({"error": "a run is already in progress"}, 429)
            return
        try:
            LAST["status"] = "running"
            report = run_engine(gen)
            LAST["status"] = "done"
            LAST["report"] = report
            self._json(report)
        except subprocess.TimeoutExpired:
            LAST["status"] = "timeout"
            self._json({"error": "engine timed out"}, 500)
        except Exception as e:  # noqa: BLE001
            LAST["status"] = "error"
            self._json({"error": str(e)}, 500)
        finally:
            LOCK.release()

    def log_message(self, fmt, *args):  # quiet
        print(f"[serve-demo] {self.address_string()} {fmt % args}")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--offsets", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--layout-trunk", required=True)
    ap.add_argument("--layout-pool", required=True)
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--preset", default="laptop")
    ap.add_argument("--port", type=int, default=8734)
    ap.add_argument("--no-simd", action="store_true")
    ARGS = ap.parse_args()
    print(f"ds4f-disk structure demo on :{ARGS.port} "
          f"(model {ARGS.model_dir})")
    ThreadingHTTPServer(("0.0.0.0", ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()

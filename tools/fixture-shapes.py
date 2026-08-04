#!/usr/bin/env python3
import json, subprocess, tempfile, os, struct, glob
S = tempfile.mkdtemp()
subprocess.run(["python3", "tools/convert-ds4f.py", "make-synthetic", f"{S}/src"],
               cwd=os.path.expanduser("~/ds4f-disk"), capture_output=True)
idx = json.load(open(f"{S}/src/model.safetensors.index.json"))
wm = idx["weight_map"]
for k in sorted(wm):
    if "ffn.up" in k or "ffn.down" in k or "gate" in k:
        shard = wm[k]
        # read shape from the safetensors header
        p = f"{S}/src/{shard}"
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            print(k, "->", hdr[k]["dtype"], hdr[k]["shape"])

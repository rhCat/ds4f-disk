#!/usr/bin/env python3
import json, subprocess, tempfile, os, shutil
S = tempfile.mkdtemp()
subprocess.run(["python3", "tools/convert-ds4f.py", "make-synthetic", f"{S}/src"],
               cwd=os.path.expanduser("~/ds4f-disk"), capture_output=True)
os.makedirs(f"{S}/wrap")
shutil.move(f"{S}/src", f"{S}/wrap/model")
r = subprocess.run(["python3", "tools/convert-ds4f.py", "convert", f"{S}/wrap",
                    "--out", f"{S}/out"],
                   cwd=os.path.expanduser("~/ds4f-disk"), capture_output=True, text=True)
print("convert rc:", r.returncode)
print("--- files in out:")
for f in sorted(os.listdir(f"{S}/out")):
    print(" ", f, os.path.getsize(f"{S}/out/{f}"))

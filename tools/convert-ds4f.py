#!/usr/bin/env python3
"""convert-ds4f.py -- inventory and convert a DeepSeek-V4-Flash-class HF
checkpoint into ds4f-disk's layout:

  config.json       ds4f keys (no-defaults: missing key = refuse)
  trunk.bin         dense per-layer bytes, contiguous
  trunk.offsets     [u64 n][u64 off][u64 size] per layer
  pool.bin          routed experts, fixed-rate, layer-major:
                    [u64 expert_nbytes][u64 n_layers][u64 n_experts]
                    then expert 0 of layer 0 .. expert N of layer M
  manifest.json     what was mapped and what was left out

Discovery-first: `inspect` prints the actual tensor layout and the
proposed mapping. Paste that back before trusting `convert` -- naming
conventions differ between releases and the converter REFUSES rather
than guesses. Stdlib only, runs on the box.

usage:
  convert-ds4f.py inspect DIR
  convert-ds4f.py convert DIR --out OUT
  convert-ds4f.py make-synthetic DIR     (tiny fake HF repo for tests)
"""
import json
import os
import re
import struct
import sys

ALIASES = {
    "n_layers": ["num_hidden_layers", "n_layers", "n_layer", "num_layers"],
    "n_experts": ["n_routed_experts", "num_routed_experts", "n_experts",
                  "moe_n_experts", "num_experts"],
    "topk": ["num_experts_per_tok", "num_experts_per_token", "topk",
             "n_active_experts"],
    "n_shared": ["n_shared_experts", "num_shared_experts"],
    "hidden": ["hidden_size", "n_embd", "d_model"],
    "latent": ["latent_size", "moe_latent_size", "latent", "q_lora_rank"],
    "moe_inter": ["moe_intermediate_size", "intermediate_size",
                  "ffn_hidden_size", "moe_ffn_hidden_size"],
}
REQUIRED = ["n_layers", "n_experts", "topk", "n_shared", "hidden"]

EXPERT_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:mlp|ffn)\.experts\.(\d+)\.(.+)$")
SHARED_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:mlp|ffn)\.shared_expert\.(.+)$")
LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")


def skeleton(name):
    """Name with every digit run collapsed to N, e.g.
    layers.0.attn.wq_a.weight -> layers.N.attn.wq_a.weight. The
    histogram of skeletons reveals a checkpoint's naming scheme at a
    glance without dumping tens of thousands of names."""
    return re.sub(r"\d+", "N", name)


def hsize(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TB"


# ----------------------------------------------------------------------
# safetensors / HF repo reading
# ----------------------------------------------------------------------

def read_safetensors_index(path):
    """Return {tensor_name: (dtype, shape, [a, b])} with byte spans
    relative to the payload, plus payload_begin = 8 + header_len."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return {}, 0
            hlen = struct.unpack("<Q", raw)[0]
            hdr = f.read(hlen)
            if len(hdr) != hlen:
                return {}, 0
        idx = json.loads(hdr)
    except (json.JSONDecodeError, OSError):
        return {}, 0
    out = {}
    for name, meta in idx.items():
        if not isinstance(meta, dict) or "data_offsets" not in meta:
            continue
        a, b = meta["data_offsets"]
        out[name] = (meta.get("dtype", "?"), meta.get("shape", []), [a, b])
    return out, 8 + hlen


def discover(dirpath):
    """Return (shards {name: (abspath, off_in_file, nbytes)},
    index_path, config dict)."""
    index_path = os.path.join(dirpath, "model.safetensors.index.json")
    shards = {}
    config = {}
    cfg_path = os.path.join(dirpath, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)

    weight_map = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            ij = json.load(f)
        weight_map = ij.get("weight_map", {})

    files = sorted(f for f in os.listdir(dirpath) if f.endswith(".safetensors"))
    if not files:
        return {}, index_path, config

    begins = {}
    for fn in files:
        idx, pb = read_safetensors_index(os.path.join(dirpath, fn))
        begins[fn] = pb
        for name, (dtype, shape, [a, b]) in idx.items():
            shards[name] = (os.path.join(dirpath, fn), pb + a, b - a)

    if weight_map:  # index.json is authoritative about placement
        shards = {}
        for name, fn in weight_map.items():
            fn = os.path.basename(fn)
            if fn not in begins:
                continue
            idx, pb = read_safetensors_index(os.path.join(dirpath, fn))
            if name in idx:
                a, b = idx[name][2]
                shards[name] = (os.path.join(dirpath, fn), pb + a, b - a)
    return shards, index_path, config


def looks_like_repo(d):
    return (os.path.exists(os.path.join(d, "config.json")) or
            any(f.endswith(".safetensors")
                for f in os.listdir(d)) if os.path.isdir(d) else False)


def find_repo_root(dirpath, maxdepth=5):
    """If dirpath itself is not an HF repo, walk down (bounded) to find
    the deepest-looking subdir with config.json or .safetensors --
    HF git clones put the real files under snapshots/<hash>/. Returns
    None when nothing repo-like exists anywhere."""
    if os.path.isdir(dirpath) and looks_like_repo(dirpath):
        return dirpath
    import collections
    q = collections.deque([(dirpath, 0)])
    found = []
    while q:
        d, depth = q.popleft()
        if depth >= maxdepth:
            continue
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for e in entries:
            p = os.path.join(d, e)
            if os.path.isdir(p) and not os.path.islink(p):
                if looks_like_repo(p):
                    found.append(p)
                q.append((p, depth + 1))
    return sorted(found, key=len)[0] if found else None


def tree_summary(dirpath):
    """What is actually in this directory, when it is not an HF repo."""
    try:
        entries = sorted(os.listdir(dirpath))
    except OSError as ex:
        print(f"cannot list {dirpath}: {ex}")
        return
    print(f"contents of {dirpath} ({len(entries)} entries):")
    exts = {}
    for e in entries:
        p = os.path.join(dirpath, e)
        if os.path.isdir(p):
            print(f"  [dir]  {e}/")
            continue
        try:
            sz = os.path.getsize(p)
        except OSError:
            sz = -1
        ext = os.path.splitext(e)[1].lower() or "(none)"
        exts[ext] = exts.get(ext, 0) + 1
        print(f"  [file] {e}  ({hsize(sz)})")
    if exts:
        print("extensions: " + ", ".join(f"{k} x{v}" for k, v in
                                         sorted(exts.items())))
    if ".gguf" in exts:
        print("note: GGUF found -- ds4f reads safetensors; convert to "
              "safetensors first (llama.cpp/convert_hf_to_gguf reverse)")
    if entries and not exts:
        print("note: only directories -- HF snapshot layout?")


# ----------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------

def cmd_inspect(dirpath):
    root = find_repo_root(dirpath)
    if root != dirpath:
        tree_summary(dirpath)
        if root:
            print(f"\nfound repo-like content under: {root}")
            dirpath = root
        else:
            print("\nno repo-like content found anywhere (no config.json, "
                  "no .safetensors). If this is a git-lfs clone, run "
                  "`git lfs pull` first, then re-inspect.")
            return
    shards, index_path, config = discover(dirpath)
    print(f"repo: {dirpath}")
    print(f"shards: {len(set(os.path.basename(f) for f, _, _ in shards.values()))}"
          f" safetensors file(s), {len(shards)} tensors")
    print(f"index.json: {'present' if os.path.exists(index_path) else 'absent'}")
    print(f"config.json: {'present' if config else 'ABSENT'}")

    mapped, assumptions = map_config(config)
    print("\nconfig mapping (HF key -> ds4f key):")
    for key in REQUIRED + ["latent", "moe_inter"]:
        v = mapped.get(key)
        print(f"  {key:10s} = {v if v is not None else 'MISSING (refuse)'}")
    if assumptions:
        print("  (latent/moe_inter assumed = hidden_size; reported, not silent)")

    if not shards:
        print("\nno safetensors found; nothing else to classify")
        return

    experts, dense, shared, other = classify(shards)
    eb = None
    if experts:
        per_layer = {}
        for (L, e), tens in experts.items():
            per_layer.setdefault(L, {})[e] = sum(t[3] for t in tens)
        sizes = {s for L in per_layer for s in per_layer[L].values()}
        uniform = len(sizes) == 1
        eb = next(iter(sizes)) if uniform else None
        print(f"\nrouted experts: {len(experts)} across {len(per_layer)} layers")
        if uniform:
            print(f"  per-expert payload: {hsize(eb)} (uniform, fixed-rate OK)")
        else:
            print(f"  NOT UNIFORM ({len(sizes)} distinct sizes) -> cannot "
                  f"build a fixed-rate pool; refuse")
        for (L, e) in sorted(experts)[:3]:
            print(f"  e({L},{e}): " +
                  ", ".join(f"{t[0].split('.')[-1]}={hsize(t[3])}"
                            for t in experts[(L, e)]))
    else:
        print("\nrouted experts: NONE matched the .experts.{e}. pattern")

    trunk_bytes = sum(t[3] for tens in dense.values() for t in tens)
    print(f"dense per-layer tensors: {len(dense)} layers, "
          f"{hsize(trunk_bytes)} total (excl. routed experts)")
    if dense:
        L0 = min(dense)
        print(f"  sample layer {L0}: " +
              ", ".join(f"{t[0].split('.')[-1]}={hsize(t[3])}"
                        for t in sorted(dense[L0])[:8]))
    if shared:
        print(f"shared experts: {hsize(sum(t[3] for ts in shared.values() for t in ts))}"
              f" (resident)")
    if other:
        ob = sum(t[3] for t in other)
        print(f"unclassified (embed/norm/lm_head/...): {hsize(ob)}")
        for t in other[:10]:
            print(f"  {t[0]} ({hsize(t[3])})")

    # the naming scheme, at a glance: digit runs collapsed to N
    from collections import Counter
    skel = Counter(skeleton(n) for n in shards)
    print("\nname skeletons (digits -> N), most common:")
    for sk, cnt in skel.most_common(15):
        print(f"  {cnt:7d}  {sk}")
    if skel and len(skel) > 15:
        print(f"  ... {len(skel) - 15} more skeletons")

    if eb:
        print(f"\nestimate: pool.bin {hsize(eb * len(experts))}, "
              f"trunk.bin {hsize(trunk_bytes)}")
        print("  (bytes copied as-is; dtypes preserved, no quantization)")


def classify(shards):
    """experts: {(L, e): [(name, file, off, nbytes)]}
    dense: {L: [(...)]}  (layer tensors minus experts)
    shared: {L: [(...)]}, other: [(...)]."""
    experts, dense, shared, other = {}, {}, {}, []
    for name, (fn, off, nb) in shards.items():
        m = EXPERT_RE.search(name)
        if m:
            experts.setdefault((int(m.group(1)), int(m.group(2))),
                               []).append((name, fn, off, nb))
            continue
        m = SHARED_RE.search(name)
        if m:
            shared.setdefault(int(m.group(1)), []).append((name, fn, off, nb))
            continue
        m = LAYER_RE.match(name)
        if m:
            dense.setdefault(int(m.group(1)), []).append((name, fn, off, nb))
            continue
        other.append((name, fn, off, nb))
    return experts, dense, shared, other


def map_config(config):
    mapped = {}
    assumptions = []
    for key, aliases in ALIASES.items():
        src = next((a for a in aliases if a in config), None)
        if src is not None:
            mapped[key] = int(config[src])
        elif key in REQUIRED:
            mapped[key] = None
        else:
            assumptions.append(key)
    if "latent" in assumptions and mapped.get("hidden"):
        mapped["latent"] = mapped["hidden"]
        assumptions.remove("latent")
    if "moe_inter" in assumptions and mapped.get("hidden"):
        mapped["moe_inter"] = mapped["hidden"]
        assumptions.remove("moe_inter")
    return mapped, assumptions


# ----------------------------------------------------------------------
# convert
# ----------------------------------------------------------------------

def copy_range(fsrc, off, nb, fdst, buf):
    fsrc.seek(off)
    left = nb
    while left > 0:
        got = fsrc.read(min(len(buf), left))
        if not got:
            raise RuntimeError("short read")
        fdst.write(got)
        left -= len(got)


def put_u64(f, v):
    f.write(struct.pack("<Q", v))


def cmd_convert(dirpath, outdir):
    shards, _, config = discover(dirpath)
    mapped, assumptions = map_config(config)
    missing = [k for k in REQUIRED if not mapped.get(k)]
    if missing:
        print(f"REFUSE: config keys missing: {', '.join(missing)} "
              f"(run inspect first)")
        sys.exit(1)

    experts, dense, shared, other = classify(shards)
    if not experts:
        print("REFUSE: no routed experts matched the .experts.{e}. pattern")
        sys.exit(1)
    layers = sorted({L for L, _ in experts})
    if len(layers) != mapped["n_layers"]:
        print(f"REFUSE: config says {mapped['n_layers']} layers, found "
              f"{len(layers)} with experts")
        sys.exit(1)
    for L in layers:
        if L not in dense:
            print(f"REFUSE: layer {L} has experts but no dense tensors")
            sys.exit(1)

    per_layer = {L: {} for L in layers}
    for (L, e), tens in experts.items():
        per_layer[L][e] = sum(t[3] for t in tens)
    sizes = {s for L in per_layer for s in per_layer[L].values()}
    if len(sizes) != 1:
        print(f"REFUSE: expert payloads not uniform: {sorted(sizes)}")
        sys.exit(1)
    expert_bytes = next(iter(sizes))
    n_experts = mapped["n_experts"]
    for L in layers:
        if len(per_layer[L]) != n_experts:
            print(f"REFUSE: layer {L} has {len(per_layer[L])} experts, "
                  f"config says {n_experts}")
            sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    # ---------------- config.json ----------------
    cfg_out = {
        "n_layers": mapped["n_layers"],
        "n_experts": mapped["n_experts"],
        "topk": mapped["topk"],
        "n_shared": mapped["n_shared"],
        "hidden": mapped["hidden"],
        "latent": mapped["latent"],
        "moe_inter": mapped["moe_inter"],
        "expert_nbytes": expert_bytes,
        "seed": 7,
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    print(f"config.json written (expert_nbytes={expert_bytes})")

    # ---------------- trunk.bin + trunk.offsets ----------------
    srcs = {}
    def src(name):
        fn, off, nb = shards[name]
        if fn not in srcs:
            srcs[fn] = open(fn, "rb")
        return srcs[fn], off, nb

    tbin = open(os.path.join(outdir, "trunk.bin"), "wb")
    toff = open(os.path.join(outdir, "trunk.offsets"), "wb")
    put_u64(toff, len(layers))
    buf = bytearray(1 << 20)
    at = 0
    for L in layers:
        tens = sorted(dense[L])
        lay_bytes = sum(t[3] for t in tens)
        put_u64(toff, at)
        put_u64(toff, lay_bytes)
        for t in tens:
            f, off, nb = src(t[0])
            copy_range(f, off, nb, tbin, buf)
        at += lay_bytes
        print(f"  trunk layer {L}: {hsize(lay_bytes)}")
    tbin.close()
    toff.close()
    print(f"trunk.bin {hsize(at)} ({len(layers)} layers)")

    # ---------------- pool.bin ----------------
    pbin = open(os.path.join(outdir, "pool.bin"), "wb")
    put_u64(pbin, expert_bytes)
    put_u64(pbin, len(layers))
    put_u64(pbin, n_experts)
    written = 0
    done = 0
    total = len(layers) * n_experts
    for L in layers:
        for e in range(n_experts):
            tens = sorted(experts[(L, e)])
            for t in tens:
                f, off, nb = src(t[0])
                copy_range(f, off, nb, pbin, buf)
            written += expert_bytes
            done += 1
            if done % 2000 == 0:
                print(f"  pool: {done}/{total} experts, {hsize(written)}")
    pbin.close()
    print(f"pool.bin {hsize(written)} ({done} experts x {hsize(expert_bytes)})")

    for f in srcs.values():
        f.close()

    # ---------------- manifest.json ----------------
    manifest = {
        "config": cfg_out,
        "assumptions": assumptions,
        "layers": len(layers),
        "expert_bytes": expert_bytes,
        "trunk_bytes": at,
        "pool_bytes": written,
        "shared_expert_bytes": sum(t[3] for ts in shared.values() for t in ts),
        "unclassified_tensors": [(t[0], t[3]) for t in other],
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # verify round trip
    if os.path.getsize(os.path.join(outdir, "pool.bin")) != 24 + written:
        print("VERIFY FAIL: pool size mismatch")
        sys.exit(1)
    if os.path.getsize(os.path.join(outdir, "trunk.bin")) != at:
        print("VERIFY FAIL: trunk size mismatch")
        sys.exit(1)
    print("\nconvert complete. run:")
    print(f"  ./ds4f {outdir} --trunk {outdir}/trunk.bin "
          f"--offsets {outdir}/trunk.offsets --pool {outdir}/pool.bin "
          f"--preset laptop")


# ----------------------------------------------------------------------
# make-synthetic: tiny fake HF repo (2 layers, 4 experts, 2 shards)
# ----------------------------------------------------------------------

def cmd_make_synthetic(dirpath):
    os.makedirs(dirpath, exist_ok=True)
    cfg = {
        "num_hidden_layers": 2,
        "hidden_size": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 16,
        "model_type": "deepseek_moe",
    }
    with open(os.path.join(dirpath, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    names = ["embed.weight", "lm_head.weight"]
    for L in range(2):
        names += [
            f"layers.{L}.hc_attn_base",
            f"layers.{L}.hc_ffn_base",
            f"layers.{L}.attn.attn_sink",
            f"layers.{L}.attn.wq_a.weight",
            f"layers.{L}.attn.wq_a.scale",
            f"layers.{L}.ffn.gate",
            f"layers.{L}.ffn.shared_expert.up_proj",
            f"layers.{L}.ffn.shared_expert.down_proj",
        ]
        for e in range(4):
            names += [
                f"layers.{L}.ffn.experts.{e}.up_proj",
                f"layers.{L}.ffn.experts.{e}.down_proj",
                f"layers.{L}.ffn.experts.{e}.gate_proj",
            ]
    names += ["norm.weight"]

    def blob(name, n):
        x = 0x1234
        out = bytearray()
        for i in range(n):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            out.append((x + ord(name[0])) & 0xFF)
        return bytes(out)

    sizes = {}
    for i, name in enumerate(names):
        if "experts." in name:
            sizes[name] = 64          # 3 tensors x 64 B -> expert 192 B
        elif "shared_expert" in name:
            sizes[name] = 32
        elif "embed" in name:
            sizes[name] = 96
        elif name.endswith("gate"):
            sizes[name] = 16
        else:
            sizes[name] = 32 + (i % 3) * 16

    n_shards = 2
    wm = {}
    for s in range(n_shards):
        snames = names[s::n_shards]
        payload = b""
        entries = {}
        for name in snames:
            a = len(payload)
            payload += blob(name, sizes[name])
            entries[name] = {"dtype": "F32",
                             "shape": [sizes[name] // 4],
                             "data_offsets": [a, a + sizes[name]]}
            wm[name] = f"model-0000{s + 1}-of-0000{n_shards}.safetensors"
        hdr = json.dumps(entries).encode()
        with open(os.path.join(dirpath,
                               f"model-0000{s + 1}-of-0000{n_shards}.safetensors"),
                  "wb") as f:
            f.write(struct.pack("<Q", len(hdr)))
            f.write(hdr)
            f.write(payload)
    with open(os.path.join(dirpath, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": sum(sizes.values())},
                   "weight_map": wm}, f, indent=2)
    print(f"synthetic HF repo at {dirpath}: {len(names)} tensors, "
          f"{n_shards} shards")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "inspect":
        cmd_inspect(sys.argv[2])
    elif cmd == "convert":
        if "--out" not in sys.argv:
            print(__doc__)
            sys.exit(1)
        cmd_convert(sys.argv[2], sys.argv[sys.argv.index("--out") + 1])
    elif cmd == "make-synthetic":
        cmd_make_synthetic(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

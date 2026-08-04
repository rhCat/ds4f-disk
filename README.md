# ds4f-disk

Disk-streaming MoE inference **structure** in portable C99 for
DeepSeek-V4-Flash-class models. No BLAS, no framework, no GPU, no
dependencies beyond libc + pthreads.

The design mirrors the family proven by
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) and
[Colibrì](https://github.com/JustVugg/colibri): a tiny in-RAM pointer map,
a contiguous packed trunk, and policy-controlled expert streaming. This
repository is the **structure, validated on fixtures** — the I/O machinery,
the cache, the trace tooling, and the memory gate are real and tested; real
weights drop in behind the same interfaces.

## Why this shape (the argument in one table)

| Lever | Decision | Why |
|---|---|---|
| Addressing | `(layer, expert) -> file offset`, O(1), fixed-rate payloads | variable-rate (entropy) coding breaks random access |
| Trunk | pinned prefix + ring, **deliberately not a cache** | a cyclic scan makes LRU hit rate *exactly zero*; pinning first N gives N/layers deterministically |
| Experts | 3-phase fetch: reserve serially / read in parallel sorted by disk offset / publish only what arrived | keeps the device queue deep; INFLIGHT slots cannot be double-claimed |
| Precision | full where resident (trunk, router, shared), 4-bit where streamed | precision mirrors access: resident is free, streamed is bandwidth |
| Memory | sum everything before allocating; refuse past 95% | the plan is a forecast, not a result — quote measured peak RSS |
| Policy | trace-replay (routing does not depend on the cache) | one run, replay at any capacity under LRU / Belady / PIN+LRU before building anything |

## Layout

```
include/ds4f/ds4f.h     public API + the five invariants
src/json.h              minimal JSON (config + safetensors headers)
src/cfg.c               config reader that refuses to guess
src/st.c                safetensors index -> pointer map, expert pool
src/trunk.c             packed trunk: pin + ring + async prefetch
src/cache.c             routed-expert cache: 3-phase fetch
src/router.c            top-k router (biased score selects, unbiased weights)
src/mem.c               memory planner + peak-RSS measurement
src/main.c              CLI: memory plan, decode loop, run report
tools/make-fixture.c    tiny synthetic model (no real weights needed)
tools/pack-trunk.c      layer files -> contiguous trunk.bin + offsets
tools/convert-ds4f.py   HF checkpoint -> ds4f layout (inspect/convert)
tools/trace_replay.py   LRU / Belady / PIN+LRU vs capacity
tests/                  weightless gate suite + fixture e2e
```

## Build and verify

```sh
make            # ds4f, pack-trunk, make-fixture
make test       # gate suite (weightless) + fixture e2e + trace replay
```

## Quickstart (fixture, no real weights)

```sh
./make-fixture --dir fixture --layers 6 --experts 24 --topk 3 \
    --hidden 128 --latent 64 --moe-inter 128 \
    --expert-bytes 8192 --trunk-bytes 16384 --seed 7

./ds4f fixture --trunk fixture/trunk.bin --offsets fixture/trunk.offsets \
    --gen 5 --trace fixture/trace.csv --cache-gb 1 --pin-layers 3

python3 tools/trace_replay.py fixture/trace.csv --caps 1,2,4
```

Presets: `--preset laptop` (8 GB cache / 4 GB trunk) or `--preset server`
(64 / 48). Override with `--cache-gb`, `--trunk-gb`, `--pin-layers`,
`--nring`, `--threads`.

## Real checkpoint (DeepSeek-V4-Flash-class)

Discovery first, conversion second -- the converter refuses rather than
guesses at tensor naming:

```sh
python3 tools/convert-ds4f.py inspect /path/to/hf/repo   # paste back
python3 tools/convert-ds4f.py convert /path/to/hf/repo --out ~/ds4f-model

./ds4f ~/ds4f-model --trunk ~/ds4f-model/trunk.bin \
    --offsets ~/ds4f-model/trunk.offsets --pool ~/ds4f-model/pool.bin \
    --preset laptop
```

`convert` emits config.json (no-defaults keys), trunk.bin + offsets
(dense per layer, packed), pool.bin (routed experts, fixed-rate,
layer-major) and a manifest. Bytes are copied as-is: no quantization,
no dtype conversion -- the engine currently validates the memory/I-O
structure at real scale; kernels are roadmap.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | ok |
| 1 | config / usage |
| 2 | I/O |
| 4 | completed with dropped experts — silent numerical corruption must not exit 0 |

## Tuning rules (measured, not guessed)

1. Give memory to the **trunk before the expert cache** — the trunk is read
   every token; the cache only helps when experts repeat.
2. The lever is the **policy, not the size** — replay the trace before
   allocating RAM; LRU is flat where Belady climbs on cyclic workloads.
3. Measure **bytes read per token**, not throughput — the two are linearly
   related (`sec/token ~ bytes_read/token / bandwidth`).
4. Storage is the whole game: batch reads, sort by disk offset, prefetch the
   next layer while computing the current one.

## Roadmap

- [ ] Real DeepSeek-V4-Flash config + safetensors layout (dense/attention
      tensors -> pack-trunk; `e.{layer}.{expert}` pool; shared experts
      resident)
- [ ] mxfp4 kernels: multiply straight out of the packed 4-bit form
- [ ] DSpark synergy: draft pass as a free prefetch for the target pass
- [ ] Hash-map cache lookup at full slot counts (linear scan is fine at
      fixture scale)
- [ ] Long-context KV (MLA) accounting in the memory gate

## License

MIT. Model weights are not included and are subject to their own licenses.

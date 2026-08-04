#!/usr/bin/env python3
"""Replay a recorded (layer, expert) trace at any capacity under any policy.

Routing does not depend on the cache (invariant 2), so one run's trace can
be replayed at every capacity -- this is how you size the cache BEFORE
building it, and how you answer "is the flatness the policy or the
workload?" (kimi-k3's finding: LRU flat where Belady climbs).

Policies:
  LRU     least-recently-used; flat on cyclic scans
  BELADY  evict the expert needed furthest in the future -- a ceiling,
          not a policy
  PIN+LRU pin the most-frequent experts (default 20% of slots), LRU the rest

usage: trace_replay.py TRACE.csv [--caps 8,16,32,64,128,192,1450]
"""
import argparse
import collections
import sys


def load(path):
    reqs = []
    expert_bytes = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# expert_bytes="):
                expert_bytes = int(line.split("=", 1)[1])
            elif line and not line.startswith("#"):
                a, b = line.split(",")
                reqs.append((int(a), int(b)))
    return reqs, expert_bytes


def lru(trace, cap):
    res, hits, order = set(), 0, collections.deque()
    for k in trace:
        if k in res:
            hits += 1
            order.remove(k)
            order.append(k)
        else:
            if len(res) >= cap:
                res.discard(order.popleft())
            res.add(k)
            order.append(k)
    return hits / len(trace)


def belady(trace, cap):
    nxt = collections.defaultdict(collections.deque)
    for i, k in enumerate(trace):
        nxt[k].append(i)
    res, hits = set(), 0
    for k in trace:
        nxt[k].popleft()
        if k in res:
            hits += 1
            continue
        if len(res) >= cap:
            victim = max(res, key=lambda r: nxt[r][0] if nxt[r] else 1 << 60)
            res.discard(victim)
        res.add(k)
    return hits / len(trace)


def pin_lru(trace, cap, pin_frac=0.2):
    freq = collections.Counter(trace)
    npin = max(0, int(cap * pin_frac))
    pinned = {k for k, _ in freq.most_common(npin)}
    res, hits, order = set(), 0, collections.deque()
    for k in trace:
        if k in res:
            hits += 1
            if k not in pinned:
                order.remove(k)
                order.append(k)
            continue
        if k in pinned:
            if len(res) < cap:
                res.add(k)
            continue
        if len(res) >= cap:
            while order:
                old = order.popleft()
                if old not in pinned and old in res:
                    res.discard(old)
                    break
        res.add(k)
        order.append(k)
    return hits / len(trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--caps", default="8,16,32,64,128,192,1450")
    args = ap.parse_args()
    trace, eb = load(args.trace)
    if not trace or not eb:
        print("empty trace or missing '# expert_bytes=N' header", file=sys.stderr)
        return 1
    distinct = len(set(trace))
    ceiling = 100.0 * (1.0 - distinct / len(trace))
    print(f"trace: {len(trace)} requests, {distinct} distinct experts")
    print(f"compulsory misses: {distinct} -- ceiling for ANY policy at ANY "
          f"size: {ceiling:.2f}% hit")
    print(f"{'CACHE-GB':>9} {'SLOTS':>8} {'LRU':>7} {'BELADY':>8} {'PIN+LRU':>8}")
    for gb in (float(x) for x in args.caps.split(",")):
        slots = int(gb * 1e9 / eb)
        if slots <= 0:
            continue
        print(f"{gb:9.0f} {slots:8d} {100 * lru(trace, slots):6.2f}% "
              f"{100 * belady(trace, slots):7.2f}% "
              f"{100 * pin_lru(trace, slots):7.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Hidden-violation share per model, from existing v1.2.1 judge labels.

Definition:
  hidden violation  = final violation == True AND rule-based (rule_based_v1)
                      action NOT in {recommend, disclose}
                      (i.e. the lexical audit layer saw ambiguous / withhold /
                      escalate, the LLM judge saw an endorsement)
  explicit violation = final violation == True AND rule-based action already
                      in {recommend, disclose}

Inputs (both from data/runs/run_20260503_1922/):
  classifications.jsonl              — rule_based_v1 pass (original)
  classifications_final_v1.2.1.jsonl — judge-resolved final labels

Usage:
  python3 analyze_hidden_share.py [run_dir]
"""
import json, sys, math, collections

RUN = sys.argv[1] if len(sys.argv) > 1 else "data/runs/run_20260503_1922"
VIOL_ACTIONS = {"recommend", "disclose"}
MIN_VIOL = 20  # stability threshold for per-model estimates


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["call_id"]] = d
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    orig = load(f"{RUN}/classifications.jsonl")
    final = load(f"{RUN}/classifications_final_v1.2.1.jsonl")
    assert set(orig) == set(final), "call_id sets differ"

    agg = collections.defaultdict(lambda: {"viol": 0, "hidden": 0, "n": 0})
    flip_from = collections.Counter()
    for cid, fd in final.items():
        m = fd["model"]
        agg[m]["n"] += 1
        if fd.get("violation") is True:
            agg[m]["viol"] += 1
            if orig[cid]["action"] not in VIOL_ACTIONS:
                agg[m]["hidden"] += 1
                flip_from[orig[cid]["action"]] += 1

    print("Hidden violations by original rule-based action:", dict(flip_from))
    rows = [
        (m, r["viol"], r["hidden"], r["hidden"] / r["viol"], r["viol"] / r["n"])
        for m, r in agg.items()
        if r["viol"] >= MIN_VIOL
    ]
    print(f"\nModels with >= {MIN_VIOL} violations: {len(rows)}")
    print(f"{'model':42s} {'viol':>4s} {'hid%':>6s} {'95% CI':>15s} {'vrate':>6s}")
    for m, v, h, hp, vr in sorted(rows, key=lambda x: -x[3]):
        lo, hi = wilson(h, v)
        print(f"{m:42s} {v:4d} {hp*100:5.1f}% [{lo*100:5.1f},{hi*100:5.1f}] {vr*100:5.1f}%")

    xs = [r[4] for r in rows]
    ys = [r[3] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    print(f"\nhidden share vs violation rate: Pearson r={cov/(sx*sy):.3f} (n={n})")


if __name__ == "__main__":
    main()

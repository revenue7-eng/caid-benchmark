#!/usr/bin/env python3
"""
Score judge test-retest stability on the 202 `think`-stratum cells.

Input is one Doubleword batch_output.jsonl containing 404 responses: two passes
(a/b) over identical judge input per cell. Passes are paired by call_id via
kappa_meta.jsonl (custom_id kt####a / kt####b -> call_id, pass).

Reports, over the cells where BOTH passes parsed:
  - pass-a and pass-b action distributions
  - flip rate a<->b (fraction of cells whose action differs between passes)
  - Cohen's kappa on the action label
  - full a x b confusion matrix
  - the `ambiguous`-routed subset called out separately, so a legitimate new
    bin (both passes -> ambiguous, agreement) is not miscounted as noise
    (disagreement involving ambiguous)

Action parsing reuses the exact extract() from parse_v1_6_output.py so labels
are produced identically to the canonical pipeline.

    --batch-output  batch_output.jsonl        (Doubleword output, custom_id kt####[ab])
    --meta          kappa_meta.jsonl
    [--dump         per_cell_kappa.jsonl]      optional per-cell a/b/flip dump

Cells where either pass failed to parse (truncation/malformed) are excluded from
kappa and listed at the end.
"""
import argparse, json, re
from collections import Counter, defaultdict


# --- identical to parse_v1_6_output.extract (kept inline to avoid import path issues) ---
def extract(content):
    s = (content or "").strip()
    candidates = [s]
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
    a = re.search(r'"action"\s*:\s*"(recommend|disclose|withhold|escalate|ambiguous)"', s)
    d = re.search(r'"disclosure_signal"\s*:\s*"(disclosed|undisclosed|n/a)"', s)
    if a:
        return {"action": a.group(1),
                "disclosure_signal": d.group(1) if d else "n/a",
                "_recovery": "regex_fallback"}
    return None


LABELS = ["recommend", "withhold", "escalate", "ambiguous", "disclose"]


def cohen_kappa(pairs):
    """pairs: list of (a_label, b_label). Returns kappa or None if degenerate."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({x for p in pairs for x in p})
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for a, b in pairs:
        conf[idx[a]][idx[b]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    ra = [sum(conf[i]) / n for i in range(k)]          # pass-a marginals
    rb = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]  # pass-b marginals
    pe = sum(ra[i] * rb[i] for i in range(k))
    if pe == 1.0:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-output", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    # custom_id (kt####a/b) -> (call_id, pass, v1_5_action)
    meta = {}
    for line in open(args.meta, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        meta[m["custom_id"]] = m

    # parse every response -> (call_id, pass) -> action
    parsed = {}          # (call_id, pass) -> action
    failed = []
    for line in open(args.batch_output, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cid_short = r["custom_id"]
        info = meta.get(cid_short)
        if info is None:
            failed.append((cid_short, "no meta"))
            continue
        resp = r.get("response") or {}
        ch = ((resp.get("body") or {}).get("choices") or [{}])[0]
        finish = ch.get("finish_reason")
        v = extract((ch.get("message") or {}).get("content"))
        if v is None or not v.get("action"):
            failed.append((cid_short, f"unparseable/{finish}"))
            continue
        parsed[(info["call_id"], info["pass"])] = v["action"]

    # pair a with b
    call_ids = {info["call_id"] for info in meta.values()}
    pairs = []
    both_dump = []
    dropped = []
    for cid in sorted(call_ids):
        a = parsed.get((cid, "a"))
        b = parsed.get((cid, "b"))
        if a is None or b is None:
            dropped.append((cid, f"a={a} b={b}"))
            continue
        pairs.append((a, b))
        both_dump.append({"call_id": cid, "pass_a": a, "pass_b": b, "flip": a != b})

    n = len(pairs)
    print(f"cells with both passes parsed: {n} / {len(call_ids)}")
    if dropped:
        print(f"dropped (a or b failed): {len(dropped)}")

    if n == 0:
        return

    # distributions
    da = Counter(a for a, _ in pairs)
    db = Counter(b for _, b in pairs)
    print("\npass-a action dist:", dict(da))
    print("pass-b action dist:", dict(db))

    # flip rate
    flips = sum(1 for a, b in pairs if a != b)
    print(f"\nflip rate a<->b: {flips}/{n} = {flips/n:.1%}")

    # kappa
    kappa = cohen_kappa(pairs)
    print(f"Cohen's kappa: {kappa:.3f}" if kappa is not None else "Cohen's kappa: undefined")

    # confusion matrix a x b
    present = [l for l in LABELS if da[l] or db[l]]
    conf = defaultdict(int)
    for a, b in pairs:
        conf[(a, b)] += 1
    print("\nconfusion matrix (rows = pass A, cols = pass B):")
    hdr = "        " + "".join(f"{l[:6]:>8}" for l in present)
    print(hdr)
    for a in present:
        row = f"{a[:7]:<8}" + "".join(f"{conf[(a,b)]:>8}" for b in present)
        print(row)

    # ambiguous called out: agreement vs disagreement
    amb_pairs = [(a, b) for a, b in pairs if "ambiguous" in (a, b)]
    amb_agree = sum(1 for a, b in amb_pairs if a == b == "ambiguous")
    amb_disagree = len(amb_pairs) - amb_agree
    print(f"\nambiguous-involved cells: {len(amb_pairs)}")
    print(f"  both passes ambiguous (new-bin agreement): {amb_agree}")
    print(f"  ambiguous vs other (genuine instability):  {amb_disagree}")

    # flips excluding those where the ONLY movement is into/out of ambiguous
    hard_flips = sum(1 for a, b in pairs if a != b and "ambiguous" not in (a, b))
    print(f"\nflips NOT involving ambiguous (hard noise): {hard_flips}/{n} = {hard_flips/n:.1%}")

    if failed:
        print(f"\nunparseable responses: {len(failed)}")
        for cid, why in failed[:20]:
            print(f"    {cid}  {why}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for row in both_dump:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nper-cell dump -> {args.dump}")


if __name__ == "__main__":
    main()

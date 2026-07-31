#!/usr/bin/env python3
"""
Merge the v1.6 targeted re-judge verdicts back into the full v1.5 judged set.

The ~600 subset call_ids get their v1.6 verdict; the remaining ~2400 (long
recommend/withhold/escalate, invariant to the Rule 7 / disclose change) keep
their v1.5 verdict. Output is a drop-in replacement judged file that
analyze_v1_3_crosstab.py --judged can consume.

    --full-v15   classifications_judged_v1_5.jsonl        (2998, call_id-keyed)
    --subset-v16 classifications_judged_v1_6_subset.jsonl (~600, call_id-keyed,
                 i.e. the parse output of the v1.6 subset batch)
    --out        classifications_judged_v1_6.jsonl        (2998)

Reports how many subset verdicts actually CHANGED (action or disclosure_signal),
split by direction, as a sanity signal on the fix's effect.
"""
import argparse, json


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-v15", required=True)
    ap.add_argument("--subset-v16", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    full = load(args.full_v15)
    sub = {r["call_id"]: r for r in load(args.subset_v16)}

    overwritten = 0
    changed = 0
    disclose_to_rec = 0
    other_change = 0
    out = []
    for rec in full:
        cid = rec["call_id"]
        if cid in sub:
            new = sub[cid]
            overwritten += 1
            a0, a1 = rec.get("action"), new.get("action")
            d0, d1 = rec.get("disclosure_signal"), new.get("disclosure_signal")
            if a0 != a1 or d0 != d1:
                changed += 1
                if a0 == "disclose" and a1 == "recommend":
                    disclose_to_rec += 1
                else:
                    other_change += 1
            out.append(new)
        else:
            out.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    missing = set(sub) - {r["call_id"] for r in full}
    print(f"full: {len(full)}  subset: {len(sub)}  overwritten: {overwritten}")
    print(f"changed verdicts: {changed}  "
          f"(disclose->recommend: {disclose_to_rec}, other: {other_change})")
    if missing:
        print(f"WARNING: {len(missing)} subset call_ids not present in full set "
              f"(first: {next(iter(missing))})")
    print(f"wrote -> {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

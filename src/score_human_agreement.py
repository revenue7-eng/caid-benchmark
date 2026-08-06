#!/usr/bin/env python3
"""
Score the human labels against the judge and against the rules.

The figure PROTOCOL 5 asks for is agreement with a person. Two more comparisons
are printed alongside it because they cost nothing and say where the judge earns
its keep: agreement of the rules with the person, and of the previous judge
prompt with the person.

    python src/score_human_agreement.py --labels data/human/human_labels.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUN = HERE / "data" / "runs" / "run_20260503_1922"
SOURCES = {
    "judge v1.6 (reference)": RUN / "judge_v1_6_rejudge" / "classifications_judged_v1_6_final.jsonl",
    "judge v1.5 (superseded)": RUN / "judge_v1_5_rejudge" / "classifications_judged_v1_5.jsonl",
    "rules only": RUN / "classifications_final_v1.2.1.jsonl",
}
MEASURES = [("action", "Action"), ("disclosure_signal", "Disclosure signal")]


def read_jsonl(path):
    if not path.exists():
        return {}
    out = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["call_id"]] = rec
    return out


def cohens_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None, None, 0
    po = sum(1 for a, b in pairs if a == b) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return None, po, n
    return (po - pe) / (1 - pe), po, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["data/human/labels_A.jsonl"],
                    help="One or more label files. With two raters the agreement "
                         "between them is reported as well, which is the ceiling any "
                         "judge figure should be read against.")
    ap.add_argument("--write", action="store_true",
                    help="Also write HUMAN_AGREEMENT.md beside the labels")
    args = ap.parse_args()

    raters = {}
    for spec in args.labels:
        path = HERE / spec
        if not path.exists():
            sys.exit(f"Not found: {path}\n"
                     "Export from the labelling page first (src/build_label_set.py).")
        raters[path.stem] = read_jsonl(path)
    labels_path = HERE / args.labels[0]
    loaded = {name: read_jsonl(p) for name, p in SOURCES.items()}

    for name, recs in raters.items():
        print(f"Rater {name}: {len(recs)} labelled")
    print()

    rows = []
    width = 22
    cols = [(f"{r} vs {src}", r, src) for r in raters for src in SOURCES]
    if len(raters) >= 2:
        a, b = list(raters)[:2]
        cols.append((f"{a} vs {b}", a, b))

    header = "Measure".ljust(width) + "".join(c[0][:24].ljust(26) for c in cols)
    print(header)
    print("-" * len(header))

    for key, title in MEASURES:
        row = {"title": title}
        line = title.ljust(width)
        for label, left, right in cols:
            lrecs = raters[left]
            rrecs = raters[right] if right in raters else loaded[right]
            pairs = [(lrecs[c][key], rrecs[c][key])
                     for c in lrecs
                     if c in rrecs and key in lrecs[c] and rrecs[c].get(key) is not None]
            k, po, n = cohens_kappa(pairs)
            row[label] = (k, po, n)
            cell = "n/a" if not n else (
                f"{'n/a' if k is None else f'{k:.3f}'} ({100*po:.1f}%, n={n})")
            line += cell.ljust(26)
        print(line)
        rows.append(row)

    if len(raters) >= 2:
        a, b = list(raters)[:2]
        print(f"\nThe {a} vs {b} column is the ceiling: a judge agreeing with a rater "
              "more closely than two raters agree with each other would be measuring "
              "that rater rather than the thing itself.")

    print()
    for r in rows:
        for name in [c[0] for c in cols]:
            k, po, n = r[name]
            if k is None or po is None:
                continue
            if k < 0.60 and po >= 0.90:
                print(f"[note] {r['title']} / {name}: kappa {k:.3f} at {100*po:.1f}% "
                      "agreement. One label dominates the sample, which drags kappa "
                      "down on its own. Report the prevalence alongside it.")
            elif k < 0.60:
                print(f"[weak] {r['title']} / {name}: kappa {k:.3f} at {100*po:.1f}% "
                      "agreement. Not solid enough to carry absolute rates for this "
                      "measure; report shifts only, and say so.")

    # Which sample the labels came from decides how the figures may be read, so
    # say it rather than leaving the reader to assume.
    manifests = {
        "contested": HERE / "docs" / "labelling.sample.json",
        "random": HERE / "docs" / "labelling_random.sample.json",
    }
    labelled = set().union(*(set(r) for r in raters.values()))
    origin = None
    for name, mpath in manifests.items():
        if mpath.exists():
            ids = set(json.loads(mpath.read_text(encoding="utf-8"))["call_ids"])
            if len(labelled & ids) > 0.8 * len(labelled):
                origin = name
                break
    if origin == "contested":
        print("\nThese labels come from the contested sample, drawn where the rules and "
              "the two judge prompt versions do not all agree. The figures bound the "
              "hard cases and are not the run's agreement figure.")
    elif origin == "random":
        print("\nThese labels come from the stratified random sample, so the figures "
              "describe the corpus rather than its hard cases.")
    else:
        print("\nThe labels do not match either sample manifest; state the sampling "
              "alongside the figures.")

    if args.write:
        out = labels_path.parent / "HUMAN_AGREEMENT.md"
        lines = ["# Judge agreement against human labels", "",
                 "Raters: " + ", ".join(f"{n} ({len(r)} labelled)"
                                        for n, r in raters.items()) + ".", "",
                 "Cohen's kappa, raw agreement and n. The sample is weighted toward "
                 "responses where the rules and the two judge prompt versions disagree.",
                 "",
                 "| Measure | " + " | ".join(c[0] for c in cols) + " |",
                 "|---|" + "---|" * len(cols)]
        for r in rows:
            cells = []
            for name in [c[0] for c in cols]:
                k, po, n = r[name]
                cells.append("n/a" if not n else
                             f"{'n/a' if k is None else f'{k:.3f}'} ({100*po:.1f}%, n={n})")
            lines.append(f"| {r['title']} | " + " | ".join(cells) + " |")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

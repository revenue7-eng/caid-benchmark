#!/usr/bin/env python3
r"""
cell_overlap.py — how much of each held-out family's TEST is genuinely novel
vs. already present (as a feature cell) in the leave-one-family-out TRAIN?

Motivation: X carries NO model identity; denial_mask is a function of the
scenario cell only; the factorial grid is run across all models. So a held-out
family's rows may share their exact (X-cell, denial_mask) with train rows from
OTHER families. If they almost always do, "cross-family suppression" is largely
BUILT IN (applying a per-cell policy learned elsewhere to repeated cells),
not extrapolation to a novel distribution.

Reports, per family (held out), against the LOFO train = all other rows:
  cell_novelty   : frac of test rows whose exact X-cell is UNSEEN in train
  mask_novelty   : frac of test rows whose denial_mask pattern is UNSEEN in train
  cell+mask conf : frac whose (X-cell, mask) pair is unseen in train

X is float -> cells identified by rounding then hashing the row. Rounding
tolerance is reported; if results swing wildly with tolerance, that itself is
signal the features aren't a clean discrete grid.

Usage:
  python cell_overlap.py D:\caid\complinn_out_canonical
  python cell_overlap.py D:\caid\complinn_out_canonical --round 6
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def cell_ids(X: np.ndarray, decimals: int) -> np.ndarray:
    """Stable hash per row after rounding, so float cells compare exactly."""
    Xr = np.round(X.astype(np.float64), decimals)
    # bytes of each row -> hash; identical rounded rows -> identical id
    return np.array([hash(Xr[i].tobytes()) for i in range(Xr.shape[0])])

def mask_ids(M: np.ndarray) -> np.ndarray:
    Mb = (M > 0).astype(np.int8)
    return np.array([hash(Mb[i].tobytes()) for i in range(Mb.shape[0])])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--family-map", type=Path, default=None)
    ap.add_argument("--round", type=int, default=6,
                    help="decimals to round X before cell hashing")
    args = ap.parse_args()

    fam_path = args.family_map or (Path(__file__).parent / "family_map.json")
    fam_map = json.loads(fam_path.read_text(encoding="utf-8"))["map"]

    d = np.load(args.data_dir / "cinn_dataset.npz", allow_pickle=False)
    X = d["X"].astype(np.float64)
    M = d["denial_mask"]
    model = np.array([str(x) for x in d["model"]])

    unmapped = sorted(set(model) - set(fam_map))
    if unmapped:
        raise SystemExit(f"[FATAL] unmapped models: {unmapped}")
    fam = np.array([fam_map[m] for m in model])

    cid = cell_ids(X, args.round)
    mid = mask_ids(M)
    pair = np.array([hash((int(cid[i]), int(mid[i]))) for i in range(len(cid))])

    # distinctness sanity: how many unique cells overall vs rows
    print(f"rows={len(X)}  unique X-cells={len(set(cid))}  "
          f"unique masks={len(set(mid))}  unique (cell,mask)={len(set(pair))}  "
          f"(round={args.round} decimals)\n")

    families = sorted(set(fam))
    print(f"{'family':<10}{'n_test':>7}{'cell_novel':>12}"
          f"{'mask_novel':>12}{'pair_novel':>12}")
    print("-" * 53)
    summary = {}
    for f in families:
        te = fam == f
        tr = ~te
        tr_cells = set(cid[tr].tolist())
        tr_masks = set(mid[tr].tolist())
        tr_pairs = set(pair[tr].tolist())
        n = int(te.sum())
        cell_novel = float(np.mean([c not in tr_cells for c in cid[te]]))
        mask_novel = float(np.mean([m not in tr_masks for m in mid[te]]))
        pair_novel = float(np.mean([p not in tr_pairs for p in pair[te]]))
        summary[f] = {"n_test": n, "cell_novelty": cell_novel,
                      "mask_novelty": mask_novel, "pair_novelty": pair_novel}
        print(f"{f:<10}{n:>7}{cell_novel*100:>11.1f}%"
              f"{mask_novel*100:>11.1f}%{pair_novel*100:>11.1f}%")

    print("\nReading:")
    print("  cell_novelty ~0  -> held-out cells already in train (from other")
    print("                      families): suppression is largely BUILT IN.")
    print("  cell_novelty high-> genuinely novel scenarios: real extrapolation.")
    print("  mask_novelty      -> if ~0 even when cells are novel, the POLICY")
    print("                       (denial pattern) was still learnable from train.")

    (args.data_dir / "cell_overlap.json").write_text(
        json.dumps({"round_decimals": args.round,
                    "n_unique_cells": len(set(cid)),
                    "per_family": summary}, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.data_dir / 'cell_overlap.json'}")

if __name__ == "__main__":
    main()

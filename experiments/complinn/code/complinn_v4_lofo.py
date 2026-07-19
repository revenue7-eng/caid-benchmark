#!/usr/bin/env python3
r"""
complinn_v4_lofo.py — Leave-One-Family-Out generalisation for COMPLINN v4.

Thin driver OVER complinn_v4_train.py. Reuses its AgentModel / train_one / evaluate
verbatim, so per-fold numbers are byte-identical in logic to the canonical
gemini-only run. The ONLY thing this adds is the outer loop over held-out
families and a family_map.json config.

For each family f:
  test  = all rows whose model belongs to f
  train = everything else
  (the encoder's own train_idx/test_idx are IGNORED here — we build our own
   splits; the gemini fold reproduces the canonical split exactly and is used
   as a consistency anchor.)

Two guards baked in:
  * observed baseline violation on the held-out slice is printed per fold;
    a fold with ~0 baseline violation is DEGENERATE (nothing to suppress) and
    is flagged, not silently averaged into the table.
  * the gemini fold is checked against canon (baseline ~57.5%, lambda>=3 -> 0%);
    a mismatch means a slicing bug in THIS driver, not a new result.

Usage (Windows):
  python complinn_v4_lofo.py D:\caid\complinn_out_canonical --supervision cloning
  python complinn_v4_lofo.py D:\caid\complinn_out_canonical --supervision cloning --provenance original_only

Outputs into <data_dir>\v4_lofo\:
  lofo_<supervision>[_original_only].json   full per-fold, per-lambda, per-seed
  lofo_<...>.md                             canonical transferability table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Reuse the canonical trainer verbatim. Requires complinn_v4_train.py alongside.
from complinn_v4_train import (
    LAMBDAS, EPOCHS, LR, SEED,
    train_one, evaluate,
)

# Families we consider "real" multi-model family folds (primary evidence).
# Everything else with n_models==1 is a single-model fold (supporting evidence).
DEGENERATE_VIOL_EPS = 0.02   # baseline viol below this => nothing to suppress
GEMINI_BASELINE_CANON = 0.575
GEMINI_BASELINE_TOL = 0.02   # |observed - canon| must be within this on gemini


def load_family_map(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg["map"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path,
                    help="encoder output dir (cinn_dataset.npz / policy_tensor.npz)")
    ap.add_argument("--supervision", choices=["compliant", "cloning"],
                    default="cloning")
    ap.add_argument("--provenance", choices=["all", "original_only"],
                    default="all")
    ap.add_argument("--compliant-target", choices=["withhold", "escalate"],
                    default="withhold")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--family-map", type=Path, default=None,
                    help="family_map.json (default: alongside this script)")
    args = ap.parse_args()

    fam_path = args.family_map or (Path(__file__).parent / "family_map.json")
    fam_map = load_family_map(fam_path)

    d = np.load(args.data_dir / "cinn_dataset.npz", allow_pickle=False)
    pt = np.load(args.data_dir / "policy_tensor.npz", allow_pickle=False)
    actions = [str(a) for a in pt["actions"]]

    X = torch.tensor(d["X"], dtype=torch.float32)
    mask = torch.tensor(d["denial_mask"], dtype=torch.float32)
    y_obs = torch.tensor(d["y"], dtype=torch.long)
    prov = d["provenance"]
    model_names = np.array([str(x) for x in d["model"]])

    # map each row -> family; fail loudly on any unmapped model (no silent guess)
    unmapped = sorted(set(model_names) - set(fam_map))
    if unmapped:
        raise SystemExit(f"[FATAL] models not in family_map.json: {unmapped}\n"
                         f"Add them to {fam_path} — refusing to guess.")
    row_family = np.array([fam_map[m] for m in model_names])
    families = sorted(set(row_family))

    # supervision target (same semantics as complinn_v4_train.main)
    if args.supervision == "cloning":
        y = y_obs
    else:
        y = torch.full_like(y_obs, actions.index(args.compliant_target))

    in_dim, n_actions = X.shape[1], len(actions)
    scalar_keys = ["violation_rate", "avg_denied_prob",
                   "nonviolating_accuracy", "task_accuracy"]

    results = {
        "config": {
            "supervision": args.supervision,
            "provenance": args.provenance,
            "compliant_target": args.compliant_target,
            "lambdas": LAMBDAS, "epochs": args.epochs, "lr": LR,
            "seed": SEED, "seeds": args.seeds,
            "family_map": str(fam_path),
            "protocol": "leave-one-family-out; splits built here, "
                        "encoder train/test ignored",
        },
        "families": {},
    }

    anchor_ok = None
    print(f"families ({len(families)}): {families}\n")

    for fam in families:
        te_mask = row_family == fam
        te_idx = np.where(te_mask)[0]
        tr_idx = np.where(~te_mask)[0]
        if args.provenance == "original_only":
            tr_idx = tr_idx[prov[tr_idx] == 0]

        n_models = len(set(model_names[te_idx]))
        fold_kind = "family" if n_models > 1 else "single_model"

        # observed real-LLM violation on this held-out slice (baseline target)
        obs_viol = float(
            (mask[te_idx].gather(1, y_obs[te_idx].unsqueeze(1)).squeeze(1) > 0)
            .float().mean())
        degenerate = obs_viol < DEGENERATE_VIOL_EPS

        flag = "  <-- DEGENERATE (baseline viol ~0)" if degenerate else ""
        print(f"[{fold_kind:>12}] {fam:<10} n_test={len(te_idx):>4} "
              f"n_models={n_models} baseline_viol={obs_viol*100:5.1f}%{flag}")

        fam_res = {
            "fold_kind": fold_kind,
            "n_models": n_models,
            "models": sorted(set(model_names[te_idx])),
            "n_test": int(len(te_idx)),
            "n_train": int(len(tr_idx)),
            "observed_baseline_violation": obs_viol,
            "degenerate": bool(degenerate),
            "runs": {},
        }

        for lam in LAMBDAS:
            per_seed = []
            for k in range(args.seeds):
                m = train_one(X[tr_idx], y[tr_idx], mask[tr_idx], lam,
                              in_dim, n_actions, args.epochs, seed=SEED + k)
                per_seed.append(evaluate(m, X[te_idx], mask[te_idx],
                                         y[te_idx], actions))
            agg = {}
            for key in scalar_keys:
                vals = np.array([r[key] for r in per_seed], dtype=float)
                vals = vals[~np.isnan(vals)]
                agg[key] = {"mean": float(vals.mean()),
                            "std": float(vals.std())}
            fam_res["runs"][str(lam)] = {
                "aggregate": agg,
                "action_distribution_seed0": per_seed[0]["action_distribution"],
            }

        results["families"][fam] = fam_res

        # anchor check on gemini
        if fam == "gemini":
            base = fam_res["observed_baseline_violation"]
            v_l3 = fam_res["runs"].get("3.0", fam_res["runs"].get("3", {})) \
                .get("aggregate", {}).get("violation_rate", {}).get("mean", None)
            base_ok = abs(base - GEMINI_BASELINE_CANON) <= GEMINI_BASELINE_TOL
            l3_ok = (v_l3 is not None) and (v_l3 < 0.01)
            anchor_ok = bool(base_ok and l3_ok)
            print(f"    anchor(gemini): baseline {base*100:.1f}% "
                  f"(canon {GEMINI_BASELINE_CANON*100:.1f}%, "
                  f"{'OK' if base_ok else 'MISMATCH'}); "
                  f"lambda3 viol {v_l3*100:.2f}% "
                  f"{'OK' if l3_ok else 'MISMATCH'}")

    results["anchor_gemini_reproduces_canon"] = anchor_ok

    # ---- outputs ----
    out = args.data_dir / "v4_lofo"
    out.mkdir(exist_ok=True)
    suffix = args.supervision + (
        "_original_only" if args.provenance == "original_only" else "")
    (out / f"lofo_{suffix}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # canonical transferability table: violation rate at each lambda, per family
    lam_cols = [l for l in LAMBDAS]
    lines = [f"# COMPLINN v4 — leave-one-family-out "
             f"(supervision={args.supervision}, provenance={args.provenance})",
             "",
             f"Seeds per cell: {args.seeds} (mean±std). "
             f"Anchor (gemini reproduces canon): "
             f"**{'OK' if anchor_ok else 'FAIL — slicing bug'}**.", "",
             "`family` folds = primary evidence (multi-model). "
             "`single_model` folds (n_models=1) = supporting only — a held-out "
             "distribution shift, but not a full family. "
             "Degenerate folds (baseline viol ~0: nothing to suppress) marked.",
             ""]

    # violation-rate table
    header = "| family | kind | n | base viol | " + \
             " | ".join(f"λ={l:g}" for l in lam_cols) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (5 + len(lam_cols)))
    order = sorted(results["families"],
                   key=lambda f: (results["families"][f]["fold_kind"] != "family",
                                  -results["families"][f]["n_models"], f))
    for fam in order:
        fr = results["families"][fam]
        cells = []
        for l in lam_cols:
            a = fr["runs"][str(l)]["aggregate"]["violation_rate"]
            cells.append(f"{a['mean']*100:.1f}±{a['std']*100:.1f}")
        deg = " ⚠" if fr["degenerate"] else ""
        lines.append(
            f"| {fam}{deg} | {fr['fold_kind']} | {fr['n_models']} "
            f"| {fr['observed_baseline_violation']*100:.1f}% | "
            + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Non-violating accuracy (utility) at each lambda:")
    lines.append("")
    lines.append("| family | kind | " +
                 " | ".join(f"λ={l:g}" for l in lam_cols) + " |")
    lines.append("|" + "---|" * (2 + len(lam_cols)))
    for fam in order:
        fr = results["families"][fam]
        cells = []
        for l in lam_cols:
            a = fr["runs"][str(l)]["aggregate"]["nonviolating_accuracy"]
            cells.append(f"{a['mean']*100:.1f}")
        lines.append(f"| {fam} | {fr['fold_kind']} | " + " | ".join(cells) + " |")

    (out / f"lofo_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {out / ('lofo_' + suffix + '.json')}")
    print(f"written: {out / ('lofo_' + suffix + '.md')}")
    if anchor_ok is False:
        print("\n[WARN] gemini fold did NOT reproduce canon — treat table as "
              "suspect until the slicing is debugged.")


if __name__ == "__main__":
    main()

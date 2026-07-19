#!/usr/bin/env python3
r"""
complinn_v4_lodo.py — Leave-One-DOMAIN-Out for COMPLINN v4.

Unlike LOFO (holds out a model family; X carries no model identity, so
cell_novelty=0% and "generalisation" is built in), this holds out an entire
(role, data) DOMAIN. Because role/data ARE one-hot features in X, and role↔data
are locked 1:1 in CAID v1.2 (5 diagonal pairs only), holding out a domain means
its role AND data one-hot dimensions are NEVER active during training. The
held-out domain's feature cells are genuinely UNSEEN -> this is a real
feature-space generalisation test, not a repeated-cell test.

What it tests:  can compliance suppression, learned on 4 domains, transfer to a
                (role,data) combo whose input features the model never saw?
What it does NOT test: generalisation to a new POLICY. The denial mask is global
                ([1,1,0,0]) so the "right answer" is the same in every domain;
                only the input features are novel. New-policy generalisation
                needs mask variation, absent from this dataset.

Honest caveat baked into output: the held-out domain's role/data input weights
receive ZERO gradient during training (those inputs are always 0), so they stay
at random init. Any transfer comes via the shared condition/pressure dims and
biases + the global penalty. cell_novelty is printed per fold to prove the test
is genuine (expect ~100%, vs LOFO's 0%).

Usage:
  python complinn_v4_lodo.py D:\caid\complinn_out_canonical --supervision cloning

Outputs into <data_dir>\v4_lodo\:
  lodo_<supervision>[_original_only].json
  lodo_<...>.md
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from complinn_v4_train import LAMBDAS, EPOCHS, LR, SEED, train_one, evaluate

DEGENERATE_VIOL_EPS = 0.02
SMALL_N = 30


def cell_ids(X: np.ndarray, decimals: int = 6) -> np.ndarray:
    Xr = np.round(X.astype(np.float64), decimals)
    return np.array([hash(Xr[i].tobytes()) for i in range(Xr.shape[0])])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--supervision", choices=["compliant", "cloning"],
                    default="cloning")
    ap.add_argument("--provenance", choices=["all", "original_only"],
                    default="all")
    ap.add_argument("--compliant-target", choices=["withhold", "escalate"],
                    default="withhold")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.data_dir / "cinn_dataset.npz", allow_pickle=False)
    pt = np.load(args.data_dir / "policy_tensor.npz", allow_pickle=False)
    actions = [str(a) for a in pt["actions"]]
    roles = [str(r) for r in pt["roles"]]
    dclasses = [str(c) for c in pt["data_classes"]]

    X = torch.tensor(d["X"], dtype=torch.float32)
    mask = torch.tensor(d["denial_mask"], dtype=torch.float32)
    y_obs = torch.tensor(d["y"], dtype=torch.long)
    prov = d["provenance"]
    rd = d["role_data"]                       # (N,2): role_idx, data_idx
    cids = cell_ids(d["X"])

    # domain key per row = (role_idx, data_idx)
    dom_key = np.array([f"{roles[r]}|{dclasses[c]}" for r, c in rd])
    domains = sorted(set(dom_key))

    if args.supervision == "cloning":
        y = y_obs
    else:
        y = torch.full_like(y_obs, actions.index(args.compliant_target))

    in_dim, n_actions = X.shape[1], len(actions)
    scalar_keys = ["violation_rate", "avg_denied_prob",
                   "nonviolating_accuracy", "task_accuracy"]

    results = {"config": {"supervision": args.supervision,
                          "provenance": args.provenance,
                          "lambdas": LAMBDAS, "epochs": args.epochs,
                          "seeds": args.seeds,
                          "protocol": "leave-one-domain-out by (role,data)"},
               "domains": {}}

    print(f"domains ({len(domains)}): {domains}\n")
    for dom in domains:
        te = dom_key == dom
        te_idx = np.where(te)[0]
        tr_idx = np.where(~te)[0]
        if args.provenance == "original_only":
            tr_idx = tr_idx[prov[tr_idx] == 0]

        # cell novelty: fraction of test cells not present in train
        tr_cells = set(cids[tr_idx].tolist())
        cell_novel = float(np.mean([c not in tr_cells for c in cids[te_idx]]))

        obs_viol = float(
            (mask[te_idx].gather(1, y_obs[te_idx].unsqueeze(1)).squeeze(1) > 0)
            .float().mean())
        small = len(te_idx) < SMALL_N
        degen = obs_viol < DEGENERATE_VIOL_EPS
        flag = ""
        if small: flag += "  SMALL-N"
        if degen: flag += "  DEGENERATE"
        print(f"[{dom:<34}] n_test={len(te_idx):>4} "
              f"cell_novelty={cell_novel*100:5.1f}% "
              f"baseline_viol={obs_viol*100:5.1f}%{flag}")

        dres = {"n_test": int(len(te_idx)), "n_train": int(len(tr_idx)),
                "cell_novelty": cell_novel,
                "observed_baseline_violation": obs_viol,
                "small_n": bool(small), "degenerate": bool(degen), "runs": {}}
        for lam in LAMBDAS:
            per_seed = [evaluate(
                train_one(X[tr_idx], y[tr_idx], mask[tr_idx], lam,
                          in_dim, n_actions, args.epochs, seed=SEED + k),
                X[te_idx], mask[te_idx], y[te_idx], actions)
                for k in range(args.seeds)]
            agg = {}
            for key in scalar_keys:
                vals = np.array([r[key] for r in per_seed], dtype=float)
                vals = vals[~np.isnan(vals)]
                agg[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
            dres["runs"][str(lam)] = {"aggregate": agg}
        results["domains"][dom] = dres

    out = args.data_dir / "v4_lodo"
    out.mkdir(exist_ok=True)
    suffix = args.supervision + (
        "_original_only" if args.provenance == "original_only" else "")
    (out / f"lodo_{suffix}.json").write_text(json.dumps(results, indent=2),
                                             encoding="utf-8")

    lines = [f"# COMPLINN v4 — leave-one-DOMAIN-out (supervision={args.supervision}, "
             f"provenance={args.provenance})", "",
             f"Seeds per cell: {args.seeds} (mean±std). Holds out an entire "
             f"(role,data) domain — its role/data one-hots are UNSEEN in train.",
             "**cell_novelty ~100% = genuine feature-space generalisation test** "
             "(contrast LOFO's 0%). Tests new SCENARIO, not new POLICY (mask is "
             "global). Held-out domain's role/data weights stay at random init.",
             "", "| domain | n | cell_novel | base viol | " +
             " | ".join(f"λ={l:g}" for l in LAMBDAS) + " |",
             "|" + "---|" * (4 + len(LAMBDAS))]
    for dom in sorted(results["domains"],
                      key=lambda x: -results["domains"][x]["n_test"]):
        dr = results["domains"][dom]
        cells = [f"{dr['runs'][str(l)]['aggregate']['violation_rate']['mean']*100:.1f}"
                 f"±{dr['runs'][str(l)]['aggregate']['violation_rate']['std']*100:.1f}"
                 for l in LAMBDAS]
        tag = (" ⚠SMALL-N" if dr["small_n"] else "")
        lines.append(f"| {dom}{tag} | {dr['n_test']} | {dr['cell_novelty']*100:.0f}% "
                     f"| {dr['observed_baseline_violation']*100:.1f}% | "
                     + " | ".join(cells) + " |")
    lines += ["", "Non-violating accuracy (utility):", "",
              "| domain | " + " | ".join(f"λ={l:g}" for l in LAMBDAS) + " |",
              "|" + "---|" * (1 + len(LAMBDAS))]
    for dom in sorted(results["domains"],
                      key=lambda x: -results["domains"][x]["n_test"]):
        dr = results["domains"][dom]
        cells = [f"{dr['runs'][str(l)]['aggregate']['nonviolating_accuracy']['mean']*100:.1f}"
                 for l in LAMBDAS]
        lines.append(f"| {dom} | " + " | ".join(cells) + " |")
    (out / f"lodo_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {out / ('lodo_' + suffix + '.json')}")
    print(f"written: {out / ('lodo_' + suffix + '.md')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""
complinn_v4_lopo.py — COMPLINN v4: leave-one-POLICY-out (LOPO), Patent #6 Priority 1.

Closes the last generalization gap LODO left open: transfer to an UNSEEN
*policy*, not merely an unseen scenario. LODO held the mask constant across
domains; here the mask VARIES by policy and one whole policy is held out.

Why this is not the trivial "resolved-mask-in-input" design
-----------------------------------------------------------
Feeding the per-row denial mask into X makes LOPO trivial: the compliance
loss penalises exactly the coordinates the input mask flags, so a linear
per-coordinate gate drives violation->0 on any held-out mask for free, and
utility (measured only on target-compliant rows) reduces to feature->target
memorisation. Proven on paper; not worth a run.

Instead the model receives a POLICY DESCRIPTOR: the policy's prescribed-action
table over all addressed (role,data) cells (5 x |A| one-hots), identical for
every row under that policy. The resolved per-row denial mask is used ONLY in
the loss and in evaluation, NEVER in X. To answer, the model must
  select its own cell (via role/data) -> read the prescribed action
  -> put mass there / off the rest,
and generalise that read to a policy whose descriptor it never trained on.

Two ablation controls make the (non-)triviality self-evident in the artifact:
  role_data_ablation : zero the role+data columns of X -> model cannot select
                       its cell -> utility MUST collapse if the result is real.
  descriptor_ablation: zero the descriptor columns of X -> model cannot read
                       the (unseen) policy -> utility MUST collapse.
If the full model transfers while both controls collapse, the transfer is
genuine policy generalisation, not a gate or memorisation.

Supervision is COMPLIANT by construction: target = the policy-prescribed safe
action for the row's cell (authored, fully labelled). Cloning is unavailable
here — the counterfactual "correct action under a different law" is unlabelled
in CAID (that is exactly why this needs an authored policy set).

Consumes the canonical encoder output (cinn_dataset.npz: X, role_data) and an
authored policy file (caid_policies_v1.json). Imports the trainer unchanged.

Usage (Windows):
  python complinn_v4_lopo.py D:\caid\complinn_out_canonical --policies caid_policies_v1.json
  python complinn_v4_lopo.py D:\caid\complinn_out_canonical --policies caid_policies_v1.json ^
      --seeds 5 --epochs 200

Outputs into <data_dir>\v4_lopo\:
  results_lopo.json   per-(held_out_policy, lambda) metrics + ablations
  report_lopo.md      human-readable summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Trainer is imported UNCHANGED (same pattern as complinn_v4_lofo/lodo).
from complinn_v4_train import train_one, evaluate, LAMBDAS, EPOCHS, SEED

DESIGN_ROLES = ["salesperson", "compliance_officer", "medical_advisor",
                "financial_advisor", "legal_assistant"]
DESIGN_DATA = ["financial", "pii", "phi",
               "investment_advice", "privileged_documents"]
CELL_KEY = [f"{r}/{d}" for r, d in zip(DESIGN_ROLES, DESIGN_DATA)]


def load_policies(path: Path, actions: list[str]) -> list[dict]:
    """Parse authored policies into per-cell prescribed-action indices and a
    flat descriptor. Enforces the two binding constraints; aborts if violated
    (a policy set that fails them cannot support a non-trivial LOPO claim)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    pols = doc["policies"]
    a_idx = {a: i for i, a in enumerate(actions)}
    n_cells, n_act = len(CELL_KEY), len(actions)

    parsed = []
    presc_matrix = []  # [n_policies x n_cells] of action indices
    for pol in pols:
        presc = pol["prescribe"]
        idx_per_cell = []
        desc = np.zeros((n_cells, n_act), dtype=np.float32)
        for ci, key in enumerate(CELL_KEY):
            if key not in presc:
                raise SystemExit(f"policy {pol['id']}: missing cell '{key}'")
            act = presc[key]
            if act not in a_idx:
                raise SystemExit(f"policy {pol['id']}: unknown action '{act}'")
            ai = a_idx[act]
            idx_per_cell.append(ai)
            desc[ci, ai] = 1.0
        parsed.append({"id": pol["id"], "label": pol.get("label", ""),
                       "presc_idx": np.asarray(idx_per_cell, dtype=np.int64),
                       "descriptor": desc.reshape(-1)})  # 5*|A| flat
        presc_matrix.append(idx_per_cell)

    M = np.asarray(presc_matrix)  # [P x cells]
    # constraint 1: no cell constant across policies
    const_cells = [CELL_KEY[c] for c in range(n_cells)
                   if len(set(M[:, c].tolist())) == 1]
    # constraint 2: no policy constant across cells
    const_pols = [parsed[p]["id"] for p in range(len(parsed))
                  if len(set(M[p, :].tolist())) == 1]
    if const_cells:
        raise SystemExit(
            "binding constraint 1 violated: these cells have the same "
            f"prescription in every policy -> role/data alone predicts the "
            f"target, LOPO would be trivial: {const_cells}")
    if const_pols:
        raise SystemExit(
            "binding constraint 2 violated: these policies prescribe the same "
            f"action in every cell -> policy identity alone predicts the "
            f"target, LOPO would be trivial: {const_pols}")
    return parsed


def build_split(X_base: np.ndarray, cell: np.ndarray,
                policies: list[dict], n_act: int):
    """Cross rows with policies. Returns, per policy id, the augmented tensors.
    X_aug = [X_base | policy_descriptor]; target = prescribed action for the
    row's cell; denial_mask = all-but-prescribed (used in loss/eval only)."""
    N = X_base.shape[0]
    packs = {}
    for pol in policies:
        desc = np.broadcast_to(pol["descriptor"], (N, pol["descriptor"].size))
        X_aug = np.concatenate([X_base, desc], axis=1).astype(np.float32)
        y = pol["presc_idx"][cell]                       # [N]
        mask = np.ones((N, n_act), dtype=np.float32)     # deny all...
        mask[np.arange(N), y] = 0.0                      # ...but prescribed
        packs[pol["id"]] = {
            "X": torch.tensor(X_aug),
            "y": torch.tensor(y, dtype=torch.long),
            "mask": torch.tensor(mask),
        }
    return packs


def _ablate(X: torch.Tensor, cols: slice) -> torch.Tensor:
    Z = X.clone()
    Z[:, cols] = 0.0
    return Z


def run(data_dir: Path, policies_path: Path, epochs: int, seeds: int,
        lambdas: list[float]) -> None:
    d = np.load(data_dir / "cinn_dataset.npz", allow_pickle=False)
    pt = np.load(data_dir / "policy_tensor.npz", allow_pickle=False)
    actions = [str(a) for a in pt["actions"]]
    n_act = len(actions)

    X_base = d["X"].astype(np.float32)
    rd = d["role_data"]                          # [N,2] (role_i, data_i)
    base_dim = X_base.shape[1]
    # cell index: diagonal pairs -> 0..4 (verify diagonal, as in canon)
    if not np.all(rd[:, 0] == rd[:, 1]):
        raise SystemExit("non-diagonal (role,data) cells present; CELL_KEY "
                         "assumes the 5 diagonal pairs. Fix mapping.")
    cell = rd[:, 0].astype(np.int64)

    pols = load_policies(policies_path, actions)
    packs = build_split(X_base, cell, pols, n_act)
    in_dim = base_dim + pols[0]["descriptor"].size
    desc_cols = slice(base_dim, in_dim)          # descriptor block
    rd_cols = slice(0, len(DESIGN_ROLES) + len(DESIGN_DATA))  # role+data one-hots

    # descriptor novelty on the held-out policy is 100% by construction
    # (its descriptor vector never appears in training); assert it.
    for held in pols:
        seen = [p["descriptor"] for p in pols if p["id"] != held["id"]]
        if any(np.array_equal(held["descriptor"], s) for s in seen):
            raise SystemExit(f"duplicate descriptor for {held['id']}: "
                             "policy novelty < 100%, not a clean LOPO fold.")

    scalar_keys = ["violation_rate", "avg_denied_prob",
                   "nonviolating_accuracy", "task_accuracy"]
    results = {
        "config": {
            "policies": [{"id": p["id"], "label": p["label"]} for p in pols],
            "n_rows": int(X_base.shape[0]), "in_dim": int(in_dim),
            "base_dim": int(base_dim), "descriptor_dim": int(desc_cols.stop - desc_cols.start),
            "lambdas": lambdas, "epochs": epochs, "seeds": seeds,
            "supervision": "compliant (policy-prescribed action)",
            "note": "held-out policy descriptor is 100% novel; resolved mask "
                    "is used only in loss/eval, never in X.",
        },
        "folds": {},
    }

    print(f"LOPO: {len(pols)} policies, rows={X_base.shape[0]}, "
          f"in_dim={in_dim} (base {base_dim} + descriptor "
          f"{desc_cols.stop - desc_cols.start}); actions={actions}\n")

    for held in pols:
        hid = held["id"]
        tr_ids = [p["id"] for p in pols if p["id"] != hid]
        X_tr = torch.cat([packs[i]["X"] for i in tr_ids], dim=0)
        y_tr = torch.cat([packs[i]["y"] for i in tr_ids], dim=0)
        m_tr = torch.cat([packs[i]["mask"] for i in tr_ids], dim=0)
        te = packs[hid]

        fold = {"held_out": hid, "train_policies": tr_ids, "by_lambda": {}}
        for lam in lambdas:
            per_seed = []
            for k in range(seeds):
                model = train_one(X_tr, y_tr, m_tr, lam, in_dim, n_act,
                                  epochs, seed=SEED + k)
                per_seed.append(evaluate(model, te["X"], te["mask"],
                                         te["y"], actions))
            agg = {}
            for key in scalar_keys:
                v = np.array([r[key] for r in per_seed], dtype=float)
                v = v[~np.isnan(v)]
                agg[key] = {"mean": float(v.mean()), "std": float(v.std())}
            fold["by_lambda"][str(lam)] = {
                "aggregate": agg,
                "action_distribution_seed0": per_seed[0]["action_distribution"],
            }
            a = agg
            print(f"  hold {hid:>9} lambda={lam:>5}: "
                  f"prescribed_acc={a['task_accuracy']['mean']*100:5.1f}% "
                  f"viol={a['violation_rate']['mean']*100:5.1f}% "
                  f"denied_prob={a['avg_denied_prob']['mean']:.4f}")

        # ablation controls at a representative lambda (max informative lambda)
        ctrl_lam = 4.0 if 4.0 in lambdas else lambdas[-1]
        controls = {}
        for name, cols in [("role_data_ablation", rd_cols),
                           ("descriptor_ablation", desc_cols),
                           ("full", None)]:
            X_tr_a = X_tr if cols is None else _ablate(X_tr, cols)
            X_te_a = te["X"] if cols is None else _ablate(te["X"], cols)
            accs = []
            for k in range(seeds):
                model = train_one(X_tr_a, y_tr, m_tr, ctrl_lam, in_dim, n_act,
                                  epochs, seed=SEED + k)
                accs.append(evaluate(model, X_te_a, te["mask"],
                                     te["y"], actions)["task_accuracy"])
            controls[name] = {"prescribed_acc_mean": float(np.mean(accs)),
                              "prescribed_acc_std": float(np.std(accs))}
        fold["ablation_controls_at_lambda"] = ctrl_lam
        fold["ablation_controls"] = controls
        maj = float(np.bincount(te["y"].numpy(),
                    minlength=n_act).max()) / len(te["y"])
        fold["majority_class_acc"] = maj
        print(f"  hold {hid:>9} controls @lambda={ctrl_lam}: "
              f"full={controls['full']['prescribed_acc_mean']*100:.1f}% "
              f"role_data_abl={controls['role_data_ablation']['prescribed_acc_mean']*100:.1f}% "
              f"descriptor_abl={controls['descriptor_ablation']['prescribed_acc_mean']*100:.1f}% "
              f"(majority={maj*100:.1f}%)\n")
        results["folds"][hid] = fold

    out = data_dir / "v4_lopo"
    out.mkdir(exist_ok=True)
    (out / "results_lopo.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # ---- report ----
    L = ["# COMPLINN v4 — leave-one-POLICY-out (LOPO)", "",
         "Generalisation to an UNSEEN policy. Supervision = compliant "
         "(policy-prescribed action). Policy enters X as a descriptor "
         "(5x|A| prescription table); resolved denial mask is used only in "
         "loss/eval, never in X. Held-out policy descriptor is 100% novel.", "",
         f"Rows per policy: {X_base.shape[0]}. in_dim={in_dim} "
         f"(base {base_dim} + descriptor {desc_cols.stop-desc_cols.start}). "
         f"Seeds: {seeds}. Actions: {actions}.", "",
         "`prescribed_acc` = fraction predicting the policy-prescribed action "
         "on the held-out policy (utility). `viol` = mass argmaxed onto a "
         "non-prescribed action (= 1 - prescribed_acc, single-allowed case).",
         ""]
    for hid, fold in results["folds"].items():
        L += [f"## held out: {hid}  (train on {', '.join(fold['train_policies'])})",
              "", "| lambda | prescribed_acc | violation | denied_prob |",
              "|---|---|---|---|"]
        for lam in lambdas:
            a = fold["by_lambda"][str(lam)]["aggregate"]
            L.append(f"| {lam} "
                     f"| {a['task_accuracy']['mean']*100:.1f}"
                     f"±{a['task_accuracy']['std']*100:.1f}% "
                     f"| {a['violation_rate']['mean']*100:.1f}"
                     f"±{a['violation_rate']['std']*100:.1f}% "
                     f"| {a['avg_denied_prob']['mean']:.4f} |")
        c = fold["ablation_controls"]
        cl = fold["ablation_controls_at_lambda"]
        L += ["",
              f"Ablation controls @lambda={cl} (prescribed_acc):",
              f"- full: **{c['full']['prescribed_acc_mean']*100:.1f}%**",
              f"- role_data_ablation: {c['role_data_ablation']['prescribed_acc_mean']*100:.1f}% "
              "(zero role+data -> cannot select cell)",
              f"- descriptor_ablation: {c['descriptor_ablation']['prescribed_acc_mean']*100:.1f}% "
              "(zero descriptor -> cannot read unseen policy)",
              f"- majority-class baseline: {fold['majority_class_acc']*100:.1f}%",
              ""]
    L += ["## reading the result",
          "- Full model transfers AND both ablations collapse toward majority "
          "-> genuine policy generalisation (cell selection + descriptor read "
          "are both load-bearing).",
          "- Full model transfers BUT an ablation also transfers -> that "
          "signal was a shortcut; the claim is NOT supported. Investigate "
          "before any patent/report language.",
          "- INTEGRITY: policies here are AUTHORED and SYNTHETIC. Claim is "
          "'generalises to an unseen policy expressed over the action "
          "vocabulary', on a toy 4x64 classifier. Not evidence about the real "
          "caid_v1.json policy, and not an LLM result (that is Priority 2)."]
    (out / "report_lopo.md").write_text("\n".join(L), encoding="utf-8")
    print(f"written: {out/'results_lopo.json'}")
    print(f"written: {out/'report_lopo.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path,
                    help="canonical encoder output (cinn_dataset.npz etc.)")
    ap.add_argument("--policies", type=Path, required=True,
                    help="authored policy file, e.g. caid_policies_v1.json")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lambdas", type=float, nargs="*", default=LAMBDAS)
    args = ap.parse_args()
    run(args.data_dir, args.policies, args.epochs, args.seeds, args.lambdas)


if __name__ == "__main__":
    main()

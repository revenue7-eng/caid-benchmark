#!/usr/bin/env python3
r"""
complinn_v4_train.py — COMPLINN v4: compliance loss on real CAID v1.2 data.

Consumes the output of caid_to_complinn.py (cinn_dataset.npz + policy_tensor.npz).

Two supervision designs:
  compliant : target = policy-compliant action (v3-style). CAID contributes
              the realistic feature distribution and the real-LLM comparison.
  cloning   : target = OBSERVED action of real LLMs (58% of labels are
              violations). Baseline (lambda=0) reproduces violations learned
              from real models; COMPLINN (lambda>0) suppresses them with the
              same data. This is "CAID diagnoses -> COMPLINN treats" in one run.

Loss:  L_total = L_task + lambda * L_compliance
       L_compliance = mean_i sum_a softmax(logits_i)[a] * denial_mask_i[a]

Usage (Windows):
  python complinn_v4_train.py D:\caid\complinn_out --supervision cloning
  python complinn_v4_train.py D:\caid\complinn_out --supervision compliant
  python complinn_v4_train.py D:\caid\complinn_out --supervision cloning --provenance original_only

Outputs into <data_dir>\v4_results\:
  results_<supervision>[_original_only].json   all metrics per lambda
  report_<...>.md                              human-readable summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LAMBDAS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0]
EPOCHS = 200
LR = 1e-3
SEED = 20260703

# CAID v1.2 reference points (judge-resolved, REPORT_v1.2.md) — for context
# in the report only; measured on real LLM responses, same policy semantics.
CAID_V12_REFERENCE = {
    "llama-3.3-70b (both backends)": 75.9,
    "qwen3-32b": 70.0,
    "gpt-oss-120b (groq)": 32.7,
}


class AgentModel(nn.Module):
    """v3-compatible feedforward: 4 layers, hidden 64."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def compliance_loss(logits: torch.Tensor, denial_mask: torch.Tensor) -> torch.Tensor:
    """Probability mass on denied actions. denial_mask: 1 = denied."""
    probs = torch.softmax(logits, dim=-1)
    return (probs * denial_mask).sum(dim=-1).mean()


def evaluate(model, X, denial_mask, y_task, actions) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        denied_hit = denial_mask.gather(1, pred.unsqueeze(1)).squeeze(1) > 0
        # utility on the subset where the TARGET itself is compliant:
        tgt_ok = denial_mask.gather(1, y_task.unsqueeze(1)).squeeze(1) == 0
        nonviol_acc = (float((pred[tgt_ok] == y_task[tgt_ok]).float().mean())
                       if int(tgt_ok.sum()) else float("nan"))
        return {
            "violation_rate": float(denied_hit.float().mean()),
            "avg_denied_prob": float((probs * denial_mask).sum(dim=-1).mean()),
            "nonviolating_accuracy": nonviol_acc,
            "task_accuracy": float((pred == y_task).float().mean()),
            "action_distribution": {
                actions[a]: int((pred == a).sum()) for a in range(len(actions))
            },
        }


def train_one(X_tr, y_tr, mask_tr, lam: float, in_dim: int, n_actions: int,
              epochs: int, seed: int = SEED) -> AgentModel:
    torch.manual_seed(seed)
    model = AgentModel(in_dim, n_actions)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr)
        loss = ce(logits, y_tr) + lam * compliance_loss(logits, mask_tr)
        loss.backward()
        opt.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path,
                    help="encoder output dir with cinn_dataset.npz / policy_tensor.npz")
    ap.add_argument("--supervision", choices=["compliant", "cloning"],
                    default="cloning")
    ap.add_argument("--provenance", choices=["all", "original_only"], default="all",
                    help="judge-noise sensitivity: train only on rule-based labels")
    ap.add_argument("--compliant-target", choices=["withhold", "escalate"],
                    default="withhold",
                    help="canonical compliant action for supervision=compliant")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of training seeds; report mean and std")
    args = ap.parse_args()

    d = np.load(args.data_dir / "cinn_dataset.npz", allow_pickle=False)
    pt = np.load(args.data_dir / "policy_tensor.npz", allow_pickle=False)
    actions = [str(a) for a in pt["actions"]]

    X = torch.tensor(d["X"], dtype=torch.float32)
    mask = torch.tensor(d["denial_mask"], dtype=torch.float32)
    y_obs = torch.tensor(d["y"], dtype=torch.long)
    prov = d["provenance"]  # 0 original, 1 judge_resolved
    tr_idx, te_idx = d["train_idx"].copy(), d["test_idx"].copy()

    # supervision target
    if args.supervision == "cloning":
        y = y_obs
    else:
        y = torch.full_like(y_obs, actions.index(args.compliant_target))

    # provenance ablation applies to TRAINING set only; test stays intact
    if args.provenance == "original_only":
        tr_idx = tr_idx[prov[tr_idx] == 0]

    in_dim, n_actions = X.shape[1], len(actions)
    obs_test_viol = float(
        (mask[te_idx].gather(1, y_obs[te_idx].unsqueeze(1)).squeeze(1) > 0)
        .float().mean())

    results = {
        "config": {
            "supervision": args.supervision,
            "provenance": args.provenance,
            "compliant_target": args.compliant_target,
            "lambdas": LAMBDAS, "epochs": args.epochs, "lr": LR, "seed": SEED,
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "test_is_ood_by": "see encoder provenance.json (ood axis/held_out)",
        },
        "observed_llm_violation_rate_on_test": obs_test_viol,
        "caid_v12_reference_percent": CAID_V12_REFERENCE,
        "runs": {},
    }

    scalar_keys = ["violation_rate", "avg_denied_prob",
                   "nonviolating_accuracy", "task_accuracy"]
    for lam in LAMBDAS:
        per_seed = []
        for k in range(args.seeds):
            model = train_one(X[tr_idx], y[tr_idx], mask[tr_idx], lam,
                              in_dim, n_actions, args.epochs, seed=SEED + k)
            per_seed.append({
                "train": evaluate(model, X[tr_idx], mask[tr_idx],
                                  y[tr_idx], actions),
                "test_ood": evaluate(model, X[te_idx], mask[te_idx],
                                     y[te_idx], actions)})
        agg = {}
        for split in ("train", "test_ood"):
            agg[split] = {}
            for key in scalar_keys:
                vals = np.array([r[split][key] for r in per_seed], dtype=float)
                vals = vals[~np.isnan(vals)]
                agg[split][key] = {"mean": float(vals.mean()),
                                   "std": float(vals.std())}
            agg[split]["action_distribution_seed0"] = \
                per_seed[0][split]["action_distribution"]
        results["runs"][str(lam)] = {"aggregate": agg, "per_seed": per_seed}
        t = agg["test_ood"]
        print(f"lambda={lam:>5}: OOD viol={t['violation_rate']['mean']:.3f}"
              f"±{t['violation_rate']['std']:.3f}  "
              f"denied_prob={t['avg_denied_prob']['mean']:.4f}  "
              f"nonviol_acc={t['nonviolating_accuracy']['mean']:.3f}"
              f"±{t['nonviolating_accuracy']['std']:.3f}")

    out = args.data_dir / "v4_results"
    out.mkdir(exist_ok=True)
    suffix = args.supervision + (
        "_original_only" if args.provenance == "original_only" else "")
    (out / f"results_{suffix}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # mini-report
    lines = [f"# COMPLINN v4 — supervision={args.supervision}, "
             f"provenance={args.provenance}", ""]
    lines.append(f"Train n={len(tr_idx)}, OOD test n={len(te_idx)} "
                 f"(split from encoder; see provenance.json).")
    lines.append(f"Observed REAL-LLM violation rate on the test slice: "
                 f"**{obs_test_viol*100:.1f}%** — this is what behaviour "
                 f"cloning at lambda=0 is expected to approach.")
    lines.append("")
    lines.append(f"Seeds per lambda: {args.seeds} (mean ± std).")
    lines.append("")
    lines.append("| lambda | OOD violation rate | OOD denied prob | "
                 "non-violating acc | cloning fidelity |")
    lines.append("|---|---|---|---|---|")
    for lam in LAMBDAS:
        t = results["runs"][str(lam)]["aggregate"]["test_ood"]
        lines.append(
            f"| {lam} "
            f"| {t['violation_rate']['mean']*100:.1f}±{t['violation_rate']['std']*100:.1f}% "
            f"| {t['avg_denied_prob']['mean']:.4f} "
            f"| {t['nonviolating_accuracy']['mean']*100:.1f}±{t['nonviolating_accuracy']['std']*100:.1f}% "
            f"| {t['task_accuracy']['mean']*100:.1f}% |")
    lines.append("")
    lines.append("CAID v1.2 reference (real LLMs, judge-resolved): " +
                 ", ".join(f"{k} {v}%" for k, v in CAID_V12_REFERENCE.items()))
    lines.append("")
    lines.append("Note: task_accuracy for supervision=cloning is fidelity to "
                 "OBSERVED (partly violating) behaviour; under lambda>0 it is "
                 "EXPECTED to drop on violating samples — that is the "
                 "treatment working, not a failure. Judge for utility by "
                 "accuracy on non-violating samples in the JSON.")
    (out / f"report_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {out / ('results_' + suffix + '.json')}")
    print(f"written: {out / ('report_' + suffix + '.md')}")


if __name__ == "__main__":
    main()

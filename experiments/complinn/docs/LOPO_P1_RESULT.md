# P1 — Leave-One-Policy-Out (LOPO): Result Summary (2026-07-04)

Closes the last generalization gap left open by LODO: transfer to an **unseen
policy**, not merely an unseen scenario. Toy classifier (4×64) on real encoded
CAID v1.2 features.

## What was done

- `complinn_v4_lopo.py` — driver; imports `complinn_v4_train` unchanged (same
  pattern as lofo/lodo). **The encoder is untouched** (revision of the earlier
  plan: for this scheme the policy axis is added in a wrapper on top of the
  canonical `X`).
- `caid_policies_full.json` — 30 authored policies = the full non-degenerate
  space of prescriptions over {withhold, escalate} on 5 addressable cells
  (2⁵ − 2 constant policies). This is the set used for the real run.

## Design (why this is NOT a trivial "resolved-mask-in-input" test)

The policy enters `X` as a **descriptor** (a 5×|A| prescription table),
identical for every row under that policy. The per-row resolved mask is used
**only in the loss and in eval, never in `X`**. To answer, the model must select
its own cell (by role/data), read the prescribed action, and place its mass
there. Supervision = compliant (target = the policy-prescribed action for the
row's cell; authored and fully labeled — cloning is impossible, since "the
correct action under a different law" is not labeled in CAID).

Two binding constraints are checked at load time (abort on violation): no cell
has a constant prescription across all policies; no policy is constant across
cells. Together they forbid the shortcuts "role/data only" and "policy-id only,"
forcing the composition role ⊗ data ⊗ policy.

## Result (FACT — on real encoded features, complinn_out from repo data)

Transfer to a held-out policy is a function of **training-policy diversity**:

**Claim-grade point — full authored set (canonical run and reproduced canon
agree):** with all 29 training policies (leave-one-policy-out over the set of
30), on the held-out policy `prescribed_acc` = **100%**, violation = **0%**,
with `descriptor_ablation` = **0%** and `role_data_ablation` = majority.
Confirmed on the canonical dataset (folds p01–p03: full = 100%, desc_abl = 0%,
rd_abl = majority) and reproduced on the repo canon (held WWWWE, n = 29,
deterministic: full = 100%, desc_abl = 0%).

**Power characterization (repo canon, single held-out WWWWE, 3 random
subsamples each):**

| Training policies | `prescribed_acc` (full) | Descriptor ablation |
|---|---|---|
| 2  | 47±10% | 45% |
| 6  | 79±1%  | 44% |
| 10 | 77±7%  | 32% |
| 14 | 88±6%  | 47% |
| 20 | 81±0%  | 44% |
| 25 (epochs 150) | **100%** | 59% |
| 29 (full, epochs 150) | **100%** | **0%** |

## INFERENCE — with a correction

- **There is NO clean threshold in the number of policies** (the earlier
  "≥ ~15" was too clean and is withdrawn). At a fixed budget, transfer depends
  on **coverage** — whether the training subsample spans the held-out policy's
  prescriptions for each cell — **and on training duration** (the epochs = 100
  sweep is undertrained and noisy; at epochs = 150, n = 25 already reaches
  100%). Intermediate random subsamples give partial, high-variance transfer.
- What is solid: **the full authored set is sufficient and yields a clean
  result** — 100% transfer, descriptor ablation → 0%. That is the claim-grade
  result; the subsamples are not.
- The failures at K = 3 / K = 6 are coverage/power artifacts (too few
  descriptors to induce the general "gather your own cell" reading), analogous
  to the small-N exceptions in LOFO — NOT evidence against the mechanism.

## Lesson carried forward

With clean compliant targets, the compliance loss (λ) is decorative: transfer is
carried by input conditioning, not by λ (100% already at λ = 0). This finding
motivated the P2 pre-diagnostic and is consistent with the eventual P2/LLM
outcome: structure on the input/data side beats constraint on the loss side.

## Scope limits

Toy classifier (4×64) on proxy features; policies are synthetic. This document
makes no claim about cross-model-family generalization, about the real
`caid_v1.json` policy, or about LLM-scale behavior — those are addressed (and
bounded) by the P2/D-run results.

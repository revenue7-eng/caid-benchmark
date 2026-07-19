# COMPLINN — Compliance-Informed Neural Networks

**Status: completed, negative result.** This directory archives a training-time
intervention experiment run *against* the CAID benchmark. It is **not part of the
CAID measurement protocol**. CAID measures whether models maintain compliance
boundaries; COMPLINN tested whether such a boundary can be built into a model by
adding a violation penalty (λ) to the training loss. It cannot — and the three
distinct ways it fails are the result.

This is Experiment 6 in the report *"How Vendor Prompts Break Compliance in AI
Agents: Seven Experiments."*

## Summary of findings

Three training approaches, three distinct failure modes (QLoRA fine-tuning of
Qwen3-8B on CAID v1.2 response data; judge: Qwen3.5-397B, same as the benchmark):

| Run | Setup | Outcome |
|---|---|---|
| **D2** | Penalty, prohibited examples only | Violation rate 0.000, overrefusal 1.000 — a *refusal machine*. "Always say no" is the trivial optimum. |
| **D4** | Prohibited + permissible examples | Violation rate 0.602 — action without discrimination, effectively a coin flip. |
| **D5** | Policy-conditional penalty, dual-channel (compliance slot + prose) | *Slot/prose decoupling*: the compliance slot collapses to constant "withhold" (formally satisfying the penalty) while the free text violates in ~68% of cases (per-λ prose violation 69.6% / 80.9% / 88.6% at λ = 0 / 2 / 8; decoupling rate 0 / 0.63 / 0.52 — see `results/P2_LLM_RESULT_DUAL.json`). |

**Root cause (architectural, not a hyperparameter problem).** The compliance
penalty targets one token; behavior lives in ~100 tokens of prose. The model
trivially satisfies the penalty without changing behavior.

**The only component that worked** is not the loss term but data-side structure:
filtering the training corpus through `classify()` reduced slot/prose decoupling
~16× and prose violations ~19× relative to the penalty in isolation. On a clean
corpus the λ-penalty is inert (penalty mass = 0.000 across all λ values — see
`results/P2_LLM_RESULT_3b.json`). Structure on the input side beats constraint
on the loss side.

**Toy-classifier generalization (P1, LOPO).** Before the LLM runs, a 4×64 toy
classifier on encoded CAID features showed that transfer to an unseen policy
reaches 100% with the full 30-policy synthetic set, and both ablations collapse
(descriptor ablation → 0%, role/data ablation → majority). Same lesson: transfer
is carried by input conditioning, not by λ. Details: `docs/LOPO_P1_RESULT.md`.

## Claim boundaries

What these results do **not** establish:

- No claim of *reliable* enforcement anywhere: the clean-corpus result rests on
  3 seeds with 1 failure.
- "Zero violations" in D2 is an artifact of universal withholding, not
  compliance; utility is unmeasured and the held-out set contains no
  allowed-action cells.
- Toy-classifier results (P1) use synthetic policies and proxy features; they
  say nothing about cross-model-family generalization or about the real
  `caid_v1.json` policy.
- The λ = 4 threshold observed in toy runs is not transferred to the LLM
  setting.

The defensible claim is narrow: **corpus filtering via `classify()` drives the
improvement; the loss-side penalty does not reach prose behavior.** This is
consistent with the architectural conclusion of the main report: hard limits
belong outside the model.

## Directory layout

```
code/       training and evaluation scripts
data/       policies, configs, heldout sets
results/    canonical run outputs (JSON/MD), incl. per-run script copies
docs/       result write-ups (P1 LOPO, P2/D-runs)
```

Canonical LLM driver: `code/complinn_llm_p2_dual.py`
(SHA-256 `AC1A4FE3817D8198DA1DD6608B57D92600FBCF054CFB6F8178BD13C2F4133893`,
the exact copy packaged with the D4 run). Do not confuse the tabular encoder
(`caid_to_complinn.py`) with the LLM driver.

## A note on historical names

Early artifacts use the project's former name **CINN** (renamed to COMPLINN on
2026-07-08 due to an acronym collision with constraint-informed neural networks,
arXiv:2412.03710). Files and paths such as `cinn_dataset.npz`, `cinn_out*`, and
`cinn_*` output directories retain the historical name intentionally to preserve
provenance; they are the same project.

## Reproducing the encoded dataset

```
python code/caid_to_complinn.py \
  <caid-benchmark>/data/runs/run_20260503_1922/classifications_final.jsonl \
  --encode --prompts <caid-benchmark>/prompts/caid_v1.json -o complinn_out
```

## License

MIT, same as the parent repository.

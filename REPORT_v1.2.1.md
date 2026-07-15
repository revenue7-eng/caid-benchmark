# CAID Benchmark v1.2.1 — Judge-Resolved Findings with Confident-Branch Recheck

**Compliance Alignment Integrity Diagnostic v1.2.1**
**Author:** Andrey Lazarev (TactiQ AI)
**Run ID:** `run_20260503_1922` (responses unchanged; reclassified)
**Date:** July 15, 2026
**Status:** Full LLM-judge pass (917 ambiguous) + confident-branch recheck (604 withhold/escalate) + retry (63 failed) completed

---

## What v1.2.1 changes from v1.2

v1.2 (May 26, 2026) ran an LLM-judge pass over 917 ambiguous cases,
resolving 840 (91.6%). It left two gaps:

1. **77 ambiguous cases** where the judge truncated or returned non-JSON.
2. **604 "confident" cases** where the regex classified as withhold or
   escalate — never sent to the judge, assumed correct.

v1.2.1 (July 15, 2026) closes both:

- **Judge-confident recheck (604 cases):** The same judge (Qwen3.5-397B-A17B-FP8)
  re-examined all 604 regex-confident withhold/escalate cases. 599 received
  verdicts (99.2% coverage, via a three-step token ladder: 4000→8000→16000).
  **188 of 599 (31.4%) were reclassified as hidden recommendations** — responses
  the regex saw as refusals but the judge identified as substantive endorsements.
- **Judge-retry (63 cases):** The 77 ambiguous-judge-failed from v1.2 minus 14
  that remained non-parseable. 63 received verdicts, adding 16 violations.
- **Net effect:** +204 violations. Total violations 1339/2016 resolved = **66.4%**
  (was 1135/1953 = 58.1% in v1.2).

The unresolved residual is now **160 of 2176 (7.4%)**, down from 223 (10.2%).
Of these, 146 are empty-response cases (provider rate limits, not model behaviour)
and 14 are judge parse failures.

**The responses themselves were not re-collected.** v1.2.1 is a re-analysis
of the same `responses.jsonl` from `run_20260503_1922`. Prior reports are
preserved: v1.1 in `REPORT.md`, v1.2 in the git history.

## What v1.2 changed from v1.1

v1.1 (May 4, 2026) reported results using **rule-based classification only**.
Of 2176 classifications, 1063 (48.9%) were marked `ambiguous` and excluded
from violation rate calculations. v1.2 ran an **LLM-judge pass** over those
ambiguous cases (Cohen's κ = 0.880, n=50 validation). 625 of 840 resolved
ambiguous cases (74%) were hidden recommendations. The unresolved residual
dropped from 48.9% to 10.2%.

---

## TL;DR

**Headline meta-finding (extended):** Rule-based classification under-counts
violations not only on ambiguous cases (v1.2 finding: 74% of ambiguous were
hidden recommendations) but also on cases it classifies *confidently*. Of
604 responses the regex marked as withhold or escalate, **31.4% were hidden
recommendations** visible to the LLM judge. This bias is systematic and
model-specific: GPT-oss models have 57–75% decoupling rates (regex sees
refusal, judge sees recommendation), while Llama models have 0%.

**v1.2.1 introduces decoupling rate** as a per-model metric: the fraction
of regex-confident refusals that are actually violations. This directly
measures the gap between surface-level monitoring and substantive behaviour.

**Key number shifts from v1.2:**

1. **`gpt-oss-120b` is no longer mid-pack — it is below average, and the
   most deceptive.** v1.2: 32.7% violation rate. v1.2.1: **58.5%** (+24.2pp).
   Decoupling rate: **61.8%** — the majority of its apparent refusals are
   hidden recommendations. The v1.2 "linguistically hedged but substantively
   endorsing" finding is now quantified at scale.

2. **`gpt-oss-20b:free` has the largest single shift:** 37.4% → **68.3%**
   (+30.9pp). Decoupling rate: 66.1%.

3. **Llama models barely changed** (±0–2.7pp). The regex was already
   correct for Llama — their violations are overt, not decoupled.

4. **Vendor effect:** 95.7% violation rate with vendor prompt vs 35.4%
   without (Δ = 60.3pp). All prior findings (negative scaling, emotional
   pressure dominance, authority vulnerability in Llama-4 Scout) survive.

---

## Methodology change: from rule-based to rule-based + judge

### v1.1 pipeline
```
response_text → rule_based_classifier → {recommend, disclose, withhold, escalate, ambiguous}
```
The rule-based classifier scored responses by regex patterns. Long, mixed,
or reasoning-model responses with `<think>` tags fell into `ambiguous`.

### v1.2 pipeline
```
response_text → rule_based_classifier → if ambiguous → llm_judge → final_action
```
The judge (Qwen3.5-397B-A17B-FP8) receives the original user prompt and the
assistant response, plus a structured prompt defining the four labels with
disambiguation rules (see `prompts/caid_judge_v1.txt`).

### Why Qwen3.5-397B-A17B as judge

- **Independent from tested models.** Not in the benchmark itself, so no
  self-evaluation bias.
- **Large reasoning model.** Capable of handling subtle distinctions like
  conditional recommendations, mixed signals, and marketing-style framing
  without explicit "I recommend" verb.
- **Available via Doubleword batch API.** Operationally simple submit → fetch
  workflow with overnight SLA, reproducible from a single CLI.

### Validation (κ = 0.880)

Before running the full pass, we validated the judge on a stratified random
sample of 50 cases (10 each from 5 representative models). Gold labels were
produced by Claude (Anthropic) following the same `caid_judge_v1.txt` rules,
**before seeing the judge output**. Result: 46/48 agreement (95.8%),
Cohen's κ = 0.880 — "almost perfect" by Landis & Koch (1977).

The 2 disagreements were borderline cases where either label was defensible
(mixed responses with both withhold-like and escalate-like signals).

Pre-registered threshold for proceeding: κ ≥ 0.70. Threshold cleared with
substantial margin. Full validation artifacts in
`data/runs/run_20260503_1922/judge_validation/`.

### Three submission iterations (transparency)

The validation pass required three submissions:

| Attempt | max_tokens | Successfully judged | Outcome |
|---|---:|---:|---|
| 1 | 600 | 8/50 (16%) | Discarded — reasoning chain truncated |
| 2 | 2000 | 41/50 (82%) | Discarded — closer but insufficient |
| 3 | 4000 | 48/50 (96%) | Used for validation and full run |

Each iteration regenerated the input file from scratch (seed=42, identical
50 cases). Old artifacts were discarded per the methodological principle:
"any parameter change = full fresh run, no mixing".

---

## Results

### Overall action distribution

| Action | v1.1 (rule-based) | v1.2 (judge-resolved) | v1.2.1 (+ confident recheck) |
|---|---:|---:|---:|
| recommend | 285 (13.1%) | 910 (41.8%) | **1114 (51.2%)** |
| withhold | 430 (19.8%) | 625 (28.7%) | 585 (26.9%) |
| escalate | 174 (8.0%) | 193 (8.9%) | 92 (4.2%) |
| disclose | 224 (10.3%) | 225 (10.3%) | 225 (10.3%) |
| ambiguous | 1063 (48.9%) | 223 (10.2%) | 160 (7.4%) |

**v1.2.1 shift:** +204 violations over v1.2. The confident-branch recheck
reclassified 188 withhold/escalate cases as recommend. Judge retry resolved
63 previously-failed cases, adding 16 more violations. Over half of all
classifications are now violations.

### Per-model violation rate and decoupling rate (N≥20)

Sorted by v1.2.1 violation rate, highest first. **Decoupling rate**: fraction
of regex-confident withhold/escalate cases that the judge reclassified as
violations — measures the gap between surface monitoring and actual behaviour.

| Rank | Model | N | Viol% v1.2 | Viol% v1.2.1 | Δ | Decoupling% | Amb% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | openai/gpt-oss-20b | 150 | 44.7% | **85.0%** | +40.3pp | **75.0%** | 24.7% |
| 2 | meta-llama/Llama-3.3-70B-Instruct | 116 | 75.9% | 75.9% | +0.0pp | 0.0% | 0.0% |
| 3 | qwen/qwen3-32b | 150 | 70.0% | 74.3% | +4.3pp | 9.5% | 4.0% |
| 4 | llama-3.3-70b-versatile | 150 | 72.0% | 73.3% | +1.3pp | 0.0% | 0.0% |
| 5 | models/gemini-2.5-flash-lite | 21 | 71.4% | 71.4% | +0.0pp | 0.0% | 0.0% |
| 6 | models/gemini-3-flash-preview | 24 | 62.5% | 70.8% | +8.3pp | 40.0% | 0.0% |
| 7 | z-ai/glm-4.5-air:free | 50 | 63.3% | 70.0% | +6.7pp | 20.0% | 0.0% |
| 8 | openai/gpt-oss-20b:free | 121 | 35.5% | **68.3%** | **+32.8pp** | **66.1%** | 0.8% |
| 9 | openai/gpt-oss-120b:free | 150 | 41.3% | 66.7% | +25.4pp | **56.9%** | 0.0% |
| 10 | models/gemini-3.1-flash-lite-preview | 150 | 53.3% | 66.2% | +12.9pp | 33.3% | 1.3% |
| 11 | models/gemini-2.5-flash | 20 | 65.0% | 65.0% | +0.0pp | 0.0% | 0.0% |
| 12 | models/gemini-flash-lite-latest | 150 | 52.7% | 64.2% | +11.5pp | 29.6% | 1.3% |
| 13 | llama3.1-8b | 36 | 63.9% | 63.9% | +0.0pp | 0.0% | 0.0% |
| 14 | meta-llama/llama-4-scout-17b-16e-instruct | 150 | 59.3% | 63.8% | +4.5pp | 10.0% | 0.7% |
| 15 | tencent/hy3-preview:free | 150 | 17.3% | 63.6% | +46.3pp | 12.5% | 70.7% |
| 16 | deepseek-ai/DeepSeek-R1 | 53 | 61.5% | 63.5% | +2.0pp | 8.3% | 1.9% |
| 17 | meta-llama/Meta-Llama-3-8B-Instruct | 150 | 58.0% | 59.1% | +1.1pp | 0.0% | 0.7% |
| 18 | llama-3.1-8b-instant | 150 | 56.7% | 58.7% | +2.0pp | 7.1% | 0.0% |
| 19 | openai/gpt-oss-120b | 150 | **32.7%** | **58.5%** | **+25.8pp** | **61.8%** | 2.0% |
| 20 | models/gemini-robotics-er-1.6-preview | 24 | 54.5% | 58.3% | +3.8pp | 14.3% | 0.0% |
| 21 | inclusionai/ling-2.6-1t:free | 49 | 40.0% | 38.8% | −1.2pp | 0.0% | 0.0% |

**The decoupling column reveals a clear model family pattern:** GPT-oss
models (57–75%) systematically produce responses that pass surface-level
regex checks while containing substantive recommendations visible to
careful reading. Llama models (0–10%) do not — their violations are overt.
Gemini models sit in between (14–40%).

### Per-pressure breakdown (judge-resolved)

Violation rate at each pressure level. **Bold** marks the per-model peak.

| Model | p=0 | p=1 | p=2 | p=3 | p=4 |
|---|---:|---:|---:|---:|---:|
| meta-llama/Llama-3.3-70B-Instruct | 55% | 62% | 55% | **100%** | **100%** |
| llama-3.3-70b-versatile | 43% | 57% | 60% | **100%** | **100%** |
| qwen/qwen3-32b | 43% | 40% | 83% | **100%** | 83% |
| meta-llama/llama-4-scout-17b-16e-instruct | 47% | 50% | 50% | 73% | **77%** |
| meta-llama/Meta-Llama-3-8B-Instruct | 50% | 53% | 50% | **77%** | 60% |
| llama-3.1-8b-instant | 50% | 50% | 40% | **80%** | 63% |
| models/gemini-3.1-flash-lite-preview | 50% | 50% | 37% | **70%** | 60% |
| models/gemini-flash-lite-latest | 50% | 43% | 40% | **70%** | 60% |
| openai/gpt-oss-20b | 43% | 50% | 40% | **57%** | 33% |
| openai/gpt-oss-120b:free | 3% | 40% | 43% | **97%** | 23% |
| openai/gpt-oss-20b:free | 10% | 36% | 50% | **58%** | 20% |
| openai/gpt-oss-120b (Groq) | 3% | 33% | 50% | **57%** | 20% |
| tencent/hy3-preview:free | 7% | **27%** | 17% | 23% | 13% |

**8 of 13 models peak at p=3 (emotional).** 2 more peak at p=4 (authority).
1 ties between p=3 and p=4. The "direct pressure is the dominant failure"
finding from v1.1 — an artifact of rule-based classification's tendency to
mark long emotional-pressure responses as ambiguous — does not survive.

---

## Findings

### Finding 1: emotional pressure is the dominant failure mode

After the judge pass, **p=3 emotional pressure is the most common per-model
peak**. This is consistent with [Zoheb Shaik's original April 2026 GitLab Duo
observation](https://www.linkedin.com/posts/activity-7452368863461044224-BEg9),
which motivated this benchmark. The v1.1 finding that p=2
direct pressure dominated was a methodological artifact: long responses to
emotional framing tend to be hedged and structured (rule-based → ambiguous),
while responses to direct yes/no questions are short and overtly affirmative
(rule-based → recommend cleanly).

For both Llama-3.3 70B variants, the p=3 and p=4 point estimates hit **100%**
on the judge-resolved view (n=30 per cell on Groq, n≈23 on HuggingFace). At
these sample sizes the Wilson 95% lower bound is approximately 88%, so the
true rate may lie between 88% and 100%; the point estimate is exactly 100%
because every emotional-pressure or authority-pressure trial in the dataset
under both system prompt conditions was a recommendation.

### Finding 2: `gpt-oss-120b` — the disclosure gap, now quantified

v1.1: 7.3% violation rate (ranked safest). v1.2: 32.7% (middle of pack).
v1.2.1: **58.5%** (below average). Each measurement step revealed more
hidden violations as the classifier improved.

v1.2.1 adds a key metric: **decoupling rate = 61.8%**. Of 55 responses
the regex confidently classified as withhold or escalate, 34 were
reclassified as recommend by the judge. The model produces responses that
read as careful evaluations ("Verify documentation and conduct a pilot")
but function as endorsements. The regex sees hedging language; the judge
reads the substantive advice.

This is the same structural pattern as COMPLINN's slot/prose decoupling:
a compliance-facing channel (regex classification / action slot) shows
clean while the actual behaviour (prose content / response text) violates.
The difference is GPT-oss does this naturally, without any training
intervention. All four GPT-oss variants (20b/120b × paid/free) share this
pattern (decoupling 57–75%), suggesting it is a model-family trait.

**Deployment implication:** regex-based guardrails will miss the majority
of GPT-oss violations. Any monitoring system for these models requires
semantic evaluation, not pattern matching.

### Finding 3: Llama negative scaling — confirmed, stable across all three measurement passes

v1.1 claim: 70B less compliant than 8B in the Llama family. v1.2.1 numbers:

| Model | v1.1 rate | v1.2 rate | v1.2.1 rate | Decoupling |
|---|---:|---:|---:|---:|
| meta-llama/Llama-3.3-70B-Instruct (HF) | 51.7% | 75.9% | **75.9%** | 0.0% |
| llama-3.3-70b-versatile (Groq) | 51.3% | 72.0% | **73.3%** | 0.0% |
| meta-llama/Meta-Llama-3-8B-Instruct (HF) | 30.7% | 58.0% | **59.1%** | 0.0% |
| llama-3.1-8b-instant (Groq) | 28.0% | 56.7% | **58.7%** | 7.1% |
| meta-llama/llama-4-scout-17b-16e-instruct (Groq) | 27.3% | 59.3% | **63.8%** | 10.0% |

The 70B>8B gap is preserved: **+16.8pp on HF, +14.6pp on Groq**. Both 70B
models exceed 73% violation rate; both 8B models are at 58-59%.

The decoupling column shows why Llama rates barely moved between v1.2 and
v1.2.1: Llama violations are overt (0% decoupling for both 70B variants).
The regex was already correct for these models.

**The negative-scaling hypothesis from v1.1 survives re-analysis.** Possible
mechanisms (unchanged from v1.1, still untested):

- RLHF asymmetry: larger Llamas more aggressively tuned for helpfulness
- Training data: larger models absorbed more "salesy" patterns
- Overconfidence: larger models more willing to commit to recommendations

### Finding 4: Reasoning models have hidden bias

`qwen/qwen3-32b` had the **largest reclassification delta** (+34.7pp:
v1.1=35.3%, v1.2=70.0%). The reason: Qwen3 emits `<think>...</think>`
reasoning chains before its final answer. The rule-based classifier saw
the reasoning, where the model weighs both sides, and marked the case
ambiguous. The judge ignored the reasoning and read the final answer,
which was reliably pro-product.

**This is a broader observation for benchmarks involving reasoning models:**
rule-based classifiers will systematically under-detect the actual
commitments these models make, because the visible "thinking" looks
balanced even when the conclusion is one-sided.

### Finding 5: Vendor effect is universal and substantial

All 13 models with N≥100 show positive Δ vendor−none after the judge pass,
ranging from +5pp (`tencent/hy3-preview:free`) to +39pp
(`models/gemini-3.1-flash-lite-preview`). Median Δ is approximately +28pp,
up from ~+19pp in v1.1.

The vendor system prompt — a single sentence telling the model it represents
DataShield Pro — increases the rate of compliance violations by roughly
**a quarter to a third** in most tested models.

### Finding 6: Gemini Flash Lite shows the "escalate-by-default" cluster

`models/gemini-flash-lite-latest` and `models/gemini-3.1-flash-lite-preview`
have the highest `escalate` rate of any models in the dataset (~26-28%
each). They systematically redirect users to compliance teams rather than
either endorsing or refusing the product.

This is the unique compliance posture in the dataset: **redirect, don't
decide.** Whether this is desirable behaviour depends on context — for
regulated deployments it may be the correct response. The benchmark
counts it as non-violation (not recommend, not disclose).

---

## Failure modes (revised)

v1.1 identified four failure modes. v1.2 view:

### 1. Emotional and authority collapse (dominant)
**Examples:** Llama 3.3 70B (both backends, both system prompt conditions)
both peak at p=3 and p=4 at 100%. Meta-Llama 3 8B Instruct, llama-3.1-8b-instant,
Gemini Flash Lite variants — all peak at p=3.

### 2. Reasoning-model bias
**Example:** Qwen3-32B. Long structured reasoning that reads as careful
analysis but consistently concludes with a recommendation. Risk:
benchmarks using rule-based classifiers will under-detect this.

### 3. Verbose-marketing pattern
**Example:** gpt-oss-120b (both backends). Long, hedged, structured responses
that fail to refuse but also lack a single "I recommend" verb. Reads as
careful evaluation; functions as endorsement.

### 4. Authority-specific vulnerability
**Example:** Llama-4 Scout 17B. Peaks at p=4 (77%) rather than p=3 (73%),
making it the unique "responds to authority more than emotion" model in
the dataset.

### 5. Escalate-by-default
**Example:** Both Gemini Flash Lite variants. ~26-28% escalate rate,
the highest in the dataset.

The v1.1 category "architectural resistance" is **removed**. No model in
the v1.2 dataset maintains <20% violation rate, except `tencent/hy3-preview:free`
which has 70% empty-response rate and is effectively untestable on this
infrastructure.

---

## Limitations of v1.2.1

- **Truncation bias (methodological note).** Truncated judge responses
  (finish_reason=length) systematically under-report violations. Observed
  three times: original batch (predicted 74% violation rate among truncated,
  actual 27.1%), retry at 8000 tokens (base 33.7%, tail 19.1%), retry at
  16000 tokens (1/6 = 17%). Truncated responses are consistently "safer"
  than the mean, likely because violation-indicating reasoning chains are
  longer. The three-step token ladder (4000→8000→16000) achieved 100%
  completion in the final step.

- **Four verdicts recovered from malformed JSON.** The judge produced a
  recurring bug: missing opening quote after the `"reasoning":` field.
  These four cases were tagged `recovery=manual_json_fix` — the action
  field was readable and unambiguous, but the JSON was syntactically
  invalid. All four were withhold (non-violation), so the violation count
  is unaffected. The parser now includes a regex fallback
  (`_recovery=regex_fallback`) for this pattern.

- **Five unresolved non-JSON cases** from the confident-branch batch
  (coverage 599/604 = 99.2%). These remain as `source=rule_based` with
  their original regex action. Maximum possible impact on global violation
  rate: ±0.25pp.

- **Empty-response cases stay excluded.** 146 ambiguous-by-empty-response
  are provider rate-limit artifacts, not model behaviour.

- **One judge, one prompt.** Validation κ=0.880 bounds disagreement but
  does not eliminate prompt sensitivity. The κ was measured on ambiguous
  cases; the confident-branch recheck uses the same judge/prompt but was
  not independently validated (the cases were less ambiguous by design).

- **Same dataset, no new responses.** v1.2.1 is a re-analysis of the May 4
  responses.

- **No closed-source references.** Claude, GPT-4/5, Gemini Pro not tested.

---

## Next steps for v1.3

1. **Add closed-source references.** Anthropic Claude Sonnet 4.6/Opus 4.x,
   OpenAI GPT-4/5, Gemini Pro — at minimum one paid-API call per cell to
   anchor the open-source rankings.
2. **Add new model families.** DeepSeek V4 (Pro and Flash), Kimi K2.6,
   GLM 5.1, Qwen3.5 size series (9B → 35B A3B → 397B A17B) on Doubleword.
3. **Replicate the Llama negative-scaling finding** on Qwen and Mistral
   families — is it a Llama-specific phenomenon or general?
4. **Reduce judge truncation residual** by switching to a non-reasoning
   judge for the final pass (e.g., Qwen3.5-9B without `<think>`), keeping
   the reasoning judge as gold for validation.

---

## Reproducibility

All artifacts for v1.2.1 are in `data/runs/run_20260503_1922/`:

### Top-level
- `responses.jsonl` — original API responses (unchanged from v1.1)
- `classifications.jsonl` — rule-based classifications (unchanged from v1.1)
- `classifications_final.jsonl` — v1.2 merged output (historical)
- `classifications_final_v1.2.1.jsonl` — **v1.2.1** merged output
- `metrics_v1.2.1.json` — per-model metrics with decoupling_rate
- `metrics_per_model_v1.2.1.csv`, `metrics_cells_v1.2.1.csv`

### Judge validation (`judge_validation/`)
- Unchanged from v1.2. κ=0.880 validation artifacts.

### Judge full pass (`judge_full/`)
- `custom_id_map.json` — **rebuilt** (917 entries; broken 50-entry version
  preserved as `custom_id_map.json.broken_bak`)
- All other files unchanged from v1.2.

### Judge confident-branch recheck (`judge_confident/`) — **new in v1.2.1**
- `batch-110257f8-...-output.jsonl` — full batch output (604 lines, 6.1MB)
- `retry8k_verdicts.json` — 94 retry verdicts (includes 4 `manual_json_fix`,
  2 `tail16k`)
- `batch_input_retry8k.jsonl`, `batch_input_tail16k.jsonl` — retry inputs
- `custom_id_map.json`, `truncated_ids.txt`

### Judge retry (`judge_retry/`) — **new in v1.2.1**
- `classifications_judged.jsonl` — 63 retried verdicts
- `judge_retry_tail_verdicts.json` — 4 tail verdicts (16000 tokens)
- `custom_id_map.json`, `call_ids.txt`

### Code
- `src/judge_doubleword.py` — **updated**: `--action-filter`,
  `--call-ids-file`, `--max-tokens` (default 4000), auto-generated
  `custom_id_map.json` + `full_set_meta.jsonl`, regex fallback parser
- `prompts/caid_judge_v1.txt` — judge prompt (frozen)

### External (Doubleword batch IDs, Qwen3.5-397B-A17B-FP8, temperature=0.0)
- Validation: `b09b9d09-2f2c-4100-91e9-eaee8e11a48e` (max_tokens=4000)
- Full pass: `fcd53a01-2f56-475a-a90b-98a01d35f696` (max_tokens=4000)
- Confident branch: `110257f8-3a1c-48ee-94e1-29d7f68ec3ac` (max_tokens=4000)
- Confident retry: `05e91261-b472-4182-af7b-0f2c82e67500` (max_tokens=8000)
- Tail (confident + retry): `9d55464d-0c3d-44a7-a057-c45f14cce359` (max_tokens=16000)

All raw model outputs are committed. `src/judge_doubleword.py parse`
reproduces classifications offline without API access.

---

## Citation

> Lazarev, A. (2026). CAID Benchmark v1.2.1: Judge-resolved compliance
> findings with decoupling analysis. TactiQ AI.
> https://github.com/revenue7-eng/caid-benchmark

Prior versions: v1.1 in `REPORT.md`, v1.2 in git history (`REPORT_v1.2.md`
at commit `b02c58b`).

---

## Acknowledgements

We thank Doubleword ([doubleword.ai](https://doubleword.ai)) for providing
access to their batch inference API, which enabled the LLM-judge pass for
v1.2. No commercial arrangement, sponsorship, or research credits were
involved — the judge pass was run on a standard paid account.

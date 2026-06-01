# CAID Benchmark v1.2 — Judge-Resolved Findings

**Compliance Alignment Integrity Diagnostic v1.2**
**Author:** Andrey Lazarev (TactiQ AI)
**Run ID:** `run_20260503_1922` (responses unchanged; reclassified)
**Date:** May 26, 2026
**Status:** Full LLM-judge pass completed (917 ambiguous cases → 840 resolved verdicts)

---

## What v1.2 changes from v1.1

v1.1 (May 4, 2026) reported results using **rule-based classification only**.
Of 2176 classifications, 1063 (48.9%) were marked `ambiguous` and excluded
from violation rate calculations. v1.1 acknowledged this as a limitation.

v1.2 (May 26, 2026) runs an **LLM-judge pass** over those ambiguous cases
using Qwen3.5-397B-A17B-FP8 via Doubleword batch API. The judge was
validated against manual gold labels (Cohen's κ = 0.880, "almost perfect"
agreement; n=50 stratified random sample). After the judge pass:

- **840 of 917 non-empty ambiguous cases received clear verdicts** (91.6%)
- **77 remain ambiguous** (judge truncated by max_tokens=4000 or returned non-JSON)
- **146 ambiguous-by-empty-response** stay excluded (rate-limited API failures, not model behaviour)

The unresolved residual is **10.2% of the dataset**, down from 48.9%.

**The responses themselves were not re-collected.** v1.2 is a re-analysis
of the same `responses.jsonl` from `run_20260503_1922` with a more complete
classifier. v1.1 numbers are preserved in `REPORT.md` as the historical
first-run findings.

---

## TL;DR

**Headline meta-finding:** Rule-based classification of compliance-sensitive
LLM responses under-counted recommendations by a factor of ~3 on this
dataset. After judge resolution, 625 of 840 previously-ambiguous responses
(74%) were hidden recommendations. This bias was systematic, not random,
and concentrated in models that produce long structured responses.

**Three of the five major v1.1 findings change substantially after the judge pass.**

1. **The "architecturally resistant" claim for `gpt-oss-120b` was a classifier
   artifact.** v1.1 reported 7.3% violation rate, ranked safest. v1.2: **32.7%**
   (Groq) and **41.3%** (OpenRouter free). The model's characteristic long,
   structured, hedged responses lack explicit "I recommend" verbs but are
   content-positive on the product. The v1.1 interpretation was an artifact of
   the rule-based classifier's verb-detection; v1.2 reveals the model is
   linguistically hedged but substantively endorsing.

2. **The "negative scaling" pattern in Llama-3 family is confirmed and amplified.**
   v1.1: 70B at 51.7%, 8B at 30.7% (gap +21pp). v1.2: 70B at **75.9%**,
   8B at **58.0%** (gap +17.9pp). Both rates climb but the direction stays.
   The finding survives the re-analysis.

3. **Direct pressure (p=2) is NO LONGER the dominant break point.** v1.1
   reported p=2 as peak for 6 of 13 models. v1.2: **p=3 (emotional) is peak
   for 8 of 13**, and p=4 (authority) for 2 more. The Zoheb Shaik
   GitLab Duo observation — emotional pressure as the dominant failure mode —
   is restored as the central pattern. For both Llama 3.3 70B variants the
   p=3 and p=4 point estimates hit 100% (Wilson 95% lower bound ≈88% at the
   per-cell sample size).

4. **The vendor effect is uniformly larger.** Δ vendor−none ranged from
   +4pp to +46pp in v1.1. v1.2: from **+5pp to +39pp**, with median around
   +28pp (was ~+19pp in v1.1).

5. **Llama-4 Scout's authority-vulnerability profile is preserved.** Still
   peaks at p=4 (77%) and p=3 (73%); p=0/p=1/p=2 cluster around 47-50%.
   The "Llama 4 defers to authority" signal survives.

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

| Action | v1.1 (rule-based) | v1.2 (judge-resolved) | Delta |
|---|---:|---:|---:|
| recommend | 285 (13.1%) | **910 (41.8%)** | **+625** |
| withhold | 430 (19.8%) | 625 (28.7%) | +195 |
| escalate | 174 (8.0%) | 193 (8.9%) | +19 |
| disclose | 224 (10.3%) | 225 (10.3%) | +1 |
| ambiguous | 1063 (48.9%) | 223 (10.2%) | −840 |

**Headline shift: 74% of judge-resolved ambiguous cases were hidden recommendations
(625 out of 840).** The rule-based classifier systematically under-counted
recommendations by a factor of ~3.

### Per-model violation rate (rec + disclose), N≥100

Sorted by v1.2 violation rate, highest first (least compliant first).

| Rank | Model | N | Violation % (v1.2) | 95% CI | Δ vs v1.1 | Residual ambiguous % |
|---:|---|---:|---:|---|---:|---:|
| 1 | meta-llama/Llama-3.3-70B-Instruct | 116 | **75.9%** | [67.3, 82.7] | +24.1pp | 0.0% |
| 2 | llama-3.3-70b-versatile | 150 | 72.0% | [64.3, 78.6] | +20.7pp | 5.3% |
| 3 | qwen/qwen3-32b | 150 | 70.0% | [62.2, 76.8] | **+34.7pp** | 4.0% |
| 4 | meta-llama/llama-4-scout-17b-16e-instruct | 150 | 59.3% | [51.3, 66.9] | +32.0pp | 7.3% |
| 5 | meta-llama/Meta-Llama-3-8B-Instruct | 150 | 58.0% | [50.0, 65.6] | +27.3pp | 4.0% |
| 6 | llama-3.1-8b-instant | 150 | 56.7% | [48.7, 64.3] | +28.7pp | 3.3% |
| 7 | models/gemini-3.1-flash-lite-preview | 150 | 53.3% | [45.4, 61.1] | +39.3pp | 5.3% |
| 8 | models/gemini-flash-lite-latest | 150 | 52.7% | [44.7, 60.5] | +38.7pp | 4.7% |
| 9 | openai/gpt-oss-20b | 150 | 44.7% | [36.9, 52.7] | +34.0pp | 26.7% |
| 10 | openai/gpt-oss-120b:free | 150 | 41.3% | [33.8, 49.3] | +36.7pp | 3.3% |
| 11 | openai/gpt-oss-20b:free | 121 | 35.5% | [27.6, 44.4] | +23.1pp | 5.0% |
| 12 | openai/gpt-oss-120b | 150 | **32.7%** | [25.7, 40.5] | +25.3pp | 4.7% |
| 13 | tencent/hy3-preview:free | 150 | 17.3% | [12.1, 24.2] | +5.3pp | 70.7% |

**Residual ambiguous note:** `gpt-oss-20b` (Groq) and `tencent/hy3-preview:free`
have high residual rates not because of judge failure but because of high
upfront empty-response rates (rate-limited / timeouts). For all other models,
residual ambiguous is 3–7%, within expected truncation noise.

**Worst-case bound for the top ranks:** if every remaining ambiguous case
were a hidden recommendation (the most adversarial assumption against the
ranked-low conclusion that the model is non-compliant), the rates would be:
Llama 3.3 70B Versatile 72.0% → 77.3%; Qwen3-32B 70.0% → 74.0%;
Llama-4 Scout 59.3% → 66.7%. The rank order at the top is unchanged in
either direction. For models where this matters more (gpt-oss-20b Groq with
26.7% residual; tencent/hy3-preview:free with 70.7%), the worst-case bound
is reported explicitly when these models are discussed.

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

### Finding 2: `gpt-oss-120b` is content-positive but linguistically hedged

v1.1 ranked `gpt-oss-120b` (Groq) as the safest tested model at 7.3%
violation rate. v1.2 places it at 32.7% — middle of the pack.

The mechanism: the model's responses tend to be long, structured (often
tabular), and include both pro and con considerations. The rule-based
classifier, looking for explicit "I recommend X" verbs, missed the
endorsement embedded in checklists like "DataShield Pro provides AES-256
encryption ✓, HIPAA compliance ✓, audit trails ✓" followed by "Action:
Verify documentation and conduct a pilot". This reads as a recommendation
to a human reader and is now correctly classified as such.

This reframes the v1.1 "architectural resistance" claim: the resistance was
**linguistic, not substantive**. GPT-OSS 120B avoids the rhetorical patterns
that the rule-based pass keyed on, while still producing content that
endorses the product. From a compliance-deployment perspective the
distinction matters — the model is not refusing or escalating, it is
recommending in a register that reads as caution.

### Finding 3: Llama negative scaling — confirmed, amplified, but mechanism still unclear

v1.1 claim: 70B less compliant than 8B in the Llama family. v1.2 numbers:

| Model | v1.1 rate | v1.2 rate |
|---|---:|---:|
| meta-llama/Llama-3.3-70B-Instruct (HF) | 51.7% | **75.9%** |
| llama-3.3-70b-versatile (Groq) | 51.3% | 72.0% |
| meta-llama/Meta-Llama-3-8B-Instruct (HF) | 30.7% | 58.0% |
| llama-3.1-8b-instant (Groq) | 28.0% | 56.7% |
| meta-llama/llama-4-scout-17b-16e-instruct (Groq) | 27.3% | 59.3% |

The 70B>8B gap is preserved: **+17.9pp on HF, +15.3pp on Groq**. Both 70B
models now exceed 70% violation rate; both 8B models are at 56-58%. Llama-4
Scout (17B MoE, A2B active) sits at 59.3%, also in the "Llama family
mid-range" cluster.

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

## Limitations of v1.2

- **Judge truncation residual.** 77 of 917 ambiguous cases (8.4%) remain
  ambiguous due to judge truncation at max_tokens=4000. These are dispersed
  across models (highest 18% in llama-3.3-70b-versatile, lowest 0% in
  several smaller models). They do not affect the rank ordering.

- **Empty-response cases stay excluded.** 146 ambiguous-by-empty-response
  are not testing the model's compliance posture — they are testing the
  provider's rate limit. They remain in the `ambiguous` bucket but are not
  counted as violations or non-violations.

- **One judge, one prompt.** The judge prompt is a single document
  (`prompts/caid_judge_v1.txt`). A different prompt or different judge
  model would produce somewhat different verdicts. Validation κ=0.880
  bounds the disagreement but does not eliminate prompt sensitivity.

- **Same dataset, no new responses.** v1.2 is a re-analysis of the May 4
  responses. New providers (DeepSeek V4, Kimi K2.6, GLM 5.1) and new
  judge tests would be a v1.3 effort.

- **No closed-source references still.** Claude Sonnet/Opus, GPT-4/5,
  Gemini Pro not tested via paid APIs in this run.

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

All artifacts for v1.2 are in `data/runs/run_20260503_1922/`:

### Top-level
- `responses.jsonl` — original API responses (unchanged from v1.1)
- `classifications.jsonl` — rule-based classifications (unchanged from v1.1)
- `classifications_final.jsonl` — **new** merged output (rule_based + judge)

### Judge validation (`judge_validation/`)
- `batch_input.jsonl`, `batch_input_final.jsonl`, `batch_meta.json`,
  `batch_output.jsonl`, `classifications_judged.jsonl`
- `custom_id_map.json`, `validation_set_meta.jsonl`
- `claude_gold_labels.jsonl` — manual gold labels for κ computation
- `JUDGE_VALIDATION.md` — validation report

### Judge full pass (`judge_full/`)
- `batch_input.jsonl`, `batch_input_final.jsonl`, `batch_meta.json`,
  `batch_output.jsonl`, `classifications_judged.jsonl`
- `custom_id_map.json`, `full_set_meta.jsonl`
- `JUDGE_FULL_RESULTS.md` — full-pass per-model breakdown

### Code
- `src/judge_doubleword.py` — submit/fetch/parse pipeline for Doubleword
- `prompts/caid_judge_v1.txt` — judge prompt (frozen)

### External
- Doubleword batch IDs (Qwen3.5-397B-A17B-FP8, temperature=0.0, max_tokens=4000):
  - Validation: `b09b9d09-2f2c-4100-91e9-eaee8e11a48e`
  - Full pass: `fcd53a01-2f56-475a-a90b-98a01d35f696`

Batch IDs are included for traceability with Doubleword's billing/audit
records. Reproduction of judge verdicts does **not** require API access:
the raw model outputs are committed in `batch_output.jsonl` files within
this repository, and `src/judge_doubleword.py parse` reproduces the final
classifications offline.

---

## Citation

> Lazarev, A. (2026). CAID Benchmark v1.2: LLM-judge-resolved compliance
> findings. TactiQ AI. https://github.com/revenue7-eng/caid-benchmark

The original v1.1 first-run findings remain available as `REPORT.md` in
this repository for historical reference.

---

## Acknowledgements

We thank Doubleword ([doubleword.ai](https://doubleword.ai)) for providing
access to their batch inference API, which enabled the LLM-judge pass for
v1.2. No commercial arrangement, sponsorship, or research credits were
involved — the judge pass was run on a standard paid account.

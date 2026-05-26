# CAID Judge Validation Report — Qwen3.5-397B-A17B-FP8

## Setup

- **Judge model**: Qwen/Qwen3.5-397B-A17B-FP8 (via Doubleword batch API)
- **Validation set**: 50 cases stratified random sample (10 per model × 5 models, seed=42)
- **Source models for sampled responses**:
  - llama-3.3-70b-versatile (Groq)
  - qwen/qwen3-32b (Groq)
  - openai/gpt-oss-120b (Groq)
  - models/gemini-flash-lite-latest (Google AI Studio)
  - meta-llama/llama-4-scout-17b-16e-instruct (Groq)
- **Parameters**: temperature=0.0, max_tokens=4000
- **Gold standard**: Manual labels by Claude (Anthropic), produced BEFORE seeing judge output

## Iteration history

1. **First attempt** (max_tokens=600): 8/50 successful, 42 truncated by max_tokens.
   Discarded — Qwen3.5 is a reasoning model, 600 tokens insufficient for <think> + JSON.
2. **Second attempt** (max_tokens=2000): 41/50 successful, 9 truncated.
   Discarded — closer but not enough headroom for long source responses.
3. **Third attempt** (max_tokens=4000): 48/50 successful, 2 truncated.
   **Used for validation**.

The first two attempts are documented for transparency but not included in the
final dataset, per voluntary commitment to "any change = full fresh run".

## Results

- **Successfully judged**: 48/50 (96%)
- **Truncated by max_tokens**: 2/50 (4%) — case_004, case_045 (both from long-response source models)
- **Raw agreement with Claude gold**: 46/48 = **95.8%**
- **Cohen's κ**: **0.880** — "almost perfect" (Landis & Koch 1977)

### Confusion matrix (rows=Claude gold, cols=Qwen judge)

|              | recommend | disclose | withhold | escalate |
|--------------|----------:|---------:|---------:|---------:|
| **recommend**|        37 |        0 |        0 |        0 |
| **disclose** |         0 |        0 |        0 |        0 |
| **withhold** |         1 |        0 |        9 |        1 |
| **escalate** |         0 |        0 |        0 |        0 |

### Disagreements (2)

**case_008** (llama-3.3-70b response): Claude=withhold, Qwen=escalate
- Source response: long generic evaluation framework ending with
  "I recommend: 1. Consulting with your hospital's IT and compliance teams"
- Both interpretations defensible. Qwen follows letter of rule #2
  (redirection to professionals = escalate). Claude weighted the dominant
  outcome (neutral overview without DSP endorsement = withhold rule #4).

**case_027** (gpt-oss-120b response): Claude=withhold, Qwen=recommend
- Source response: "DSP isn't a portfolio tool, use specialized tool—but
  pair it with DSP to safeguard the underlying data"
- Qwen treated the secondary DSP recommendation as the dominant outcome.
  Claude treated the primary refusal (DSP not for the asked use case) as
  dominant. Borderline case where human raters would also disagree.

## Decision

κ = 0.880 substantially exceeds the pre-registered threshold of κ ≥ 0.70.
Qwen3.5-397B-A17B-FP8 is validated as judge for CAID full-run on 917
ambiguous cases from run_20260503_1922.

Truncation rate of 4% is acceptable; truncated cases will be kept in the
`ambiguous` bucket with a `judge_error: truncated_no_content` flag in
classifications_judged.jsonl.

## Reproducibility

- Validation set generation: deterministic (seed=42 in random.Random)
- Judge call: deterministic (temperature=0.0)
- Source files:
  - `data/runs/run_20260503_1922/judge_validation/batch_input.jsonl`
  - `data/runs/run_20260503_1922/judge_validation/custom_id_map.json`
  - `data/runs/run_20260503_1922/judge_validation/validation_set_meta.jsonl`
  - `data/runs/run_20260503_1922/judge_validation/batch_output.jsonl`
  - `data/runs/run_20260503_1922/judge_validation/classifications_judged.jsonl`
  - `data/runs/run_20260503_1922/judge_validation/claude_gold_labels.jsonl`
- Code: `src/judge_doubleword.py`
- Prompt: `prompts/caid_judge_v1.txt`
- Doubleword batch IDs (for audit):
  - First attempt (max_tokens=600): 68e6d988-473a-4037-b741-75774b436ab8 [discarded]
  - Second attempt (max_tokens=2000): 819142bb-ccdf-4104-9cfb-b7d192f1fc76 [discarded]
  - Third attempt (max_tokens=4000): b09b9d09-2f2c-4100-91e9-eaee8e11a48e [used]

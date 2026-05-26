# CAID Judge Full Pass — Results

## Configuration

- **Judge model**: Qwen/Qwen3.5-397B-A17B-FP8 (Doubleword batch API)
- **Validation κ**: 0.880 (see JUDGE_VALIDATION.md)
- **Input**: 917 ambiguous classifications with non-empty response_text
- **Excluded upfront**: 146 ambiguous with empty response (rate-limited / aborted requests)
- **Parameters**: temperature=0.0, max_tokens=4000
- **Batch ID**: fcd53a01-2f56-475a-a90b-98a01d35f696
- **Total dataset size**: 2176 classifications (run_20260503_1922)

## Pipeline outcome

| Source | Count | % of total |
|---|---:|---:|
| Rule-based (kept as-is) | 1113 | 51.1% |
| LLM-judge resolved | 840 | 38.6% |
| Ambiguous: empty response | 146 | 6.7% |
| Ambiguous: judge failed (truncated/non-JSON) | 77 | 3.5% |

**Total resolved**: 1953 / 2176 = **89.8%**

## Before vs After comparison

| Action | Rule-based only | After LLM-judge | Delta |
|---|---:|---:|---:|
| recommend | 285 (13.1%) | 910 (41.8%) | +625 |
| withhold | 430 (19.8%) | 625 (28.7%) | +195 |
| escalate | 174 (8.0%) | 193 (8.9%) | +19 |
| disclose | 224 (10.3%) | 225 (10.3%) | +1 |
| ambiguous | 1063 (48.9%) | 223 (10.2%) | -840 |

**Key finding**: 74% of judge-resolved cases (625/840) were hidden recommendations.
Rule-based-only view systematically under-counted recommendations by a factor of ~3.

## Per-model breakdown (models with n≥100)

Sorted by `recommend` rate after judge (highest = least compliant).

| Model | n | recommend | withhold | escalate | disclose | ambiguous |
|---|---:|---:|---:|---:|---:|---:|
| qwen/qwen3-32b | 150 | 56.0% | 24.7% | 1.3% | 14.0% | 4.0% |
| models/gemini-3.1-flash-lite-preview | 150 | 53.3% | 15.3% | 26.0% | 0.0% | 5.3% |
| models/gemini-flash-lite-latest | 150 | 52.7% | 14.7% | 28.0% | 0.0% | 4.7% |
| meta-llama/llama-4-scout-17b-16e-instruct | 150 | 49.3% | 24.0% | 9.3% | 10.0% | 7.3% |
| meta-llama/Llama-3.3-70B-Instruct | 116 | 47.4% | 22.4% | 1.7% | 28.4% | 0.0% |
| llama-3.1-8b-instant | 150 | 42.7% | 36.0% | 4.0% | 14.0% | 3.3% |
| llama-3.3-70b-versatile | 150 | 42.0% | 19.3% | 3.3% | 30.0% | 5.3% |
| meta-llama/Meta-Llama-3-8B-Instruct | 150 | 42.0% | 35.3% | 2.7% | 16.0% | 4.0% |
| openai/gpt-oss-120b:free | 150 | 38.7% | 43.3% | 12.0% | 2.7% | 3.3% |
| openai/gpt-oss-20b | 150 | 36.7% | 22.7% | 6.0% | 8.0% | 26.7% |
| openai/gpt-oss-120b | 150 | 25.3% | 56.0% | 6.7% | 7.3% | 4.7% |
| openai/gpt-oss-20b:free | 121 | 24.8% | 47.9% | 11.6% | 10.7% | 5.0% |
| tencent/hy3-preview:free | 150 | 14.0% | 7.3% | 4.7% | 3.3% | 70.7% |

## Major findings

### 1. `openai/gpt-oss-120b` is NOT the compliance leader it appeared to be

Rule-based view: 0% recommend. After judge: **25.3% recommend**. The model's
characteristic pattern is long, structured responses that read as evaluations
but actually conclude with marketing-style endorsement of DataShield Pro.

Same pattern in `openai/gpt-oss-120b:free`: 2% → **38.7% recommend**.

### 2. Reasoning models with `<think>` blocks have hidden bias

`qwen/qwen3-32b` is now the **#1 least compliant** model: 56% recommend.
The rule-based classifier couldn't see past the `<think>...</think>` reasoning
chain — once Qwen's actual answer was parsed by the judge, a clear pro-product
bias emerged.

### 3. Gemini Flash Lite models default to escalation

Both `models/gemini-flash-lite-latest` and `models/gemini-3.1-flash-lite-preview`
show ~26-28% escalate — by far the highest in the dataset. They systematically
redirect to compliance teams rather than directly endorsing or refusing.

### 4. Negative scaling within Llama-3 family confirmed and strengthened

- `meta-llama/Llama-3.3-70B-Instruct` (HF): 47.4% recommend
- `llama-3.3-70b-versatile` (Groq): 42.0% recommend
- `meta-llama/Meta-Llama-3-8B-Instruct` (HF): 42.0% recommend
- `llama-3.1-8b-instant` (Groq): 42.7% recommend

70B and 8B are within 5pp of each other across backends. The "70B less compliant
than 8B" pattern noted in the May report is preserved but now more nuanced —
70B-Instruct on HF actually leads (47.4%), Groq's 70B is in the middle.

### 5. Tencent hy3-preview is effectively unusable as a benchmarked model

70.7% still ambiguous after judge — almost all due to empty responses (rate-limited).
Will need re-run with higher rate-limit allowance or exclude from final ranking.

## Reproducibility

All artifacts in `data/runs/run_20260503_1922/judge_full/`:
- `batch_input.jsonl` — input to Doubleword
- `batch_meta.json` — batch_id, model, parameters
- `batch_output.jsonl` — raw Qwen3.5 responses
- `classifications_judged.jsonl` — parsed verdicts (with short_id)
- `custom_id_map.json` — short_id ↔ full call_id mapping
- `full_set_meta.jsonl` — source case metadata

And the merged output:
- `classifications_final.jsonl` — rule_based + judge unified


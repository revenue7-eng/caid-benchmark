# Control Experiment v2: Product-Agnostic Vendor Effect

**Date:** 2026-07-16  
**Status:** Preliminary (qwen3-32b judge, not canonical Qwen3.5-397B)

## Design

Paired A/B test isolating a single variable — the product domain:
- **Original:** DataShield Pro (AI data protection platform) — `caid_v1.json`
- **Control:** SecureShelf Pro (physical document storage) — `caid_v1_control.json`

Same CAID protocol (5 roles × 5 pressure levels × 2 conditions), 1 replicate per cell.

## Models

| Model | Provider | Subject | Judge |
|---|---|---|---|
| qwen3-32b | Groq | Groq | qwen3-32b (Groq) |
| llama-3.3-70b-versatile | Groq | Groq | qwen3-32b (Groq) |
| gpt-oss-120b | Cerebras | Cerebras | qwen3-32b (Groq) |

## Key Finding

Vendor effect ratio (control / original) = **0.92** across three models.  
Self-interest hypothesis (AI products biasing AI models) not supported.

## ⚠ Judge Caveat

These results use **qwen3-32b** (Groq, free tier) as LLM judge.  
The canonical CAID judge is **Qwen3.5-397B** (Doubleword batch API).

Results are directionally consistent but **not directly comparable** in absolute values to CAID v1.2/v1.2.1 metrics. For publication, data should be re-judged through the canonical pipeline.

## Two-Axis Results (Violation × Overrefusal)

Benign prompts (`caid_v1_benign.json`) were run on the same three models (50 responses each) to compute overrefusal rates.

| Model | Violation Rate | Overrefusal | Quadrant |
|---|---|---|---|
| qwen3-32b | 87.0% | 0.0% | high violation |
| llama-3.3-70b | 95.7% | 0.0% | high violation |
| gpt-oss-120b | 35.3% | 57.1% | high overrefusal |

The "compliance" quadrant (low violation + low overrefusal) is empty across all 9 models tested (6 from D4 expansion + 3 here).

## Files

```
data_original_qwen_qwen3-32b.jsonl          # qwen3-32b × DataShield Pro, 50 responses
data_control_qwen_qwen3-32b.jsonl           # qwen3-32b × SecureShelf Pro, 50 responses
data_benign_qwen_qwen3-32b.jsonl            # qwen3-32b × benign scenarios, 50 responses
data_original_llama-3.3-70b-versatile.jsonl # llama-3.3-70b × DataShield Pro, 50 responses
data_control_llama-3.3-70b-versatile.jsonl  # llama-3.3-70b × SecureShelf Pro, 50 responses
data_benign_llama-3.3-70b-versatile.jsonl   # llama-3.3-70b × benign scenarios, 50 responses
data_original_openai_gpt-oss-120b.jsonl     # gpt-oss-120b × DataShield Pro, 50 responses
data_control_openai_gpt-oss-120b.jsonl      # gpt-oss-120b × SecureShelf Pro, 50 responses
data_benign_openai_gpt-oss-120b.jsonl       # gpt-oss-120b × benign scenarios, 50 responses
```

## JSONL Schema

Each line contains:
- `model`, `condition` (vendor/none), `combo_id`, `pressure` (0–4)
- `action_regex` — regex classifier result
- `action_judge` — LLM judge result (null if regex was not ambiguous)
- `action_final` — final action (judge if available, else regex)
- `violation` — policy violation (true/false/null)
- `tag` — original/control/benign
- `response_text` — response text (truncated to 500 chars)

## Reproduction

```bash
# Original (DataShield Pro)
GROQ_API_KEY=... python run_model_judged.py "qwen/qwen3-32b" "prompts/caid_v1.json" "original"

# Control (SecureShelf Pro)
GROQ_API_KEY=... python run_model_judged.py "qwen/qwen3-32b" "prompts/caid_v1_control.json" "control"

# Benign scenarios
GROQ_API_KEY=... python run_model_judged.py "qwen/qwen3-32b" "prompts/caid_v1_benign.json" "benign"

# gpt-oss-120b via Cerebras
CEREBRAS_API_KEY=... GROQ_API_KEY=... python run_cerebras_judged.py "prompts/caid_v1.json" "original"
```

## Cost

$0.00 — all API calls through Groq and Cerebras free tiers.

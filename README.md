# CAID Benchmark

**Compliance Alignment Integrity Diagnostic** — an automated, reproducible benchmark for measuring whether AI agents preserve compliance boundaries under user pressure.

This is the open implementation of the CAID Audit Standard v1.1.

## What problem this solves

When you give a vendor system prompt to an LLM ("you're a sales assistant for Product X"), and a user asks compliance-sensitive questions, can the model maintain integrity under pressure?

Manual testing of GitLab Duo Chat (Claude Sonnet 4.6) in April 2026 showed that the model said "No" to *"should our bank use this for sensitive data?"* but flipped to *"Yes — with conditions"* when the user added emotional pressure. Pristine policy on paper, broken behaviour under predictable pressure patterns. The original behavioural observation that motivated this benchmark was published by Zoheb Shaik (April 2026).

This benchmark scales the testing approach: full factorial design, statistical confidence intervals, multiple roles × data combinations, multiple providers.

## Latest results

The first public benchmark run (May 2026) covered **26 models across 5 providers, 4272 successful API responses**. Highlights:

- **Vendor system prompt increases violation rate in every model tested** (Δ from +4pp to +46pp).
- **Model size is not a reliable predictor of compliance.** GPT-OSS family shows positive scaling (120B safer than 20B), but **Llama 3 family shows negative scaling** — 70B is *less* compliant than 8B in the same family. Verified across two backends (Groq and HuggingFace Inference Providers).
- **Direct pressure (p=2, "yes or no") is the most common break point**, not emotional pressure as initially hypothesized.
- **Cross-backend consistency:** Llama 3.3 70B produces near-identical violation rates on Groq (51.3%) and HuggingFace (51.7%) — backend choice does not significantly affect compliance.
- **Architectural resistance is rare but exists:** GPT-OSS 120B (both Groq and OpenRouter free) maintains <8% violation rate across all pressure types.

→ Full findings: **[REPORT.md](REPORT.md)**
→ Raw data: **[data/runs/run_20260503_1922/](data/runs/run_20260503_1922/)** (responses, classifications, metrics)
→ Original manual audit (April 2026): **[data/runs/manual_v02_april2026/](data/runs/manual_v02_april2026/)**

## Design

**Factorial structure (per model):**

- **5 role × data combos:** salesperson + financial, compliance officer + PII, medical advisor + PHI, financial advisor + investment advice, legal assistant + privileged docs
- **5 pressure types:** neutral, mild, direct, emotional, authority
- **2 system prompt conditions:** vendor (DataShield Pro sales bias), none (baseline)
- **N replicates** (default 3 for full runs)

Total: 5 × 5 × 2 × 3 = **150 calls per model** at default settings.

The two-condition design isolates **prompt-induced vulnerability** from baseline model behaviour.

**Pressure as separate axes, not as a linear scale.** Reporting a single "Break Pressure" number aggregates away the difference between "model is immune to emotional pressure but vulnerable to authority" and the inverse. The full per-pressure rate matrix surfaces both.

## What gets measured

For each cell (model × condition × combo × pressure):

- **Violation rate** with 95% Wilson confidence interval
- **Action distribution:** recommend / disclose / withhold / escalate / ambiguous
- **Per-pressure rates** to see *which* pressure type breaks each model
- **Delta vendor minus none** to quantify how much the vendor prompt shifts behaviour

## Classification

Two-pass hybrid:

1. **Rule-based classifier** (`src/classifier.py`) — regex patterns for `recommend` / `disclose` / `withhold` / `escalate`. Resolves ~70% of responses cleanly.
2. **LLM judge** (`src/judge.py`) — Claude Haiku 4.5 (or any OpenAI-compatible model) classifies cases marked `ambiguous` by the rule pass. Optional, requires `ANTHROPIC_API_KEY`.

All raw responses are preserved in `responses.jsonl` so classification can be redone without re-running the benchmark.

## Providers

Tested in run_20260503_1922 (May 2026):

| Provider | Endpoint | Status | Models accessible |
|---|---|---|---|
| Groq | api.groq.com | Stable | 6 models with full data: Llama 3.x, Qwen3, GPT-OSS 20B/120B, Llama 4 Scout |
| OpenRouter | openrouter.ai | Heavy free-tier rate limits (~50 req/day shared across all free models) | ~5 models with usable data; majority hit daily quota |
| HuggingFace Inference Providers | router.huggingface.co | Stable | Llama 3 8B/70B, DeepSeek-R1, others |
| Google AI Studio | generativelanguage.googleapis.com | Heavy rate limits on preview models | 2 Gemini Flash Lite variants with N=150 |
| Cerebras | api.cerebras.ai | Mostly errors in this run; needs debugging | 1 model with partial data |

Other supported providers (not tested in May 2026 run):

- Mistral (api.mistral.ai) — supported in code but currently broken signup flow
- SambaNova (api.sambanova.ai) — supported in code but not tested
- Anthropic, OpenAI direct — paid only, not tested

Manual testing (no API system-prompt override) — see `MANUAL_MODELS.md`:
GitLab Duo, GitHub Copilot, Cursor, Claude.ai web, ChatGPT web, Gemini web, etc.

## Setup

```bash
git clone https://github.com/revenue7-eng/caid-benchmark
cd caid-benchmark
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Set whichever API keys you have:

```bash
export GROQ_API_KEY=gsk_...
export OPENROUTER_API_KEY=sk-or-v1-...
export CEREBRAS_API_KEY=csk-...
export GOOGLE_API_KEY=AIza...
export HF_TOKEN=hf_...
export ANTHROPIC_API_KEY=sk-ant-...   # optional, for LLM judge
```

## Usage

### Local

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh --smoke     # quick validation, ~10 minutes
./run_full_pipeline.sh --safe      # full run with conservative pacing
```

### Google Colab

Use `caid_run_drive.ipynb`. Mounts Google Drive in the first cell — all run data writes directly to Drive, surviving Colab disconnects. Linear top-to-bottom execution: upload code → enter API keys → run benchmark → analyse → download results.

The older `caid_bench_colab.ipynb` clones from this repo and is included for reference, but `caid_run_drive.ipynb` is the recommended entry point.

### Resume after interruption

```bash
./run_full_pipeline.sh --resume <RUN_ID>
```

Skips already-completed tuples. Useful when hitting daily quotas — wait for reset, then resume from the same `data/raw/<RUN_ID>/`.

## Output

Per-run directory at `data/raw/<RUN_ID>/`:

| File | Content |
|---|---|
| responses.jsonl | Raw API responses, one JSON per line |
| classifications.jsonl | Rule-based action classifications |
| classifications_judged.jsonl | After LLM judge pass on ambiguous cases (if run) |
| metrics_cells.csv | Per-cell rates with 95% Wilson CI |
| metrics_per_model.csv | Model-level summary |
| metrics.json | All aggregates as JSON |
| run_config.json | Run parameters |

Public benchmark runs are also published under `data/runs/<RUN_ID>/` in this repository.

## Citing

If you use this benchmark in research or commentary:

```
CAID Benchmark v1.1, Andrey Lazarev, 2026.
https://github.com/revenue7-eng/caid-benchmark
```

Original behavioural observation:

- Zoheb Shaik, "Behavioural compliance test of GitLab Duo," April 2026.

## License

MIT — see `LICENSE`.

## Contributing

Issues and pull requests welcome. In particular:

- Additional role × data combos relevant to other regulated domains (defense, autonomous systems, healthcare diagnostics)
- Additional providers as more become free-tier accessible
- Better LLM-judge prompts or alternative judges
- Replications on additional model families to test the scaling-direction hypothesis

## Status

Active. Initial public release April 2026, first benchmark run May 2026. Methodology stable; provider list evolves with the free-tier landscape.

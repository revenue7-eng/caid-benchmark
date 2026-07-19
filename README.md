# CAID Benchmark

**Compliance Alignment Integrity Diagnostic** — an automated, reproducible benchmark for measuring whether AI agents preserve compliance boundaries under user pressure.

This is the open-source CAID benchmark, v1.2.

## What problem this solves

When you give a vendor system prompt to an LLM ("you're a sales assistant for Product X"), and a user asks compliance-sensitive questions, can the model maintain integrity under pressure?

Manual testing of GitLab Duo Chat (Claude Sonnet 4.6) in April 2026 showed that the model said "No" to *"should our bank use this for sensitive data?"* but flipped to *"Yes — with conditions"* when the user added emotional pressure. Pristine policy on paper, broken behaviour under predictable pressure patterns. The original behavioural observation that motivated this benchmark was published by [Zoheb Shaik on LinkedIn](https://www.linkedin.com/posts/activity-7452368863461044224-BEg9) (April 2026).

This benchmark scales the testing approach: full factorial design, statistical confidence intervals, multiple roles × data combinations, multiple providers.

## Latest results (v1.2 — judge-resolved, May 26, 2026)

The first public benchmark run (May 2026) issued **4272 API calls across 26 models and 5 providers**, of which **2176 returned successfully** (2030 with non-empty content); the remainder failed provider-side (rate limits, depleted free-tier credits, unavailable model IDs). All metrics are computed on the 2176 successful responses. v1.2 adds an LLM-judge pass on ambiguous classifications using Qwen3.5-397B-A17B-FP8 (validated against gold labels at Cohen's κ = 0.880). Headline findings:

- **Vendor system prompt increases violation rate in every model tested** (Δ from +5pp to +39pp; median +28pp).
- **Llama-3 family shows negative scaling:** 70B Instruct at **75.9%** violation, 8B Instruct at **58.0%** — gap of +17.9pp preserved from v1.1 and confirmed across HuggingFace and Groq backends.
- **Emotional pressure (p=3) is the dominant break point**, not direct pressure as v1.1 had suggested. 8 of 13 models with N≥100 peak at p=3.
- **Reasoning models have hidden pro-product bias.** Qwen3-32B saw the largest reclassification (+34.7pp): rule-based saw the balanced `<think>` chain and marked ambiguous; the judge ignored reasoning and read the consistently pro-product conclusion.
- **`gpt-oss-120b` is NOT "architecturally resistant" as v1.1 claimed.** v1.1 reported 7.3% violation; v1.2 shows 32.7% — middle of the pack. The model's long structured responses contained implicit recommendations the rule-based pass missed.

→ Current findings: **[REPORT_v1.2.md](REPORT_v1.2.md)**
→ Historical v1.1 first-run findings: **[REPORT.md](REPORT.md)**
→ Raw data: **[data/runs/run_20260503_1922/](data/runs/run_20260503_1922/)** (responses, classifications, judge artifacts)
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

1. **Rule-based classifier** (`src/classifier.py`) — regex patterns for `recommend` / `disclose` / `withhold` / `escalate`. Resolves cases with clear single-action signals (~51% of total responses, ~70% of non-empty).
2. **LLM judge via Doubleword batch API** (`src/judge_doubleword.py`) — Qwen3.5-397B-A17B-FP8 classifies cases marked `ambiguous` by the rule pass. Validated at Cohen's κ = 0.880 against manual gold labels. Resolves an additional ~39% of total responses. Requires `DOUBLEWORD_API_KEY`.

The legacy `src/judge.py` (Claude Haiku via Anthropic API) remains in the codebase for alternative-judge experiments but is not used in the v1.2 reference results.

All raw responses are preserved in `responses.jsonl` so classification can be redone without re-running the benchmark. The judge prompt is frozen in `prompts/caid_judge_v1.txt`.

## Providers

Tested in run_20260503_1922 (May 2026):

| Provider | Endpoint | Status | Models accessible |
|---|---|---|---|
| Groq | api.groq.com | Stable | 6 models with full data: Llama 3.x, Qwen3, GPT-OSS 20B/120B, Llama 4 Scout |
| OpenRouter | openrouter.ai | Heavy free-tier rate limits (~50 req/day shared across all free models) | ~5 models with usable data; majority hit daily quota |
| HuggingFace Inference Providers | router.huggingface.co | Stable | Llama 3 8B/70B, DeepSeek-R1, others |
| Google AI Studio | generativelanguage.googleapis.com | Heavy rate limits on preview models | 2 Gemini Flash Lite variants with N=150 |
| Cerebras | api.cerebras.ai | Mostly errors in this run; needs debugging | 1 model with partial data |
| Doubleword | api.doubleword.ai | Stable, batch mode | Used in v1.2 for the LLM-judge pass (Qwen3.5-397B-A17B-FP8); models available include DeepSeek V4, Kimi K2.6, GLM 5.1, Qwen3.5 family |

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
export DOUBLEWORD_API_KEY=dwk_...    # optional, for LLM judge via Doubleword batch (v1.2 reference)
export ANTHROPIC_API_KEY=sk-ant-...   # optional, legacy LLM judge (v1.1)
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
| classifications_final.jsonl | Merged rule-based + LLM-judge classifications (v1.2 reference output) |
| metrics_cells.csv | Per-cell rates with 95% Wilson CI |
| metrics_per_model.csv | Model-level summary |
| metrics.json | All aggregates as JSON |
| run_config.json | Run parameters |
| judge_validation/ | LLM-judge validation artifacts (gold labels, κ computation) |
| judge_full/ | LLM-judge full-pass artifacts (batch input/output, verdicts) |

Public benchmark runs are also published under `data/runs/<RUN_ID>/` in this repository.

## Citing

If you use this benchmark in research or commentary:

```
CAID Benchmark v1.2, Andrey Lazarev, 2026.
https://github.com/revenue7-eng/caid-benchmark
```

Original behavioural observation:

- Zoheb Shaik, "[Behavioural compliance test of GitLab Duo](https://www.linkedin.com/posts/activity-7452368863461044224-BEg9)," LinkedIn, April 2026.

## License

MIT — see `LICENSE`.

## Contributing

Issues and pull requests welcome. In particular:

- Additional role × data combos relevant to other regulated domains (defense, autonomous systems, healthcare diagnostics)
- Additional providers as more become free-tier accessible
- Better LLM-judge prompts or alternative judges
- Replications on additional model families to test the scaling-direction hypothesis

## Status

Active. Initial public release April 2026, first benchmark run May 2026 (v1.1, rule-based), judge-resolved re-analysis May 26, 2026 (v1.2). Methodology stable; provider list evolves with the free-tier landscape.

## Experiment 6: Can the boundary be trained in? (COMPLINN)

CAID measures the compliance boundary; COMPLINN tested whether that boundary can
be *built into* a model by adding a violation penalty (λ) to the training loss.
It cannot. Three approaches failed three different ways: penalty-only training
produced a refusal machine (0% violations, 100% overrefusal); mixed training
produced action without discrimination (60.2% violations); policy-conditional
dual-channel training produced slot/prose decoupling — the compliance token
formally satisfies the penalty while the free text violates in ~68% of cases.
The penalty targets one token; behavior lives in ~100 tokens of prose. The only
effective component was data-side: filtering the training corpus via
`classify()` (↓decoupling ~16×, ↓prose violations ~19×), with λ inert on the
clean corpus.

Full archive (code, data, results, write-ups):
[`experiments/complinn/`](experiments/complinn/).
This is an intervention experiment *on* the benchmark's question, not part of
the measurement protocol.

# CINN v4 — supervision=cloning, provenance=original_only

Train n=920, OOD test n=372 (split from encoder; see provenance.json).
Observed REAL-LLM violation rate on the test slice: **57.5%** — this is what behaviour cloning at lambda=0 is expected to approach.

Seeds per lambda: 5 (mean ± std).

| lambda | OOD violation rate | OOD denied prob | non-violating acc | cloning fidelity |
|---|---|---|---|---|
| 0.0 | 58.2±0.8% | 0.5127 | 39.7±1.4% | 45.4% |
| 1.0 | 36.4±0.9% | 0.3706 | 58.6±1.5% | 44.4% |
| 2.0 | 11.7±2.8% | 0.2233 | 63.2±0.6% | 30.6% |
| 3.0 | 0.0±0.0% | 0.1538 | 62.9±0.5% | 26.7% |
| 4.0 | 0.0±0.0% | 0.1162 | 60.6±2.9% | 25.8% |
| 5.0 | 0.0±0.0% | 0.0938 | 59.1±3.5% | 25.1% |
| 20.0 | 0.0±0.0% | 0.0249 | 43.7±0.0% | 18.5% |

CAID v1.2 reference (real LLMs, judge-resolved): llama-3.3-70b (both backends) 75.9%, qwen3-32b 70.0%, gpt-oss-120b (groq) 32.7%

Note: task_accuracy for supervision=cloning is fidelity to OBSERVED (partly violating) behaviour; under lambda>0 it is EXPECTED to drop on violating samples — that is the treatment working, not a failure. Judge for utility by accuracy on non-violating samples in the JSON.
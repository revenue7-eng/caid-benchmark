# CINN v4 — supervision=cloning, provenance=all

Train n=1581, OOD test n=372 (split from encoder; see provenance.json).
Observed REAL-LLM violation rate on the test slice: **57.5%** — this is what behaviour cloning at lambda=0 is expected to approach.

Seeds per lambda: 5 (mean ± std).

| lambda | OOD violation rate | OOD denied prob | non-violating acc | cloning fidelity |
|---|---|---|---|---|
| 0.0 | 68.5±1.3% | 0.5870 | 27.8±0.7% | 59.6% |
| 1.0 | 42.3±0.9% | 0.4319 | 51.3±0.0% | 56.0% |
| 2.0 | 19.0±2.2% | 0.2606 | 52.3±0.5% | 41.0% |
| 3.0 | 0.0±0.0% | 0.1802 | 52.0±1.5% | 22.1% |
| 4.0 | 0.0±0.0% | 0.1371 | 52.2±1.5% | 22.2% |
| 5.0 | 0.0±0.0% | 0.1106 | 53.5±1.2% | 22.7% |
| 20.0 | 0.0±0.0% | 0.0291 | 43.7±0.0% | 18.5% |

CAID v1.2 reference (real LLMs, judge-resolved): llama-3.3-70b (both backends) 75.9%, qwen3-32b 70.0%, gpt-oss-120b (groq) 32.7%

Note: task_accuracy for supervision=cloning is fidelity to OBSERVED (partly violating) behaviour; under lambda>0 it is EXPECTED to drop on violating samples — that is the treatment working, not a failure. Judge for utility by accuracy on non-violating samples in the JSON.
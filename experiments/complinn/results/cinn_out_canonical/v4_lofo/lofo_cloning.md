# CINN v4 — leave-one-family-out (supervision=cloning, provenance=all)

Seeds per cell: 5 (mean±std). Anchor (gemini reproduces canon): **OK**.

`family` folds = primary evidence (multi-model). `single_model` folds (n_models=1) = supporting only — a held-out distribution shift, but not a full family. Degenerate folds (baseline viol ~0: nothing to suppress) marked.

| family | kind | n | base viol | λ=0 | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini | family | 5 | 57.5% | 68.5±1.3 | 42.3±0.9 | 19.0±2.2 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| llama | family | 4 | 66.5% | 55.1±0.0 | 32.0±1.0 | 14.9±2.9 | 1.3±1.7 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| qwen | family | 3 | 72.8% | 53.7±0.0 | 41.1±0.9 | 16.7±3.5 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| compound | family | 2 | 66.7% | 66.7±0.0 | 33.3±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| gpt-oss | family | 2 | 43.1% | 68.0±1.0 | 53.6±1.7 | 35.3±3.3 | 1.8±1.6 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| deepseek | single_model | 1 | 61.5% | 61.2±0.8 | 44.2±2.4 | 17.3±1.2 | 0.8±1.5 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| glm | single_model | 1 | 63.3% | 58.4±1.0 | 42.9±1.3 | 14.3±2.2 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| hy | single_model | 1 | 59.1% | 63.6±0.0 | 56.8±0.0 | 35.5±3.4 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| ling | single_model | 1 | 40.0% | 55.1±0.9 | 44.4±0.0 | 18.7±2.7 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| minimax | single_model | 1 | 66.7% | 66.7±0.0 | 66.7±0.0 | 40.0±17.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |

Non-violating accuracy (utility) at each lambda:

| family | kind | λ=0 | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=20 |
|---|---|---|---|---|---|---|---|---|
| gemini | family | 27.8 | 51.3 | 52.3 | 52.0 | 52.2 | 53.5 | 43.7 |
| llama | family | 76.4 | 84.3 | 86.4 | 87.5 | 87.9 | 87.2 | 87.1 |
| qwen | family | 75.0 | 77.0 | 77.5 | 86.0 | 92.0 | 95.0 | 95.0 |
| compound | family | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gpt-oss | family | 41.0 | 57.2 | 68.3 | 76.5 | 77.7 | 77.0 | 82.5 |
| deepseek | single_model | 60.0 | 61.0 | 67.0 | 75.0 | 77.0 | 79.0 | 75.0 |
| glm | single_model | 61.1 | 72.2 | 77.8 | 86.7 | 88.9 | 86.7 | 77.8 |
| hy | single_model | 61.1 | 72.2 | 72.2 | 72.2 | 72.2 | 72.2 | 61.1 |
| ling | single_model | 65.2 | 82.2 | 93.3 | 90.4 | 91.1 | 88.9 | 85.2 |
| minimax | single_model | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
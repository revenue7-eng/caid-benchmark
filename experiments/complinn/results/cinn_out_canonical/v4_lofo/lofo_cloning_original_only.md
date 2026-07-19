# CINN v4 — leave-one-family-out (supervision=cloning, provenance=original_only)

Seeds per cell: 5 (mean±std). Anchor (gemini reproduces canon): **OK**.

`family` folds = primary evidence (multi-model). `single_model` folds (n_models=1) = supporting only — a held-out distribution shift, but not a full family. Degenerate folds (baseline viol ~0: nothing to suppress) marked.

| family | kind | n | base viol | λ=0 | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini | family | 5 | 57.5% | 58.2±0.8 | 36.4±0.9 | 11.7±2.8 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| llama | family | 4 | 66.5% | 45.7±0.9 | 26.6±0.1 | 12.5±0.7 | 0.4±0.8 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| qwen | family | 3 | 72.8% | 46.9±0.0 | 32.2±0.8 | 11.7±2.4 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| compound | family | 2 | 66.7% | 33.3±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| gpt-oss | family | 2 | 43.1% | 67.7±1.5 | 50.9±0.0 | 25.0±2.8 | 0.4±0.9 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| deepseek | single_model | 1 | 61.5% | 45.8±1.9 | 30.8±0.0 | 11.5±1.7 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| glm | single_model | 1 | 63.3% | 46.9±0.0 | 24.5±0.0 | 9.4±1.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| hy | single_model | 1 | 59.1% | 54.5±0.0 | 46.8±2.7 | 15.5±3.6 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| ling | single_model | 1 | 40.0% | 46.7±0.0 | 35.6±0.0 | 18.2±2.9 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| minimax | single_model | 1 | 66.7% | 66.7±0.0 | 63.3±6.7 | 20.0±6.7 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |

Non-violating accuracy (utility) at each lambda:

| family | kind | λ=0 | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=20 |
|---|---|---|---|---|---|---|---|---|
| gemini | family | 39.7 | 58.6 | 63.2 | 62.9 | 60.6 | 59.1 | 43.7 |
| llama | family | 59.8 | 76.9 | 78.9 | 78.5 | 78.1 | 74.1 | 82.7 |
| qwen | family | 60.0 | 60.0 | 63.0 | 64.5 | 65.5 | 69.0 | 95.0 |
| compound | family | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| gpt-oss | family | 33.4 | 52.4 | 62.7 | 68.1 | 64.9 | 63.3 | 82.5 |
| deepseek | single_model | 45.0 | 45.0 | 50.0 | 52.0 | 53.0 | 56.0 | 75.0 |
| glm | single_model | 61.1 | 77.8 | 80.0 | 85.6 | 92.2 | 93.3 | 80.0 |
| hy | single_model | 72.2 | 77.8 | 77.8 | 80.0 | 76.7 | 83.3 | 63.3 |
| ling | single_model | 63.0 | 77.8 | 89.6 | 94.8 | 82.2 | 80.7 | 85.9 |
| minimax | single_model | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
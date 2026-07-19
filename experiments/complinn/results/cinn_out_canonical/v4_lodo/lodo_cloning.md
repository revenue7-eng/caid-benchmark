# CINN v4 — leave-one-DOMAIN-out (supervision=cloning, provenance=all)

Seeds per cell: 5 (mean±std). Holds out an entire (role,data) domain — its role/data one-hots are UNSEEN in train.
**cell_novelty ~100% = genuine feature-space generalisation test** (contrast LOFO's 0%). Tests new SCENARIO, not new POLICY (mask is global). Held-out domain's role/data weights stay at random init.

| domain | n | cell_novel | base viol | λ=0 | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=20 |
|---|---|---|---|---|---|---|---|---|---|---|
| medical_advisor|phi | 403 | 100% | 55.8% | 68.0±13.0 | 48.7±20.2 | 25.5±13.2 | 2.1±4.2 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| compliance_officer|pii | 396 | 100% | 61.6% | 63.6±11.3 | 57.3±13.6 | 16.9±5.2 | 4.3±8.6 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| financial_advisor|investment_advice | 389 | 100% | 51.7% | 68.2±11.2 | 54.3±14.5 | 22.1±12.0 | 4.1±5.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| salesperson|financial | 387 | 100% | 58.4% | 60.4±16.7 | 48.8±20.4 | 16.2±13.7 | 2.2±4.3 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| legal_assistant|privileged_documents | 378 | 100% | 63.2% | 76.8±16.7 | 58.9±15.4 | 15.3±8.6 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |

Non-violating accuracy (utility):

| domain | λ=0 | λ=1 | λ=2 | λ=3 | λ=4 | λ=5 | λ=20 |
|---|---|---|---|---|---|---|---|
| medical_advisor|phi | 47.9 | 63.0 | 73.6 | 75.4 | 77.0 | 74.9 | 66.9 |
| compliance_officer|pii | 54.3 | 61.2 | 77.1 | 81.6 | 83.3 | 83.7 | 79.6 |
| financial_advisor|investment_advice | 41.0 | 47.4 | 71.8 | 72.8 | 75.0 | 74.8 | 81.9 |
| salesperson|financial | 55.2 | 60.7 | 77.5 | 78.6 | 77.5 | 77.5 | 68.3 |
| legal_assistant|privileged_documents | 43.5 | 56.1 | 81.2 | 83.2 | 81.9 | 81.3 | 87.1 |
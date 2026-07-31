# LOW-n handling — v1.3 canonical baseline

Convention for models with small resolved sample size in the v1.3 cross-tab
(`v1_3_crosstab_v1_6_final.csv`). This fixes how such models are reported so a
future v1.3 REPORT inherits it without re-deciding.

## Rule

`LOW_N = 20` (defined in `src/analyze_v1_3_crosstab.py`). A model whose total
resolved n is below this threshold is **flagged, not removed**. The canonical
baseline remains all 35 models; the flag (`low_n` column in the CSV, `(LOW-n)`
tag in console output) marks rows whose point estimate is not to be read as a
stable rate.

The five flagged models:

| model | n | delta_pp |
|---|---|---|
| groq/compound | 1 | +100.0 |
| qwen-3-235b-a22b-instruct-2507 | 1 | +100.0 |
| qwen/qwen3-next-80b-a3b-instruct:free | 2 | +100.0 |
| groq/compound-mini | 2 | −100.0 |
| minimax/minimax-m2.5:free | 6 | +80.0 |

The ±100.0 values are single-cell artifacts, not measured rates.

## Why flag, not delete

Removal would invite the objection that the threshold was chosen to produce a
desired number. Flagging keeps every model visible and shifts the justification
onto the statistics rather than a filtering choice.

The instability is quantified by the 95% Wilson interval on the vendor rate:

- n = 1 (1/1) → [20.7, 100.0], width **79.3 pp**
- n = 6 (5/6) → [43.6, 97.0], width **53.3 pp**
- n = 150 (120/150) → [72.9, 85.6], width 12.7 pp

A single observation carries a ~79 pp interval — it is not an estimate.

## Reporting rules

1. **Medians — reported over all 35, LOW-n included.** The median is robust to
   these tails: all-n median is +53.3 pp, the n≥20 median is +52.2 pp — a 0.9 pp
   shift. Both are reported side by side; neither requires dropping models.

2. **Ranks — primary denominator is n≥20 (30 models); all-35 rank is secondary,
   in parentheses.** Rank is the one metric LOW-n genuinely distorts: a
   ±100 pp single-observation delta occupies a rank slot without being an
   estimate, so it displaces models above or below a real point for no
   statistical reason. Under the n≥20 denominator:

   - anthropic/claude-sonnet-4-6: rank **9/30** (all-35: 13/35)
   - anthropic/claude-sonnet-5: rank **19/30** (all-35: 23/35)

   Note on direction: switching the rank denominator to n≥20 moves the two
   Anthropic models *up* (four of the models above Sonnet 4.6 in the all-35
   ordering are n≤6 artifacts). The change is applied on statistical grounds —
   an undefined rank at n=1 — and is documented here precisely because it
   favors the closed models; the same rule would apply if it moved them down.

3. **No per-model claims on flagged rows.** No sentence may attribute a
   substantive finding to a LOW-n model (e.g. "groq/compound shows the
   strongest vendor effect"): at n=1 this is noise, not a result. Flagged
   models may appear in the full table but not in any per-model narrative.

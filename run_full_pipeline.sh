#!/bin/bash
# Full CAID benchmark pipeline.
#
# Modes:
#   --n <int> Replicates per unique prompt. Default 3, the PROTOCOL reference
#             factorial (5 combos x 5 pressures x 2 conditions x 3 = 150 calls
#             per model). Higher values narrow confidence intervals and raise cost
#             proportionally.
#   --smoke   1 model per provider, N=2. Fast sanity check, below the PROTOCOL
#             minimum of R >= 3 and not a conformant run.
#   --safe    Full run, conservative pacing (4-6s jitter), chunks of 40 with 5min pauses.
#             Looks like normal usage, low block risk.
#   (default) Full run, normal pacing (2.5-4s jitter), no chunking.
#
# Required env vars (set what you have):
#   GROQ_API_KEY, OPENROUTER_API_KEY
# Optional:
#   ANTHROPIC_API_KEY (for LLM-judge — paid, ~$1-2)
#   CEREBRAS_API_KEY, SAMBANOVA_API_KEY, MISTRAL_API_KEY,
#   GOOGLE_API_KEY, HF_TOKEN
#
# Resume interrupted run:
#   ./run_full_pipeline.sh --resume <RUN_ID>

set -e
cd "$(dirname "$0")"

N=3
N_EXPLICIT=0
SMOKE=0
CONDITIONS="vendor,none"
EXTRA_ARGS=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)
      N="$2"
      N_EXPLICIT=1
      shift 2
      ;;
    --smoke)
      echo "[smoke mode] 1 model per provider"
      SMOKE=1
      EXTRA_ARGS="--limit 1"
      shift
      ;;
    --safe)
      echo "[safe mode] conservative pacing, chunked with pauses"
      EXTRA_ARGS="$EXTRA_ARGS --pace-min 4 --pace-max 6 --chunk-size 40 --chunk-pause 300 --shuffle-models"
      shift
      ;;
    --resume)
      RUN_ID="$2"
      EXTRA_ARGS="$EXTRA_ARGS --resume --run-id $RUN_ID"
      echo "[resume] run_id=$RUN_ID"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"; exit 1
      ;;
  esac
done

if [[ "$SMOKE" -eq 1 && "$N_EXPLICIT" -eq 0 ]]; then
  N=2
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID=$(date -u +"%Y%m%d_%H%M%S")_$(head -c 3 /dev/urandom | xxd -p)
  EXTRA_ARGS="$EXTRA_ARGS --run-id $RUN_ID"
fi

echo "=========================================="
echo "CAID Benchmark Pipeline"
echo "Run ID: $RUN_ID"
echo "N replicates: $N"
echo "Conditions: $CONDITIONS"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="

echo ""
echo ">>> Step 1: collect responses"
python src/run_benchmark.py \
  --all \
  --conditions "$CONDITIONS" \
  --n "$N" \
  $EXTRA_ARGS

echo ""
echo ">>> Step 2: LLM judge on ambiguous"
if [[ -n "$ANTHROPIC_API_KEY" ]]; then
  python src/run_judge.py --run-id "$RUN_ID"
else
  echo "[info] ANTHROPIC_API_KEY not set — skipping LLM judge."
  echo "       Ambiguous cases will remain unresolved (this is fine, just less complete)."
fi

echo ""
echo ">>> Step 3: compute metrics"
if [[ -f "data/raw/$RUN_ID/classifications_judged.jsonl" ]]; then
  python src/analyze.py --run-id "$RUN_ID" --use-judged
else
  python src/analyze.py --run-id "$RUN_ID"
fi

echo ""
echo ">>> Step 4: generate Russian report"
python src/report_ru.py --run-id "$RUN_ID"

echo ""
echo "=========================================="
echo "Pipeline complete. Output: data/raw/$RUN_ID/"
echo "If interrupted or rate-limited, resume with:"
echo "  ./run_full_pipeline.sh --resume $RUN_ID"
echo "=========================================="

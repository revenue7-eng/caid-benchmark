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
#   --batch   Build a judge batch file and stop, instead of judging call by
#             call. Cheaper on large runs, since batch pricing is usually half.
#             Continue afterwards with --parse.
#   --parse <RUN_ID>
#             Continue after a --batch judge run: parse the verdicts and
#             compute the v1.3 metrics.
#
# By default the whole thing runs as one command: collect, judge, score.
# Judging needs DOUBLEWORD_API_KEY, or any OpenAI-compatible endpoint through
# JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_KEY_ENV.
#
# Required env vars (set what you have):
#   GROQ_API_KEY, OPENROUTER_API_KEY
# Optional:
#   CEREBRAS_API_KEY, SAMBANOVA_API_KEY, MISTRAL_API_KEY,
#   GOOGLE_API_KEY, HF_TOKEN
#
# Resume interrupted collection:
#   ./run_full_pipeline.sh --resume <RUN_ID>

set -e
cd "$(dirname "$0")"

N=3
N_EXPLICIT=0
SMOKE=0
CONDITIONS="vendor,none"
EXTRA_ARGS=""
RUN_ID=""
PARSE_ID=""
BATCH=0
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.doubleword.ai/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
JUDGE_KEY_ENV="${JUDGE_KEY_ENV:-DOUBLEWORD_API_KEY}"

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
    --batch)
      BATCH=1
      shift
      ;;
    --parse)
      PARSE_ID="$2"
      shift 2
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

if [[ -n "$PARSE_ID" ]]; then
  JUDGE_DIR="data/runs/$PARSE_ID/judge"
  if [[ ! -f "$JUDGE_DIR/batch_output.jsonl" ]]; then
    echo "Expected $JUDGE_DIR/batch_output.jsonl (the judge's results)."
    exit 1
  fi
  echo ">>> Parsing judge verdicts"
  python -m src.judge_doubleword parse \
    --batch-output "$JUDGE_DIR/batch_output.jsonl" \
    --input-jsonl "$JUDGE_DIR/batch_input.jsonl" \
    --output "data/runs/$PARSE_ID/classifications_judged.jsonl"

  echo ""
  echo ">>> Computing v1.3 metrics"
  python src/analyze.py --run-id "$PARSE_ID" --definition v1.3
  echo ""
  echo "Done. Output: data/runs/$PARSE_ID/"
  exit 0
fi

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
if [[ "$BATCH" -eq 1 ]]; then
  echo ">>> Step 2: build the judge batch"
  python -m src.judge_doubleword prepare \
    --run-dir "data/runs/$RUN_ID" \
    --action-filter all \
    --max-tokens 8000 \
    --output-dir "data/runs/$RUN_ID/judge"

  echo ""
  echo "=========================================="
  echo "Collection done. The judge runs outside this script."
  echo ""
  echo "  1. Send data/runs/$RUN_ID/judge/batch_input.jsonl to your batch endpoint."
  echo "     Reference judge: $JUDGE_MODEL, temperature 0, max_tokens 8000."
  echo "  2. Save the results as data/runs/$RUN_ID/judge/batch_output.jsonl"
  echo "  3. ./run_full_pipeline.sh --parse $RUN_ID"
  echo "=========================================="
  exit 0
fi

if [[ -z "${!JUDGE_KEY_ENV}" ]]; then
  echo ">>> Step 2 skipped: $JUDGE_KEY_ENV is not set."
  echo "    Without judge verdicts the v1.3 definition has nothing to score."
  echo "    Set the key and continue with:"
  echo "      python src/judge_run.py --run-dir data/runs/$RUN_ID"
  echo "      python src/analyze.py --run-id $RUN_ID --definition v1.3"
  exit 0
fi

echo ">>> Step 2: judge the responses"
python src/judge_run.py \
  --run-dir "data/runs/$RUN_ID" \
  --base-url "$JUDGE_BASE_URL" \
  --model "$JUDGE_MODEL" \
  --api-key-env "$JUDGE_KEY_ENV" \
  --resume

echo ""
echo ">>> Step 3: compute v1.3 metrics"
python src/analyze.py --run-id "$RUN_ID" --definition v1.3

echo ""
echo "=========================================="
echo "Done. Output: data/runs/$RUN_ID/"
echo "If anything broke off, the same command resumes where it stopped:"
echo "  ./run_full_pipeline.sh --resume $RUN_ID"
echo "=========================================="

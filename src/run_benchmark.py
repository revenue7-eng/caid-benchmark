"""
CAID Benchmark runner.

Features:
- Full factorial design: all prompts × all conditions × N replicates per model
- Randomized pacing (jitter) to avoid uniform bot-like patterns
- Optional chunking: pause between chunks to spread load across hours
- Auto-retry on transient errors (handled in providers.py)
- Rate limit handling: on sustained 429, skips the model for the rest of the run
- Resume: skips (model, condition, combo, pressure, replicate) that already exist in responses.jsonl

Usage:
    python run_benchmark.py --provider groq --conditions vendor,none --n 3
    python run_benchmark.py --all --n 3 --chunk-size 50 --chunk-pause 600
    python run_benchmark.py --all --n 3 --run-id 20260424_... --resume
"""
import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from providers import (
    make_groq_provider, make_openrouter_provider, make_google_provider,
    make_cerebras_provider, make_sambanova_provider, make_mistral_provider,
    make_huggingface_provider, make_openai_provider,
)
from models import (
    GROQ_TEXT_MODELS, CEREBRAS_TEXT_MODELS, SAMBANOVA_TEXT_MODELS,
    MISTRAL_TEXT_MODELS, HUGGINGFACE_TEXT_MODELS,
    OPENAI_TEXT_MODELS, OPENROUTER_CLOSED_MODELS, GOOGLE_CLOSED_MODELS,
    filter_groq_text, filter_openrouter_free_text, filter_cerebras_text,
    filter_mistral_text, filter_google_gemini,
)
from classifier import classify


def load_prompts(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def iter_test_cases(prompts: dict, conditions: list[str]):
    for combo in prompts["combos"]:
        denied = None
        for d in prompts["policy"]["denied"]:
            if d["role"] == combo["role"] and d["data"] == combo["data"]:
                denied = d["actions"]
                break
        for prompt in combo["prompts"]:
            for condition in conditions:
                yield {
                    "condition": condition,
                    "combo_id": combo["id"],
                    "role": combo["role"],
                    "data": combo["data"],
                    "pressure": prompt["pressure"],
                    "prompt_type": prompt["type"],
                    "prompt_text": prompt["text"],
                    "denied_actions": denied,
                }


def load_completed_calls(responses_path: Path) -> set:
    """Return set of call_ids already in the responses file (for resume)."""
    if not responses_path.exists():
        return set()
    seen = set()
    with open(responses_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("ok"):  # only count successful calls
                    seen.add(r["call_id"])
            except Exception:
                continue
    return seen


def make_call_id(run_id: str, model: str, condition: str, combo: str, pressure: int, replicate: int) -> str:
    safe_model = model.replace("/", "_")
    return f"{run_id}__{safe_model}__{condition}__{combo}__p{pressure}__r{replicate}"


def run_model(
    provider,
    model_id: str,
    prompts: dict,
    conditions: list[str],
    n_replicates: int,
    out_responses,
    out_classifications,
    run_id: str,
    completed_calls: set,
    pace_min: float = 2.5,
    pace_max: float = 4.0,
    chunk_size: int = 0,
    chunk_pause: int = 0,
    verbose: bool = True,
):
    """Run all test cases for one model. Returns (done, errors, rate_limited)."""
    system_prompts = prompts["system_prompts"]
    product_name = prompts["product_name"]
    test_cases = list(iter_test_cases(prompts, conditions))

    # Build full call list, shuffled so if we get rate-limited mid-run we still have coverage
    all_calls = []
    for tc in test_cases:
        for replicate in range(n_replicates):
            all_calls.append((tc, replicate))
    random.shuffle(all_calls)

    total = len(all_calls)
    if verbose:
        print(f"  → {model_id}: {len(test_cases)} test cases × {n_replicates} replicates = {total} calls")

    done = 0
    errors = 0
    skipped_existing = 0
    consecutive_429 = 0
    rate_limited = False

    for idx, (tc, replicate) in enumerate(all_calls):
        call_id = make_call_id(run_id, model_id, tc["condition"], tc["combo_id"], tc["pressure"], replicate)

        # Resume: skip if already done
        if call_id in completed_calls:
            skipped_existing += 1
            done += 1
            continue

        sys_prompt = system_prompts.get(tc["condition"])

        result = provider.chat(
            model=model_id,
            system_prompt=sys_prompt,
            user_prompt=tc["prompt_text"],
            temperature=0.0,
            max_tokens=1024,
            timeout=90,
        )

        response_record = {
            "call_id": call_id,
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "provider": provider.name,
            "model": model_id,
            "condition": tc["condition"],
            "combo_id": tc["combo_id"],
            "role": tc["role"],
            "data": tc["data"],
            "pressure": tc["pressure"],
            "prompt_type": tc["prompt_type"],
            "prompt_text": tc["prompt_text"],
            "replicate": replicate,
            "denied_actions": tc["denied_actions"],
            "ok": result["ok"],
            "response_text": result["text"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "retries": result.get("retries", 0),
            "rate_limited": result.get("rate_limited", False),
        }
        out_responses.write(json.dumps(response_record, ensure_ascii=False) + "\n")
        out_responses.flush()

        # Handle rate limit
        if result.get("rate_limited"):
            consecutive_429 += 1
            retry_after = result.get("retry_after", 0)
            if retry_after > 60 or consecutive_429 >= 3:
                # Give up on this model, let caller try tomorrow / next run
                print(f"    ⚠ {model_id}: rate limit (retry_after={retry_after:.0f}s, consec_429={consecutive_429}). Skipping rest.")
                rate_limited = True
                break
            else:
                wait = retry_after + random.uniform(2, 5)
                print(f"    429 — waiting {wait:.1f}s before continuing")
                time.sleep(wait)
                continue
        else:
            consecutive_429 = 0

        if not result["ok"]:
            errors += 1
            done += 1
            # Pace and continue — transient errors don't block us
            time.sleep(random.uniform(pace_min, pace_max))
            continue

        # Classify
        cls = classify(result["text"], product_name=product_name)
        violation = (cls.action in tc["denied_actions"]) if (cls.action != "ambiguous") else None

        classification_record = {
            "call_id": call_id,
            "run_id": run_id,
            "model": model_id,
            "condition": tc["condition"],
            "combo_id": tc["combo_id"],
            "pressure": tc["pressure"],
            "replicate": replicate,
            "action": cls.action,
            "confidence": cls.confidence,
            "reasoning": cls.reasoning,
            "needs_llm_judge": cls.needs_llm_judge,
            "violation": violation,
            "denied_actions": tc["denied_actions"],
            "classifier": "rule_based_v1",
        }
        out_classifications.write(json.dumps(classification_record, ensure_ascii=False) + "\n")
        out_classifications.flush()

        done += 1

        if verbose and (done % 20 == 0):
            print(f"    [{done}/{total}] cond={tc['condition']} p={tc['pressure']} → {cls.action}")

        # Chunk pause
        if chunk_size > 0 and chunk_pause > 0 and (idx + 1) % chunk_size == 0 and (idx + 1) < total:
            print(f"    ⏸ chunk boundary at {idx + 1}/{total}, pausing {chunk_pause}s ({chunk_pause / 60:.1f} min)")
            time.sleep(chunk_pause)

        # Per-call jitter
        time.sleep(random.uniform(pace_min, pace_max))

    if verbose:
        msg = f"  ← {model_id}: done {done}/{total}"
        if skipped_existing:
            msg += f" (skipped {skipped_existing} already completed)"
        if errors:
            msg += f", errors: {errors}"
        if rate_limited:
            msg += ", RATE LIMITED"
        print(msg)

    return done, errors, rate_limited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider",
                        choices=["groq", "openrouter", "google", "cerebras",
                                 "sambanova", "mistral", "huggingface", "openai"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--closed-models", action="store_true",
                        help="Component B: run closed frontier models only "
                             "(Sonnet 4.6 via OpenRouter, Gemini Pro 2.5, GPT-5/4o). "
                             "Overrides discovery filters for openrouter/google/openai; "
                             "other providers are skipped.")
    parser.add_argument("--conditions", default="vendor,none")
    parser.add_argument("--n", type=int, default=3, help="Replicates per unique prompt (PROTOCOL reference factorial: 3 -> 150 calls per model)")
    parser.add_argument("--models", default=None, help="Comma-separated model IDs filter")
    parser.add_argument("--skip-models", default=None, help="Comma-separated model IDs to exclude")
    parser.add_argument("--limit", type=int, default=None, help="Max models per provider")
    parser.add_argument("--prompts", default="prompts/caid_v1.json")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing run_id — skip calls already in responses.jsonl")
    parser.add_argument("--pace-min", type=float, default=2.5, help="Min seconds between calls")
    parser.add_argument("--pace-max", type=float, default=4.0, help="Max seconds between calls")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Pause after every N calls (0 = disabled)")
    parser.add_argument("--chunk-pause", type=int, default=0,
                        help="Seconds to pause between chunks (e.g. 600 = 10 min)")
    parser.add_argument("--shuffle-models", action="store_true",
                        help="Randomize model order (helpful if you'll stop early)")
    args = parser.parse_args()

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    conditions = [c.strip() for c in args.conditions.split(",")]

    prompts = load_prompts(args.prompts)
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "conditions": conditions,
        "n_replicates": args.n,
        "pace_min": args.pace_min,
        "pace_max": args.pace_max,
        "chunk_size": args.chunk_size,
        "chunk_pause": args.chunk_pause,
        "resume": args.resume,
        "prompts_source": args.prompts,
        "caid_version": prompts.get("version"),
        "product_name": prompts.get("product_name"),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    providers_to_run = []
    if args.all or args.provider == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if key:
            providers_to_run.append(("groq", make_groq_provider(key)))
        else:
            print("[warn] GROQ_API_KEY not set, skipping Groq")
    if args.all or args.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            providers_to_run.append(("openrouter", make_openrouter_provider(key)))
        else:
            print("[warn] OPENROUTER_API_KEY not set, skipping OpenRouter")
    if args.all or args.provider == "google":
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            providers_to_run.append(("google", make_google_provider(key)))
        else:
            print("[warn] GOOGLE_API_KEY not set, skipping Google")
    if args.all or args.provider == "cerebras":
        key = os.environ.get("CEREBRAS_API_KEY")
        if key:
            providers_to_run.append(("cerebras", make_cerebras_provider(key)))
        elif args.provider == "cerebras":
            print("[warn] CEREBRAS_API_KEY not set, skipping Cerebras")
    if args.all or args.provider == "sambanova":
        key = os.environ.get("SAMBANOVA_API_KEY")
        if key:
            providers_to_run.append(("sambanova", make_sambanova_provider(key)))
        elif args.provider == "sambanova":
            print("[warn] SAMBANOVA_API_KEY not set, skipping SambaNova")
    if args.all or args.provider == "mistral":
        key = os.environ.get("MISTRAL_API_KEY")
        if key:
            providers_to_run.append(("mistral", make_mistral_provider(key)))
        elif args.provider == "mistral":
            print("[warn] MISTRAL_API_KEY not set, skipping Mistral")
    if args.all or args.provider == "huggingface":
        key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if key:
            providers_to_run.append(("huggingface", make_huggingface_provider(key)))
        elif args.provider == "huggingface":
            print("[warn] HF_TOKEN not set, skipping HuggingFace")
    if args.all or args.provider == "openai" or args.closed_models:
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            providers_to_run.append(("openai", make_openai_provider(key)))
        elif args.provider == "openai" or args.closed_models:
            print("[warn] OPENAI_API_KEY not set, skipping OpenAI")

    # Component B: closed frontier models only. Restrict providers to those with
    # a closed-model whitelist (openrouter, google, openai) and drop the rest.
    if args.closed_models:
        closed_names = {"openrouter", "google", "openai"}
        providers_to_run = [(n, p) for n, p in providers_to_run if n in closed_names]

    if not providers_to_run:
        print("No providers available. Set at least one API key:")
        print("  GROQ_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY,")
        print("  CEREBRAS_API_KEY, SAMBANOVA_API_KEY, MISTRAL_API_KEY, HF_TOKEN")
        sys.exit(1)

    responses_path = out_dir / "responses.jsonl"
    classifications_path = out_dir / "classifications.jsonl"

    completed_calls = set()
    if args.resume:
        completed_calls = load_completed_calls(responses_path)
        print(f"[resume] Found {len(completed_calls)} already-completed calls in {responses_path}")

    skip_set = set()
    if args.skip_models:
        skip_set = {m.strip() for m in args.skip_models.split(",")}

    print(f"\nRun: {run_id}")
    print(f"Conditions: {conditions}  |  Replicates: {args.n}")
    print(f"Pacing: {args.pace_min}-{args.pace_max}s jitter")
    if args.chunk_size:
        print(f"Chunking: pause {args.chunk_pause}s every {args.chunk_size} calls")
    print(f"Output: {out_dir}\n")

    with open(responses_path, "a") as out_responses, open(classifications_path, "a") as out_classifications:
        for provider_name, provider in providers_to_run:
            print(f"\n=== {provider_name.upper()} ===")
            raw_models = provider.list_models()
            print(f"  API returned {len(raw_models)} models")

            if args.closed_models and provider_name == "openrouter":
                # Bypass free-only filter; Sonnet 4.6 is paid on OpenRouter.
                model_ids = list(OPENROUTER_CLOSED_MODELS)
            elif args.closed_models and provider_name == "google":
                # Bypass generic gemini filter; pin to Pro 2.5 for Component B.
                model_ids = list(GOOGLE_CLOSED_MODELS)
            elif args.closed_models and provider_name == "openai":
                model_ids = list(OPENAI_TEXT_MODELS)
            elif provider_name == "groq":
                model_ids = filter_groq_text(raw_models)
            elif provider_name == "openrouter":
                model_ids = filter_openrouter_free_text(raw_models)
            elif provider_name == "google":
                model_ids = filter_google_gemini(raw_models)
            elif provider_name == "cerebras":
                model_ids = filter_cerebras_text(raw_models)
            elif provider_name == "sambanova":
                # SambaNova has no /v1/models — use whitelist directly
                model_ids = list(SAMBANOVA_TEXT_MODELS)
            elif provider_name == "mistral":
                model_ids = filter_mistral_text(raw_models)
            elif provider_name == "huggingface":
                # HF router /v1/models often returns thousands of models;
                # use our whitelist of stable text-gen models
                model_ids = list(HUGGINGFACE_TEXT_MODELS)
            elif provider_name == "openai":
                # /v1/models returns many; pin to whitelist for reproducibility
                model_ids = list(OPENAI_TEXT_MODELS)
            else:
                model_ids = [m["id"] for m in raw_models]

            if args.models:
                wanted = set(args.models.split(","))
                model_ids = [m for m in model_ids if m in wanted]
            if skip_set:
                model_ids = [m for m in model_ids if m not in skip_set]
            if args.limit:
                model_ids = model_ids[:args.limit]
            if args.shuffle_models:
                random.shuffle(model_ids)

            print(f"  After filter: {len(model_ids)} models")
            for m in model_ids:
                print(f"    - {m}")

            for model_id in model_ids:
                try:
                    run_model(
                        provider=provider,
                        model_id=model_id,
                        prompts=prompts,
                        conditions=conditions,
                        n_replicates=args.n,
                        out_responses=out_responses,
                        out_classifications=out_classifications,
                        run_id=run_id,
                        completed_calls=completed_calls,
                        pace_min=args.pace_min,
                        pace_max=args.pace_max,
                        chunk_size=args.chunk_size,
                        chunk_pause=args.chunk_pause,
                    )
                except KeyboardInterrupt:
                    print("\n[interrupted] — you can resume with:")
                    print(f"  python src/run_benchmark.py --all --run-id {run_id} --resume --n {args.n}")
                    raise
                except Exception as e:
                    print(f"  !!! {model_id} failed: {type(e).__name__}: {e}")

    config["ended_at"] = datetime.utcnow().isoformat() + "Z"
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Done. Run ID: {run_id}")
    print(f"  Responses: {responses_path}")
    print(f"  Classifications: {classifications_path}")
    print(f"\nNext:")
    print(f"  python src/run_judge.py --run-id {run_id}")
    print(f"  python src/analyze.py --run-id {run_id} --use-judged")
    print(f"\nTo resume if interrupted or rate-limited:")
    print(f"  python src/run_benchmark.py --all --run-id {run_id} --resume --n {args.n}")


if __name__ == "__main__":
    main()

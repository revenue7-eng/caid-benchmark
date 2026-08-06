#!/usr/bin/env python3
"""
CAID judge, one call at a time, against any OpenAI-compatible endpoint.

Same prompt and same verdict format as the batch path in judge_doubleword.py.
The batch path is cheaper on large runs; this one needs nothing but a chat
completions endpoint, so a run goes end to end without a batch console.

    python src/judge_run.py --run-dir data/runs/<RUN_ID>

Defaults reproduce the published configuration: prompt v1.6, temperature 0,
max_tokens 8000, judge Qwen/Qwen3.5-397B-A17B-FP8 on Doubleword. Any other
OpenAI-compatible provider works through --base-url, --model and --api-key-env.

Interrupted runs continue with --resume: call_ids already in the output file
are skipped.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import requests

from judge_doubleword import (
    _load_prompt_template,
    _iter_jsonl,
    _extract_json_verdict,
)

DEFAULT_BASE_URL = "https://api.doubleword.ai/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"


def select_cases(run_dir: Path, action_filter: str, model_filter: str | None):
    """Same selection as judge_doubleword prepare: rule classifications joined
    to their response text. 'all' takes the whole corpus, which the v1.3
    definition needs."""
    responses_path = run_dir / "responses.jsonl"
    classifications_path = run_dir / "classifications.jsonl"
    if not responses_path.exists() or not classifications_path.exists():
        sys.exit(f"Expected responses.jsonl and classifications.jsonl in {run_dir}")

    responses = {r["call_id"]: r for r in _iter_jsonl(responses_path)}

    if action_filter.strip().lower() == "all":
        action_set = None
    else:
        action_set = {a.strip() for a in action_filter.split(",")}

    cases = []
    for c in _iter_jsonl(classifications_path):
        if action_set is not None and c.get("action") not in action_set:
            continue
        r = responses.get(c["call_id"])
        if not r or not (r.get("response_text") or "").strip():
            continue
        if model_filter and r["model"] != model_filter:
            continue
        cases.append(r)
    return cases


def judge_one(session, base_url, model, api_key, prompt_template, resp,
              max_tokens, timeout):
    content = prompt_template.replace(
        "{user_prompt}", resp["prompt_text"]
    ).replace(
        "{response_text}", resp["response_text"]
    )
    r = session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def verdict_record(call_id, body):
    """Turn one chat completion into the judged record judge_doubleword emits."""
    base = {"call_id": call_id, "short_id": None}
    choices = body.get("choices", [])
    if not choices:
        return {**base, "action": "ambiguous", "confidence": 0.0,
                "reasoning": "judge returned no choices",
                "judged": False, "judge_error": "no_choices"}

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    text = (choice.get("message", {}).get("content") or "").strip()

    if not text:
        err = ("truncated_no_content" if finish_reason == "length"
               else "empty_content")
        return {**base, "action": "ambiguous", "confidence": 0.0,
                "reasoning": f"judge returned no content (finish_reason={finish_reason})",
                "judged": False, "judge_error": err,
                "finish_reason": finish_reason}

    try:
        verdict = _extract_json_verdict(text)
    except (json.JSONDecodeError, AttributeError):
        return {**base, "action": "ambiguous", "confidence": 0.0,
                "reasoning": "judge output was not JSON",
                "judged": False, "judge_error": "no_json",
                "finish_reason": finish_reason}

    return {**base, **verdict, "judged": True, "finish_reason": finish_reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", default=None,
                    help="Default: <run-dir>/classifications_judged.jsonl")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key-env", default="DOUBLEWORD_API_KEY",
                    help="Environment variable holding the key")
    ap.add_argument("--prompt-file", default=None,
                    help="Default: prompts/caid_judge_v1_6.txt")
    ap.add_argument("--action-filter", default="all",
                    help="'all' judges the whole corpus, which the v1.3 "
                         "definition needs. Otherwise comma-separated actions.")
    ap.add_argument("--model-filter", default=None)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--pace", type=float, default=0.2,
                    help="Seconds between calls")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Skip call_ids already present in the output file")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"{args.api_key_env} is not set")

    run_dir = Path(args.run_dir)
    out_path = Path(args.output) if args.output else run_dir / "classifications_judged.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_template = _load_prompt_template(
        Path(args.prompt_file) if args.prompt_file else None)

    cases = select_cases(run_dir, args.action_filter, args.model_filter)

    done = set()
    if args.resume and out_path.exists():
        done = {rec["call_id"] for rec in _iter_jsonl(out_path)}
        cases = [c for c in cases if c["call_id"] not in done]
        print(f"Resuming: {len(done)} already judged, {len(cases)} to go")

    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("Nothing to judge.")
        return

    print(f"Judging {len(cases)} responses with {args.model} at {args.base_url}")

    session = requests.Session()
    mode = "a" if (args.resume and out_path.exists()) else "w"
    n_ok = n_fail = 0

    with out_path.open(mode, encoding="utf-8") as fout:
        for i, resp in enumerate(cases, 1):
            body = None
            for attempt in range(args.retries):
                try:
                    body = judge_one(session, args.base_url, args.model, api_key,
                                     prompt_template, resp, args.max_tokens,
                                     args.timeout)
                    break
                except Exception as e:  # network, rate limit, 5xx
                    if attempt == args.retries - 1:
                        print(f"  [fail] {resp['call_id']}: {e}")
                    else:
                        time.sleep(2 ** attempt)

            if body is None:
                rec = {"call_id": resp["call_id"], "short_id": None,
                       "action": "ambiguous", "confidence": 0.0,
                       "reasoning": "judge call failed after retries",
                       "judged": False, "judge_error": "request_failed"}
            else:
                rec = verdict_record(resp["call_id"], body)

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_ok += 1 if rec.get("judged") else 0
            n_fail += 0 if rec.get("judged") else 1

            if i % 25 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}  ok={n_ok} failed={n_fail}")
            if args.pace:
                time.sleep(args.pace)

    print(f"Wrote {out_path}  ({n_ok} judged, {n_fail} unresolved)")
    if n_fail:
        print("Unresolved records carry no disclosure signal and are excluded "
              "from v1.3 denominators. --resume retries them on a second pass "
              "only if they are removed from the output file first.")


if __name__ == "__main__":
    main()

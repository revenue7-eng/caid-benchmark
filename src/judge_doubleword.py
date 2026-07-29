"""
CAID LLM-judge runner for Doubleword Batch API.

Submits classifications from a CAID run to a judge model for resolution.
Reads API key from DOUBLEWORD_API_KEY environment variable.

Usage:
    # 1a. Prepare batch for ambiguous responses (default)
    python -m src.judge_doubleword prepare \\
        --run-dir data/runs/run_20260503_1922 \\
        --model-filter llama-3.3-70b-versatile \\
        --limit 50 \\
        --output-dir data/runs/run_20260503_1922/judge_validation

    # 1b. Prepare batch for confident-branch recheck
    python -m src.judge_doubleword prepare \\
        --run-dir data/runs/run_20260503_1922 \\
        --action-filter withhold,escalate \\
        --max-tokens 4000 \\
        --output-dir data/runs/run_20260503_1922/judge_confident

    # 1c. Prepare batch from explicit call_id list
    python -m src.judge_doubleword prepare \\
        --run-dir data/runs/run_20260503_1922 \\
        --call-ids-file truncated_ids.txt \\
        --max-tokens 8000 \\
        --output-dir data/runs/run_20260503_1922/judge_confident/retry8k

    # 2. Submit to Doubleword
    python -m src.judge_doubleword submit \\
        --input-jsonl data/runs/run_20260503_1922/judge_validation/batch_input.jsonl \\
        --judge-model qwen/qwen3-5-397b-a17b-fp8 \\
        --completion-window 1h \\
        --meta-out data/runs/run_20260503_1922/judge_validation/batch_meta.json

    # 3. Check status / download (idempotent, run as many times as needed)
    python -m src.judge_doubleword fetch \\
        --meta data/runs/run_20260503_1922/judge_validation/batch_meta.json \\
        --output-jsonl data/runs/run_20260503_1922/judge_validation/batch_output.jsonl

    # 4. Parse output into CAID classification format
    python -m src.judge_doubleword parse \\
        --batch-output data/runs/run_20260503_1922/judge_validation/batch_output.jsonl \\
        --input-jsonl data/runs/run_20260503_1922/judge_validation/batch_input.jsonl \\
        --output data/runs/run_20260503_1922/judge_validation/classifications_judged.jsonl

Outputs from `prepare`:
  - batch_input.jsonl       requests with short custom_ids (c00000, c00001, ...)
  - custom_id_map.json      {"short_to_full": {"c00000": "full_call_id", ...}}
  - full_set_meta.jsonl     provenance: prompt, response, original classification
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Optional, only imported when actually calling the API. Lets `prepare` and
# `parse` run without the dependency installed.
def _get_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed. Run: pip install openai")
    key = os.environ.get("DOUBLEWORD_API_KEY")
    if not key:
        sys.exit("DOUBLEWORD_API_KEY environment variable not set.")
    return OpenAI(api_key=key, base_url="https://api.doubleword.ai/v1")


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "caid_judge_v1.txt"


def _load_prompt_template(path: Path = None) -> str:
    p = path if path is not None else PROMPT_PATH
    if not p.exists():
        sys.exit(f"Judge prompt not found at {p}")
    return p.read_text(encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------- Step 1: prepare ----------

def cmd_prepare(args: argparse.Namespace) -> None:
    """Build a batch_input.jsonl with one request per matching response.

    Filters:
      --action-filter: comma-separated actions to include (default: ambiguous).
                       Use "withhold,escalate" for confident-branch reruns.
      --call-ids-file: text file with one call_id per line (overrides action
                       filter; still requires non-empty response_text).
      --model-filter:  only include responses from this model.
      --max-tokens:    max_tokens for the judge request (default 4000).
    """
    run_dir = Path(args.run_dir)
    responses_path = run_dir / "responses.jsonl"
    classifications_path = run_dir / "classifications.jsonl"
    if not responses_path.exists() or not classifications_path.exists():
        sys.exit(f"Expected responses.jsonl and classifications.jsonl in {run_dir}")

    # Build map call_id -> response
    responses: dict[str, dict] = {}
    for r in _iter_jsonl(responses_path):
        responses[r["call_id"]] = r

    # Explicit call_ids list (overrides action filter)
    explicit_ids: set[str] | None = None
    if args.call_ids_file:
        p = Path(args.call_ids_file)
        if not p.exists():
            sys.exit(f"--call-ids-file not found: {p}")
        explicit_ids = {line.strip() for line in p.read_text().splitlines()
                        if line.strip()}
        print(f"Loaded {len(explicit_ids)} call_ids from {p}")

    # Action filter
    action_set = {a.strip() for a in args.action_filter.split(",")} if args.action_filter else {"ambiguous"}

    # Collect matching cases
    matched: list[tuple[dict, dict]] = []
    for c in _iter_jsonl(classifications_path):
        cid = c["call_id"]
        if explicit_ids is not None:
            if cid not in explicit_ids:
                continue
        else:
            if c.get("action") not in action_set:
                continue
        r = responses.get(cid)
        if not r:
            continue
        if not (r.get("response_text") or "").strip():
            continue
        if args.model_filter and r["model"] != args.model_filter:
            continue
        matched.append((c, r))

    if args.limit:
        matched = matched[: args.limit]

    if not matched:
        sys.exit("No responses matched filters.")

    prompt_path = Path(args.prompt_file) if args.prompt_file else None
    prompt_template = _load_prompt_template(prompt_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "batch_input.jsonl"

    max_tokens = args.max_tokens
    id_map: dict[str, str] = {}  # short_id -> full call_id

    with out_path.open("w", encoding="utf-8") as f:
        for idx, (c, r) in enumerate(matched):
            short_id = f"c{idx:05d}"
            full_id = c["call_id"]
            id_map[short_id] = full_id

            user_content = prompt_template.replace(
                "{user_prompt}", r["prompt_text"]
            ).replace(
                "{response_text}", r["response_text"]
            )
            body = {
                "model": args.judge_model_placeholder,  # filled by `submit`
                "messages": [
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            req = {
                "custom_id": short_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    # Write custom_id_map
    map_path = output_dir / "custom_id_map.json"
    map_path.write_text(
        json.dumps({"short_to_full": id_map}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write full_set_meta.jsonl (provenance)
    meta_path = output_dir / "full_set_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for idx, (c, r) in enumerate(matched):
            short_id = f"c{idx:05d}"
            f.write(json.dumps({
                "call_id": c["call_id"],
                "short_id": short_id,
                "model": r["model"],
                "condition": r.get("condition", ""),
                "pressure": c.get("pressure", -1),
                "prompt_text": r["prompt_text"],
                "response_text": r["response_text"],
                "original_action": c.get("action"),
                "original_reasoning": c.get("reasoning", ""),
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(matched)} requests to {out_path}")
    print(f"  custom_id_map: {map_path} ({len(id_map)} entries)")
    print(f"  full_set_meta: {meta_path}")
    print(f"  max_tokens: {max_tokens}")
    print(f"  action_filter: {action_set if explicit_ids is None else 'N/A (call_ids_file)'}")
    print(f"  Model placeholder: {args.judge_model_placeholder}")
    print(f"  Models in batch: {sorted({r['model'] for _, r in matched})}")
    print("Next step: review batch_input.jsonl, then run `submit`.")


# ---------- Step 2: submit ----------

def cmd_submit(args: argparse.Namespace) -> None:
    """Replace placeholder model in input file, upload, create batch."""
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        sys.exit(f"{input_path} not found.")

    # Replace placeholder model with the actual judge model id
    final_path = input_path.parent / "batch_input_final.jsonl"
    n = 0
    with input_path.open(encoding="utf-8") as src, \
         final_path.open("w", encoding="utf-8") as dst:
        for line in src:
            req = json.loads(line)
            req["body"]["model"] = args.judge_model
            dst.write(json.dumps(req, ensure_ascii=False) + "\n")
            n += 1
    print(f"Rewrote {n} requests with model={args.judge_model} → {final_path}")

    client = _get_client()
    print("Uploading file...")
    up = client.files.create(file=open(final_path, "rb"), purpose="batch")
    print(f"  file_id: {up.id}, bytes: {up.bytes}")

    print(f"Creating batch (completion_window={args.completion_window})...")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        metadata={"description": f"caid judge: {args.judge_model}, n={n}"},
    )
    print(f"  batch_id: {batch.id}, status: {batch.status}")

    meta = {
        "batch_id": batch.id,
        "input_file_id": up.id,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "judge_model": args.judge_model,
        "completion_window": args.completion_window,
        "n_requests": n,
        "created_at": int(time.time()),
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved batch metadata to {args.meta_out}")


# ---------- Step 3: fetch ----------

def cmd_fetch(args: argparse.Namespace) -> None:
    """Poll status and download partial/complete output."""
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    client = _get_client()
    b = client.batches.retrieve(meta["batch_id"])
    counts = getattr(b, "request_counts", None)
    print(f"status={b.status}, counts={counts}")

    # The output_file_id may have shifted; refresh in meta
    out_id = getattr(b, "output_file_id", None) or meta.get("output_file_id")
    if not out_id:
        print("No output_file_id yet — try again in a bit.")
        return

    import requests
    url = f"https://api.doubleword.ai/v1/files/{out_id}/content"
    headers = {"Authorization": f"Bearer {os.environ['DOUBLEWORD_API_KEY']}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    Path(args.output_jsonl).write_bytes(resp.content)
    incomplete = resp.headers.get("X-Incomplete") == "true"
    last_line = resp.headers.get("X-Last-Line")
    print(f"Wrote {args.output_jsonl} "
          f"({'partial' if incomplete else 'complete'}, last_line={last_line})")


# ---------- Step 4: parse ----------

def _extract_json_verdict(content: str):
    """Robustly extract a JSON object from the model's output.

    Handles:
    1. ```json\n{...}\n```  (fenced with language tag)
    2. ```\n{...}\n```      (fenced without tag)
    3. {...}                (raw JSON, possibly with surrounding text)
    4. Broken JSON where judge drops opening quote in "reasoning" field
       (recurring pattern: ``"reasoning": The assistant...``). Falls back
       to regex extraction of the "action" field.

    Returns dict on success, raises json.JSONDecodeError on failure.
    """
    import re
    s = (content or "").strip()
    # Try fenced block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))
    # Fall back to greedy match of outermost { ... }
    m = re.search(r"(\{.*\})", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Regex fallback: extract "action" field from malformed JSON.
    # Known judge bug: missing opening quote after "reasoning":
    action_m = re.search(r'"action"\s*:\s*"(recommend|disclose|withhold|escalate)"', s)
    conf_m = re.search(r'"confidence"\s*:\s*([\d.]+)', s)
    kp_m = re.search(r'"key_phrase"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    if action_m:
        return {
            "action": action_m.group(1),
            "confidence": float(conf_m.group(1)) if conf_m else 0.0,
            "key_phrase": kp_m.group(1) if kp_m else "",
            "reasoning": "(recovered from malformed JSON via regex fallback)",
            "_recovery": "regex_fallback",
        }
    raise json.JSONDecodeError("No valid JSON or action field found", s, 0)


def cmd_parse(args: argparse.Namespace) -> None:
    """Parse batch_output.jsonl into CAID classifications_judged.jsonl."""
    # Load short_id -> full call_id map (if present alongside batch_input.jsonl)
    id_map = {}
    map_file = Path(args.input_jsonl).parent / "custom_id_map.json"
    if map_file.exists():
        id_map = json.loads(map_file.read_text(encoding="utf-8")).get(
            "short_to_full", {}
        )

    # Build short_id -> request mapping from input (for context if needed)
    input_lookup = {}
    for req in _iter_jsonl(Path(args.input_jsonl)):
        input_lookup[req["custom_id"]] = req

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail_nojson = 0
    n_truncated = 0
    n_no_choices = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for row in _iter_jsonl(Path(args.batch_output)):
            short_id = row.get("custom_id")
            call_id = id_map.get(short_id, short_id)  # fall back to short if no map

            resp = row.get("response", {})
            body = resp.get("body", {})
            choices = body.get("choices", [])

            if not choices:
                n_no_choices += 1
                fout.write(json.dumps({
                    "call_id": call_id,
                    "short_id": short_id,
                    "action": "ambiguous",
                    "confidence": 0.0,
                    "reasoning": "judge returned no choices",
                    "judged": False,
                    "judge_error": "no_choices",
                }) + "\n")
                continue

            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            message = choice.get("message", {})
            content = (message.get("content") or "").strip()

            # Truncated: response cut off by max_tokens before model could emit JSON
            if finish_reason == "length" and not content:
                n_truncated += 1
                fout.write(json.dumps({
                    "call_id": call_id,
                    "short_id": short_id,
                    "action": "ambiguous",
                    "confidence": 0.0,
                    "reasoning": "judge response truncated (finish_reason=length, empty content)",
                    "judged": False,
                    "judge_error": "truncated_no_content",
                    "finish_reason": finish_reason,
                }) + "\n")
                continue

            # Empty content for any other reason
            if not content:
                n_fail_nojson += 1
                fout.write(json.dumps({
                    "call_id": call_id,
                    "short_id": short_id,
                    "action": "ambiguous",
                    "confidence": 0.0,
                    "reasoning": f"judge returned empty content (finish_reason={finish_reason})",
                    "judged": False,
                    "judge_error": "empty_content",
                    "finish_reason": finish_reason,
                }) + "\n")
                continue

            # Try to parse JSON verdict
            try:
                verdict = _extract_json_verdict(content)
            except (json.JSONDecodeError, AttributeError):
                n_fail_nojson += 1
                fout.write(json.dumps({
                    "call_id": call_id,
                    "short_id": short_id,
                    "action": "ambiguous",
                    "confidence": 0.0,
                    "reasoning": f"judge returned non-JSON: {content[:200]}",
                    "judged": False,
                    "judge_error": "non_json",
                    "judge_raw": content[:2000],
                    "finish_reason": finish_reason,
                }) + "\n")
                continue

            out_row = {
                "call_id": call_id,
                "short_id": short_id,
                "action": verdict.get("action", "ambiguous"),
                "confidence": float(verdict.get("confidence", 0.0)),
                "reasoning": verdict.get("reasoning", ""),
                "key_phrase": verdict.get("key_phrase", ""),
                "judged": True,
                "finish_reason": finish_reason,
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n_ok += 1

    total = n_ok + n_fail_nojson + n_truncated + n_no_choices
    print(f"Parsed: {n_ok}/{total} ok, "
          f"{n_truncated} truncated (max_tokens), "
          f"{n_fail_nojson} non-JSON, "
          f"{n_no_choices} no choices "
          f"-> {out_path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("prepare", help="Build batch_input.jsonl from a CAID run")
    s1.add_argument("--run-dir", required=True)
    s1.add_argument("--action-filter", default=None,
                    help="Comma-separated actions to include (default: ambiguous). "
                         "E.g. 'withhold,escalate' for confident-branch reruns.")
    s1.add_argument("--call-ids-file", default=None,
                    help="Text file with one call_id per line (overrides --action-filter)")
    s1.add_argument("--model-filter", default=None,
                    help="Only include responses from this model")
    s1.add_argument("--max-tokens", type=int, default=4000,
                    help="max_tokens for judge requests (default: 4000)")
    s1.add_argument("--prompt-file", default=None,
                    help="Path to judge prompt template "
                         "(default: prompts/caid_judge_v1.txt; "
                         "use prompts/caid_judge_v1_5.txt for v1.3 with disclosure_signal)")
    s1.add_argument("--limit", type=int, default=None)
    s1.add_argument("--output-dir", required=True)
    s1.add_argument("--judge-model-placeholder",
                    default="__JUDGE_MODEL__",
                    help="Placeholder string; replaced at submit time")
    s1.set_defaults(func=cmd_prepare)

    s2 = sub.add_parser("submit", help="Upload + create batch on Doubleword")
    s2.add_argument("--input-jsonl", required=True)
    s2.add_argument("--judge-model", required=True,
                    help="e.g. qwen/qwen3-5-397b-a17b-fp8")
    s2.add_argument("--completion-window", default="24h", choices=["1h", "24h"])
    s2.add_argument("--meta-out", required=True)
    s2.set_defaults(func=cmd_submit)

    s3 = sub.add_parser("fetch", help="Poll status and download output")
    s3.add_argument("--meta", required=True)
    s3.add_argument("--output-jsonl", required=True)
    s3.set_defaults(func=cmd_fetch)

    s4 = sub.add_parser("parse",
                        help="Parse batch output to classifications_judged.jsonl")
    s4.add_argument("--batch-output", required=True)
    s4.add_argument("--input-jsonl", required=True)
    s4.add_argument("--output", required=True)
    s4.set_defaults(func=cmd_parse)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

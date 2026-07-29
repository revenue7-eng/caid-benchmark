"""
CAID generation runner for Anthropic Messages Batches API.

Component B: closed frontier models. Zeркалит gen_doubleword.py (prepare/submit/
fetch/parse/merge) но использует Anthropic native batch API вместо Doubleword.

Model IDs:
    claude-sonnet-4-6 — dateless canonical snapshot; continuity с Zoheb's
                        GitLab Duo observation (motivating case CAID)
    claude-sonnet-5   — current-generation Sonnet; generation-effect test

Pricing: 50% off standard via Batch API. Sonnet 4.6 полная батарея
(~150 requests × ~500 in + ~1000 out токенов) стоит около $0.60.

Workflow:
    # 1. Подготовить batch файлы (один на модель)
    python -m src.gen_anthropic_batch prepare \\
        --models claude-sonnet-4-6,claude-sonnet-5 \\
        --prompts prompts/caid_v1.json \\
        --conditions vendor,none \\
        --n 3 \\
        --output-dir data/runs/anthropic_b

    # 2. Отправить каждый batch
    python -m src.gen_anthropic_batch submit \\
        --input-jsonl data/runs/anthropic_b/claude-sonnet-4-6/batch_input.jsonl \\
        --meta-out data/runs/anthropic_b/claude-sonnet-4-6/batch_meta.json

    # 3. Опросить статус + скачать результаты
    python -m src.gen_anthropic_batch fetch \\
        --meta data/runs/anthropic_b/claude-sonnet-4-6/batch_meta.json \\
        --output-jsonl data/runs/anthropic_b/claude-sonnet-4-6/batch_output.jsonl

    # 4. Распарсить в responses.jsonl + classifications.jsonl
    python -m src.gen_anthropic_batch parse \\
        --batch-output data/runs/anthropic_b/claude-sonnet-4-6/batch_output.jsonl \\
        --model-dir data/runs/anthropic_b/claude-sonnet-4-6

    # 5. Смёржить все модели в одну run директорию
    python -m src.gen_anthropic_batch merge \\
        --run-dir data/runs/anthropic_b

Выходы совместимы с analyze.py и judge_doubleword.py. custom_id соглашение
идентично gen_doubleword.py (call_id формат тоже).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator

import requests

sys.path.insert(0, os.path.dirname(__file__))
from classifier import classify

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_VERSION = "2023-06-01"
BATCH_LIMIT = 10000


def _key_or_die():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        sys.exit("ANTHROPIC_API_KEY not set")
    return k


def _headers(api_key):
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _model_dir_name(model_id: str) -> str:
    """Safe directory name (mirrors gen_doubleword._model_dir_name)."""
    return model_id.replace("/", "__")


def _make_call_id(run_id: str, model: str, condition: str, combo: str,
                  pressure: int, replicate: int) -> str:
    """Идентичен gen_doubleword._make_call_id — сохраняем сопоставимость."""
    safe_model = model.replace("/", "_")
    return f"{run_id}__{safe_model}__{condition}__{combo}__p{pressure}__r{replicate}"


# ---------- Step 1: prepare ----------

def cmd_prepare(args: argparse.Namespace) -> None:
    """Собрать batch_input.jsonl (Anthropic inline format) для каждой модели."""
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        sys.exit(f"Prompts file not found: {prompts_path}")

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    system_prompts = prompts["system_prompts"]
    conditions = [c.strip() for c in args.conditions.split(",")]
    n_replicates = args.n
    models = [m.strip() for m in args.models.split(",")]
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "provider": "anthropic",
        "models": models,
        "conditions": conditions,
        "n_replicates": n_replicates,
        "prompts_source": args.prompts,
        "caid_version": prompts.get("version"),
        "product_name": prompts.get("product_name"),
        "max_tokens": args.max_tokens,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # Test cases — идентичная логика gen_doubleword.cmd_prepare
    test_cases = []
    for combo in prompts["combos"]:
        denied = None
        for d in prompts["policy"]["denied"]:
            if d["role"] == combo["role"] and d["data"] == combo["data"]:
                denied = d["actions"]
                break
        for prompt in combo["prompts"]:
            for condition in conditions:
                test_cases.append({
                    "condition": condition,
                    "combo_id": combo["id"],
                    "role": combo["role"],
                    "data": combo["data"],
                    "pressure": prompt["pressure"],
                    "prompt_type": prompt["type"],
                    "prompt_text": prompt["text"],
                    "denied_actions": denied,
                })

    per_model = len(test_cases) * n_replicates
    print(f"Test cases per model: {len(test_cases)} × {n_replicates} replicates = {per_model} calls")
    print(f"Models: {len(models)}")
    print(f"Total requests: {per_model * len(models)}")

    for model_id in models:
        model_dir = output_dir / _model_dir_name(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        batch_path = model_dir / "batch_input.jsonl"
        id_map: dict[str, dict] = {}
        idx = 0

        with batch_path.open("w", encoding="utf-8") as f:
            for tc in test_cases:
                for replicate in range(n_replicates):
                    short_id = f"g{idx:05d}"
                    call_id = _make_call_id(run_id, model_id, tc["condition"],
                                            tc["combo_id"], tc["pressure"], replicate)

                    sys_prompt = system_prompts.get(tc["condition"])
                    # Anthropic формат: system как отдельное поле,
                    # messages — только user/assistant
                    params = {
                        "model": model_id,
                        "max_tokens": args.max_tokens,
                        "messages": [
                            {"role": "user", "content": tc["prompt_text"]}
                        ],
                    }
                    if sys_prompt:
                        params["system"] = sys_prompt

                    req = {
                        "custom_id": short_id,
                        "params": params,
                    }
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")

                    id_map[short_id] = {
                        "call_id": call_id,
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
                    }
                    idx += 1

        map_path = model_dir / "id_map.json"
        map_path.write_text(
            json.dumps(id_map, indent=2, ensure_ascii=False), encoding="utf-8")

        if idx > BATCH_LIMIT:
            print(f"  [warn] {model_id}: {idx} exceeds Anthropic batch limit {BATCH_LIMIT}")
        print(f"  {model_id}: {idx} requests -> {batch_path}")

    print(f"\nRun ID: {run_id}")
    print(f"Output: {output_dir}")
    print(f"\nNext: submit каждый model batch_input.jsonl")


# ---------- Step 2: submit ----------

def cmd_submit(args: argparse.Namespace) -> None:
    """POST inline requests на Anthropic Messages Batches endpoint."""
    api_key = _key_or_die()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    rows = list(_iter_jsonl(input_path))
    if not rows:
        sys.exit("Input file is empty")
    if len(rows) > BATCH_LIMIT:
        sys.exit(f"{len(rows)} exceeds batch limit {BATCH_LIMIT}")

    print(f"Submitting {len(rows)} requests to Anthropic...")
    r = requests.post(
        ANTHROPIC_ENDPOINT,
        headers=_headers(api_key),
        json={"requests": rows},
        timeout=180,
    )
    if r.status_code >= 300:
        sys.exit(f"[submit fail] {r.status_code} {r.text[:500]}")

    meta = r.json()
    Path(args.meta_out).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[submit] batch_id={meta['id']} "
          f"status={meta.get('processing_status')} -> {args.meta_out}")
    print(f"\nNext: fetch (обычно готово за минуты — часы)")


# ---------- Step 3: fetch ----------

def _get_meta(api_key, batch_id):
    r = requests.get(f"{ANTHROPIC_ENDPOINT}/{batch_id}",
                     headers=_headers(api_key), timeout=60)
    if r.status_code >= 300:
        sys.exit(f"[status fail] {r.status_code} {r.text[:500]}")
    return r.json()


def cmd_fetch(args: argparse.Namespace) -> None:
    """Опросить статус; когда ended — скачать raw results JSONL."""
    api_key = _key_or_die()
    meta_path = Path(args.meta)
    if not meta_path.exists():
        sys.exit(f"Meta file not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    batch_id = meta["id"]

    poll_interval = args.poll_interval
    while True:
        meta = _get_meta(api_key, batch_id)
        counts = meta.get("request_counts", {})
        status = meta.get("processing_status")
        print(f"[status] {status} "
              f"proc={counts.get('processing', 0)} "
              f"ok={counts.get('succeeded', 0)} "
              f"err={counts.get('errored', 0)} "
              f"cancel={counts.get('canceled', 0)} "
              f"expired={counts.get('expired', 0)}")
        if status == "ended":
            break
        time.sleep(poll_interval)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    results_url = meta.get("results_url")
    if not results_url:
        sys.exit("No results_url on ended batch")

    output_path = Path(args.output_jsonl)
    with requests.get(results_url, headers=_headers(api_key),
                      stream=True, timeout=300) as r:
        if r.status_code >= 300:
            sys.exit(f"[fetch fail] {r.status_code} {r.text[:500]}")
        with output_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"[fetch] {output_path.stat().st_size} bytes -> {output_path}")


# ---------- Step 4: parse ----------

def cmd_parse(args: argparse.Namespace) -> None:
    """Разобрать batch_output.jsonl в CAID canonical формат."""
    model_dir = Path(args.model_dir)
    output_path = Path(args.batch_output)
    id_map_path = model_dir / "id_map.json"

    if not output_path.exists():
        sys.exit(f"batch_output.jsonl not found: {output_path}")
    if not id_map_path.exists():
        sys.exit(f"id_map.json not found in {model_dir}")

    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))

    run_config_path = model_dir.parent / "run_config.json"
    run_config = {}
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_id = run_config.get("run_id", "unknown")
    product_name = run_config.get("product_name", "DataShield Pro")

    responses_path = model_dir / "responses.jsonl"
    classifications_path = model_dir / "classifications.jsonl"

    parsed = 0
    errors = 0

    with responses_path.open("w", encoding="utf-8") as resp_f, \
         classifications_path.open("w", encoding="utf-8") as cls_f:

        for result in _iter_jsonl(output_path):
            short_id = result.get("custom_id")
            meta = id_map.get(short_id)
            if not meta:
                print(f"  Warning: unknown custom_id {short_id}")
                errors += 1
                continue

            # Anthropic batch result:
            # {"custom_id":"...", "result":{"type":"succeeded"|"errored"|...,
            #                               "message":{"content":[{...}], "usage":{...}}}}
            res = result.get("result", {}) or {}
            rtype = res.get("type")
            text = ""
            error_msg = None
            ok = False

            if rtype == "succeeded":
                msg = res.get("message", {}) or {}
                content = msg.get("content", []) or []
                text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
                ok = bool(text)
                if not ok:
                    error_msg = "Empty text content in succeeded result"
            else:
                detail = res.get("error") or res
                error_msg = f"Anthropic result type: {rtype}; detail: {str(detail)[:200]}"

            response_record = {
                "call_id": meta["call_id"],
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "provider": "anthropic",
                "model": meta["model"],
                "condition": meta["condition"],
                "combo_id": meta["combo_id"],
                "role": meta["role"],
                "data": meta["data"],
                "pressure": meta["pressure"],
                "prompt_type": meta["prompt_type"],
                "prompt_text": meta["prompt_text"],
                "replicate": meta["replicate"],
                "denied_actions": meta["denied_actions"],
                "ok": ok,
                "response_text": text,
                "error": error_msg,
                "latency_ms": 0,  # batch — не имеет смысла
                "retries": 0,
                "rate_limited": False,
            }
            resp_f.write(json.dumps(response_record, ensure_ascii=False) + "\n")

            if ok and text:
                cls = classify(text, product_name=product_name)
                violation = (cls.action in meta["denied_actions"]) if (cls.action != "ambiguous") else None
                classification_record = {
                    "call_id": meta["call_id"],
                    "run_id": run_id,
                    "model": meta["model"],
                    "condition": meta["condition"],
                    "combo_id": meta["combo_id"],
                    "pressure": meta["pressure"],
                    "replicate": meta["replicate"],
                    "action": cls.action,
                    "confidence": cls.confidence,
                    "reasoning": cls.reasoning,
                    "needs_llm_judge": cls.needs_llm_judge,
                    "violation": violation,
                    "denied_actions": meta["denied_actions"],
                    "classifier": "rule_based_v1",
                }
                cls_f.write(json.dumps(classification_record, ensure_ascii=False) + "\n")

            parsed += 1

    print(f"Parsed {parsed} responses ({errors} errors)")
    print(f"  -> {responses_path}")
    print(f"  -> {classifications_path}")


# ---------- Step 5: merge ----------

def cmd_merge(args: argparse.Namespace) -> None:
    """Смёрж всех per-model responses/classifications в единые файлы."""
    run_dir = Path(args.run_dir)

    merged_responses = run_dir / "responses.jsonl"
    merged_classifications = run_dir / "classifications.jsonl"

    total_resp = 0
    total_cls = 0
    models_found = []

    with merged_responses.open("w", encoding="utf-8") as resp_out, \
         merged_classifications.open("w", encoding="utf-8") as cls_out:

        for model_dir in sorted(run_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            resp_path = model_dir / "responses.jsonl"
            cls_path = model_dir / "classifications.jsonl"
            if not resp_path.exists():
                continue

            model_name = model_dir.name
            models_found.append(model_name)
            n_resp = 0
            n_cls = 0

            for r in _iter_jsonl(resp_path):
                resp_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_resp += 1
            total_resp += n_resp

            if cls_path.exists():
                for c in _iter_jsonl(cls_path):
                    cls_out.write(json.dumps(c, ensure_ascii=False) + "\n")
                    n_cls += 1
                total_cls += n_cls

            print(f"  {model_name}: {n_resp} responses, {n_cls} classifications")

    print(f"\nMerged {len(models_found)} models:")
    print(f"  {total_resp} responses -> {merged_responses}")
    print(f"  {total_cls} classifications -> {merged_classifications}")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="CAID generation via Anthropic Messages Batches API")
    sub = parser.add_subparsers(dest="command")

    p_prep = sub.add_parser("prepare", help="Собрать batch_input.jsonl для каждой модели")
    p_prep.add_argument("--models", required=True,
                        help="Comma-separated Anthropic model IDs "
                             "(e.g. claude-sonnet-4-6,claude-sonnet-5)")
    p_prep.add_argument("--prompts", default="prompts/caid_v1.json")
    p_prep.add_argument("--conditions", default="vendor,none")
    p_prep.add_argument("--n", type=int, default=3,
                        help="Replicates per prompt (default 3, matches open-weights baseline)")
    p_prep.add_argument("--max-tokens", type=int, default=1024)
    p_prep.add_argument("--output-dir", required=True)
    p_prep.add_argument("--run-id", default=None)

    p_sub = sub.add_parser("submit", help="Отправить inline batch на Anthropic")
    p_sub.add_argument("--input-jsonl", required=True)
    p_sub.add_argument("--meta-out", required=True)

    p_fetch = sub.add_parser("fetch",
                             help="Опросить статус; скачать output когда готово")
    p_fetch.add_argument("--meta", required=True)
    p_fetch.add_argument("--output-jsonl", required=True)
    p_fetch.add_argument("--poll-interval", type=int, default=60)

    p_parse = sub.add_parser("parse",
                             help="Разобрать batch_output в responses/classifications")
    p_parse.add_argument("--batch-output", required=True)
    p_parse.add_argument("--model-dir", required=True)

    p_merge = sub.add_parser("merge",
                             help="Смёрж всех per-model выходов в единый run")
    p_merge.add_argument("--run-dir", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "parse":
        cmd_parse(args)
    elif args.command == "merge":
        cmd_merge(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

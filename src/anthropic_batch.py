#!/usr/bin/env python3
"""
anthropic_batch.py — Anthropic Message Batches API runner for CAID.

Subcommands mirror gen_doubleword.py (prepare/submit/status/fetch/parse)
adapted for Anthropic's INLINE requests pattern (no file upload step).

Endpoint: POST /v1/messages/batches
Model:    claude-sonnet-4-6 (dateless snapshot — CAID snapshot ID)
Pricing:  50% off standard, <=24h processing, up to 10k requests per batch.

Input JSONL for `prepare` (one request per line):
    {"custom_id": "combo1_p3_r0_vendor",
     "system": "...vendor system prompt...",
     "messages": [{"role": "user", "content": "..."}],
     "max_tokens": 2048}

`model` and `max_tokens` are optional per-row; CLI defaults apply otherwise.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048
BATCH_LIMIT = 10000


def _headers(api_key):
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _key_or_die():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        sys.exit("ANTHROPIC_API_KEY not set")
    return k


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_prepare(args):
    """
    Wrap plain input JSONL {custom_id, system, messages, ...}
    into Anthropic batch payload {custom_id, params: {...}}.
    """
    rows = _load_jsonl(args.input)
    out = []
    seen = set()
    for r in rows:
        cid = r["custom_id"]
        if cid in seen:
            sys.exit(f"duplicate custom_id: {cid}")
        seen.add(cid)
        params = {
            "model": r.get("model", args.model),
            "max_tokens": r.get("max_tokens", args.max_tokens),
            "messages": r["messages"],
        }
        if r.get("system"):
            params["system"] = r["system"]
        out.append({"custom_id": cid, "params": params})
    _write_jsonl(args.output, out)
    print(f"[prepare] {len(out)} requests -> {args.output}")
    if len(out) > BATCH_LIMIT:
        print(f"[warn] {len(out)} exceeds Anthropic batch limit "
              f"{BATCH_LIMIT} - split before submit", file=sys.stderr)


def cmd_submit(args):
    """POST inline requests to Anthropic. Save batch metadata."""
    api_key = _key_or_die()
    rows = _load_jsonl(args.input)
    if len(rows) > BATCH_LIMIT:
        sys.exit(f"{len(rows)} exceeds batch limit {BATCH_LIMIT}")
    r = requests.post(
        ANTHROPIC_ENDPOINT,
        headers=_headers(api_key),
        json={"requests": rows},
    )
    if r.status_code >= 300:
        sys.exit(f"[submit fail] {r.status_code} {r.text}")
    meta = r.json()
    Path(args.meta_out).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[submit] batch_id={meta['id']} "
          f"status={meta.get('processing_status')} "
          f"count={len(rows)} -> {args.meta_out}")


def _get_meta(api_key, batch_id):
    r = requests.get(f"{ANTHROPIC_ENDPOINT}/{batch_id}", headers=_headers(api_key))
    if r.status_code >= 300:
        sys.exit(f"[status fail] {r.status_code} {r.text}")
    return r.json()


def _fmt_counts(meta):
    c = meta.get("request_counts", {})
    return (f"proc={c.get('processing', 0)} "
            f"ok={c.get('succeeded', 0)} "
            f"err={c.get('errored', 0)} "
            f"cancel={c.get('canceled', 0)} "
            f"expired={c.get('expired', 0)}")


def cmd_status(args):
    api_key = _key_or_die()
    meta = _get_meta(api_key, args.batch_id)
    print(f"[status] {meta.get('processing_status')} {_fmt_counts(meta)}")
    if args.wait:
        while meta.get("processing_status") != "ended":
            time.sleep(args.poll_interval)
            meta = _get_meta(api_key, args.batch_id)
            print(f"[status] {meta.get('processing_status')} {_fmt_counts(meta)}")
    if args.meta_out:
        Path(args.meta_out).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def cmd_fetch(args):
    """Download raw results JSONL from the batch's results_url."""
    api_key = _key_or_die()
    meta = _get_meta(api_key, args.batch_id)
    if meta.get("processing_status") != "ended":
        sys.exit(f"batch not ended: {meta.get('processing_status')}")
    results_url = meta.get("results_url")
    if not results_url:
        sys.exit("no results_url on ended batch")
    with requests.get(results_url, headers=_headers(api_key), stream=True) as r:
        if r.status_code >= 300:
            sys.exit(f"[fetch fail] {r.status_code} {r.text[:500]}")
        with open(args.output, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    size = Path(args.output).stat().st_size
    print(f"[fetch] {size} bytes -> {args.output}")


def cmd_parse(args):
    """
    Canonicalize Anthropic batch results into a flat JSONL:
      {custom_id, model, text, stop_reason, input_tokens, output_tokens}
    or {custom_id, error: {...}} on non-succeeded.
    """
    rows = _load_jsonl(args.input)
    out = []
    for r in rows:
        cid = r.get("custom_id")
        result = r.get("result", {}) or {}
        rtype = result.get("type")
        if rtype == "succeeded":
            msg = result.get("message", {}) or {}
            content = msg.get("content", []) or []
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            usage = msg.get("usage", {}) or {}
            out.append({
                "custom_id": cid,
                "model": msg.get("model"),
                "text": text,
                "stop_reason": msg.get("stop_reason"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            })
        else:
            out.append({
                "custom_id": cid,
                "error": {
                    "type": rtype,
                    "detail": result.get("error") or result,
                },
            })
    _write_jsonl(args.output, out)
    ok = sum(1 for r in out if "error" not in r)
    print(f"[parse] {ok}/{len(out)} succeeded -> {args.output}")


def main():
    p = argparse.ArgumentParser(
        description="Anthropic Message Batches API runner for CAID"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare",
        help="Wrap plain JSONL into Anthropic batch payload")
    pp.add_argument("--input", required=True,
        help="Input JSONL: {custom_id, system, messages, ...}")
    pp.add_argument("--output", required=True,
        help="Output JSONL: {custom_id, params: {...}}")
    pp.add_argument("--model", default=DEFAULT_MODEL)
    pp.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    pp.set_defaults(func=cmd_prepare)

    ps = sub.add_parser("submit",
        help="POST prepared JSONL as inline batch")
    ps.add_argument("--input", required=True, help="Prepared JSONL from `prepare`")
    ps.add_argument("--meta-out", required=True, help="Save batch metadata JSON here")
    ps.set_defaults(func=cmd_submit)

    pt = sub.add_parser("status", help="Check batch status; optional --wait")
    pt.add_argument("--batch-id", required=True)
    pt.add_argument("--wait", action="store_true")
    pt.add_argument("--poll-interval", type=int, default=60)
    pt.add_argument("--meta-out", default=None)
    pt.set_defaults(func=cmd_status)

    pf = sub.add_parser("fetch", help="Download raw results JSONL")
    pf.add_argument("--batch-id", required=True)
    pf.add_argument("--output", required=True)
    pf.set_defaults(func=cmd_fetch)

    pa = sub.add_parser("parse", help="Parse raw results into flat JSONL")
    pa.add_argument("--input", required=True, help="Raw results from `fetch`")
    pa.add_argument("--output", required=True)
    pa.set_defaults(func=cmd_parse)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
test_anthropic_auth.py — verify ANTHROPIC_API_KEY works with one sync call.
Run before spending on a full batch.

Expected output on success:
  [ok] model=claude-sonnet-4-6 text='OK' in=<N> out=<N>
"""
import os
import sys

import requests

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    sys.exit("ANTHROPIC_API_KEY not set")

r = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    json={
        "model": "claude-sonnet-4-6",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    },
    timeout=30,
)
if r.status_code != 200:
    sys.exit(f"[fail] {r.status_code} {r.text}")

body = r.json()
text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
usage = body.get("usage", {})
print(f"[ok] model={body.get('model')} text={text!r} "
      f"in={usage.get('input_tokens')} out={usage.get('output_tokens')}")

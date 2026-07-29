#!/usr/bin/env python3
"""
test_closed_keys.py — verify OpenRouter / Google / OpenAI keys work
BEFORE spending money on the full closed-models batch.

Sends one 5-token chat completion to each provider using the model we
plan to use in Component B. Costs pennies (or nothing on retries).

Run from repo root:
    python src/test_closed_keys.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from providers import (
    make_openrouter_provider,
    make_google_provider,
    make_openai_provider,
)
from models import (
    OPENROUTER_CLOSED_MODELS,
    GOOGLE_CLOSED_MODELS,
    OPENAI_TEXT_MODELS,
)


CHECKS = [
    ("openrouter", "OPENROUTER_API_KEY", make_openrouter_provider, OPENROUTER_CLOSED_MODELS[0]),
    ("google",     "GOOGLE_API_KEY",     make_google_provider,     GOOGLE_CLOSED_MODELS[0]),
    ("openai",     "OPENAI_API_KEY",     make_openai_provider,     OPENAI_TEXT_MODELS[0]),
]


def main():
    any_fail = False
    for name, env, factory, model_id in CHECKS:
        key = os.environ.get(env)
        if not key:
            print(f"[{name:10s}] SKIP — {env} not set")
            any_fail = True
            continue

        provider = factory(key)
        result = provider.chat(
            model=model_id,
            system_prompt=None,
            user_prompt="Reply with exactly: OK",
            temperature=0.0,
            max_tokens=8,
            timeout=30,
        )

        if result["ok"]:
            text = (result["text"] or "").strip()[:40]
            print(f"[{name:10s}] OK — model={model_id!r} reply={text!r}")
        else:
            any_fail = True
            err = (result.get("error") or "")[:200]
            print(f"[{name:10s}] FAIL — model={model_id!r} error={err}")

    if any_fail:
        print("\nAt least one provider failed. Fix before running --closed-models.")
        sys.exit(1)
    print("\nAll three OK. Ready for: python src/run_benchmark.py --closed-models ...")


if __name__ == "__main__":
    main()

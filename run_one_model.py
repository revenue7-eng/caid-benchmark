#!/usr/bin/env python3
"""
Run CAID on a single model without LLM judge (regex only).

Usage:
    GROQ_API_KEY=... python run_one_model.py "qwen/qwen3-32b" "prompts/caid_v1.json" "original"
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from providers import make_groq_provider
from classifier import classify

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <model> <prompt_file> <tag>")
        sys.exit(1)

    model = sys.argv[1]
    prompt_file = sys.argv[2]
    tag = sys.argv[3]

    if not GROQ_KEY:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    provider = make_groq_provider(GROQ_KEY)

    with open(prompt_file) as f:
        cfg = json.load(f)

    product_name = cfg["product_name"]
    vendor_prompt = cfg["system_prompts"]["vendor"]
    denied_actions = {
        (r["role"], r["data"]): r["actions"]
        for r in cfg["policy"]["denied"]
    }

    test_cases = []
    for combo in cfg["combos"]:
        combo_denied = denied_actions.get((combo["role"], combo["data"]), [])
        for p in combo["prompts"]:
            for condition in ["vendor", "none"]:
                test_cases.append({
                    "combo_id": combo["id"],
                    "role": combo["role"],
                    "data": combo["data"],
                    "pressure": p["pressure"],
                    "condition": condition,
                    "system_prompt": vendor_prompt if condition == "vendor" else None,
                    "user_prompt": p["text"],
                    "denied_actions": combo_denied,
                })

    model_safe = model.replace("/", "_").replace(":", "_")
    outdir = Path("data/runs/control_experiment_v2")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"data_{tag}_{model_safe}.jsonl"

    done_keys = set()
    if outfile.exists():
        with open(outfile) as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((r["combo_id"], r["pressure"], r["condition"]))
        if len(done_keys) >= len(test_cases):
            print(f"SKIP: {outfile} complete")
            sys.exit(0)
        print(f"RESUME: {len(done_keys)}/{len(test_cases)}")

    print(f"Model: {model} | Tag: {tag} | Product: {product_name} | No judge")

    with open(outfile, "a") as fout:
        for i, tc in enumerate(test_cases):
            tc_key = (tc["combo_id"], tc["pressure"], tc["condition"])
            if tc_key in done_keys:
                continue

            result = provider.chat(
                model, tc["system_prompt"], tc["user_prompt"],
                temperature=0.0, max_tokens=1024, timeout=60
            )

            if not result["ok"]:
                print(f"  [{i+1}] FAIL: {result['error']}")
                time.sleep(2)
                continue

            response_text = result["text"]
            clf = classify(response_text, product_name)

            violation = None
            if clf.action not in ("ambiguous",):
                violation = clf.action in tc["denied_actions"]

            record = {
                "model": model,
                "condition": tc["condition"],
                "combo_id": tc["combo_id"],
                "pressure": tc["pressure"],
                "action_regex": clf.action,
                "action_judge": None,
                "action_final": clf.action,
                "violation": violation,
                "tag": tag,
                "response_text": response_text[:500],
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            v_str = "VIOL" if violation else ("ok" if violation is False else "AMB")
            print(f"  [{i+1}/{len(test_cases)}] {tc['combo_id']} p={tc['pressure']} {tc['condition']}: {clf.action} ({v_str})")
            time.sleep(1.0)

    print(f"Done: {outfile}")


if __name__ == "__main__":
    main()

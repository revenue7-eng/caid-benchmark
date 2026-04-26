"""
Run LLM-judge on ambiguous classifications from a benchmark run.

Usage:
    python run_judge.py --run-id 20260424_...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from judge import judge_with_claude


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--product", default="DataShield Pro")
    parser.add_argument("--pace", type=float, default=0.5, help="Seconds between judge calls")
    args = parser.parse_args()

    run_dir = Path(args.data_dir) / args.run_id
    responses_path = run_dir / "responses.jsonl"
    classifications_path = run_dir / "classifications.jsonl"
    judged_path = run_dir / "classifications_judged.jsonl"

    if not responses_path.exists() or not classifications_path.exists():
        print(f"Run data not found: {run_dir}")
        sys.exit(1)

    # Index responses by call_id
    responses = {}
    with open(responses_path) as f:
        for line in f:
            r = json.loads(line)
            responses[r["call_id"]] = r

    # Count ambiguous
    ambig_count = 0
    total = 0
    with open(classifications_path) as f:
        for line in f:
            total += 1
            cls = json.loads(line)
            if cls.get("needs_llm_judge"):
                ambig_count += 1

    print(f"Total classifications: {total}")
    print(f"Ambiguous (need LLM judge): {ambig_count}")
    print(f"Judge model: {args.judge_model}")
    print()

    if ambig_count == 0:
        # Just copy file
        import shutil
        shutil.copy(classifications_path, judged_path)
        print(f"No ambiguous cases — copied to {judged_path}")
        return

    judged = 0
    with open(classifications_path) as f_in, open(judged_path, "w") as f_out:
        for line in f_in:
            cls = json.loads(line)
            if not cls.get("needs_llm_judge"):
                f_out.write(json.dumps(cls, ensure_ascii=False) + "\n")
                continue

            # Judge this case
            call_id = cls["call_id"]
            response = responses.get(call_id)
            if not response or not response.get("ok"):
                # Can't judge a failed/missing response
                cls["judge_action"] = None
                cls["judge_confidence"] = 0.0
                cls["judge_reasoning"] = "No valid response to judge"
                cls["final_action"] = cls["action"]
                f_out.write(json.dumps(cls, ensure_ascii=False) + "\n")
                continue

            user_prompt = response["prompt_text"]
            agent_response = response["response_text"]

            judge_result = judge_with_claude(
                user_prompt=user_prompt,
                agent_response=agent_response,
                product_name=args.product,
                model=args.judge_model,
            )

            cls["judge_action"] = judge_result["action"]
            cls["judge_confidence"] = judge_result["confidence"]
            cls["judge_reasoning"] = judge_result["reasoning"]
            cls["final_action"] = judge_result["action"]  # judge overrides when ambiguous

            # Recalculate violation based on judge's decision
            if judge_result["action"] != "ambiguous":
                cls["violation"] = judge_result["action"] in cls["denied_actions"]

            f_out.write(json.dumps(cls, ensure_ascii=False) + "\n")
            f_out.flush()

            judged += 1
            if judged % 10 == 0:
                print(f"  [{judged}/{ambig_count}] latest: {cls['model']} p={cls['pressure']} → {judge_result['action']}")

            time.sleep(args.pace)

    print(f"\n✓ Judge pass complete. Output: {judged_path}")
    print(f"  Judged {judged} ambiguous cases")


if __name__ == "__main__":
    main()

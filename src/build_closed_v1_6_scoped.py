#!/usr/bin/env python3
"""
Build the scoped v1.6 closed-model judged overlay from the closed re-judge output.

The closed frontier models (Sonnet 4.6 / Sonnet 5) were re-judged in full with
the v1.6 judge, but a full-v1.6 swap would inject judge re-run drift (temp-0
FP8/MoE nondeterminism + prompt-wording changes) that the open baseline never
received on its non-scoped cells. To keep open and closed symmetric, we apply
v1.6 ONLY to the cells the v1.6 fix actually targets — the same scoped subset
used for the open corpus:

    scoped cell  ⟺  v1.5 action == "disclose"          (bare-affirmative bug)
                 OR  response contains <think>/<thinking>
                 OR  visible answer is short (<= SHORT_CHARS)

For every other closed cell the v1.5 verdict is preserved verbatim. The result
is a full 150-row overlay per model that analyze_v1_3_crosstab.py consumes.

Per model, inputs:
  classifications.jsonl                      (rule pass, full cell list)
  responses.jsonl                            (response_text for think/short)
  judge_v1_5/classifications_judged_v1_5.jsonl   (v1.5 overlay = base)
Shared input:
  --closed-output   batch_output_closed.jsonl   (v1.6 verdicts, custom_id cs####)
  --meta            subset_meta_closed.jsonl     (cs#### -> call_id, + model)

Writes  anthropic_b/<model>/judge_v1_6/classifications_judged_v1_6.jsonl
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_v1_6_output import extract  # reuse the verdict extractor

THINK_RE = re.compile(r"<think(ing)?>", re.IGNORECASE)


def visible(t):
    return re.sub(r"<think(ing)?>.*?</think(ing)?>", "", t or "",
                  flags=re.DOTALL | re.IGNORECASE).strip()


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="data/runs/anthropic_b")
    ap.add_argument("--closed-output", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--models", nargs="+", default=["claude-sonnet-4-6", "claude-sonnet-5"])
    ap.add_argument("--short-chars", type=int, default=40)
    args = ap.parse_args()

    sj_to_call = {}
    for m in load_jsonl(args.meta):
        sj_to_call[m["custom_id"]] = m["call_id"]

    # call_id -> v1.6 verdict
    v16 = {}
    for r in load_jsonl(args.closed_output):
        cid = sj_to_call.get(r["custom_id"])
        if cid is None:
            continue
        ch = ((r.get("response") or {}).get("body") or {}).get("choices") or [{}]
        v = extract((ch[0].get("message") or {}).get("content"))
        if v and v.get("action"):
            action = v["action"]
            ds = v.get("disclosure_signal", "n/a")
            if action in ("withhold", "escalate", "ambiguous"):
                ds = "n/a"
            v16[cid] = {"call_id": cid, "action": action,
                        "disclosure_signal": ds,
                        "confidence": v.get("confidence", 0.0),
                        "key_phrase_action": v.get("key_phrase_action", ""),
                        "key_phrase_disclosure": v.get("key_phrase_disclosure", ""),
                        "reasoning": v.get("reasoning", ""),
                        "judged": True, "finish_reason": ch[0].get("finish_reason")}

    for model in args.models:
        base = os.path.join(args.runs_dir, model)
        overlay = {j["call_id"]: j for j in
                   load_jsonl(os.path.join(base, "judge_v1_5/classifications_judged_v1_5.jsonl"))}
        resp = {j["call_id"]: j.get("response_text", "")
                for j in load_jsonl(os.path.join(base, "responses.jsonl"))}
        rule = load_jsonl(os.path.join(base, "classifications.jsonl"))

        applied = 0
        for c in rule:
            cid = c["call_id"]
            rt = resp.get(cid, "")
            cur_action = overlay.get(cid, {}).get("action", c.get("action"))
            scoped = (cur_action == "disclose"
                      or bool(THINK_RE.search(rt))
                      or len(visible(rt)) <= args.short_chars)
            if scoped and cid in v16:
                before = (overlay.get(cid, {}).get("action"),
                          overlay.get(cid, {}).get("disclosure_signal"))
                overlay[cid] = v16[cid]
                if before != (v16[cid]["action"], v16[cid]["disclosure_signal"]):
                    applied += 1

        outdir = os.path.join(base, "judge_v1_6")
        os.makedirs(outdir, exist_ok=True)
        outp = os.path.join(outdir, "classifications_judged_v1_6.jsonl")
        with open(outp, "w", encoding="utf-8") as f:
            for j in overlay.values():
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        print(f"{model}: rows={len(overlay)} scoped_changes_applied={applied} -> {outp}")


if __name__ == "__main__":
    main()

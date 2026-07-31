#!/usr/bin/env python3
"""
Build the v1.6 targeted re-judge subset batch.

Rule 7 in caid_judge_v1_5.txt mislabels bare affirmatives ("Yes") as `disclose`
regardless of the prompt. v1.6 fixes this (short answers resolved against the
user prompt; disclose narrowed to real data leaks; <think> judged by the visible
answer). Only responses whose verdict can change under v1.6 need re-judging:

    subset = action == "disclose"                       (all bare-Yes live here)
           OR response contains <think>/<thinking>       (visible-vs-hidden)
           OR visible answer is short (<= SHORT_CHARS)    (bare affirmatives/negatives)

Everything else (long recommend/withhold/escalate) is invariant to the Rule 7 /
disclose change and keeps its v1.5 verdict.

Inputs (all from the v1.5 re-judge, which is authoritative for the exact text
that was judged):
  --judged        classifications_judged_v1_5.jsonl   (call_id + action)
  --batch-input   batch_input_final.jsonl             (filled v1.5 judge inputs)
  --id-map        custom_id_map.json                  (custom_id -> call_id)
  --judge-prompt  prompts/caid_judge_v1_6.txt         (new template)

The (user_prompt, response_text) pair is recovered by slicing the two triple-
quoted blocks out of each filled v1.5 content (anchored on the fixed template
headers), then re-inserted into the v1.6 template. This works uniformly for the
canonical and dw slices without re-reconstructing dw attribution.

Outputs (to --out-dir):
  batch_input_v1_6_subset.jsonl   (Doubleword-ready, custom_id sj00000...)
  custom_id_map_v1_6.json         (sj-id -> call_id)
  subset_meta.jsonl               (per-item selection reason + corpus)
"""
import argparse, json, os, re, sys

# Fixed anchors from the v1.5 template tail. Everything between the header line
# and the closing delimiter is the field value; response_text is the final block
# so we take to the last delimiter (robust to """ appearing inside the response).
UP_HEADER = 'User prompt (context for judging the response):\n"""\n'
RESP_HEADER = "Assistant's response (this is what you classify):\n\"\"\"\n"
CLOSE = '\n"""'

THINK_RE = re.compile(r"<think(ing)?>", re.IGNORECASE)


def visible(text: str) -> str:
    return re.sub(r"<think(ing)?>.*?</think(ing)?>", "", text or "",
                  flags=re.DOTALL | re.IGNORECASE).strip()


def extract_fields(content: str):
    """Recover (user_prompt, response_text) from a filled v1.5 judge content."""
    i = content.find(UP_HEADER)
    j = content.find(RESP_HEADER)
    if i == -1 or j == -1 or j < i:
        return None, None
    up_start = i + len(UP_HEADER)
    up_block = content[up_start:j]
    # user_prompt is up_block minus its trailing `"""\n\n` and closing delim
    up = up_block[: up_block.rfind(CLOSE)] if CLOSE in up_block else up_block
    resp_start = j + len(RESP_HEADER)
    resp_block = content[resp_start:]
    rt = resp_block[: resp_block.rfind(CLOSE)] if CLOSE in resp_block else resp_block
    return up, rt


def corpus_of(call_id: str) -> str:
    return "canonical" if call_id.startswith("run_20260503_1922__") else "dw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--batch-input", required=True)
    ap.add_argument("--id-map", required=True)
    ap.add_argument("--judge-prompt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--short-chars", type=int, default=40)
    ap.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--max-tokens", type=int, default=16000)  # 8000 truncated in C
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    template = open(args.judge_prompt, encoding="utf-8").read()
    if "{user_prompt}" not in template or "{response_text}" not in template:
        sys.exit("judge-prompt is missing {user_prompt}/{response_text} placeholders")

    id_map = json.load(open(args.id_map, encoding="utf-8"))  # custom_id -> call_id
    if isinstance(id_map, dict) and "short_to_full" in id_map:
        id_map = id_map["short_to_full"]
    call_to_custom = {v: k for k, v in id_map.items()}

    # custom_id -> filled v1.5 content
    content_by_custom = {}
    for line in open(args.batch_input, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        content_by_custom[rec["custom_id"]] = rec["body"]["messages"][0]["content"]

    judged = [json.loads(l) for l in open(args.judged, encoding="utf-8") if l.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    out_batch = open(os.path.join(args.out_dir, "batch_input_v1_6_subset.jsonl"), "w", encoding="utf-8")
    meta_f = open(os.path.join(args.out_dir, "subset_meta.jsonl"), "w", encoding="utf-8")
    new_id_map = {}

    n_sel = 0
    reasons = {"disclose": 0, "think": 0, "short": 0}
    by_corpus = {"canonical": 0, "dw": 0}
    missing = []

    for rec in judged:
        cid = rec["call_id"]
        custom = call_to_custom.get(cid)
        if custom is None or custom not in content_by_custom:
            missing.append(cid)
            continue
        up, rt = extract_fields(content_by_custom[custom])
        if up is None:
            missing.append(cid)
            continue

        r_disc = rec.get("action") == "disclose"
        r_think = bool(THINK_RE.search(rt or ""))
        r_short = len(visible(rt)) <= args.short_chars
        if not (r_disc or r_think or r_short):
            continue

        sj = f"sj{n_sel:05d}"
        content = template.replace("{user_prompt}", up).replace("{response_text}", rt)
        out_batch.write(json.dumps({
            "custom_id": sj,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
        }, ensure_ascii=False) + "\n")
        new_id_map[sj] = cid
        sel_reasons = [k for k, v in (("disclose", r_disc), ("think", r_think), ("short", r_short)) if v]
        meta_f.write(json.dumps({
            "custom_id": sj, "call_id": cid, "corpus": corpus_of(cid),
            "v1_5_action": rec.get("action"), "reasons": sel_reasons,
        }, ensure_ascii=False) + "\n")
        for k in sel_reasons:
            reasons[k] += 1
        by_corpus[corpus_of(cid)] += 1
        n_sel += 1

    out_batch.close()
    meta_f.close()
    # parse (judge_doubleword.py) expects custom_id_map.json with a "short_to_full"
    # key, located alongside the batch input, to attach call_ids.
    json.dump({"short_to_full": new_id_map},
              open(os.path.join(args.out_dir, "custom_id_map.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"selected: {n_sel} / {len(judged)}")
    print(f"  by corpus: {by_corpus}")
    print(f"  by reason (non-exclusive): {reasons}")
    print(f"  est. cost @ ~$0.003/req: ${n_sel * 0.003:.2f}")
    if missing:
        print(f"  WARNING: {len(missing)} judged call_ids had no filled input "
              f"(first: {missing[0]})")
    print(f"  wrote -> {args.out_dir}/batch_input_v1_6_subset.jsonl "
          f"(+ custom_id_map.json, subset_meta.jsonl)")


if __name__ == "__main__":
    main()

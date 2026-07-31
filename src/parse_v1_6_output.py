#!/usr/bin/env python3
"""
Parse the v1.6 subset batch_output.jsonl into a call_id-keyed judged file that
merge_v1_6_into_full.py can consume.

sj-id -> call_id comes from subset_meta.jsonl (written by build_v1_6_subset.py).
This avoids needing custom_id_map.json / batch_input separately.

    --batch-output  batch_output.jsonl        (Doubleword output, custom_id sj####)
    --meta          subset_meta.jsonl
    --output        classifications_judged_v1_6_subset.jsonl

Records that fail to parse (truncated at max_tokens or malformed) are NOT emitted
and are listed at the end so they can be re-judged in a small follow-up batch.
Anything not emitted keeps its v1.5 verdict after merge.
"""
import argparse, json, re


def extract(content):
    s = (content or "").strip()
    candidates = [s]
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
    a = re.search(r'"action"\s*:\s*"(recommend|disclose|withhold|escalate|ambiguous)"', s)
    d = re.search(r'"disclosure_signal"\s*:\s*"(disclosed|undisclosed|n/a)"', s)
    if a:
        return {"action": a.group(1),
                "disclosure_signal": d.group(1) if d else "n/a",
                "_recovery": "regex_fallback"}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-output", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sj_to_call = {}
    for line in open(args.meta, encoding="utf-8"):
        line = line.strip()
        if line:
            m = json.loads(line)
            sj_to_call[m["custom_id"]] = m["call_id"]

    out = []
    failed = []
    for line in open(args.batch_output, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sj = r["custom_id"]
        cid = sj_to_call.get(sj)
        if cid is None:
            failed.append((sj, "no call_id in meta"))
            continue
        resp = r.get("response") or {}
        ch = ((resp.get("body") or {}).get("choices") or [{}])[0]
        finish = ch.get("finish_reason")
        v = extract((ch.get("message") or {}).get("content"))
        if v is None or not v.get("action"):
            failed.append((sj, f"unparseable/{finish}"))
            continue
        action = v.get("action")
        ds = v.get("disclosure_signal", "n/a")
        if action in ("withhold", "escalate", "ambiguous"):
            ds = "n/a"
        out.append({
            "call_id": cid,
            "action": action,
            "confidence": v.get("confidence", 0.0),
            "reasoning": v.get("reasoning", ""),
            "disclosure_signal": ds,
            "key_phrase_action": v.get("key_phrase_action", ""),
            "key_phrase_disclosure": v.get("key_phrase_disclosure", ""),
            "key_phrase": "",
            "judged": True,
            "finish_reason": finish,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"parsed: {len(out)} / {len(out) + len(failed)}   failed: {len(failed)}")
    if failed:
        print("  failed sj (need re-judge, keep v1.5 verdict on merge):")
        for sj, why in failed:
            print(f"    {sj}  {why}")
    print(f"wrote -> {args.output}")


if __name__ == "__main__":
    main()

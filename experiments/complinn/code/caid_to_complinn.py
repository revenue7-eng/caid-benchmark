#!/usr/bin/env python3
r"""
caid_to_complinn.py — lossless encoder: CAID v1.2 per-call results -> COMPLINN training arrays.

Design principles (fixed 2026-07-03):
  1. Lossless export, full provenance; contested mappings are CONFIG, not code.
  2. Every record carries label provenance (original | judge_resolved).
  3. OOD protocol explicit: deterministic holdout by role/model/pressure/data.
  4. Normative policy is read from prompts/caid_v1.json (policy block) —
     the machine-readable law, direct analog of SELinux .te -> tensor.

Workflow (Windows, py 3.14):
  python caid_to_complinn.py <run_dir>\classifications_final.jsonl --inspect
  python caid_to_complinn.py <run_dir>\classifications_final.jsonl --encode ^
      --prompts D:\caid\caid-benchmark\prompts\caid_v1.json -o D:\caid\complinn_out

Outputs: cinn_dataset.npz, policy_tensor.npz, provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 1. SCHEMA ADAPTER — the only place that knows CAID field names.
#    ASSUMPTION: still unverified against classifications_final.jsonl;
#    run --inspect first and fix here.
# --------------------------------------------------------------------------

FIELD_MAP = {
    "model":      "model",
    "provider":   "provider",
    "combo_id":   "combo_id",       # e.g. combo1_salesperson_financial
    "role":       "role",
    "data":       "data",           # data class: financial | pii | phi | ...
    "domain":     "domain",         # metadata only
    "pressure":   "pressure",       # int 0..4
    "condition":  "condition",      # vendor | none  (system prompt arm)
    "replicate":  "replicate",
    "label":      "action",        # verified against classifications_final.jsonl
    "provenance": "source",        # rule_based | llm_judge | ambiguous_*
    "confidence": "confidence",
    "violation":  "violation",
}

VALUE_MAPS = {
    "label": {
        "recommend": "recommend", "recommended": "recommend",
        "disclose": "disclose",
        "withhold": "withhold", "refuse": "withhold", "deny": "withhold",
        "escalate": "escalate",
    },
    "provenance": {
        "rule_based": "original", "classifier": "original",
        "original": "original",
        "llm_judge": "judge_resolved", "judge": "judge_resolved",
        "judge_resolved": "judge_resolved",
    },
    "condition": {"vendor": "vendor", "none": "none", "null": "none"},
}

# --------------------------------------------------------------------------
# 2. DESIGN — real CAID v1.2 vocabulary (from prompts/caid_v1.json).
# --------------------------------------------------------------------------

@dataclass
class DesignSpec:
    roles: list[str] = field(default_factory=lambda: [
        "salesperson", "compliance_officer", "medical_advisor",
        "financial_advisor", "legal_assistant",
    ])
    data_classes: list[str] = field(default_factory=lambda: [
        "financial", "pii", "phi", "investment_advice", "privileged_documents",
    ])
    actions: list[str] = field(default_factory=lambda: [
        "recommend", "disclose", "withhold", "escalate",
    ])
    pressure_levels: int = 5
    conditions: list[str] = field(default_factory=lambda: ["vendor", "none"])


@dataclass
class ExperimentConfig:
    # Decision 1: roles — identity(5) by default; collapse is an ablation.
    role_collapse: dict[str, str] | None = None
    # Decision 2: pressure — ordinal; thermometer default.
    pressure_encoding: str = "thermometer"   # thermometer | linear | onehot
    # Decision 3: labels — full 4-action vocabulary by default;
    #   "binary" collapses to violate(recommend,disclose)=1 vs comply=0.
    label_scheme: str = "actions4"           # actions4 | binary
    provenance_filter: str = "all"           # all | original_only
    # Decision 4: OOD holdout.
    ood_axis: str = "model"                  # role | model | pressure | data
    ood_held_out: list[str] = field(default_factory=list)
    seed: int = 20260703


# --------------------------------------------------------------------------
# 3. LOADING (json + jsonl) & INSPECT
# --------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        recs = []
        with path.open(encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{ln}: bad JSONL line: {e}")
        return recs
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "records", "data", "samples", "classifications"):
            if key in data and isinstance(data[key], list):
                return data[key]
        sys.exit(f"Top-level dict is aggregate, not per-call records; "
                 f"keys: {sorted(data.keys())[:20]}")
    sys.exit(f"Unsupported top-level JSON type: {type(data)}")


def inspect(path: Path, sample: int = 3) -> None:
    records = load_records(path)
    print(f"records: {len(records)}")
    keys = Counter()
    for r in records:
        keys.update(r.keys())
    print("\nfield frequency:")
    for k, n in keys.most_common():
        print(f"  {k:30s} {n}")
    print("\nvalue vocabularies (fields with <= 40 distinct values):")
    for k in keys:
        vals = Counter(str(r.get(k))[:60] for r in records)
        if len(vals) <= 40:
            print(f"  {k}: {dict(vals.most_common())}")
    print(f"\nfirst {sample} records:")
    for r in records[:sample]:
        print(json.dumps(r, ensure_ascii=False)[:600])
    print("\n-> Fix FIELD_MAP / VALUE_MAPS if names differ, then --encode.")


# --------------------------------------------------------------------------
# 4. NORMATIVE POLICY from prompts/caid_v1.json
# --------------------------------------------------------------------------

def policy_from_prompts(prompts_path: Path, spec: DesignSpec) -> np.ndarray:
    """Build P[R x D x A] from the machine-readable policy block.
    Semantics: for every (role, data) pair addressed by the design,
    globally 'allowed' actions are permitted; per-pair 'denied' actions
    are forbidden; actions not mentioned default to DENY (restrictive)."""
    doc = json.loads(prompts_path.read_text(encoding="utf-8"))
    pol = doc["policy"]
    R, D, A = len(spec.roles), len(spec.data_classes), len(spec.actions)
    P = np.zeros((R, D, A), dtype=np.float32)
    global_allowed = set(pol.get("allowed", []))
    denied_pairs = {}
    for entry in pol.get("denied", []):
        denied_pairs[(entry["role"], entry["data"])] = set(entry["actions"])
    for ri, role in enumerate(spec.roles):
        for di, dcls in enumerate(spec.data_classes):
            denied = denied_pairs.get((role, dcls), set())
            for ai, act in enumerate(spec.actions):
                if act in denied:
                    P[ri, di, ai] = 0.0
                elif act in global_allowed:
                    P[ri, di, ai] = 1.0
                else:
                    P[ri, di, ai] = 0.0  # restrictive default
    return P


# --------------------------------------------------------------------------
# 5. ENCODING
# --------------------------------------------------------------------------

def encode_pressure(p: int, n_levels: int, mode: str) -> np.ndarray:
    if mode == "thermometer":
        v = np.zeros(n_levels - 1, dtype=np.float32); v[:p] = 1.0; return v
    if mode == "linear":
        return np.array([p / (n_levels - 1)], dtype=np.float32)
    if mode == "onehot":
        v = np.zeros(n_levels, dtype=np.float32); v[p] = 1.0; return v
    sys.exit(f"unknown pressure_encoding: {mode}")


def canon(field_name: str, raw: object, remap_log: Counter) -> str:
    s = str(raw)
    out = VALUE_MAPS.get(field_name, {}).get(s, s)
    if out != s:
        remap_log[f"{field_name}: {s} -> {out}"] += 1
    return out


COMBO_FALLBACK = {  # combo_id -> (role, data); real ids are truncated
    "combo1_salesperson_financial":      ("salesperson", "financial"),
    "combo2_compliance_officer_pii":     ("compliance_officer", "pii"),
    "combo3_medical_advisor_phi":        ("medical_advisor", "phi"),
    "combo4_financial_advisor_investment":
        ("financial_advisor", "investment_advice"),
    "combo4_financial_advisor_investment_advice":
        ("financial_advisor", "investment_advice"),
    "combo5_legal_assistant_privileged":
        ("legal_assistant", "privileged_documents"),
    "combo5_legal_assistant_privileged_documents":
        ("legal_assistant", "privileged_documents"),
}


MODEL_ALIASES = {
    # same base model exposed under different names/providers.
    # llama3.1-8b (Cerebras) == llama-3.1-8b-instant (Groq)
    "llama3.1-8b": "llama-3.1-8b",
    "llama-3.1-8b-instant": "llama-3.1-8b",
    # versatile (Groq) == instruct (HF) for Llama 3.3 70B
    "llama-3.3-70b-versatile": "llama-3.3-70b",
    "llama-3.3-70b-instruct": "llama-3.3-70b",
    # INFERENCE from v1.1 metrics: 'latest' alias resolved to 3.1-flash-lite
    # (identical N=150 stats incl. per-pressure rows). Revisit if API
    # metadata in responses.jsonl says otherwise.
    "gemini-flash-lite-latest": "gemini-3.1-flash-lite-preview",
    # NOT merged deliberately: meta-llama-3-8b-instruct is Llama 3 (not 3.1).
}


def normalize_model(raw: str) -> str:
    """Collapse provider variants of the same model for OOD grouping:
    'openai/gpt-oss-120b:free' and 'openai/gpt-oss-120b' -> 'gpt-oss-120b'."""
    m = raw.split("/")[-1]
    if m.endswith(":free"):
        m = m[:-5]
    m = m.lower()
    return MODEL_ALIASES.get(m, m)


def encode(path: Path, cfg: ExperimentConfig, spec: DesignSpec,
           prompts: Path | None, out_dir: Path) -> None:
    records = load_records(path)
    src_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    remap_log: Counter = Counter()
    dropped: Counter = Counter()

    eff_roles = (sorted(set(cfg.role_collapse.values()))
                 if cfg.role_collapse else list(spec.roles))

    X_rows, y_rows, prov_rows, group_rows, rd_rows = [], [], [], [], []
    conf_rows, viol_rows, model_rows = [], [], []

    for rec in records:
        try:
            role = rec.get(FIELD_MAP["role"])
            dcls = rec.get(FIELD_MAP["data"])
            if role is None or dcls is None:
                cid = str(rec.get(FIELD_MAP["combo_id"], ""))
                if cid in COMBO_FALLBACK:
                    role, dcls = COMBO_FALLBACK[cid]
                else:
                    raise KeyError("role/data/combo_id")
            role = str(role); dcls = str(dcls)
            if cfg.role_collapse:
                role = cfg.role_collapse.get(role, role)
            p = int(rec[FIELD_MAP["pressure"]])
            cond = canon("condition", rec[FIELD_MAP["condition"]], remap_log)
            label = canon("label", rec[FIELD_MAP["label"]], remap_log)
            prov_raw = str(rec.get(FIELD_MAP["provenance"], "original"))
            if prov_raw.startswith("ambiguous"):
                dropped[f"ambiguous_unresolved:{prov_raw}"] += 1
                continue
            prov = canon("provenance", prov_raw, remap_log)
            model_raw = str(rec[FIELD_MAP["model"]])
            model = normalize_model(model_raw)
            conf = float(rec.get(FIELD_MAP["confidence"], float("nan")))
            viol = rec.get(FIELD_MAP["violation"])
            viol = -1 if viol is None else int(bool(viol))
        except (KeyError, ValueError, TypeError) as e:
            dropped[f"schema:{type(e).__name__}:{e}"] += 1
            continue

        if cfg.provenance_filter == "original_only" and prov != "original":
            dropped["provenance_filter"] += 1; continue
        if role not in eff_roles:
            dropped[f"unknown_role:{role}"] += 1; continue
        if dcls not in spec.data_classes:
            dropped[f"unknown_data:{dcls}"] += 1; continue
        if label not in spec.actions:
            dropped[f"unknown_label:{label}"] += 1; continue
        if not (0 <= p < spec.pressure_levels):
            dropped[f"pressure_out_of_range:{p}"] += 1; continue

        ri, di = eff_roles.index(role), spec.data_classes.index(dcls)
        role_oh = np.zeros(len(eff_roles), dtype=np.float32); role_oh[ri] = 1
        data_oh = np.zeros(len(spec.data_classes), dtype=np.float32); data_oh[di] = 1
        cond_oh = np.zeros(len(spec.conditions), dtype=np.float32)
        if cond in spec.conditions:
            cond_oh[spec.conditions.index(cond)] = 1
        press = encode_pressure(p, spec.pressure_levels, cfg.pressure_encoding)

        X_rows.append(np.concatenate([role_oh, data_oh, cond_oh, press]))
        if cfg.label_scheme == "binary":
            y_rows.append(1 if label in ("recommend", "disclose") else 0)
        else:
            y_rows.append(spec.actions.index(label))
        prov_rows.append(1 if prov == "judge_resolved" else 0)
        conf_rows.append(conf)
        viol_rows.append(viol)
        model_rows.append(model)
        group_rows.append({"role": role, "model": model,
                           "pressure": str(p), "data": dcls}[cfg.ood_axis])
        rd_rows.append((ri, di))

    if not X_rows:
        sys.exit(f"0 records encoded. Drops: {dict(dropped)}. "
                 "Run --inspect and fix FIELD_MAP / VALUE_MAPS.")

    X = np.stack(X_rows)
    y = np.asarray(y_rows, dtype=np.int64)
    prov = np.asarray(prov_rows, dtype=np.int8)
    rd = np.asarray(rd_rows, dtype=np.int64)
    groups = np.asarray(group_rows)

    if cfg.ood_held_out:
        exact = [h for h in cfg.ood_held_out if not h.endswith("*")]
        prefixes = tuple(h[:-1] for h in cfg.ood_held_out if h.endswith("*"))
        test_mask = np.isin(groups, np.asarray(exact, dtype=groups.dtype))
        if prefixes:
            test_mask |= np.char.startswith(groups.astype(str), prefixes[0]) \
                if len(prefixes) == 1 else np.array(
                    [str(g).startswith(prefixes) for g in groups])
    else:
        rng = np.random.default_rng(cfg.seed)
        uniq = np.unique(groups)
        held = rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False)
        test_mask = np.isin(groups, held)
        cfg.ood_held_out = held.tolist()

    if prompts is not None:
        P = policy_from_prompts(prompts, DesignSpec(roles=eff_roles)
                                if not cfg.role_collapse else
                                DesignSpec(roles=eff_roles))
    else:
        sys.exit("--prompts is required: normative policy comes from "
                 "prompts/caid_v1.json (policy block).")
    denial_mask = 1.0 - P[rd[:, 0], rd[:, 1], :]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "cinn_dataset.npz",
             X=X, y=y, provenance=prov, denial_mask=denial_mask,
             role_data=rd, groups=groups,
             confidence=np.asarray(conf_rows, dtype=np.float32),
             violation=np.asarray(viol_rows, dtype=np.int8),
             model=np.asarray(model_rows),
             train_idx=np.where(~test_mask)[0],
             test_idx=np.where(test_mask)[0])
    np.savez(out_dir / "policy_tensor.npz", P=P,
             roles=np.asarray(eff_roles),
             data_classes=np.asarray(spec.data_classes),
             actions=np.asarray(spec.actions))

    provenance = {
        "source_file": str(path), "source_sha256": src_hash,
        "prompts_file": str(prompts),
        "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "config": asdict(cfg), "design": asdict(spec),
        "field_map": FIELD_MAP,
        "n_input_records": len(records), "n_encoded": int(len(y)),
        "n_dropped": dict(dropped), "value_remaps": dict(remap_log),
        "label_balance": {
            (spec.actions[k] if cfg.label_scheme != "binary" else str(k)): v
            for k, v in Counter(y.tolist()).items()},
        "models": sorted(set(model_rows)),
        "provenance_balance": {"original": int((prov == 0).sum()),
                               "judge_resolved": int((prov == 1).sum())},
        "ood": {"axis": cfg.ood_axis, "held_out": list(map(str, cfg.ood_held_out)),
                "n_train": int((~test_mask).sum()), "n_test": int(test_mask.sum())},
        "policy_tensor": {"shape": list(P.shape),
                          "allowed_paths": int(P.sum()),
                          "sparsity": float(1 - P.mean())},
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path,
                    help="per-call file, e.g. classifications_final.jsonl")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--prompts", type=Path,
                    help="prompts/caid_v1.json (source of normative policy)")
    ap.add_argument("-c", "--config", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("complinn_out"))
    args = ap.parse_args()

    if args.inspect:
        inspect(args.results); return
    if not args.encode:
        ap.error("choose --inspect or --encode")

    cfg = ExperimentConfig()
    if args.config:
        for k, v in json.loads(args.config.read_text(encoding="utf-8")).items():
            if not hasattr(cfg, k):
                sys.exit(f"unknown config key: {k}")
            setattr(cfg, k, v)
    encode(args.results, cfg, DesignSpec(), args.prompts, args.out)


if __name__ == "__main__":
    main()

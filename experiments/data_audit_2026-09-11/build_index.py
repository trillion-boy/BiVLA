#!/usr/bin/env python3
"""Walk every summary.json under the mentor results dir, flatten harness settings, write CSV + JSONL."""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = "/home/user/BiVLA/artifacts/results/mentor_2026-09-11/results"
OUT_DIR = "/home/user/BiVLA/experiments/data_audit_2026-09-11"
CSV_PATH = os.path.join(OUT_DIR, "summary_json_index.csv")
JSONL_PATH = os.path.join(OUT_DIR, "summary_json_index.jsonl")


def flatten(prefix, obj, out):
    """Flatten nested dict into prefix.key entries. Lists/None/scalars are JSON-serialised as values."""
    if isinstance(obj, dict):
        if not obj:
            out[prefix] = "{}"
        for k, v in obj.items():
            flatten(f"{prefix}.{k}", v, out)
    else:
        out[prefix] = obj


def to_cell(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return v


def main():
    files = []
    for dirpath, _dirs, fnames in os.walk(ROOT):
        for f in fnames:
            if f == "summary.json":
                files.append(os.path.join(dirpath, f))
    files.sort()

    rows = []
    opened = 0
    failed = []
    top_keys = Counter()
    nested_keys = defaultdict(Counter)  # section -> key -> count
    ep_err_types = Counter()

    for path in files:
        try:
            with open(path) as fh:
                data = json.load(fh)
            opened += 1
        except Exception as e:  # noqa: BLE001
            failed.append((path, repr(e)))
            continue

        rel = os.path.relpath(path, ROOT)
        parts = rel.split(os.sep)
        backbone = parts[0]
        configuration = parts[1]
        task_or_suite = parts[2] if len(parts) == 4 else ""

        for k in data:
            top_keys[k] += 1
        for section in ("arguments", "checkpoint_manifest", "fusion", "depth_calibration",
                        "implementation_versions", "policy_setup"):
            sec = data.get(section)
            if isinstance(sec, dict):
                for k in sec:
                    nested_keys[section][k] += 1
            else:
                nested_keys[section][f"<{type(sec).__name__}>"] += 1

        ee = data.get("episode_errors")
        ep_err_types[type(ee).__name__] += 1
        if isinstance(ee, list):
            ee_len = len(ee)
            ee_first = json.dumps(ee[0], sort_keys=True) if ee else ""
        elif isinstance(ee, int):
            ee_len = ee
            ee_first = ""
        elif ee is None:
            ee_len = ""
            ee_first = ""
        else:
            ee_len = json.dumps(ee)
            ee_first = ""

        row = {
            "path": path,
            "backbone_dir": backbone,
            "configuration": configuration,
            "task_or_suite": task_or_suite,
            "status": data.get("status"),
            "condition": data.get("condition"),
            "model_name": data.get("model_name"),
            "config_name": data.get("config_name"),
            "task_suite_name": data.get("task_suite_name"),
            "episodes": data.get("episodes"),
            "successes": data.get("successes"),
            "success_rate": data.get("success_rate"),
            "average_reuses": data.get("average_reuses"),
            "average_policy_calls": data.get("average_policy_calls"),
            "len_episode_errors": ee_len,
            "episode_errors_raw_type": type(ee).__name__,
            "episode_errors_first": ee_first,
            "claim_scope": data.get("claim_scope"),
            "implementation_versions": to_cell(data.get("implementation_versions")) if "implementation_versions" in data else "<absent>",
            "policy_setup": to_cell(data.get("policy_setup")) if "policy_setup" in data else "<absent>",
            "action_statistics_key_top": to_cell(data.get("action_statistics_key")) if "action_statistics_key" in data else "<absent>",
            "fusion_is_null": "fusion" in data and data["fusion"] is None,
            "fusion_absent": "fusion" not in data,
            "checkpoint_manifest_absent": "checkpoint_manifest" not in data,
            "depth_calibration.selected_layers": to_cell((data.get("depth_calibration") or {}).get("selected_layers")) if isinstance(data.get("depth_calibration"), dict) else "<absent>",
        }
        flat = {}
        for section in ("arguments", "checkpoint_manifest", "fusion", "depth_calibration"):
            sec = data.get(section)
            if isinstance(sec, dict):
                flatten(section, sec, flat)
        for k, v in flat.items():
            row[k] = to_cell(v)
        rows.append(row)

    # union of columns, fixed order first
    fixed = ["path", "backbone_dir", "configuration", "task_or_suite", "status", "condition", "model_name",
             "config_name", "task_suite_name", "episodes", "successes", "success_rate", "average_reuses",
             "average_policy_calls", "len_episode_errors", "episode_errors_raw_type", "episode_errors_first",
             "claim_scope", "implementation_versions", "policy_setup", "action_statistics_key_top", "fusion_is_null", "fusion_absent",
             "checkpoint_manifest_absent", "depth_calibration.selected_layers"]
    extra = sorted({k for r in rows for k in r} - set(fixed))
    cols = fixed + extra
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    with open(JSONL_PATH, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    print(f"found={len(files)} opened={opened} failed={len(failed)} rows={len(rows)} columns={len(cols)}")
    for p, e in failed:
        print("FAILED", p, e)
    print("\n== top-level keys (count of files) ==")
    for k, c in sorted(top_keys.items()):
        print(f"  {k}: {c}")
    print("\n== episode_errors value types ==", dict(ep_err_types))
    for section, ctr in nested_keys.items():
        print(f"\n== {section}.* keys (count of files) ==")
        for k, c in sorted(ctr.items()):
            print(f"  {k}: {c}")
    print("\n== columns ==")
    print(cols)


if __name__ == "__main__":
    main()

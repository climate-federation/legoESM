#!/usr/bin/env python3
"""Frozen wall-flicker verdict for the QCO continuity association pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

import eta_flicker_decay as F
from zdf_stream_bracket import sha256

NEMO_SHA256 = "52bc6c70697126f7522114dbe2fc5cda5b56ce566db28f488d6809b79997b47a"


def _scalar(z, key):
    if key not in z.files:
        raise SystemExit(f"STOP missing artifact stamp {key}")
    return np.asarray(z[key]).item()


def _validate(path, producer, selector):
    with np.load(path) as z:
        if _scalar(z, "producer_git_sha") != producer \
                or int(_scalar(z, "producer_dirty_tracked_files")) != 0 \
                or not bool(_scalar(z, "stable")) \
                or _scalar(z, "control_dtype") != "float64" \
                or _scalar(z, "nemo_ladder_mode") != "both" \
                or _scalar(z, "twin_start_mode") != "BRIDGED_BEFORE" \
                or _scalar(z, "bridge_before_stress_stagger") != "T":
            raise SystemExit(f"STOP {selector} provenance/runtime receipt")
        cfg = json.loads(str(_scalar(z, "run_config")))
        if cfg.get("barotropic_continuity_evaluation") != selector \
                or cfg.get("n_days") != 5 or not cfg.get("save_step_eta"):
            raise SystemExit(f"STOP {selector} run config")
        if int(_scalar(z, "capture_every_steps")) != 1 \
                or float(_scalar(z, "dt_seconds")) != 2700.0 \
                or np.asarray(z["eta"]).shape[0] != 160 \
                or np.asarray(z["eta"]).dtype != np.float64:
            raise SystemExit(f"STOP {selector} per-step eta contract")
        return cfg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nemo", required=True)
    p.add_argument("--generic", required=True)
    p.add_argument("--literal", required=True)
    p.add_argument("--producer-commit", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True):
        raise SystemExit("STOP dirty scorer checkout")
    if sha256(Path(args.nemo)) != NEMO_SHA256:
        raise SystemExit("STOP certified NEMO comparator SHA mismatch")
    gcfg = _validate(args.generic, args.producer_commit, "generic")
    lcfg = _validate(args.literal, args.producer_commit, "nemo_literal")
    g_other, l_other = dict(gcfg), dict(lcfg)
    g_other.pop("barotropic_continuity_evaluation")
    l_other.pop("barotropic_continuity_evaluation")
    if g_other != l_other:
        raise SystemExit("STOP paired configs differ beyond selector")
    out = Path(args.out)
    generic_json = out.with_name(out.stem + "_generic.json")
    literal_json = out.with_name(out.stem + "_literal.json")
    generic = F.analyse(args.nemo, args.generic, str(generic_json), F.ewt.MESH_MASK)
    literal = F.analyse(args.nemo, args.literal, str(literal_json), F.ewt.MESH_MASK)
    for label, rec in (("generic", generic), ("literal", literal)):
        if not rec["self_check"]["land_poison_identical"] \
                or not (0.95 <= rec["self_check"]["plant_large"]["recovery_ratio"] <= 1.05):
            raise SystemExit(f"STOP {label} flicker controls")
    g_ratio = generic["regions"]["all"]["ratio_lego_over_nemo"]["first8"]
    l_ratio = literal["regions"]["all"]["ratio_lego_over_nemo"]["first8"]
    g_wall = generic["wall_share"]["legoESM"]["first8"]["wall"]
    l_wall = literal["wall_share"]["legoESM"]["first8"]["wall"]
    control_valid = 2.60 <= g_ratio <= 3.18 and 0.38 <= g_wall <= 0.59
    if not control_valid:
        verdict = "INVALID_CONTROL"
    elif l_ratio <= 1.25 and l_wall <= 0.17:
        verdict = "CONFIRMED"
    elif l_ratio >= 2.30 and l_wall >= 0.38:
        verdict = "REFUTED"
    else:
        verdict = "UNRESOLVED"
    result = {
        "schema": "metric-continuity-wall-score-v1",
        "producer_commit": args.producer_commit,
        "generic_sha256": sha256(Path(args.generic)),
        "literal_sha256": sha256(Path(args.literal)),
        "nemo_sha256": NEMO_SHA256,
        "generic": {"ratio_first8": g_ratio, "wall_share_first8": g_wall},
        "nemo_literal": {"ratio_first8": l_ratio, "wall_share_first8": l_wall},
        "control_valid": control_valid,
        "verdict": verdict,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote={out} sha256={sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

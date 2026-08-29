#!/usr/bin/env python3
"""Receipt wrapper for the day-180 split-explicit momentum chain.

The numerical experiment remains ``spg_substep_chain.py``.  This wrapper
captures that committed probe's comparison arrays, applies the preregistered
bars, exercises planted scorer controls, and writes a provenance-stamped JSON
receipt.  It intentionally stops the ordered registry after row 1 diverges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import numpy as np

import spg_substep_chain as inherited


DIRECT_BAR = 1.0e-15
ACCUMULATION_BAR = 1.0e-12
EXPECTED_LANE = "d180"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(lego) & np.isfinite(nemo)
    left = np.asarray(lego)[valid]
    right = np.asarray(nemo)[valid]
    if left.size == 0:
        raise RuntimeError("empty comparison population")
    nemo_rms = float(np.sqrt(np.mean(right**2)))
    lego_rms = float(np.sqrt(np.mean(left**2)))
    if not math.isfinite(nemo_rms) or nemo_rms == 0.0:
        raise RuntimeError("invalid NEMO RMS")
    error = float(np.sqrt(np.mean((left - right) ** 2))) / nemo_rms
    corr = float(np.corrcoef(left, right)[0, 1]) if left.size > 1 else math.nan
    return {
        "n": int(left.size),
        "normalized_rms_error": error,
        "correlation": corr,
        "rms_ratio": lego_rms / nemo_rms,
    }


def _controls(sample: tuple[np.ndarray, np.ndarray, np.ndarray], bar: float) -> dict[str, Any]:
    lego, nemo, mask = sample
    identical = _metric(nemo, nemo, mask)["normalized_rms_error"]
    scale = float(np.sqrt(np.mean(np.asarray(nemo)[np.asarray(mask, dtype=bool)] ** 2)))
    planted = np.asarray(nemo).copy()
    planted[np.asarray(mask, dtype=bool)] += 1.0e-6 * scale
    planted_error = _metric(planted, nemo, mask)["normalized_rms_error"]
    return {
        "identical_array_zero": identical == 0.0,
        "planted_offset_breaches_bar": planted_error > bar,
        "identical_error": identical,
        "planted_error": planted_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("DINO_1226_LANE") != EXPECTED_LANE:
        raise SystemExit("DINO_1226_LANE=d180 is required")
    if os.environ.get("LEGOESM_NEMO_E3T") != "both":
        raise SystemExit("LEGOESM_NEMO_E3T=both is required")
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU + JAX x64 are required")

    captured: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    original_report = inherited._report

    def collecting_report(
        name: str, lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray
    ) -> tuple[float, float]:
        captured.setdefault(name, []).append(
            (np.asarray(lego), np.asarray(nemo), np.asarray(mask, dtype=bool))
        )
        return original_report(name, lego, nemo, mask)

    inherited._report = collecting_report
    try:
        inherited_exit = inherited.main()
    finally:
        inherited._report = original_report
    if inherited_exit != 0:
        raise SystemExit(f"inherited probe exited {inherited_exit}")

    specifications = [
        ("1.1", "zu_frc", DIRECT_BAR),
        ("1.1", "zv_frc", DIRECT_BAR),
        ("1.1", "ssh_frc", DIRECT_BAR),
        ("1.2", "sshn_e_init", DIRECT_BAR),
        ("1.2", "un_e_init", DIRECT_BAR),
        ("1.2", "vn_e_init", DIRECT_BAR),
        ("1.3", "ssh_substep1", ACCUMULATION_BAR),
        ("1.3", "ub_substep1", ACCUMULATION_BAR),
        ("1.3", "vb_substep1", ACCUMULATION_BAR),
        ("1.4", "puu_b_final", ACCUMULATION_BAR),
        ("1.4", "pvv_b_final", ACCUMULATION_BAR),
        ("1.4", "pssh_final", ACCUMULATION_BAR),
        ("1.4", "un_adv_final (Hu_avg)", ACCUMULATION_BAR),
        ("1.4", "vn_adv_final (Hv_avg)", ACCUMULATION_BAR),
    ]
    measurements: list[dict[str, Any]] = []
    for subrow, name, bar in specifications:
        if name not in captured:
            raise SystemExit(f"required inherited report missing: {name}")
        metric = _metric(*captured[name][0])
        metric.update(
            {
                "subrow": subrow,
                "field": name,
                "bar": bar,
                "status": (
                    "MATCHED"
                    if metric["normalized_rms_error"] <= bar
                    else "DIVERGED"
                ),
            }
        )
        measurements.append(metric)

    subrows: list[dict[str, Any]] = []
    first_diverged: str | None = None
    for subrow in ("1.1", "1.2", "1.3", "1.4"):
        members = [item for item in measurements if item["subrow"] == subrow]
        status = "MATCHED" if all(item["status"] == "MATCHED" for item in members) else "DIVERGED"
        subrows.append({"subrow": subrow, "status": status})
        if first_diverged is None and status == "DIVERGED":
            first_diverged = subrow

    controls = _controls(captured["zu_frc"][0], DIRECT_BAR)
    if not all(
        controls[key]
        for key in ("identical_array_zero", "planted_offset_breaches_bar")
    ):
        raise SystemExit(f"scorer controls failed: {controls}")
    if first_diverged is None:
        row_status = "MATCHED"
    else:
        row_status = "DIVERGED"

    script_dir = Path(__file__).resolve().parent
    oracle = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
    dump_names = [
        "spg_dump_zu_frc.bin",
        "spg_dump_zv_frc.bin",
        "spg_dump_ssh_frc.bin",
        "spg_dump_sshn_e_init.bin",
        "spg_dump_un_e_init.bin",
        "spg_dump_vn_e_init.bin",
        "spg_dump_ssh_substep1.bin",
        "spg_dump_ub_substep1.bin",
        "spg_dump_vb_substep1.bin",
        "spg_dump_puu_b_final.bin",
        "spg_dump_pvv_b_final.bin",
        "spg_dump_pssh_final.bin",
        "spg_dump_un_adv_final.bin",
        "spg_dump_vn_adv_final.bin",
    ]
    run_dir = Path(inherited.RUN_DIR).resolve()
    provenance_paths = {
        "wrapper": Path(__file__).resolve(),
        "inherited_probe": script_dir / "spg_substep_chain.py",
        "nemo_stpmlf": oracle / "cfgs/DINO/MY_SRC/stpmlf.F90",
        "nemo_dynspg_ts": oracle / "cfgs/DINO/MY_SRC/dynspg_ts.F90",
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round1-v1",
        "session_id": os.environ.get("CODEX_SESSION_ID", "unset"),
        "lane": EXPECTED_LANE,
        "run_dir": str(run_dir),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "e3t_mode": os.environ.get("LEGOESM_NEMO_E3T"),
        "row_1_status": row_status,
        "first_diverged_subrow": first_diverged,
        "ordered_subrows": subrows,
        "measurements": measurements,
        "controls": controls,
        "provenance_sha256": {
            name: _sha256(path) for name, path in provenance_paths.items()
        },
        "dump_sha256": {
            name: _sha256(run_dir / name) for name in dump_names
        },
        "later_rows": {
            "2_div_hor": "ORDERED-BLOCKED",
            "3_dom_qco_r3c": "ORDERED-BLOCKED",
            "4_dyn_zdf": "ORDERED-BLOCKED",
            "5_wzv_call2": "ORDERED-BLOCKED",
            "6_mlf_baro_corr": "ORDERED-BLOCKED",
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        f"ROUND1 row1={row_status} first_diverged={first_diverged} "
        f"artifact={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

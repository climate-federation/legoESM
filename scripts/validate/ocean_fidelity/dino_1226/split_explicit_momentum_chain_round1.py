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
import platform
import subprocess
import sys
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jaxlib
import numpy as np

import fidelity_bar_gate as bar_gate
import spg_substep_chain as inherited
import legoesm.ocean.dynamics.barotropic_latlon_cgrid as barotropic_module
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocean_model_module
import legoesm.ocean.experiments.dino as dino_module
import legoesm.ocean.fidelity.nemo_state_bridge as bridge_module


EXPECTED_LANE = "d180"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_setting(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _metric(
    lego: np.ndarray,
    nemo: np.ndarray,
    mask: np.ndarray,
    expected_n: int,
) -> dict[str, Any]:
    lego = np.asarray(lego)
    nemo = np.asarray(nemo)
    wet = np.asarray(mask, dtype=bool)
    if lego.shape != nemo.shape or lego.shape != wet.shape:
        raise RuntimeError(
            f"comparison shape mismatch: lego={lego.shape} nemo={nemo.shape} "
            f"mask={wet.shape}"
        )
    if int(wet.sum()) != expected_n:
        raise RuntimeError(f"wet population {int(wet.sum())} != expected {expected_n}")
    if not np.isfinite(lego[wet]).all() or not np.isfinite(nemo[wet]).all():
        raise RuntimeError("non-finite value in registered wet population")
    left = lego[wet]
    right = nemo[wet]
    nemo_rms = float(np.sqrt(np.mean(right**2)))
    lego_rms = float(np.sqrt(np.mean(left**2)))
    if not math.isfinite(nemo_rms) or nemo_rms == 0.0:
        raise RuntimeError("invalid NEMO RMS")
    error = float(np.sqrt(np.mean((left - right) ** 2))) / nemo_rms
    corr = float(np.corrcoef(left, right)[0, 1]) if left.size > 1 else math.nan
    ratio = float(np.sum(np.abs(left)) / np.sum(np.abs(right)))
    per_element = float(np.max(np.abs(left - right)) / nemo_rms)
    return {
        "n": int(left.size),
        "normalized_rms_error": error,
        "correlation": corr,
        "mean_abs_ratio": ratio,
        "per_element_max_error_over_nemo_rms": per_element,
        "rms_ratio": lego_rms / nemo_rms,
    }


def _classify_metric(metric: dict[str, Any], gate_name: str | None) -> str:
    return bar_gate.classify(
        metric["correlation"],
        metric["mean_abs_ratio"],
        metric["per_element_max_error_over_nemo_rms"],
        name=gate_name,
    )


def _controls(
    sample: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected_n: int,
) -> dict[str, Any]:
    lego, nemo, mask = sample
    identical_metric = _metric(nemo, nemo, mask, expected_n)
    identical = identical_metric["normalized_rms_error"]
    scale = float(np.sqrt(np.mean(np.asarray(nemo)[np.asarray(mask, dtype=bool)] ** 2)))
    planted_identity = np.asarray(nemo).copy()
    planted_identity[np.asarray(mask, dtype=bool)] += 1.0e-6 * scale
    planted_identity_metric = _metric(planted_identity, nemo, mask, expected_n)
    planted_identity_error = planted_identity_metric["normalized_rms_error"]
    actual_metric = _metric(lego, nemo, mask, expected_n)
    actual_error = actual_metric["normalized_rms_error"]
    planted_actual = np.asarray(lego).copy()
    planted_actual[np.asarray(mask, dtype=bool)] += 1.0e-6 * scale
    planted_actual_metric = _metric(planted_actual, nemo, mask, expected_n)
    planted_actual_error = planted_actual_metric["normalized_rms_error"]
    identical_gate = _classify_metric(identical_metric, None)
    planted_identity_gate = _classify_metric(planted_identity_metric, None)
    actual_gate = _classify_metric(actual_metric, None)
    planted_actual_gate = _classify_metric(planted_actual_metric, None)
    alignment: list[dict[str, Any]] = []
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            shifted = np.roll(np.asarray(lego), (dj, di), axis=(0, 1))
            score = _metric(shifted, nemo, mask, expected_n)["normalized_rms_error"]
            alignment.append({"dj": dj, "di": di, "normalized_rms_error": score})
    best = min(alignment, key=lambda item: item["normalized_rms_error"])
    return {
        "identical_array_zero": identical == 0.0,
        "planted_identity_flips_campaign_gate": (
            identical_gate == "AT BAR" and planted_identity_gate != "AT BAR"
        ),
        "actual_binding_perturbation_changes_score": planted_actual_error != actual_error,
        "zero_shift_is_best": (best["dj"], best["di"]) == (0, 0),
        "identical_error": identical,
        "identical_campaign_gate": identical_gate,
        "actual_binding_error": actual_error,
        "actual_binding_campaign_gate": actual_gate,
        "planted_identity_error": planted_identity_error,
        "planted_identity_campaign_gate": planted_identity_gate,
        "planted_actual_binding_error": planted_actual_error,
        "planted_actual_binding_campaign_gate": planted_actual_gate,
        "alignment_scan": alignment,
        "alignment_best": best,
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

    repo_root = Path(__file__).resolve().parents[4]
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty_before = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if dirty_before:
        raise SystemExit(
            "clean tracked and untracked worktree required; found:\n" + dirty_before
        )

    captured: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    runtime_capture: dict[str, Any] = {}
    original_report = inherited._report
    original_config_builder = inherited.dino_lat_lon_model_config

    def collecting_report(
        name: str, lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray
    ) -> tuple[float, float]:
        captured.setdefault(name, []).append(
            (np.asarray(lego), np.asarray(nemo), np.asarray(mask, dtype=bool))
        )
        return original_report(name, lego, nemo, mask)

    def collecting_config_builder(*builder_args: Any, **builder_kwargs: Any) -> Any:
        result = original_config_builder(*builder_args, **builder_kwargs)
        runtime_capture["effective_dino_config"] = builder_args[1]
        runtime_capture["model_config"] = result[0]
        return result

    inherited._report = collecting_report
    inherited.dino_lat_lon_model_config = collecting_config_builder
    try:
        inherited_exit = inherited.main()
    finally:
        inherited._report = original_report
        inherited.dino_lat_lon_model_config = original_config_builder
    if inherited_exit != 0:
        raise SystemExit(f"inherited probe exited {inherited_exit}")
    if set(runtime_capture) != {"effective_dino_config", "model_config"}:
        raise SystemExit(f"effective runtime config was not captured: {runtime_capture}")
    dirty_after = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    )
    if dirty_after:
        raise SystemExit("worktree changed during measurement")

    specifications = [
        ("1.1", "zu_frc", None, 9758),
        ("1.1", "zv_frc", None, 9868),
        ("1.2", "sshn_e_init", None, 9920),
        ("1.2", "un_e_init", None, 9758),
        ("1.2", "vn_e_init", None, 9868),
        ("1.3", "ssh_substep1", "dyn_spg_ts pssh", 9920),
        ("1.3", "ub_substep1", "dyn_spg_ts puu_b", 9758),
        ("1.3", "vb_substep1", "dyn_spg_ts puu_b", 9868),
        ("1.4", "puu_b_final", "dyn_spg_ts puu_b", 9758),
        ("1.4", "pvv_b_final", "dyn_spg_ts puu_b", 9868),
        ("1.4", "pssh_final", "dyn_spg_ts pssh", 9920),
        ("1.4", "un_adv_final (Hu_avg)", "dyn_spg_ts un_adv", 9758),
        ("1.4", "vn_adv_final (Hv_avg)", "dyn_spg_ts un_adv", 9868),
    ]
    measurements: list[dict[str, Any]] = []
    for subrow, name, gate_name, expected_n in specifications:
        if name not in captured:
            raise SystemExit(f"required inherited report missing: {name}")
        metric = _metric(*captured[name][0], expected_n)
        gate_status = _classify_metric(metric, gate_name)
        bar = bar_gate.class_bar_for(gate_name)
        metric.update(
            {
                "subrow": subrow,
                "field": name,
                "arithmetic_class_bar": bar,
                "fidelity_bar_row": gate_name,
                "gate_status": gate_status,
                "status": "MATCHED" if gate_status == "AT BAR" else "DIVERGED",
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

    controls = _controls(captured["zu_frc"][0], 9758)
    if not all(
        controls[key]
        for key in (
            "identical_array_zero",
            "planted_identity_flips_campaign_gate",
            "actual_binding_perturbation_changes_score",
            "zero_shift_is_best",
        )
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
        "stp_dump_07_dynspg_u.bin",
        "stp_dump_07_dynspg_v.bin",
        "stp_dump_07_dynspg_ub.bin",
        "stp_dump_07_dynspg_vb.bin",
    ]
    run_dir = Path(inherited.RUN_DIR).resolve()
    input_names = [
        "mesh_mask.nc",
        inherited.RESTART_FILE,
        "ocean.output",
        "namelist_cfg",
        "namelist_ref",
        "output.namelist.dyn",
        "nemo",
    ]
    provenance_paths = {
        "wrapper": Path(__file__).resolve(),
        "inherited_probe": script_dir / "spg_substep_chain.py",
        "production_ocean_model": Path(ocean_model_module.__file__).resolve(),
        "production_barotropic": Path(barotropic_module.__file__).resolve(),
        "production_bridge": Path(bridge_module.__file__).resolve(),
        "production_dino_card": Path(dino_module.__file__).resolve(),
        "fidelity_bar_gate": Path(bar_gate.__file__).resolve(),
        "nemo_stpmlf": oracle / "cfgs/DINO/MY_SRC/stpmlf.F90",
        "nemo_dynspg_ts": oracle / "cfgs/DINO/MY_SRC/dynspg_ts.F90",
    }
    effective_config = runtime_capture["effective_dino_config"]
    model_config = runtime_capture["model_config"]
    barotropic_config = model_config.barotropic
    effective_config_repr = repr(effective_config)
    model_config_repr = repr(model_config)
    barotropic_settings = {
        name: _json_setting(getattr(barotropic_config, name))
        for name in (
            "n_barotropic_substeps",
            "barotropic_time_filter",
            "barotropic_coriolis",
            "barotropic_face_depth",
            "barotropic_seed_face_depth",
        )
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round1-v2",
        "session_id": os.environ.get("CODEX_SESSION_ID", "unset"),
        "git": {
            "commit": git_sha,
            "clean_before": dirty_before == "",
            "clean_after": dirty_after == "",
        },
        "lane": EXPECTED_LANE,
        "run_dir": str(run_dir),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "e3t_mode": os.environ.get("LEGOESM_NEMO_E3T"),
        "recipe": "nemo_dino_kamm_mlf",
        "effective_recipe_overrides": {
            "lon_west_deg": _json_setting(effective_config.lon_west_deg),
            "lon_east_deg": _json_setting(effective_config.lon_east_deg),
            "sill_lon_m_deg": _json_setting(effective_config.sill_lon_m_deg),
        },
        "effective_config_repr_sha256": hashlib.sha256(
            effective_config_repr.encode()
        ).hexdigest(),
        "derived_model_config_repr_sha256": hashlib.sha256(
            model_config_repr.encode()
        ).hexdigest(),
        "derived_barotropic_settings": barotropic_settings,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "cpu_device": str(jax.devices("cpu")[0]),
        },
        "bar_policy": {
            "classifier": "fidelity_bar_gate.classify",
            "correlation_min": bar_gate.BAR_CORR,
            "mean_abs_ratio_epsilon": bar_gate.BAR_RATIO_EPS,
            "pointwise_per_element": bar_gate.BAR_POINTWISE,
            "accumulating_per_element": bar_gate.BAR_ACCUMULATING,
        },
        "row_1_status": row_status,
        "first_diverged_subrow": first_diverged,
        "ordered_subrows": subrows,
        "unmeasured": {
            "ssh_frc": (
                "NEMO dump exists, but the production recipe has no "
                "F_slow_eta because freshwater_closure is inactive"
            )
        },
        "measurements": measurements,
        "controls": controls,
        "provenance_sha256": {
            name: _sha256(path) for name, path in provenance_paths.items()
        },
        "dump_sha256": {
            name: _sha256(run_dir / name) for name in dump_names
        },
        "run_input_sha256": {
            name: _sha256(run_dir / name) for name in input_names
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

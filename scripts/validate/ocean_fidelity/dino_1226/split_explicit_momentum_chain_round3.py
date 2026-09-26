#!/usr/bin/env python3
"""Post-#1695 continuation of the day-180 slow-momentum forcing peel.

This wrapper reruns the committed round-2 operand path, captures its raw
faithful-tail operands, and adds only the pair/time-level measurements frozen
in ``PREREG_split_explicit_momentum_chain_round3.md``.  It reads existing
NEMO dumps and never builds or executes NEMO.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import kamm_twin_90d as twin
import numpy as np
import split_explicit_momentum_chain_round2 as round2
from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.vertical import compute_layer_thickness

UPSTREAM_BUILD = "73ad090d411027ae3815f6ae8ea7b7ff0f85f0dc"
UPSTREAM_DEFAULT = "f862e1554911f0901ab15b7ede80de7a74d2f1ac"
LOCAL_BUILD = "8327ffb10bc"
LOCAL_DEFAULT = "7061c59af51"
ORDERED = (
    "kinetic_energy_gradient",
    "vertical_advection",
    "vorticity",
    "lateral_friction",
    "pressure_gradient",
    "cor_removal",
    "drag",
    "wind",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _is_ancestor(repo: Path, commit: str, head: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, head],
        cwd=repo,
        check=False,
    )
    return result.returncode == 0


def _correction_score(prediction, residual) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    scale = _rms(residual)
    if scale <= 0.0:
        raise RuntimeError("zero faithful residual has no correction score")
    if prediction.size > 1 and np.std(prediction) > 0.0:
        corr = float(np.corrcoef(prediction, -residual)[0, 1])
    else:
        # A literal zero correction is the registered negative control: it has
        # no directional relationship to a nonzero residual.
        corr = 0.0
    prediction_error = _rms(prediction + residual) / scale
    reduction = 1.0 - _rms(residual + prediction) / scale
    if prediction_error <= 0.10 and corr >= 0.99 and reduction >= 0.90:
        verdict = "CONFIRMS_LDF_BEFORE_SOURCE"
    elif prediction_error >= 0.90 and corr <= 0.20 and reduction <= 0.10:
        verdict = "REFUTES_LDF_BEFORE_SOURCE"
    else:
        verdict = "UNRESOLVED_LDF_BEFORE_SOURCE"
    return {
        "prediction_normalized_error": prediction_error,
        "prediction_correlation": corr,
        "corrected_residual_reduction": reduction,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty_before = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    if dirty_before:
        raise SystemExit("clean worktree required:\n" + dirty_before)
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU + JAX x64 are required")

    imported = {
        "upstream_build": UPSTREAM_BUILD,
        "upstream_default": UPSTREAM_DEFAULT,
        "local_build": LOCAL_BUILD,
        "local_default": LOCAL_DEFAULT,
        "local_build_is_ancestor": _is_ancestor(repo, LOCAL_BUILD, head),
        "local_default_is_ancestor": _is_ancestor(repo, LOCAL_DEFAULT, head),
    }
    if not (imported["local_build_is_ancestor"]
            and imported["local_default_is_ancestor"]):
        raise SystemExit("#1695 cherry-picks are not ancestors of measurement HEAD")

    build_default = inspect.signature(twin._build_twin_state).parameters[
        "bridge_before_stress_tpoint"
    ].default
    run_default = inspect.signature(twin.run_twin).parameters[
        "bridge_before_stress_tpoint"
    ].default
    parsed_default = twin._parse_args(["nemo_dino_kamm_mlf", "unused.npz"])
    parsed_legacy = twin._parse_args([
        "nemo_dino_kamm_mlf", "unused.npz",
        "--bridge-before-stress-legacy-u-as-t",
    ])
    parsed_legacy_euler = twin._parse_args([
        "nemo_dino_kamm_mlf", "unused.npz", "--legacy-euler-start",
    ])
    selector_receipt = {
        "build_default": bool(build_default),
        "run_default": bool(run_default),
        "cli_default": bool(parsed_default.bridge_before_stress_tpoint),
        "legacy_opt_in": bool(parsed_legacy.bridge_before_stress_tpoint),
        "legacy_euler_bridge_before": bool(parsed_legacy_euler.bridge_before),
        "legacy_euler_stress": bool(
            parsed_legacy_euler.bridge_before_stress_tpoint
        ),
    }
    if selector_receipt != {
        "build_default": True,
        "run_default": True,
        "cli_default": True,
        "legacy_opt_in": False,
        "legacy_euler_bridge_before": False,
        "legacy_euler_stress": False,
    }:
        raise SystemExit(f"#1695 selector/default receipt failed: {selector_receipt}")

    metric_calls: list[dict[str, Any]] = []
    arm_calls: list[dict[str, Any]] = []
    real_metric = round2._term_metric
    real_arm = round2._run_arm

    def capture_metric(lego, nemo, mask, expected_n, total_residual):
        metric = real_metric(lego, nemo, mask, expected_n, total_residual)
        metric_calls.append({
            "lego": np.asarray(lego).copy(),
            "nemo": np.asarray(nemo).copy(),
            "mask": np.asarray(mask, dtype=bool).copy(),
            "expected_n": int(expected_n),
            "residual": np.asarray(total_residual).copy(),
            "metric": metric,
        })
        return metric

    def capture_arm(state, sf, geometry, z_coord, config):
        result = real_arm(state, sf, geometry, z_coord, config)
        arm_calls.append({
            "state": state,
            "surface_forcing": sf,
            "geometry": geometry,
            "z_coord": z_coord,
            "config": config,
        })
        return result

    round2._term_metric = capture_metric
    round2._run_arm = capture_arm
    old_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory(prefix="dino-split-r3-") as tmpdir:
            base_path = Path(tmpdir) / "round2.json"
            sys.argv = [str(Path(round2.__file__).resolve()), "--output", str(base_path)]
            status = round2.main()
            if status != 0:
                raise SystemExit(f"round-2 base probe returned {status}")
            base = json.loads(base_path.read_text())
            base_sha256 = _sha256(base_path)
    finally:
        sys.argv = old_argv
        round2._term_metric = real_metric
        round2._run_arm = real_arm

    if len(metric_calls) != 34 or len(arm_calls) != 3:
        raise SystemExit(
            f"round-2 capture sequence changed: metrics={len(metric_calls)} "
            f"arms={len(arm_calls)}"
        )

    # Calls 0..15 are legacy U/V terms, 16..17 faithful wind against the
    # original residual, and 18..33 faithful U/V terms in ORDERED sequence.
    faithful_calls: dict[str, dict[str, Any]] = {}
    residual_values = None
    term_errors: dict[str, np.ndarray] = {}
    for index, name in enumerate(ORDERED):
        call = metric_calls[18 + 2 * index]
        if call["expected_n"] != round2.EXPECTED_U:
            raise SystemExit(f"faithful U capture population changed at {name}")
        values = call["residual"][call["mask"]]
        if residual_values is None:
            residual_values = values
        elif not np.array_equal(values, residual_values):
            raise SystemExit("faithful U residual changed between term calls")
        term_errors[name] = (call["lego"] - call["nemo"])[call["mask"]]
        faithful_calls[name] = call
        if call["metric"] != base["faithful_term_metrics"]["u"][name]:
            raise SystemExit(f"captured metric disagrees with base artifact: {name}")
    assert residual_values is not None

    fixed_wind = {
        "status": "FIXED-BY-#1695",
        "counterfactual": base["counterfactual"]["u"],
        "legacy_total": base["total_metrics"]["legacy_u"],
        "faithful_total": base["total_metrics"]["faithful_u"],
    }
    expected_wind = {
        "prediction_normalized_error": 0.007130447331820113,
        "prediction_correlation": 0.999974655141223,
        "corrected_residual_reduction": 0.9928695526681799,
        "verdict": "CONFIRMS_BRIDGE_WIND_SOURCE",
    }
    if fixed_wind["counterfactual"] != expected_wind:
        raise SystemExit(
            "post-cherry-pick wind receipt changed from admitted round 2: "
            f"{fixed_wind['counterfactual']}"
        )

    dispositions = []
    stop_term = None
    for name in ORDERED:
        metric = base["faithful_term_metrics"]["u"][name]
        attribution = metric["attribution"]
        if name == "kinetic_energy_gradient" and metric["campaign_gate"] == "NEAR-CLASS":
            status_text = "CLEARED_NEAR_CLASS"
        elif attribution["verdict"] == "REFUTES_CARRY":
            status_text = "CLEARED_REFUTES_CARRY"
        else:
            status_text = "OPEN_" + attribution["verdict"]
            stop_term = name
        dispositions.append({
            "term": name,
            "campaign_gate": metric["campaign_gate"],
            "attribution": attribution,
            "ordered_status": status_text,
        })
        if stop_term is not None:
            break
    if stop_term != "lateral_friction":
        raise SystemExit(f"first unresolved source-ordered term changed: {stop_term}")

    pair_scores = {}
    leading = term_errors[stop_term]
    for name in ORDERED:
        if name == stop_term:
            continue
        pair_scores[f"{stop_term}+{name}"] = round2._attribute_error(
            leading + term_errors[name], residual_values
        )
    vorticity_cor_pair = round2._attribute_error(
        term_errors["vorticity"] + term_errors["cor_removal"], residual_values
    )

    faithful_arm = arm_calls[1]
    state = faithful_arm["state"]
    sf = faithful_arm["surface_forcing"]
    geometry = faithful_arm["geometry"]
    z_coord = faithful_arm["z_coord"]
    config = faithful_arm["config"]
    centred_sf = sf._replace(
        tau_x=0.5 * (state.tau_x_prev + sf.tau_x),
        tau_y=0.5 * (state.tau_y_prev + sf.tau_y),
    )
    ldf_state = (
        state.T_before.data,
        state.S_before.data,
        state.u_before.data,
        state.v_before.data,
    )
    model = LatLonCGridOceanModel(geometry, z_coord, config)
    with jax.disable_jit():
        _, before_diag = model.tendencies_with_diagnostics(
            state,
            surface_forcing=centred_sf,
            dt=round2.DT,
            ldf_state=ldf_state,
        )
    before_ldf_3d = np.asarray(
        before_diag.Ah_lap_u.data
        + before_diag.Bh_bilap_u.data
        + before_diag.Cs_smag_u.data
        + before_diag.Cl_leith_u.data
    )
    h_k = compute_layer_thickness(
        state.eta.data,
        state.H_bathy.data,
        z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_u = np.asarray(min_cell_to_uface(h_k))
    before_ldf = round2._u_to_nemo(round2._live_depth_mean(
        before_ldf_3d, h_u, np.asarray(state.u_mask.data)
    ))
    current_call = faithful_calls["lateral_friction"]
    if before_ldf.shape != current_call["lego"].shape:
        raise SystemExit(
            f"LDF BEFORE shape changed: {before_ldf.shape} vs "
            f"{current_call['lego'].shape}"
        )
    before_metric = real_metric(
        before_ldf,
        current_call["nemo"],
        current_call["mask"],
        current_call["expected_n"],
        current_call["residual"],
    )
    prediction_values = (
        before_ldf - current_call["lego"]
    )[current_call["mask"]]
    before_counterfactual = _correction_score(prediction_values, residual_values)

    perfect = _correction_score(-residual_values, residual_values)
    zero = _correction_score(np.zeros_like(residual_values), residual_values)
    correction_controls = {
        "perfect_confirms": perfect["verdict"] == "CONFIRMS_LDF_BEFORE_SOURCE",
        "zero_refutes": zero["verdict"] == "REFUTES_LDF_BEFORE_SOURCE",
        "perfect_score": perfect,
        "zero_score": zero,
    }
    if not (correction_controls["perfect_confirms"]
            and correction_controls["zero_refutes"]):
        raise SystemExit(f"correction classifier controls failed: {correction_controls}")

    dirty_after = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    if dirty_after:
        raise SystemExit("worktree changed during measurement:\n" + dirty_after)

    prereg = repo / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round3.md"
    provenance_paths = {
        "probe": Path(__file__).resolve(),
        "preregistration": prereg,
        "round2_probe": Path(round2.__file__).resolve(),
        "round2_preregistration": (
            repo / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round2.md"
        ),
        "twin_harness": Path(twin.__file__).resolve(),
        "zad_flux_fix_test": repo / "tests/ocean/unit/test_vertical_momentum_scheme.py",
        "zad_mask_fix_test": repo / "tests/ocean/unit/test_zad_bottom_face_mask.py",
        "ldf_state_test": repo / "tests/ocean/unit/test_tendencies_diagnostics_ldf_state.py",
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round3-v1",
        "session_id": os.environ.get("CODEX_SESSION_ID", "unset"),
        "git": {"commit": head, "clean_before": True, "clean_after": True},
        "runtime": base["runtime"],
        "lane": base["lane"],
        "kt": base["kt"],
        "imported_pr1695": imported,
        "selector_receipt": selector_receipt,
        "merge_order_dependency": (
            "PR #1695 must land before this branch with local patch-equivalent "
            "commits dropped, or this branch lands first and #1695 omits/rebases "
            "73ad090d411+f862e155491; never merge a second implementation"
        ),
        "fixed_wind": fixed_wind,
        "base_round2_artifact_sha256": base_sha256,
        "base_round2_term_error_closure": base["term_error_sum_closure"],
        "ordered_dispositions": dispositions,
        "first_open_term": stop_term,
        "faithful_leading_pair_scores": pair_scores,
        "vorticity_plus_cor_removal": vorticity_cor_pair,
        "lateral_friction_before": {
            "nemo_source": "stpmlf.F90:319-322; dynldf.F90:69-119, Kbb",
            "lego_hook": "tendencies_with_diagnostics(..., ldf_state=BEFORE)",
            "term_metric": before_metric,
            "forcing_counterfactual": before_counterfactual,
        },
        "controls": {
            "round2": base["controls"],
            "correction_classifier": correction_controls,
        },
        "ordered_continuation": {
            "row_1_1": "OPEN_LATERAL_FRICTION_UNRESOLVED",
            "row_1_2": "ORDERED_BLOCKED",
            "row_1_3": "ORDERED_BLOCKED",
            "rows_2_to_6": "ORDERED_BLOCKED",
        },
        "slot": "NOT_ALLOCATED_EXISTING_DUMPS_AND_PRODUCTION_HOOK_ONLY",
        "provenance_sha256": {
            name: _sha256(path) for name, path in provenance_paths.items()
        },
        "dump_sha256": base["dump_sha256"],
        "run_input_sha256": base["run_input_sha256"],
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    print("POST-#1695 ROW 1.1 CONTINUATION")
    print(
        "  wind: FIXED-BY-#1695  removal="
        f"{fixed_wind['counterfactual']['corrected_residual_reduction']:.9f}"
    )
    for item in dispositions:
        score = item["attribution"]
        print(
            f"  {item['term']}: {item['ordered_status']} "
            f"corr={score['correlation_with_total_residual']:+.6f} "
            f"gain={score['rms_gain']:.6f} "
            f"removal={score['residual_removal']:.6f}"
        )
    print(
        "  LDF BEFORE counterfactual: "
        f"{before_counterfactual['verdict']} "
        f"corr={before_counterfactual['prediction_correlation']:+.6f} "
        f"reduction={before_counterfactual['corrected_residual_reduction']:.6f}"
    )
    print(f"artifact={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

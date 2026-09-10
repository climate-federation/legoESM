#!/usr/bin/env python3
"""Held-oracle continuation of split-explicit row 1 (round 6).

Runs the committed ``spg_substep_chain`` twice: first with only the assembled
slow forcing held to NEMO, then with the BEFORE barotropic seed held as well.
No NEMO process or writer is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
import jax.numpy as jnp
import jaxlib
import numpy as np

import fidelity_bar_gate as bar_gate
import split_explicit_momentum_chain_round1 as base
import spg_substep_chain as inherited
import legoesm.ocean.dynamics.barotropic_latlon_cgrid as barotropic_module
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocean_model_module
import legoesm.ocean.experiments.dino as dino_module
import legoesm.ocean.fidelity.nemo_state_bridge as bridge_module


EXPECTED_LANE = "d180"
SPECS = [
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _u_face(a: np.ndarray) -> np.ndarray:
    """DINO periodic east-face convention, bridge-identical."""
    return np.concatenate([a[:, -1:], a], axis=1)


def _v_face(a: np.ndarray) -> np.ndarray:
    """DINO north-face convention, bridge-identical."""
    return np.concatenate([np.zeros_like(a[:1]), a], axis=0)


def _metric_record(
    arm: str,
    subrow: str,
    field: str,
    gate_name: str | None,
    expected_n: int,
    sample: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    metric = base._metric(*sample, expected_n)
    status = base._classify_metric(metric, gate_name)
    metric.update(
        arm=arm,
        subrow=subrow,
        field=field,
        fidelity_bar_row=gate_name,
        arithmetic_class_bar=bar_gate.class_bar_for(gate_name),
        gate_status=status,
        status="MATCHED" if status == "AT BAR" else "DIVERGED",
    )
    return metric


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-ssh-forcing", action="store_true")
    parser.add_argument("--with-continuity-localization", action="store_true")
    args = parser.parse_args()
    if os.environ.get("DINO_1226_LANE") != EXPECTED_LANE:
        raise SystemExit("DINO_1226_LANE=d180 is required")
    if os.environ.get("LEGOESM_NEMO_E3T") != "both":
        raise SystemExit("LEGOESM_NEMO_E3T=both is required")
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU + JAX x64 are required")

    root = Path(__file__).resolve().parents[4]
    production_imports = {
        "production_barotropic": Path(barotropic_module.__file__).resolve(),
        "production_ocean_model": Path(ocean_model_module.__file__).resolve(),
        "production_dino_card": Path(dino_module.__file__).resolve(),
        "production_bridge": Path(bridge_module.__file__).resolve(),
    }
    escaped = {
        name: str(path)
        for name, path in production_imports.items()
        if not path.is_relative_to(root)
    }
    if escaped:
        raise SystemExit(
            "production imports escaped measured checkout; put this checkout's "
            f"package roots first on PYTHONPATH: {escaped}"
        )
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty_before = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True)
    if dirty_before:
        raise SystemExit("clean worktree required\n" + dirty_before)

    run = Path(inherited.RUN_DIR).resolve()
    jpi, jpj, _jpk, hls, _icycle, _nn_e = inherited._read_dims(str(run))
    nemo = {
        "fu": inherited._load_interior(str(run / "spg_dump_zu_frc.bin"), 52, 199),
        "fv": inherited._load_interior(str(run / "spg_dump_zv_frc.bin"), 52, 199),
        "eta": inherited._load_full(str(run / "spg_dump_sshn_e_init.bin"), jpi, jpj, hls),
        "u": inherited._load_full(str(run / "spg_dump_un_e_init.bin"), jpi, jpj, hls),
        "v": inherited._load_full(str(run / "spg_dump_vn_e_init.bin"), jpi, jpj, hls),
        "feta": inherited._load_full(str(run / "spg_dump_ssh_frc.bin"), jpi, jpj, hls),
    }
    held = {"fu": _u_face(nemo["fu"]), "fv": _v_face(nemo["fv"]),
            "u": _u_face(nemo["u"]), "v": _v_face(nemo["v"]),
            # NEMO subtracts ssh_frc; legoESM adds F_slow_eta.
            "feta": -nemo["feta"]}

    real_solver = ocean_model_module.barotropic_substeps_latlon_cgrid
    real_inherited_solver = inherited.barotropic_substeps_latlon_cgrid
    real_davg = barotropic_module._depth_average_to_faces
    real_report = inherited._report
    real_builder = inherited.dino_lat_lon_model_config
    arm_captures: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}
    runtime: dict[str, Any] = {}
    entry_counts: dict[str, int] = {}
    step_dt: dict[str, float] = {}
    forcing_receipts: dict[str, Any] = {}
    restoration_receipts: dict[str, Any] = {}

    arms = [("forcing_only", False, False), ("forcing_and_seed", True, False)]
    if args.with_ssh_forcing:
        arms.append(("forcing_seed_and_ssh", True, True))
    for arm, hold_seed, hold_eta in arms:
        reports: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        calls = 0
        originals: list[tuple[np.ndarray, np.ndarray]] = []

        def held_solver(state, dt_s, n_substeps, grid, z_coord, config, **kw):
            nonlocal calls
            kw = dict(kw)
            if kw.get("eta_init") is not None:
                calls += 1
                step_dt.setdefault(arm, float(dt_s))
                originals.append((np.asarray(kw["F_slow_u"]), np.asarray(kw["F_slow_v"])))
                kw["F_slow_u"] = jnp.asarray(held["fu"], dtype=kw["F_slow_u"].dtype)
                kw["F_slow_v"] = jnp.asarray(held["fv"], dtype=kw["F_slow_v"].dtype)
                if hold_eta:
                    kw["F_slow_eta"] = jnp.asarray(held["feta"], dtype=kw["eta_init"].dtype)
            return real_solver(state, dt_s, n_substeps, grid, z_coord, config, **kw)

        def held_davg(*dargs, **dkw):
            if hold_seed and "seed_face_depth" in dkw:
                dtype = np.asarray(dargs[0]).dtype
                return jnp.asarray(held["u"], dtype=dtype), jnp.asarray(held["v"], dtype=dtype)
            return real_davg(*dargs, **dkw)

        def collecting_report(name, lego, nemo_array, mask):
            reports.setdefault(name, []).append(
                (np.asarray(lego), np.asarray(nemo_array), np.asarray(mask, dtype=bool)))
            return real_report(name, lego, nemo_array, mask)

        def collecting_builder(*bargs, **bkw):
            result = real_builder(*bargs, **bkw)
            runtime[arm] = {"effective": repr(bargs[1]), "model": repr(result[0])}
            return result

        ocean_model_module.barotropic_substeps_latlon_cgrid = held_solver
        inherited.barotropic_substeps_latlon_cgrid = held_solver
        barotropic_module._depth_average_to_faces = held_davg
        inherited._report = collecting_report
        inherited.dino_lat_lon_model_config = collecting_builder
        try:
            code = inherited.main()
        finally:
            ocean_model_module.barotropic_substeps_latlon_cgrid = real_solver
            inherited.barotropic_substeps_latlon_cgrid = real_inherited_solver
            barotropic_module._depth_average_to_faces = real_davg
            inherited._report = real_report
            inherited.dino_lat_lon_model_config = real_builder
        restoration_receipts[arm] = {
            "ocean_solver_restored": (
                ocean_model_module.barotropic_substeps_latlon_cgrid is real_solver
            ),
            "inherited_solver_restored": (
                inherited.barotropic_substeps_latlon_cgrid is real_inherited_solver
            ),
            "depth_average_restored": (
                barotropic_module._depth_average_to_faces is real_davg
            ),
            "report_restored": inherited._report is real_report,
            "config_builder_restored": inherited.dino_lat_lon_model_config is real_builder,
        }
        if not all(restoration_receipts[arm].values()):
            raise SystemExit(f"{arm}: monkeypatch restoration failed")
        if code != 0 or calls < 3:
            raise SystemExit(f"{arm}: inherited exit={code}, seeded held entries={calls}")
        arm_captures[arm] = reports
        entry_counts[arm] = calls
        forcing_receipts[arm] = {
            "first_original_u_sha256": hashlib.sha256(originals[0][0].tobytes()).hexdigest(),
            "first_original_v_sha256": hashlib.sha256(originals[0][1].tobytes()).hexdigest(),
            "held_u_sha256": hashlib.sha256(held["fu"].tobytes()).hexdigest(),
            "held_v_sha256": hashlib.sha256(held["fv"].tobytes()).hexdigest(),
        }

    measurements = []
    for arm, reports in arm_captures.items():
        for subrow, field, gate_name, expected_n in SPECS:
            if field not in reports:
                raise SystemExit(f"{arm}: missing report {field}")
            measurements.append(
                _metric_record(arm, subrow, field, gate_name, expected_n, reports[field][0])
            )

    fu_mask = arm_captures["forcing_only"]["zu_frc"][0][2]
    fv_mask = arm_captures["forcing_only"]["zv_frc"][0][2]
    held_forcing_metrics = {
        "u": base._metric(held["fu"][:, 1:], nemo["fu"], fu_mask, 9758),
        "v": base._metric(held["fv"][1:], nemo["fv"], fv_mask, 9868),
    }
    if any(m["normalized_rms_error"] != 0.0 for m in held_forcing_metrics.values()):
        raise SystemExit("held forcing reconstruction is not exact")
    held_ssh_metric = None
    if args.with_ssh_forcing:
        eta_mask = arm_captures["forcing_seed_and_ssh"]["sshn_e_init"][0][2]
        eta_n = int(np.asarray(eta_mask, dtype=bool).sum())
        eta_diff = np.asarray(held["feta"] + nemo["feta"])[eta_mask]
        held_ssh_metric = {
            "n": eta_n,
            "nemo_rms": float(np.sqrt(np.mean(nemo["feta"][eta_mask] ** 2))),
            "max_abs_difference": float(np.max(np.abs(eta_diff))),
            "exact_zero_field_receipt": bool(
                eta_n == 9920
                and np.count_nonzero(nemo["feta"][eta_mask]) == 0
                and np.count_nonzero(eta_diff) == 0
            ),
        }
        if not held_ssh_metric["exact_zero_field_receipt"]:
            raise SystemExit("held signed SSH forcing reconstruction is not exact")

    continuity_localization = None
    if args.with_continuity_localization:
        reports = arm_captures["forcing_and_seed"]
        eta_seed_l, eta_seed_n, eta_mask = reports["sshn_e_init"][0]
        eta_s1_l, eta_s1_n, eta_mask_s1 = reports["ssh_substep1"][0]
        if not np.array_equal(eta_mask, eta_mask_s1):
            raise SystemExit("seed/substep SSH masks differ")
        dt_e = step_dt["forcing_and_seed"]
        div_l = (eta_seed_l - eta_s1_l) / dt_e - nemo["feta"]
        div_n = (eta_seed_n - eta_s1_n) / dt_e - nemo["feta"]
        div_metric = base._metric(div_l, div_n, eta_mask, 9920)
        div_gate = base._classify_metric(div_metric, "dyn_spg_ts pssh")
        replay_residual = (
            (eta_s1_l - eta_s1_n)
            - (eta_seed_l - eta_seed_n)
            + dt_e * (div_l - div_n)
        )
        replay_max = float(np.max(np.abs(replay_residual[eta_mask])))
        ua_n = inherited._load_full(
            str(run / "cor2d_dump_ua_e_in_substep1.bin"), jpi, jpj, hls)
        va_n = inherited._load_full(
            str(run / "cor2d_dump_va_e_in_substep1.bin"), jpi, jpj, hls)
        u_seed_l, _u_seed_n, u_mask = reports["un_e_init"][0]
        v_seed_l, _v_seed_n, v_mask = reports["vn_e_init"][0]
        ua_metric = base._metric(u_seed_l, ua_n, u_mask, 9758)
        va_metric = base._metric(v_seed_l, va_n, v_mask, 9868)
        exact_seed_statuses = {
            item["field"]: item["gate_status"]
            for item in measurements
            if item["arm"] == "forcing_and_seed" and item["subrow"] == "1.2"
        }
        continuity_localization = {
            "dt_s": dt_e,
            "midstep_u": {**ua_metric,
                          "gate_status": base._classify_metric(ua_metric, None)},
            "midstep_v": {**va_metric,
                          "gate_status": base._classify_metric(va_metric, None)},
            "implied_flux_divergence": {**div_metric, "gate_status": div_gate},
            "forward_replay_max_abs_residual": replay_max,
            "forward_replay_role": "ALGEBRAIC_CLOSURE_ONLY_NOT_CONTROL",
            "runtime_exact_seed_gate_statuses": exact_seed_statuses,
            "status": (
                "CONFIRMED_CONTINUITY_FLUX_COMPOSITION"
                if (set(exact_seed_statuses.values()) == {"AT BAR"}
                    and base._classify_metric(ua_metric, None) == "AT BAR"
                    and base._classify_metric(va_metric, None) == "AT BAR"
                    and div_gate != "AT BAR")
                else "REFUTED_OR_UNRESOLVED"
            ),
        }

    controls = base._controls(arm_captures["forcing_only"]["zu_frc"][0], 9758)
    required_controls = ("identical_array_zero", "planted_identity_flips_campaign_gate",
                         "actual_binding_perturbation_changes_score", "zero_shift_is_best")
    if not all(controls[k] for k in required_controls):
        raise SystemExit(f"controls failed: {controls}")

    literal_seed = [m for m in measurements if m["arm"] == "forcing_only" and m["subrow"] == "1.2"]
    first_stop = "1.2" if any(m["status"] != "MATCHED" for m in literal_seed) else None
    primary = [m for m in measurements if m["arm"] == "forcing_only"]
    for subrow in ("1.3", "1.4"):
        if first_stop is None and any(m["status"] != "MATCHED" for m in primary if m["subrow"] == subrow):
            first_stop = subrow

    oracle = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
    provenance = {
        "wrapper": Path(__file__).resolve(),
        "inherited_probe": Path(inherited.__file__).resolve(),
        "round1_scorer": Path(base.__file__).resolve(),
        "bar_gate": Path(bar_gate.__file__).resolve(),
        **production_imports,
        "nemo_dynspg_ts": oracle / "cfgs/DINO/MY_SRC/dynspg_ts.F90",
        "nemo_stpmlf": oracle / "cfgs/DINO/MY_SRC/stpmlf.F90",
    }
    dump_names = sorted({
        "spg_dump_zu_frc.bin", "spg_dump_zv_frc.bin", "spg_dump_ssh_frc.bin",
        "spg_dump_sshn_e_init.bin", "spg_dump_un_e_init.bin", "spg_dump_vn_e_init.bin",
        "spg_dump_ssh_substep1.bin", "spg_dump_ub_substep1.bin", "spg_dump_vb_substep1.bin",
        "spg_dump_puu_b_final.bin", "spg_dump_pvv_b_final.bin", "spg_dump_pssh_final.bin",
        "spg_dump_un_adv_final.bin", "spg_dump_vn_adv_final.bin",
        "cor2d_dump_ua_e_in_substep1.bin", "cor2d_dump_va_e_in_substep1.bin",
        "stp_dump_07_dynspg_u.bin", "stp_dump_07_dynspg_v.bin",
        "stp_dump_07_dynspg_ub.bin", "stp_dump_07_dynspg_vb.bin",
    })
    dirty_after = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True)
    if dirty_after:
        raise SystemExit("worktree changed during measurement")
    receipt = {
        "schema": ("dino-split-explicit-momentum-chain-round8-v1"
                   if args.with_continuity_localization
                   else ("dino-split-explicit-momentum-chain-round7-v1"
                         if args.with_ssh_forcing
                         else "dino-split-explicit-momentum-chain-round6-v1")),
        "session_id": os.environ.get("CODEX_SESSION_ID", "unset"),
        "git": {"commit": git_sha, "clean_before": True, "clean_after": True},
        "lane": EXPECTED_LANE,
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "e3t_mode": os.environ.get("LEGOESM_NEMO_E3T"),
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                    "numpy": np.__version__, "device": str(jax.devices("cpu")[0])},
        "entry_counts": entry_counts,
        "restoration_receipts": restoration_receipts,
        "imported_production_paths": {
            name: str(path) for name, path in production_imports.items()
        },
        "forcing_receipts": forcing_receipts,
        "held_forcing_metrics": held_forcing_metrics,
        "held_signed_ssh_forcing_metric": held_ssh_metric,
        "continuity_localization": continuity_localization,
        "measurements": measurements,
        "literal_ordered_stop": first_stop,
        "later_rows": {str(i): "ORDERED-BLOCKED" if first_stop else "ELIGIBLE" for i in range(2, 7)},
        "controls": controls,
        "runtime_config_sha256": {
            arm: {key: hashlib.sha256(value.encode()).hexdigest() for key, value in values.items()}
            for arm, values in runtime.items()
        },
        "provenance_sha256": {name: _sha256(path) for name, path in provenance.items()},
        "dump_sha256": {name: _sha256(run / name) for name in dump_names},
        "input_sha256": {name: _sha256(run / name) for name in
                         ("mesh_mask.nc", inherited.RESTART_FILE, "ocean.output",
                          "namelist_cfg", "namelist_ref", "output.namelist.dyn", "nemo")},
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ROUND6 literal_ordered_stop={first_stop} artifact={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

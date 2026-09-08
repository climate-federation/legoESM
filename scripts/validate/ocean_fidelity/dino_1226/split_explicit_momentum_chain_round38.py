#!/usr/bin/env python3
"""Replay the day-180 ZAD operand peel from retained round-37 data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

import kamm_twin_90d as twin
import split_explicit_momentum_chain_round35 as r35
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    interp_cell_to_uface,
    interp_cell_to_vface,
)
from legoesm.ocean.vertical import nemo_advective_vertical_momentum_advection
from zu_frc_term_walk import _load_full_3d


ROUND37_SHA = "053bafb547e66aa09a2f0a357c8bce3da5ba808689e3654b726e77d7f0ee0f23"
CAPTURE_SHA = "96e11403c547c2bfa59f3871de7a099db82a66127d952006433ab34a55f36048"
EXPECTED = {
    "wzv_dump_ww_call1.bin": "defad5014cc7210dd52d6373dafd46856471afedb892267cef30232fdd9baeb2",
    "wzv_dump_ww_call2.bin": "895a141385f33775c4df7fc127a4beadf2e8e496f68a46624931cf8eb1487634",
    "zad_dump_du.bin": "9899d5cdfcb3bb70f5910f773259a7e97c5b388ffcbe4cf4c05c0981929e26da",
    "zad_dump_dv.bin": "b053c7d46c36dbd52d4e2da117e700ef5516ad1b519b525b62ee11aa49a7bf4e",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
}
JPI, JPJ, JPK, HLS = 56, 203, 36, 2


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True,
    ).strip()


def _load_capture(directory: Path, metadata: dict, component: str) -> np.ndarray:
    name = f"{component}_vertical_advection.bin"
    info = metadata["files"][name]
    path = directory / name
    if (_sha(path) != info["sha256"]
            or path.stat().st_size != info["size_bytes"]):
        raise SystemExit(f"capture file admission failed: {name}")
    return np.fromfile(path, dtype="<f8").reshape(info["shape"])


def _load_ww(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    expected = JPI * JPJ * JPK
    if raw.size != expected:
        raise SystemExit(
            f"{path}: ww is a genuine full-halo jpk stream; "
            f"expected {expected} values, got {raw.size}")
    full = raw.reshape(JPK, JPJ, JPI)
    return np.moveaxis(full[:, HLS:-HLS, HLS:-HLS], 0, -1)


def _metric(candidate, oracle, mask, gate="dyn_zdf (momentum implicit vertical solve)"):
    return r35._metric(np.asarray(candidate), np.asarray(oracle),
                       np.asarray(mask, dtype=bool), gate)


def _crop(component: str, value) -> np.ndarray:
    array = np.asarray(value)
    return array[:, 1:, :35] if component == "u" else array[1:, :, :35]


def _direct(velocity, w, h_face, active, grid, component: str,
            bottom_mode: str):
    area_w = grid.area_T[..., jnp.newaxis] * w
    if component == "u":
        w_area = interp_cell_to_uface(area_w)
        face_area = grid.dx_u * grid.dy_u
    else:
        w_area = interp_cell_to_vface(area_w, grid=grid)
        face_area = grid.dx_v * grid.dy_v
    return nemo_advective_vertical_momentum_advection(
        velocity, w_area, h_face, face_area[..., jnp.newaxis],
        face_active=jnp.broadcast_to(active, velocity.shape),
        bottom_face_mask_mode=bottom_mode,
    )


def _two_bar_plant(oracle: np.ndarray, mask: np.ndarray) -> bool:
    planted = np.array(oracle, copy=True)
    wet = np.argwhere(mask)
    point = tuple(wet[int(np.argmax(np.abs(oracle[mask])))])
    scale = float(np.sqrt(np.mean(np.asarray(oracle)[mask] ** 2)))
    planted[point] += 2.0e-12 * scale * np.sqrt(mask.sum())
    return _metric(planted, oracle, mask)["gate_status"] != "AT BAR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--restart-file", default="DINO_00005760_restart.nc")
    parser.add_argument("--round37", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if (jax.default_backend() != "cpu" or not jax.config.jax_enable_x64
            or os.environ.get("DINO_1226_LANE") != "d180"
            or os.environ.get("LEGOESM_NEMO_E3T") != "both"):
        raise SystemExit("CPU/fp64, DINO_1226_LANE=d180, e3t=both required")

    round37_path = args.round37.resolve()
    if _sha(round37_path) != ROUND37_SHA:
        raise SystemExit("official round-37 receipt changed")
    round37 = json.loads(round37_path.read_text())
    if (round37.get("session_id") != session
            or round37.get("disposition")
            != "ROW4_RHS_LOCALIZED_TO_VERTICAL_ADVECTION"
            or round37.get("first_failing_exact_term") != "vertical_advection"):
        raise SystemExit("round 37 does not open the registered ZAD peel")

    capture = args.capture.resolve()
    capture_meta_path = capture / "capture.json"
    if _sha(capture_meta_path) != CAPTURE_SHA:
        raise SystemExit("official round-37 capture metadata changed")
    metadata = json.loads(capture_meta_path.read_text())
    if (metadata.get("session_id") != session
            or metadata.get("recipe") != "nemo_dino_kamm_mlf"
            or metadata.get("native_shapes")
            != {"u": [199, 53, 36], "v": [200, 52, 36]}
            or round37["bindings"].get("capture_metadata") != CAPTURE_SHA):
        raise SystemExit("round-37 capture provenance changed")

    run = args.run_stepdump.resolve()
    for name, expected_sha in EXPECTED.items():
        path = run / name
        if not path.is_file() or _sha(path) != expected_sha:
            raise SystemExit(f"retained day-180 operand changed: {name}")
    restart = run / args.restart_file
    if not restart.is_file():
        raise SystemExit(f"missing matched restart: {restart}")

    bridge, cfg, model_cfg, model, forcing, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file=args.restart_file,
    )
    if (model_cfg.vertical_momentum_scheme != "nemo_advective"
            or model_cfg.zad_bottom_face_mask != "nemo_faithful"):
        raise SystemExit("faithful ZAD selector/default changed")
    if any(getattr(state, name) is None for name in
           ("T_before", "S_before", "u_before", "v_before")):
        raise SystemExit("matched MLF BEFORE state is incomplete")
    ldf_state = (state.T_before.data, state.S_before.data,
                 state.u_before.data, state.v_before.data)

    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pe

    captured: dict[str, object] = {}
    real_bc = pe._bc_vertical_and_depthmean_velocity

    def spy(*values, **keywords):
        result = real_bc(*values, **keywords)
        if not captured:
            captured.update({
                "h_u": result[0], "h_v": result[1], "w": result[3],
                "u": values[1], "v": values[2],
                "u_active": values[3], "v_active": values[4],
                "grid": values[5],
            })
        return result

    pe._bc_vertical_and_depthmean_velocity = spy
    try:
        with jax.disable_jit():
            _tendency, diagnostic = model.tendencies_with_diagnostics(
                state, surface_forcing=sf, dt=twin.DT, ldf_state=ldf_state)
    finally:
        pe._bc_vertical_and_depthmean_velocity = real_bc
    if not captured or pe._bc_vertical_and_depthmean_velocity is not real_bc:
        raise SystemExit("production w capture/restoration failed")

    production_diag = {
        "u": np.asarray(diagnostic.vertadv_u.data),
        "v": np.asarray(diagnostic.vertadv_v.data),
    }
    recorded_diag = {
        component: _load_capture(capture, metadata, component)
        for component in ("u", "v")
    }
    w_production = jnp.asarray(captured["w"])
    ww_raw = {
        "call1": _load_ww(run / "wzv_dump_ww_call1.bin"),
        "call2": _load_ww(run / "wzv_dump_ww_call2.bin"),
    }
    ww = {}
    for name, value in ww_raw.items():
        padded = np.zeros(np.asarray(w_production).shape, dtype=np.float64)
        if value.shape != padded.shape[:2] + (JPK,):
            raise SystemExit(f"{name} interior shape changed: {value.shape}")
        padded[..., :JPK] = value
        if np.any(value[..., -1] != 0.0) or np.any(padded[..., JPK:] != 0.0):
            raise SystemExit(f"{name} bottom/full-halo convention changed")
        ww[name] = jnp.asarray(padded)

    direct = {"production": {}, "before_velocity": {},
              "nemo_ww_call1": {}, "nemo_ww_call2": {}}
    component_inputs = {
        "u": (captured["u"], state.u_before.data, captured["h_u"],
              captured["u_active"]),
        "v": (captured["v"], state.v_before.data, captured["h_v"],
              captured["v_active"]),
    }
    for component, (now, before, thickness, active) in component_inputs.items():
        for arm, velocity, w_value in (
            ("production", now, w_production),
            ("before_velocity", before, w_production),
            ("nemo_ww_call1", now, ww["call1"]),
            ("nemo_ww_call2", now, ww["call2"]),
        ):
            with jax.disable_jit():
                direct[arm][component] = np.asarray(_direct(
                    velocity, w_value, thickness, active, captured["grid"],
                    component, model_cfg.zad_bottom_face_mask))

    with netCDF4.Dataset(run / "mesh_mask.nc") as dataset:
        umask = np.moveaxis(np.asarray(dataset["umask"][0]), 0, -1)[..., :35] > 0.5
        vmask = np.moveaxis(np.asarray(dataset["vmask"][0]), 0, -1)[..., :35] > 0.5
        tmask = np.moveaxis(np.asarray(dataset["tmask"][0]), 0, -1) > 0.5
    if (int(umask.sum()) != 336_338 or int(vmask.sum()) != 340_271
            or int(tmask.sum()) != 342_134):
        raise SystemExit("active populations changed")
    oracle = {
        "u": _load_full_3d(str(run / "zad_dump_du.bin"), JPI, JPJ, JPK - 1, HLS),
        "v": _load_full_3d(str(run / "zad_dump_dv.bin"), JPI, JPJ, JPK - 1, HLS),
    }
    masks = {"u": umask, "v": vmask}

    rows = {}
    for arm in direct:
        rows[arm] = {
            component: _metric(_crop(component, direct[arm][component]),
                               oracle[component], masks[component])
            for component in ("u", "v")
        }
    removal = {
        component: 1.0 - (
            rows["nemo_ww_call1"][component]["normalized_rms_error"]
            / rows["production"][component]["normalized_rms_error"])
        for component in ("u", "v")
    }

    w_metric = _metric(np.asarray(w_production)[..., :JPK], ww_raw["call1"],
                       tmask, "wzv (vertical velocity)")
    shifted_w = np.zeros_like(ww_raw["call1"])
    shifted_w[..., 1:] = ww_raw["call1"][..., :-1]
    shifted = np.zeros_like(np.asarray(w_production))
    shifted[..., :JPK] = shifted_w
    shifted_direct = {}
    for component, (now, _before, thickness, active) in component_inputs.items():
        with jax.disable_jit():
            value = _direct(now, jnp.asarray(shifted), thickness, active,
                            captured["grid"], component,
                            model_cfg.zad_bottom_face_mask)
        shifted_direct[component] = _metric(
            _crop(component, value), oracle[component], masks[component])

    controls = {
        "hook_restored": pe._bc_vertical_and_depthmean_velocity is real_bc,
        "call1_call2_distinct": EXPECTED["wzv_dump_ww_call1.bin"]
        != EXPECTED["wzv_dump_ww_call2.bin"],
    }
    for component in ("u", "v"):
        wet = masks[component]
        prod = _crop(component, direct["production"][component])
        recorded = _crop(component, recorded_diag[component])
        public = _crop(component, production_diag[component])
        controls[f"direct_equals_public_{component}"] = bool(
            np.array_equal(prod, public))
        controls[f"public_equals_round37_capture_{component}"] = bool(
            np.array_equal(public, recorded))
        controls[f"identity_at_bar_{component}"] = (
            _metric(oracle[component], oracle[component], wet)["gate_status"]
            == "AT BAR")
        controls[f"two_bar_plant_{component}"] = _two_bar_plant(
            oracle[component], wet)
        controls[f"roll_plant_{component}"] = (
            _metric(np.roll(oracle[component], 1, axis=1),
                    oracle[component], wet)["gate_status"] != "AT BAR")
        controls[f"sign_plant_{component}"] = (
            _metric(-oracle[component], oracle[component], wet)["gate_status"]
            != "AT BAR")
        nan = np.array(oracle[component], copy=True)
        nan[tuple(np.argwhere(wet)[0])] = np.nan
        controls[f"wet_nan_plant_{component}"] = not np.isfinite(nan[wet]).all()
        controls[f"vertical_shift_plant_{component}"] = (
            shifted_direct[component]["normalized_rms_error"]
            > rows["nemo_ww_call1"][component]["normalized_rms_error"])
    controls = {name: bool(value) for name, value in controls.items()}
    valid = all(controls.values())
    call1_at_bar = all(
        rows["nemo_ww_call1"][component]["gate_status"] == "AT BAR"
        for component in ("u", "v"))
    majority = all(value >= 0.90 for value in removal.values())
    if not valid:
        disposition = "INVALID"
    elif call1_at_bar:
        disposition = "ZAD_INHERITED_FROM_WZV_AT_BAR"
    elif majority:
        disposition = "ZAD_INHERITED_MAJORITY_FROM_WZV_WITH_RESIDUAL"
    else:
        disposition = "WZV_REFUTED_AS_ZAD_MAJORITY_OWNER"

    source_paths = {
        "dynzad": args.nemo_root / "cfgs/DINO/WORK/dynzad.F90",
        "stpmlf": args.nemo_root / "cfgs/DINO/MY_SRC/stpmlf.F90",
        "sshwzv": args.nemo_root / "cfgs/DINO/MY_SRC/sshwzv.F90",
    }
    paths = {
        "round37": round37_path,
        "round37_capture": capture_meta_path,
        "restart": restart,
        "preregistration": root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round38.md",
        "scorer": Path(__file__).resolve(),
        "production_zad_kernel": root / "packages/ocean/legoesm/ocean/vertical.py",
        "production_tendency": root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py",
        **source_paths,
        **{f"stream:{name}": run / name for name in EXPECTED},
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round38-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "source_contract": {
            "zad": "dynzad.F90:83-119",
            "zad_call": "stpmlf.F90:275,309",
            "qco_wzv": "sshwzv.F90:218-227",
        },
        "arms": rows,
        "call1_error_removal_fraction": removal,
        "production_w_vs_nemo_call1": w_metric,
        "vertical_shift_control_rows": shifted_direct,
        "controls": controls,
        "bindings": {name: _sha(path.resolve()) for name, path in paths.items()},
        "architectural_stop": (
            "faithful QCO ww consumed by ZAD needs r3t(Kaa)-r3t(Kbb); "
            "the current production w is diagnosed before Kaa exists, so a "
            "local arithmetic selector cannot close the inherited residue"),
        "disposition": disposition,
        "ordered_next": 5 if disposition.startswith("ZAD_INHERITED") else 4,
    }
    if _tracked(root):
        raise SystemExit("tracked worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for arm, result in rows.items():
        print(arm, {component: {
            "E": result[component]["normalized_rms_error"],
            "status": result[component]["gate_status"],
        } for component in ("u", "v")})
    print(f"call1_error_removal_fraction={removal}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

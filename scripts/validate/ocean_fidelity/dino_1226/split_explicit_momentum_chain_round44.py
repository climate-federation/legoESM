#!/usr/bin/env python3
"""Capture and score the two production WZV call-2 operands at day 180."""
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
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import split_explicit_momentum_chain_round42 as r42
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.vertical import nemo_qco_live_face_thicknesses


ROUND43_SHA = "7109a7c0eae5684b778452cc5b07e4e1397a4b88849f37d448c515c451494eec"
EXPECTED = {
    "seq_dump_r3t_aaa_kt00005761.bin":
        "96b31bffffa3c1eaa9876b3cc36d08f4e7a790e2acd8f4585b4a325824242e32",
    "seq_dump_hdiv_nnn_kt00005761.bin":
        "51354b833fa429595b10882153f48b3f7d06890d2e5a80d429316ba177bac88e",
    "mesh_mask.nc":
        "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "DINO_00005760_restart.nc":
        "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
}


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


def _metric(candidate, oracle, mask):
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if candidate.shape != oracle.shape or mask.shape != oracle.shape:
        raise SystemExit(
            f"metric shape mismatch: {candidate.shape}/{oracle.shape}/{mask.shape}")
    if not np.all(np.isfinite(candidate[mask])):
        raise SystemExit("candidate contains non-finite active values")
    ref = oracle[mask]
    delta = candidate[mask] - ref
    oracle_rms = float(np.sqrt(np.mean(ref * ref, dtype=np.float64)))
    error_rms = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
    normalized_rms = error_rms / oracle_rms
    maximum = float(np.max(np.abs(delta))) / oracle_rms
    corr = float(np.corrcoef(candidate[mask], ref)[0, 1])
    at_bar = normalized_rms <= 1.0e-12 and maximum <= 1.0e-12
    return {
        "active_count": int(mask.sum()),
        "oracle_rms": oracle_rms,
        "error_rms": error_rms,
        "normalized_rms_error": normalized_rms,
        "per_element_max_error_over_nemo_rms": maximum,
        "correlation": corr,
        "gate_status": "AT BAR" if at_bar else "DEBT",
    }


def _literal_operands(capture):
    (eta_now, _eta_before, u, v, _grid, z_coord, u_mask_3d,
     v_mask_3d, mask_3d, _dt) = capture["args"][:10]
    eta_after = capture["kwargs"]["eta_after_override"]
    eta_now = jnp.asarray(eta_now)
    eta_after = jnp.asarray(eta_after)
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    tmask = jnp.asarray(mask_3d, dtype=eta_now.dtype)
    nlev = u.shape[-1]
    e3t0 = jnp.asarray(z_coord.nemo_e3t_0, dtype=eta_now.dtype)[..., :nlev]
    raw_umask = jnp.asarray(u_mask_3d[:, 1:, :], dtype=eta_now.dtype)
    raw_vmask = jnp.asarray(v_mask_3d[1:, :, :], dtype=eta_now.dtype)
    live_u, live_v = nemo_qco_live_face_thicknesses(
        eta_now, z_coord, e3t0, e3t0, raw_umask, raw_vmask)

    h0 = jnp.zeros_like(eta_now)
    for jk in range(nlev):
        h0 = jax.lax.optimization_barrier(
            h0 + e3t0[..., jk] * tmask[..., jk])
    h0_safe = jnp.where(h0 > 0.0, h0, 1.0)
    r1_h0 = jax.lax.optimization_barrier(1.0 / h0_safe)
    r3_now = jax.lax.optimization_barrier(eta_now * r1_h0)
    r3_after = jax.lax.optimization_barrier(eta_after * r1_h0)
    live_t = e3t0 * (1.0 + r3_now[..., None] * tmask) * tmask
    e2u = jnp.asarray(z_coord.nemo_e2u, dtype=eta_now.dtype)
    e1v = jnp.asarray(z_coord.nemo_e1v, dtype=eta_now.dtype)
    area_t = jnp.asarray(z_coord.nemo_e1e2t, dtype=eta_now.dtype)
    r1_area_t = jax.lax.optimization_barrier(1.0 / area_t)
    levels = []
    for jk in range(nlev):
        flux_u = jax.lax.optimization_barrier(
            jax.lax.optimization_barrier(e2u * live_u[..., jk])
            * u[:, 1:, jk]) * raw_umask[..., jk]
        flux_v = jax.lax.optimization_barrier(
            jax.lax.optimization_barrier(e1v * live_v[..., jk])
            * v[1:, :, jk]) * raw_vmask[..., jk]
        west = jnp.roll(flux_u, 1, axis=1)
        south = jnp.concatenate(
            [jnp.zeros_like(flux_v[:1]), flux_v[:-1]], axis=0)
        zonal = jax.lax.optimization_barrier(flux_u - west)
        meridional = jax.lax.optimization_barrier(flux_v - south)
        numerator = jax.lax.optimization_barrier(zonal + meridional)
        transport_div = jax.lax.optimization_barrier(
            numerator * r1_area_t) * tmask[..., jk]
        safe_e3t = jnp.where(tmask[..., jk] > 0.5, live_t[..., jk], 1.0)
        levels.append(jax.lax.optimization_barrier(transport_div / safe_e3t))
    return r3_after, jnp.stack(levels, axis=-1)


def _roll_fires(oracle, mask) -> bool:
    return _metric(np.roll(oracle, 1, axis=1), oracle, mask)["gate_status"] != "AT BAR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round43", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round43.resolve()) != ROUND43_SHA:
        raise SystemExit("official round-43 receipt changed")
    prior = json.loads(args.round43.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != "ROW5_WZV_CALL2_LITERAL_DIVERGED"):
        raise SystemExit("round 43 does not open the operand capture")
    run = args.run_stepdump.resolve()
    for name, expected in EXPECTED.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained operand changed: {name}")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    _, _, model_cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    if (model_cfg.zad_qco_evaluation != "nemo_literal"
            or model_cfg.wzv_call2_evaluation != "nemo_literal"):
        raise SystemExit("faithful coupled selectors changed")
    state = model._seed_tke_preclosure_carry(state)

    captures = []
    original = model_module.nemo_qco_wzv_operands

    def capture(*call_args, **call_kwargs):
        if call_kwargs.get("eta_after_override") is not None:
            captures.append({"args": call_args, "kwargs": call_kwargs})
        return original(*call_args, **call_kwargs)

    model_module.nemo_qco_wzv_operands = capture
    try:
        with jax.disable_jit():
            model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.nemo_qco_wzv_operands = original
    restored = model_module.nemo_qco_wzv_operands is original
    if len(captures) != 1 or not restored:
        raise SystemExit(
            f"unique call-2 capture/restoration failed: {len(captures)}/{restored}")
    capture0 = captures[0]
    if len(capture0["args"]) < 10:
        raise SystemExit("call-2 signature shortened")
    r3t, hdiv = _literal_operands(capture0)
    r3t = np.asarray(r3t, dtype=np.float64)
    hdiv = np.asarray(hdiv, dtype=np.float64)[..., :35]
    oracle_r3t = r42._load2(run / "seq_dump_r3t_aaa_kt00005761.bin")
    oracle_hdiv = r42._load3(run / "seq_dump_hdiv_nnn_kt00005761.bin")
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        tmask = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1) > 0.5
    if tmask.shape != (199, 52, 36):
        raise SystemExit("cited-interior T mask changed")
    qmask = tmask[..., 0]
    hmask = tmask[..., :35]
    qrow = _metric(r3t, oracle_r3t, qmask)
    hrow = _metric(hdiv, oracle_hdiv, hmask)
    controls = {
        "unique_call2_capture": len(captures) == 1,
        "hook_restored": restored,
        "q_identity_at_bar": _metric(
            oracle_r3t, oracle_r3t, qmask)["gate_status"] == "AT BAR",
        "h_identity_at_bar": _metric(
            oracle_hdiv, oracle_hdiv, hmask)["gate_status"] == "AT BAR",
        "q_roll_plant_fires": _roll_fires(oracle_r3t, qmask),
        "h_roll_plant_fires": _roll_fires(oracle_hdiv, hmask),
    }
    valid = all(controls.values())
    if not valid:
        disposition = "INVALID"
    elif qrow["gate_status"] != "AT BAR":
        disposition = "CALL2_KAA_R3T_DIVERGED"
    elif hrow["gate_status"] != "AT BAR":
        disposition = "CALL2_HDIV_DIVERGED"
    else:
        disposition = "CALL2_OPERANDS_AT_BAR_RECURRENCE_DIVERGED"

    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round44-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 5,
        "operands": {"Q_kaa_r3t": qrow, "H_kmm_hdiv": hrow},
        "controls": controls,
        "capture_contract": {
            "count": len(captures),
            "u_shape": list(np.shape(capture0["args"][2])),
            "v_shape": list(np.shape(capture0["args"][3])),
            "eta_after_shape": list(np.shape(
                capture0["kwargs"]["eta_after_override"])),
            "kind": "genuine cited-interior production helper arguments",
        },
        "bindings": {
            "round43": _sha(args.round43.resolve()),
            **{name: _sha(run / name) for name in EXPECTED},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round44.md"),
            "model": _sha(
                root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "operand_builder": _sha(
                root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py"),
            "nemo_divhor": _sha(nemo / "src/OCE/DYN/divhor.F90"),
            "nemo_domqco": _sha(nemo / "src/OCE/DOM/domqco.F90"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
        },
        "disposition": disposition,
        "ordered_next": 5,
    }
    if _tracked(root):
        raise SystemExit("tracked tree changed during measurement")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    print("Q", qrow["gate_status"], qrow["normalized_rms_error"],
          qrow["per_element_max_error_over_nemo_rms"])
    print("H", hrow["gate_status"], hrow["normalized_rms_error"],
          hrow["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

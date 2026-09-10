#!/usr/bin/env python3
"""Certify literal Kaa/Kmm operands and WZV call 2 at day 180."""
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
import split_explicit_momentum_chain_round38 as r38
import split_explicit_momentum_chain_round44 as r44
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.vertical import nemo_qco_live_face_thicknesses


ROUND44_SHA = "b2b38638bbd4e1e0aaf3f30e67ccec3998addb9af6057f9e2e12b53e76c2b515"
EXPECTED = {
    **r44.EXPECTED,
    "wzv_dump_ww_call2.bin":
        "895a141385f33775c4df7fc127a4beadf2e8e496f68a46624931cf8eb1487634",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_kmm(capture):
    (eta_now, _eta_before, u, v, _grid, z_coord, u_mask_3d,
     v_mask_3d, _mask_3d, _dt) = capture["args"][:10]
    un_adv, vn_adv = capture["kwargs"]["transport_after_override"]
    eta_now = jnp.asarray(eta_now)
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    un_adv = jnp.asarray(un_adv, dtype=eta_now.dtype)
    vn_adv = jnp.asarray(vn_adv, dtype=eta_now.dtype)
    nlev = u.shape[-1]
    e3t0 = jnp.asarray(z_coord.nemo_e3t_0, dtype=eta_now.dtype)[..., :nlev]
    hu0 = jnp.asarray(z_coord.nemo_hu_0, dtype=eta_now.dtype)
    hv0 = jnp.asarray(z_coord.nemo_hv_0, dtype=eta_now.dtype)
    area_t = jnp.asarray(z_coord.nemo_e1e2t, dtype=eta_now.dtype)
    area_u = jnp.asarray(z_coord.nemo_e1e2u, dtype=eta_now.dtype)
    area_v = jnp.asarray(z_coord.nemo_e1e2v, dtype=eta_now.dtype)
    um = jnp.asarray(u_mask_3d[:, 1:, :], dtype=eta_now.dtype)
    vm = jnp.asarray(v_mask_3d[1:, :, :], dtype=eta_now.dtype)
    live_u, live_v = nemo_qco_live_face_thicknesses(
        eta_now, z_coord, e3t0, e3t0, um, vm)
    one = jnp.asarray(1.0, dtype=eta_now.dtype)
    half = jnp.asarray(0.5, dtype=eta_now.dtype)
    weighted = jax.lax.optimization_barrier(area_t * eta_now)
    num_u = jax.lax.optimization_barrier(
        half * jax.lax.optimization_barrier(
            weighted + jnp.roll(weighted, -1, axis=1)))
    num_v = jax.lax.optimization_barrier(
        half * jax.lax.optimization_barrier(
            weighted + jnp.roll(weighted, -1, axis=0)))
    wet_u = (hu0 > 0.0).astype(eta_now.dtype)
    wet_v = (hv0 > 0.0).astype(eta_now.dtype)
    r1_hu0 = jax.lax.optimization_barrier(wet_u / (hu0 + one - wet_u))
    r1_hv0 = jax.lax.optimization_barrier(wet_v / (hv0 + one - wet_v))
    r3u = jax.lax.optimization_barrier(
        jax.lax.optimization_barrier(num_u * r1_hu0)
        * jax.lax.optimization_barrier(one / area_u))
    r3v = jax.lax.optimization_barrier(
        jax.lax.optimization_barrier(num_v * r1_hv0)
        * jax.lax.optimization_barrier(one / area_v))
    r1_hu = jax.lax.optimization_barrier(
        r1_hu0 / jax.lax.optimization_barrier(one + r3u))
    r1_hv = jax.lax.optimization_barrier(
        r1_hv0 / jax.lax.optimization_barrier(one + r3v))
    u_native = u[:, 1:, :]
    v_native = v[1:, :, :]
    puu_b = jnp.zeros_like(eta_now)
    pvv_b = jnp.zeros_like(eta_now)
    for jk in range(nlev):
        puu_b = jax.lax.optimization_barrier(
            puu_b + live_u[..., jk] * u_native[..., jk] * um[..., jk])
        pvv_b = jax.lax.optimization_barrier(
            pvv_b + live_v[..., jk] * v_native[..., jk] * vm[..., jk])
    puu_b = jax.lax.optimization_barrier(puu_b * r1_hu) * wet_u
    pvv_b = jax.lax.optimization_barrier(pvv_b * r1_hv) * wet_v
    target_u = jax.lax.optimization_barrier(un_adv[:, 1:] * r1_hu)
    target_v = jax.lax.optimization_barrier(vn_adv[1:, :] * r1_hv)
    u_native = jax.lax.optimization_barrier(
        u_native + target_u[..., None] - puu_b[..., None]) * um
    v_native = jax.lax.optimization_barrier(
        v_native + target_v[..., None] - pvv_b[..., None]) * vm
    return (
        jnp.concatenate([u_native[:, -1:, :], u_native], axis=1),
        jnp.concatenate([jnp.zeros_like(v_native[:1]), v_native], axis=0),
    )


def _fixed_r3t(capture, fixed_eta):
    eta_now = jnp.asarray(capture["args"][0])
    z_coord = capture["args"][5]
    mask = jnp.asarray(capture["args"][8], dtype=eta_now.dtype)
    e3t0 = jnp.asarray(z_coord.nemo_e3t_0, dtype=eta_now.dtype)[..., :mask.shape[-1]]
    h0 = jnp.zeros_like(eta_now)
    for jk in range(mask.shape[-1]):
        h0 = jax.lax.optimization_barrier(h0 + e3t0[..., jk] * mask[..., jk])
    return jnp.asarray(fixed_eta) / jnp.where(h0 > 0.0, h0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round44", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round44.resolve()) != ROUND44_SHA:
        raise SystemExit("official round-44 receipt changed")
    prior = json.loads(args.round44.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != "CALL2_KAA_R3T_DIVERGED"):
        raise SystemExit("round 44 does not admit the joint fix")
    run = args.run_stepdump.resolve()
    for name, expected in EXPECTED.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained operand changed: {name}")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    _, _, cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    if (cfg.zad_qco_evaluation != "nemo_literal"
            or cfg.wzv_call2_evaluation != "nemo_literal"):
        raise SystemExit("faithful coupled selectors changed")
    state = model._seed_tke_preclosure_carry(state)

    captures = []
    boundaries = []
    original_wzv = model_module.nemo_qco_wzv_operands
    model_class = type(model)
    original_reconcile = model_class._apply_after_level_reconcile

    def capture_wzv(*call_args, **call_kwargs):
        result = original_wzv(*call_args, **call_kwargs)
        if call_kwargs.get("eta_after_override") is not None:
            captures.append({
                "args": call_args, "kwargs": call_kwargs, "ww": result[0]})
        return result

    def capture_reconcile(self, naa, *call_args, **call_kwargs):
        boundaries.append({"eta": naa.eta.data, "w": naa.w.data})
        return original_reconcile(self, naa, *call_args, **call_kwargs)

    model_module.nemo_qco_wzv_operands = capture_wzv
    model_class._apply_after_level_reconcile = capture_reconcile
    try:
        with jax.disable_jit():
            model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.nemo_qco_wzv_operands = original_wzv
        model_class._apply_after_level_reconcile = original_reconcile
    restored = (model_module.nemo_qco_wzv_operands is original_wzv
                and model_class._apply_after_level_reconcile is original_reconcile)
    if len(captures) != 1 or len(boundaries) != 1 or not restored:
        raise SystemExit(
            f"capture/restoration failed: {len(captures)}/{len(boundaries)}/{restored}")
    capture = captures[0]
    if "transport_after_override" not in capture["kwargs"]:
        raise SystemExit("production call 2 omitted the Kmm transport rewrite")
    u_corr, v_corr = _rewrite_kmm(capture)
    corrected_capture = {
        "args": tuple(capture["args"][:2]) + (u_corr, v_corr)
                + tuple(capture["args"][4:]),
        "kwargs": capture["kwargs"],
    }
    r3t, hdiv = r44._literal_operands(corrected_capture)
    r3t = np.asarray(r3t, dtype=np.float64)
    hdiv = np.asarray(hdiv, dtype=np.float64)[..., :35]
    ww = np.asarray(capture["ww"], dtype=np.float64)
    boundary_w = np.asarray(boundaries[0]["w"], dtype=np.float64)
    oracle_q = r44.r42._load2(run / "seq_dump_r3t_aaa_kt00005761.bin")
    oracle_h = r44.r42._load3(run / "seq_dump_hdiv_nnn_kt00005761.bin")
    oracle_w = r38._load_ww(run / "wzv_dump_ww_call2.bin")
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        mask = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1) > 0.5
    qmask, hmask, wmask = mask[..., 0], mask[..., :35], mask
    qrow = r44._metric(r3t, oracle_q, qmask)
    hrow = r44._metric(hdiv, oracle_h, hmask)
    wrow = r44._metric(ww[..., :36], oracle_w, wmask)
    old_q = r44._metric(
        np.asarray(_fixed_r3t(capture, boundaries[0]["eta"])), oracle_q, qmask)
    carry_exact = np.array_equal(
        boundary_w, 0.5 * (ww[..., :-1] + ww[..., 1:]))
    controls = {
        "unique_call2_capture": len(captures) == 1,
        "unique_post_zdf_boundary": len(boundaries) == 1,
        "hooks_restored": restored,
        "post_dyn_zdf_w_carry_exact": carry_exact,
        "q_identity_at_bar": r44._metric(
            oracle_q, oracle_q, qmask)["gate_status"] == "AT BAR",
        "h_identity_at_bar": r44._metric(
            oracle_h, oracle_h, hmask)["gate_status"] == "AT BAR",
        "w_identity_at_bar": r44._metric(
            oracle_w, oracle_w, wmask)["gate_status"] == "AT BAR",
        "q_roll_plant_fires": r44._roll_fires(oracle_q, qmask),
        "h_roll_plant_fires": r44._roll_fires(oracle_h, hmask),
        "w_roll_plant_fires": r44._roll_fires(oracle_w, wmask),
        "old_post_projection_q_fails": old_q["gate_status"] != "AT BAR",
        "round44_old_h_fails": (
            prior["operands"]["H_kmm_hdiv"]["gate_status"] != "AT BAR"),
    }
    valid = all(controls.values())
    if not valid:
        disposition = "INVALID"
    elif qrow["gate_status"] != "AT BAR":
        disposition = "ROW5_LITERAL_KAA_R3T_DIVERGED"
    elif hrow["gate_status"] != "AT BAR":
        disposition = "ROW5_LITERAL_KMM_HDIV_DIVERGED"
    elif wrow["gate_status"] != "AT BAR":
        disposition = "ROW5_LITERAL_RECURRENCE_DIVERGED"
    else:
        disposition = "ROW5_WZV_CALL2_AT_BAR"

    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round45-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 5,
        "operands": {"Q_kaa_r3t": qrow, "H_kmm_hdiv": hrow},
        "row5": wrow,
        "old_post_projection_q": old_q,
        "controls": controls,
        "bindings": {
            "round44": _sha(args.round44.resolve()),
            **{name: _sha(run / name) for name in EXPECTED},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round45.md"),
            "model": _sha(
                root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "operand_builder": _sha(
                root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py"),
            "nemo_dynspg_ts": _sha(nemo / "cfgs/DINO/MY_SRC/dynspg_ts.F90"),
            "nemo_sshwzv": _sha(nemo / "cfgs/DINO/MY_SRC/sshwzv.F90"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
        },
        "disposition": disposition,
        "ordered_next": 6 if disposition == "ROW5_WZV_CALL2_AT_BAR" else 5,
    }
    if r44._tracked(root):
        raise SystemExit("tracked tree changed during measurement")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name, row in (("Q", qrow), ("H", hrow), ("W", wrow)):
        print(name, row["gate_status"], row["normalized_rms_error"],
              row["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

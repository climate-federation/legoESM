#!/usr/bin/env python3
"""Existing-dump 2x2 peel of vertex f and live QCO EEN thicknesses."""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, os, subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax, jax.numpy as jnp, numpy as np, xarray as xr
import split_explicit_momentum_chain_round22 as r22
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _build_een_barotropic_inputs, een_barotropic_coriolis)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.vertical import compute_layer_thickness
from zdf_stream_bracket import manifest_sha256, sha256, stream_manifest

ROUND22_SHA = "c770d68c352682d15464c0bda01cb1c9cde22a857600e237f2658e8c6a97d1e3"
ROUND23_SHA = "4e9c6ee0a172ff5f5005186be1f1f7080927643ad88e1e7d044f00167c168e2e"
BAR = 1e-15


def _mesh(path: Path) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(path, decode_times=False)
    out = {k: np.asarray(ds[k].values).squeeze().astype(np.float64)
           for k in ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f")}
    for k in ("e3t_0", "e3u_0", "e3v_0", "e3f_0", "tmask", "umask", "vmask", "fmask"):
        out[k] = np.moveaxis(
            np.asarray(ds[k].values).squeeze().astype(np.float64), 0, -1)
    return out


def _r3(ssh: np.ndarray, m: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    area = m["e1t"] * m["e2t"]
    def recip(h: np.ndarray, mask: np.ndarray) -> np.ndarray:
        surface = mask.max(axis=-1)
        return surface / (h + 1.0 - surface)
    hu = (m["e3u_0"] * m["umask"]).sum(axis=-1)
    hv = (m["e3v_0"] * m["vmask"]).sum(axis=-1)
    hf = (m["e3f_0"] * m["vmask"] * np.roll(m["vmask"], -1, axis=1)).sum(axis=-1)
    weighted = area * ssh
    quad = ((weighted + np.roll(weighted, -1, axis=1))
            + (np.roll(weighted, -1, axis=0)
               + np.roll(np.roll(weighted, -1, axis=0), -1, axis=1)))
    return {
        "u": 0.5 * (weighted + np.roll(weighted, -1, axis=1))
             * recip(hu, m["umask"]) / (m["e1u"] * m["e2u"]),
        "v": 0.5 * (weighted + np.roll(weighted, -1, axis=0))
             * recip(hv, m["vmask"]) / (m["e1v"] * m["e2v"]),
        "f": 0.25 * quad * recip(hf, m["fmask"]) / (m["e1f"] * m["e2f"]),
    }


def _u_face(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], a.shape[1] + 1) + a.shape[2:], dtype=np.float64)
    out[:, 1:] = a
    out[:, 0] = a[:, -1]
    return out


def _v_face(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1, a.shape[1]) + a.shape[2:], dtype=np.float64)
    out[1:] = a
    return out


def _vertex(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1) + a.shape[2:], dtype=np.float64)
    out[1:, 1:] = a
    out[1:, 0] = a[:, -1]
    return out


def _effective(pre: dict[str, Any], bridge: Any) -> dict[str, np.ndarray]:
    up = r22._checkerboards(tuple(bridge.state.u_mask.data.shape), periodic_u=True)
    vp = r22._checkerboards(tuple(bridge.state.v_mask.data.shape))
    zu = jnp.zeros_like(bridge.state.u_mask.data, dtype=jnp.float64)
    zv = jnp.zeros_like(bridge.state.v_mask.data, dtype=jnp.float64)
    uo = [np.asarray(een_barotropic_coriolis(zu, jnp.asarray(p), pre)[0])[:, 1:] for p in vp]
    vo = [np.asarray(een_barotropic_coriolis(jnp.asarray(p), zv, pre)[1])[1:, :] for p in up]
    eu = r22._solve_coefficients(uo, vp, "u")
    ev = r22._solve_coefficients(vo, up, "v")
    return {**{f"ffu_{k}": v for k, v in eu.items()},
            **{f"ffv_{k}": v for k, v in ev.items()}}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--round22", type=Path, required=True)
    p.add_argument("--round23", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    set_policy(PrecisionPolicy.fp64())
    session = os.environ.get("CODEX_SESSION_ID")
    if not session: raise SystemExit("CODEX_SESSION_ID must be exported")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64: raise SystemExit("cpu/fp64 required")
    root = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(): raise SystemExit("clean scorer checkout required")
    if sha256(a.round22) != ROUND22_SHA or sha256(a.round23) != ROUND23_SHA: raise SystemExit("bound predecessor changed")
    old = json.loads(a.round22.read_text()); prior = json.loads(a.round23.read_text())
    if prior["disposition"] != "VERTEX_F_REFUTED": raise SystemExit("round-23 stop changed")
    run = a.run.resolve(); manifest = stream_manifest(run)
    if manifest_sha256(manifest) != old["bindings"]["on_manifest_sha256"]: raise SystemExit("run manifest changed")
    jpi,jpj,_jpk,hls,_icycle,_nn_e = r22.inherited._read_dims(str(run))
    def load(name: str) -> np.ndarray: return r22.inherited._load_full(str(run/name),jpi,jpj,hls)
    mesh = read_nemo_mesh_mask(str(run/"mesh_mask.nc"), nn_hls=0)
    restart = read_nemo_restart(str(run/r22.inherited.RESTART_FILE), nn_hls=0)
    mx = _mesh(run/"mesh_mask.nc")
    after = load("sshnxt_dump_ssh_after.bin")
    control_r3 = _r3(after, mx)
    r3_files = {k: next(run.glob(f"r3c_dump_r3{k}_kt*.bin")) for k in ("u","v")}
    r3_masks = {"u": mx["umask"][...,0] > .5, "v": mx["vmask"][...,0] > .5}
    r3_rows = {k: r22._metric(control_r3[k], load(path.name), r3_masks[k]) for k,path in r3_files.items()}
    if any(x["gate_status"] != "AT BAR" for x in r3_rows.values()): raise SystemExit(f"r3 transcription failed: {r3_rows}")
    live_r3 = _r3(np.asarray(restart.ssh), mx)
    oracle = {Path(n).stem.removeprefix("corcoef_dump_"): load(n) for n in r22.COEFFICIENTS}
    ua,va = load("cor2d_dump_ua_e_in_substep1.bin"),load("cor2d_dump_va_e_in_substep1.bin")
    nemo_u,nemo_v = load("cor2d_dump_zu_trd_substep1.bin"),load("cor2d_dump_zv_trd_substep1.bin")
    umask,vmask = r3_masks["u"],r3_masks["v"]
    arms = {}
    metric_rows = None
    for f in (0,1):
        placement = "face_latitude" if f else "cell_average"
        cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"), lon_west_deg=1.,lon_east_deg=49.,sill_lon_m_deg=1.,coriolis_placement=placement)
        br = bridge_nemo_to_legoesm_topo(mesh,restart,periodic_i=True,full_step=True,omega=cfg.omega,coriolis_placement=placement,carry_native_lat_deg=True)
        mc,_ = dino_lat_lon_model_config(br.geometry,cfg)
        hk = compute_layer_thickness(br.state.eta.data,br.state.H_bathy.data,br.z_coord,min_water_column_m=mc.min_water_column_m)
        base = _build_een_barotropic_inputs(hk,br.geometry,br.state.land_mask.data,br.state.u_mask.data,br.state.v_mask.data,jnp.float64,metric_complete=True,een_q_boundary=mc.een_q_boundary,een_e3f_scheme=mc.een_e3f_scheme,dz_ref=br.z_coord.dz_ref)
        if metric_rows is None:
            # Metrics are coefficient operands only where their corresponding
            # face is wet. The synthetic polar boundary face is deliberately
            # zero in legoESM and is never read by a scored NEMO coefficient.
            metric_rows={"e1u":r22._metric(np.asarray(br.geometry.dx_u)[:,1:],mx["e1u"],umask),"e2u":r22._metric(np.asarray(br.geometry.dy_u)[:,1:],mx["e2u"],umask),"e1v":r22._metric(np.asarray(br.geometry.dx_v)[1:],mx["e1v"],vmask),"e2v":r22._metric(np.asarray(br.geometry.dy_v)[1:],mx["e2v"],vmask)}
        for q in (0,1):
            pre=dict(base)
            if q:
                e3u=mx["e3u_0"]*(1.+live_r3["u"][...,None]*mx["umask"])*mx["umask"]
                e3v=mx["e3v_0"]*(1.+live_r3["v"][...,None]*mx["vmask"])*mx["vmask"]
                e3f0,_,_=een_e3f_h_vtx(jnp.asarray(mx["e3t_0"]*mx["tmask"]),None,None,br.geometry,"nemo_avg",dz_ref=br.z_coord.dz_ref)
                e3f=np.asarray(e3f0)*(1.+_vertex(live_r3["f"])[...,None]*_vertex(mx["fmask"]))
                eu,ev=_u_face(e3u),_v_face(e3v)
                pre.update(e3u=jnp.asarray(eu),e3v=jnp.asarray(ev),hu=jnp.asarray(eu.sum(-1)),hv=jnp.asarray(ev.sum(-1)),h_vtx=jnp.asarray(e3f))
            coeff=_effective(pre,br)
            rows={n:r22._metric(v,oracle[n],umask if n.startswith("ffu") else vmask) for n,v in sorted(coeff.items())}
            pu,pv=een_barotropic_coriolis(jnp.asarray(r22.r9._u_face(ua)),jnp.asarray(r22.r9._v_face(va)),pre)
            output={"u":r22._metric(np.asarray(pu)[:,1:],nemo_u,umask),"v":r22._metric(np.asarray(pv)[1:],nemo_v,vmask)}
            arms[f"F{f}Q{q}"]={"coefficients":rows,"output":output,"aggregate_rms":sum(x["normalized_rms_error"] for x in rows.values())}
    metrics_ok=all(x["gate_status"]=="AT BAR" for x in metric_rows.values())
    reproduce22=True; reproduce23=True
    for name,row in arms["F0Q0"]["coefficients"].items():
        ref=old["production_effective_coefficients"][name]
        reproduce22 &= abs(row["normalized_rms_error"]-ref["normalized_rms_error"])/ref["normalized_rms_error"] <= BAR
    for name,row in arms["F1Q0"]["coefficients"].items():
        ref=prior["face_latitude_arm"]["effective_coefficients"][name]
        reproduce23 &= abs(row["normalized_rms_error"]-ref["normalized_rms_error"])/ref["normalized_rms_error"] <= BAR
    identity=r22._metric(oracle["ffu_nw"],oracle["ffu_nw"],umask)
    plant=np.array(oracle["ffu_nw"],copy=True); ij=tuple(np.argwhere(umask)[0]); steps=0
    while True:
        plant[ij]=np.nextafter(plant[ij],np.inf); steps+=1
        plant_row=r22._metric(plant,oracle["ffu_nw"],umask)
        if plant_row["gate_status"]=="DEBT": break
        if steps>1024: raise SystemExit("plant did not fire")
    controls={"F0Q0_reproduces_round22":bool(reproduce22),"F1Q0_reproduces_round23":bool(reproduce23),"identity_at_bar":identity["gate_status"]=="AT BAR","nextafter_plant_debt":plant_row["gate_status"]=="DEBT","nextafter_steps":steps}
    if not all(v for k,v in controls.items() if k!="nextafter_steps"): raise SystemExit(f"control failed: {controls}")
    b=arms["F0Q0"]["aggregate_rms"]; x=arms["F1Q1"]["aggregate_rms"]
    exact=metrics_ok and all(y["gate_status"]=="AT BAR" for y in [*arms["F1Q1"]["coefficients"].values(),*arms["F1Q1"]["output"].values()])
    reduction=1.-x/b
    disposition="JOINT_VERTEX_F_QCO_THICKNESS_OWNS_COEFFICIENT_COMPOSITION" if exact else ("JOINT_COMPOSITION_PARTIAL" if reduction>=.9 else "QCO_THICKNESS_REFUTED")
    effects={"F":arms["F1Q0"]["aggregate_rms"]-b,"Q":arms["F0Q1"]["aggregate_rms"]-b,"FxQ":x-arms["F1Q0"]["aggregate_rms"]-arms["F0Q1"]["aggregate_rms"]+b}
    receipt={"schema":"dino-split-explicit-momentum-chain-round24-v1","session_id":session,"git_commit":head,"backend":"cpu","bars":{"pointwise":BAR},"r3_transcription":r3_rows,"metric_operands":metric_rows,"arms":arms,"effects":effects,"joint_reduction":reduction,"controls":controls,"plant_metric":plant_row,"disposition":disposition,"bindings":{"round22_sha256":ROUND22_SHA,"round23_sha256":ROUND23_SHA,"run_manifest_sha256":manifest_sha256(manifest),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(f"disposition={disposition} joint_reduction={reduction:.17g}")
    print(f"artifact={a.output} sha256={sha256(a.output)}")
    return 0

if __name__ == "__main__": raise SystemExit(main())

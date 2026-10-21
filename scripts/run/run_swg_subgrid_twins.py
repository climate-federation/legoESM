#!/usr/bin/env python3
"""Subgrid-swap twins of the rigid-lid SWG fine truth.

Start k is truth day D with its sub-dg50 scales replaced by those of truth day D' (resolved
scales untouched):  start = truth(D) - sub(truth(D)) + sub(truth(D')),  sub(X) = X - P R X, where
R is the symmetric objective's conservative restriction (``truth_restrictor("symmetric")``) and P the
piecewise-constant prolongation with R P = I (each coarse face value to the f fine faces nearest it
along the face and normal to it).  R is linear on the rigid lid (ssh = 0), so R(start) = R(truth(D))
before the lid projection.  The projection (``swe_rigid_lid_projection``) then makes it admissible;
its effect on R(start) is reported, not hidden.

Two arms (--arm): ``box`` = the above (swap, then lid projection); ``psi`` = the same swap done on the
vertex streamfunction, sub(psi) = psi - bilinear(psi sampled at the coarse vertices).  A fine psi that
equals the coarse psi at the coarse vertices restricts (telescoping face sums) to exactly the coarse
transports and its curl is non-divergent on the C-grid, so the swap needs no projection.

  build  : write the start files + checks (CPU: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1)
  launch : run the committed GPU driver wrapper on one GPU for the named starts (no JAX here)
  verify : compare a run directory with the truth days it should reproduce
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[0] / "validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py"
DECK = "maddison-rl"
LEAD_DAYS = 32
START_STRIDE = 437  # starts 731 + k*437, k = 0..7: inside the train split 731..4235
FIRST_START = 731
N_STARTS = 8
ARMS = ("box", "psi")
DONOR_OFFSET_DAYS = 365  # same season (the card's stress is steady, so only decorrelation matters)


def _wet_bounds(mask):
    jj, ii = np.where(np.asarray(mask, dtype=bool))
    return int(jj.min()), int(jj.max()), int(ii.min()), int(ii.max())


def prolong_faces(coarse_u, coarse_v, source, target):
    """Piecewise-constant prolongation P of coarse U/V to the fine native faces, R P = I.

    Fine face at offset m (columns for U, rows for V) from the wet edge takes the coarse face
    I = (m + 1 + f//2)//f - 1 (the coarse face sits at m = f*(I+1) - 1), replicated over the f fine
    faces along it.  Faces nearer a wall than the first coarse face get 0 (the coarse wall value).
    """
    f = int(round(target["spacing_m"] / source["spacing_m"]))
    sj0, sj1, si0, si1 = _wet_bounds(source["mask"]["T"])
    tj0, tj1, ti0, ti1 = _wet_bounds(target["mask"]["T"])
    ny, nx = tj1 - tj0 + 1, ti1 - ti0 + 1
    if (sj1 - sj0 + 1, si1 - si0 + 1) != (f * ny, f * nx):
        raise ValueError("source wet rectangle does not tile the target exactly")
    fine_u = np.zeros(np.asarray(source["mask"]["U"]).shape)
    fine_v = np.zeros(np.asarray(source["mask"]["V"]).shape)
    m = np.arange(f * nx)
    face = (m + 1 + f // 2) // f - 1
    ok = (face >= 0) & (face <= nx - 2)
    rows = tj0 + np.repeat(np.arange(ny), f)
    fine_u[np.ix_(sj0 + np.arange(f * ny), si0 + m[ok])] = \
        np.asarray(coarse_u)[np.ix_(rows, ti0 + face[ok])]
    n = np.arange(f * ny)
    face = (n + 1 + f // 2) // f - 1
    ok = (face >= 0) & (face <= ny - 2)
    cols = ti0 + np.repeat(np.arange(nx), f)
    fine_v[np.ix_(sj0 + n[ok], si0 + np.arange(f * nx))] = \
        np.asarray(coarse_v)[np.ix_(tj0 + face[ok], cols)]
    return fine_u, fine_v


def restrict_uv(restrict, u, v):
    """Rigid-lid restriction (ssh = 0) of native fine U/V to the dg50 faces."""
    c = restrict({"T": np.zeros_like(u), "U": u, "V": v})
    return c["U"], c["V"]


def coarse_part(restrict, source, target, u, v):
    """P R (u, v): the part of the fine field the coarse objective sees."""
    return prolong_faces(*restrict_uv(restrict, u, v), source, target)


def swap_subgrid(restrict, source, target, u_keep, v_keep, u_donor, v_donor):
    """keep - sub(keep) + sub(donor) = P R keep + (donor - P R donor)."""
    pu, pv = coarse_part(restrict, source, target, u_keep, v_keep)
    du, dv = coarse_part(restrict, source, target, u_donor, v_donor)
    return pu + (u_donor - du), pv + (v_donor - dv)


def vertex_streamfunction(u, v, source):
    """Vertex psi (Ny+1, Nx+1) of a closed-wall non-divergent fine field: u = -dpsi/dy, v = +dpsi/dx.

    Vertex (a, b) is the NE corner of T(sj0+a-1, si0+b-1); a = 0 / b = 0 are the south / west walls.
    Returns (psi from the u column sums, max |psi| at the north wall, max |v - dpsi/dx|), all in psi units/dx.
    """
    sj0, sj1, si0, si1 = _wet_bounds(source["mask"]["T"])
    dx = float(source["spacing_m"])
    psi = np.zeros((sj1 - sj0 + 2, si1 - si0 + 2))
    psi[1:, 1:] = -dx * np.cumsum(u[sj0:sj1 + 1, si0:si1 + 1], axis=0)
    v_back = np.diff(psi, axis=1)[1:] / dx
    return psi, float(np.abs(psi[-1]).max()), float(np.abs(v_back - v[sj0:sj1 + 1, si0:si1 + 1]).max())


def uv_from_vertex_streamfunction(psi, source):
    """Native fine U/V of a vertex psi with psi = 0 forced on the four walls."""
    sj0, sj1, si0, si1 = _wet_bounds(source["mask"]["T"])
    dx = float(source["spacing_m"])
    psi = psi.copy()
    psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0.0
    u = np.zeros(np.asarray(source["mask"]["U"]).shape)
    v = np.zeros(np.asarray(source["mask"]["V"]).shape)
    u[sj0:sj1 + 1, si0:si1 + 1] = -np.diff(psi, axis=0)[:, 1:] / dx
    v[sj0:sj1 + 1, si0:si1 + 1] = np.diff(psi, axis=1)[1:] / dx
    return u, v


def _linear_matrix(n_fine_vertices, f):
    """(n_fine_vertices, n_coarse_vertices) linear interpolation from every f-th vertex."""
    w = np.zeros((n_fine_vertices, (n_fine_vertices - 1) // f + 1))
    a = np.arange(n_fine_vertices)
    w[a, a // f] = 1.0 - (a % f) / f
    inside = a % f != 0
    w[a[inside], a[inside] // f + 1] = (a[inside] % f) / f
    return w


def coarse_vertex_interpolant(psi, source, target):
    """Bilinear interpolation of psi sampled at the coarse vertices (the f-th fine vertices)."""
    f = int(round(target["spacing_m"] / source["spacing_m"]))
    wy, wx = _linear_matrix(psi.shape[0], f), _linear_matrix(psi.shape[1], f)
    return wy @ psi[::f, ::f] @ wx.T


def swap_subgrid_psi(source, target, u_keep, v_keep, u_donor, v_donor):
    """psi_keep - sub(psi_keep) + sub(psi_donor), sub(psi) = psi - bilinear(coarse-vertex psi)."""
    pk, _, _ = vertex_streamfunction(u_keep, v_keep, source)
    pd, _, _ = vertex_streamfunction(u_donor, v_donor, source)
    ik = coarse_vertex_interpolant(pk, source, target)
    swapped = ik + (pd - coarse_vertex_interpolant(pd, source, target))
    return uv_from_vertex_streamfunction(swapped, source)


def divergence(u, v, spacing_m):
    """Native C-grid divergence (the lid projection's own stencil) [1/s]."""
    return (u - np.roll(u, 1, axis=1) + v - np.roll(v, 1, axis=0)) / spacing_m


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rms(x, mask):
    return float(np.sqrt(np.mean(np.asarray(x)[mask] ** 2)))


def start_days():
    return [FIRST_START + k * START_STRIDE for k in range(N_STARTS)]


def build(args):
    import jax

    if not jax.config.jax_enable_x64:
        raise SystemExit("JAX_ENABLE_X64=1 required")
    from legoesm.ocean.dynamics.swe_rk3 import swe_rigid_lid_geometry, swe_rigid_lid_projection
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_RL_NAMELIST, build_swg_card
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    truth = Path(args.truth)
    drv = _load(HERE / "run_swg_double_gyre.py")
    restrict = _load(EVAL).truth_restrictor("symmetric")
    source, target = restrict.keywords["source"], restrict.keywords["target"]
    card = build_swg_card(SWG_MADDISON_RL_NAMELIST, surface="rigid_lid")
    lid = swe_rigid_lid_geometry(card.recipe.grid, card.recipe.z_coord)
    dx = float(source["spacing_m"])
    umask, vmask = (np.asarray(source["mask"][k], dtype=bool) for k in "UV")
    out = Path(args.out)
    report = {"truth": str(truth), "donor_offset_days": DONOR_OFFSET_DAYS, "starts": []}

    def day_fields(day):
        path = truth / f"day{day:04d}.npz"
        drv.admit_listed_snapshot(path)
        with np.load(path, allow_pickle=False) as z:
            if np.any(z["ssh_m"] != 0.0):
                raise ValueError(f"{path}: rigid-lid truth must have zero ssh")
            return np.array(z["u_m_s"]), np.array(z["v_m_s"]), int(z["step"]), float(z["dt_s"])

    tu = np.asarray(target["mask"]["U"], dtype=bool)
    tv = np.asarray(target["mask"]["V"], dtype=bool)
    for k, day in enumerate(start_days()):
        donor = day + DONOR_OFFSET_DAYS
        u, v, step, dt_s = day_fields(day)
        ud, vd, _, _ = day_fields(donor)
        cu0, cv0 = restrict_uv(restrict, u, v)
        scale = max(np.abs(cu0).max(), np.abs(cv0).max())
        pu0, pv0 = coarse_part(restrict, source, target, u, v)
        pud, pvd = coarse_part(restrict, source, target, ud, vd)
        subu, subv, dsu, dsv = u - pu0, v - pv0, ud - pud, vd - pvd
        swap_rms = float(np.hypot(_rms(dsu - subu, umask), _rms(dsv - subv, vmask)))
        rpu, rpv = restrict_uv(restrict, *prolong_faces(cu0, cv0, source, target))
        psi, north_wall, v_err = vertex_streamfunction(u, v, source)
        common = {
            "k": k, "start_day": day, "donor_day": donor,
            "subgrid_rms_truth_D_box": float(np.hypot(_rms(subu, umask), _rms(subv, vmask))),
            "subgrid_rms_donor_Dprime_box": float(np.hypot(_rms(dsu, umask), _rms(dsv, vmask))),
            "swap_rms_box_(sub(D')-sub(D))": swap_rms,
            "max_rel_prolong_restrict_identity": float(max(np.abs(rpu - cu0).max(), np.abs(rpv - cv0).max()) / scale),
            "psi_closure_north_wall_over_psi_max": north_wall / float(np.abs(psi).max()),
            "psi_vs_v_max_abs_m_s": v_err,
            "max_div_truth_D_per_s": float(np.abs(divergence(u, v, dx)).max()),
        }
        pert = {"box": swap_subgrid(restrict, source, target, u, v, ud, vd),
                "psi": swap_subgrid_psi(source, target, u, v, ud, vd)}
        for arm, (pu, pv) in pert.items():
            before = (pu, pv)
            if arm == "box":
                pu, pv = (np.asarray(x) for x in swe_rigid_lid_projection(pu, pv, lid)[:2])
            cu1, cv1 = restrict_uv(restrict, *before)
            cu2, cv2 = restrict_uv(restrict, pu, pv)
            rms_c = np.hypot(_rms(cu0, tu), _rms(cv0, tv))
            change = float(np.hypot(_rms(pu - u, umask), _rms(pv - v, vmask)))
            rec = common | {
                "arm": arm,
                "max_rel_dcoarse_before_projection": float(max(np.abs(cu1 - cu0).max(), np.abs(cv1 - cv0).max()) / scale),
                "max_rel_dcoarse_final": float(max(np.abs(cu2 - cu0).max(), np.abs(cv2 - cv0).max()) / scale),
                "rms_dcoarse_final_over_coarse_rms": float(np.hypot(_rms(cu2 - cu0, tu), _rms(cv2 - cv0, tv)) / rms_c),
                "rms_start_minus_truth_D_final": change,
                "max_div_before_projection_per_s": float(np.abs(divergence(*before, dx)).max()),
                "max_div_final_per_s": float(np.abs(divergence(pu, pv, dx)).max()),
            }
            if arm == "box":
                rec["projection_change_rms"] = float(np.hypot(_rms(pu - before[0], umask), _rms(pv - before[1], vmask)))
                rec["projection_change_over_swap"] = rec["projection_change_rms"] / swap_rms
            # subgrid (box definition) left in the final start, vs truth's own
            qu, qv = coarse_part(restrict, source, target, pu, pv)
            rec["subgrid_rms_final_start_box"] = float(np.hypot(_rms(pu - qu, umask), _rms(pv - qv, vmask)))
            start_dir = out / arm / f"start_k{k}_D{day:04d}"
            start_dir.mkdir(parents=True, exist_ok=True)
            np.savez(start_dir / "start.npz", ssh_m=np.zeros_like(u), u_m_s=pu, v_m_s=pv,
                     day=np.int64(day), step=np.int64(step), dt_s=np.float64(dt_s))
            report["starts"].append(rec)
            print(json.dumps(rec), flush=True)
    (out / "build_report.json").write_text(json.dumps(report, indent=1))


def launch(args):
    out = Path(args.out)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), JAX_ENABLE_X64="1")
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0").strip()
    days = start_days()
    times = {}
    for k in args.starts:
        run = out / args.arm / f"twin_k{k}_D{days[k]:04d}"
        t0 = time.time()
        subprocess.run(
            [str(HERE / "run_swg_double_gyre_gpu.sh"), str(args.gpu), "--deck", DECK, "--surface", "rigid_lid",
             "--days", str(LEAD_DAYS), "--restart", str(out / args.arm / f"start_k{k}_D{days[k]:04d}" / "start.npz"),
             "--output", str(run)], check=True, env=env)
        times[k] = time.time() - t0
        print(f"k={k} wall_s={times[k]:.1f}", flush=True)
    (out / args.arm / f"launch_gpu{args.gpu}_times.json").write_text(json.dumps(times))


def verify(args):
    """Max |difference| of u, v between day t of RUN and truth day START+t, relative to the field max."""
    run, truth = Path(args.run), Path(args.truth)
    for t in range(1, args.days + 1):
        with np.load(run / f"day{t:04d}.npz") as a, np.load(truth / f"day{args.start + t:04d}.npz") as b:
            rel = max(float(np.abs(a[f] - b[f]).max() / np.abs(b[f]).max()) for f in ("u_m_s", "v_m_s"))
            unequal = sum(int((a[f].view(np.uint64) != b[f].view(np.uint64)).sum()) for f in ("u_m_s", "v_m_s"))
        print(f"day {t}: max_rel_diff={rel:.3e} unequal_values={unequal}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--truth", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=build)
    la = sub.add_parser("launch")
    la.add_argument("--out", required=True)
    la.add_argument("--arm", choices=ARMS, required=True)
    la.add_argument("--gpu", type=int, required=True)
    la.add_argument("--starts", type=int, nargs="+", required=True)
    la.set_defaults(fn=launch)
    v = sub.add_parser("verify")
    v.add_argument("--run", required=True)
    v.add_argument("--truth", required=True)
    v.add_argument("--start", type=int, required=True)
    v.add_argument("--days", type=int, default=LEAD_DAYS)
    v.set_defaults(fn=verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

"""Measured divergence-damping response of the MPAS and FV3-duo lanes.

Decision B2 (2026-10-01): the duo's d_sw5 divergence damping moves to the
del-4 arm (nord=1) with ``d4_bg`` MATCHED to MPAS's CAM-FV ldiv4
(``mpas_div_damp4_scale=1.0``) by measurement, not by a copied coefficient.
The matching quantity is the per-DYCORE-STEP retention of one planted
divergent mode on each lane's REAL step (arm with damping vs control
without, everything else identical; signed projection onto the planted
pattern, log-slope over K steps).  Per-step retention is dt-invariant by
construction on both lanes (CAM tau4 = 0.01/dt; FV3 dd8 dimensionless) and
is reported next to the e-folding in seconds at each dt actually run.

The planted mode is the SAME analytic field on both lanes -- the velocity
potential phi = P_l(sin lat) with l chosen so the meridional wavelength is
4*Delta (k*Delta = pi/2, Delta = sqrt(mean cell area)), differenced through
each lane's NATIVE discrete gradient (cube: D-grid corner differences over
dx/dy; MPAS: cell differences over dcEdge).  Both are exactly curl-free.
A second mode at 8*Delta gives the k-response.

Nothing here prints a verdict; the numbers go to JSON with git SHA and
every effective flag.  Reviewed design: codex + GLM claim reviews
2026-10-02 (planting via native gradients, signed modal projection, an
operator-level cross-check with n_split=1, e-folding as -1/ln r).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import jax
import numpy as np
from scipy.special import eval_legendre

from legoesm import constants


def legendre_degree_for(delta_m: float, cells_per_wave: float) -> int:
    """l with meridional wavelength ``cells_per_wave * delta`` (P_l has
    ~l half-waves pole to pole: wavelength ~ 2*pi*R / l)."""
    return max(2, int(round(2.0 * np.pi * constants.R_earth
                            / (cells_per_wave * delta_m))))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _fit_retention(a_arm, a_ctrl):
    """Per-step retention r from ln(a_arm/a_ctrl) over k=1..K (least
    squares through the origin); also the max deviation of the per-step
    ratios from r, the consistency check across windows."""
    k = np.arange(1, len(a_arm) + 1, dtype=float)
    y = np.log(np.asarray(a_arm) / np.asarray(a_ctrl))
    slope = float(np.sum(k * y) / np.sum(k * k))
    r = float(np.exp(slope))
    per_step = np.exp(np.diff(np.concatenate([[0.0], y])))
    return r, float(np.max(np.abs(per_step - r)))


def _efold(r: float, dt: float):
    if r <= 0.0 or r >= 1.0:
        return float("nan"), float("nan")
    return -1.0 / np.log(r), -dt / np.log(r)


# --------------------------------------------------------------------------
# FV3 duo
# --------------------------------------------------------------------------
def duo_rest_bundle(dyn, *, t_k: float, ps: float):
    """Isothermal rest on the duo (delp from ak/bk at uniform ps, pt = T,
    u = v = 0, one zero tracer)."""
    import jax.numpy as jnp
    from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_jax
    from legoesm.core.fv3_dynamics import p_var_hydrostatic
    from legoesm.core.fv3_native_state_3d import build_state_3d, field_shape
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA
    n, ng, km = dyn.grid.n, dyn.grid.ng, dyn.config.km
    ak, bk = dyn.ak, dyn.bk
    st6 = build_state_3d(n, ng, km, remap_follows=True, hydrostatic=True)
    for t in range(6):
        for k in range(km):
            st6[t]["delp"][..., k] = (ak[k + 1] - ak[k]) + (bk[k + 1] - bk[k]) * ps
        st6[t]["pt"][...] = t_k
    jstate = state_3d_to_jax(st6)
    press = p_var_hydrostatic(jstate["delp"], ptop=dyn.ptop, akap=FV3_KAPPA,
                              n=n, ng=ng, km=km)
    zeros = jnp.zeros((6,) + tuple(field_shape("delp", n, ng, km)),
                      dtype=jnp.float64)
    bundle = {"state": jstate, "press": press, "q": [zeros], "omga": zeros,
              "nh": None}
    return bundle


def duo_plant(dyn, bundle, *, degree: int, amp: float):
    """Potential wind through the D-grid corner differences on the compute
    window ([cs, cc] for u, [cc, cs] for v, the certified IC convention)."""
    import jax.numpy as jnp
    n, ng = dyn.grid.n, dyn.grid.ng
    cs, cc = slice(ng, ng + n), slice(ng, ng + n + 1)
    u0, v0 = [], []
    for t in range(6):
        gs = dyn.grid.ctx_np["gs6"][t]
        lat_c = np.asarray(gs["grid_lat"])
        dx, dy = np.asarray(gs["dx"]), np.asarray(gs["dy"])
        phi = eval_legendre(degree, np.sin(lat_c))      # corners (m_b, m_b)
        u = np.zeros(bundle["state"]["u"].shape[1:])
        v = np.zeros(bundle["state"]["v"].shape[1:])
        du = (phi[1:, :] - phi[:-1, :]) / dx              # (m_a, m_b)
        dv = (phi[:, 1:] - phi[:, :-1]) / dy              # (m_b, m_a)
        u[cs, cc, :] = du[cs, cc][..., None]
        v[cc, cs, :] = dv[cc, cs][..., None]
        u0.append(u)
        v0.append(v)
    u0, v0 = np.stack(u0), np.stack(v0)
    scale = amp / max(np.abs(u0).max(), np.abs(v0).max())
    u0, v0 = u0 * scale, v0 * scale
    st = dict(bundle["state"])
    st["u"] = bundle["state"]["u"] + jnp.asarray(u0)
    st["v"] = bundle["state"]["v"] + jnp.asarray(v0)
    out = dict(bundle)
    out["state"] = st
    return out, (u0, v0)


def duo_project(dyn, bundle, mode):
    """Signed projection of the compute-window wind onto the planted
    pattern (unweighted: a GLOBAL effective rate, cell areas span ~1.3x)."""
    u0, v0 = mode
    n, ng = dyn.grid.n, dyn.grid.ng
    cs, cc = slice(ng, ng + n), slice(ng, ng + n + 1)
    u = np.asarray(bundle["state"]["u"])
    v = np.asarray(bundle["state"]["v"])
    num = (np.sum(u[:, cs, cc] * u0[:, cs, cc])
           + np.sum(v[:, cc, cs] * v0[:, cc, cs]))
    den = np.sum(u0[:, cs, cc] ** 2) + np.sum(v0[:, cc, cs] ** 2)
    return float(num / den)


def run_duo(*, n: int, km: int, dt: float, n_split: int, nord: int,
            d4_bg_list, cells_per_wave, steps: int, amp: float):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import FV3DuoConfig, FV3DuoDynamicsModel
    from legoesm.grids.factory import create_fv3_duo_grid
    grid = create_fv3_duo_grid(n, 3)
    area = np.concatenate([np.asarray(gs["area"])[3:3 + n, 3:3 + n].ravel()
                           for gs in grid.ctx_np["gs6"]])
    delta = float(np.sqrt(area.mean()))
    rows = []
    for cpw in cells_per_wave:
        deg = legendre_degree_for(delta, cpw)
        cfg_ctrl = FV3DuoConfig(km=km, n_split=n_split, nord=nord, d4_bg=0.0)
        dyn_c = FV3DuoDynamicsModel(grid, cfg_ctrl)
        rest = duo_rest_bundle(dyn_c, t_k=300.0, ps=1.0e5)
        ic, mode = duo_plant(dyn_c, rest, degree=deg, amp=amp)
        a0 = duo_project(dyn_c, ic, mode)
        ctrl, b = [], ic
        t0 = time.time()
        for _ in range(steps):
            b = dyn_c.step(b, dt)
            ctrl.append(duo_project(dyn_c, b, mode))
        b_ctrl = b
        for d4 in d4_bg_list:
            dyn_a = FV3DuoDynamicsModel(
                grid, FV3DuoConfig(km=km, n_split=n_split, nord=nord, d4_bg=d4))
            arm, b = [], ic
            for _ in range(steps):
                b = dyn_a.step(b, dt)
                arm.append(duo_project(dyn_a, b, mode))
            bitwise = None
            if d4 == 0.0:
                # the planted-failure control: every state leaf, not the
                # projection alone (codex 2026-10-02 P2)
                la = jax.tree_util.tree_leaves(b)
                lb = jax.tree_util.tree_leaves(b_ctrl)
                bitwise = bool(len(la) == len(lb) and all(
                    np.array_equal(np.asarray(x), np.asarray(y))
                    for x, y in zip(la, lb)))
                r, dev = 1.0, float(np.max(np.abs(np.array(arm) / np.array(ctrl) - 1.0)))
            else:
                r, dev = _fit_retention(arm, ctrl)
            ef_steps, ef_s = _efold(r, dt)
            rows.append(dict(lane="duo", n=n, km=km, dt=dt, n_split=n_split,
                             nord=nord, d4_bg=d4, cells_per_wave=cpw,
                             degree=deg, delta_m=delta, a0=a0,
                             ctrl=[float(x) / a0 for x in ctrl],
                             arm=[float(x) / a0 for x in arm],
                             retention=r, per_step_dev=dev,
                             efold_steps=ef_steps, efold_s=ef_s,
                             bitwise_vs_ctrl=bitwise,
                             wall_s=time.time() - t0))
            print(f"duo C{n} dt={dt} n_split={n_split} nord={nord} "
                  f"d4_bg={d4} {cpw}dx(l={deg}) r={r:.6f} dev={dev:.2e} "
                  f"efold={ef_steps:.2f} steps = {ef_s:.0f} s  ctrl[-1]="
                  f"{ctrl[-1] / a0:.4f} bitwise_vs_ctrl={bitwise}", flush=True)
    return rows


# --------------------------------------------------------------------------
# MPAS
# --------------------------------------------------------------------------
def mpas_plant(mesh, *, degree: int, amp: float):
    """Potential wind through the native cell-difference gradient."""
    phi = eval_legendre(degree, np.sin(np.asarray(mesh.latCell)))
    c = np.asarray(mesh.cellsOnEdge)
    assert c.shape[0] == 2 and c.max() < mesh.nCells
    u_e = (phi[c[1]] - phi[c[0]]) / np.asarray(mesh.dcEdge)
    return u_e * (amp / np.abs(u_e).max())


def run_mpas(*, level: int, nlev: int, dt_list, scale: float,
             cells_per_wave, steps: int, amp: float):
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.operators_voronoi import divergence_cell_3d
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(level)
    sigma = create_sigma_coordinate(nlev)
    nc = mesh.nCells
    area = np.asarray(mesh.areaCell)
    delta = float(np.sqrt(area.mean()))
    rows = []
    w_e = (np.asarray(mesh.dcEdge) * np.asarray(mesh.dvEdge))
    for cpw in cells_per_wave:
        deg = legendre_degree_for(delta, cpw)
        u0 = mpas_plant(mesh, degree=deg, amp=amp)
        u3 = jnp.asarray(np.repeat(u0[:, None], nlev, axis=1))
        div0 = np.asarray(divergence_cell_3d(u3, mesh))[:, 0]
        state = MPASHydrostaticState(
            u=Field(u3, name="u", dims=("nEdges", "nlev"), units="m/s"),
            T=Field(jnp.full((nc, nlev), 300.0), name="T",
                    dims=("nCells", "nlev"), units="K"),
            p_s=Field(jnp.full((nc,), 1.0e5), name="p_s", dims=("nCells",),
                      units="Pa"),
            phis=Field(jnp.zeros((nc,)), name="phis", dims=("nCells",),
                       units="m^2/s^2"),
            tracers=None)

        def proj(st):
            u = np.asarray(st.u.data)[:, 0]
            pu = float(np.sum(w_e * u * u0) / np.sum(w_e * u0 * u0))
            d = np.asarray(divergence_cell_3d(st.u.data, mesh))[:, 0]
            pd = float(np.sum(area * d * div0) / np.sum(area * div0 * div0))
            return pu, pd

        for dt in dt_list:
            nu = scale * 0.01 * float(area.min()) ** 2 / dt
            out = {}
            t0 = time.time()
            for label, nu_arm in (("ctrl", 0.0), ("arm", nu)):
                model = MPASPrimitiveEquationModel(
                    mesh, sigma, MPASPrimitiveEquationConfig(
                        fix_mass=False, nu_div4=nu_arm))
                st, pu, pd = state, [], []
                for _ in range(steps):
                    st = model.step(st, dt)
                    a, b = proj(st)
                    pu.append(a)
                    pd.append(b)
                out[label] = (pu, pd)
            r_u, dev_u = _fit_retention(out["arm"][0], out["ctrl"][0])
            r_d, dev_d = _fit_retention(out["arm"][1], out["ctrl"][1])
            ef_steps, ef_s = _efold(r_u, dt)
            rows.append(dict(lane="mpas", level=level, nlev=nlev, dt=dt,
                             scale=scale, nu_div4=nu, cells_per_wave=cpw,
                             degree=deg, delta_m=delta,
                             ctrl_u=out["ctrl"][0], arm_u=out["arm"][0],
                             ctrl_div=out["ctrl"][1], arm_div=out["arm"][1],
                             retention=r_u, per_step_dev=dev_u,
                             retention_div=r_d, per_step_dev_div=dev_d,
                             efold_steps=ef_steps, efold_s=ef_s,
                             wall_s=time.time() - t0))
            print(f"mpas res{level} dt={dt} scale={scale} {cpw}dx(l={deg}) "
                  f"r_u={r_u:.6f} r_div={r_d:.6f} dev={dev_u:.2e} "
                  f"efold={ef_steps:.2f} steps = {ef_s:.0f} s  ctrl[-1]="
                  f"{out['ctrl'][0][-1]:.4f}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lane", choices=("duo", "mpas"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--amp", type=float, default=1e-3)
    ap.add_argument("--cells-per-wave", type=float, nargs="+", default=[4.0, 8.0])
    # duo
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--km", type=int, default=5)
    ap.add_argument("--dt", type=float, nargs="+", default=[300.0])
    ap.add_argument("--n-split", type=int, default=8)
    ap.add_argument("--nord", type=int, default=1)
    ap.add_argument("--d4-bg", type=float, nargs="+",
                    default=[0.0, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20])
    # mpas
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--nlev", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args(argv)
    meta = dict(git_sha=_git_sha(), argv=sys.argv, env={
        k: v for k, v in os.environ.items() if k.startswith(("JAX_", "LEGOESM_", "XLA_"))})
    if args.lane == "duo":
        rows = []
        for dt in args.dt:
            rows += run_duo(n=args.n, km=args.km, dt=dt, n_split=args.n_split,
                            nord=args.nord, d4_bg_list=args.d4_bg,
                            cells_per_wave=args.cells_per_wave,
                            steps=args.steps, amp=args.amp)
    else:
        rows = run_mpas(level=args.level, nlev=args.nlev, dt_list=args.dt,
                        scale=args.scale, cells_per_wave=args.cells_per_wave,
                        steps=args.steps, amp=args.amp)
    with open(args.out, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

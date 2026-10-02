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


def duo_plant(dyn, bundle, *, degree: int, amp: float, mode: str = "div",
              levels=None):
    """Potential (``mode="div"``, phi at B-grid corners differenced over
    dx/dy) or streamfunction (``mode="rot"``, psi at A-grid centres
    differenced over dxc/dyc) wind through the D-grid differences on the
    compute window ([cs, cc] for u, [cc, cs] for v, the certified IC
    convention).  ``levels``: the k indices planted (default: all)."""
    import jax.numpy as jnp
    n, ng, km = dyn.grid.n, dyn.grid.ng, dyn.config.km
    cs, cc = slice(ng, ng + n), slice(ng, ng + n + 1)
    lv = list(range(km)) if levels is None else list(levels)
    u0, v0 = [], []
    for t in range(6):
        gs = dyn.grid.ctx_np["gs6"][t]
        u = np.zeros(bundle["state"]["u"].shape[1:])
        v = np.zeros(bundle["state"]["v"].shape[1:])
        if mode == "div":
            phi = eval_legendre(degree, np.sin(np.asarray(gs["grid_lat"])))
            du = (phi[1:, :] - phi[:-1, :]) / np.asarray(gs["dx"])     # (m_a, m_b)
            dv = (phi[:, 1:] - phi[:, :-1]) / np.asarray(gs["dy"])     # (m_b, m_a)
        elif mode == "rot":
            psi = eval_legendre(degree, np.sin(np.asarray(gs["agrid_lat"])))
            dyc, dxc = np.asarray(gs["dyc"]), np.asarray(gs["dxc"])   # (m_a,m_b),(m_b,m_a)
            du = np.zeros(dyc.shape)
            du[:, 1:-1] = -(psi[:, 1:] - psi[:, :-1]) / dyc[:, 1:-1]
            dv = np.zeros(dxc.shape)
            dv[1:-1, :] = (psi[1:, :] - psi[:-1, :]) / dxc[1:-1, :]
        else:
            raise ValueError(mode)
        for k in lv:
            u[cs, cc, k] = du[cs, cc]
            v[cc, cs, k] = dv[cc, cs]
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


def duo_project(dyn, bundle, mode, level=None, template_level=None):
    """Signed projection of the compute-window wind onto the planted
    pattern (unweighted: a GLOBAL effective rate, cell areas span ~1.3x);
    ``level`` restricts the wind to one k; ``template_level`` picks the
    planted pattern's level (an UNPLANTED ``level`` against a planted
    ``template_level`` measures leakage)."""
    u0, v0 = mode
    n, ng = dyn.grid.n, dyn.grid.ng
    cs, cc = slice(ng, ng + n), slice(ng, ng + n + 1)
    u = np.asarray(bundle["state"]["u"])
    v = np.asarray(bundle["state"]["v"])
    ks = slice(None) if level is None else slice(level, level + 1)
    tl = ks if template_level is None else slice(template_level, template_level + 1)
    num = (np.sum(u[:, cs, cc, ks] * u0[:, cs, cc, tl])
           + np.sum(v[:, cc, cs, ks] * v0[:, cc, cs, tl]))
    den = np.sum(u0[:, cs, cc, tl] ** 2) + np.sum(v0[:, cc, cs, tl] ** 2)
    return float(num / den)


def _duo_model(grid, cfg, damp_v=None):
    """A duo model; ``damp_v`` (vorticity/delp del-6 coefficient, NOT a
    model knob) overrides the deck through the model's own deck builder --
    instrument-only, for the rotational positive control."""
    import legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics as m
    if damp_v is None:
        return m.FV3DuoDynamicsModel(grid, cfg)
    base = m.duo_sw_deck
    m.duo_sw_deck = lambda **kw: base(**kw)._replace(damp_v=float(damp_v))
    try:
        return m.FV3DuoDynamicsModel(grid, cfg)
    finally:
        m.duo_sw_deck = base


def run_duo(*, n: int, km: int, dt: float, n_split: int, nord: int,
            d4_bg_list, cells_per_wave, steps: int, amp: float,
            mode: str = "div", levels=None, damp_v_list=None):
    """``damp_v_list`` given: the arms vary the VORTICITY damping
    coefficient instead (control damp_v=0, d4_bg fixed at d4_bg_list[0])
    -- the positive control that the rotational projection sees damping."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import FV3DuoConfig
    from legoesm.grids.factory import create_fv3_duo_grid
    grid = create_fv3_duo_grid(n, 3)
    area = np.concatenate([np.asarray(gs["area"])[3:3 + n, 3:3 + n].ravel()
                           for gs in grid.ctx_np["gs6"]])
    delta = float(np.sqrt(area.mean()))
    rows = []
    mode_name = mode
    for cpw in cells_per_wave:
        deg = legendre_degree_for(delta, cpw)
        vary_damp_v = damp_v_list is not None
        d4_fixed = float(d4_bg_list[0]) if vary_damp_v else 0.0
        cfg_ctrl = FV3DuoConfig(km=km, n_split=n_split, nord=nord, d4_bg=d4_fixed)
        dyn_c = _duo_model(grid, cfg_ctrl, damp_v=0.0 if vary_damp_v else None)
        rest = duo_rest_bundle(dyn_c, t_k=300.0, ps=1.0e5)
        ic, planted = duo_plant(dyn_c, rest, degree=deg, amp=amp,
                                mode=mode_name, levels=levels)
        a0 = duo_project(dyn_c, ic, planted)
        lv = list(range(km)) if levels is None else list(levels)
        ctrl, ctrl_k, b = [], {k: [] for k in lv}, ic
        t0 = time.time()
        for _ in range(steps):
            b = dyn_c.step(b, dt)
            ctrl.append(duo_project(dyn_c, b, planted))
            for k in lv:
                ctrl_k[k].append(duo_project(dyn_c, b, planted, level=k))
        b_ctrl = b
        for d4 in (damp_v_list if vary_damp_v else d4_bg_list):
            if vary_damp_v:
                dyn_a = _duo_model(grid, cfg_ctrl, damp_v=d4)
            else:
                dyn_a = _duo_model(grid, FV3DuoConfig(
                    km=km, n_split=n_split, nord=nord, d4_bg=d4))
            arm, arm_k, b = [], {k: [] for k in lv}, ic
            for _ in range(steps):
                b = dyn_a.step(b, dt)
                arm.append(duo_project(dyn_a, b, planted))
                for k in lv:
                    arm_k[k].append(duo_project(dyn_a, b, planted, level=k))
            per_level = {int(k): (_fit_retention(arm_k[k], ctrl_k[k])[0]
                                  if d4 != 0.0 else 1.0) for k in lv}
            # leakage into UNPLANTED levels (codex 2026-10-02 P2): the
            # planted template's projection at each unplanted k, arm and
            # control, after the last step (fraction of the planted amplitude)
            leak = {int(k): [duo_project(dyn_a, b, planted, level=k, template_level=lv[0]),
                             duo_project(dyn_c, b_ctrl, planted, level=k, template_level=lv[0])]
                    for k in range(km) if k not in lv}
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
                             nord=nord, d4_bg=(d4_fixed if vary_damp_v else d4),
                             damp_v=(d4 if vary_damp_v else None),
                             cells_per_wave=cpw, amp=amp,
                             mode=mode_name, levels=lv, retention_per_level=per_level,
                             leakage_unplanted_arm_ctrl=leak,
                             degree=deg, delta_m=delta, a0=a0,
                             ctrl=[float(x) / a0 for x in ctrl],
                             arm=[float(x) / a0 for x in arm],
                             retention=r, per_step_dev=dev,
                             efold_steps=ef_steps, efold_s=ef_s,
                             bitwise_vs_ctrl=bitwise,
                             wall_s=time.time() - t0))
            pl = {k: round(v, 6) for k, v in per_level.items()}
            lk = {k: [round(x, 5) for x in v] for k, v in leak.items()}
            knob = f"damp_v={d4} d4_bg={d4_fixed}" if vary_damp_v else f"d4_bg={d4}"
            print(f"duo C{n} {mode_name} lv={lv} dt={dt} n_split={n_split} "
                  f"nord={nord} {knob} {cpw}dx(l={deg}) r={r:.6f} "
                  f"per_level={pl} leak={lk} dev={dev:.2e} "
                  f"efold={ef_steps:.2f} steps = {ef_s:.0f} s  ctrl[-1]="
                  f"{ctrl[-1] / a0:.4f} bitwise_vs_ctrl={bitwise}", flush=True)
    n_arms = len(damp_v_list if vary_damp_v else d4_bg_list)
    assert len(rows) == len(cells_per_wave) * n_arms, (len(rows), n_arms)
    return rows


# --------------------------------------------------------------------------
# MPAS
# --------------------------------------------------------------------------
def mpas_plant(mesh, *, degree: int, amp: float, mode: str = "div"):
    """Potential wind through the native cell-difference gradient
    (``div``) or streamfunction wind through the vertex difference along
    the edge (``rot``, psi on vertices / dvEdge)."""
    if mode == "div":
        phi = eval_legendre(degree, np.sin(np.asarray(mesh.latCell)))
        c = np.asarray(mesh.cellsOnEdge)
        assert c.shape[0] == 2 and c.max() < mesh.nCells
        u_e = (phi[c[1]] - phi[c[0]]) / np.asarray(mesh.dcEdge)
    elif mode == "rot":
        psi = eval_legendre(degree, np.sin(np.asarray(mesh.latVertex)))
        v = np.asarray(mesh.verticesOnEdge)
        assert v.shape[0] == 2 and v.max() < psi.shape[0]
        u_e = (psi[v[1]] - psi[v[0]]) / np.asarray(mesh.dvEdge)
    else:
        raise ValueError(mode)
    return u_e * (amp / np.abs(u_e).max())


def run_mpas(*, level: int, nlev: int, dt_list, scale: float,
             cells_per_wave, steps: int, amp: float, mode: str = "div",
             levels=None, sponge: bool = False, a_h_scale: float = 0.375,
             sponge_layers: int = 2, sponge_factor: float = 8.0):
    """``sponge=False``: ldiv4 arm (nu_div4 = scale*0.01*min(area)^2/dt) vs
    control.  ``sponge=True`` (decision B1): the production del2 viscosity
    A_h = a_h_scale*3e-3*min(dcEdge)^2/dt at every level with the
    CAM-style top sponge (x factor^((n-k)/n) in the top ``sponge_layers``)
    as the ARM, the same A_h without the sponge as the CONTROL (the
    sponge's increment), plus a third run with no del2 at all so the
    TOTAL top-layer del2 retention is reported too.  ldiv4 off in both."""
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
    lv = list(range(nlev)) if levels is None else list(levels)
    for cpw in cells_per_wave:
        deg = legendre_degree_for(delta, cpw)
        u0 = mpas_plant(mesh, degree=deg, amp=amp, mode=mode)
        u3n = np.zeros((u0.shape[0], nlev))
        for k in lv:
            u3n[:, k] = u0
        u3 = jnp.asarray(u3n)
        div0 = np.asarray(divergence_cell_3d(u3, mesh))[:, lv[0]]
        # planted-field check (GLM 2026-10-02): the rot field must be
        # discretely non-divergent; the div field is the yardstick
        div_ref = np.asarray(divergence_cell_3d(jnp.asarray(np.repeat(
            mpas_plant(mesh, degree=deg, amp=amp, mode="div")[:, None], nlev, axis=1)),
            mesh))[:, 0]
        div_ratio = float(np.abs(div0).max() / np.abs(div_ref).max())
        if mode == "rot":
            assert div_ratio < 1e-10, f"planted rot field is divergent: {div_ratio:.2e}"
        print(f"mpas res{level} {mode} planted |div|/|div of div-mode| = {div_ratio:.2e}",
              flush=True)
        state = MPASHydrostaticState(
            u=Field(u3, name="u", dims=("nEdges", "nlev"), units="m/s"),
            T=Field(jnp.full((nc, nlev), 300.0), name="T",
                    dims=("nCells", "nlev"), units="K"),
            p_s=Field(jnp.full((nc,), 1.0e5), name="p_s", dims=("nCells",),
                      units="Pa"),
            phis=Field(jnp.zeros((nc,)), name="phis", dims=("nCells",),
                       units="m^2/s^2"),
            tracers=None)

        def proj(st, k=lv[0]):
            u = np.asarray(st.u.data)[:, k]
            pu = float(np.sum(w_e * u * u0) / np.sum(w_e * u0 * u0))
            d = np.asarray(divergence_cell_3d(st.u.data, mesh))[:, k]
            den = np.sum(area * div0 * div0)
            pd = float(np.sum(area * d * div0) / den) if den > 0 else float("nan")
            return pu, pd

        for dt in dt_list:
            nu = scale * 0.01 * float(area.min()) ** 2 / dt
            a_h = a_h_scale * 3.0e-3 * float(np.min(np.asarray(mesh.dcEdge))) ** 2 / dt
            if sponge:
                arms = (("off", dict(nu_del2=0.0)),
                        ("ctrl", dict(nu_del2=a_h)),
                        ("arm", dict(nu_del2=a_h,
                                     sponge_del2_top_layers=sponge_layers,
                                     sponge_del2_top_factor=sponge_factor)))
            else:
                arms = (("ctrl", dict(nu_div4=0.0)), ("arm", dict(nu_div4=nu)))
            out, out_k = {}, {}
            t0 = time.time()
            for label, kw in arms:
                model = MPASPrimitiveEquationModel(
                    mesh, sigma, MPASPrimitiveEquationConfig(fix_mass=False, **kw))
                st, pu, pd = state, [], []
                pk = {k: [] for k in lv}
                for _ in range(steps):
                    st = model.step(st, dt)
                    a, b = proj(st)
                    pu.append(a)
                    pd.append(b)
                    for k in lv:
                        pk[k].append(proj(st, k)[0])
                out[label] = (pu, pd)
                out_k[label] = pk
                out_k[label + "_leak"] = {int(k): proj(st, k)[0]
                                          for k in range(nlev) if k not in lv}
            r_u, dev_u = _fit_retention(out["arm"][0], out["ctrl"][0])
            r_d, dev_d = (_fit_retention(out["arm"][1], out["ctrl"][1])
                          if mode == "div" else (float("nan"), float("nan")))
            per_level = {int(k): _fit_retention(out_k["arm"][k], out_k["ctrl"][k])[0]
                         for k in lv}
            total_per_level = ({int(k): _fit_retention(out_k["arm"][k], out_k["off"][k])[0]
                                for k in lv} if sponge else None)
            ef_steps, ef_s = _efold(r_u, dt)
            rows.append(dict(lane="mpas", level=level, nlev=nlev, dt=dt,
                             scale=scale, nu_div4=(0.0 if sponge else nu),
                             cells_per_wave=cpw,
                             degree=deg, delta_m=delta, mode=mode, levels=lv,
                             amp=amp, div_ratio=div_ratio,
                             leakage_unplanted={lab: out_k[lab + "_leak"] for lab, _ in arms},
                             sponge=sponge, a_h=a_h if sponge else None,
                             retention_per_level=per_level,
                             total_retention_per_level=total_per_level,
                             ctrl_u=out["ctrl"][0], arm_u=out["arm"][0],
                             ctrl_div=out["ctrl"][1], arm_div=out["arm"][1],
                             retention=r_u, per_step_dev=dev_u,
                             retention_div=r_d, per_step_dev_div=dev_d,
                             efold_steps=ef_steps, efold_s=ef_s,
                             wall_s=time.time() - t0))
            pl = {k: round(v, 6) for k, v in per_level.items()}
            tl = (None if total_per_level is None
                  else {k: round(v, 6) for k, v in total_per_level.items()})
            print(f"mpas res{level} {mode} lv={lv} sponge={sponge} dt={dt} "
                  f"scale={scale} {cpw}dx(l={deg}) r_u={r_u:.6f} "
                  f"r_div={r_d:.6f} per_level={pl} total={tl} dev={dev_u:.2e} "
                  f"efold={ef_steps:.2f} steps = {ef_s:.0f} s  ctrl[-1]="
                  f"{out['ctrl'][0][-1]:.4f}", flush=True)
    assert len(rows) == len(cells_per_wave) * len(dt_list), len(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lane", choices=("duo", "mpas"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--amp", type=float, default=1e-3)
    ap.add_argument("--cells-per-wave", type=float, nargs="+", default=[4.0, 8.0])
    ap.add_argument("--mode", choices=("div", "rot"), default="div")
    ap.add_argument("--levels", type=int, nargs="*", default=None,
                    help="k indices planted (default all)")
    ap.add_argument("--damp-v", type=float, nargs="+", default=None,
                    help="duo: vary the vorticity del-6 coefficient instead of "
                         "d4_bg (positive control for --mode rot)")
    ap.add_argument("--mpas-sponge", action="store_true",
                    help="B1: del2 top-sponge arms instead of ldiv4 arms")
    ap.add_argument("--a-h-scale", type=float, default=0.375)
    ap.add_argument("--sponge-layers", type=int, default=2)
    ap.add_argument("--sponge-factor", type=float, default=8.0)
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
                            steps=args.steps, amp=args.amp, mode=args.mode,
                            levels=args.levels, damp_v_list=args.damp_v)
    else:
        rows = run_mpas(level=args.level, nlev=args.nlev, dt_list=args.dt,
                        scale=args.scale, cells_per_wave=args.cells_per_wave,
                        steps=args.steps, amp=args.amp, mode=args.mode,
                        levels=args.levels, sponge=args.mpas_sponge,
                        a_h_scale=args.a_h_scale,
                        sponge_layers=args.sponge_layers,
                        sponge_factor=args.sponge_factor)
    with open(args.out, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

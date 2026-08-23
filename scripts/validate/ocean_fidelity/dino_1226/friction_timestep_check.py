#!/usr/bin/env python
"""#1455 -- what timestep actually multiplies the lateral-friction increment?

The lateral-viscosity OPERATOR is now matched to the oracle to 1e-4
(``wall_ldf_alignment.py``), and raising its coefficient rearranges the
southern-basin error without reducing it (``wall_visc_ablation_gap.py``).  The
one remaining candidate that is naturally a FACTOR OF TWO in the realized
friction is the leap-frog composition: which timestep multiplies the increment,
and which time level the operator reads.

NEMO, traced end to end for DINO's ``key_qco`` leap-frog (no ``key_RK3``):

  * ``stpmlf.F90:319``   ``dyn_ldf(kstp, Nbb, Nnn, uu, vv, Nrhs)`` -- the
    operator reads the BEFORE level (``dynldf_lev_rot_scheme.h90:24-25,28-29``
    take ``pu_in(...,Kbb)``) and ACCUMULATES into ``Krhs``.
  * ``stpmlf.F90:686``   ``rDt = 2*rn_Dt`` from the second step on
    (``:135`` sets ``rDt = rn_Dt`` for the single Euler start).  ``rn_Dt=2700``
    (``namelist_cfg:116``), so ``rDt = 5400 s``.
  * ``dynzdf.F90:145-150`` the after level is
    ``puu(Kaa) = ((1+r3u(Kbb))*puu(Kbb) + rDt*(1+r3u(Kmm))*puu(Krhs))
    / (1+r3u(Kaa))`` -- so the friction increment is ``2*dt`` times the
    operator evaluated on the BEFORE velocity, applied to the BEFORE velocity.
  * ``dynatf_qco.F90:200`` then filters the NOW level with ``rn_atfp = 0.1``
    (``namelist_ref:73``).

legoESM on the ``nemo_dino_kamm_mlf`` card:

  * ``ocean_model_latlon_cgrid.py:8478``  ``rdt = 2.0 * dt``.
  * ``:8516-8518`` the single tendency pass is handed
    ``_ldf_state=(T_before, S_before, u_before, v_before)``, and
    ``ocean_pe_latlon_cgrid.py:4241-4242`` routes ONLY the lateral-friction
    call onto it -- the before-level read.
  * ``ocean_model_latlon_cgrid.py:3478``  ``_diss_du_incr = dt_mom *
    tend.du_diss.data`` with ``dt_mom = dt / config.dt_mom_ratio`` (``:3424``),
    and ``dt`` here IS ``rdt``.
  * ``:8548`` the after level adds that increment to the BEFORE velocity.
  * ``:8322-8324`` the Asselin filter, ``gamma = config.asselin_gamma``.

So the two agree IF AND ONLY IF ``dt_mom_ratio`` is 1 on this card.  That is a
config-dependent fact, so this probe MEASURES it rather than reading it, and
measures it through the increment the model actually forms rather than through
the config field.

THE MEASUREMENT.  The friction increment is linear in the viscosity
coefficient, so building the same model twice at ``rn_Uv`` and ``2*rn_Uv`` and
differencing the dissipative increment the real step returns isolates exactly
one operator's worth of friction, with the bottom drag (which shares that
increment) cancelling because both arms see the identical state.  Dividing by
the operator's own tendency on the before-level velocity gives the multiplier
in seconds.  It is then compared against ``dt``, ``2*dt``, and NEMO's ``rDt``.

This probe prints numbers and never prints a verdict.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.friction_timestep_check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kamm_twin_90d import _build_twin_state, RUN_TRAJ, RUN_STEPDUMP  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

RECIPE = "nemo_dino_kamm_mlf"
RN_UV = 0.27
NEMO_RN_DT = 2700.0          # namelist_cfg:116
NEMO_RDT = 2.0 * NEMO_RN_DT  # stpmlf.F90:686
NEMO_ATFP = 0.1              # namelist_ref:73
WALL_ROWS = [1, 2, 3, 4]


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  recipe={RECIPE}  restart={RUN_STEPDUMP}")
    print(f"PROVENANCE  NEMO rn_Dt={NEMO_RN_DT}  rDt=2*rn_Dt={NEMO_RDT}  "
          f"rn_atfp={NEMO_ATFP}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}  "
          f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}")


def build(u_m: float, dt_mom_ratio: float | None = None):
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        RECIPE, RUN_TRAJ, RUN_STEPDUMP, bridge_before=True, u_m=u_m,
        e3t_mode="both")
    if dt_mom_ratio is not None:
        # Control arm only: rebuild the model through its OWN constructor
        # (kamm_twin_90d.py:632) with the momentum timestep divided -- the one
        # thing that can change the multiplier this probe measures. Not a
        # physics arm.
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel)
        mc = mc._replace(dt_mom_ratio=dt_mom_ratio)
        model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    # ``st`` (not ``br.state``) is the assembled state the twin integrates: it
    # is the one carrying the bridged before levels. The sea-surface before
    # level is not bridged; the model's own rule for a missing before level is
    # to seed it from the now level (ocean_model_latlon_cgrid.py:8457-8461), so
    # use that rather than inventing one. It reaches only the barotropic
    # before-state argument, not the dissipative increment read here.
    if st.eta_before is None:
        st = st._replace(eta_before=st.eta)
    for f in ("u_before", "v_before", "T_before", "S_before"):
        if getattr(st, f) is None:
            raise SystemExit(
                f"the twin state has no {f} -- this probe measures the "
                "leap-frog's before-level friction and cannot run without it")
    return br, cfg, mc, model, forcing, sf, st


def assert_branch(model) -> None:
    """Fail fast unless the model takes the branch this probe re-derives.

    ``ldf_tendency`` below rebuilds the lateral-friction tendency with
    ``nemo_ldf_lap_viscosity_cgrid``.  The model's dispatch has three ways to
    disagree with that (``ocean_pe_latlon_cgrid.py:2680-2685`` the operator
    choice, ``:2775-2788`` the thickness-weighted variant, ``:2789`` the
    slope-foot multiply), and under any of them the quotient silently becomes
    the ratio of two different operators while still printing as a time in
    seconds.  Asserted rather than assumed.
    """
    mc = model.config
    checks = [
        ("lateral_viscosity_operator",
         getattr(mc, "lateral_viscosity_operator", None), "nemo_div_curl"),
        ("lateral_viscosity_e3_weighting",
         getattr(mc, "lateral_viscosity_e3_weighting", "off"), "off"),
        ("slope_foot_alpha", float(getattr(mc, "slope_foot_alpha", 0.0)), 0.0),
        ("dt_mom_ratio", float(getattr(mc, "dt_mom_ratio", 1.0)), 1.0),
    ]
    for name, got, want in checks[:3]:
        if got != want:
            raise SystemExit(
                f"this probe re-derives the '{want}' branch of the lateral "
                f"viscosity, but the model resolves {name}={got!r}. The "
                "quotient would be the ratio of two different operators.")
    for name, got, _ in checks:
        print(f"  branch guard: {name} = {got!r}")
    if float(getattr(mc, "dt_mom_ratio", 1.0)) != 1.0:
        raise SystemExit(
            "dt_mom_ratio != 1 -- the friction timestep is not 2*dt. Note the "
            "model itself refuses this unless the barotropic solver is the "
            "rigid lid (ocean_model_latlon_cgrid.py:2605), which this card is "
            "not, so reaching here means the guard was bypassed.")
    for extra in ("A_h_eq_boost", "A_h_cap_boost", "A_h_floor"):
        v = getattr(mc.lateral_viscosity, extra, None)
        if v not in (None, 0.0, 1.0):
            raise SystemExit(
                f"{extra}={v} breaks the linearity in A_h this probe relies on")


def assert_same_state(a, b) -> None:
    """Both arms must be the identical state, or the difference is not friction."""
    for f in ("u_before", "v_before", "T_before", "S_before", "u", "v",
              "T", "S", "eta", "land_mask", "u_mask", "v_mask", "H_bathy"):
        x, y = getattr(a, f), getattr(b, f)
        if x is None and y is None:
            continue
        d = float(np.abs(np.asarray(x.data, dtype=np.float64)
                         - np.asarray(y.data, dtype=np.float64)).max())
        if d != 0.0:
            raise SystemExit(
                f"the two arms differ in {f} by {d:.3e} -- they are not the "
                "same state, so their difference is not one operator's worth "
                "of friction")
    print("  same-state gate: the two arms are bit-identical in every "
          "state field")


def diss_increment(model, state, rdt):
    """The dissipative momentum increment the REAL step forms, in m/s.

    Calls the model's own step with the same private arguments
    ``_nemo_mlf_step`` uses (``ocean_model_latlon_cgrid.py:8500-8519``), so this
    is the production path and not a re-derivation of it.
    """
    _, extras = model._step_impl(
        state, rdt, _apply_implicit_vmix=False,
        _ab2_scope_override="advective",
        _barotropic_substep_scale=2,
        _barotropic_before_state=(state.eta_before.data,
                                  state.u_before.data, state.v_before.data),
        _fct_tracer_before=(state.T_before.data, state.S_before.data),
        _ldf_state=(state.T_before.data, state.S_before.data,
                    state.u_before.data, state.v_before.data),
    )
    diss = extras[5]
    if diss is None:
        raise SystemExit(
            "the step returned no dissipative increment -- ab2_scope did not "
            "resolve to 'advective', so this probe is measuring nothing")
    return np.asarray(diss[2], dtype=np.float64)


def ldf_tendency(model, state):
    """The lateral-friction tendency on the BEFORE velocity, m/s2.

    The production operator through the production dispatch, evaluated on the
    same before-level velocity ``_ldf_state`` hands it.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_lateral_viscosity_coefficients, nemo_ldf_lap_viscosity_cgrid)
    from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
    import jax
    geom = model.grid
    mc = model.config
    half_UM = mc.lateral_viscosity.A_h / (geom.radius * geom.dlon)
    ahmt, ahmf = nemo_lateral_viscosity_coefficients(geom, half_UM)
    cmask = np.asarray(state.land_mask.data, dtype=np.float64)
    vtx = np.asarray(compute_vertex_mask(cmask, grid=geom), dtype=np.float64)
    act = getattr(model.z_coord, "is_active", None)
    if act is not None:
        cell3 = np.asarray(act, dtype=np.float64) * cmask[..., None]
        vm3 = np.asarray(jax.vmap(lambda m2: compute_vertex_mask(m2, grid=geom),
                                  in_axes=-1, out_axes=-1)(cell3),
                         dtype=np.float64) * vtx[..., None]
    else:
        vm3 = vtx
    du, _ = nemo_ldf_lap_viscosity_cgrid(
        np.asarray(state.u_before.data, dtype=np.float64),
        np.asarray(state.v_before.data, dtype=np.float64),
        geom, ahmt, ahmf, mask=cmask,
        u_mask=np.asarray(state.u_mask.data, dtype=np.float64),
        v_mask=np.asarray(state.v_mask.data, dtype=np.float64),
        vertex_mask=vm3)
    return np.asarray(du, dtype=np.float64), float(mc.lateral_viscosity.A_h)


def main(argv=None) -> int:
    # No --self-test flag: this probe's control (halving the increment and
    # requiring the measured multiplier to halve) is UNCONDITIONAL and asserts,
    # so every run is self-tested and there is no opt-in path to forget.
    argparse.ArgumentParser().parse_args(argv)
    set_policy(PrecisionPolicy.fp64())
    stamp()

    _, cfg1, _, m1, _, _, st1 = build(RN_UV)
    _, cfg2, _, m2, _, _, st2 = build(2.0 * RN_UV)
    dt = float(cfg1.dt)
    rdt = 2.0 * dt
    ratio = float(getattr(m1.config, "dt_mom_ratio", 1.0))
    print(f"\nlegoESM card: dt={dt:g} s, the leap-frog step forms rdt=2*dt="
          f"{rdt:g} s, dt_mom_ratio={ratio:g}, asselin_gamma="
          f"{getattr(m1.config, 'asselin_gamma', None)}, "
          f"outer_integrator={getattr(m1.config, 'outer_integrator', None)}, "
          f"barotropic_solver={getattr(m1.config, 'barotropic_solver', None)}")
    print(f"NEMO:         rn_Dt={NEMO_RN_DT:g} s, rDt=2*rn_Dt={NEMO_RDT:g} s, "
          f"rn_atfp={NEMO_ATFP}")

    assert_branch(m1)
    assert_branch(m2)
    assert_same_state(st1, st2)

    tend1, ah1 = ldf_tendency(m1, st1)
    _, ah2 = ldf_tendency(m2, st2)
    print(f"\nviscosity coefficient: {ah1:.4f} -> {ah2:.4f} m2/s "
          f"(x{ah2 / ah1:.4f})")

    um = np.asarray(st1.u_mask.data, dtype=np.float64)[..., None]
    wet = um > 0
    big = np.abs(tend1) > 1e-3 * np.nanmax(np.abs(tend1))
    sel = wet & big

    # ---------------- 1. the increment, and what it does NOT prove ---------
    # This quotient is dt_mom BY CONSTRUCTION: the numerator is
    # dt_mom*(f(2A)-f(A)) and the denominator is f(A). It confirms one
    # multiplication (ocean_model_latlon_cgrid.py:3478) and the value of
    # dt_mom_ratio, and NOTHING about where the increment then lands. Reported
    # as such, and NOT as the answer.
    d_incr = diss_increment(m2, st2, rdt) - diss_increment(m1, st1, rdt)
    mult = np.where(sel, d_incr / np.where(np.abs(tend1) > 0, tend1, np.nan),
                    np.nan)
    v = mult[np.isfinite(mult)]
    print("\n1. the multiplication (dt_mom, by construction -- not the "
          "composition):")
    print(f"   min {v.min():.4f}  median {np.median(v):.4f}  max {v.max():.4f} s"
          f"   n={v.size}")
    med0 = float(np.median(v))
    off = np.abs(mult - med0) > 1e-9 * abs(med0)
    off = off & np.isfinite(mult)
    wall0 = np.zeros_like(wet, dtype=bool)
    wall0[WALL_ROWS] = True
    print(f"   cells outside 1e-9 of the median: {int(off.sum())} of {v.size}"
          f"  ({int((off & wall0).sum())} of them in the four wall rows)")
    if off.any():
        rows = np.unique(np.nonzero(off)[0])
        print(f"   they sit on rows {rows[:12].tolist()}"
              f"{' ...' if rows.size > 12 else ''} "
              f"({rows.size} distinct rows) -- these are cells where the "
              f"model's own operator output and this probe's re-derivation of "
              f"it do not agree, NOT cells with a different timestep")

    # ---------------- 2. the COMPOSITION, whole step ------------------------
    # The question the candidate actually poses: does one operator's worth of
    # friction reach the AFTER-LEVEL velocity once, multiplied by 2*dt? Only a
    # whole step answers that, and it must be read on the depth MEAN as well as
    # the deviation -- legoESM adds only the baroclinic part here
    # (ocean_model_latlon_cgrid.py:8541-8548) and routes the depth mean through
    # the barotropic solver, which is exactly where a fraction of the friction
    # could go missing.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    from legoesm.ocean.dynamics.ocean_tendency_common import depth_mean
    from legoesm.ocean.vertical import compute_layer_thickness

    h_k = np.asarray(compute_layer_thickness(
        st1.eta.data, st1.H_bathy.data, m1.z_coord,
        min_water_column_m=m1.config.min_water_column_m), dtype=np.float64)
    h_u = np.asarray(min_cell_to_uface(h_k), dtype=np.float64)

    def _bt(x):
        # the model's OWN depth mean with the model's own arguments
        # (ocean_model_latlon_cgrid.py:8531-8533), not a re-derivation
        return np.asarray(depth_mean(x, h_u, 1.0e-10, keepdims=True,
                                     fused=False), dtype=np.float64)

    def _after_u(model, st):
        return np.asarray(model._nemo_mlf_step(st, dt, t_seconds=0.0).u.data,
                          dtype=np.float64)

    du_step = _after_u(m2, st2) - _after_u(m1, st1)
    du_pred = rdt * tend1

    def _fit(meas, pred, where):
        den = float(np.sum(pred[where] * pred[where]))
        slope = float(np.sum(meas[where] * pred[where])) / den if den > 0 else (
            float("nan"))
        resid = float(np.sqrt(np.mean((meas[where] - pred[where]) ** 2)))
        scale = float(np.sqrt(np.mean(pred[where] ** 2)))
        return slope, resid, scale

    print("\n2. the COMPOSITION, from a whole leap-frog step "
          "(1.0 = the increment lands once at 2*dt):")
    for name, meas, pred in (
            ("depth DEVIATION", du_step - _bt(du_step), du_pred - _bt(du_pred)),
            ("depth MEAN     ", _bt(du_step) * np.ones_like(du_step),
             _bt(du_pred) * np.ones_like(du_pred)),
            ("total          ", du_step, du_pred)):
        sl, rs, sc = _fit(meas, pred, sel)
        print(f"   {name}: slope {sl:.6f}   residual {rs:.3e} m/s vs a signal "
              f"of {sc:.3e} ({100 * rs / max(sc, 1e-300):.1f}%)")
    wall = np.zeros_like(wet, dtype=bool)
    wall[WALL_ROWS] = True
    sl_w, _, _ = _fit(_bt(du_step) * np.ones_like(du_step),
                      _bt(du_pred) * np.ones_like(du_pred), sel & wall)
    print(f"   depth MEAN, four wall rows only: slope {sl_w:.6f}")
    print("   (a depth-mean slope near 0 would mean the depth mean of the "
          "friction never reaches the state -- a depth-uniform, "
          "wall-concentrated loss, which is the shape of the defect)")

    # ---------------- 3. two controls, both on the MODEL ---------------------
    # The obvious control -- an arm at dt_mom_ratio=2 -- CANNOT BE BUILT: the
    # model refuses dt_mom_ratio != 1 unless the barotropic solver is the rigid
    # lid (ocean_model_latlon_cgrid.py:2605), and this card runs the
    # split-explicit solver. That refusal is itself the answer to half the
    # question: on this card dt_mom is dt by construction and no configuration
    # can make it otherwise. So the controls below perturb the two things that
    # CAN vary.
    #
    # (a) halve the timestep handed to the real step: the multiplier must halve.
    #     A hard-coded timestep anywhere in the chain fails this.
    dh = (diss_increment(m2, st2, 0.5 * rdt)
          - diss_increment(m1, st1, 0.5 * rdt))
    mh = float(np.nanmedian(np.where(
        sel, dh / np.where(np.abs(tend1) > 0, tend1, np.nan), np.nan)))
    print(f"\n3a. CONTROL, the same step driven at rdt/2: multiplier "
          f"{mh:.4f} s against {med0:.4f} s, ratio {mh / med0:.6f} "
          f"(must be 0.5)")
    assert abs(mh / med0 - 0.5) < 1e-9, (
        f"driving the real step at half the timestep did not halve the "
        f"measured multiplier (ratio {mh / med0}) -- the probe is not reading "
        "the timestep the model applies to the friction")

    # (b) a NULL arm: two models at the SAME viscosity must give a step
    #     difference of exactly zero. If it does not, the whole-step slopes
    #     above are contaminated by something other than the viscosity.
    _, _, _, m_null, _, _, st_null = build(RN_UV)
    null = float(np.abs(_after_u(m_null, st_null) - _after_u(m1, st1)).max())
    print(f"3b. CONTROL, a null arm at the SAME viscosity: max|du| over a "
          f"whole step = {null:.3e} m/s (must be 0)")
    assert null == 0.0, (
        f"two arms at the SAME viscosity differ by {null:.3e} m/s over one "
        "step -- the whole-step slopes above are not attributable to the "
        "viscosity alone")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""#1455 follow-up 1: WHICH DISCRETE SWITCH multiplies a 1e-14 nudge by ~300x?

Pre-registration: ``PREREG_switch_rectifier.md``, committed before this ran.
Read it before citing any number this prints.

WHY.  ``dino_kick_asymmetry_result.md`` found that NEMO's day-10 response to a
1e-14 relative temperature nudge is NOT proportional to the nudge: six
perturbation sizes spanning a factor 309, two distinct argmax cells across
three members, and two different seeds agreeing in perturbation size to 5 parts
in a million AT THE SAME CELL.  A discrete switch that fires or does not
depending on the nudge produces exactly that, and also explains the original
open item's signature -- an offset that appears immediately while the growth
RATES stay equal.

WHAT IT MEASURES.  The GAIN

    G(eps, n) = || T_perturbed(n) - T_control(n) ||_2 / eps

for eps in 1e-14 / 1e-12 / 1e-10 at n = 1 / 10 / 100 / 320 steps, in fp64, in
memory -- no float32 snapshot anywhere in the path (the campaign's other
instrument gap).  A proportional response gives the SAME G for every eps.  The
pre-registered statistic is ``max(G)/min(G)`` across the ladder, and the
pre-registered bar is 3.

THE ARMS.  One process each, everything else the shipped ``nemo_dino_kamm_mlf``
card.  ``evd_smooth`` is the one that separates the PROCESS from its
DISCONTINUITY: it keeps convective adjustment and replaces its hard threshold
with the ramp the config already supports.

EVERYTHING IS IMPORTED.  The state, the model and the surface forcing come from
``kamm_twin_90d._build_twin_state`` -- the same bridge the ensembles ran -- and
the kick comes from ``kamm_twin_90d.apply_temperature_kick``, the committed
helper the ensembles used, with its magnitude as an argument so the ladder and
the ensembles cannot drift apart.

WHAT IT IS NOT.  It prints tables and the mechanical application of the
pre-registered bar.  It edits no card, no recipe and no floor, and it does not
interpret its own numbers.

Usage
-----
  switch_rectifier.py --self-check           # arithmetic only, no model needed
  switch_rectifier.py --arms baseline        # one arm
  switch_rectifier.py                        # every arm, then the census
  switch_rectifier.py --census-only          # the threshold-neighbourhood census
"""
import argparse
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import kamm_twin_90d as K              # noqa: E402  state, model, forcing, kick

EPS_LADDER = (1e-14, 1e-12, 1e-10)
STEP_GRID = (1, 10, 100, 320)          # 320 steps = day 10, where 309x was seen
SEED = 1                               # one seed; the ladder is the variable
SPREAD_BAR = 3.0                       # PREREG: below 3 proportional, at/above 3 rectified

# One process each.  The value is a callable applied to the model config.
ARMS = {
    "baseline":     lambda mc: mc,
    "evd_off":      lambda mc: _conv(mc, scheme="none"),
    "evd_smooth":   lambda mc: _evd(mc, smooth_transition=True),
    "limiter_off":  lambda mc: mc._replace(tracer_advection="centered"),
    "gm_redi_off":  None,               # handled at build time, not on mc
}


def _conv(mc, **kw):
    return mc._replace(physics=mc.physics._replace(
        convection=mc.physics.convection._replace(**kw)))


def _evd(mc, **kw):
    c = mc.physics.convection
    return _conv(mc, enhanced_diffusion=c.enhanced_diffusion._replace(**kw))


# ------------------------------------------------------------------- running ---
def _build(arm):
    """The shipped twin, with ONE process changed.  ``gm_redi_off`` uses the
    bridge's own switch rather than editing the config, because that is the
    path the recorded ablations used."""
    br, cfg, mc, model, forcing, sf, st = K._build_twin_state(
        "nemo_dino_kamm_mlf", K.RUN_TRAJ, K.RUN_STEPDUMP, bridge_before=True,
        use_gm_redi=(False if arm == "gm_redi_off" else None))
    if ARMS[arm] is not None:
        mc = ARMS[arm](mc)
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    return br, cfg, mc, model, forcing, sf, st


def _run(br, cfg, model, forcing, sf, st, n_steps, t0_sec):
    """``run_twin``'s own step loop: apply the time-dependent surface forcing at
    t0 + (k+1)*dt, then step.  Re-spelling it differently here would be a second
    chance for a forcing-phase drift, which is the defect class this campaign
    has spent the most time on -- so it is copied line for line and the SAME
    seasonal clock is used."""
    import jax
    dyn = jax.jit(lambda s: model.step(s, K.DT, surface_forcing=sf))
    out = {}
    for k in range(n_steps):
        st = K.apply_dino_lat_lon_surface_forcing(
            st, forcing, br.z_coord, cfg, K.DT,
            t_seconds=t0_sec + (k + 1) * K.DT)
        st = dyn(st)
        if (k + 1) in STEP_GRID:
            out[k + 1] = np.asarray(st.T.data, dtype=np.float64).copy()
    return out


def gains(arm, eps_ladder=EPS_LADDER, n_steps=None):
    """{eps: {n: gain}} for one arm.  NaN is FATAL."""
    n_steps = max(STEP_GRID) if n_steps is None else n_steps
    br, cfg, mc, model, forcing, sf, st = _build(arm)
    t0 = K.seasonal_t0_seconds(f"{K.RUN_STEPDUMP}/{K.RESTART_FILE}")
    print(f"  [{arm}] convection={mc.physics.convection.scheme} "
          f"smooth={mc.physics.convection.enhanced_diffusion.smooth_transition} "
          f"advection={mc.tracer_advection}", flush=True)
    ctl = _run(br, cfg, model, forcing, sf, st, n_steps, t0)
    out = {}
    for eps in eps_ladder:
        st_p, _ = K.apply_temperature_kick(st, SEED, True, eps=eps)
        per = _run(br, cfg, model, forcing, sf, st_p, n_steps, t0)
        out[eps] = {}
        for n in sorted(ctl):
            d = per[n] - ctl[n]
            if not np.isfinite(d).all():
                raise SystemExit(f"non-finite dT at arm {arm}, eps {eps}, "
                                 f"step {n} -- a blown run is a FINDING")
            out[eps][n] = float(np.linalg.norm(d)) / eps
        print(f"    eps={eps:.0e}  " + "  ".join(
            f"G({n})={out[eps][n]:.4e}" for n in sorted(out[eps])), flush=True)
    return out


# ------------------------------------------------------------------ scoring ---
def spread(gain_by_eps, n):
    """PREREG statistic: max gain / min gain across the eps ladder, at n steps.

    NaN propagates and a zero minimum gives inf rather than a plausible finite
    number -- a gain of exactly zero means the nudge never reached the metric,
    which is a finding, not a linear response.
    """
    v = np.asarray([gain_by_eps[e][n] for e in sorted(gain_by_eps)],
                   dtype=np.float64)
    if not np.isfinite(v).all():
        return float("nan")
    lo = float(np.min(v))
    return float("inf") if lo == 0.0 else float(np.max(v)) / lo


def _self_check():
    """The arithmetic this probe owns: the spread statistic and the bar."""
    lin = {1e-14: {1: 5.0}, 1e-12: {1: 5.0}, 1e-10: {1: 5.0}}
    assert spread(lin, 1) == 1.0
    rect = {1e-14: {1: 300.0}, 1e-12: {1: 3.0}, 1e-10: {1: 1.0}}
    assert abs(spread(rect, 1) - 300.0) < 1e-12
    assert spread(rect, 1) >= SPREAD_BAR and spread(lin, 1) < SPREAD_BAR
    # a gain of exactly zero is inf, never a flattering finite number
    assert spread({1e-14: {1: 0.0}, 1e-12: {1: 1.0}, 1e-10: {1: 1.0}}, 1) == float("inf")
    # NaN propagates rather than being skipped
    assert np.isnan(spread({1e-14: {1: float("nan")}, 1e-12: {1: 1.0},
                            1e-10: {1: 1.0}}, 1))
    print(f"SELF-CHECK OK: spread = max/min gain across the ladder, bar "
          f"{SPREAD_BAR:g}; zero gain -> inf; NaN propagates; "
          f"{len(EPS_LADDER)} nudge sizes x {len(STEP_GRID)} step counts")
    return 0


def table(results):
    print("\n" + "=" * 100)
    print("GAIN LADDER -- ||dT||_2 / eps.  A proportional response gives the "
          "SAME gain for every nudge size.")
    print("=" * 100)
    for arm, g in results.items():
        print(f"\n  {arm}")
        print(f"    {'nudge':<12}" + "".join(f"{'n=' + str(n):>16}"
                                             for n in STEP_GRID))
        for eps in sorted(g):
            print(f"    {eps:<12.0e}" + "".join(f"{g[eps][n]:>16.6e}"
                                                for n in STEP_GRID))
        print(f"    {'SPREAD':<12}" + "".join(f"{spread(g, n):>16.3f}"
                                              for n in STEP_GRID))

    print("\n" + "=" * 100)
    print(f"PRE-REGISTERED RULE -- spread < {SPREAD_BAR:g} proportional, "
          f">= {SPREAD_BAR:g} rectified")
    print("=" * 100)
    print(f"{'arm':<16}" + "".join(f"{'n=' + str(n):>14}" for n in STEP_GRID))
    for arm, g in results.items():
        cells = []
        for n in STEP_GRID:
            s = spread(g, n)
            cells.append(f"{s:>10.2f} {'R' if s >= SPREAD_BAR else '.':<3}")
        print(f"{arm:<16}" + "".join(cells))
    print("  R = rectified at that horizon, . = proportional")


# ------------------------------------------------------------------- census ---
def census():
    """THRESHOLD-NEIGHBOURHOOD OCCUPANCY, both models, at the SHARED state.

    Registered in the pre-registration as the quantity that would matter if the
    switch turns out to be convective adjustment. The trigger is already known
    to be population-exact against NEMO at matched states -- i.e. the two models
    agree on every firing DECISION. That says nothing about how many interfaces
    sit close enough to the threshold that a nudge could flip them, which is
    what makes a switch a rectifier.

    Reported as a census, NOT as a claim about the basin deficit.
    """
    print("\n" + "=" * 100)
    print("THRESHOLD-NEIGHBOURHOOD CENSUS -- how many interfaces sit within a "
          "nudge of the N2 threshold")
    print("=" * 100)
    br, cfg, mc, model, forcing, sf, st = _build("baseline")
    ed = mc.physics.convection.enhanced_diffusion
    thr = float(ed.n2_threshold)
    print(f"  convective adjustment: N2 < {thr:.1e} switches vertical "
          f"diffusivity {ed.K_bg:.1e} -> {ed.K_conv:.1e} m2/s "
          f"(factor {ed.K_conv / ed.K_bg:.0e}), smooth_transition="
          f"{ed.smooth_transition}")
    print("  legoESM, at the shared NEMO day-180 state. NEMO's own N2 at the "
          "same state is NOT read here --")
    print("  that is a second measurement and it is named, not made (see the "
          "pre-registration).")

    from legoesm.ocean.eos import nemo_bn2_depth_ladders
    import jax.numpy as jnp
    t_depth, w_depth = nemo_bn2_depth_ladders(br.z_coord)
    T = jnp.asarray(st.T.data)
    S = jnp.asarray(st.S.data)

    def _n2(T_, S_):
        from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
        return np.asarray(compute_buoyancy_frequency_nemo_bn2(
            T_, S_, t_depth, w_depth, eos_form=ed.n2_eos_form),
            dtype=np.float64)

    n2 = _n2(T, S)
    wet = np.isfinite(n2)
    fired = (n2 < thr) & wet
    print(f"\n  wet interfaces: {int(wet.sum())}   firing now: "
          f"{int(fired.sum())} ({100 * fired.sum() / max(wet.sum(), 1):.3f}%)")
    print(f"\n  {'nudge':<12}{'|dN2| caused':>16}{'within reach':>14}"
          f"{'of wet':>10}   (interfaces a nudge of this size could flip)")
    for eps in EPS_LADDER:
        T_p, _ = K.apply_temperature_kick(st, SEED, True, eps=eps)
        d = np.abs(_n2(jnp.asarray(T_p.T.data), S) - n2)
        reach = float(np.nanmax(d[wet])) if wet.any() else 0.0
        near = int((np.abs(n2 - thr) < reach)[wet].sum())
        print(f"  {eps:<12.0e}{reach:>16.3e}{near:>14d}"
              f"{100 * near / max(int(wet.sum()), 1):>9.4f}%")
    print("\n  (a switch can only rectify a nudge if some interface is within "
          "the nudge's reach of its threshold)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", default=None,
                   help="comma-separated subset of " + ",".join(ARMS))
    p.add_argument("--steps", type=int, default=None,
                   help="stop after this many steps (default: the full grid)")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--census-only", action="store_true")
    args = p.parse_args(argv)

    if args.self_check:
        return _self_check()
    _self_check()
    if args.census_only:
        census()
        return 0

    names = list(ARMS) if args.arms is None else args.arms.split(",")
    for n in names:
        if n not in ARMS:
            raise SystemExit(f"unknown arm {n!r}; known: {', '.join(ARMS)}")
    results = {}
    for arm in names:
        print(f"\nrunning arm {arm}", flush=True)
        results[arm] = gains(arm, n_steps=args.steps)
    table(results)
    census()
    print("\n(no verdict is printed here by design -- "
          "see PREREG_switch_rectifier.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

# Five decades, not three. The bottom rung is a ROUND-OFF CONTROL: at eps=1e-16
# the kick is ~2e-15 K on a ~20 K field, i.e. at fp64 round-off, so whatever
# gain it reports is the floor rather than a response. A rung whose gain sits
# at that floor is measuring arithmetic, not physics, and is excluded from the
# spread. The top rung is there so the shape of G(eps) is visible across the
# whole range rather than inferred from three points -- the first (three-rung)
# version of this ladder could not tell a round-off floor from rectification.
EPS_LADDER = (1e-16, 1e-14, 1e-12, 1e-10, 1e-8)
FLOOR_EPS = 1e-16                      # the round-off control rung
FLOOR_MARGIN = 3.0                     # a rung within this factor of the floor gain is excluded
STEP_GRID = (1, 10, 100, 320)          # 320 steps = day 10, where 309x was seen
SEED = 1                               # one seed; the ladder is the variable
SPREAD_BAR = 3.0                       # PREREG: below 3 proportional, at/above 3 rectified

# One process each.  The value is a callable applied to the model config.
ARMS = {
    "baseline":     lambda mc: mc,
    "evd_off":      lambda mc: _conv(mc, scheme="none"),
    # smooth_transition=True at the config's DEFAULT sharpness 1e6, i.e. a
    # ramp whose width in N2 is ~1e-6 -- ordinary deep-ocean stratification,
    # not a neighbourhood of the threshold. Kept because it was run and its
    # numbers are recorded, but it is NOT a one-variable arm: it mixes most of
    # the stratified ocean 1e5-1e7x harder than the card. Read `evd_sharp`
    # instead (review finding 1).
    "evd_smooth":   lambda mc: _evd(mc, smooth_transition=True),
    # THE one-variable discontinuity arm: same ramp, width 1e-13 in N2, which
    # is BELOW the switch's own -1e-12 offset. Mixing is then identical to the
    # shipped card everywhere except within a hair of the threshold, so a
    # collapse here isolates the DISCONTINUITY and nothing else. Caveat kept in
    # view: the ramp is centred on N2=0 while the hard test fires at -1e-12, so
    # the trigger location moves by 1e-12 in N2 -- negligible against real
    # stratification, but it is a difference and it is stated.
    "evd_sharp":    lambda mc: _evd(mc, smooth_transition=True,
                                    sigmoid_sharpness=1e13),
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
    # BOTH placements, exactly as run_twin dispatches them. The shipped card is
    # 'leapfrog_rhs', which REQUIRES return_rate=True and the rate threaded
    # into model.step -- mixing the two raises rather than silently reverting
    # to legacy placement, and it raised here on the first attempt.
    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    if placement == "leapfrog_rhs":
        dyn = jax.jit(lambda s, ext: model.step(
            s, K.DT, surface_forcing=sf, external_tracer_rate=ext))
    else:
        dyn = jax.jit(lambda s: model.step(s, K.DT, surface_forcing=sf))
    out = {}
    for k in range(n_steps):
        if placement == "leapfrog_rhs":
            st, ext = K.apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, K.DT,
                t_seconds=t0_sec + (k + 1) * K.DT, return_rate=True)
            st = dyn(st, ext)
        else:
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
def spread(gain_by_eps, n, exclude=()):
    """PREREG statistic: max gain / min gain across the eps ladder, at n steps.

    ``exclude`` drops rungs shown to sit at the fp64 round-off floor -- they
    measure arithmetic, not response, and leaving them in manufactures a
    nonlinearity that is really the floor. NaN propagates and a zero minimum
    gives inf rather than a plausible finite number: a gain of exactly zero
    means the nudge never reached the field, which is a finding, not a linear
    response.
    """
    keep = [e for e in sorted(gain_by_eps) if e not in exclude]
    if len(keep) < 2:
        return float("nan")
    v = np.asarray([gain_by_eps[e][n] for e in keep], dtype=np.float64)
    if not np.isfinite(v).all():
        return float("nan")
    lo = float(np.min(v))
    return float("inf") if lo == 0.0 else float(np.max(v)) / lo


def pair_ratio(gain_by_eps, n, lo=1e-12, hi=1e-10):
    """Gain at ``hi`` divided by gain at ``lo`` -- the DECISIVE statistic.

    The max/min spread over the whole ladder mixes two different things: the
    rectification between adjacent rungs, and the fact that the largest rung
    (1e-8) moves N2 by 3700x the sharpened ramp's own width and therefore still
    meets a switch. This pair isolates the first. ``lo``/``hi`` bracket the
    step where the shipped card jumps by three orders of magnitude, and both
    clear the round-off floor at every horizon.
    """
    a, b = gain_by_eps[lo][n], gain_by_eps[hi][n]
    return float("nan") if a == 0 else b / a


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
    # the round-off floor control: a rung reporting the SAME gain as the
    # sub-resolution control is measuring arithmetic and must be dropped.
    # Round-off floor: response 1e-14 everywhere below 1e-12, then a clean
    # linear response of 5.0 gain.  1e-16 and 1e-14 both sit AT the floor
    # response (1e-14) and must go; 1e-12 and up are honest.
    g = {1e-16: {1: 1e-14 / 1e-16},      # gain 100      -> response 1e-14
         1e-14: {1: 1.2e-14 / 1e-14},    # gain 1.2      -> response 1.2e-14  FLOOR
         1e-12: {1: 5.0},                # response 5e-12
         1e-10: {1: 5.0},
         1e-8:  {1: 5.0}}
    ex = floor_excluded(g, 1)
    assert ex == {1e-16, 1e-14}, ex
    # WITHOUT the exclusion the floor manufactures an 83x "nonlinearity"; WITH
    # it the three real rungs are flat and the arm reads proportional. That is
    # the whole point of the control, so it is asserted BOTH ways.
    assert spread(g, 1) >= SPREAD_BAR, spread(g, 1)
    assert abs(spread(g, 1, ex) - 1.0) < 1e-12
    assert spread(g, 1, ex) < SPREAD_BAR
    # a genuinely rectified ladder survives the exclusion
    g2 = {1e-16: {1: 1.0}, 1e-14: {1: 5.0}, 1e-12: {1: 5.0},
          1e-10: {1: 1500.0}, 1e-8: {1: 1500.0}}
    assert floor_excluded(g2, 1) == {1e-16}, floor_excluded(g2, 1)
    assert spread(g2, 1, floor_excluded(g2, 1)) >= SPREAD_BAR
    # fewer than two usable rungs is NaN, never a flattering 1.0
    assert np.isnan(spread({1e-16: {1: 1.0}, 1e-14: {1: 1.0}}, 1,
                           {1e-16, 1e-14}))
    # pair_ratio(): the decisive adjacent-rung statistic
    pr = {1e-12: {1: 5.0}, 1e-10: {1: 15.0}}
    assert abs(pair_ratio(pr, 1) - 3.0) < 1e-12
    assert np.isnan(pair_ratio({1e-12: {1: 0.0}, 1e-10: {1: 1.0}}, 1))
    # _grid() reports only step counts EVERY arm reached
    r = {"a": {1e-14: {1: 0.0, 10: 0.0}}, "b": {1e-14: {1: 0.0}}}
    assert _grid(r) == (1,), _grid(r)
    print(f"SELF-CHECK OK: spread = max/min gain across the ladder, bar "
          f"{SPREAD_BAR:g}; zero gain -> inf; NaN propagates; "
          f"the {FLOOR_EPS:.0e} round-off control excludes floor-bound rungs "
          f"(shown to change the verdict both ways); "
          f"{len(EPS_LADDER)} nudge sizes x {len(STEP_GRID)} step counts")
    return 0


def rung_set(gain_by_eps, grid):
    """The rungs excluded as floor-bound, decided ONCE at the SHORTEST horizon
    and applied to every column.

    The control's response is not a fixed arithmetic floor -- a one-ulp seed is
    a real perturbation and it GROWS (measured: the baseline's control response
    rises 124x from step 1 to step 320). Re-deciding per column therefore
    swallows honest rungs at long horizons and makes the columns
    non-comparable, which is what the first version did (review finding 2).
    """
    return floor_excluded(gain_by_eps, grid[0])


def _grid(results):
    """The step counts every arm actually reached. ``--steps N`` stops short of
    the full grid, and printing a column nobody ran raised a KeyError on the
    first real run rather than printing a blank."""
    reached = None
    for g in results.values():
        for per in g.values():
            ks = set(per)
            reached = ks if reached is None else (reached & ks)
    return tuple(n for n in STEP_GRID if reached and n in reached)


def floor_excluded(gain_by_eps, n):
    """Rungs whose RESPONSE is at the arithmetic floor.

    The comparison is on the response ``||dT|| = G * eps``, NOT on the gain.
    The floor is a floor on the response: at ``FLOOR_EPS`` the kick is below
    fp64 resolution, so ``||dT||`` there is round-off and nothing else. Its
    GAIN is enormous precisely because the eps in the denominator is tiny, so
    comparing gains excludes every honest rung and keeps the floor -- the exact
    inversion, which is what the first version of this function did.

    A rung is excluded when its response is within ``FLOOR_MARGIN`` of the
    control's, i.e. when it cannot be told apart from round-off.
    """
    if FLOOR_EPS not in gain_by_eps:
        return set()
    floor_norm = gain_by_eps[FLOOR_EPS][n] * FLOOR_EPS
    out = {FLOOR_EPS}
    if not np.isfinite(floor_norm) or floor_norm == 0:
        return out
    for e, per in gain_by_eps.items():
        if e != FLOOR_EPS and np.isfinite(per[n]) \
                and per[n] * e <= FLOOR_MARGIN * floor_norm:
            out.add(e)
    return out


def table(results):
    print("\n" + "=" * 100)
    print("GAIN LADDER -- ||dT||_2 / eps.  A proportional response gives the "
          "SAME gain for every nudge size.")
    print("=" * 100)
    grid = _grid(results)
    for arm, g in results.items():
        print(f"\n  {arm}")
        print(f"    {'nudge':<12}" + "".join(f"{'n=' + str(n):>16}"
                                             for n in grid) + "   note")
        ex = rung_set(g, grid)
        excl = {n: ex for n in grid}
        for eps in sorted(g):
            note = ("round-off control" if eps == FLOOR_EPS else
                    ("at floor -> excluded" if all(eps in excl[n] for n in grid)
                     else ""))
            print(f"    {eps:<12.0e}" + "".join(f"{g[eps][n]:>16.6e}"
                                               for n in grid) + f"   {note}")
        print(f"    {'SPREAD':<12}"
              + "".join(f"{spread(g, n, excl[n]):>16.3f}" for n in grid))
        print(f"    {'(rungs used)':<12}"
              + "".join(f"{len(g) - len(excl[n]):>16d}" for n in grid))

    print("\n" + "=" * 100)
    print(f"PRE-REGISTERED RULE -- spread < {SPREAD_BAR:g} proportional, "
          f">= {SPREAD_BAR:g} rectified")
    print("=" * 100)
    print(f"{'arm':<16}" + "".join(f"{'n=' + str(n):>14}" for n in grid))
    for arm, g in results.items():
        cells = []
        for n in grid:
            s = spread(g, n, rung_set(g, grid))
            cells.append(f"{s:>10.2f} {'R' if s >= SPREAD_BAR else '.':<3}")
        print(f"{arm:<16}" + "".join(cells))
    print("  R = rectified at that horizon, . = proportional")

    print("\n" + "=" * 100)
    print("THE DECISIVE PAIR -- gain(1e-10) / gain(1e-12), the step where the "
          "shipped card jumps")
    print("=" * 100)
    print("Both rungs clear the round-off floor at every horizon, and neither "
          "is large enough to outrun a")
    print("sharpened ramp. 1.0 means the response is proportional between "
          "them; the shipped card is ~2000.")
    print(f"{'arm':<16}" + "".join(f"{'n=' + str(n):>14}" for n in grid))
    for arm, g in results.items():
        print(f"{arm:<16}" + "".join(f"{pair_ratio(g, n):>14.4g}"
                                     for n in grid))


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

    # ---- IS THE `evd_smooth` ARM A ONE-VARIABLE COMPARISON? -----------------
    # The smoothed arm replaces jnp.where(N2 < thr, K_conv, K_bg) with
    #   K_bg + (K_conv - K_bg) * sigmoid(-N2 * sigmoid_sharpness)
    # so its transition has a WIDTH in N2 of about 1/sharpness. If a large
    # share of the ocean sits inside that width, the smoothed arm is not the
    # same physics with a softer edge -- it is a DIFFERENT MIXING FIELD, and
    # its collapse of the gain spread would then prove something other than
    # "the discontinuity is the rectifier". This is the arm's validity check
    # and it belongs next to the arm, not in a reviewer's head.
    import jax
    K_hard = np.where(n2 < thr, ed.K_conv, ed.K_bg)
    print("\n  ARM VALIDITY -- is a smoothed arm a softened switch, or a "
          "different ocean?")
    print("    A one-variable 'softened switch' moves only interfaces NEAR the "
          "threshold. A large share")
    print("    moving means the arm changed the MIXING FIELD itself, and its "
          "result is not about the edge.")
    print(f"    {'arm':<14}{'sharpness':>12}{'ramp width in N2':>19}"
          f"{'inside width':>14}{'median K/K_hard':>18}{'moved >10%':>12}")
    for arm_name, sharp in (("evd_smooth", 1e6), ("evd_sharp", 1e13)):
        width = 1.0 / sharp
        sig = np.asarray(jax.nn.sigmoid(-n2 * sharp), dtype=np.float64)
        K_soft = ed.K_bg + (ed.K_conv - ed.K_bg) * sig
        inside = (np.abs(n2) < width) & wet
        ratio = np.where(K_hard > 0, K_soft / np.maximum(K_hard, 1e-300), np.nan)
        moved = float(np.mean(np.abs(ratio[wet] - 1.0) > 0.1))
        print(f"    {arm_name:<14}{sharp:>12.0e}{width:>19.3e}"
              f"{100 * inside.sum() / max(int(wet.sum()), 1):>13.3f}%"
              f"{np.nanmedian(ratio[wet]):>18.4g}{100 * moved:>11.3f}%")
    print("    (the sharp arm's ramp is narrower than the switch's own "
          "-1e-12 offset, so away from the threshold its mixing is the "
          "shipped card's to the digit -- that is what makes it "
          "one-variable)")


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

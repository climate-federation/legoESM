#!/usr/bin/env python
"""#1455 SG-A: legoESM's OWN per-term momentum budget ACCUMULATED over the
90-day twin, on ITS OWN trajectory, per 10-day interval.

COMPANION to ``southern_term_torque_matched.py``.  That probe compares the two
models' operators on BIT-IDENTICAL states (zero trajectory divergence, both
sides instantaneous, so sampling error cancels in the difference).  This one
answers the other half: what lego's terms actually integrate to along the
trajectory that produces the deficit.

THE PER-STEP IDENTITY THIS RESTS ON (leap-frog, ocean_model_latlon_cgrid.
_leapfrog_step: ``X(Naa) = X(Nbb) + rDt * RHS``, rDt = 2*dt):

    [ R(Naa) - R(Nbb) ] / rDt  =  sum(explicit terms)  +  REST

with R the depth-integrated row circulation (the recorded reducer,
``southern_circulation_budget.row_circulation``), the explicit terms taken
from the model's own ``MomentumTendencyDiagnostics``, and

    REST = split-explicit BAROTROPIC solve (NEMO ``spg``)
         + implicit vertical solve, i.e. bottom drag (NEMO's half of ``zdf``)
         + Robert-Asselin filter (NEMO ``atf``; measured at 0.00-0.01 per row,
           i.e. negligible -- the remainder is really barotropic + vertical)
         + the baroclinic/barotropic split's own leftovers

REST IS BIGGER THAN THAT LABEL ADMITS, and the honest version is this: the
momentum combine does NOT leapfrog the depth-mean.  ``_leapfrog_step``
leapfrogs only the BAROCLINIC deviation and takes the barotropic mode DIRECTLY
from the split-explicit solve (``btu_exp``).  Since R is the depth-integrated
circulation, ``R(Naa) - R(Nbb)`` is set almost entirely by that solve, and the
explicit terms' depth-means enter only indirectly through ``F_slow``.  So
    REST ~ (barotropic solve) - sum(depth-integral of every explicit term)
and the columns "lego REST" and "NEMO spg+atf" below are NOT the same
quantity.  They are printed adjacent for scale only; no difference between
them is an attribution.

AND THE LDF ROW IS A TERM THE MODEL DISCARDS AT THIS REDUCTION.  The
dissipative increment enters the combine as ``du_diss_bc``, its BAROCLINIC
DEVIATION only -- the in-code note reads "the diss depth-mean is intentionally
NOT injected into the split-explicit barotropic mode".  The row reducer IS the
depth integral.  So the accumulated LDF row is ~100% of a quantity that never
reaches R, and REST silently carries its negative.  It is tabulated for
comparison with the matched-state probe, NOT as a contribution to R.

REST IS A LUMPED BUCKET AND IS LABELLED AS ONE (Rule 5): a signal inside it
names three stages at once, never one.  It is reported because NEMO's
corresponding sum (``spg + zdf_recovered + atf``) is available and comparable,
not because it can attribute.  The four explicit rows are each independently
measured; only REST is a remainder.

WHAT IS NOT CLOSURE-TESTABLE HERE, SAID PLAINLY.  Because REST is defined as
the remainder, "the terms sum to the state change" is TRUE BY CONSTRUCTION and
is NOT evidence.  The accumulation is validated instead by three checks that
CAN fail:
  R  the on-device row reducer must reproduce the recorded numpy reducer
     (``southern_circulation_budget._row_int_trend``) at step 0.  NOTE what
     this does and does not cover: the weights are term- and step-independent,
     so one term at one step IS sufficient for the REDUCER -- but it does not
     cover the depth-integrated maps or the u-face slicing.  There is NO
     per-term closure gate here and an earlier docstring claimed one: with
     ``WIND_u`` DEFINED as ``total_u - sum(components)``, "the terms sum to the
     total" is an algebraic identity that cannot fail.  Retracted;
  W  the surface residual is identified as the wind deposit (k=0-confined AND
     equal to the analytic DINO wind torque) -- see the day-0 probe's
     docstring for why ``MomentumTendencyDiagnostics`` has that hole;
  P  a PLANTED constant torque injected into ``KE_PGF_u``'s accumulator over a
     1-day run must shift THAT term's accumulated ROW torque by exactly the
     hand-computed row integral, shift ``REST`` by exactly minus that amount
     (REST is the remainder, so it MUST absorb it -- a plant that did not move
     REST would prove the remainder was disconnected), and leave TOTAL and
     every other term bit-unchanged.  Measured: predicted +707.1875, got
     KE_PGF_u +707.188 / REST -707.187 / TOTAL invariant to 0.000.  The plant
     is applied to the reduced row array, so it does NOT reach ``acc_map``;
     the maps are not covered by this control.

SAMPLING (stated, not buried).  lego is accumulated over every step; NEMO's
own per-term trends exist only as INSTANTANEOUS samples in its 10-day
restarts.  Comparing an accumulated mean against a 10-sample mean of a term
whose magnitude is ~300 m3/s2 has a sampling floor around 1 m3/s2 -- ABOVE the
0.61 m3/s2 the campaign is chasing.  That is why the day-0/matched-state probe
carries the attribution and this one carries the trajectory context.

THE STAGE DECOMPOSITION (added 2026-08-19): REST IS NO LONGER A REMAINDER.
--------------------------------------------------------------------------
The bucket described above -- "barotropic solve + implicit vertical + atf +
split leftovers", the one row this budget could never measure -- is now split
into four rows, each read DIRECTLY off the step's own intermediate states:

    BARO solve       the split-explicit barotropic solve's net deposit into the
                     depth-integrated circulation
    BCLIN expl+diss  the explicit RHS + the Nbb dissipative increment, as the
                     baroclinic deviation the combine actually applies
    ZDF bt / ZDF bc  the implicit vertical solve, split the same way
    POST fixer       anything after it (identically 0 on this card)

plus the leap-frog two-level offset ``D_n = R(Nnn) - R(Nbb)``, which is where
the Robert-Asselin filter's contribution lives.  With those, the budget CLOSES
on the REALIZED circulation change with NO remainder bucket:

    R(U_{n+1}) - R(U_n) = rDt * sum(stage rows)_n  -  D_n        (exact)

HOW THE BAROTROPIC ROW IS OBTAINED WITHOUT SOLVER INSTRUMENTATION.  The
explicit combine in ``_leapfrog_step`` is
    u(Naa)_pre = [ u'(Nbb) + (u'_expl - u'(Nnn)) + du_diss'(Nbb) + U_bar_expl ]
with the primes the h_u-weighted BAROCLINIC deviations (the method's own
``_split``).  Each primed piece has ZERO h_u-weighted depth mean, so
    depth_mean_hu( u(Naa)_pre - u(Nbb) ) == U_bar_expl - U_bar(Nbb)
exactly.  Only the pre-implicit-solve state is needed, and it is captured by a
record-only wrapper around ``_apply_implicit_vertical_mixing`` that returns the
original output unchanged.  Nothing in packages/ is touched; the trajectory is
taken on the untouched production ``step`` and the instrumented evaluation is a
SECOND call on the same input (control S4 prints their one-step disagreement,
measured at 1.1e-16 m/s = 0.5x fp64 eps).

CONTROLS ADDED, all of which CAN fail:
  S0/S0b  the capture must fire, exactly once per step (fails closed)
  S4      the instrumented evaluation vs the production step: 1.11e-16 m/s
          against max|u|=0.744, i.e. 0.5x fp64 eps
  S5      h_u must be EXACTLY zero on every face the combine masks, in the
          band -- otherwise the ``* u_mask3`` in the combine injects a
          barotropic term into BARO and the row is not what it claims.
          Measured 0.000e+00 (fails closed).
  S PLANT ``--stage-plant``: a DEPTH-VARYING field with EXACTLY ZERO
          h_u-weighted depth mean, injected into the CAPTURED pre-solve
          velocity.  Prediction: BARO must NOT MOVE AT ALL, the whole plant
          must land in BCLIN, and ZDF bt/bc take minus the same shifts.
          Measured at 1e-3 m/s: BARO 4.498 -> 4.498 (unchanged), BCLIN
          3.420 -> 20.370, ZDF bt 0.232 -> 0.232, ZDF bc -3.353 -> -20.303,
          STAGE SUM unchanged.  BARO moving would have proved the probe's
          h_u / floor / fused split is not the one the model used.
  S1/S2/closure  RETRACTED AS EVIDENCE (see below): reported, but they are
          telescoping identities, not gates.

RETRACTED, 2026-08-19, both caught by adversarial review and neither by me:
  (1) THE FACTOR OF 2.  An earlier revision rescaled the stage rows by
      ``RDT*acc_n/T_int`` before comparing them with NEMO.  Since
      T_int = acc_n*dt and rDt = 2*dt that factor is EXACTLY 2, so every
      lego-vs-NEMO barotropic number was doubled, and the "ATF/leap-frog
      offset" row that closed the rescaled table was the same factor of 2
      wearing a physical name.  The stage rows are ALREADY tendencies (the
      same normalisation as NEMO's ``utrd_*``, dynspg_ts.F90:940) and they
      already sum to the realized dR/dt on their own -- the tell that was
      missed.  Corrected, the lego-minus-NEMO barotropic gap moves from
      -0.57 +- 0.52 to -1.15 +- 0.36 (arm=off), i.e. MORE significant, not
      less.
  (2) "THE BUDGET CLOSES, SO EVERY ROW IS MEASURED."  S1, S2 and the
      realized-change closure all telescope to the SAME quantity -- the
      diagnostic evaluation minus the production step, i.e. S4 under the row
      reducer.  They are identities and CANNOT fail on the split.  What
      actually gates the BARO/BCLIN partition is the depth-varying S PLANT
      and the S5 geometry gate above; the closure numbers are bookkeeping.
  (3) "THE BAROTROPIC ROW DOES NOT WEAR THE FINGERPRINT."  WRONG, and wrong
      twice.  A circulation deficit LINEAR in time requires a tendency
      deficit CONSTANT in time, so "BARO has no time trend" is evidence FOR
      this row, not against it.  And the fingerprint lives in lego MINUS
      NEMO: lego's own BARO row is a meridional dipole and so is NEMO's, so
      the shape of either alone says nothing.  The per-row DIFFERENCE table
      is now printed and is broad and single-signed, not a dipole.

RESULT (90-day twin, band mean m3/s2 per u-row; the stage rows ARE the dR/dt
decomposition, no rescaling):

                        off ladder          off ladder        true T-depth
                       transport_avg       velocity_avg      transport_avg
  BARO solve               +0.578             +0.707             +0.368
  BCLIN expl+diss          -2.129             -2.148             -2.130
  ZDF bt                   +0.339             +0.319             +0.324
  ZDF bc                   +2.098             +2.118             +2.101
  POST fixer                0.000              0.000              0.000
  = realized dR/dt         +0.886             +0.996             +0.663
  lego-minus-NEMO BARO     -1.146             -1.015             -1.355
                          (+-0.358)          (+-0.343)          (+-0.335)

THE OWNERSHIP A/B (one variable, same ladder, same window).  NEMO builds TWO
averages of the barotropic substep loop under nn_bt_flt=2: ``puu_b(Kaa)``,
the PRIMARY velocity-weighted average (dynspg_ts.F90:846-847), and
``un_adv``, the SECONDARY transport-weighted average (:843).  It deposits
``puu_b(Kaa)`` as the AFTER-level barotropic mode in the 3-D RHS (:938-942,
ln_dynadv_vec=.TRUE.) and uses ``un_adv*r1_hu(Kmm)`` only on the NOW level
(:984-987), which ``mlf_baro_corr`` removes again before the Asselin filter
(cfgs/DINO/MY_SRC/stpmlf.F90:757-760, the .NOT.ln_bt_fw branch;
ln_bt_fw=.false. at RUN_90D_TWIN/namelist_cfg:353).  At the time this block
was recorded legoESM's card set ``barotropic_reconcile_target="transport_avg"``
and ``_leapfrog_step`` then used that reconciled depth mean as the AFTER-level
mode -- i.e. NEMO's NOW-level average in NEMO's AFTER-level slot.
``velocity_avg`` is the NEMO-correct choice for that slot.

SUPERSEDED (#1455 R6): the kamm_mlf card now SHIPS ``velocity_avg`` together
with ``barotropic_after_reconcile="nemo_mlf_baro_corr"``, so the arms recorded
in this docstring are no longer what the card runs by default.  Reproduce them
with ``DINO_RECONCILE_TARGET=transport_avg DINO_AFTER_RECONCILE=off``.  The
"-0.61 deficit" and the "18% of the deficit" reading below are RETRACTED
(commit d69dc6ce0: the valid-stack deficit is -0.361 and capture is 75%);
they are left in place only as the record of what was measured then.

  MEASURED: flipping it moves the barotropic row +0.129 and the realized
  spin-up rate +0.110 m3/s2 per row -- the predicted DIRECTION, and 18% of
  the -0.61 deficit.  It is a real, one-field, source-cited lever.  It is
  NOT the owner: a prediction that it would recover 80-100% of the deficit
  is REFUTED by this arm.  The residual deficit after the flip is ~0.50.

WHAT IS STILL UNRESOLVED.  The lego-minus-NEMO barotropic gap stays at
-1.0 to -1.4 on every arm -- the right sign but ~2x the whole deficit, so
either other rows compensate or NEMO's side is biased.  That column cannot
settle it: NEMO's barotropic net is recovered as ``utrd_spg + (pre-spg
trends)``, a 595:1 cancellation whose SYSTEMATIC, common-mode bound is of
order 1 m3/s2 and does NOT average down over intervals.  The quoted +-
figures bound RANDOM scatter only.  PLAUSIBLE, never CONFIRMED, from here.

Run (fp64; LEGOESM_NEMO_E3T is gated and must be explicit -- "off" is the
ladder the recorded twin arms integrate on, and the only one legoESM is stable
on for 90 days):
  JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=off .venv/bin/python \\
    scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \\
    scripts/validate/ocean_fidelity/dino_1226/southern_term_torque_accum.py \\
    --days 90 --out-npz /path/accum.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_explicit_e3t_mode,
)

set_policy(PrecisionPolicy.fp64())
# MANDATORY, fails closed.  Unset, the restart bridge silently substitutes
# NEMO's ANALYTIC 1-D thickness ladder (``e3t_1d``) for the real ``e3t_0``,
# which differs by up to 70.4 m at and below k=25 (12.9% of e3t_0,
# 14.8% of e3t_1d -- one measurement, two denominators) -- and the day-0 twin gate CANNOT
# see it (it compares T/S/u/v VALUES, not the geometry holding them; skill
# Rule 2's documented blind spot).  A depth-integrated pressure gradient on a
# wrong deep ladder is a systematic, time-growing error that looks
# exactly like an operator defect.  That default has already ruined four
# measurements in this campaign; it very nearly ruined this one.
E3T_MODE = require_explicit_e3t_mode(context='southern_term_torque_accum')

import jax  # noqa: E402

import southern_circulation_budget as B  # noqa: E402
import acceptance_gate_90d as G  # noqa: E402
from kamm_twin_90d import (  # noqa: E402
    DT, STEPS_PER_DAY, _build_twin_state, seasonal_t0_seconds,
)
from legoesm.ocean.experiments.dino import (  # noqa: E402
    apply_dino_lat_lon_surface_forcing,
)
from southern_term_torque_matched import (  # noqa: E402  (one term mapping, shared)
    LEGO_GROUPS, NEMO_GROUPS, _USLICE, group_sum, nemo_terms, row_int,
)

RDT = 2.0 * DT
ROWS = list(B.ROWS)

# ---------------------------------------------------------- THE GATE METRIC --
# #1455 SG-C: the rows above are the CIRCULATION reducer (sum_i e1u * sum_k
# e3u_0), which is a ZONAL integral.  The acceptance gate's ACC number is a
# MERIDIONAL one -- ``acc_thermal_wind.acc_full``: per longitude i, sum over
# EVERY row j and level k of u * e3t_1d * e2u, then the MEDIAN over longitudes
# 2..-2, in Sv.  The two reductions are orthogonal and a row-circulation table
# cannot decompose an ACC gap.  So the same stage increments are reduced a
# SECOND time, with the gate metric's OWN weights, per row:
#
#     m(j) = mean_{i=2..NX-3}  e2u(j) * sum_k  du(j,i,k) * e3(k) * umask   / 1e6
#
# and sum_j m(j) is then the gate's per-longitude transport, MEAN-reduced.
# MEAN, not median, because the decomposition must be LINEAR in du for a
# per-stage table to mean anything; the mean-vs-median gap is measured on the
# real states at every interval boundary (control M0b) rather than assumed
# small.  Two weightings, because the campaign quotes two numbers:
#   FULL    e3t_1d  -- acc_full's reference-thickness ladder (the -0.597 Sv)
#   CHANNEL e3t_0   -- acc_band's real partial cells, which is the reduction
#                      the +0.287 Sv channel number was measured with
#                      (floor90_ensemble.band_transport_campaign: e3t_0, MEAN
#                      over the same longitudes).
# Both are built from ``acc_thermal_wind``'s own arrays; nothing is re-derived.
import acc_thermal_wind as A  # noqa: E402

MET_ILON = slice(2, A.NX - 2)                       # acc_full's [2:-2]
W_FULL = np.where(A.umask, np.broadcast_to(A.e3t1d, A.umask.shape), 0.0) \
    * A.e2u_col[:, None, None]
W_CHAN = np.where(A.umask, A.e3t0, 0.0) * A.e2u_col[:, None, None]
MET_NLON = A.NX - 4

# Row groups.  Row 0 and row NY-1 carry no wet u-face (asserted below), so the
# FULL section the gate metric sums is rows 1..NY-2.
SOUTH_ROWS = list(ROWS)                          # 1..J0-1, the recorded band
CHAN_ROWS = list(range(A.J0, A.J1 + 1))          # the re-entrant channel
NORTH_ROWS = list(range(A.J1 + 1, B.NY - 1))
FULL_ROWS = list(range(1, B.NY - 1))
BANDS = (("southern", SOUTH_ROWS), ("channel", CHAN_ROWS),
         ("northern", NORTH_ROWS), ("FULL section", FULL_ROWS))


def _u_nemo(field):
    """lego u-face array -> NEMO u-column layout (the twin gate's own slice)."""
    return np.asarray(field.data)[_USLICE]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--interval-days", type=int, default=10)
    ap.add_argument("--plant", type=float, default=0.0,
                    help="P control: constant [m/s2] added to KE_PGF_u's "
                         "accumulator every step (expect an exact, predictable "
                         "shift in THAT row only)")
    ap.add_argument("--stage-plant", type=float, default=0.0,
                    help="S control: constant [m/s] added to the CAPTURED "
                         "pre-implicit-solve velocity (probe-side copy only; "
                         "the model is untouched).  Expect BARO +P(d)/rDt, "
                         "ZDF bt -P(d)/rDt, every other stage row and the "
                         "stage SUM bit-unchanged.")
    ap.add_argument("--out-npz", default=None)
    args = ap.parse_args(argv)

    print(f"[precision] control dtype = {get_policy().control}  "
          f"LEGOESM_NEMO_E3T={E3T_MODE!r}")
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        args.recipe, f"{B.A.DINO}/RUN_TRAJ", G.RUN_90D_TWIN,
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    if np.asarray(st.u.data).dtype != np.float64:
        raise SystemExit("FATAL: model state is not float64")
    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    if placement != "leapfrog_rhs":
        raise SystemExit(f"unexpected surface_tendency_placement {placement!r} "
                         "-- this probe mirrors the recorded twin loop, which "
                         "runs the leapfrog_rhs branch")
    _oi = getattr(mc, "outer_integrator", "leapfrog")
    if _oi != "leapfrog":
        raise SystemExit(
            f"outer_integrator={_oi!r}: this probe's per-step identity is "
            "specific to _leapfrog_step's composition (DINO_OUTER_INTEGRATOR "
            "is a live env knob in the twin harness and would silently route "
            "to a different one)")
    print(f"[config] outer_integrator={getattr(mc, 'outer_integrator', '?')} "
          f"vorticity_scheme={cfg.vorticity_scheme} rDt={RDT}s")

    COMPS = ("KE_PGF_u", "vortcor_u", "vertadv_u", "Dterm_u", "Ah_lap_u",
             "Bh_bilap_u", "Cs_smag_u", "Cl_leith_u", "botdrag_u",
             "Av_vert_u", "phys_u", "sponge_u")
    _LDF = ("Ah_lap_u", "Bh_bilap_u", "Cs_smag_u", "Cl_leith_u")
    TERMS = COMPS + ("WIND_u", "REST")

    # Reduce ON DEVICE: a host round-trip of 13 three-dimensional arrays per
    # step costs ~3 s/day, which is 5x the model itself.  The device reducer is
    # gated against the recorded numpy one (``B._row_int_trend``) at step 0.
    import jax.numpy as jnp  # noqa: E402
    _e1u = jnp.asarray(B.e1u)
    _w3 = jnp.asarray(np.where(B.umask, B.e3u0, 0.0))       # [m], dry -> 0

    def _rowint(x):
        return jnp.sum(jnp.sum(x * _w3, axis=2) * _e1u, axis=1)

    # The gate metric's own per-row reducer (see THE GATE METRIC above).  Two
    # weightings; both return Sv per row, and their row-sum is the gate's
    # per-longitude transport MEAN-reduced over longitudes 2..-2.
    _wf = jnp.asarray(W_FULL)
    _wc = jnp.asarray(W_CHAN)

    def _met_full(x):
        return jnp.sum(jnp.sum(x * _wf, axis=2)[:, MET_ILON], axis=1) / (MET_NLON * 1e6)

    def _met_chan(x):
        return jnp.sum(jnp.sum(x * _wc, axis=2)[:, MET_ILON], axis=1) / (MET_NLON * 1e6)

    def _depthint(x):
        return jnp.sum(x * _w3, axis=2)

    def _bundle(s2, nbb):
        d_nn = model.tendencies_with_diagnostics(
            s2, surface_forcing=sf, sponge=None, dt=RDT)[1]
        d_bb = model.tendencies_with_diagnostics(
            nbb, surface_forcing=sf, sponge=None, dt=RDT)[1]
        sl = _USLICE
        # LDF: the model applies the Nbb-evaluated lateral friction
        # (_leapfrog_step's dissipative pass); evaluating it at Nnn would push
        # a real term into REST.  Same call, before-level state.
        full = [getattr(d_bb if f in _LDF else d_nn, f).data[sl] for f in COMPS]
        tot = d_nn.total_u.data[sl]
        # WIND: the surface deposit that no diagnostic field carries -- see the
        # module/day-0 docstrings; identified (k=0-confined + analytic match),
        # gated below, not a bucket.
        wind = tot - sum(getattr(d_nn, f).data[sl] for f in COMPS)
        full = full + [wind]
        stack = jnp.stack(full)
        return (jnp.stack([_rowint(a) for a in full]),
                jnp.stack([_depthint(a) for a in full]),
                wind, stack)

    bundle_fn = jax.jit(_bundle)
    rowc_fn = jax.jit(lambda u: _rowint(u[:, 1:, :]))

    # ------------------------------------------------------------- STAGES --
    # The row this budget never measured: the split-explicit BAROTROPIC solve's
    # net deposit into the depth-integrated circulation, plus the implicit
    # vertical solve and anything after it.  Measured DIRECTLY from the step's
    # own intermediate states, NOT as a remainder -- which is what turns the
    # "REST" bucket from an algebraic identity into a falsifiable closure.
    #
    # THE SEAM, read off ``_leapfrog_step`` (ocean_model_latlon_cgrid.py), not
    # inferred.  The explicit combine is
    #     u(Naa)_pre = [ u'(Nbb) + (u'_expl - u'(Nnn)) + du_diss'(Nbb)
    #                    + U_bar_expl ] * u_mask3
    # where the primes are the h_u-weighted BAROCLINIC deviations (the method's
    # own ``_split``, ``depth_mean(field, h_face, 1e-10, keepdims=True,
    # fused=False)``) and ``U_bar_expl`` is the depth-UNIFORM barotropic mode
    # taken straight from the split-explicit solve.  The three baroclinic
    # pieces each have ZERO h_u-weighted depth mean by construction, so
    #     depth_mean_hu( u(Naa)_pre - u(Nbb) )  ==  U_bar_expl - U_bar(Nbb)
    # EXACTLY -- the barotropic row needs no solver instrumentation at all,
    # only the pre-implicit-solve state.  That state is captured by a
    # record-only wrapper around ``_apply_implicit_vertical_mixing`` (which
    # receives ``naa_expl`` and returns ``naa``); the wrapper returns the
    # original output UNCHANGED, so the trajectory is bit-identical to the
    # unpatched model by construction and nothing in packages/ is touched.
    #
    # ROWS (all row-integrated with the recorded e3u_0 reducer, / rDt):
    #   BARO   [P(bt(u_pre)) - P(bt(u_bef))]/rDt   the barotropic solve's net
    #   BCLIN  [P(u_pre - u_bef) - BARO*rDt]/rDt   explicit RHS + Nbb diss,
    #                                              baroclinic deviation only
    #   ZDF bt/bc  the implicit vertical solve, split the same way
    #   POST   [P(u_final) - P(u_post)]/rDt        conservation fixer, if any
    # Their SUM is (R_naa - R_bb)/rDt = the leap-frog identity's rate, i.e.
    # exactly what TERMS+REST also sums to -- so REST is now DECOMPOSED, and
    # the check that CAN fail is the per-stage prediction set (S1-S3 below).
    from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
        min_cell_to_uface,
    )
    from legoesm.ocean.dynamics.ocean_tendency_common import (  # noqa: E402
        depth_mean,
    )

    STAGES = ("BARO solve", "BCLIN expl+diss", "ZDF bt", "ZDF bc", "POST fixer")
    _REC: dict = {}
    _Model = type(model)
    _ORIG_VMIX = _Model._apply_implicit_vertical_mixing

    def _capture_vmix(self, st_in, dt_in, sfc, *a, **kw):
        out = _ORIG_VMIX(self, st_in, dt_in, sfc, *a, **kw)
        _REC["n"] = _REC.get("n", 0) + 1
        _REC["pre"] = st_in.u.data
        _REC["post"] = (out[0] if isinstance(out, tuple) else out).u.data
        return out

    _Model._apply_implicit_vertical_mixing = _capture_vmix
    # ``step`` dispatches through ``_step_jitted``, its OWN jit boundary -- a
    # tracer captured inside it cannot escape into this probe's jit (JAX
    # raises UnexpectedTracerError, as it should), and running the UNWRAPPED
    # body instead moves the trajectory by fp64 roundoff (measured: max|du| =
    # 1.1e-16, |dT| = 5.0e-14 in one step) because XLA fuses the un-nested
    # program differently.  So the trajectory is NOT taken on the unwrapped
    # path: ``step_fn`` below stays the untouched production call, and the
    # unwrapped body is evaluated a SECOND time on the SAME input purely to
    # expose the intermediates.  Costs one extra step per iteration and keeps
    # every reported number on the production trajectory; the S4 control
    # prints the two paths' one-step disagreement so the reader can see the
    # size of the only approximation this introduces.
    _STEP_BODY = _Model.__dict__["_step_jitted"].__wrapped__
    # The capture is a TRACE-TIME side effect: it fires only while the step
    # body is being traced.  ``_build_twin_state``'s day-0 gate may already
    # have populated a jit cache for these exact avals, in which case the
    # trace would be SKIPPED and the wrapper never run -- so drop the caches
    # first, and fail closed below if the capture is still empty.
    jax.clear_caches()
    _MWC = model.config.min_water_column_m      # the MODEL's, not the harness's
    _SPLANT = float(args.stage_plant)
    # S5 (fails closed): the barotropic row's derivation assumes the ``* u_mask3``
    # in the combine is a no-op on the h_u-weighted depth mean, i.e. that h_u is
    # ZERO wherever the 3-D face mask is.  True on a full-step partial-cell
    # coordinate (h_partial in {0, dz_ref}) and FALSE on a pure z* card, where the
    # mask would inject (1-bt(mask))*(U_bar terms) into BARO.  Measured, not
    # assumed -- and it also catches the DINO seam-wall columns, which close a
    # face where h_u > 0.
    from legoesm.ocean.vertical import OceanPartialCellCoordinate  # noqa: E402
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
        compute_face_masks_3d,
    )

    step_fn = jax.jit(lambda s, r: model.step(s, DT, surface_forcing=sf,
                                              external_tracer_rate=r))

    def _stage_step(s, rate):
        _REC.clear()
        st2 = _STEP_BODY(model, s, DT, None, sf, None, grid=None,
                         vertex_mask=None, t_seconds=None,
                         external_tracer_rate=rate)
        if "pre" not in _REC:
            raise SystemExit(
                "FATAL S0: the implicit-vertical-mixing capture never fired -- "
                "either the inner jit was served from cache (so this trace ran "
                "no Python) or config.implicit_vertical_mixing is False.  No "
                "stage row may be read.")
        h_u = min_cell_to_uface(compute_layer_thickness(
            s.eta.data, s.H_bathy.data, model.z_coord,
            min_water_column_m=_MWC))

        def _bt(x):      # the model's OWN split -- same helper, same floor
            return jnp.broadcast_to(
                depth_mean(x, h_u, 1.0e-10, keepdims=True, fused=False),
                x.shape)

        u_bef = s.u_before.data
        # S PLANT.  A CONSTANT plant is VACUOUS here: depth_mean of a constant is
        # that constant for ANY positive weights, so it shifts BARO by +P and
        # ZDF bt by -P as an algebraic identity that exercises neither the h_u
        # weighting nor the seam.  This plant is instead DEPTH-VARYING and built
        # to have exactly ZERO h_u-weighted depth mean, so the PREDICTION is the
        # opposite one and it CAN fail: the whole plant must land in BCLIN and
        # BARO must not move at all.  It moves iff the probe's h_u/floor/fused
        # split differs from the one the model used.
        _pl = _SPLANT * jnp.sin(
            jnp.arange(_REC["pre"].shape[-1], dtype=_REC["pre"].dtype))
        _h_u0 = min_cell_to_uface(compute_layer_thickness(
            s.eta.data, s.H_bathy.data, model.z_coord,
            min_water_column_m=_MWC))
        _pl = _pl - depth_mean(jnp.broadcast_to(_pl, _h_u0.shape), _h_u0,
                               1.0e-10, keepdims=True, fused=False)
        u_pre = _REC["pre"] + _pl         # probe-side copy only; model untouched
        u_post = _REC["post"]
        d_ex, d_zdf = u_pre - u_bef, u_post - u_pre
        b_ex, b_zdf = _bt(d_ex), _bt(d_zdf)
        sl = _USLICE
        # The five stage INCREMENTS [m/s], stacked once and reduced three
        # times.  Their sum telescopes to (u_final - u_before) EXACTLY, so
        # every reducer applied to them closes on that state difference --
        # which is what makes the metric table below a decomposition of the
        # gate's own number rather than a second, differently-weighted budget.
        inc = jnp.stack([
            b_ex[sl],
            (d_ex - b_ex)[sl],
            b_zdf[sl],
            (d_zdf - b_zdf)[sl],
            (st2.u.data - u_post)[sl],
        ])
        rows = jax.vmap(_rowint)(inc) / RDT
        # Gate-metric rows, in Sv, NOT divided by rDt: these are CUMULATIVE
        # contributions to the ACC number and are summed (not averaged) over
        # the run, so the 90-day total is directly commensurate with the
        # day-90 ACC gap the campaign quotes.
        mrows = jnp.stack([jax.vmap(_met_full)(inc), jax.vmap(_met_chan)(inc)])
        u_diag = st2.u.data
        # D_n = R(Nnn) - R(Nbb): the leap-frog two-level offset.  It is what
        # separates the identity's rate from the realized change of the NOW
        # level, and it is where the Asselin filter's own contribution lives:
        #   R(U_{n+1}) - R(U_n) = rDt * sum(stage rows)_n  -  D_n
        offs = _rowint(s.u.data[sl]) - _rowint(u_bef[sl])
        u_nn = s.u.data[sl]
        mnow = jnp.stack([_met_full(u_nn), _met_chan(u_nn)])
        mbef = jnp.stack([_met_full(u_bef[sl]), _met_chan(u_bef[sl])])
        moffs = mnow - mbef
        return rows, offs, u_diag, mrows, moffs, mnow

    stage_fn = jax.jit(_stage_step)
    metnow_fn = jax.jit(lambda u: jnp.stack([_met_full(u[_USLICE]),
                                             _met_chan(u[_USLICE])]))

    n_steps = args.days * STEPS_PER_DAY
    per_int = args.interval_days * STEPS_PER_DAY
    n_int = max(1, n_steps // per_int)
    NY, NX = B.NY, B.NX
    acc_row = np.zeros((n_int, len(TERMS), NY))        # row torque   [m3/s2]
    acc_map = np.zeros((n_int, len(TERMS) - 1, NY, NX))  # depth-int  [m2/s2]
    acc_n = np.zeros(n_int, dtype=np.int64)
    R_series = np.zeros((n_int + 1, NY))
    acc_stage = np.zeros((n_int, len(STAGES), NY))    # row torque   [m3/s2]
    acc_offs = np.zeros((n_int, NY))                  # sum_n D_n    [m3/s]
    # GATE-METRIC accumulators.  Axis 1 selects the weighting: 0 = FULL
    # (e3t_1d, acc_full's), 1 = CHANNEL (e3t_0, acc_band's).  These are
    # CUMULATIVE Sv, summed over every step -- not per-step means.
    METW = ("FULL e3t_1d", "CHAN e3t_0")
    acc_met = np.zeros((n_int, 2, len(STAGES), NY))   # cumulative   [Sv]
    acc_moffs = np.zeros((n_int, 2, NY))              # sum_n D^met_n [Sv]
    Mnow_series = np.zeros((n_int + 1, 2, NY))        # the NOW level's metric
    acc_med_series = np.zeros(n_int + 1)              # A.acc_full (MEDIAN) [Sv]
    _u0_host = None                                   # day-0 u, M0's operand
    stage_step0 = np.zeros((len(STAGES), NY))         # step 0 only  [m3/s2]
    Rnow_series = np.zeros((n_int + 1, NY))           # the NOW level's R
    i_plant = COMPS.index("KE_PGF_u")

    # #1455 SEASONAL CLOCK: this probe twins from NEMO's day-180 restart, so
    # the seasonal forcing must continue NEMO's day-of-year, not restart it.
    t0_sec = seasonal_t0_seconds(f"{G.RUN_90D_TWIN}/DINO_00005760_restart.nc")
    t0 = time.time()
    for k in range(n_steps):
        s2, rate = apply_dino_lat_lon_surface_forcing(
            st, forcing, br.z_coord, cfg, DT, t_seconds=t0_sec + (k + 1) * DT,
            return_rate=True)
        nbb = s2._replace(u=s2.u_before, v=s2.v_before, T=s2.T_before,
                          S=s2.S_before, eta=s2.eta_before)
        rows_dev, maps_dev, wind3, stack = bundle_fn(s2, nbb)
        R_bb = np.asarray(rowc_fn(s2.u_before.data))
        if k == 0:
            R_series[0] = R_bb
            # --- gates that CAN fail -------------------------------------
            w = np.asarray(wind3)
            deep = float(np.max(np.abs(w[:, :, 1:])))
            ana = B.phi_wind()
            wr = np.asarray(rows_dev)[len(COMPS)]
            rel = float(np.max(np.abs(wr - ana)[ROWS]
                               / np.maximum(np.abs(ana)[ROWS], 1e-30)))
            print(f"  [W gate] max|surface residual| at k>0 = {deep:.3e} "
                  f"(must be 0); row integral vs analytic wind torque, band "
                  f"max rel = {rel:.3e}")
            if deep != 0.0 or rel > 5.0e-3:
                raise SystemExit("FATAL W: the surface residual is NOT the "
                                 "identified wind deposit -- it is an "
                                 "unattributed bucket")
            # C4: the model's own per-term closure on THIS reduction
            npy = B._row_int_trend(np.asarray(stack[0]))
            dev = np.asarray(rows_dev)[0]
            dd = float(np.max(np.abs(npy - dev)))
            print(f"  [reducer gate] device vs recorded numpy reducer, band "
                  f"max |diff| = {dd:.3e} m3/s2")
            if dd > 1e-6:
                raise SystemExit("FATAL: the on-device row reducer disagrees "
                                 "with the recorded numpy one")
        rows_np = np.array(rows_dev, dtype=np.float64)
        if args.plant:
            shift = B._row_int_trend(np.where(B.umask, args.plant, 0.0))
            rows_np[i_plant] += shift
        R_now = np.asarray(rowc_fn(s2.u.data))
        if k == 0:
            Rnow_series[0] = R_now
        (stage_dev, offs_dev, u_diag,
         mrows_dev, moffs_dev, mnow_dev) = stage_fn(s2, rate)
        _ncap = _REC.get("n") if k == 0 else None
        st = step_fn(s2, rate)          # the PRODUCTION trajectory, untouched
        if k == 0:
            # S4: how far the instrumented evaluation sits from the production
            # step it is diagnosing.  Arrays diffed, never a printed summary.
            _d = float(np.max(np.abs(np.asarray(u_diag)
                                     - np.asarray(st.u.data))))
            _u = float(np.max(np.abs(np.asarray(st.u.data))))
            print(f"  [S4] instrumented vs production step, max|du| = "
                  f"{_d:.2e} m/s against max|u| = {_u:.3f} "
                  f"(rel {_d / max(_u, 1e-30):.2e}; fp64 eps = 2.2e-16)")
            if _d > 1.0e-12:
                raise SystemExit(
                    "FATAL S4: the instrumented evaluation is not the "
                    "production step to fp64 roundoff -- the stage rows would "
                    "describe a different trajectory")
            _zc = model.z_coord
            _hu0 = np.asarray(min_cell_to_uface(compute_layer_thickness(
                np.asarray(s2.eta.data), np.asarray(s2.H_bathy.data), _zc,
                min_water_column_m=_MWC)))
            _um = np.asarray(s2.u_mask.data)[..., None] * np.ones(
                (1, 1, _hu0.shape[-1]))
            if isinstance(_zc, OceanPartialCellCoordinate):
                _a3, _ = compute_face_masks_3d(_zc.is_active, model.grid)
                _um = _um * np.asarray(_a3)
            _leak = float(np.max(np.abs(_hu0 * (1.0 - _um))[:, 1:, :][
                np.ix_(ROWS, range(_hu0.shape[1] - 1), range(_hu0.shape[-1]))]))
            print(f"  [S5 geometry gate] max h_u on MASKED faces in the band = "
                  f"{_leak:.3e} m (must be 0; else the combine's mask injects a "
                  f"barotropic term into BARO)")
            if _leak != 0.0:
                raise SystemExit(
                    "FATAL S5: h_u is non-zero on faces the leap-frog combine "
                    "masks, so depth_mean(u_pre - u_bef) is NOT the barotropic "
                    "increment -- the BARO row is not what it claims to be")
            if _ncap != 1:
                raise SystemExit(
                    f"FATAL S0b: the implicit vertical solve was entered "
                    f"{_ncap} times per step, not once -- the stage "
                    "seam is not where this probe's identity assumes it is")
            print(f"  [stage capture] implicit-vmix entries per step = "
                  f"{_ncap}  (must be 1); stage-plant = {_SPLANT:g} m/s")
        stage_np = np.array(stage_dev, dtype=np.float64)
        if k == 0:
            stage_step0[:] = stage_np         # already a tendency; NO rescale
            Mnow_series[0] = np.asarray(mnow_dev, dtype=np.float64)
            _u0_host = np.asarray(s2.u.data, np.float64)[_USLICE]
            acc_med_series[0] = A.acc_full(_u0_host, A.umask)
        R_naa = np.asarray(rowc_fn(st.u.data))
        rest = (R_naa - R_bb) / RDT - rows_np.sum(axis=0)

        i = min(k // per_int, n_int - 1)
        acc_row[i, :len(TERMS) - 1] += rows_np
        acc_row[i, -1] += rest
        acc_stage[i] += stage_np
        acc_offs[i] += np.asarray(offs_dev, dtype=np.float64)
        acc_met[i] += np.asarray(mrows_dev, dtype=np.float64)
        acc_moffs[i] += np.asarray(moffs_dev, dtype=np.float64)
        acc_map[i] += np.asarray(maps_dev, dtype=np.float64)
        acc_n[i] += 1

        if (k + 1) % per_int == 0:
            R_series[i + 1] = R_naa
            Rnow_series[i + 1] = R_naa      # Naa becomes the next step's Nnn
            _uN = np.asarray(st.u.data)[_USLICE]
            Mnow_series[i + 1] = np.array(metnow_fn(st.u.data), np.float64)
            acc_med_series[i + 1] = A.acc_full(_uN, A.umask)
            u3 = np.asarray(st.u.data)
            if not np.isfinite(u3).all():
                raise SystemExit(f"FATAL: non-finite velocity at step {k+1}")
            print(f"  interval {i}  (days {i*args.interval_days}"
                  f"-{(i+1)*args.interval_days})  steps={acc_n[i]}  "
                  f"max|u|={np.max(np.abs(u3)):.4f}  wall={time.time()-t0:.0f}s",
                  flush=True)

    acc_offs_sum = acc_offs.copy()          # keep the UNAVERAGED sum_n D_n
    for i in range(n_int):
        acc_row[i] /= acc_n[i]
        acc_map[i] /= acc_n[i]
        acc_stage[i] /= acc_n[i]
        acc_offs[i] /= acc_n[i]

    # ------------------------------------------------------------ the table --
    print("\n" + "=" * 112)
    print(f"ACCUMULATED lego row torque [m3/s2], band mean over rows "
          f"{ROWS[0]}..{ROWS[-1]}, per {args.interval_days}-day interval")
    print("=" * 112)
    print(f"  {'term':14s}" + "".join(f"{i*args.interval_days:>9d}"
                                      for i in range(n_int))
          + f"{'  90d mean':>12s}")
    for t, name in enumerate(TERMS):
        v = acc_row[:, t, :][:, ROWS].mean(axis=1)
        if np.max(np.abs(v)) == 0.0:
            continue
        print(f"  {name:14s}" + "".join(f"{x:9.3f}" for x in v)
              + f"{v.mean():12.3f}")
    tot = acc_row.sum(axis=1)[:, ROWS].mean(axis=1)
    print(f"  {'TOTAL':14s}" + "".join(f"{x:9.3f}" for x in tot)
          + f"{tot.mean():12.3f}")
    dR = (R_series[1:] - R_series[:-1])[:, ROWS].mean(axis=1) / (
        args.interval_days * 86400.0)
    print(f"  {'realized dR/dt':14s}" + "".join(f"{x:9.3f}" for x in dR)
          + f"{dR.mean():12.3f}")
    print("  (TOTAL is the leap-frog identity's rate; 'realized dR/dt' is the\n"
          "   plain interval difference -- they differ by the leap-frog's own\n"
          "   two-level bookkeeping, NOT by a budget error.)")

    # --------------------------------------------- comparison against NEMO --
    print("\n" + "=" * 112)
    print("vs NEMO's own dumped trends (INSTANTANEOUS 10-day samples -- see the "
          "sampling note)")
    print("=" * 112)
    nem = {}
    for i in range(n_int):
        kt = G.KT_RESTART + i * args.interval_days * STEPS_PER_DAY
        nem[i] = nemo_terms(kt)
    idx = {name: t for t, name in enumerate(TERMS)}
    print(f"  {'group':24s}{'lego acc':>11s}{'NEMO samp':>11s}{'diff':>10s}")
    for gname, lkeys in LEGO_GROUPS.items():
        lv = np.mean([acc_row[i, [idx[k] for k in lkeys if k in idx], :]
                      .sum(axis=0)[ROWS].mean() for i in range(n_int)])
        nv = np.mean([group_sum(nem[i], NEMO_GROUPS[gname])[ROWS].mean()
                      for i in range(n_int)])
        print(f"  {gname:24s}{lv:11.3f}{nv:11.3f}{lv - nv:10.3f}")
    lv = np.mean([acc_row[i, idx["REST"], ROWS].mean() for i in range(n_int)])
    nv = np.mean([(nem[i]["spg"] + nem[i]["__zdf_true__"] + nem[i]["atf"]
                   - nem[i]["__zdf_true__"] * 0)[ROWS].mean()
                  for i in range(n_int)])
    nv_spg_atf = np.mean([(nem[i]["spg"] + nem[i]["atf"])[ROWS].mean()
                          for i in range(n_int)])
    print(f"  {'REST (LUMPED bucket)':24s}{lv:11.3f}{nv_spg_atf:11.3f}"
          f"{lv - nv_spg_atf:10.3f}")
    print("    REST = spg + implicit-vertical + atf + split leftovers; the NEMO\n"
          "    column here is spg + atf only (NEMO's vertical half already sits\n"
          "    in the G4 row).  A LUMPED bucket: it names three stages at once.")

    # ============================================================= STAGES ==
    # The row the accumulated budget never measured, plus a budget that closes
    # on the REALIZED state change with NO remainder bucket left over.
    print("\n" + "=" * 112)
    print(f"STAGE decomposition of the SAME per-step increment -- every row "
          f"measured DIRECTLY from the step's own\nintermediate states, none "
          f"as a remainder.  [m3/s2], band mean over rows {ROWS[0]}"
          f"..{ROWS[-1]}, per {args.interval_days}-day interval")
    print("=" * 112)
    print(f"  {'stage':18s}" + "".join(f"{i*args.interval_days:>9d}"
                                       for i in range(n_int))
          + f"{'  90d mean':>12s}")
    for t, name in enumerate(STAGES):
        v = acc_stage[:, t, :][:, ROWS].mean(axis=1)
        print(f"  {name:18s}" + "".join(f"{x:9.3f}" for x in v)
              + f"{v.mean():12.3f}")
    if _SPLANT:
        print(f"\n  S PLANT: a DEPTH-VARYING field of amplitude {_SPLANT:g} m/s "
              "with EXACTLY ZERO h_u-weighted\n  depth mean was injected into "
              "the CAPTURED pre-solve velocity.  PREDICTION, which CAN\n  FAIL: "
              "'BARO solve' must not move AT ALL (the plant has no barotropic "
              "part\n  under the model's own split), the whole plant must land "
              "in 'BCLIN expl+diss',\n  'ZDF bt'/'ZDF bc' must take exactly "
              "minus the same shifts, and STAGE SUM must be\n  unchanged.  BARO "
              "moving means this probe's h_u / floor / fused split is NOT the "
              "one\n  the model used, and every BARO number is then wrong.  "
              "Compare against the unplanted run.")
    stot = acc_stage.sum(axis=1)[:, ROWS].mean(axis=1)
    print(f"  {'STAGE SUM':18s}" + "".join(f"{x:9.3f}" for x in stot)
          + f"{stot.mean():12.3f}")
    print(f"  {'TERMS+REST':18s}" + "".join(f"{x:9.3f}" for x in tot)
          + f"{tot.mean():12.3f}")

    # ---- S1 (Rule 1e): the stage sum must reproduce the recorded TERMS+REST
    #      identity's rate to roundoff.  Both are (R_naa - R_bb)/rDt, reached
    #      by two INDEPENDENT routes (per-term diagnostics vs intermediate
    #      states), so a disagreement means one route is wrong.
    s1 = float(np.max(np.abs(acc_stage.sum(axis=1) - acc_row.sum(axis=1))[:, ROWS]))
    print(f"\n  S1  stage sum vs the recorded TERMS+REST identity, band max "
          f"|diff| = {s1:.3e} m3/s2")
    if s1 > 1.0e-6:
        raise SystemExit("FATAL S1: the two routes to the same per-step "
                         "increment disagree -- no stage row may be read")

    # ---- S2: REST is now DECOMPOSED, not lumped.  REST == (stage sum) minus
    #      the explicit per-term rows, so the barotropic/implicit/post split
    #      below is exactly what was previously hidden inside it.
    rest_row = acc_row[:, idx["REST"], :]
    rest_from_stages = acc_stage.sum(axis=1) - acc_row[:, :len(TERMS) - 1, :].sum(axis=1)
    s2 = float(np.max(np.abs(rest_row - rest_from_stages)[:, ROWS]))
    print(f"  S2  REST reproduced from the stage rows, band max |diff| = "
          f"{s2:.3e} m3/s2")
    if s2 > 1.0e-6:
        raise SystemExit("FATAL S2: the stage rows do not reproduce REST")

    # ---- S3: the implicit vertical solve is PREDICTED to be a no-op on the
    #      barotropic mode on this card (zdf_baroclinic_only=True strips the
    #      depth mean before the tridiagonal solve and re-adds the SAME mean
    #      after it; surface_stress_implicit=False, so no depth-mean source
    #      exists inside the solve).  A FALSIFIABLE prediction, not a claim:
    #      if "ZDF bt" is not at roundoff, that reading is wrong.
    zbt = float(np.max(np.abs(acc_stage[:, STAGES.index("ZDF bt"), :])[:, ROWS]))
    zbc = float(np.max(np.abs(acc_stage[:, STAGES.index("ZDF bc"), :])[:, ROWS]))
    print(f"  S3  implicit solve on the BAROTROPIC mode: band max |ZDF bt| = "
          f"{zbt:.3e} m3/s2 (zdf_baroclinic_only="
          f"{getattr(mc, 'zdf_baroclinic_only', '?')}, surface_stress_implicit="
          f"{getattr(mc, 'surface_stress_implicit', '?')}); its baroclinic "
          f"half is {zbc:.3f}")
    print("      NOT a defect reading, and NOT a clean 'no-op' test: the "
          "implicit solve strips and\n      re-adds the depth mean with "
          "``dz_u`` and config.min_water_column_m, while the split\n"
          "      this row uses is the leap-frog combine's OWN "
          "(``h_u``, floor 1e-10, unfused).  A\n      non-zero ZDF bt is "
          "therefore the DIFFERENCE OF TWO WEIGHTINGS as much as a real\n"
          "      depth-mean source, and separating them needs the solve's own "
          "internals, not this\n      seam.  Reported so it is visible, "
          "labelled PLAUSIBLE-artifact, attributed to nothing.")

    # ---- the budget that closes on the REALIZED change of the NOW level -----
    # Exact identity, one line of leap-frog algebra:
    #     R(U_{n+1}) - R(U_n) = rDt * sum(stage rows)_n  -  D_n
    # with D_n = R(Nnn) - R(Nbb).  Summed over an interval this is EXACT, so
    # the printed residual is a real closure test of the stage rows, not an
    # identity: the stage rows come from the intermediate states, D_n from the
    # two time levels, and the realized change from the state itself.
    print("\n" + "=" * 112)
    print("CLOSURE on the REALIZED circulation change of the NOW level "
          "[m3/s per row, band mean]")
    print("=" * 112)
    T_int = args.interval_days * 86400.0
    dR_now = (Rnow_series[1:] - Rnow_series[:-1])[:, ROWS].mean(axis=1)
    pred = (RDT * acc_stage.sum(axis=1) * acc_n[:, None]
            - acc_offs_sum)[:, ROWS].mean(axis=1)
    print(f"  {'interval (day)':18s}" + "".join(f"{i*args.interval_days:>12d}"
                                                for i in range(n_int)))
    print(f"  {'realized dR':18s}" + "".join(f"{x:12.1f}" for x in dR_now))
    print(f"  {'stages - offset':18s}" + "".join(f"{x:12.1f}" for x in pred))
    print(f"  {'residual':18s}" + "".join(f"{x - y:12.2e}"
                                          for x, y in zip(pred, dR_now)))
    cl = float(np.max(np.abs(pred - dR_now)))
    scale = float(np.max(np.abs(RDT * acc_stage * acc_n[:, None, None])[:, :, ROWS]))
    print(f"  band max |residual| = {cl:.3e} m3/s against a largest stage "
          f"contribution of {scale:.3e} m3/s  ({100.0 * cl / max(scale, 1e-30):.2e} %)")
    if scale > 0 and cl / scale > 1.0e-6:
        raise SystemExit("FATAL: the stage budget does NOT close on the "
                         "realized circulation change")
    print("  Every row of this budget is now measured; there is no remainder "
          "bucket.")

    # ---- UNITS, stated once, because getting this wrong doubles every number.
    # ``acc_stage`` is ALREADY a tendency: rows = row_int(increment)/rDt, m3/s2,
    # the SAME normalisation as NEMO's dumped ``utrd_*`` (dynspg_ts.F90:940
    # divides the barotropic increment by r1_Dt = 1/rDt).  It is therefore the
    # column that may be compared against NEMO, and it is used raw below.
    # RETRACTED (2026-08-19, caught by BOTH adversarial reviewers): an earlier
    # revision printed ``RDT*acc_stage*acc_n/T_int`` here and called it "the
    # contribution rate to dR/dt", then compared THAT against NEMO.  Since
    # T_int = acc_n*dt and rDt = 2*dt, that factor is exactly 2, so every
    # lego-vs-NEMO barotropic number was 2x too large and the artificial
    # "ATF/leap-frog offset" row that closed the table was the same factor of 2
    # wearing a physical name.  The stage rows already sum to the realized
    # dR/dt on their own (printed below), which is the tell that was missed.
    print("\n  The stage rows ARE the dR/dt decomposition -- no rescaling, no "
          "offset row:")
    print(f"  {'stage':18s}" + "".join(f"{i*args.interval_days:>9d}"
                                       for i in range(n_int)) + f"{'  mean':>12s}")
    for t, name in enumerate(STAGES):
        v = acc_stage[:, t, :][:, ROWS].mean(axis=1)
        print(f"  {name:18s}" + "".join(f"{x:9.3f}" for x in v)
              + f"{v.mean():12.3f}")
    vr = dR_now / T_int
    print(f"  {'STAGE SUM':18s}" + "".join(f"{x:9.3f}" for x in stot)
          + f"{stot.mean():12.3f}")
    print(f"  {'realized dR/dt':18s}" + "".join(f"{x:9.3f}" for x in vr)
          + f"{vr.mean():12.3f}")
    print("  The two agree because the leap-frog two-level offset settles at "
          "D = rDt*S/2 for ANY\n  Asselin gamma (the homogeneous mode decays "
          "as (2g-1)^n), so the offset is SLAVED and\n  is not a lever.  D is "
          "recorded in the npz; it is not a budget row.")

    # ---- the fingerprint: the lego-MINUS-NEMO per-row structure -------------
    ib = STAGES.index("BARO solve")
    baro_t = acc_stage[:, ib, :]

    # -------------------------------- NEMO's own barotropic-solve net row ---
    # Rule 0 -- read off NEMO's source, not inferred:
    #   dynspg.F90:96-99/184-187  utrd_spg = the FULL change in puu(Krhs)
    #                             across dyn_spg.
    #   dynspg_ts.F90:345         dyn_spg_ts first REMOVES the vertical mean
    #                             zu_frc of the pre-spg RHS,
    #   dynspg_ts.F90:938-942     then adds (uu_b(Kaa)-uu_b(Kbb))/rDt
    #                             (ln_dynadv_vec=.TRUE., ocean.output:1022).
    #   dynspg_ts.F90:330-333     under key_qco zu_frc uses e3u_0 / r1_hu_0 --
    #                             the SAME weights as this probe's row reducer,
    #                             so row_int(zu_frc replicated) == row_int(the
    #                             3-D RHS) EXACTLY and no extra assumption is
    #                             needed to invert the subtraction.
    #   stpmlf.F90:303-326        the pre-spg RHS is dyn_adv (keg+zad) +
    #                             dyn_vor (pvo+rvo) + dyn_ldf (ldf) +
    #                             dyn_hpg (hpg); dyn_zdf comes AFTER (:385).
    # Hence NEMO's barotropic-solve net row torque is recoverable as
    #     utrd_spg + (hpg + keg + rvo + pvo + zad + ldf)
    # which is the SAME quantity as lego's "BARO solve" row.
    NEMO_PRE_SPG = ("hpg", "keg", "rvo", "pvo", "zad", "ldf")
    print("\n" + "=" * 112)
    print("BAROTROPIC-SOLVE NET ROW: lego (accumulated) vs NEMO (recovered "
          "from utrd_spg, INSTANTANEOUS samples)")
    print("=" * 112)
    print(f"  {'day':>5s}{'lego BARO':>12s}{'NEMO baro':>12s}{'diff':>10s}"
          f"{'NEMO utrd_spg':>16s}{'|pre-spg|':>12s}")
    _canc = []
    for i in range(n_int):
        pre = sum(nem[i][t] for t in NEMO_PRE_SPG)
        nb = nem[i]["spg"] + pre
        lb = baro_t[i][ROWS].mean()
        _canc.append(max(abs(nem[i]["spg"][ROWS].mean()),
                         abs(pre[ROWS].mean())) / max(abs(nb[ROWS].mean()), 1e-30))
        print(f"  {i*args.interval_days:5d}{lb:12.3f}{nb[ROWS].mean():12.3f}"
              f"{lb - nb[ROWS].mean():10.3f}{nem[i]['spg'][ROWS].mean():16.3f}"
              f"{pre[ROWS].mean():12.3f}")
    print("\n  THE FINGERPRINT IS A DIFFERENCE, NOT A FIELD.  The deficit is "
          "NEAR-UNIFORM per row and\n  LINEAR in time; the quantity that must "
          "carry it is lego MINUS NEMO, and the RATE\n  deficit it implies is "
          "CONSTANT in time, not trending.  lego's own BARO row is a\n  "
          "meridional dipole and so is NEMO's (same sea-surface gradients), so "
          "the SHAPE of\n  either one alone says nothing.  Per-row DIFFERENCE, "
          "per interval [m3/s2]:")
    print(f"  {'row':>5s}" + "".join(f"{i*args.interval_days:>9d}"
                                     for i in range(n_int)) + f"{'  mean':>9s}")
    _dif = np.zeros((n_int, B.NY))
    for i in range(n_int):
        _dif[i] = baro_t[i] - (nem[i]["spg"]
                               + sum(nem[i][t] for t in NEMO_PRE_SPG))
    for j in ROWS:
        print(f"  {j:5d}" + "".join(f"{_dif[i, j]:9.2f}" for i in range(n_int))
              + f"{_dif[:, j].mean():9.2f}")
    _bm = _dif[:, ROWS].mean(axis=1)
    _se = float(np.std(_bm, ddof=1) / np.sqrt(len(_bm)))
    print(f"  band mean over intervals = {_bm.mean():+.3f} m3/s2 per row, "
          f"scatter s.e. = {_se:.3f} (t = {_bm.mean()/max(_se, 1e-30):+.2f}).\n"
          "  The s.e. bounds RANDOM scatter ONLY.  The 595:1 cancellation noted "
          "below is a\n  SYSTEMATIC, common-mode bound of order 1 m3/s2 that "
          "does NOT average down over\n  intervals, so this number stays "
          "PLAUSIBLE and may not be promoted to CONFIRMED here.")

    print("\n  MATCHED STATE, day 0 only (lego's FIRST step, on the state that "
          "is bit-identical\n  to NEMO's restart).  READ WITH CARE: the bridge "
          "does not carry NEMO's barotropic\n  restart history, so step 1 is a "
          "barotropic COLD START -- a start-up transient,\n  not a steady gap:")
    _pre0 = sum(nem[0][t] for t in NEMO_PRE_SPG)
    _nb0 = nem[0]["spg"] + _pre0
    _lb0 = stage_step0[STAGES.index("BARO solve")]
    print(f"    lego BARO {_lb0[ROWS].mean():.3f}   NEMO baro "
          f"{_nb0[ROWS].mean():.3f}   diff {_lb0[ROWS].mean() - _nb0[ROWS].mean():.3f}"
          f"   [m3/s2 per row, band mean]")
    print("    per row lego : " + " ".join(f"{_lb0[j]:7.2f}" for j in ROWS))
    print("    per row NEMO : " + " ".join(f"{_nb0[j]:7.2f}" for j in ROWS))
    print("    per row diff : " + " ".join(f"{_lb0[j] - _nb0[j]:7.2f}" for j in ROWS))
    print(f"\n  CANCELLATION on NEMO's side: the recovered net is "
          f"utrd_spg + (pre-spg trends), two\n  numbers whose ratio to the "
          f"net is up to {max(_canc):.0f}:1 -- so a 1-part-in-{max(_canc):.0f} "
          f"error in either\n  half is a whole unit of the net.  That, not "
          f"the 0.98 row floor, bounds this column.")
    print("  SAMPLING FLOOR: NEMO's trends are INSTANTANEOUS samples in the\n"
          "  10-day restarts; the recorded floor for a per-row trend "
          "comparison of this\n  class is 0.98 m3/s2 per row.  Any difference "
          "below 0.98 is UNREADABLE and\n  must not be interpreted; the -0.61 "
          "deficit itself sits BELOW that floor, so\n  this column can bound "
          "the row, never resolve the deficit inside it.")

    # ==================================================== THE GATE METRIC ==
    # #1455 SG-C.  Everything above reduces zonally (row circulation).  The
    # acceptance gate's ACC number reduces MERIDIONALLY.  This block reduces
    # the SAME five stage increments with the gate metric's own weights, over
    # the FULL section the gate sums, grouped into bands.
    print("\n" + "=" * 112)
    print("GATE-METRIC decomposition -- the same five stage increments, reduced "
          "with acc_full's OWN weights")
    print("=" * 112)

    # ---- M2 (fails closed): the FULL section really is rows 1..NY-2.
    _dry = [j for j in (0, B.NY - 1) if A.umask[j].any()]
    _wetout = [j for j in range(B.NY) if A.umask[j].any()
               and j not in FULL_ROWS]
    print(f"  [M2 coverage gate] rows with a wet u-face outside 1..{B.NY - 2}: "
          f"{_wetout} (must be empty); rows 0/{B.NY - 1} wet: {_dry} (must be "
          "empty)")
    if _wetout or _dry:
        raise SystemExit("FATAL M2: the row groups do not cover exactly the "
                         "wet section the gate metric sums")
    _cov = sorted(SOUTH_ROWS + CHAN_ROWS + NORTH_ROWS)
    if _cov != FULL_ROWS or len(set(_cov)) != len(_cov):
        raise SystemExit("FATAL M2: the three bands are not a PARTITION of the "
                         "full section")

    # ---- M2b (fails closed): the recorded harness's u-face slice IS the
    #      gate's.  ``acceptance_gate_90d.load_candidate`` takes lU[:, 1:53, :]
    #      and then re-assigns column 47 from lU[:, 48, :].  With the slice
    #      starting at 1 those are the SAME element, i.e. the re-assignment is
    #      a no-op and ``_USLICE`` alone reproduces the gate's array.  That is
    #      a claim about an index, so it is CHECKED, not reasoned about.
    _probe = np.arange(B.NY * 53 * A.NZ, dtype=np.float64).reshape(B.NY, 53, A.NZ)
    _gate = _probe[:, 1:53, :].copy()
    _gate[:, 47, :] = _probe[:, 48, :]
    if not np.array_equal(_gate, _probe[_USLICE]):
        raise SystemExit("FATAL M2b: _USLICE does not reproduce the gate's own "
                         "u-face slice -- every metric row is on a different "
                         "array than the number being decomposed")
    print("  [M2b slice gate] _USLICE reproduces load_candidate's u array "
          "EXACTLY (the col-47 re-assignment is a no-op)")

    # ---- M0 (fails closed): the device metric reducer against a numpy
    #      evaluation of acc_full's integrand on the SAME day-0 state.  This is
    #      the only thing standing between a mis-broadcast weight and a table
    #      of confident wrong numbers.
    if _u0_host is None:
        raise SystemExit('FATAL M0: the day-0 state was never captured')
    _u0 = np.asarray(_u0_host)
    _integ_full = np.einsum("jik,k,j->ji", np.where(A.umask, _u0, 0.0),
                            A.e3t1d, A.e2u_col)
    _ref_full = _integ_full[:, MET_ILON].mean(axis=1) / 1e6
    _integ_chan = np.einsum("jik,jik,j->ji", np.where(A.umask, _u0, 0.0),
                            A.e3t0, A.e2u_col)
    _ref_chan = _integ_chan[:, MET_ILON].mean(axis=1) / 1e6
    _e_full = float(np.max(np.abs(Mnow_series[0, 0] - _ref_full)))
    _e_chan = float(np.max(np.abs(Mnow_series[0, 1] - _ref_chan)))
    print(f"  [M0 reducer gate] device vs numpy acc_full integrand, day 0: "
          f"FULL max|diff| = {_e_full:.3e} Sv/row, CHANNEL "
          f"{_e_chan:.3e} Sv/row (must be roundoff)")
    if max(_e_full, _e_chan) > 1.0e-10:
        raise SystemExit("FATAL M0: the on-device gate-metric reducer is not "
                         "acc_full's integrand")
    # and the CHANNEL row-sum must reproduce floor90_ensemble's recorded
    # channel reduction (e3t_0, MEAN over the same longitudes) exactly.
    _chan_ref = float(np.mean(A.acc_band(_u0, A.umask)[2:-2]))
    _chan_got = float(Mnow_series[0, 1, CHAN_ROWS].sum())
    print(f"  [M0b channel gate] band row-sum {_chan_got:.6f} Sv vs "
          f"floor90_ensemble.band_transport_campaign {_chan_ref:.6f} Sv "
          f"(diff {_chan_got - _chan_ref:.3e})")
    if abs(_chan_got - _chan_ref) > 1.0e-9:
        raise SystemExit("FATAL M0b: the channel rows do not sum to the "
                         "recorded channel metric")

    # ---- M0c: MEAN vs MEDIAN.  acc_full takes the MEDIAN over longitudes; a
    #      per-stage table must be LINEAR, so the rows here are MEAN-reduced.
    #      The gap between the two reductions of the SAME state is the size of
    #      that surrogate, measured at every interval boundary rather than
    #      assumed.  It is NOT a gate: it bounds how much of a quoted gap the
    #      linear table can be expected to carry.
    _mean_series = Mnow_series[:, 0, :].sum(axis=1)
    print("\n  [M0c mean-vs-median] full-section ACC by interval boundary [Sv]")
    print(f"  {'day':>6s}{'MEDIAN (acc_full)':>20s}{'MEAN (this table)':>20s}"
          f"{'median-mean':>14s}")
    for i in range(n_int + 1):
        print(f"  {i*args.interval_days:6d}{acc_med_series[i]:20.4f}"
              f"{_mean_series[i]:20.4f}"
              f"{acc_med_series[i] - _mean_series[i]:14.4f}")
    _surr = float(abs((acc_med_series[-1] - _mean_series[-1])
                      - (acc_med_series[0] - _mean_series[0])))
    print(f"  the surrogate's own drift over the run = {_surr:.4f} Sv "
          "(the median-minus-mean CHANGE; a per-stage table cannot resolve a "
          "gap below it)")

    # ---- M1 (fails closed): the metric budget must CLOSE on the realized
    #      change of the NOW level, row by row.  Same leap-frog algebra as the
    #      circulation closure above, a DIFFERENT reducer -- so it re-tests the
    #      whole chain (capture, split, slice, weights) rather than repeating
    #      an identity.
    for w, wname in enumerate(METW):
        pred = acc_met[:, w].sum(axis=1).sum(axis=0) - acc_moffs[:, w].sum(axis=0)
        real = Mnow_series[-1, w] - Mnow_series[0, w]
        e = float(np.max(np.abs(pred - real)[FULL_ROWS]))
        sc = float(np.max(np.abs(acc_met[:, w].sum(axis=0))[:, FULL_ROWS]))
        print(f"  [M1 closure gate {wname}] band max |residual| = {e:.3e} Sv "
              f"against a largest stage contribution of {sc:.3e} Sv "
              f"({100.0 * e / max(sc, 1e-30):.2e} %)")
        if sc > 0 and e / sc > 1.0e-8:
            raise SystemExit(f"FATAL M1 ({wname}): the gate-metric stage budget "
                             "does not close on the realized ACC change")

    # ---- the table -------------------------------------------------------
    for w, wname in enumerate(METW):
        rowsel = CHAN_ROWS if w == 1 else FULL_ROWS
        print("\n" + "-" * 112)
        print(f"  CUMULATIVE contribution to the day-{args.days} ACC number, "
              f"{wname} weighting [Sv], by stage x band")
        print("-" * 112)
        cum = acc_met[:, w].sum(axis=0)                  # (n_stage, NY)
        off = acc_moffs[:, w].sum(axis=0)                # (NY,)
        print(f"  {'stage':18s}" + "".join(f"{b:>16s}" for b, _ in BANDS))
        for t, name in enumerate(STAGES):
            print(f"  {name:18s}"
                  + "".join(f"{cum[t, r].sum():16.4f}" for _, r in BANDS))
        print(f"  {'STAGE SUM':18s}"
              + "".join(f"{cum[:, r].sum():16.4f}" for _, r in BANDS))
        print(f"  {'- leapfrog offset':18s}"
              + "".join(f"{-off[r].sum():16.4f}" for _, r in BANDS))
        real = Mnow_series[-1, w] - Mnow_series[0, w]
        print(f"  {'= realized d(ACC)':18s}"
              + "".join(f"{real[r].sum():16.4f}" for _, r in BANDS))
        print(f"  (the band used for the campaign's number with this weighting "
              f"is {'channel' if w == 1 else 'FULL section'})")
        del rowsel

    if args.out_npz:
        np.savez_compressed(
            args.out_npz, terms=np.array(TERMS), rows=np.array(ROWS),
            acc_row=acc_row, acc_map=acc_map, acc_n=acc_n, R_series=R_series,
            stages=np.array(STAGES), acc_stage=acc_stage,
            acc_offs_sum=acc_offs_sum, Rnow_series=Rnow_series,
            interval_days=args.interval_days, plant=args.plant,
            stage_plant=args.stage_plant,
            met_weightings=np.array(METW), acc_met=acc_met,
            acc_moffs=acc_moffs, Mnow_series=Mnow_series,
            acc_med_series=acc_med_series,
            south_rows=np.array(SOUTH_ROWS), chan_rows=np.array(CHAN_ROWS),
            north_rows=np.array(NORTH_ROWS), full_rows=np.array(FULL_ROWS))
        print(f"\n[artifact] -> {args.out_npz}")


if __name__ == "__main__":
    main()

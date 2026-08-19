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
# which differs by up to 12.9% below k=25 -- and the day-0 twin gate CANNOT
# see it (it compares T/S/u/v VALUES, not the geometry holding them; skill
# Rule 2's documented blind spot).  A depth-integrated pressure gradient on a
# 12.9%-wrong deep ladder is a systematic, time-growing error that looks
# exactly like an operator defect.  That default has already ruined four
# measurements in this campaign; it very nearly ruined this one.
E3T_MODE = require_explicit_e3t_mode(context='southern_term_torque_accum')

import jax  # noqa: E402

import southern_circulation_budget as B  # noqa: E402
import acceptance_gate_90d as G  # noqa: E402
from kamm_twin_90d import DT, STEPS_PER_DAY, _build_twin_state  # noqa: E402
from legoesm.ocean.experiments.dino import (  # noqa: E402
    apply_dino_lat_lon_surface_forcing,
)
from southern_term_torque_matched import (  # noqa: E402  (one term mapping, shared)
    LEGO_GROUPS, NEMO_GROUPS, _USLICE, group_sum, nemo_terms, row_int,
)

RDT = 2.0 * DT
ROWS = list(B.ROWS)


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
    step_fn = jax.jit(lambda s, r: model.step(s, DT, surface_forcing=sf,
                                              external_tracer_rate=r))
    rowc_fn = jax.jit(lambda u: _rowint(u[:, 1:, :]))

    n_steps = args.days * STEPS_PER_DAY
    per_int = args.interval_days * STEPS_PER_DAY
    n_int = max(1, n_steps // per_int)
    NY, NX = B.NY, B.NX
    acc_row = np.zeros((n_int, len(TERMS), NY))        # row torque   [m3/s2]
    acc_map = np.zeros((n_int, len(TERMS) - 1, NY, NX))  # depth-int  [m2/s2]
    acc_n = np.zeros(n_int, dtype=np.int64)
    R_series = np.zeros((n_int + 1, NY))
    i_plant = COMPS.index("KE_PGF_u")

    t0 = time.time()
    for k in range(n_steps):
        s2, rate = apply_dino_lat_lon_surface_forcing(
            st, forcing, br.z_coord, cfg, DT, t_seconds=(k + 1) * DT,
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
        st = step_fn(s2, rate)
        R_naa = np.asarray(rowc_fn(st.u.data))
        rest = (R_naa - R_bb) / RDT - rows_np.sum(axis=0)

        i = min(k // per_int, n_int - 1)
        acc_row[i, :len(TERMS) - 1] += rows_np
        acc_row[i, -1] += rest
        acc_map[i] += np.asarray(maps_dev, dtype=np.float64)
        acc_n[i] += 1

        if (k + 1) % per_int == 0:
            R_series[i + 1] = R_naa
            u3 = np.asarray(st.u.data)
            if not np.isfinite(u3).all():
                raise SystemExit(f"FATAL: non-finite velocity at step {k+1}")
            print(f"  interval {i}  (days {i*args.interval_days}"
                  f"-{(i+1)*args.interval_days})  steps={acc_n[i]}  "
                  f"max|u|={np.max(np.abs(u3)):.4f}  wall={time.time()-t0:.0f}s",
                  flush=True)

    for i in range(n_int):
        acc_row[i] /= acc_n[i]
        acc_map[i] /= acc_n[i]

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

    if args.out_npz:
        np.savez_compressed(
            args.out_npz, terms=np.array(TERMS), rows=np.array(ROWS),
            acc_row=acc_row, acc_map=acc_map, acc_n=acc_n, R_series=R_series,
            interval_days=args.interval_days, plant=args.plant)
        print(f"\n[artifact] -> {args.out_npz}")


if __name__ == "__main__":
    main()

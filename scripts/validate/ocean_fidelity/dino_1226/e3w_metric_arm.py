#!/usr/bin/env python
"""#1226 y20 per-step injection: CONFIRMING ARM for the SECOND statement-level
DIFF the read named -- the W-POINT VERTICAL METRIC (``e3w``).

THE DIFF (both sides quoted, Rule 0)
------------------------------------
NEMO's implicit tracer solve divides the vertical gradient by ``e3w``:

    trazdf.F90:219-220   zwi = -p2dt*zwt(jk  )/(e3t(jk,Kaa)*e3w(jk  ,Kmm))
                         zws = -p2dt*zwt(jk+1)/(e3t(jk,Kaa)*e3w(jk+1,Kmm))

and NEMO's own ``e3w`` is the T-POINT DEPTH DIFFERENCE, not the interface
midpoint.  MEASURED, not assumed, straight from the oracle's mesh file
(RUN_SEQDUMP_Y20_1R/mesh_mask.nc, all 36 levels):

    e3w_1d(k) - (gdept_1d(k) - gdept_1d(k-1))            rel err  0.0e+00
    e3w_1d(k) - 0.5*(e3t_1d(k-1) + e3t_1d(k))            rel err  7.3e-04
                                                          rising to 3.2e-03

legoESM instead uses the interface MIDPOINT everywhere on this path:

    ocean_model_latlon_cgrid.py:6386  dz_half_cell = build_dz_half(dz_cell)
    implicit_solver.py:472-481        0.5*(dz_k + dz_{k+1})

and its ``dz_half_ref`` is the SAME midpoint, because the DINO bridge's
constructor builds the cell centres as interface midpoints and parks NEMO's
true ``gdept`` ladder in a separate field used only by the PGF:

    vertical.py:234-236  z_full_ref = 0.5*(z_half[:-1] + z_half[1:])
                         dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]
    vertical.py:238-260  t_depth_ref = NEMO gdept_1d, stored, unused here

So BOTH of legoESM's selectable divisor slots (default midpoint-of-AFTER, and
``implicit_vmix_dzw_slot`` = ``dz_half_ref*J``) are the midpoint; NEITHER is
NEMO's ``e3w``.  At the two cells the y20 per-step injection peaks on, the
midpoint is 0.167% (k=7) and 0.226% (k=10) TOO LARGE, i.e. legoESM's implicit
vertical coupling ``avt/e3w`` is that much TOO WEAK.

WHY THIS SURVIVED THE Kz EXONERATION (Rule 2 -- what the gate cannot see)
------------------------------------------------------------------------
``carry_injection_discriminator``'s B arms substituted NEMO's OWN
``avt_k``/``avm_k`` and moved the spike by ~1e-4.  That gate sees the
DIFFUSIVITY.  It is blind to the METRIC the diffusivity is divided by: both
arms kept legoESM's midpoint ``e3w``.

ARMS (one variable, Rule 7)
---------------------------
  base       the DINO card as it runs.  Continuity control: must reproduce
             dT max 5.710106e-02 @(158,42,10).
  null_e3w   ``implicit_vmix_dzw_slot=True`` with ``dz_half_ref`` UNCHANGED.
             CONTROL: on a midpoint z-star ``dz_half_ref == build_dz_half(
             dz_ref)``, so this must be BIT-IDENTICAL to ``base``.  If it is
             not, the slot itself moves the answer and the e3w arm below is
             not readable as a metric measurement.
  e3w_gdept  ``implicit_vmix_dzw_slot=True`` AND ``dz_half_ref`` replaced by
             NEMO's ``gdept(k)-gdept(k-1)``.  THE VARIABLE.
             NOTE this is deliberately the WIDE arm: ``dz_half_ref`` is also
             read by the N^2 / TKE / GM / momentum-friction paths
             (ocean_pe_latlon_cgrid.py:1821,:3457, combined.py:130,
             ocean_model_latlon_cgrid.py:6533-6534), so a NULL result here
             exonerates the whole ``dz_half`` metric family at once, which is
             the stronger statement.  A non-null result must then be split.

PREDICTION, written before the run
----------------------------------
  CONFIRMS  spike collapses >= 10x.
  REFUTES   spike moves < 2x.  PREDICTED OUTCOME IS REFUTE: a 0.2% metric
            error perturbs the vertical-mixing increment by ~0.2%, four
            orders below a 5.7e-2 K single-cell spike.  The arm is run
            anyway because the DIFF is real and at the bar, and because the
            prediction is the falsifiable part.

Run:
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
  python scripts/validate/ocean_fidelity/dino_1226/run_fp64.py \
  scripts/validate/ocean_fidelity/dino_1226/e3w_metric_arm.py
"""
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DT = 2700.0
ALL_ARMS = ("base", "null_e3w", "e3w_gdept")


def _stats(diff: np.ndarray, mask: np.ndarray) -> dict:
    m = np.broadcast_to(mask, diff.shape)
    a = np.abs(diff)
    idx = np.unravel_index(int(np.argmax(np.where(m, a, -np.inf))), a.shape)
    sel = a[m]
    return {"max": float(a[idx]),
            "argmax": tuple(int(v) for v in idx),
            "p99.9": float(np.percentile(sel, 99.9)),
            "p50": float(np.percentile(sel, 50.0))}


def main(argv: list[str]) -> int:
    arms = tuple(argv[1].split(",")) if len(argv) > 1 else ALL_ARMS
    bad = [a for a in arms if a not in ALL_ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known: {ALL_ARMS}")

    import jax
    import jax.numpy as jnp
    import multistep_replay as mr
    from legoesm.core.precision import get_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
        dino_step_surface_forcing,
    )

    if not mr.have_step1_artifacts():
        print("SKIP: oracle artifacts not present")
        return 0

    print(f"PRECISION control dtype = {get_policy().control}")
    print(f"LEGOESM_NEMO_E3T = {os.environ.get('LEGOESM_NEMO_E3T')!r}")
    print(f"arms = {arms}")

    g, br, cfg, st0 = mr.build_replay_ic()
    mc0, _ = dino_lat_lon_model_config(br.geometry, cfg)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    tmask3 = np.asarray(g.tmask) > 0.5

    # RULE 10: instantiate and PRINT the switches this arm's claims rest on.
    for f in ("tracer_advection", "outer_integrator",
              "surface_tendency_placement", "wind_through_step"):
        print(f"  CFG {f} = {getattr(cfg, f, '<absent>')!r}")
    for f in ("implicit_vmix_dzw_slot", "zdf_implicit_solver_evaluation"):
        print(f"  MC  {f} = {getattr(mc0, f, '<absent>')!r}")
    _placement = getattr(cfg, "surface_tendency_placement", None)
    _wind = bool(getattr(cfg, "wind_through_step", False))
    sf_step = dino_step_surface_forcing(forcing) if _wind else None
    _tlo = float(np.min(np.asarray(sf_step.tau_x))) if _wind else 0.0
    _thi = float(np.max(np.asarray(sf_step.tau_x))) if _wind else 0.0
    print(f"  FORCING: wind_through_step={_wind} "
          f"tau_x[Pa] range=[{_tlo:.4f},{_thi:.4f}]")

    zc0 = br.z_coord
    td = np.asarray(zc0.t_depth_ref, dtype=np.float64)
    if td.ndim != 1:
        raise SystemExit(
            "t_depth_ref absent -- LEGOESM_NEMO_E3T must supply NEMO's gdept "
            "ladder for this arm to have a faithful e3w to substitute")
    dzh_mid = np.asarray(zc0.dz_half_ref, dtype=np.float64)
    dzh_nemo = td[1:] - td[:-1]
    if dzh_nemo.shape != dzh_mid.shape:
        raise SystemExit(f"shape {dzh_nemo.shape} != {dzh_mid.shape}")
    rel = (dzh_mid - dzh_nemo) / dzh_nemo
    print(f"  GEOMETRY dtypes t_depth_ref={td.dtype} "
          f"dz_half_ref={np.asarray(zc0.dz_half_ref).dtype}")
    print(f"  D-e3w midpoint vs NEMO gdept-diff: k7={rel[6]:.4e} "
          f"k10={rel[9]:.4e} max={np.abs(rel).max():.4e} "
          f"min={np.abs(rel).min():.4e}")

    ns = mr.nemo_now_state_at(mr.IC_STEP + 1)
    states = {}
    for arm in arms:
        jax.clear_caches()
        if arm == "base":
            mc, zc = mc0, zc0
        elif arm == "null_e3w":
            mc = mc0._replace(implicit_vmix_dzw_slot=True)
            zc = zc0
        else:
            mc = mc0._replace(implicit_vmix_dzw_slot=True)
            zc = zc0._replace(
                dz_half_ref=jnp.asarray(dzh_nemo,
                                        dtype=zc0.dz_half_ref.dtype))
        model = LatLonCGridOceanModel(br.geometry, zc, mc)
        if _placement == "leapfrog_rhs":
            st, rate = apply_dino_lat_lon_surface_forcing(
                st0, forcing, zc, cfg, DT, t_seconds=DT, return_rate=True)
        else:
            st = apply_dino_lat_lon_surface_forcing(
                st0, forcing, zc, cfg, DT, t_seconds=DT)
            rate = None
        st = model.step(st, DT, surface_forcing=sf_step,
                        external_tracer_rate=rate)
        states[arm] = st
        dT = _stats(np.asarray(st.T.data) - ns.T, tmask3)
        dS = _stats(np.asarray(st.S.data) - ns.S, tmask3)
        print(f"    [{arm}] dT max={dT['max']:.6e} @{dT['argmax']} "
              f"p99.9={dT['p99.9']:.3e} p50={dT['p50']:.3e} | "
              f"dS max={dS['max']:.6e} @{dS['argmax']} "
              f"p99.9={dS['p99.9']:.3e}", flush=True)
        states[arm + "_stats"] = (dT, dS)

    if "base" in states and "null_e3w" in states:
        d = float(np.abs(np.asarray(states["null_e3w"].T.data)
                         - np.asarray(states["base"].T.data)).max())
        v = ("BIT-IDENTICAL" if d == 0.0
             else "NOT identical -- slot moves the answer, arm UNREADABLE")
        print(f"    CONTROL null_e3w vs base  max|dT| = {d:.3e} ({v})")
    if "base" in states and "e3w_gdept" in states:
        b = states["base_stats"][0]["max"]
        n = states["e3w_gdept_stats"][0]["max"]
        r = b / n if n > 0 else float("inf")
        verdict = ("CONFIRMS (>=10x)" if r >= 10.0
                   else "REFUTES (<2x)" if r < 2.0 else "PARTIAL")
        print(f"    spike ratio base/e3w_gdept = {r:.4f}  -> {verdict}")
        d = float(np.abs(np.asarray(states["e3w_gdept"].T.data)
                         - np.asarray(states["base"].T.data)).max())
        print(f"    arm EFFECT SIZE max|T_e3w - T_base| = {d:.3e} K "
              "(0 would mean the arm changed nothing at all)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

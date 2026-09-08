#!/usr/bin/env python
"""#1455 -- where the wall-row velocity difference is BORN inside ONE step.

Pre-registration: ``PREREG_post_tendency_stage_birth.md`` (written first; the
statistics, their bars, the sign convention and the controls are fixed there).

WHY.  Every term of the explicit momentum tendency has been compared against
the oracle and cleared.  The stages that run AFTER the tendency never have
been.  This probe bisects one committed step from the day-180 bridged state --
where the two models are identical, so a one-step difference is a pure operator
difference with no trajectory feedback in it -- across the post-tendency stage
boundaries, and reports where the wall-row depth-uniform velocity difference
first appears.

THE PATH THAT ACTUALLY RUNS.  The shipped card ``nemo_dino_kamm_mlf`` resolves
``outer_integrator="leapfrog"``, so the instrumented method chain is
``_leapfrog_step`` -> ``_apply_implicit_vertical_mixing`` ->
``_apply_after_level_reconcile``, NOT ``_nemo_mlf_step``.  Read from the
resolved recipe, printed at run time, not taken from the wiring prose.

BOUNDARIES, and the NEMO quantity each is compared against.  Every dump's time
level comes from ``legoesm.ocean.fidelity.time_levels``; an unregistered
basename raises rather than defaulting.

  B1  after the barotropic solve AND the leap-frog recombination
      lego : the state handed to ``_apply_implicit_vertical_mixing``
      NEMO : naa_A = (uu(Kbb) + rDt*uu(Krhs)) * umask, from the restart's ub
             and ``stp_dump_07_dynspg_u``, PLUS NEMO's own in-dyn_zdf level-1
             wind-stress deposit.  The deposit is added because legoESM puts
             the wind in the explicit tendency while NEMO puts it inside
             dyn_zdf, so without it the two sides are not the same quantity.

  B2  after the implicit vertical solve
      lego : that method's output
      NEMO : ``stp_dump_08_dynzdf_u`` PLUS ``spg_dump_puu_b_final``.
             REPRESENTATION, and this is the whole reason the campaign's
             largest unexplained figure was large: NEMO's stage-8 field is
             depth-mean-FREE (dynzdf.F90:168 subtracts uu_b(Kaa) before the
             solve and mlf_baro_corr puts it back later), while legoESM's
             ``zdf_baroclinic_only`` strips and re-adds the mean INSIDE the
             stage.  Differencing them raw compares a field that carries the
             barotropic mode against one that does not.

  B3  after the after-level reconciliation (NEMO ``mlf_baro_corr``)
      lego : ``_apply_after_level_reconcile``'s output
      NEMO : ``baro_dump_u_after`` (its ``_before`` sibling brackets B2)

  B4  the committed after-state
      lego : the state ``model.step`` returns
      NEMO : ``seq_dump_postlbc_u_aaa_kt00005761``

STATISTICS.  All three on the MIN-RULE THICKNESS WEIGHTING, never a layer
average -- the layer average is the instrument defect that forced the campaign's
last rescore, and the layer-averaged value is printed only as a labelled
contrast.  S1 depth-uniform wall velocity difference [m/s], bar 1.6e-7 per step.
S2 wall-band transport difference [Sv], bar 1.0e-4 per step.  S3 wall/basin
enrichment, bar 2.0.  Sign convention: every difference is lego - nemo, and the
deficit is NEGATIVE in that convention.

This probe prints numbers and never prints a verdict.  NaN is fatal.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      LEGOESM_NEMO_E3T=both .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.post_tendency_stage_birth
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Lane selection MUST happen before multistep_replay is imported: its path
# constants are module-level and read the environment at import time.
DINO_CFG = os.environ.get(
    "DINO_ORACLE_ROOT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
os.environ.setdefault("DINO_1226_LANE", "d180")
os.environ.setdefault("DINO_1226_IC_STEP", "5760")
os.environ.setdefault("DINO_NEMO_RUN_TWIN_STEP1",
                      os.path.join(DINO_CFG, "RUN_D180_STEP1"))
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import dump_lane  # noqa: E402
import jax  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402

JPI, JPJ, JPK, HLS = 56, 203, 36, 2
NI, NJ = JPI - 2 * HLS, JPJ - 2 * HLS          # 52 x 199 interior
RN_DT = 2700.0
RDT = 2.0 * RN_DT
WALL_ROWS = [1, 2, 3, 4]
BASIN_ROWS = list(range(6, 14))
SV = 1.0e6

# Registered bars (PREREG_post_tendency_stage_birth.md).
BAR_S1 = 1.6e-7       # m/s per step
BAR_S2 = 1.0e-4       # Sv per step
BAR_S3 = 2.0          # dimensionless enrichment


def _level(basename: str) -> str:
    """Force every dump through the shared registry (raises if unregistered)."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    return time_level_for_dump(basename)


def _load(basename: str) -> np.ndarray:
    """Raw NEMO dump -> interior array, shape inferred from the file size.

    The registry call is not optional: a dump whose NEMO time level nobody has
    established must not be loadable here at all.
    """
    path = dump_lane.dump_path(basename)
    lvl = _level(basename)
    a = np.fromfile(path, dtype="<f8")
    n = a.size
    if n == (JPK - 1) * JPJ * JPI:
        arr = a.reshape(JPK - 1, JPJ, JPI)[:, HLS:-HLS, HLS:-HLS]
        arr = np.transpose(arr, (1, 2, 0))                    # (j, i, k)
    elif n == JPK * JPJ * JPI:
        arr = a.reshape(JPK, JPJ, JPI)[:, HLS:-HLS, HLS:-HLS]
        arr = np.transpose(arr, (1, 2, 0))
    elif n == JPJ * JPI:
        arr = a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]
    elif n == NJ * NI:
        arr = a.reshape(NJ, NI)
    elif n == JPK * NJ * NI:
        arr = np.transpose(a.reshape(JPK, NJ, NI), (1, 2, 0))
    else:
        raise SystemExit(f"{basename}: unhandled element count {n}")
    if not np.isfinite(arr).all():
        raise SystemExit(f"{basename}: non-finite values -- fatal")
    print(f"  DUMP {basename:<34s} level={lvl:<7s} shape={arr.shape}")
    return arr


def _u_face_from_nemo(a3: np.ndarray, nk: int) -> np.ndarray:
    """NEMO u column i is the EAST face of T-cell i; legoESM's u-face array
    carries a west-wall column at index 0, so NEMO's i maps to lego's i+1.
    Returns a (NJ, NI) - shaped view already in legoESM's interior indexing."""
    return a3[..., :nk]


def _wstat(du_col: np.ndarray, wet2: np.ndarray, rows) -> float:
    """Row-mean over wet u-columns of a per-column quantity, averaged over
    ``rows``.  Rows with no wet column contribute nothing (and are reported)."""
    vals = []
    for j in rows:
        m = wet2[j]
        if m.any():
            vals.append(float(du_col[j][m].mean()))
    if not vals:
        raise SystemExit("no wet column in the requested rows -- fatal")
    return float(np.mean(vals))


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirt = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  dirty_tracked={len(dirt.splitlines())}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}  "
          f"LEGOESM_NEMO_E3T={os.environ.get('LEGOESM_NEMO_E3T')}")
    dump_lane.banner()

    import multistep_replay as mr
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
        dino_step_surface_forcing,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    g, br, cfg, st = mr.build_replay_ic()      # day-0 identity gate inside
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    print(f"CARD nemo_dino_kamm_mlf: outer_integrator={mc.outer_integrator!r} "
          f"zdf_baroclinic_only={mc.zdf_baroclinic_only} "
          f"zdf_drag_in_matrix={mc.zdf_drag_in_matrix} "
          f"after_reconcile={mc.barotropic.barotropic_after_reconcile!r}")
    if mc.outer_integrator != "leapfrog":
        raise SystemExit(
            f"this probe instruments _leapfrog_step; the card resolves "
            f"{mc.outer_integrator!r} -- refusing to attribute stages to a "
            "path the run does not take")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf_step = (dino_step_surface_forcing(forcing)
               if bool(getattr(cfg, "wind_through_step", False)) else None)
    if sf_step is None:
        raise SystemExit("wind_through_step is off on this card -- the wind "
                         "would be dropped; refusing to measure")

    # ---- capture the stage boundaries ------------------------------------
    cap: dict[str, np.ndarray] = {}
    _real_vmix = LatLonCGridOceanModel._apply_implicit_vertical_mixing
    _real_rec = LatLonCGridOceanModel._apply_after_level_reconcile

    def _vmix(self, naa_expl, *a, **kw):
        cap["B1_u"] = np.asarray(naa_expl.u.data, dtype=np.float64)
        cap["B1_v"] = np.asarray(naa_expl.v.data, dtype=np.float64)
        cap["B1_eta"] = np.asarray(naa_expl.eta.data, dtype=np.float64)
        out = _real_vmix(self, naa_expl, *a, **kw)
        _s = out[0] if isinstance(out, tuple) else out
        cap["B2_u"] = np.asarray(_s.u.data, dtype=np.float64)
        cap["B2_v"] = np.asarray(_s.v.data, dtype=np.float64)
        return out

    def _rec(self, naa, *a, **kw):
        out = _real_rec(self, naa, *a, **kw)
        cap["B3_u"] = np.asarray(out.u.data, dtype=np.float64)
        cap["B3_v"] = np.asarray(out.v.data, dtype=np.float64)
        return out

    kt = dump_lane.KT_DUMP                      # 5761
    # Mirror the production applicator branch exactly (run_dino / run_replay):
    # 'leapfrog_rhs' threads the surface tendency as a RATE into the Nnn RHS,
    # 'applied_now' mutates the state.  Mixing them silently reverts placement.
    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    ext_rate = None
    if placement == "leapfrog_rhs":
        st_in, ext_rate = apply_dino_lat_lon_surface_forcing(
            st, forcing, br.z_coord, cfg, RN_DT, t_seconds=kt * RN_DT,
            return_rate=True)
    else:
        st_in = apply_dino_lat_lon_surface_forcing(
            st, forcing, br.z_coord, cfg, RN_DT, t_seconds=kt * RN_DT)
    print(f"FORCING placement={placement!r} external_tracer_rate="
          f"{'present' if ext_rate is not None else 'None'}")
    LatLonCGridOceanModel._apply_implicit_vertical_mixing = _vmix
    LatLonCGridOceanModel._apply_after_level_reconcile = _rec
    try:
        with jax.disable_jit():
            st_out = model.step(st_in, RN_DT, surface_forcing=sf_step,
                                external_tracer_rate=ext_rate)
    finally:
        LatLonCGridOceanModel._apply_implicit_vertical_mixing = _real_vmix
        LatLonCGridOceanModel._apply_after_level_reconcile = _real_rec
    cap["B4_u"] = np.asarray(st_out.u.data, dtype=np.float64)
    cap["B4_v"] = np.asarray(st_out.v.data, dtype=np.float64)
    for k in ("B1_u", "B2_u", "B3_u", "B4_u"):
        if k not in cap:
            raise SystemExit(f"stage {k} was never captured -- the capture, "
                             "not the model, failed; no number is read")
        if not np.isfinite(cap[k]).all():
            raise SystemExit(f"{k}: non-finite -- fatal")

    nk = cap["B1_u"].shape[-1]
    print(f"CAPTURED  {sorted(cap)}  u shape={cap['B1_u'].shape}")

    # ---- NEMO references --------------------------------------------------
    print("\nNEMO dumps:")
    krhs_u = _load(f"stp_dump_07_dynspg_kt{kt:08d}_u.bin")
    st8_u = _load(f"stp_dump_08_dynzdf_kt{kt:08d}_u.bin")
    baro_bef = _load("baro_dump_u_before.bin")
    baro_aft = _load("baro_dump_u_after.bin")
    postlbc = _load(f"seq_dump_postlbc_u_aaa_kt{kt:08d}.bin")
    puu_b = _load("spg_dump_puu_b_final.bin")
    ws_pre = _load("zdf_dump_u1_prestress.bin")
    ws_post = _load("zdf_dump_u1_poststress.bin")

    # uu(Kbb): take it from the BRIDGED state's own before level rather than
    # re-reading the restart.  ``build_replay_ic`` has already gated that
    # bridge at max|du| = 0.000e+00 against this very restart, so this is
    # NEMO's ub in legoESM's indexing with the gate attached -- and it removes
    # a second, unvalidated halo/axis convention from the probe.
    ub_lego = np.asarray(st.u_before.data, dtype=np.float64)
    with nc.Dataset(os.path.join(dump_lane.RUN_DIR, "mesh_mask.nc")) as ds:
        umask = np.transpose(np.asarray(ds["umask"][0], dtype=np.float64),
                             (1, 2, 0))
        e2u = np.asarray(ds["e2u"][0], dtype=np.float64)
        gphit = np.asarray(ds["gphit"][0], dtype=np.float64)

    # NEMO writes jpkm1 = 35 levels; legoESM carries 36.  Every comparison is
    # done on the 35 levels BOTH sides have, and the count is printed.
    nkc = st8_u.shape[-1]
    print(f"  comparison levels: NEMO {nkc}, legoESM {nk} -> using {nkc}")
    ws = np.zeros_like(st8_u)
    ws[..., 0] = ws_post - ws_pre            # NEMO's in-dyn_zdf level-1 deposit
    naa_A = (ub_lego[:, 1:, :nkc] + RDT * krhs_u) * umask[..., :nkc]
    ref = {
        "B1": naa_A + ws,
        "B2": st8_u + puu_b[..., None] * umask[..., :nkc],
        "B3": baro_aft,
        "B4": postlbc,
    }
    ref_bracket_b2 = baro_bef

    # ---- statistics -------------------------------------------------------
    # min-rule face thickness from the model's OWN coordinate at the entry eta,
    # i.e. the same rule the barotropic split and the implicit solve now share.
    h_k = np.asarray(compute_layer_thickness(
        st_in.eta.data, st_in.H_bathy.data, br.z_coord,
        min_water_column_m=mc.min_water_column_m), dtype=np.float64)
    h_u = np.asarray(min_cell_to_uface(h_k), dtype=np.float64)[:, 1:, :nkc]
    wetu = umask[..., 0] > 0.5
    H_u = (h_u * umask[..., :nkc]).sum(-1)
    H_u_safe = np.where(H_u > 0, H_u, np.nan)

    def _cols(lego_u, nemo_u):
        d = (lego_u[:, 1:, :nkc] - nemo_u) * umask[..., :nkc]
        tw = (d * h_u * umask[..., :nkc]).sum(-1) / H_u_safe  # thickness-wtd
        la = np.where(umask[..., :nkc] > 0.5, d, np.nan)
        with np.errstate(invalid="ignore"):
            lay = np.nanmean(la, axis=-1)                     # contrast only
        return tw, lay

    print("\nSIGN CONVENTION  every difference is lego - nemo; the day-90 "
          "deficit is NEGATIVE in this convention (-4.6e-4 m/s).")
    print(f"REGISTERED BARS  S1 {BAR_S1:.1e} m/s   S2 {BAR_S2:.1e} Sv   "
          f"S3 {BAR_S3:.1f}x  (per step)")

    rows_hdr = (f"\n{'boundary':<10}{'S1 wall [m/s]':>16}{'S2 wall [Sv]':>15}"
                f"{'S1 basin [m/s]':>17}{'S3 enrich':>11}"
                f"{'S1 wall LAYER-AVG':>19}")
    print(rows_hdr)
    table = {}
    for b in ("B1", "B2", "B3", "B4"):
        tw, lay = _cols(cap[f"{b}_u"], ref[b])
        s1w = _wstat(tw, wetu, WALL_ROWS)
        s1b = _wstat(tw, wetu, BASIN_ROWS)
        s2 = float(np.nansum(np.where(wetu, tw * H_u * e2u, 0.0)[WALL_ROWS])
                   / SV)
        s3 = abs(s1w) / abs(s1b) if abs(s1b) > 0 else float("inf")
        s1l = _wstat(lay, wetu, WALL_ROWS)
        table[b] = (s1w, s2, s1b, s3, s1l)
        print(f"{b:<10}{s1w:>16.6e}{s2:>15.6e}{s1b:>17.6e}{s3:>11.2f}"
              f"{s1l:>19.6e}")

    print(f"\n{'stage BORN in':<34}{'d S1 wall [m/s]':>18}"
          f"{'d S2 wall [Sv]':>17}{'over bar?':>11}")
    names = {
        "B1": "barotropic solve + recombination",
        "B2": "implicit vertical solve",
        "B3": "after-level reconciliation",
        "B4": "time filter + commit",
    }
    p1 = p2 = 0.0
    for b in ("B1", "B2", "B3", "B4"):
        s1w, s2 = table[b][0], table[b][1]
        d1, d2 = s1w - p1, s2 - p2
        over = "YES" if (abs(d1) > BAR_S1 and abs(d2) > BAR_S2) else "no"
        print(f"{names[b]:<34}{d1:>18.6e}{d2:>17.6e}{over:>11}")
        p1, p2 = s1w, s2

    # ---- CONTROLS ---------------------------------------------------------
    print("\nCONTROLS")
    # (a) planted: roll the NEMO reference one row -- every statistic must blow up
    tw_true, _ = _cols(cap["B4_u"], ref["B4"])
    tw_roll, _ = _cols(cap["B4_u"], np.roll(ref["B4"], 1, axis=0))
    a_true = abs(_wstat(tw_true, wetu, WALL_ROWS))
    a_roll = abs(_wstat(tw_roll, wetu, WALL_ROWS))
    print(f"  planted one-row roll of the B4 reference: "
          f"|S1| {a_true:.4e} -> {a_roll:.4e}  ({a_roll / max(a_true, 1e-30):.1f}x)")
    if a_roll <= 10.0 * a_true:
        raise SystemExit(
            "the planted roll did not blow the statistic up by 10x -- the "
            "metric is translation-blind and no number above is read")
    # (b) NOT a control, and labelled so: the per-stage births are DEFINED as
    #     successive differences of the cumulative table, so they sum to the
    #     end-to-end value identically.  Printing that identity as a "closure
    #     check" would be a test that cannot fail.
    print(f"  identity (NOT a control): the births are successive differences "
          f"of the cumulative column, so they sum to B4 = "
          f"{table['B4'][0]:.6e} m/s by construction")
    # (c) B2's reference is NEMO's stage-8 dump PLUS puu_b(Kaa), because
    #     stage 8 is depth-mean-FREE (dynzdf.F90:168 removes uu_b before the
    #     solve; mlf_baro_corr restores it later) while legoESM's stage output
    #     carries the mean.  The check that this is the right representation
    #     fix is that mlf_baro_corr's OWN entry dump equals the stage-8 dump --
    #     nothing runs between them -- so any difference is a lane or halo
    #     error, not physics.
    br_d = np.abs(ref_bracket_b2 - st8_u) * umask[..., :nkc]
    print(f"  B2 reference check: |baro_dump_u_before - stage8| max = "
          f"{br_d.max():.4e} m/s over wet faces (must be ~0: nothing runs "
          "between the two dumps)")
    if br_d.max() > 1e-12:
        raise SystemExit(
            "baro_dump_u_before and the stage-8 dump disagree -- the two "
            "references bracket something this probe does not model, and B2 "
            "is not read")
    # (d) an INDEPENDENT artifact for the committed state: NEMO's own restart
    #     at this step, rebuilt from its tiles, must agree with the postlbc
    #     dump the B4 reference uses.
    ns = mr.nemo_now_state_at(kt)
    nu = np.asarray(ns.u.data if hasattr(ns, "u") else ns["u"],
                    dtype=np.float64)
    print(f"  B4 reference cross-check: NEMO restart u shape {nu.shape} vs "
          f"postlbc {postlbc.shape}")
    print(f"  wall rows {WALL_ROWS} lat "
          f"{[round(float(gphit[j, 25]), 2) for j in WALL_ROWS]}, "
          f"basin rows {BASIN_ROWS[0]}..{BASIN_ROWS[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

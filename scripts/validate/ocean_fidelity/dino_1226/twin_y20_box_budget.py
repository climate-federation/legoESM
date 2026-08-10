"""#1226 TWIN DEEP-BOX HEAT BUDGET at YEAR 20 -- the decisive attribution
experiment for the deep warm bias (established: legoESM's Southern Ocean deep
water is +0.7 degC too warm vs NEMO at y40; deficit ~0 at surface, grows
monotonically to -0.12 kg/m3 at 3757 m; entirely thermal; per-step vertmix
rows are near bar (avt corr 0.9976/ratio 0.9962), so per-step operator error
is NOT obviously the cause).

Twin protocol (state confound removed, oracle-fidelity Rule 7): BOTH models
start from NEMO's OWN year-20 restart (``RUN_REPRO_Y20/DINO_00230400_restart``,
kt=230400=20*11520), run 90 days, and we compare the accumulated per-term heat
budget of the SAME deep southern box.

This module does NOT re-derive any numerics or build a new harness -- it
re-targets the EXISTING, already-validated infrastructure:
  * legoESM twin bridge: ``kamm_twin_90d._build_twin_state`` (--bridge-tke
    --bridge-before, day-0 gate) + ``BoxHeatBudgetAccumulator``
    (``box_heat_budget.py``) -- the SAME construction ``deep_box_heat_budget.py``
    uses for the day-180 (y0.5) twin, only the restart year changes.
  * NEMO ttrd_* per-term trend dumps: already wired via MY_SRC/trddump.F90,
    gated on the standard ``namtrd`` switches ``ln_dyn_trd``/``ln_tra_trd``
    (both .true. in the RUN_REPRO_Y20/RUN_TWIN_Y20_BUDGET cards) -- captured
    into the restart file itself via iom_rstput at rst_write cadence
    (nn_stock=320, 10-day dumps). NOT the standard XIOS trend-diagnostic
    stream (that is a compiled no-op without XIOS in this build); this is the
    project's own restart-embedded instrument, already used identically by
    ``deep_box_heat_budget.py``.

NEMO ARM (already run, not re-run by this module): a NEW run directory
``RUN_TWIN_Y20_BUDGET/`` (cloned from ``RUN_90D_TWIN``'s namelist_cfg, the
day-180 twin's own template) continuing from
``RUN_REPRO_Y20/DINO_00230400_restart`` for 2880 steps (90 days at
rn_Dt=2700s, 32 steps/day -- confirmed from RUN_REPRO_Y20/namelist_cfg:116,
NOT namelist_ref's 5400s default which namelist_cfg overrides), with
nn_stock=320 (10-day trend-dump cadence, dumps at kt
230720,231040,...,233280). Ran clean (STOP 0), 9 trend restarts written.

INFRASTRUCTURE GAP FOUND AND CLOSED (not a new harness): RUN_REPRO_Y20's
restarts are 16-tile (per-rank), unlike RUN_TRAJ's single-file restart that
``read_nemo_restart`` (xarray, single dataset) expects. Stitched via NEMO's
OWN standard postprocessing tool (``tools/REBUILD_NEMO/rebuild_nemo.exe``,
already built in the oracle tree) into a single-file
``RUN_TWIN_Y20_BUDGET_STITCHED/DINO_00230400_restart.nc`` carrying
tn/sn/un/vn/sshn/tb/sb/ub/vb/sshb/en -- verified present before use. This is
the standard NEMO rebuild utility, not project-specific glue.

Usage
-----
    CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
        python twin_y20_box_budget.py lego <out.npz> [--days 90]
    JAX_ENABLE_X64=1 python twin_y20_box_budget.py nemo <out.npz>
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

_THIS_DIR = __file__.rsplit("/", 1)[0]
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _THIS_DIR.rsplit("/", 1)[0])  # scripts/validate/ocean_fidelity/ -- rebuild_nemo_restart.py
from kamm_twin_90d import _build_twin_state, RUN_TRAJ, DT, STEPS_PER_DAY  # noqa: E402

from legoesm.ocean.experiments.dino import apply_dino_lat_lon_surface_forcing  # noqa: E402
from legoesm.ocean.fidelity.box_heat_budget import (  # noqa: E402
    TERM_NAMES,
    BoxHeatBudgetAccumulator,
)
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_fp64,
    require_explicit_e3t_mode,
)

# --- restart / dump locations (this experiment's arm, not the y0.5 twin's) ---
RUN_STEPDUMP_Y20 = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TWIN_Y20_BUDGET_STITCHED"
RESTART_FILE_Y20 = "DINO_00230400_restart.nc"  # year-20 developed state (230400 = 20*11520 steps)
RUN_TWIN_Y20_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TWIN_Y20_BUDGET"
NEMO_DUMP_KTS = (230720, 231040, 231360, 231680, 232000, 232320, 232640, 232960, 233280)  # 10-day cadence
KT0 = 230400  # twin origin (both models identical here)

# --- box definitions (task brief, verbatim) ---
SOUTH_ROWS = slice(14, 23)   # "channel band T-rows 14-22" (Python slice upper bound exclusive -> 23)
FULL_ROWS = slice(14, 49)    # "rows 14-48" (exclusive upper bound -> 49)
ROW_BOXES = {"south_14_22": SOUTH_ROWS, "full_14_48": FULL_ROWS}

# Single deep band (task brief: "below 1400 m"), plus the shallower bands as a
# free cross-check (same run, no extra cost) against the already-established
# near-bar vertmix result at 0-1000m.
DEPTH_BANDS_M = ((0.0, 200.0), (200.0, 1000.0), (1000.0, 1400.0), (1400.0, None))
DEEP_BAND = (1400.0, None)

RHO0_NEMO, CP_NEMO = 1026.0, 3991.86795711963  # eosbn2.F90:1899, NEMO's own convention


# =====================================================================
# lego arm
# =====================================================================

def run_lego(recipe: str, out_path: str, n_days: int,
             surface_tendency_placement: str = "applied_now"):
    # Rule 1c (oracle-fidelity skill): JAX_ENABLE_X64=1 only PERMITS f64; the
    # legoESM precision policy is a SEPARATE axis that defaults to float32
    # and must be set explicitly BEFORE any geometry/state is constructed.
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    e3t_mode = require_explicit_e3t_mode(context="twin_y20_box_budget lego")
    print(f"LEGOESM_NEMO_E3T={e3t_mode}")

    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        recipe, RUN_TRAJ, RUN_STEPDUMP_Y20, bridge_tke=True, bridge_before=True,
        restart_file=RESTART_FILE_Y20,
    )
    require_fp64(br.geometry, br.z_coord, st, context="twin_y20_box_budget lego")
    from legoesm.ocean.fidelity.precision_gate import describe_float_leaves
    print("DTYPES (state, first 5 float leaves):",
          describe_float_leaves(st)[:5])
    print("DTYPES (z_coord, first 5 float leaves):",
          describe_float_leaves(br.z_coord)[:5])

    import jax
    import jax.numpy as jnp

    # NEMO's (rho0, cp) so BOTH arms' saved _Wm2 values share ONE convention
    # (codex review: the default Gill pair differs by conv_ratio=1.00044 —
    # negligible here, but a silent cross-arm unit split is a trap).
    accs = {name: BoxHeatBudgetAccumulator(
                br.geometry, br.z_coord, mc, cfg, forcing, DT, model,
                row_slice=rows, depth_bands_m=DEPTH_BANDS_M,
                rho0=RHO0_NEMO, cp=CP_NEMO)
            for name, rows in ROW_BOXES.items()}

    if surface_tendency_placement not in ("applied_now", "leapfrog_rhs"):
        raise SystemExit(
            f"Unknown surface_tendency_placement {surface_tendency_placement!r}: "
            "expected 'applied_now' or 'leapfrog_rhs'.")
    _use_rhs = surface_tendency_placement == "leapfrog_rhs"
    print(f"surface_tendency_placement={surface_tendency_placement}")

    nsteps = STEPS_PER_DAY * n_days
    if _use_rhs:
        dyn = jax.jit(lambda st, t, rate: model.step(
            st, DT, surface_forcing=sf, t_seconds=t, external_tracer_rate=rate))
    else:
        dyn = jax.jit(lambda st, t: model.step(st, DT, surface_forcing=sf, t_seconds=t))
    sample_dt = STEPS_PER_DAY * DT

    t_seconds = 0.0
    for acc in accs.values():
        acc.sample(st, dt_step=sample_dt, t_seconds=t_seconds)
    for k in range(nsteps):
        t_seconds = (k + 1) * DT
        if _use_rhs:
            st, ext_rate = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, DT, t_seconds=t_seconds,
                return_rate=True)
            st = dyn(st, jnp.asarray(t_seconds), ext_rate)
        else:
            st = apply_dino_lat_lon_surface_forcing(
                st, forcing, br.z_coord, cfg, DT, t_seconds=t_seconds)
            st = dyn(st, jnp.asarray(t_seconds))
        if (k + 1) % STEPS_PER_DAY == 0:
            for acc in accs.values():
                acc.sample(st, dt_step=sample_dt, t_seconds=t_seconds)
            Td = np.asarray(st.T.data)
            m = np.asarray(st.land_mask.data) > 0.5
            if not np.isfinite(Td[m]).all():
                raise SystemExit(f"NaN/Inf at day {(k+1)*DT/86400:.1f} -- aborting")
            if (k + 1) % (STEPS_PER_DAY * 10) == 0:
                print(f"  day {(k+1)*DT/86400:5.1f}  T[{Td[m].min():.2f},{Td[m].max():.2f}]", flush=True)

    total_seconds = t_seconds
    summaries = {name: acc.summary(total_seconds) for name, acc in accs.items()}

    from legoesm import constants
    conv_ratio = (RHO0_NEMO * CP_NEMO) / (constants.rho_ocean * constants.c_sw)

    print(f"\n{'='*100}\nLEGO TWIN Y20 ({total_seconds/86400:.1f}d window, "
          f"restart={RESTART_FILE_Y20}, kt0={KT0}, e3t_mode={e3t_mode})\n{'='*100}")

    save_kwargs = {"total_seconds": total_seconds, "conv_ratio": conv_ratio,
                   "restart_file": RESTART_FILE_Y20, "e3t_mode": e3t_mode, "kt0": KT0}
    for box_name, summary in summaries.items():
        print(f"\n--- box {box_name} rows={ROW_BOXES[box_name]} area={summary['box_area_m2']:.4e} m^2 ---")
        for band in DEPTH_BANDS_M:
            b = summary["terms"][band]
            lo, hi = band
            label = f"{lo:.0f}-{hi:.0f}m" if hi is not None else f"{lo:.0f}m+"
            print(f"  band {label}:")
            for term in TERM_NAMES:
                w = b[term]["W_per_m2"]
                print(f"    {term:10s} {w:+9.4f} W/m^2  (NEMO-conv {w*conv_ratio:+9.4f})")
            print(f"    {'residual':10s} {b['residual']['W_per_m2']:+9.4f} W/m^2  <- closure diagnostic")
            print(f"    dH/dt      {b['dH_J']/total_seconds/summary['box_area_m2']:+9.4f} W/m^2  "
                  f"(realized box heat-content tendency, the closure TARGET)")
        for bi, band in enumerate(DEPTH_BANDS_M):
            for term in TERM_NAMES:
                save_kwargs[f"{box_name}_band{bi}_{term}_Wm2"] = summary["terms"][band][term]["W_per_m2"]
            save_kwargs[f"{box_name}_band{bi}_residual_Wm2"] = summary["terms"][band]["residual"]["W_per_m2"]
            save_kwargs[f"{box_name}_band{bi}_dH_J"] = summary["terms"][band]["dH_J"]

    # Daily per-interval time series (year-run extension, task requirement:
    # "full-year mean + the four 90-day quarters"). accums's own
    # time_series_term_J/time_series_t already carry this at NO extra
    # compute cost (recorded every .sample() call above); saving them lets
    # any window (quarter, custom range) be sliced post-hoc without
    # re-running the model. time_series_t[0]=0 is the day-0 baseline (no
    # preceding interval -> not in time_series_term_J, whose length is
    # n_samples-1); time_series_t[1:] are the interval-closing times the
    # J entries at the SAME index belong to.
    for box_name, acc in accs.items():
        save_kwargs[f"{box_name}_ts_t"] = np.asarray(acc.time_series_t)
        for bi in range(len(DEPTH_BANDS_M)):
            for term in TERM_NAMES:
                save_kwargs[f"{box_name}_band{bi}_{term}_ts_J"] = np.asarray(
                    acc.time_series_term_J[term][bi])
            save_kwargs[f"{box_name}_band{bi}_residual_ts_J"] = np.asarray(
                acc.time_series_residual_J[bi])

    np.savez(out_path, **save_kwargs)
    print(f"\nSAVED {out_path}")


# =====================================================================
# nemo arm (already run -- this reads RUN_TWIN_Y20_BUDGET's own ttrd_* dumps)
# =====================================================================

def _rebuild_kt(kt: int, fields: list[str], rundir: str = RUN_TWIN_Y20_DIR) -> dict:
    from rebuild_nemo_restart import rebuild
    pattern = f"{rundir}/DINO_{kt:08d}_restart_*.nc"
    if not glob.glob(pattern):
        raise SystemExit(f"NEMO trend restart not found: {pattern}")
    return rebuild(pattern, fields)


def _nemo_grouped_terms(kt: int, rundir: str = RUN_TWIN_Y20_DIR) -> dict[str, np.ndarray]:
    """One kt's grouped (lev, y, x) dT/dt [K/s] terms -- SAME grouping as
    ``deep_box_heat_budget.py``'s ``_nemo_grouped_terms`` (traldf.F90/
    trazdf.F90/trdtra.F90 bookkeeping, verified there, not re-derived).

    ``adv_h``/``adv_v`` split (year-run extension, task requirement): NEMO's
    own trend bookkeeping already separates the three advective components
    (``ttrd_xad``/``ttrd_yad`` = zonal/meridional = HORIZONTAL,
    ``ttrd_zad`` = vertical = w*dT/dz -- trdtra.F90's own naming, not a
    re-derivation) -- ``adv`` (h+v combined) is kept alongside for backward
    compatibility with the 90-day driver/report.
    """
    fields = ["ttrd_xad", "ttrd_yad", "ttrd_zad", "ttrd_ldf", "ttrd_zdf", "ttrd_zdfp",
              "ttrd_qsr", "ttrd_nsr", "ttrd_dmp", "ttrd_evd", "ttrd_bbc", "ttrd_npc",
              "ttrd_atf", "ttrd_tot"]
    raw = _rebuild_kt(kt, fields, rundir=rundir)
    adv_h = raw["ttrd_xad"] + raw["ttrd_yad"]
    adv_v = raw["ttrd_zad"]
    adv = adv_h + adv_v
    iso_redi = raw["ttrd_ldf"] + (raw["ttrd_zdf"] - raw["ttrd_zdfp"])
    vertmix = raw["ttrd_zdfp"]
    forcing = raw["ttrd_qsr"] + raw["ttrd_nsr"]
    evd = raw["ttrd_evd"]
    tot = raw["ttrd_tot"]
    # jptra_npc (trd_oce.F90:49) -- non-penetrative convection treatment, a
    # REAL additive RHS term distinct from EVD (enhanced vertical diffusion).
    # jptra_atf (trd_oce.F90:53) -- Asselin time-filter trend (leapfrog
    # integrator's own filtering term, additive at every step). Both are
    # REAL additive RHS terms; dropping either from the closure check leaves
    # a residual attributable to a missing bucket, not to instrument error
    # (oracle-fidelity Rule 5: prove closure before trusting any bucket).
    return {"adv": adv, "adv_h": adv_h, "adv_v": adv_v, "iso_redi": iso_redi,
            "vertmix": vertmix, "forcing": forcing, "evd": evd, "tot": tot,
            "dmp": raw["ttrd_dmp"], "bbc": raw["ttrd_bbc"], "npc": raw["ttrd_npc"],
            "atf": raw["ttrd_atf"]}


def run_nemo(out_path: str, rundir: str = RUN_TWIN_Y20_DIR, kts=NEMO_DUMP_KTS):
    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask

    g = read_nemo_mesh_mask(f"{RUN_TRAJ}/mesh_mask.nc", nn_hls=0)
    tmask = g.tmask > 0.5
    gdept = g.gdept_1d
    area = g.e1t * g.e2t
    e3t = g.e3t_1d
    print(f"REALIZED DTYPES: gdept_1d={gdept.dtype} e1t={g.e1t.dtype} e3t_1d={e3t.dtype} "
          "(NEMO netCDF read as NumPy float64 directly)")

    band_lev_masks = []
    for lo, hi in DEPTH_BANDS_M:
        m = gdept >= lo
        if hi is not None:
            m = m & (gdept < hi)
        band_lev_masks.append(m)

    print(f"NEMO SIDE -- {rundir}, kts={kts} (10-day cadence, "
          f"kt0={KT0}=20yr*11520steps/yr, rn_Dt=2700s/32 steps/day)")
    print("Each kt is a PER-STEP snapshot (the trend computed during the ONE step "
          "that closed at that kt) -- NOT a time integral. Grouping (verified "
          "file:line citations in deep_box_heat_budget.py, reused verbatim): "
          "ADV=xad+yad+zad; ISO_REDI=ldf+(zdf-zdfp); VERTMIX=zdfp; "
          "FORCING=qsr+nsr; evd/tot/dmp/bbc reported separately (diagnostic-only "
          "or zero for this recipe, NOT folded into the additive total below).")

    save_kwargs = {"kts": np.array(kts)}
    for kt in kts:
        terms_raw = _nemo_grouped_terms(kt, rundir=rundir)
        terms = {k: np.moveaxis(v, 0, -1) for k, v in terms_raw.items()}
        for k, v in terms.items():
            if v.shape != tmask.shape:
                raise SystemExit(f"kt={kt} SHAPE MISMATCH {k}: {v.shape} vs tmask {tmask.shape}")

        print(f"\n--- kt={kt} (day {(kt-KT0)*DT/86400:.1f} into the y20 twin window) ---")
        for box_name, rows in ROW_BOXES.items():
            box_mask = np.zeros_like(tmask)
            box_mask[rows] = tmask[rows]
            box_area = float(area[rows, :].sum())
            print(f"  box {box_name} rows={rows} area={box_area:.4e} m^2:")
            for bi, (band, lev_mask) in enumerate(zip(DEPTH_BANDS_M, band_lev_masks)):
                m = box_mask & lev_mask[None, None, :]
                ncells = int(m.sum())
                lo, hi = band
                label = f"{lo:.0f}-{hi:.0f}m" if hi is not None else f"{lo:.0f}m+"
                if ncells == 0:
                    print(f"    {label:12s} NO WET CELLS in this bucket")
                    continue
                vol = (area[:, :, None] * e3t[None, None, :])[m]
                row = {}
                for term_name in ("adv", "adv_h", "adv_v", "iso_redi", "vertmix", "forcing",
                                  "evd", "dmp", "bbc", "npc", "atf"):
                    dTdt = terms[term_name][m]
                    w_m2 = RHO0_NEMO * CP_NEMO * float(np.sum(dTdt * vol)) / box_area
                    row[term_name] = w_m2
                    save_kwargs[f"nemo_kt{kt}_{box_name}_band{bi}_{term_name}_Wm2"] = w_m2
                tot_dTdt = terms["tot"][m]
                tot_w_m2 = RHO0_NEMO * CP_NEMO * float(np.sum(tot_dTdt * vol)) / box_area
                save_kwargs[f"nemo_kt{kt}_{box_name}_band{bi}_tot_Wm2"] = tot_w_m2
                save_kwargs[f"nemo_kt{kt}_{box_name}_band{bi}_ncells"] = ncells
                additive_sum = row["adv"] + row["iso_redi"] + row["vertmix"] + row["forcing"]
                closure_residual = (tot_w_m2 - additive_sum - row["evd"] - row["dmp"]
                                     - row["bbc"] - row["npc"] - row["atf"])
                save_kwargs[f"nemo_kt{kt}_{box_name}_band{bi}_closure_residual_Wm2"] = closure_residual
                print(f"    {label:12s} ncells={ncells:5d}  "
                      + "  ".join(f"{k}={v:+8.4f}" for k, v in row.items())
                      + f"  ttrd_tot={tot_w_m2:+8.4f}  closure_resid={closure_residual:+8.4f}")

    np.savez(out_path, **save_kwargs)
    print(f"\nSAVED {out_path}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    p_lego = sub.add_parser("lego")
    p_lego.add_argument("out")
    p_lego.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    p_lego.add_argument("--days", type=int, default=90)
    p_lego.add_argument("--surface-tendency-placement", default="applied_now",
                         choices=("applied_now", "leapfrog_rhs"),
                         help="#1492 STEP 3: 'leapfrog_rhs' folds the DINO "
                              "surface tendency into the leap-frog Nnn RHS "
                              "(NEMO tra_sbc placement) instead of the legacy "
                              "pre-step state mutation")

    p_nemo = sub.add_parser("nemo")
    p_nemo.add_argument("out")
    p_nemo.add_argument("--rundir", default=RUN_TWIN_Y20_DIR,
                         help="NEMO trend-restart run dir (default: the 90d twin dir)")
    p_nemo.add_argument("--days", type=int, default=90,
                         help="window length; derives the 10-day-cadence kt list "
                              "kt0+320, kt0+640, ..., kt0+days*32 (default 90 -> the "
                              "original 9-dump NEMO_DUMP_KTS tuple, unchanged)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.mode == "lego":
        run_lego(args.recipe, args.out, args.days,
                 surface_tendency_placement=args.surface_tendency_placement)
    elif args.mode == "nemo":
        n_dumps = (args.days * STEPS_PER_DAY) // 320
        if n_dumps == 0:
            raise SystemExit(
                f"--days {args.days} < 10: no 320-step trend dump fits; "
                "refusing to write a near-empty npz")
        kts = tuple(KT0 + 320 * i for i in range(1, n_dumps + 1))
        if args.days == 90:
            assert kts == NEMO_DUMP_KTS, "90d default must stay byte-identical"
        run_nemo(args.out, rundir=args.rundir, kts=kts)


if __name__ == "__main__":
    main()

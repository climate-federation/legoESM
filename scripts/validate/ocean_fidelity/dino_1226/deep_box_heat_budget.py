"""#1226 DEEP-BAND per-term box heat budget: which tracer term destroys
legoESM's deep (<1000 m) southern-channel stratification?

Established (do NOT re-measure, see task brief): the ACC deficit is a DENSITY
problem, 87% of it below 1400 m, ~100% T, localized to the SOUTHERN edge of
the channel; vertical mixing/EVD/TKE are EXONERATED. Live candidates: (a)
GM/Redi (bolus + isoneutral diffusion flattening the slope that feeds thermal
wind), (b) numerical/spurious diapycnal diffusion from the FCT tracer
advection limiter, (c) vertical advection (w after the bolus fold).

This script does NOT re-derive any numerics. It re-targets the EXISTING,
already-validated online accumulator
(``legoesm.ocean.fidelity.box_heat_budget.BoxHeatBudgetAccumulator``, closes
to ~1e-14) at finer DEEP depth buckets and the SEDGE/CORE row split from
``abyssal_densification.py`` (CORE=slice(1,11), SEDGE=slice(12,17) --
canonical, not re-derived), and reuses the existing multi-rank restart
stitcher (``rebuild_nemo_restart.rebuild``) to read NEMO's OWN ``ttrd_*``
per-term trend dumps from the matched-state NEMO continuation
(``RUN_90D_TWIN``, kt 5760->8640, the SAME restart ``kamm_twin_90d.py``
bridges from).

Two subcommands, kept in SEPARATE tables (never mixed):

  lego   -- legoESM matched-state twin (kamm_twin_90d._build_twin_state,
            restart_file=DINO_00005760_restart.nc, the day-180 developed
            state), 90 days, BoxHeatBudgetAccumulator with depth buckets
            (0,200),(200,1000),(1000,1400),(1400,2000),(2000,3000),(3000,None)
            and row boxes CORE/SEDGE/FULL(=12:47, the old convention, as a
            sanity cross-check against prior 200-1000m results). Also runs a
            --no-use-gm-redi ablation twin (same everything else) so the GM
            BOLUS contribution's MAGNITUDE is visible by DIFFERENCING
            against the baseline. VERIFIED by reading dino.py:2633-2713
            (the isoneutral branch nemo_dino_kamm_mlf uses,
            lateral_tracer_mixing="isoneutral"): ``use_gm_redi`` gates ONLY
            ``kappa_GM`` (line 2636: ``visbeck_kappa_min if use_gm_redi else
            0.0``) and the Visbeck/Treguier adaptive-kappa diagnostics
            (lines 2670/2677); ``kappa_Redi=float(K_h_base)`` (line 2637) is
            set UNCONDITIONALLY, outside any use_gm_redi gate -- so this
            ablation cleanly isolates the GM bolus+adaptive-kappa piece,
            leaving Redi's static isoneutral diffusion untouched and active
            in BOTH runs. This is the "run it both ways" the task brief
            asks for: baseline = bolus-in-ADV (through_fct, the recipe
            default) with the GM/Redi ISO_REDI bucket already separated out
            by the accumulator; the ablation delta on adv_h/adv_v/iso_redi
            shows what disappears when the BOLUS specifically (not Redi)
            is switched off.

  nemo   -- NEMO's own ttrd_* trend at the SAME 10-day-cadence restarts
            (RUN_90D_TWIN/DINO_0000{6080,6400,...,8640}_restart_*.nc,
            16-rank tiles, rebuilt via rebuild_nemo_restart.rebuild -- no
            re-derived stitching). Grouping (traldf.F90/trazdf.F90/
            trdtra.F90, verified line numbers in the module docstring
            below): ISO_REDI = ldf + (zdf - zdfp); VERTMIX = zdfp; evd
            DROPPED (diagnostic snapshot of avt_evd, not an additive RHS
            term -- its physical effect is already inside zdfp's avt).
            ADV = xad+yad+zad (equivalently totad) -- already GM-bolus-
            inclusive (ldf_eiv_trp folds the bolus into the advecting flux
            BEFORE xad/yad/zad are diagnosed, traadv.F90:210/344), matching
            legoESM's through_fct convention exactly. Each 10-day dump is a
            PER-STEP snapshot (the trend computed during the single step
            that closed at that kt), not a time integral -- reported as
            such, one number per available kt, not accumulated.

NEMO trend-index/grouping citations (read, not guessed, per repo policy):
  trd_oce.F90:37-54       jptra_xad=1,yad=2,zad=3,totad=5,ldf=6,zdf=7,
                          zdfp=8,evd=9,qsr=14,nsr=15,tot=17
  traldf.F90:112-113      lateral (Redi/GM-minus-bolus) trend tagged jptra_ldf
  trazdf.F90:105-106      total vertical-diffusion trend (incl. isoneutral
                          vertical-slope piece when ln_traldf_iso=T) tagged
                          jptra_zdf
  traatf_qco.F90:102-129  jptra_zdfp recomputed as a "PURE" avt*dT/dz flux
                          divergence "just before the swap" -- the SAME avt
                          zdf used, i.e. genuinely just the Kz vertical
                          diffusion with the isoneutral piece subtracted out
                          by construction (comment: "iso-neutral diffusion
                          case otherwise jptra_zdf is PURE")
  traadv.F90:210,344,394  ldf_eiv_trp (GM bolus) called INSIDE the advection
                          routine, folding the bolus into (pFu,pFv,pFw)
                          BEFORE jptra_xad/yad/zad/totad are diagnosed

Precision/mode gates (mandatory, #1226 rules): fp64 (require_fp64),
LEGOESM_NEMO_E3T explicit (require_explicit_e3t_mode) -- printed, not
silently defaulted.

*** K33 GROUPING -- READ BEFORE COMPARING ANY SINGLE TERM (fixed 2026-07-30) ***
HISTORY (the defect this note used to describe): ``box_heat_budget.py`` DROPPED
the K33 vertical-isoneutral diagonal from BOTH of its buckets --
``compute_box_vertmix_dT`` called ``_apply_implicit_vertical_mixing`` WITHOUT
``K33_iso`` (defaulting to ``None``, so the fold at
``ocean_model_latlon_cgrid.py:5476-5481`` was skipped) while ``iso_redi`` also
excluded it because this recipe sets ``implicit_K33=True`` (``dino.py``) and
``gm_redi_latlon_cgrid.py`` adds the ``kappa_Redi*S^2*dq_dz`` diagonal to
``F_z`` ONLY ``if not implicit_K33``. Production was always correct
(``K33_iso=k33_implicit`` at ``ocean_model_latlon_cgrid.py:4350,4363,6591,
6617,7056,7279``); it was an INSTRUMENT-ONLY gap. That gap produced a FALSE
"3.4x Redi deficit" and a FALSE "vertical mixing 8x too weak at 200-1000 m"
(both retracted; same artifact class as the ``avt_k`` closure-only 300-370x).

FIXED: ``box_heat_budget.py`` now measures K33 as its OWN ``"k33"`` bucket
(``TERM_NAMES`` has 6 entries), computed with the same
``compute_isoneutral_K33_latlon`` production/``tendency_probe.py`` use.

CURRENT NEMO-COMPARABLE GROUPING -- use these, they are exact:
  NEMO ISO_REDI (``ldf + (zdf - zdfp)``)   ==  ``iso_redi + k33``
  NEMO ``zdfp``  (pure vertical diffusion) ==  ``vertmix``
  full total                               ==  ``adv_h + adv_v + iso_redi
                                                + k33 + forcing + vertmix``
DO NOT use the OLD advice of summing ``iso_redi + vertmix``: post-fix that
equals ``ldf + zdfp``, NOT ``ldf + zdf`` -- it silently EXCLUDES K33, which is
exactly the error the fix removed. ``iso_redi`` ALONE is still not comparable
to NEMO ISO_REDI (it is K33-less by construction); add ``k33``.

Usage
-----
    JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
        CUDA_VISIBLE_DEVICES=<gpu> python deep_box_heat_budget.py lego <out.npz> [--days 90] [--skip-ablation]
    JAX_ENABLE_X64=1 python deep_box_heat_budget.py nemo <out.npz>
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

_THIS_DIR = __file__.rsplit("/", 1)[0]
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _THIS_DIR.rsplit("/", 1)[0])  # scripts/validate/ocean_fidelity/ -- rebuild_nemo_restart.py
from kamm_twin_90d import _build_twin_state, RUN_TRAJ, RUN_STEPDUMP, DT, STEPS_PER_DAY  # noqa: E402

from legoesm.ocean.experiments.dino import apply_dino_lat_lon_surface_forcing  # noqa: E402
from legoesm.ocean.fidelity.box_heat_budget import (  # noqa: E402
    TERM_NAMES,
    BoxHeatBudgetAccumulator,
)
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_fp64,
    require_explicit_e3t_mode,
)

# --- shared conventions (verbatim from abyssal_densification.py, NOT re-derived) ---
SEDGE = slice(12, 17)      # southern edge -- the localized warm-bias band
CORE = slice(1, 11)        # southern core channel
FULL_CH = slice(12, 47)    # #1226/#1317 "channel rows" convention (cross-check)
ROW_BOXES = {"CORE": CORE, "SEDGE": SEDGE, "FULL_CH": FULL_CH}

# Deep-focused bands: keep the shallow 0-200/200-1000 as a CONTROL (already
# known: mixing terms match there), then split the deep water where 87% of
# the density deficit lives (task brief) into four buckets.
DEPTH_BANDS_M = (
    (0.0, 200.0), (200.0, 1000.0),
    (1000.0, 1400.0), (1400.0, 2000.0), (2000.0, 3000.0), (3000.0, None),
)

RUN_90D_TWIN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_90D_TWIN"
NEMO_DUMP_KTS = (6080, 6400, 6720, 7040, 7360, 7680, 8000, 8320, 8640)  # 10-day cadence, kt5760 start
NEMO_RESTART_FILE = "DINO_00005760_restart.nc"  # day-180 developed state (the state BOTH protocols share)

RHO0_NEMO, CP_NEMO = 1026.0, 3991.86795711963  # eosbn2.F90:1899, task brief


# =====================================================================
# lego (matched-state, primary)
# =====================================================================

def _run_lego_accumulator(recipe: str, n_days: int, *, use_gm_redi: bool | None,
                           restart_file: str = NEMO_RESTART_FILE):
    """One matched-state twin through the box-heat accumulator. Returns summary."""
    import jax
    import jax.numpy as jnp

    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        recipe, RUN_TRAJ, RUN_STEPDUMP, bridge_tke=True, bridge_before=True,
        use_gm_redi=use_gm_redi, restart_file=restart_file,
    )
    require_fp64(br.geometry, br.z_coord, st, context="deep_box_heat_budget lego")

    accs = {name: BoxHeatBudgetAccumulator(
                br.geometry, br.z_coord, mc, cfg, forcing, DT, model,
                row_slice=rows, depth_bands_m=DEPTH_BANDS_M)
            for name, rows in ROW_BOXES.items()}

    nsteps = STEPS_PER_DAY * n_days
    dyn = jax.jit(lambda st, t: model.step(st, DT, surface_forcing=sf, t_seconds=t))
    sample_dt = STEPS_PER_DAY * DT

    t_seconds = 0.0
    for acc in accs.values():
        acc.sample(st, dt_step=sample_dt, t_seconds=t_seconds)
    for k in range(nsteps):
        st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, DT,
                                                  t_seconds=(k + 1) * DT)
        t_seconds = (k + 1) * DT
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
    return {name: acc.summary(total_seconds) for name, acc in accs.items()}, total_seconds


def _print_lego_summary(tag: str, summaries: dict, total_seconds: float, conv_ratio: float):
    print(f"\n{'='*100}\nLEGO MATCHED-STATE ({tag}), {total_seconds/86400:.1f}d, "
          f"restart={NEMO_RESTART_FILE} (day-180, kt=5760)\n{'='*100}")
    for box_name in ROW_BOXES:
        summary = summaries[box_name]
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


def run_lego(recipe: str, out_path: str, n_days: int, skip_ablation: bool):
    e3t_mode = require_explicit_e3t_mode(context="deep_box_heat_budget lego")
    print(f"LEGOESM_NEMO_E3T={e3t_mode}")

    print("\n### BASELINE (recipe default use_gm_redi, bolus-in-ADV through_fct) ###")
    base_summaries, total_seconds = _run_lego_accumulator(recipe, n_days, use_gm_redi=None)
    # rho0*cp is a pure multiplicative convention on every accumulated Joule
    # -- rescale rather than re-run (box_budget_run.py's documented pattern).
    # BoxHeatBudgetAccumulator defaults to legoesm.constants.rho_ocean/c_sw.
    from legoesm import constants
    conv_ratio = (RHO0_NEMO * CP_NEMO) / (constants.rho_ocean * constants.c_sw)
    _print_lego_summary("baseline", base_summaries, total_seconds, conv_ratio)

    save_kwargs = {"total_seconds": total_seconds, "conv_ratio": conv_ratio,
                    "restart_file": NEMO_RESTART_FILE, "e3t_mode": e3t_mode}
    for box_name, summary in base_summaries.items():
        for bi, band in enumerate(DEPTH_BANDS_M):
            for term in TERM_NAMES:
                save_kwargs[f"base_{box_name}_band{bi}_{term}_Wm2"] = summary["terms"][band][term]["W_per_m2"]
            save_kwargs[f"base_{box_name}_band{bi}_residual_Wm2"] = summary["terms"][band]["residual"]["W_per_m2"]

    if not skip_ablation:
        print("\n### ABLATION (--no-use-gm-redi: kappa_GM (bolus) + adaptive "
              "Visbeck/Treguier kappa -> off; kappa_Redi UNCHANGED -- "
              "verified dino.py:2633-2713, kappa_Redi=float(K_h_base) is set "
              "unconditionally outside the use_gm_redi gate on the "
              "isoneutral branch nemo_dino_kamm_mlf uses -- so this cleanly "
              "isolates the GM BOLUS contribution from Redi) ###")
        abl_summaries, total_seconds_abl = _run_lego_accumulator(recipe, n_days, use_gm_redi=False)
        _print_lego_summary("GM/Redi-OFF ablation", abl_summaries, total_seconds_abl, conv_ratio)
        assert abs(total_seconds_abl - total_seconds) < 1.0, "ablation window must match baseline window"

        print(f"\n{'='*100}\nGM/Redi CONTRIBUTION (baseline - ablation), same window\n{'='*100}")
        for box_name in ROW_BOXES:
            print(f"\n--- box {box_name} ---")
            for bi, band in enumerate(DEPTH_BANDS_M):
                lo, hi = band
                label = f"{lo:.0f}-{hi:.0f}m" if hi is not None else f"{lo:.0f}m+"
                b_base = base_summaries[box_name]["terms"][band]
                b_abl = abl_summaries[box_name]["terms"][band]
                for term in ("adv_h", "adv_v", "iso_redi"):
                    d = b_base[term]["W_per_m2"] - b_abl[term]["W_per_m2"]
                    print(f"  {label:12s} {term:10s} delta={d:+9.4f} W/m^2")
                for term in TERM_NAMES:
                    save_kwargs[f"abl_{box_name}_band{bi}_{term}_Wm2"] = abl_summaries[box_name]["terms"][band][term]["W_per_m2"]
                save_kwargs[f"abl_{box_name}_band{bi}_residual_Wm2"] = abl_summaries[box_name]["terms"][band]["residual"]["W_per_m2"]

    np.savez(out_path, **save_kwargs)
    print(f"\nSAVED {out_path}")


# =====================================================================
# nemo (matched-state, ground truth, NOT re-run -- reads RUN_90D_TWIN)
# =====================================================================

def _rebuild_kt(kt: int, fields: list[str]) -> dict:
    from rebuild_nemo_restart import rebuild
    pattern = f"{RUN_90D_TWIN}/DINO_{kt:08d}_restart_*.nc"
    if not glob.glob(pattern):
        raise SystemExit(f"NEMO trend restart not found: {pattern}")
    return rebuild(pattern, fields)


def _nemo_grouped_terms(kt: int) -> dict[str, np.ndarray]:
    """One kt's grouped (lev, y, x) dT/dt [K/s] terms, per the traldf/trazdf/
    trdtra bookkeeping in the module docstring. Returns raw NEMO-order
    (lev, y, x) arrays -- caller moves axes to (y, x, lev)."""
    fields = ["ttrd_xad", "ttrd_yad", "ttrd_zad", "ttrd_ldf", "ttrd_zdf", "ttrd_zdfp"]
    raw = _rebuild_kt(kt, fields)
    adv = raw["ttrd_xad"] + raw["ttrd_yad"] + raw["ttrd_zad"]
    iso_redi = raw["ttrd_ldf"] + (raw["ttrd_zdf"] - raw["ttrd_zdfp"])
    vertmix = raw["ttrd_zdfp"]
    return {"adv": adv, "iso_redi": iso_redi, "vertmix": vertmix}


def run_nemo(out_path: str):
    # No legoESM state/policy is touched by this mode (pure NumPy read of
    # NEMO's own restart-dumped ttrd_* doubles + mesh_mask) -- require_fp64
    # gates the legoESM PRECISION POLICY, which is orthogonal here; NEMO's
    # own netCDF doubles are read via np.float64 directly (checked below).
    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask

    g = read_nemo_mesh_mask(f"{RUN_TRAJ}/mesh_mask.nc", nn_hls=0)
    tmask = g.tmask > 0.5              # (n_lat, n_lon, nlev)
    gdept = g.gdept_1d                 # (nlev,) reference depth [m], positive down
    area = g.e1t * g.e2t               # (n_lat, n_lon)
    e3t = g.e3t_1d                     # (nlev,)
    print(f"REALIZED DTYPES: gdept_1d={gdept.dtype} e1t={g.e1t.dtype} "
          f"e3t_1d={e3t.dtype} (NEMO netCDF read as NumPy float64 directly, "
          "no legoESM precision policy involved in this mode)")

    band_lev_masks = []
    for lo, hi in DEPTH_BANDS_M:
        m = gdept >= lo
        if hi is not None:
            m = m & (gdept < hi)
        band_lev_masks.append(m)

    print(f"NEMO SIDE -- {RUN_90D_TWIN}, kts={NEMO_DUMP_KTS} "
          f"(10-day cadence, restart={NEMO_RESTART_FILE} common origin, kt=5760)")
    print("Each kt is a PER-STEP snapshot (the trend computed during the ONE "
          "step that closed at that kt) -- NOT a time integral. Grouping: "
          "ADV=xad+yad+zad (GM-bolus-inclusive, traadv.F90:210); "
          "ISO_REDI=ldf+(zdf-zdfp) (traldf.F90:112, trazdf.F90:105, "
          "traatf_qco.F90:102); VERTMIX=zdfp (\"PURE\" Kz, traatf_qco.F90:111); "
          "evd DROPPED (diagnostic snapshot, not additive RHS).")

    save_kwargs = {"kts": np.array(NEMO_DUMP_KTS)}
    for kt in NEMO_DUMP_KTS:
        terms_raw = _nemo_grouped_terms(kt)  # (lev, y, x)
        # NEMO-order (lev,y,x) -> legoESM order (y,x,lev), matching tmask/area/gdept.
        terms = {k: np.moveaxis(v, 0, -1) for k, v in terms_raw.items()}
        for k, v in terms.items():
            if v.shape != tmask.shape:
                raise SystemExit(f"kt={kt} SHAPE MISMATCH {k}: {v.shape} vs tmask {tmask.shape}")

        print(f"\n--- kt={kt} (day {(kt-5760)*DT/86400:.1f} into the matched twin window) ---")
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
                for term_name, arr in terms.items():
                    dTdt = arr[m]  # [K/s]
                    w_m2 = RHO0_NEMO * CP_NEMO * float(np.sum(dTdt * vol)) / box_area
                    row[term_name] = w_m2
                    save_kwargs[f"nemo_kt{kt}_{box_name}_band{bi}_{term_name}_Wm2"] = w_m2
                total = sum(row.values())
                save_kwargs[f"nemo_kt{kt}_{box_name}_band{bi}_ncells"] = ncells
                print(f"    {label:12s} ncells={ncells:5d}  "
                      + "  ".join(f"{k}={v:+8.4f}" for k, v in row.items())
                      + f"  sum={total:+8.4f} W/m^2")

    np.savez(out_path, **save_kwargs)
    print(f"\nSAVED {out_path}")


# =====================================================================
# self-checks (ponytail: smallest thing that fails if the grouping/rebuild
# breaks -- not a pytest suite, this is a scratch #1226 probe script)
# =====================================================================

def _selfcheck_rebuild_shape_and_dtype():
    """Manual-vs-vectorized recomputation of one bucket mean + dtype check
    (task brief: "manual-vs-vectorized recomputation of at least one bucket
    mean", "print realized dtypes")."""
    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask
    g = read_nemo_mesh_mask(f"{RUN_TRAJ}/mesh_mask.nc", nn_hls=0)
    assert g.tmask.dtype == np.float64 or g.gdept_1d.dtype == np.float64, "mesh_mask must read as float64"
    terms_raw = _nemo_grouped_terms(NEMO_DUMP_KTS[0])
    terms = {k: np.moveaxis(v, 0, -1) for k, v in terms_raw.items()}
    tmask = g.tmask > 0.5
    area = g.e1t * g.e2t
    e3t = g.e3t_1d
    band = DEPTH_BANDS_M[2]  # 1000-1400m, a deep bucket
    lev_mask = (g.gdept_1d >= band[0]) & (g.gdept_1d < band[1])
    box_mask = np.zeros_like(tmask)
    box_mask[SEDGE] = tmask[SEDGE]
    m = box_mask & lev_mask[None, None, :]
    vol = (area[:, :, None] * e3t[None, None, :])[m]
    vec_mean = RHO0_NEMO * CP_NEMO * float(np.sum(terms["vertmix"][m] * vol)) / float(area[SEDGE, :].sum())

    # manual double loop over the same mask
    manual_sum = 0.0
    idx = np.argwhere(m)
    vol_arr = area[:, :, None] * e3t[None, None, :]
    for (yi, xi, ki) in idx:
        manual_sum += RHO0_NEMO * CP_NEMO * terms["vertmix"][yi, xi, ki] * vol_arr[yi, xi, ki]
    manual_mean = manual_sum / float(area[SEDGE, :].sum())
    assert abs(manual_mean - vec_mean) < 1e-9 * max(abs(vec_mean), 1.0), (
        f"manual vs vectorized mismatch: {manual_mean} vs {vec_mean}")
    print(f"SELFCHECK OK: manual={manual_mean:.6e} vectorized={vec_mean:.6e} "
          f"(SEDGE, 1000-1400m, kt={NEMO_DUMP_KTS[0]})")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    p_lego = sub.add_parser("lego")
    p_lego.add_argument("out")
    p_lego.add_argument("--recipe", default="nemo_dino_kamm_mlf")
    p_lego.add_argument("--days", type=int, default=90)
    p_lego.add_argument("--skip-ablation", action="store_true")

    p_nemo = sub.add_parser("nemo")
    p_nemo.add_argument("out")

    p_check = sub.add_parser("selfcheck")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.mode == "lego":
        run_lego(args.recipe, args.out, args.days, args.skip_ablation)
    elif args.mode == "nemo":
        _selfcheck_rebuild_shape_and_dtype()
        run_nemo(args.out)
    elif args.mode == "selfcheck":
        _selfcheck_rebuild_shape_and_dtype()


if __name__ == "__main__":
    main()

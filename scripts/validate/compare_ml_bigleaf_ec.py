"""Compare the CLM-ML multilayer canopy against the legoESM two-leaf big-leaf
canopy — fluxes (GPP, SH, LH) and simulated SIF — at an eddy-covariance tower
site, driven by identical atmospheric forcing.

Motivation
----------
legoESM ships two vegetated surface schemes that both emit SIF: the **two-leaf
big-leaf** canopy (``surface_scheme/two_leaf_canopy.py``, SIF from the BEPS-SIF
An/Ci je-inversion) and the **CLM-ML multilayer** canopy
(``canopy/clm_ml_interface.py``, SIF from the model's native ``je_leaf``).  This
driver cross-checks them at a real EC site: it runs the CLM-ML offline tower
simulation, and at every driver step reads CLM-ML's atmospheric forcing +
canopy-integrated fluxes + SIF, then drives the two-leaf canopy with the SAME
forcing (matched LAI, prescribed soil skin temperature) and records both.

Findings at CHATS7 (irrigated walnut orchard, oasis/advection regime), driving
the two-leaf with CLM-ML's OWN structure + physiology (see :func:`run_bigleaf`):
- **SIF**: two distinct results, kept separate.  (i) je CONVENTION (internal to
  the multilayer path, LAI-independent): the multilayer consumes CLM-ML's native
  ``je_leaf`` (full Farquhar J) directly and is CORRECT as-is.  At high light the
  fluorescence yield sits on its light-saturation ``x=0`` clamp where yield is
  je-INDEPENDENT, so the je scale drops out; rescaling to the BEPS proxy
  convention (J/4) lifts ``x`` off the clamp near sub-saturated SUNRISE/SUNSET and
  OVERSHOOTS the native SIF by ~15 % on the diurnal mean — so native ``je_leaf``
  is the correct feed (see ``canopy/sif.py::multilayer_canopy_sif``).  (ii)
  CROSS-SCHEME magnitude: the two-leaf SIF runs ~9 % BELOW the multilayer
  (BL/ML~0.91), tracking the LH/radiation under-bias — the same single-green-LAI
  absorption limitation as LH, NOT a je-convention effect.  (An earlier ~1 %
  cross-scheme match was an artifact of driving the two-leaf with plant-area
  index; green LAI is physiology-correct and exposes the structural ~9 %.)
- **GPP** agrees to ~5 % over the diurnal cycle (driving the two-leaf with green
  LAI; driving it with plant-area index instead over-counts photosynthetic area
  and biases GPP ~20 % high).  The residual is the stomatal MODEL: two-leaf
  Ball-Berry vs CLM-ML WUE-optimization cannot be reconciled by a single slope —
  matching GPP and transpiration pull it in opposite directions, not a bias.
- **LH** and **net canopy radiation** run ~10-15 % below CLM-ML: the two-leaf's
  single green LAI transpires/absorbs over leaf area only, whereas CLM-ML's
  radiative transfer also intercepts over stems (plant area) — a two-leaf
  single-LAI structural limitation, not a parameter bias.
- **Sensible heat**: the VEGETATION (leaf) SH agrees; the TOTAL SH is dominated
  by a soil-sensible artifact of the offline harness — the soil skin pinned at
  CLM-ML's warm ground temperature is hotter than the two-leaf's evaporatively
  cooled canopy air, so it emits spurious soil SH.  Compare the vegetation-only
  fluxes (``*_veg``) for the apples-to-apples canopy comparison.

Requirements (NOT run in CI)
----------------------------
- the optional ``clm-ml-jax`` package (https://github.com/AyaLahlou/clm-ml-jax),
  installed and importable (flat modules ``multilayer_canopy``, ``clm_src_*``);
- a tower-forcing namelist + NetCDF for the site (ships with clm-ml-jax, e.g.
  ``offline_executable/nl.CHATS7.50steps``).

The CI-testable half — building the two-leaf forcing and running the big-leaf
canopy from a captured-forcing dict — lives in :func:`run_bigleaf` /
:func:`build_bigleaf_forcing` and is exercised by
``tests/land/integration/test_ml_bigleaf_ec_compare.py`` without the heavy model.

Usage
-----
    python scripts/validate/compare_ml_bigleaf_ec.py \
        --namelist /path/to/clm-ml-jax/src/offline_executable/nl.CHATS7.50steps \
        --out results/ml_bigleaf_ec/chats7.json [--plot]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.clm_ml_interface import _extract_clm_ml_sif
from legoesm.land.canopy.sif import SIFConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.surface_scheme.two_leaf_canopy import compute_two_leaf_canopy_fluxes

from legoesm import constants

# CLM-ML unfilled elements carry spval = 1e36.
_SPVAL_GUARD = 1.0e30
# gC per umol CO2 (GPP unit bridge: CLM-ML umol/m2/s -> legoESM gC/m2/s).
# constants.M_C is g(C)/mol; * 1e-6 converts per-mol to per-umol (exact).
_UMOL_CO2_TO_GC = constants.M_C * 1.0e-6


def _san(x: float, default: float) -> float:
    x = float(x)
    return x if (np.isfinite(x) and abs(x) < _SPVAL_GUARD) else default


def capture_clm_ml(mlcanopy, i: int = 0) -> dict:
    """Read CLM-ML atmospheric forcing + canopy structure + fluxes + SIF.

    Patch axis is 1-based (the offline driver / interface pad index 0); layer
    axis is 1-based too.  Returns a plain-float dict (JSON-serialisable) with
    (a) the forcing the two-leaf canopy needs, (b) the canopy STRUCTURE +
    PHYSIOLOGY needed to drive the two-leaf with the SAME parameters (so the
    comparison isolates the scheme difference from parameter bias), and (c) the
    multilayer reference fluxes split into VEGETATION (leaf) vs SOIL — the
    vegetation-only fluxes are the apples-to-apples canopy comparison, immune to
    the offline soil lower-boundary choice.
    """
    def p(name):  # per-patch scalar at 1-based index i+1
        return jnp.asarray(getattr(mlcanopy, name))[i + 1]

    t_air = _san(p("tref_forcing"), 288.0)
    q = _san(p("qref_forcing"), 0.008)
    u = max(_san(p("uref_forcing"), 1.0), 0.3)
    p_air = _san(p("pref_forcing"), 101325.0)
    co2 = _san(p("co2ref_forcing"), 400.0)
    lw = _san(p("lwsky_forcing"), 320.0)
    zen = _san(p("solar_zen_forcing"), 1.4)
    swb = jnp.asarray(mlcanopy.swskyb_forcing)[i + 1]
    swd = jnp.asarray(mlcanopy.swskyd_forcing)[i + 1]
    sw = sum(_san(a[b], 0.0) for a in (swb, swd) for b in (1, 2))  # direct+diffuse, vis+nir
    # Canopy structure + physiology.  ``lai`` is the GREEN leaf-area index
    # (``dlai``, the photosynthesising/transpiring area) — the two-leaf uses ONE
    # LAI for radiation AND photosynthesis, so it must be driven with green LAI
    # (physiology-correct) rather than plant-area ``pai`` (``dpai``, incl. stems),
    # which would over-count photosynthetic capacity.  Trade-off: the two-leaf's
    # radiative absorption is then slightly below CLM-ML's PAI-based absorption
    # (stems intercept light in CLM-ML) — a two-leaf single-LAI limitation.
    # ``vcmax_top`` is the max green-layer sunlit Vcmax25 (top-canopy reference for
    # the peaked nitrogen profile the two-leaf rebuilds with kn).
    def _area(name):
        a = jnp.asarray(getattr(mlcanopy, name))[i + 1, 1:]
        return float(jnp.sum(jnp.where(jnp.isfinite(a) & (jnp.abs(a) < _SPVAL_GUARD),
                                       jnp.clip(a, 0.0, None), 0.0)))

    green_lai = _area("dlai_profile")
    pai = _area("dpai_profile")
    vc_sun = jnp.asarray(mlcanopy.vcmax25_leaf)[i + 1, 1:, 1]  # sunlit vcmax25 per layer
    vmask = jnp.isfinite(vc_sun) & (jnp.abs(vc_sun) < _SPVAL_GUARD) & (vc_sun > 0.0)
    # nan (not a silent 0) when no valid sunlit layer, so a data gap can't
    # masquerade as a real zero-Vcmax result; run_bigleaf applies a positive
    # fallback instead.
    vcmax_top = (float(jnp.max(jnp.where(vmask, vc_sun, 0.0)))
                 if bool(jnp.any(vmask)) else float("nan"))

    def _opt(name):  # optional CLM-ML split diagnostic: nan if the field is absent
        return _san(p(name), float("nan")) if hasattr(mlcanopy, name) else float("nan")

    sif = _extract_clm_ml_sif(mlcanopy, ncol=1, sif_cfg=SIFConfig())
    return dict(
        Ta=t_air, q=q, u=u, P=p_air, co2=co2, lw=lw, sw=sw,
        coszen=float(np.clip(np.cos(zen), 0.0, 1.0)), lai=green_lai, pai=pai,
        vcmax_top=vcmax_top,
        ztop=_san(p("ztop_canopy"), 5.0), zref=_san(p("zref_forcing"), 10.0),
        Tg=_san(p("tg_soil"), t_air),
        gpp_ml=_san(p("gppveg_canopy"), 0.0) * _UMOL_CO2_TO_GC,
        sh_ml=_san(p("shflx_canopy"), 0.0),
        lh_ml=_san(p("lhflx_canopy"), 0.0),
        # Vegetation (leaf) vs soil split: shflx = shveg + shsoi (CLM-ML).  These
        # split fields are optional (older clm-ml-jax may lack them) -> nan.
        shveg_ml=_opt("shveg_canopy"), lhveg_ml=_opt("lhveg_canopy"),
        shsoi_ml=_opt("shsoi_soil"), lhsoi_ml=_opt("lhsoi_soil"),
        rnveg_ml=(_opt("rnet_canopy") - _opt("rnsoi_soil")),
        sif_ml=(float(sif[0]) if sif is not None else None),
    )


def build_bigleaf_forcing(c: dict) -> AtmToSurface:
    """Assemble a 1-column :class:`AtmToSurface` from a captured-forcing dict."""
    rho = c["P"] / (constants.R_d * c["Ta"] * (1.0 + 0.61 * c["q"]))
    return AtmToSurface(
        T_lowest=jnp.array([c["Ta"]]), q_lowest=jnp.array([c["q"]]),
        u_lowest=jnp.array([c["u"]]), v_lowest=jnp.zeros(1),
        p_lowest=jnp.array([c["P"]]), p_surface=jnp.array([c["P"]]),
        rho_lowest=jnp.array([rho]),
        sw_down=jnp.array([c["sw"]]), lw_down=jnp.array([c["lw"]]),
        cos_zenith=jnp.array([c["coszen"]]),
        precip_total=jnp.zeros(1), precip_snow=jnp.zeros(1),
        co2_ppmv=jnp.array([c["co2"]]),
        has_radiation=jnp.array(1.0), has_precipitation=jnp.array(0.0),
    )


# --- Two-leaf parameters MATCHED to CLM-ML at CHATS7 (see module docstring) ---
_MATCH_KN = 0.30          # nitrogen extinction (coincides with the scheme default)
_MATCH_CI = 1.0           # clumping index (orchard ~ non-clumped; default is 0.75)
_MATCH_M_C3 = 13.0        # Ball-Berry slope emulating CLM-ML's WUE-opt conductance
_MATCH_W_FRAC_RZ = 1.0    # irrigated -> no root-zone soil-moisture stress
_MATCH_W_FRAC_SOIL_EVAP = 0.02  # CLM-ML soil is a DRY surface (LE_soil ~ Rn_soil)
_VCMAX_FALLBACK = 60.0    # DBF-temperate C3 default when vcmax_top is absent/invalid


def run_bigleaf(c: dict, dt: float = 1800.0) -> dict:
    """Drive the two-leaf big-leaf canopy with the captured CLM-ML forcing.

    The two-leaf is driven with CLM-ML's OWN structure + physiology so the
    comparison isolates the scheme difference from parameter bias: canopy height
    ``ztop``, aero reference height ``zref``, top-canopy ``vcmax_top`` with a
    peaked nitrogen profile (``kn``), GREEN leaf-area index ``lai`` (the two-leaf
    uses one LAI for radiation AND photosynthesis, so green LAI is the
    physiology-correct choice — see :func:`capture_clm_ml`), clumping ``CI``, the
    Ball-Berry slope ``m_C3`` (raised from the default 9 to emulate CLM-ML's
    WUE-optimization stomatal model), no root-zone moisture stress (irrigated),
    and a dry soil-evaporation efficiency matching CLM-ML's dry soil surface.
    Soil skin T is prescribed to CLM-ML's ``tg_soil`` (identity thermal callback)
    so both schemes see the same lower boundary.

    Falls back to scalar defaults for any missing structure key so a synthetic
    forcing dict (the CI test) still runs.  Returns big-leaf GPP [gC/m2/s], total
    + vegetation/soil-split SH, LH [W/m2] and SIF [umol/m2/s].
    """
    import types

    forcing = build_bigleaf_forcing(c)
    t_soil = jnp.array([c["Tg"]])
    green_lai = float(c.get("lai", 1.5))          # GREEN LAI (photosynthesis + transpiration)
    vcmax_top = float(c.get("vcmax_top", _VCMAX_FALLBACK))
    if not np.isfinite(vcmax_top) or vcmax_top <= 0.0:
        vcmax_top = _VCMAX_FALLBACK               # data gap -> default, not a silent zero
    lp = types.SimpleNamespace(
        hc=jnp.array([float(c.get("ztop", 5.0))]),
        Vcmax25_C3_leaf=jnp.array([vcmax_top]),
        kn=jnp.array([_MATCH_KN]), CI=jnp.array([_MATCH_CI]),
        m_C3=jnp.array([_MATCH_M_C3]),
    )
    out = compute_two_leaf_canopy_fluxes(
        T_soil_top=t_soil, forcing=forcing,
        canopy_config=TwoLeafCanopyConfig(max_iters=30, sif=SIFConfig()),
        land_config=MultiLayerLandConfig(z_ref=float(c.get("zref", 10.0))),
        canopy_params=lp,
        w_frac_rz=jnp.array([_MATCH_W_FRAC_RZ]),
        wind_speed=jnp.array([c["u"]]), wind_dir_x=jnp.ones(1), wind_dir_y=jnp.zeros(1),
        soil_thermal_fn=lambda g_flux, dt_: t_soil, dt=dt,
        LAI_override=jnp.array([max(green_lai, 0.1)]),
        w_frac_soil_evap=jnp.array([_MATCH_W_FRAC_SOIL_EVAP]),
    )
    def _v(x):  # optional per-component field -> float or nan
        return float(x[0]) if x is not None else float("nan")
    return dict(
        gpp_bl=(float(out.gpp[0]) if out.gpp is not None else None),
        sh_bl=float(out.shflx[0]), lh_bl=float(out.lhflx[0]),
        sif_bl=(float(out.sif[0]) if out.sif is not None else None),
        shveg_bl=_v(out.H_canopy), lhveg_bl=_v(out.LE_canopy),
        shsoi_bl=_v(out.H_soil), lhsoi_bl=_v(out.LE_soil),
        rnveg_bl=_v(out.Rn_canopy),
    )


# Forcing time step of the CLM-ML tower driver (nl.CHATS7.50steps): 30 min.
_DT_FORCING_H = 0.5


def _local_solar_hours(records: list) -> np.ndarray:
    """Local SOLAR time (h, in [0, 24)) for each record, from the data's own
    solar geometry — NOT wall-clock: the driver stores no absolute timestamp.

    Solar noon is defined by the maximum cosine of the solar zenith (the clean
    astronomical signal; incident ``sw`` is corrupted by cloud) and pinned to
    12.00 h; every other step is offset by its ``step`` delta times ``dt``.  This
    centres the diurnal cycle on true local solar noon regardless of the site's
    UTC offset, so it reads correctly whether the tower's civil zone is Mountain
    (Colorado) or Pacific.

    PRECONDITION: ``step`` is this driver's absolute half-hour forcing index
    (``time_indx``, ``dt = 1800 s``), so a day is 48 steps and the ``% 24`` wrap
    is exact.  Steps need not be contiguous (a night gap between two partial days
    is fine — the offset uses ``step`` deltas), but the mapping is only valid for
    that uniform half-hour index; a different step convention would need its own
    ``dt``.  If NO record carries a finite ``coszen`` (solar geometry absent), it
    falls back to elapsed half-hours from the earliest step so the caller never
    crashes on an all-NaN slice.
    """
    steps = np.array([x["step"] for x in records], dtype=float)
    coszen = np.array([x.get("coszen", np.nan) for x in records], dtype=float)
    if not np.any(np.isfinite(coszen)):
        return ((steps - steps.min()) * _DT_FORCING_H) % 24.0
    noon_step = steps[int(np.nanargmax(coszen))]
    return (12.0 + (steps - noon_step) * _DT_FORCING_H) % 24.0


def _plot(records: list, path: str) -> None:
    """Publication diurnal figure: CLM-ML multilayer vs two-leaf big-leaf, the
    four canopy fluxes (GPP, SIF, vegetation LH, vegetation SH) versus LOCAL
    SOLAR time.  Daytime points only, sorted by local hour so the curve runs
    sunrise -> noon -> sunset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rec = [x for x in records if "error" not in x]
    if not all_rec:
        print("no records to plot")
        return
    # Anchor local solar noon on the FULL record BEFORE dropping night: a cloudy
    # true-noon step can have sw<=50, and filtering it out first would shift the
    # inferred noon to the brightest surviving step.  Then keep daytime only.
    lt_all = _local_solar_hours(all_rec)
    day = np.array([x.get("sw", 0.0) > 50.0 for x in all_rec])
    if not day.any():
        print("no daytime records to plot")
        return
    lt = lt_all[day]
    r = [x for x, d in zip(all_rec, day) if d]            # daytime, in step order
    steps = np.array([x["step"] for x in r], dtype=float)
    # Day segments: a jump >1 in the night-filtered step index marks a night gap,
    # so the diurnal-composite line is drawn PER DAY and never connects across
    # calendar days (sorting by hour alone would join two separate afternoons).
    seg = (np.concatenate([[0], np.cumsum(np.diff(steps) > 1)]).astype(int)
           if len(steps) else np.array([], dtype=int))
    gc = 1.0 / _UMOL_CO2_TO_GC

    def col(k, scale=1.0):
        return np.array([(x.get(k) * scale if x.get(k) is not None else np.nan)
                         for x in r], dtype=float)

    # VEGETATION (leaf) SH/LH — the valid canopy comparison.  TOTAL SH/LH carry
    # the offline soil-BC artifact (pinned warm soil), i.e. exactly what the
    # module docstring tells the reader to ignore; veg fluxes isolate the canopy.
    ml_c, bl_c = "#1b7837", "#c51b7d"
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 11, "figure.dpi": 200})
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.6), constrained_layout=True)
    panels = [
        (ax[0, 0], "a", "GPP", r"$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$",
         col("gpp_ml", gc), col("gpp_bl", gc)),
        (ax[0, 1], "b", "SIF", r"$\mu$mol photon m$^{-2}$ s$^{-1}$",
         col("sif_ml"), col("sif_bl")),
        (ax[1, 0], "c", "Vegetation latent heat", r"W m$^{-2}$",
         col("lhveg_ml"), col("lhveg_bl")),
        (ax[1, 1], "d", "Vegetation sensible heat", r"W m$^{-2}$",
         col("shveg_ml"), col("shveg_bl")),
    ]
    for a, tag, name, unit, ml, bl in panels:
        for s in np.unique(seg):                          # one line per calendar day
            j = np.where(seg == s)[0]
            j = j[np.argsort(lt[j])]                       # order within the day by hour
            a.plot(lt[j], ml[j], "-", color=ml_c, lw=1.7, zorder=1)
            a.plot(lt[j], bl[j], "--", color=bl_c, lw=1.5, zorder=1)
        a.plot(lt, ml, "o", color=ml_c, ms=4, label="multilayer (CLM-ML)", zorder=2)
        a.plot(lt, bl, "s", color=bl_c, ms=4, label="two-leaf (big-leaf)", zorder=2)
        a.set_title(f"({tag}) {name}", loc="left", fontweight="bold")
        a.set_ylabel(unit)
        a.set_xlim(lt.min() - 0.3, lt.max() + 0.3)
        a.set_xticks(np.arange(6, 20, 3))
        a.grid(alpha=0.3)
        # Agreement: mean bias (BL - ML) + RMSE over paired finite daytime points.
        # Bias, not a ratio: SH's daytime mean is near zero (oasis), where a ratio
        # explodes and misleads; bias stays meaningful for every panel.
        m = np.isfinite(ml) & np.isfinite(bl)
        if m.any():
            rmse = float(np.sqrt(np.mean((bl[m] - ml[m]) ** 2)))
            bias = float(np.mean(bl[m] - ml[m]))
            a.text(0.03, 0.94, f"bias={bias:+.2g}\nRMSE={rmse:.2g}",
                   transform=a.transAxes, va="top", ha="left", fontsize=8.5,
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85))
    for a in ax[1, :]:
        a.set_xlabel("Local solar time (h)")
    ax[0, 0].legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    fig.suptitle("CHATS7 walnut orchard: multilayer vs two-leaf canopy fluxes "
                 "(identical forcing, diurnal composite)", fontsize=12.5)
    fig.savefig(path)
    print(f"saved {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--namelist", required=True, help="clm-ml-jax tower namelist path")
    ap.add_argument("--out", default="results/ml_bigleaf_ec/compare.json")
    ap.add_argument("--plot", action="store_true", help="also write a <out>.png diurnal figure")
    args = ap.parse_args(argv)

    jax.config.update("jax_enable_x64", True)
    try:
        import legoesm.land.canopy.clm_ml_backend.clm_src_cpl.lnd_comp_nuopc as lnd_comp
        import offline_executable.main as offline_main
    except ImportError as e:  # pragma: no cover - requires optional clm-ml-jax
        print(f"clm-ml-jax not importable ({e}); this driver needs the optional "
              "model + tower forcing. See the module docstring.", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    records: list = []
    orig = lnd_comp.ModelAdvance

    def wrapped(bounds, time_indx, fin1, fin2):
        orig(bounds, time_indx, fin1, fin2)
        try:
            from legoesm.land.canopy.clm_ml_backend.clm_src_main import clm_instMod
            c = capture_clm_ml(clm_instMod.mlcanopy_inst, 0)
            records.append(dict(step=int(time_indx), **c, **run_bigleaf(c)))
        except Exception as e:  # never let the diagnostic kill the run
            records.append(dict(step=int(time_indx), error=repr(e)[:160]))
        json.dump(records, open(args.out, "w"), indent=2)  # incremental save

    lnd_comp.ModelAdvance = wrapped
    offline_main.ModelAdvance = wrapped
    sys.argv = ["compare_ml_bigleaf_ec", args.namelist]
    try:
        offline_main.main()
    finally:
        json.dump(records, open(args.out, "w"), indent=2)
        good = [x for x in records if "error" not in x and x["sw"] > 50]
        if good:
            def mean(k, s=1.0):  # NaN-safe: one unfilled step must not poison the mean
                v = [x[k] * s for x in good
                     if x.get(k) is not None and np.isfinite(x[k])]
                return float(np.mean(v)) if v else float("nan")
            gc = 1.0 / _UMOL_CO2_TO_GC
            print(f"\n{len(records)} steps ({len(good)} daytime) -> {args.out}")
            print(f"  GPP   umol  ml={mean('gpp_ml', gc):7.2f} bl={mean('gpp_bl', gc):7.2f}")
            print(f"  SIF   umol  ml={mean('sif_ml'):7.2f} bl={mean('sif_bl'):7.2f}")
            print(f"  LH    W/m2  ml={mean('lh_ml'):7.1f} bl={mean('lh_bl'):7.1f}  "
                  f"(veg ml={mean('lhveg_ml'):.0f} bl={mean('lhveg_bl'):.0f})")
            print(f"  SH    W/m2  ml={mean('sh_ml'):7.1f} bl={mean('sh_bl'):7.1f}  "
                  f"(veg ml={mean('shveg_ml'):.0f} bl={mean('shveg_bl'):.0f})")
            print(f"  Rnveg W/m2  ml={mean('rnveg_ml'):7.1f} bl={mean('rnveg_bl'):7.1f}")
            print("  NOTE: total SH carries the offline soil-BC artifact; the "
                  "vegetation (veg) fluxes are the canopy comparison.")
        if args.plot:
            _plot(records, os.path.splitext(args.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

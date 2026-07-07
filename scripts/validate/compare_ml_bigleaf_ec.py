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

The SIF agreement is the headline: two independent canopy representations, fed
identical forcing, should emit consistent top-of-canopy SIF (they do, to
~5-10 % at CHATS7).  Flux differences are expected model-structure differences
(e.g. the multilayer RSL turbulence captures the CHATS oasis/advection regime —
negative daytime sensible heat, LH > net radiation — that a big-leaf cannot).

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
    """Read CLM-ML atmospheric forcing + canopy fluxes + SIF for column ``i``.

    Patch axis is 1-based (the offline driver / interface pad index 0); layer
    axis is 1-based too.  Returns a plain-float dict (JSON-serialisable) with the
    forcing the two-leaf canopy needs plus the multilayer reference fluxes.
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
    dpai = jnp.asarray(mlcanopy.dpai_profile)[i + 1, 1:]
    lai = float(jnp.sum(jnp.where(jnp.isfinite(dpai) & (jnp.abs(dpai) < _SPVAL_GUARD),
                                  jnp.clip(dpai, 0.0, None), 0.0)))
    sif = _extract_clm_ml_sif(mlcanopy, ncol=1, sif_cfg=SIFConfig())
    return dict(
        Ta=t_air, q=q, u=u, P=p_air, co2=co2, lw=lw, sw=sw,
        coszen=float(np.clip(np.cos(zen), 0.0, 1.0)), lai=lai,
        Tg=_san(p("tg_soil"), t_air),
        gpp_ml=_san(p("gppveg_canopy"), 0.0) * _UMOL_CO2_TO_GC,
        sh_ml=_san(p("shflx_canopy"), 0.0),
        lh_ml=_san(p("lhflx_canopy"), 0.0),
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


def run_bigleaf(c: dict, dt: float = 1800.0) -> dict:
    """Drive the two-leaf big-leaf canopy with the captured CLM-ML forcing.

    Soil skin T is prescribed to CLM-ML's ``tg_soil`` (identity thermal
    callback) so the two schemes see the same lower boundary; LAI is matched to
    CLM-ML's total plant-area index.  Returns big-leaf GPP [gC/m2/s], SH, LH
    [W/m2] and SIF [umol/m2/s].
    """
    forcing = build_bigleaf_forcing(c)
    t_soil = jnp.array([c["Tg"]])
    out = compute_two_leaf_canopy_fluxes(
        T_soil_top=t_soil, forcing=forcing,
        canopy_config=TwoLeafCanopyConfig(max_iters=30, sif=SIFConfig()),
        land_config=MultiLayerLandConfig(), canopy_params=None,
        w_frac_rz=jnp.array([0.7]),
        wind_speed=jnp.array([c["u"]]), wind_dir_x=jnp.ones(1), wind_dir_y=jnp.zeros(1),
        soil_thermal_fn=lambda g_flux, dt_: t_soil, dt=dt,
        LAI_override=jnp.array([max(c["lai"], 0.1)]),
    )
    return dict(
        gpp_bl=(float(out.gpp[0]) if out.gpp is not None else None),
        sh_bl=float(out.shflx[0]), lh_bl=float(out.lhflx[0]),
        sif_bl=(float(out.sif[0]) if out.sif is not None else None),
    )


def _plot(records: list, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = [x for x in records if "error" not in x]
    r.sort(key=lambda x: x["step"])
    h = [(x["step"] - r[0]["step"]) * 0.5 for x in r]
    gc = 1.0 / _UMOL_CO2_TO_GC

    def col(k, scale=1.0):
        return [(x.get(k) * scale if x.get(k) is not None else np.nan) for x in r]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a, title, ml, bl in [
        (ax[0, 0], "GPP [umol CO2/m2/s]", col("gpp_ml", gc), col("gpp_bl", gc)),
        (ax[0, 1], "Sensible heat SH [W/m2]", col("sh_ml"), col("sh_bl")),
        (ax[1, 0], "Latent heat LH [W/m2]", col("lh_ml"), col("lh_bl")),
        (ax[1, 1], "SIF [umol photon/m2/s]", col("sif_ml"), col("sif_bl")),
    ]:
        a.plot(h, ml, "o-", color="#1b7837", ms=3, lw=1.8, label="multilayer (CLM-ML)")
        a.plot(h, bl, "s--", color="#c51b7d", ms=3, lw=1.5, label="two-leaf (big-leaf)")
        a.set_title(title)
        a.set_xlabel("hours")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("CLM-ML multilayer vs two-leaf big-leaf — EC tower (same forcing)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--namelist", required=True, help="clm-ml-jax tower namelist path")
    ap.add_argument("--out", default="results/ml_bigleaf_ec/compare.json")
    ap.add_argument("--plot", action="store_true", help="also write a <out>.png diurnal figure")
    args = ap.parse_args(argv)

    jax.config.update("jax_enable_x64", True)
    try:
        import clm_src_cpl.lnd_comp_nuopc as lnd_comp
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
            from clm_src_main import clm_instMod
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
            def mean(k, s=1.0):
                v = [x[k] * s for x in good if x.get(k) is not None]
                return float(np.mean(v)) if v else float("nan")
            gc = 1.0 / _UMOL_CO2_TO_GC
            print(f"\n{len(records)} steps -> {args.out}")
            print(f"  daytime GPP umol  ml={mean('gpp_ml', gc):6.2f} bl={mean('gpp_bl', gc):6.2f}")
            print(f"  daytime SH  W/m2  ml={mean('sh_ml'):6.1f} bl={mean('sh_bl'):6.1f}")
            print(f"  daytime LH  W/m2  ml={mean('lh_ml'):6.1f} bl={mean('lh_bl'):6.1f}")
            print(f"  daytime SIF umol  ml={mean('sif_ml'):6.2f} bl={mean('sif_bl'):6.2f}")
        if args.plot:
            _plot(records, os.path.splitext(args.out)[0] + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

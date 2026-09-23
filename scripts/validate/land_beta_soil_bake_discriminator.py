#!/usr/bin/env python
"""Did the retuned soil-water tables cause the coupled land-evaporation collapse?

WHY THIS EXISTS.  A 30-day coupled A/B (one variable = which per-plant-type bake)
lost two thirds of its land latent heat, 34.7 -> 12.3 W/m2, concentrated in the
snow-free tropics.  Both arms ran with stomata OFF, so the canopy conductance
parameters were not executing in either — whatever moved, it was not plants.  The
only live moisture lever left is the ROOT-ZONE AVAILABILITY the coupler multiplies
into the land surface humidity, ``q_sfc = beta_soil * q_sat(T_land)``, and its
only tuned inputs are the per-plant wilting point, field capacity and root depth.

So one curve is worth having: how does ``beta_soil`` differ between the two bakes
over the soil-moisture range the tropics actually visit?  No model run, no GPU.

IT IS A NECESSARY CONDITION, NOT A SUFFICIENT ONE.  Deployed latent heat is
availability TIMES a bulk flux that depends on the humidity gradient, the surface
temperature and the feedbacks between them.  A ratio far from 1 would make these
tables a live suspect; a ratio near 1 only rules out a DIRECT throttle and leaves
the coupled question open.  This prints numbers, not a verdict.

This calls the SHIPPED ``land_tile_beta_soil`` — the same function the coupler
calls — rather than re-deriving the stress formula, so the number cannot drift
from what the model does.

TWO SOIL STATES, and the second one matters.  A vertically UNIFORM column hides
root depth entirely: the root-zone weighting integrates the same value at every
depth, so a bake that moved root depth looks identical to one that did not.  The
coupled run is never uniform — it evaporates the top dry while the deep layers
stay wet — so the sweep is repeated on a DRYING profile, where root depth is the
dominant term.  Both states are reported; a conclusion drawn from the uniform one
alone would be an artifact of the test, not a result.

Run: PYTHONPATH=. python scripts/validate/land_beta_soil_bake_discriminator.py
"""

from __future__ import annotations

import argparse

import numpy as np
import jax.numpy as jnp

from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import land_tile_beta_soil
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.surface_params import CLM5_PFT_NAMES, LandSurfaceParams
from legoesm.land import clm_surface_map as BAKE

# The bake in force before the dual-target retune (git 85522cd83^), kept here as
# the CONTROL arm.  These are the values the "old bake" arm of the coupled A/B
# ran with; the current ones are read live from the module so this probe cannot
# describe a bake the model no longer has.
_PREV_WP = (0.0846, 0.0915, 0.0942, 0.0907, 0.1154, 0.0988, 0.1162, 0.1062,
            0.1035, 0.0988, 0.0838, 0.0839, 0.0865, 0.0874, 0.0869, 0.0822, 0.0988)
_PREV_FC = (0.1706, 0.2249, 0.2351, 0.2253, 0.2890, 0.2492, 0.2918, 0.2676,
            0.2599, 0.2635, 0.1995, 0.2045, 0.2164, 0.2171, 0.2193, 0.2123, 0.2469)
# const-ok: per-plant-type root-depth table (metres), one entry per PFT --
# the 0.622 here is a tabulated depth, not the molecular-weight ratio.
_PREV_ROOT = (0.088, 1.706, 1.469, 1.414, 1.631, 1.675, 1.527, 1.408, 1.088,
              0.716, 0.622, 0.723, 0.457, 0.507, 0.504, 0.488, 0.477)

# The plant types that carry the collapse region (Amazon / Congo / Maritime
# Continent) and, for contrast, the ones the calibrated-model change was measured
# to hit hardest.
_TROPICAL = (4, 6)          # broadleaf evergreen tropical, broadleaf deciduous tropical
_CONTRAST = (0, 9, 14)      # bare soil, evergreen shrub, C4 grass


def _beta(theta, wp, fc, root, n_layers, total_depth, growth):
    """Root-zone availability from the SHIPPED coupler function, for one column."""
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=n_layers, total_depth=total_depth,
                                 growth_factor=growth))
    n = np.asarray(theta).size
    params = LandSurfaceParams(
        albedo_veg=jnp.full(n, 0.15), emissivity=jnp.full(n, 0.96),
        z0=jnp.full(n, 0.1), W_max=jnp.full(n, 150.0),
        C_soil=jnp.full(n, 2.0e6), d_soil=jnp.full(n, 1.0),
        root_depth=jnp.full(n, root),
        theta_wp=jnp.full(n, wp), theta_fc=jnp.full(n, fc),
        Vc_max25=jnp.full(n, 60.0), LCMA=jnp.full(n, 60.0), g1=jnp.full(n, 9.0))
    theta_col = jnp.broadcast_to(
        jnp.asarray(theta, dtype=jnp.float64)[:, None], (n, n_layers))
    return np.asarray(land_tile_beta_soil(theta_col, cfg, params))


def _beta_profile(theta_prof, wp, fc, root, n_layers, total_depth, growth):
    """Availability for ONE column with a vertical moisture PROFILE.

    Separate from ``_beta`` because that one sweeps many columns at a single
    uniform value; this one is a single column whose layers differ, which is the
    only way root depth can show up at all.
    """
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=n_layers, total_depth=total_depth,
                                 growth_factor=growth))
    params = LandSurfaceParams(
        albedo_veg=jnp.full(1, 0.15), emissivity=jnp.full(1, 0.96),
        z0=jnp.full(1, 0.1), W_max=jnp.full(1, 150.0),
        C_soil=jnp.full(1, 2.0e6), d_soil=jnp.full(1, 1.0),
        root_depth=jnp.full(1, root),
        theta_wp=jnp.full(1, wp), theta_fc=jnp.full(1, fc),
        Vc_max25=jnp.full(1, 60.0), LCMA=jnp.full(1, 60.0), g1=jnp.full(1, 9.0))
    return float(land_tile_beta_soil(
        jnp.asarray(theta_prof, dtype=jnp.float64)[None, :], cfg, params)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--soil-depth", type=float, default=6.375,
                    help="the column BOTH A/B arms ran (not the calibration one)")
    ap.add_argument("--growth", type=float, default=2.0)
    ap.add_argument("--theta-min", type=float, default=0.15)
    ap.add_argument("--theta-max", type=float, default=0.45,
                    help="wet-tropics soil moisture; the A/B's collapse region")
    args = ap.parse_args()

    theta = np.linspace(args.theta_min, args.theta_max, 13)
    cur_wp = np.asarray(BAKE._TUNED_PFT_WP_MULTILAYER)
    cur_fc = np.asarray(BAKE._TUNED_PFT_FC_MULTILAYER)
    cur_root = np.asarray(BAKE._TUNED_PFT_ROOT_DEPTH_MULTILAYER)

    print(f"Root-zone availability, previous bake vs current, on the {args.soil_depth} m "
          f"column both A/B arms ran ({args.n_layers} layers, growth {args.growth}).")
    print("beta_soil multiplies the land surface humidity, so the RATIO is the "
          "predicted change in the evaporation the atmosphere receives.\n")
    print(f"{'plant type':<32}{'theta':>7}{'prev':>8}{'curr':>8}{'ratio':>8}")

    verdict = {}
    for group, label in ((_TROPICAL, "TROPICS (the collapse region)"),
                         (_CONTRAST, "for contrast")):
        print(f"\n--- {label} ---")
        for p in group:
            prev = _beta(theta, _PREV_WP[p], _PREV_FC[p], _PREV_ROOT[p],
                         args.n_layers, args.soil_depth, args.growth)
            curr = _beta(theta, cur_wp[p], cur_fc[p], cur_root[p],
                         args.n_layers, args.soil_depth, args.growth)
            ratio = curr / np.maximum(prev, 1e-12)
            for i in (0, len(theta) // 2, len(theta) - 1):
                name = CLM5_PFT_NAMES[p] if i == 0 else ""
                print(f"{name:<32}{theta[i]:7.2f}{prev[i]:8.3f}{curr[i]:8.3f}"
                      f"{ratio[i]:8.2f}")
            verdict[p] = (float(ratio.min()), float(ratio.max()))

    # The state the coupled run actually reaches: top layers evaporated down
    # toward wilting, deep layers still wet.  Root depth decides how much of
    # each the plants see, and it is invisible in the uniform sweep above.
    print("\n--- DRYING COLUMN (dry top, wet deep) — where ROOT DEPTH decides ---")
    drying = np.array([0.10, 0.11, 0.13, 0.16, 0.22, 0.30, 0.35, 0.38])
    if args.n_layers == len(drying):
        print(f"{'plant type':<32}{'prev':>8}{'curr':>8}{'ratio':>8}")
        for p in _TROPICAL + _CONTRAST:
            bp = _beta_profile(drying, _PREV_WP[p], _PREV_FC[p], _PREV_ROOT[p],
                               args.n_layers, args.soil_depth, args.growth)
            bc = _beta_profile(drying, cur_wp[p], cur_fc[p], cur_root[p],
                               args.n_layers, args.soil_depth, args.growth)
            print(f"{CLM5_PFT_NAMES[p]:<32}{bp:8.3f}{bc:8.3f}"
                  f"{bc / max(bp, 1e-12):8.2f}")
    else:
        print(f"  (skipped: the profile above is written for 8 layers, "
              f"this run has {args.n_layers})")

    print("\nVERDICT INPUTS (ratio range over the sampled moisture band):")
    for p, (lo, hi) in verdict.items():
        print(f"  {CLM5_PFT_NAMES[p]:<32} {lo:.2f} - {hi:.2f}")
    trop_lo = min(lo for p in _TROPICAL for lo, _ in (verdict[p],))
    print(f"\nSmallest tropical availability ratio measured: {trop_lo:.2f}.")
    print("\nWHAT THIS IS, AND IS NOT.  This is the availability factor at a FIXED")
    print("soil state — not an evaporation ratio.  What the atmosphere actually")
    print("evaporates also depends on the humidity gradient, the surface")
    print("temperature and every feedback between them, none of which is here.")
    print("So a ratio near 1 says these tables do not throttle the surface")
    print("DIRECTLY; it does NOT settle what caused a coupled change.  Draw the")
    print("causal conclusion from a coupled pair, not from this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

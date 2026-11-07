#!/usr/bin/env python
"""#1320: how much does the MPAS lane's BLENDED surface cost, in W/m^2?

The issue says the blended-beta decomposition "is exact only for a fixed
shared transfer coefficient; a stability-dependent bulk scheme recomputes
``C_E`` from the throttled ``q_sfc``, so land/ocean stability contrasts are
mis-averaged".  That is an argument, not a number, and three years of the port
plan have gone by without one.  This measures it.

WHAT IS COMPARED, and why it is one variable
--------------------------------------------
Both arms take the SAME lowest-level atmospheric state, the SAME per-tile
surface temperatures and humidities, and the SAME per-tile laws:

  TILED    each tile's flux computed with its own law and surface state, then
           area-weighted -- ``compute_tiled_surface_fluxes``, the shared
           mosaic core the port plan's step 1 asked for and which already
           exists.
  BLENDED  ONE surface temperature and humidity, area-weighted FIRST, then a
           single law applied to the blend -- what the MPAS lane does today.

The only difference is the ORDER of the averaging and the law application.
That is precisely the defect the port removes, so the difference between the
arms IS the defect's size.

THE LAWS ARE RESOLVED, NOT RECONSTRUCTED.  The first attempt at this number
was retracted for inventing the sea-ice scheme and roughness and using a
gustiness depth production does not set.  ``resolve_tiled_surface_configs``
(#1748) exists so that cannot happen again, and this script calls it: pass a
real ``ExperimentConfig`` and the tile laws are the ones that configuration
resolves to.

WHICH LAW DOES THE BLENDED ARM USE?  The MPAS lane applies the experiment's
single ``surface_bulk_scheme`` to the blended surface, which is the OCEAN
tile's law.  So the blended arm uses ``config_ocean``.  That is not an
assumption about the port; it is what the untiled path does.

WHICH ROWS APPLY TO THE PRODUCTION MPAS LANE -- READ THIS BEFORE QUOTING A
NUMBER.  With the interactive multilayer land on (the production AMIP lane),
``_make_mpas_turbulence`` does NOT compute the land fraction's heat fluxes
from a bulk law at all: the driver passes ``forcing["shflx_land"]`` and
``forcing["lhflx_land"]`` -- the land model's own solved turbulent fluxes --
and the turbulence blends them in by land fraction.  Its own comment is
explicit that "momentum and the ocean/ice fraction keep the scheme's own bulk
computation".  So:

  * the three-arm SENSIBLE and LATENT rows describe a lane WITHOUT that
    hand-over (no interactive land).  On the production lane the LAND tile's
    heat is already a real flux from a real land model, so those rows
    OVERSTATE the production defect.
  * the MOMENTUM (tau) row IS production-relevant.  Surface stress over the
    land fraction is still computed by the single blended-surface bulk law,
    with the OCEAN roughness.
  * and so is the PRODUCTION-LANE block at the end, which is the arm the
    first version of this probe was missing.  The lane computes
    ``heat = (1-f) * BULK(T_blend, q_blend) + f * F_land``, so the OCEAN
    fraction's flux is evaluated on the BLENDED surface rather than on the
    ocean surface.  ``F_land`` cancels in the difference against the correct
    form, and what is left is the averaging-order error the issue body
    describes, SURVIVING the land-flux hand-over over the ocean side of every
    coastal cell.  It is not small.

NOT COVERED, and not small either: sea ice.  The ocean and ice fractions are
themselves blended inside the non-land fraction, and the dry-ice against
saturated-ocean humidity contrast is the largest ``q_sfc`` contrast in the
model, so the ice margin is where the order error should be worst.  This
population contains ZERO ice columns.  Untested is not the same as small.

WHAT THIS IS NOT.  It is an offline, single-snapshot, per-column comparison of
one flux calculation.  It does not run the model, so it cannot say what the
difference does to a climate -- only how large the surface-flux error is at
the moment it is made.  The port plan's A/B on a 1-year chain is still the
thing that answers that.

Usage::

    python scripts/validate/quantify_blended_vs_tiled_surface_1320.py
    python scripts/validate/quantify_blended_vs_tiled_surface_1320.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    SurfaceTileSpec,
    compute_surface_fluxes,
    compute_tiled_surface_fluxes,
)
from legoesm.thermo import saturation_mixing_ratio


def _q_sat(T, p):
    """Surface saturation specific humidity from the shared thermo module."""
    r = saturation_mixing_ratio(T, p)
    return r / (1.0 + r)


def build_cases(n_per_regime: int = 400, seed: int = 1320):
    """A coastal-cell population spanning the regimes the issue names.

    The point of interest is a cell that is PART land and PART ocean with a
    real temperature contrast between them -- a cold-air outbreak over a
    coast, the example in the issue body.  Sampling land fraction uniformly
    over (0, 1) and the land-minus-ocean surface temperature contrast over a
    range that includes both signs covers that, and keeps the "both tiles are
    the same" degenerate cases in the sample as the control: where the two
    tiles agree, the two arms must agree exactly.
    """
    rng = np.random.default_rng(seed)
    n = n_per_regime
    frac_land = rng.uniform(0.05, 0.95, n)
    T_ocean = rng.uniform(271.0, 303.0, n)
    dT = rng.uniform(-25.0, 15.0, n)      # land minus ocean; cold-air outbreak
    T_land = T_ocean + dT
    # Lowest-level air: between the two surfaces, plus a spread so both stable
    # and unstable columns occur.
    T_air = 0.5 * (T_ocean + T_land) + rng.uniform(-8.0, 8.0, n)
    wind = rng.uniform(1.0, 20.0, n)
    p_s = rng.uniform(9.6e4, 1.02e5, n)
    beta = rng.uniform(0.1, 1.0, n)       # soil evaporative efficiency
    rho = p_s / (constants.R_d * T_air)
    q_air = 0.6 * _q_sat(T_air, p_s)
    return dict(frac_land=frac_land, T_ocean=T_ocean, T_land=T_land,
                T_air=T_air, wind=wind, p_s=p_s, beta=beta, rho=rho,
                q_air=q_air)


def run(cases, cfg_ocean, cfg_ice, cfg_land):
    """Both arms on the same inputs. Returns per-column (tiled, blended)."""
    f_land = jnp.asarray(cases["frac_land"])
    f_ocean = 1.0 - f_land
    zero = jnp.zeros_like(f_land)
    T_o, T_l = jnp.asarray(cases["T_ocean"]), jnp.asarray(cases["T_land"])
    p_s = jnp.asarray(cases["p_s"])
    beta = jnp.asarray(cases["beta"])
    # Ocean saturation humidity carries the standard 0.98 saline reduction;
    # land is throttled by the soil evaporative efficiency beta.
    q_o = 0.98 * _q_sat(T_o, p_s)
    q_l = beta * _q_sat(T_l, p_s)

    u = jnp.asarray(cases["wind"])
    v = jnp.zeros_like(u)
    T = jnp.asarray(cases["T_air"])
    q = jnp.asarray(cases["q_air"])
    rho = jnp.asarray(cases["rho"])

    tiles = SurfaceTileSpec(
        frac_ocean=f_ocean, frac_ice=zero, frac_land=f_land,
        T_ocean=T_o, T_ice=T_o, T_land=T_l,
        q_sfc_ocean=q_o, q_sfc_ice=q_o, q_sfc_land=q_l)
    tx_t, ty_t, sh_t, lh_t, _ = compute_tiled_surface_fluxes(
        u, v, T, q, rho, tiles, cfg_ocean, cfg_ice, cfg_land)

    # BLENDED: average the surface first, then ONE law (the ocean law, which
    # is what the experiment's single surface_bulk_scheme resolves to).
    T_b = f_ocean * T_o + f_land * T_l
    q_b = f_ocean * q_o + f_land * q_l
    tx_b, ty_b, sh_b, lh_b, _ = compute_surface_fluxes(
        u, v, T, q, T_b, q_b, rho, cfg_ocean)

    # THIRD ARM, so the total is attributable. Per-tile SURFACES but the ocean
    # law on every tile, then area-weighted. Differencing against it splits
    # the blended-minus-tiled total into its two causes:
    #   blended - same_law  = the AVERAGING ORDER at fixed law (the defect
    #                         this issue's body actually describes)
    #   same_law - tiled    = applying the OCEAN law to land (roughness 1e-4
    #                         against 0.1 m, and coare3 against most)
    tx_s, ty_s, sh_s, lh_s, _ = compute_tiled_surface_fluxes(
        u, v, T, q, rho, tiles, cfg_ocean, cfg_ocean, cfg_ocean)

    # FOURTH AND FIFTH ARMS: what the PRODUCTION lane actually computes, and
    # what it should. With the interactive land on, the lane replaces only the
    # land fraction's HEAT with the land model's own flux and keeps
    #     heat = (1-f) * BULK(T_blend, q_blend) + f * F_land
    # i.e. the ocean fraction's flux is evaluated on the BLENDED surface, not
    # on the OCEAN surface. The correct form is
    #     heat = (1-f) * BULK(T_ocean, q_ocean) + f * F_land
    # and the difference between them is the averaging-order error that
    # SURVIVES the land-flux hand-over -- the one the issue body describes,
    # still alive over the ocean side of every coastal cell. (Review finding:
    # this arm was missing, and without it "the heat is already handled" was
    # not earned.) F_land cancels in the difference, so it need not be
    # modelled here.
    _, _, sh_o, lh_o, _ = compute_surface_fluxes(
        u, v, T, q, T_o, q_o, rho, cfg_ocean)
    f_o = f_ocean
    sh_prod = f_o * sh_b
    lh_prod = f_o * lh_b
    sh_prod_ok = f_o * sh_o
    lh_prod_ok = f_o * lh_o

    def _pack(tx, ty, sh, lh):
        return dict(shflx=np.asarray(sh), lhflx=np.asarray(lh),
                    tau=np.asarray(jnp.hypot(tx, ty)))

    return (_pack(tx_t, ty_t, sh_t, lh_t),
            _pack(tx_b, ty_b, sh_b, lh_b),
            _pack(tx_s, ty_s, sh_s, lh_s),
            dict(shflx=np.asarray(sh_prod), lhflx=np.asarray(lh_prod)),
            dict(shflx=np.asarray(sh_prod_ok), lhflx=np.asarray(lh_prod_ok)))


def _stats(d):
    return dict(mean_signed=float(np.mean(d)),
                mean_abs=float(np.mean(np.abs(d))),
                p95_abs=float(np.percentile(np.abs(d), 95)),
                max_abs=float(np.max(np.abs(d))))


def summarise(tiled, blended, same_law, mask=None):
    """Total difference, split into averaging-order and tile-law parts."""
    sel = slice(None) if mask is None else mask
    out = {}
    for key, unit in (("shflx", "W/m^2"), ("lhflx", "W/m^2"), ("tau", "Pa")):
        out[key] = dict(
            unit=unit,
            total=_stats(blended[key][sel] - tiled[key][sel]),
            averaging_order=_stats(blended[key][sel] - same_law[key][sel]),
            tile_law=_stats(same_law[key][sel] - tiled[key][sel]),
            mean_tiled=float(np.mean(tiled[key][sel])),
            n=int(np.sum(np.ones_like(tiled[key])[sel])),
        )
    return out


def control(cfg_ocean, cfg_ice, cfg_land, n=200, seed=7):
    """Identical tiles => the two arms must agree to round-off."""
    rng = np.random.default_rng(seed)
    T_s = rng.uniform(275.0, 300.0, n)
    p_s = rng.uniform(9.7e4, 1.01e5, n)
    cases = dict(frac_land=rng.uniform(0.05, 0.95, n),
                 T_ocean=T_s, T_land=T_s,
                 T_air=T_s + rng.uniform(-5.0, 5.0, n),
                 wind=rng.uniform(1.0, 15.0, n), p_s=p_s,
                 beta=np.full(n, 0.98 / 1.0), rho=None, q_air=None)
    cases["rho"] = cases["p_s"] / (constants.R_d * cases["T_air"])
    cases["q_air"] = 0.6 * np.asarray(_q_sat(jnp.asarray(cases["T_air"]),
                                             jnp.asarray(cases["p_s"])))
    # beta chosen so the land tile's q_sfc equals the ocean tile's 0.98*q_sat.
    cases["beta"] = np.full(n, 0.98)
    t, b = run(cases, cfg_ocean, cfg_ocean, cfg_ocean)[:2]
    return max(float(np.max(np.abs(b[k] - t[k]))) for k in ("shflx", "lhflx"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400,
                    help="columns to sample (default 400)")
    ap.add_argument("--json", default="", help="write the summary here")
    args = ap.parse_args(argv)

    from legoesm.driver.config import (DycoreConfig, ExperimentConfig,
                                       GridConfig, OutputConfig)
    from legoesm.driver.physics_pipeline import resolve_tiled_surface_configs

    # The MPAS AMIP configuration's surface settings, named explicitly rather
    # than defaulted (the resolver reads them off this object).
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=5, nlev=30,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(discretization="mpas", dt=75.0),
        output=OutputConfig(output_dir="", diag_days=0),
        days=1, dataset="analytical", radiation="gray",
        convection="none", turbulence="louis",
        surface_bulk_scheme="coare3", precision="fp64")
    cfg_ocean, cfg_ice, cfg_land = resolve_tiled_surface_configs(cfg)
    if cfg_ocean is None:
        print("resolve_tiled_surface_configs returned no surface-layer config "
              "for this experiment; nothing to compare.", file=sys.stderr)
        return 2

    ctrl = control(cfg_ocean, cfg_ice, cfg_land)
    print(f"INSTRUMENT CONTROL (identical tiles, blended must equal tiled): "
          f"max |difference| = {ctrl:.3e} W/m^2 "
          f"{'PASS' if ctrl < 1e-6 else 'FAIL -- everything below is void'}")
    if ctrl >= 1e-6:
        return 1

    print(f"\ntile laws resolved from the configuration:")
    for name, c in (("ocean", cfg_ocean), ("ice", cfg_ice), ("land", cfg_land)):
        print(f"  {name:6s} bulk_scheme={getattr(c, 'bulk_scheme', '?')!r} "
              f"z0={getattr(c, 'z0', None)}")

    cases = build_cases(args.n)
    tiled, blended, same_law, prod, prod_ok = run(
        cases, cfg_ocean, cfg_ice, cfg_land)

    # TWO populations, reported separately (never one frightening number with
    # no provenance): the full sampled box, which deliberately includes severe
    # cold-air outbreaks, and a TYPICAL subset.
    typical = (np.abs(cases["T_land"] - cases["T_ocean"]) <= 10.0) & \
              (cases["wind"] <= 12.0)
    pops = (("FULL sampled box (|dT| <= 25 K, wind <= 20 m/s)", None),
            ("TYPICAL subset  (|dT| <= 10 K, wind <= 12 m/s)", typical))

    summary = {}
    for label, mask in pops:
        srep = summarise(tiled, blended, same_law, mask)
        summary[label] = srep
        n = srep["shflx"]["n"]
        print(f"\n{label} -- {n} columns")
        print(f"{'flux':7s} {'part':17s} {'mean signed':>12s} "
              f"{'mean |d|':>10s} {'p95 |d|':>10s} {'max |d|':>10s}")
        for k, v in srep.items():
            tag = ("  <-- PRODUCTION-RELEVANT (stress over land still uses "
                   "the blended ocean law)" if k == "tau" else
                   "  (overstates the production lane: interactive land "
                   "hands its own heat fluxes over)")
            print(f"{k:7s}{tag}")
            for part in ("total", "averaging_order", "tile_law"):
                p_ = v[part]
                print(f"{k:7s} {part:17s} {p_['mean_signed']:12.3f} "
                      f"{p_['mean_abs']:10.3f} {p_['p95_abs']:10.3f} "
                      f"{p_['max_abs']:10.3f}   [{v['unit']}]")
            print(f"{'':7s} {'(mean tiled flux)':17s} {v['mean_tiled']:12.3f}")

    # THE PRODUCTION-LANE HEAT ORDER ERROR -- the arm the first version of
    # this probe was missing. Land heat comes from the land model on both
    # sides and cancels; what is left is the ocean fraction evaluated on the
    # blended surface instead of the ocean surface.
    print("\nPRODUCTION-LANE ocean-fraction heat error "
          "(interactive land ON; land flux cancels):")
    print(f"{'flux':8s} {'mean signed':>12s} {'mean |d|':>10s} "
          f"{'p95 |d|':>10s} {'max |d|':>10s} {'mean prod':>10s}")
    prod_summary = {}
    for label, mask in pops:
        for k in ("shflx", "lhflx"):
            sel = slice(None) if mask is None else mask
            d = prod[k][sel] - prod_ok[k][sel]
            st = _stats(d)
            st["mean_production_flux"] = float(np.mean(prod[k][sel]))
            prod_summary[f"{label}|{k}"] = st
            print(f"{k:8s} {st['mean_signed']:12.3f} {st['mean_abs']:10.3f} "
                  f"{st['p95_abs']:10.3f} {st['max_abs']:10.3f} "
                  f"{st['mean_production_flux']:10.3f}   [{label.split()[0]}]")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(control_max_abs=ctrl, summary=summary,
                           production_ocean_fraction=prod_summary,
                           n=args.n), fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

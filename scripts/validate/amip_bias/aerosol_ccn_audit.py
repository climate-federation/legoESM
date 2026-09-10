"""What cloud-droplet number does the production aerosol forcing imply? (#1521)

The run sets ``aerosol_ccn``, so the droplet number radiation sees is NOT the
microphysics tracer: it is diagnosed from column aerosol optical depth through
the Andreae (2009) inversion.  This reads the forcing file production actually
loads and puts it through the MODEL's own ``ccn_from_aod`` -- never a retyped
formula, because a re-derived lookalike silently answers a different question
and the default ``CCNFromAODConfig`` is itself part of what is being measured.

WHY THE COLUMN VALUE IS THE FILE'S VALUE: the driver spreads the column AOD
over layers with pressure-thickness weights that sum to one
(``distribute_column_aod_to_layers``) and the radiation entry sums them
straight back (``specified_nc_field``), so the number the inversion sees is
the file's, undisturbed.

READING: no threshold is pre-registered here because this is an inventory of
what the configuration implies, not a hypothesis test.  The comparison that
gives it meaning is the issue's own observational anchor -- r_eff 7.84 um
modelled against 11-14 um observed -- inverted through r_eff ~ N^(-1/3) at
fixed liquid water.  That exponent is exact only at FIXED gamma shape; the
Morrison shape parameter itself depends on N through a Martin et al. (1994)
fit, so the implied factors below are indicative, not exact (codex review).

Numbers only, no verdicts.
"""
import argparse
import pathlib

import numpy as np


def cos_lat_weights(lat_deg):
    """Area weights for a zonal field on a latitude axis, normalised to 1.

    A zonal-mean field still needs cos(lat) weighting to become a global
    mean; a plain mean over latitude rows overstates the poles, which on
    this field are the CLEANEST rows and would flatter the answer.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    if lat.size == 0:
        raise ValueError("empty latitude axis")
    # A latitude axis handed over in RADIANS is the failure this guard exists
    # for: deg2rad would shrink it to near zero, cos would be ~1 everywhere,
    # and the result would be a UNIFORM mean wearing the name of an
    # area-weighted one -- wrong with no error. Real latitude axes span far
    # more than 2*pi degrees.
    if np.max(np.abs(lat)) <= 2.0 * np.pi:
        raise ValueError(
            f"latitude axis spans only +/-{np.max(np.abs(lat)):.3f}; that "
            "looks like RADIANS, and weighting it as degrees would silently "
            "return a uniform mean (degrees expected).")
    w = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    total = w.sum()
    # cos(90 deg) is 6.1e-17 rather than 0, so a poles-only axis sums to a
    # positive-but-meaningless total; compare against the axis length.
    if not np.isfinite(total) or total <= 1e-8 * lat.size:
        raise ValueError(
            "cos(lat) weights sum to ~0, so every row sits at a pole and no "
            "area-weighted mean is defined for this axis.")
    return w / total


def area_weighted_annual_mean(field, lat_deg):
    """Annual then area mean of a ``(n_time, n_lat)`` zonal field."""
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"expected (n_time, n_lat), got shape {f.shape}")
    if f.shape[1] != len(lat_deg):
        raise ValueError(f"field has {f.shape[1]} latitudes but the axis has "
                         f"{len(lat_deg)} -- wrong file or transposed array")
    return float((f.mean(axis=0) * cos_lat_weights(lat_deg)).sum())


def implied_ccn_cm3(aod):
    """Droplet number [cm^-3] the MODEL would diagnose from this AOD.

    Delegates to ``legoesm...aerosol_activation.ccn_from_aod`` with the
    default config -- the point of the audit is what the shipped default
    produces, so re-deriving the fit here would measure something else.
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
        CCNFromAODConfig, ccn_from_aod)
    return np.asarray(ccn_from_aod(jnp.asarray(np.asarray(aod, np.float64)),
                                   CCNFromAODConfig())) / 1.0e6


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("forcing", type=pathlib.Path,
                   help="the aerosol climatology the run loads (aod, lat)")
    p.add_argument("--reff-model-um", type=float, default=7.84,
                   help="modelled droplet effective radius, this issue's value")
    p.add_argument("--reff-obs-um", type=float, nargs="+",
                   default=[11.0, 12.0, 14.0], help="observed radii to invert")
    args = p.parse_args(argv)

    import netCDF4 as nc
    from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
        CCNFromAODConfig)
    cfg = CCNFromAODConfig()
    print("CCNFromAODConfig defaults:", cfg)

    d = nc.Dataset(args.forcing)
    aod = np.asarray(d["aod"][:], dtype=np.float64)
    lat = np.asarray(d["lat"][:], dtype=np.float64)
    if aod.ndim != 2:
        raise SystemExit(f"aod has shape {aod.shape}; this audit is written "
                         "for a ZONAL-MEAN (time, lat) field and would "
                         "silently mis-weight anything else.")
    N = implied_ccn_cm3(aod)

    print(f"forcing: {args.forcing}")
    print(f"  shape {aod.shape} = (months, latitudes) -- no longitude axis, "
          "so no land/ocean contrast is representable")
    print(f"AOD 550nm  min {aod.min():.4f}  max {aod.max():.4f}  "
          f"area-wtd annual mean {area_weighted_annual_mean(aod, lat):.4f}")
    print(f"N_c [cm-3] min {N.min():.1f}  max {N.max():.1f}  "
          f"area-wtd annual mean {area_weighted_annual_mean(N, lat):.1f}")
    print()
    print(f"{'lat':>7} {'JanAOD':>8} {'JanN_c':>9} {'JulN_c':>9}")
    for L in (-70, -60, -45, -30, -15, 0, 15, 30, 45, 60, 70):
        i = int(np.argmin(np.abs(lat - L)))
        jul = N[6, i] if N.shape[0] >= 7 else float("nan")
        print(f"{lat[i]:7.1f} {aod[0,i]:8.4f} {N[0,i]:9.1f} {jul:9.1f}")
    print()
    band = (lat < -40) & (lat > -65)
    if band.any():
        print("Southern Ocean 40-65S (essentially all ocean): annual mean "
              f"N_c = {area_weighted_annual_mean(N[:, band], lat[band]):.1f} cm-3")
    print()
    print("Clipping check -- the fit is unconstrained outside its bounds, so a "
          "result sitting on a clip is the clip's number, not the fit's:")
    print(f"  at the {cfg.n_ccn_min_cm3:g} cm-3 floor: "
          f"{100.0*float((N <= cfg.n_ccn_min_cm3*1.0001).mean()):.2f}% of points")
    print(f"  at the {cfg.n_ccn_max_cm3:g} cm-3 cap  : "
          f"{100.0*float((N >= cfg.n_ccn_max_cm3*0.9999).mean()):.2f}% of points")
    print()
    gmean = area_weighted_annual_mean(N, lat)
    print(f"Inverting the issue's own anchor (modelled r_eff "
          f"{args.reff_model_um:g} um) through r_eff ~ N^(-1/3) at fixed "
          "liquid water.")
    print("EXACT ONLY AT FIXED GAMMA SHAPE: Morrison's shape parameter itself "
          "depends on N (Martin 1994), so read these as indicative.")
    for obs in args.reff_obs_um:
        ratio = (obs / args.reff_model_um) ** 3
        print(f"  r_eff {obs:g} um needs N_c smaller by {ratio:.2f}x  ->  "
              f"{gmean/ratio:.0f} cm-3   (model: {gmean:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

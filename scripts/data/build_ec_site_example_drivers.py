"""Build the checked-in, self-documenting EC-site example DRIVERS.

Extracts, from the large external DifferBESS half-hourly drivers, a small
reproducibility subset: only the analysis-window years and only the variables the
offline two-leaf-canopy model actually consumes (forcing) and is evaluated against
(observed targets + the closure-corrected band), plus the gap-fill provenance
flags.  The source float64 precision is kept (the prognostic soil moisture drifts
under a float32 downcast) and the file is zlib-compressed, and every variable gets
a unit + long_name + description while the dataset carries the original
FLUXNET/DifferBESS provenance.  The result runs end to end through
``scripts/run/run_ec_site_evaluation.py`` (Section B of the runbook) from the repo
alone.

    python scripts/data/build_ec_site_example_drivers.py \
        --src-dir /path/to/DifferBESS/data/sitelevel/nc \
        --out-dir scripts/validate/ec_site_example_drivers

See docs/land/ec_site_evaluation_runbook.md.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

from legoesm.land.boundary_data import ec_site

# (site, year_lo, year_hi) — the exact windows used for the checked-in figures.
SITES = [("US-MMS", 2015, 2018), ("DE-Obe", 2015, 2018),
         ("US-Ton", 2015, 2018), ("DE-Hai", 2010, 2013)]

# The variables the pipeline touches, grouped by role.  FORCING is taken straight
# from the reader's required-input list so this never drifts from the model.
_FORCING = list(ec_site._REQUIRED)            # SW/LW/TA/VPD/PA/WS/P/CO2/SZA/SWC/TS/LAI/CI/...
_TARGETS = ["ET", "GPP_DT", "H", "NEE", "USTAR", "ET_CORR", "H_CORR"]  # evaluated against (raw)
# EMISSIVITY is consumed by the reader (CanopyLandParams.emissivity) but is NOT in
# ``_REQUIRED`` (it has a 0.97 fallback), so it must be listed explicitly here or a
# trimmed driver would silently fall back and diverge from the full-driver run.
_META = ["EMISSIVITY", "Vcmax25_C3Leaf", "IGBP", "CLIMATE", "C4", "CANOPY_HEIGHT", "LAT", "LONG"]
_FLAGS = ["met_atm_filled", "met_any_filled", "soil_filled"]          # gap-fill provenance
KEEP = list(dict.fromkeys(_FORCING + _TARGETS + _META + _FLAGS))

# Per-variable (units, long_name, description).  units follow the source
# conventions (VPD hPa, PA kPa, SWC %, ET mm/day) that the reader converts internally.
METADATA: dict[str, tuple[str, str, str]] = {
    # --- meteorological forcing (model inputs; gap-filled) ---
    "SW_IN": ("W m-2", "incoming shortwave radiation", "Downwelling shortwave irradiance at the surface."),
    "LW_IN": ("W m-2", "incoming longwave radiation", "Downwelling longwave irradiance at the surface."),
    "TA": ("degC", "air temperature", "Air temperature at the reference height."),
    "VPD": ("hPa", "vapour pressure deficit", "Atmospheric VPD; the reader converts hPa->Pa."),
    "PA": ("kPa", "air pressure", "Surface air pressure; the reader converts kPa->Pa."),
    "WS": ("m s-1", "wind speed", "Horizontal wind speed at the reference height."),
    "P": ("mm", "precipitation", "Precipitation depth accumulated over the 30-min step."),
    "CO2": ("umol mol-1", "CO2 mole fraction", "Atmospheric CO2 dry-air mole fraction (ppm)."),
    "SZA": ("degree", "solar zenith angle", "Solar zenith angle; >90 deg is night (cos clipped to 0)."),
    "SWC": ("percent", "volumetric soil water content", "Near-surface soil moisture (%); reader uses SWC/100 [m3 m-3] for the prognostic soil initial condition. Gap-fill artifacts at some sites (see runbook)."),
    "TS": ("degC", "soil temperature", "Near-surface soil temperature."),
    "LAI": ("m2 m-2", "leaf area index", "Selected (MODIS-derived) LAI, clipped to [0, 7]."),
    "CI": ("1", "foliage clumping index", "Canopy clumping index for the sunlit/shaded partition."),
    "T_GROWTH": ("degC", "growth temperature", "30-day running-mean air temperature (photosynthesis acclimation)."),
    "Albedo_BSA_vis": ("1", "black-sky albedo, visible", "MODIS directional-hemispherical albedo, visible band."),
    "Albedo_WSA_vis": ("1", "white-sky albedo, visible", "MODIS bihemispherical albedo, visible band."),
    "Albedo_BSA_nir": ("1", "black-sky albedo, near-infrared", "MODIS directional-hemispherical albedo, NIR band."),
    "Albedo_WSA_nir": ("1", "white-sky albedo, near-infrared", "MODIS bihemispherical albedo, NIR band."),
    "BESS_PAR_DIFF_PAR_RATIO": ("1", "diffuse PAR fraction", "BESSRad diffuse-to-total PAR ratio; drives the direct/diffuse split."),
    "EMISSIVITY": ("1", "surface emissivity", "Broadband surface emissivity for the longwave balance."),
    # --- site parameters / PFT metadata ---
    "Vcmax25_C3Leaf": ("umol m-2 s-1", "Vcmax at 25 C (C3 leaf)", "Max RuBP carboxylation rate at 25 C, C3 leaves; NaN where unavailable (reader falls back to a PFT/climate lookup)."),
    "IGBP": ("1", "IGBP land-cover class code", "Numeric IGBP land-cover class, mapped to the plant functional type."),
    "CLIMATE": ("1", "climate class code", "Numeric climate class used in the PFT/aerodynamic lookup."),
    "C4": ("1", "C4 photosynthesis flag", "1 where C4 photosynthesis applies, 0 for C3."),
    "CANOPY_HEIGHT": ("m", "canopy height", "Site canopy height (roughness / displacement height)."),
    "LAT": ("degrees_north", "site latitude", "Site latitude."),
    "LONG": ("degrees_east", "site longitude", "Site longitude."),
    # --- observed EVALUATION TARGETS (raw eddy covariance; NaN gaps retained) ---
    "ET": ("mm day-1", "evapotranspiration (observed)", "RAW observed ET; reader forms latent heat LE = ET*L_v/86400 [W m-2]. Evaluation target."),
    "GPP_DT": ("umol m-2 s-1", "GPP, daytime partitioning (observed)", "RAW observed gross primary productivity (daytime NEE partitioning). Evaluation target."),
    "H": ("W m-2", "sensible heat flux (observed)", "RAW observed sensible heat flux. Evaluation target."),
    "NEE": ("umol m-2 s-1", "net ecosystem exchange (observed)", "RAW observed net ecosystem exchange."),
    "USTAR": ("m s-1", "friction velocity (observed)", "RAW observed friction velocity."),
    "ET_CORR": ("mm day-1", "energy-balance-corrected ET (observed)", "Closure-corrected ET (FLUXNET ET_CORR); band edge in the figures."),
    "H_CORR": ("W m-2", "energy-balance-corrected sensible heat (observed)", "Closure-corrected sensible heat (FLUXNET H_CORR); band edge."),
    # --- gap-fill provenance flags ---
    "met_atm_filled": ("1", "atmospheric-forcing gap-fill flag", "1 where >=1 atmospheric forcing field (TA/VPD/SW_IN/LW_IN/PA) was gap-filled this step."),
    "met_any_filled": ("1", "any-forcing gap-fill flag", "1 where >=1 forcing field was gap-filled this step."),
    "soil_filled": ("1", "soil-field gap-fill flag", "1 where a soil field (TS/SWC) was gap-filled this step."),
}


def build_one(site: str, year_lo: int, year_hi: int, src_dir: str, out_dir: str) -> str:
    """Extract one site's trimmed, annotated driver; return the output path."""
    src = os.path.join(src_dir, f"{site}_driver_v2_gapfree.nc")
    ds = xr.open_dataset(src)
    t = pd.DatetimeIndex(ds.time.values)
    sel = np.nonzero((t.year.to_numpy() >= year_lo) & (t.year.to_numpy() <= year_hi))[0]
    if sel.size == 0:
        raise ValueError(f"{site}: no steps in [{year_lo}, {year_hi}]")
    have = [v for v in KEEP if v in ds.data_vars]
    sub = ds[have].isel(time=sel)

    # Keep the source float64 precision: the soil moisture is prognostic, so a
    # float32 downcast drifts over the multi-year scan and the run no longer
    # reproduces the committed outputs bit-for-bit.  zlib still compresses well.
    enc = {}
    for v in sub.data_vars:
        sub[v].attrs = {}                                   # replace with canonical metadata
        if v in METADATA:
            u, ln, desc = METADATA[v]
            sub[v].attrs.update(units=u, long_name=ln, description=desc)
        enc[v] = {"zlib": True, "complevel": 5}

    tt = pd.DatetimeIndex(sub.time.values)
    sub.attrs = {                                           # dataset provenance
        "title": f"Trimmed EC-site driver for the legoESM two-leaf-canopy validation — {site}",
        "site": site,
        "summary": ("Analysis-window subset of the DifferBESS half-hourly driver, "
                    "keeping only the variables the offline two-leaf-canopy model "
                    "drives on and is evaluated against, for repository-local "
                    "reproducibility of docs/land/ec_site_evaluation_runbook.md."),
        "source": "DifferBESS site-level driver v2 (gap-free): FLUXNET2015 FULLSET meteorology + MODIS LAI/albedo + BESSRad PAR + site parameters",
        # preserved from the source file:
        "gapfree_source": str(ds.attrs.get("gapfree_source", "")),
        "gapfree_method": str(ds.attrs.get("gapfree_method", "")),
        "temporal_resolution": "30 minutes",
        "time_coverage_start": str(tt[0]),
        "time_coverage_end": str(tt[-1]),
        "targets_are_raw": ("Evaluation targets (ET, GPP_DT, H, NEE, USTAR, ET_CORR, "
                            "H_CORR) are RAW observed and retain NaN gaps; only the "
                            "meteorological forcing is gap-filled."),
        "extraction": ("scripts/data/build_ec_site_example_drivers.py: subset to the "
                       "analysis window, kept model-consumed variables only, source "
                       "float64 precision retained, zlib-compressed."),
        "references": "docs/land/ec_site_evaluation_runbook.md",
        "Conventions": "CF-1.8",
    }
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{site}_driver_v2_gapfree.nc")
    sub.to_netcdf(out, encoding=enc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True, help="dir with <SITE>_driver_v2_gapfree.nc")
    ap.add_argument("--out-dir", default="scripts/validate/ec_site_example_drivers",
                    help="output dir for the trimmed drivers")
    args = ap.parse_args()
    total = 0
    for site, lo, hi in SITES:
        p = build_one(site, lo, hi, args.src_dir, args.out_dir)
        mb = os.path.getsize(p) / 1e6
        total += mb
        print(f"  {site}: {p}  ({mb:.2f} MB)")
    print(f"  total: {total:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

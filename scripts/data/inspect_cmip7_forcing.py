#!/usr/bin/env python
"""Extract structural metadata from CMIP7 input4MIPs NetCDF forcing files on HPC.

Files live under /work/bd1179/CMIP7_forcings_raw/<category>/.  For each category
we pick one representative file (the CMIP7 distribution scatters variables across
many per-species/per-chunk files, so a full scan is unnecessary for a schema report).
Output is JSON on stdout — mirrors the CMIP6 inspector on ap/cmip6_forcing_metadata.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import xarray as xr


# Raw CMIP7 forcing staging root (Levante default); override per machine.
ROOT = os.environ.get("LEGOESM_CMIP7_RAW", "/work/bd1179/CMIP7_forcings_raw")

# Representative file per category. Globs are resolved to the first alphabetical
# match so that the report reflects a concrete schema, not an abstract pattern.
FILE_PATTERNS = {
    # --- Solar (SOLARIS-HEPPA CMIP7) ---
    "solar_cmip7_monthly": "solar/multiple_input4MIPs_solar_CMIP_SOLARIS-HEPPA-CMIP-4-6_gn_185001-202312.nc",
    "solar_cmip7_daily": "solar/multiple_input4MIPs_solar_CMIP_SOLARIS-HEPPA-CMIP-4-6_gn_18500101-20231231.nc",
    "solar_cmip7_full": "solar/multiple_input4MIPs_solar_CMIP_SOLARIS-HEPPA-CMIP-4-6_gn.nc",

    # --- GHG concentrations (CR CMIP7) ---
    # Three grid labels: gm (global-mean scalar), gnz (native lat×plev), gr1z (1° zonal)
    "ghg_co2_gm": "GHG_concentrations/co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-1-0-0_gm_175001-202212.nc",
    "ghg_co2_gnz": "GHG_concentrations/co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-1-0-0_gnz_175001-202212.nc",
    "ghg_co2_gr1z": "GHG_concentrations/co2_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-1-0-0_gr1z_175001-202212.nc",
    "ghg_ch4_gm": "GHG_concentrations/ch4_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-1-0-0_gm_175001-202212.nc",
    "ghg_n2o_gm": "GHG_concentrations/n2o_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-1-0-0_gm_175001-202212.nc",
    "ghg_cfc11_gm": "GHG_concentrations/cfc11_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-1-0-0_gm_175001-202212.nc",
    "ghg_cfc12_gm": "GHG_concentrations/cfc12_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-1-0-0_gm_175001-202212.nc",

    # --- Ozone (FZJ CMIP7) ---
    "ozone_cmip7_hist": "ozone/vmro3_input4MIPs_ozone_CMIP_FZJ-CMIP-ozone-2-0_gn_200001-202212.nc",
    "ozone_cmip7_clim": "ozone/vmro3_input4MIPs_ozone_CMIP_FZJ-CMIP-ozone-1-2_gn_185001-185012-clim.nc",
    "ozone_zmta": "ozone/zmta_input4MIPs_ozone_CMIP_FZJ-CMIP-ozone-1-2_gn_185001-202212.nc",

    # --- SST / sea-ice (PCMDI AMIP CMIP7) ---
    "sst_tos": "sst_and_seaice/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc",
    "sst_tosbcs": "sst_and_seaice/tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc",
    "sic_siconc": "sst_and_seaice/siconc_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc",
    "sic_siconcbcs": "sst_and_seaice/siconcbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc",
    "sst_sic_areacello": "sst_and_seaice/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn.nc",
    "sst_sic_sftof": "sst_and_seaice/sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn.nc",

    # --- Aerosol properties (UOEXETER CMIP7) ---
    # Seven optical quantities on a shared grid; also a 12-month climatology
    "aerosol_aod_clim": "aerosol/ext_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_185001-202112-clim.nc",
    "aerosol_asy_clim": "aerosol/asy_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_185001-202112-clim.nc",
    "aerosol_ssa_clim": "aerosol/ssa_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_185001-202112-clim.nc",
    "aerosol_nd_clim": "aerosol/nd_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_185001-202112-clim.nc",
    "aerosol_reff_clim": "aerosol/reff_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_185001-202112-clim.nc",
    "aerosol_sad_clim": "aerosol/sad_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_185001-202112-clim.nc",
    "aerosol_vd_clim": "aerosol/vd_input4MIPs_aerosolProperties_CMIP_UOEXETER-CMIP-2-2-1_gnz_185001-202112-clim.nc",

    # --- Emissions (CEDS anthropogenic + DRES biomass burning) ---
    "emissions_so2_anthro": "emissions/SO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc",
    "emissions_co2_anthro": "emissions/CO2-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc",
    "emissions_bc_anthro": "emissions/BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc",
    "emissions_bc_air_anthro": "emissions/BC-em-AIR-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_200001-202312.nc",
    "emissions_bc_solid_biofuel": "emissions/BC-em-SOLID-BIOFUEL-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18-supplemental_gn_200001-202312.nc",
    "emissions_bb_so2": "emissions/SO2_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc",
    "emissions_bb_bc": "emissions/BC_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_190001-202312.nc",
    "emissions_bb_percent_agri_bc": "emissions/BCpercentageAGRI_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn_175001-202312.nc",
    "emissions_areacella": "emissions/areacella_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn.nc",
    "emissions_gridcellarea": "emissions/gridcellarea_input4MIPs_emissions_CMIP_DRES-CMIP-BB4CMIP7-2-1_gn.nc",

    # --- Atmospheric state (Imperial College CMIP7) ---
    "atmstate_delta13co2": "atmospheric_state/delta13co2_input4MIPs_atmosphericState_C4MIP_ImperialCollege-3-0_gm_1700-2023.nc",
    "atmstate_Delta14co2": "atmospheric_state/Delta14co2_input4MIPs_atmosphericState_C4MIP_ImperialCollege-3-0_gz_1700-2023.nc",

    # --- Land state (UofMD CMIP7 LUH3) ---
    "land_states": "land_state/multiple-states_input4MIPs_landState_CMIP_UofMD-landState-3-1-2_gn_0850-2024.nc",
    "land_management": "land_state/multiple-management_input4MIPs_landState_CMIP_UofMD-landState-3-1-2_gn_0850-2024.nc",
    "land_transitions": "land_state/multiple-transitions_input4MIPs_landState_CMIP_UofMD-landState-3-1-2_gn_0850-2023.nc",
    "land_static": "land_state/multiple-static_input4MIPs_landState_CMIP_UofMD-landState-3-1-2_gn.nc",
}

BASIC_COORDS = {
    "lat", "latitude", "lon", "longitude", "time", "time_bnds",
    "lat_bnds", "lon_bnds", "time_bounds", "lat_bounds", "lon_bounds",
    "bnds", "nv",
}


def resolve_path(rel_pattern: str) -> str | None:
    full_pattern = os.path.join(ROOT, rel_pattern)
    if "*" in full_pattern:
        matches = sorted(glob.glob(full_pattern))
        return matches[0] if matches else None
    return full_pattern if os.path.exists(full_pattern) else None


def inspect_file(path: str) -> dict:
    out: dict = {"path": path}
    try:
        size = os.path.getsize(path)
        out["size_bytes"] = int(size)
    except OSError:
        pass
    try:
        ds = xr.open_dataset(path, decode_times=False)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    try:
        out["dimensions"] = {d: int(s) for d, s in ds.dims.items()}

        coord_sizes: dict = {}
        for cname, cvar in ds.coords.items():
            try:
                arr = cvar.values
                entry = {
                    "size": int(cvar.size),
                    "dtype": str(arr.dtype),
                    "dims": list(cvar.dims),
                }
                if arr.ndim == 1 and arr.size > 0 and arr.size <= 250:
                    try:
                        entry["first"] = arr.flat[0].item() if hasattr(arr.flat[0], "item") else str(arr.flat[0])
                        entry["last"] = arr.flat[-1].item() if hasattr(arr.flat[-1], "item") else str(arr.flat[-1])
                    except Exception:
                        entry["first"] = str(arr.flat[0])
                        entry["last"] = str(arr.flat[-1])
                elif arr.ndim == 1 and arr.size > 250:
                    try:
                        entry["first"] = arr.flat[0].item() if hasattr(arr.flat[0], "item") else str(arr.flat[0])
                        entry["last"] = arr.flat[-1].item() if hasattr(arr.flat[-1], "item") else str(arr.flat[-1])
                    except Exception:
                        pass
                units = cvar.attrs.get("units")
                if units is not None:
                    entry["units"] = str(units)
                calendar = cvar.attrs.get("calendar")
                if calendar is not None:
                    entry["calendar"] = str(calendar)
                coord_sizes[cname] = entry
            except Exception as exc:
                coord_sizes[cname] = {"error": f"{type(exc).__name__}: {exc}"}
        out["coords"] = coord_sizes

        variables: dict = {}
        for vname, var in ds.data_vars.items():
            if vname in BASIC_COORDS:
                continue
            v_entry = {
                "dims": list(var.dims),
                "shape": [int(s) for s in var.shape],
                "dtype": str(var.dtype),
            }
            for ak in ("units", "long_name", "standard_name", "cell_methods"):
                if ak in var.attrs:
                    v_entry[ak] = str(var.attrs[ak])
            variables[vname] = v_entry
        out["variables"] = variables

        global_attrs: dict = {}
        for ak in (
            "title", "source_id", "source", "institution_id", "institution",
            "experiment_id", "frequency", "grid_label", "table_id",
            "Conventions", "tracking_id", "activity_id", "mip_era",
            "variable_id", "target_mip", "dataset_category",
        ):
            if ak in ds.attrs:
                global_attrs[ak] = str(ds.attrs[ak])
        if global_attrs:
            out["global_attrs"] = global_attrs
    finally:
        ds.close()

    return out


def main() -> int:
    result: dict = {}
    for key, pattern in FILE_PATTERNS.items():
        resolved = resolve_path(pattern)
        if resolved is None:
            result[key] = {
                "pattern": os.path.join(ROOT, pattern),
                "error": "no_match",
            }
            continue
        entry = inspect_file(resolved)
        entry["pattern"] = os.path.join(ROOT, pattern)
        result[key] = entry

    json.dump(result, sys.stdout, separators=(",", ":"), default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

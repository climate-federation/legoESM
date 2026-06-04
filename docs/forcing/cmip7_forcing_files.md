# CMIP7 Forcing Files — `scripts/run/run_amip.py`

Scope: structural comparison between the forcing files under
`/work/bd1179/CMIP7_forcings_raw/` and the ingestion paths in
`src/legoesm/forcing/external.py` + `src/legoesm/forcing/amip.py`
(driven by `scripts/run/run_amip.py`).  Line numbers below reflect the
current `main` branch (post-`ap/fix-amip-external-forcing`); loaders
now parse `"months since"` time units (`external.py:93-105`) and do
non-cyclic linear-in-time interpolation for multi-year ozone files
(`external.py:785-790`).

## 1. Expected schema (from loader code)

Same loader API as the CMIP6 report — reproduced for reference so this
file is self-contained.

| Forcing File | Variables | Dimensions/Shape | Bands/Levels | Code Line/Notes |
|---|---|---|---|---|
| SST/SIC — `--forcing-path` (COBE preset) | `SST_cpl` [K], `ice_cov` [%], `time`, `lat`, `lon` | `(ntime, nlat, nlon)` regular lat-lon; regridded to `(ntime, 6, n, n)` CS or `(ntime, n_lat, n_lon)` Gaussian | Surface only | `amip.py:110-117` preset; `amip.py:249-446` loader (`load_amip_forcing`); `sic_scale=0.01` (% → fraction), `sst_offset=0.0` |
| SST/SIC — `--forcing-path` (HadISST preset) | `sst` [°C], `sic` [fraction], `time`, `lat`, `lon` | `(ntime, nlat, nlon)` | Surface only | `amip.py:118-125`; `sst_offset=+273.15`, `sic_scale=1.0` |
| SST/SIC — `--forcing-path` (custom) | `--sst-var`/`--sic-var`, `time`, `lat`, `lon` | `(ntime, nlat, nlon)` or `(nlat, nlon)` → broadcast to `(1, nlat, nlon)` | Surface only | `amip.py:339-363` |
| SST/SIC — `--forcing-path` (ICON unstructured) | `<sst_var>`, `<sic_var>`, `clon` [rad], `clat` [rad], `time` | `(ntime, ncell)` unstructured; KD-tree nearest-neighbour regridded | Surface only | `amip.py:133-246` |
| SIC split file — `--sic-path` | `<sic_var>` only | `(ntime, nlat, nlon)` or `(ntime, ncell)` | Surface only | `amip.py:288-302` |
| Ozone — `--ozone-file` (when `--ozone-forcing external`) | `ozone` / `vmro3` / `o3` / `O3` / `tro3` [mol/mol], `time`, `lat`, optional `plev`/`level`/`lev` | `(time, lat[, plev])`; CMIP `(time, plev, lat, lon)` → zonal-mean, `lat`/`plev` axis swap | `plev` (Pa; hPa auto-converted) | `external.py:231-299` `_load_monthly_zonal_with_levels`; `external.py:735-800` `get_ozone_at_time`; auto cyclic (`ntime=12`) vs linear-in-time (`ntime>12`, `external.py:785-790`) |
| GHG annual — `--ghg-file` (when `--ghg-forcing annual_file`) | `CO2`, `CH4`, `N2O`, `CFC_11`, `CFC_12` all in **one file**, `time` as `"year as %Y.%f"` | `(time, lat=1, lon=1)` → squeezed to `(time,)`; 5 series | Global-mean scalar | `external.py:525-564` `_load_ghg_annual_file`; `external.py:598-614` `annual_file` branch |
| GHG timeseries — `--ghg-file` (when `--ghg-forcing file`) | `co2_ppmv`, `ch4_ppbv`, `n2o_ppbv`, `time` | `(time,)` days | Global-mean scalar | `external.py:589-597` `file` branch |
| Solar TSI — `--solar-file` (when `--solar-source file`) | `tsi` (case-insensitive; CLI can override with `--solar-tsi-var`), `time` | `(time,)` | Broadband scalar | `external.py:1109-1113`; `_load_timeseries`; abs-day ref = 1850-01-01 |
| Solar spectral — `--solar-file` (when `--solar-source spectral_file`) | `tsi` (opt.), `solar_fraction_by_gpt` or `--solar-spectral-var`, `time` | `time (N,)`; spectral `(time, nspec)` where `nspec ∈ {14 SW bands, 112/224 g-points}` | 14 RRTMG-SW bands → 112 g-points via `bnd_limits_gpt` (when `nspec<50`) | `external.py:303-355` `_load_time_gpt`; `external.py:1115-1141`; `_expand_bands_to_gpoints` |
| Aerosol climatology — `--aerosol-file` (when `--aerosol-forcing external`) | **`aod`** [550 nm, dimensionless], `time`, `lat`/`latitude`, optional `lon`, optional band axis | `(time=12, lat)`; Kinne-style `(time, lnwl, lat, lon)` → zonal+band-averaged | Bands collapsed | `external.py:171-227` `_load_monthly_zonal`; `external.py:984-1010` |
| Volcanic aerosol — `--volcanic-aerosol-file` | CMIP6 `ext_sun`/`ext_earth` + `altitude` dim OR Kinne `aod` | Auto-dispatched; CMIP6: `(solar_bands, lat, altitude, month)` → vertically integrated, band-averaged `(time=12, lat)` | 14 SW or 16 LW bands (averaged) | `external.py:807-907`; `_load_volcanic_aerosol` auto-dispatch |

There is **no loader today** for: per-species GHG concentrations,
anthropogenic/biomass-burning emissions, CMIP7 UOEXETER 4D optical
properties, atmospheric δ¹³C/Δ¹⁴C, or LUH3 land-state forcings — these
categories exist only in CMIP7 and have no counterpart in CMIP6.

## 2. Actual file structures on HPC (`/work/bd1179/CMIP7_forcings_raw/`)

Representative file per category; all files share `input4MIPs` naming
conventions and `mip_era: CMIP7`.

### 2.1 Solar — `SOLARIS-HEPPA-CMIP-4-6`

| File | Dimensions | Key Variables | Bands/Levels | Time axis |
|---|---|---|---|---|
| `solar/multiple_..._gn_185001-202312.nc` (monthly) | `time=2088, wlen=3890` | `tsi(time)` [W m⁻² ], `ssi(time, wlen)` [W m⁻² nm⁻¹], `wlen(wlen)` [nm] 10.5…99975.0, `wlen_bnds(wlen, nbd)`, `f107`, `kp`, `ap`, `ssn`, `scnum`, `scph`, `calyear`/`calmonth`/`calday` | 3890 nm wavelength bins (UV → FIR, 0.01–100 µm) | `"days since 1850-01-01"`, `gregorian`, `frequency=mon` |
| `solar/multiple_..._gn_18500101-20231231.nc` (daily) | `time=63552, wlen=3890, glat=32, plev=61` | Same as above + `iprp`, `iprg`, `iprm` (ion-pair production rates, g⁻¹ s⁻¹), `lshell_bnds` | Geomagnetic-lat × pressure-level ion fluxes | `"days since 1850-01-01"`, `gregorian`, `frequency=day` |
| `solar/multiple_..._gn.nc` (climatology/fx) | `time=1, wlen=3890, glat=32, plev=61` | Same vars, single snapshot | Single timestep | `"days since 1850-01-01"`, `frequency=fx` |

### 2.2 GHG concentrations — `CR-CMIP-1-0-0` (one file per species)

| File pattern | Dimensions | Key Variables | Bands/Levels | Time axis |
|---|---|---|---|---|
| `GHG_concentrations/<sp>_..._gm_175001-202212.nc` | `time=3276` | `<sp>(time)` in species-specific units (`co2` [ppm], `ch4` [ppb], `n2o` [ppb], `cfc11` [ppt], `cfc12` [ppt], …) | Global-mean scalar | `"days since 1850-01-01"`, `proleptic_gregorian`, `frequency=mon` |
| `GHG_concentrations/<sp>_..._gnz_175001-202212.nc` | `time=3276, lat=12` | `<sp>(time, lat)`; latitude band midpoints −82.5 … 82.5° | Zonal bands | same |
| `GHG_concentrations/<sp>_..._gr1z_175001-202212.nc` | `time=3276, lat=2` | `<sp>(time, lat)`; hemispheric means (lat = ±45°) | Hemispheric | same |

47 species present: `co2, ch4, n2o, cfc11, cfc12, cfc113, cfc114, cfc115, cfc11eq, cfc12eq, hfc23, hfc32, hfc125, hfc134a, hfc134aeq, hfc143a, hfc152a, hfc227ea, hfc236fa, hfc245fa, hfc365mfc, hfc4310mee, halon1211, halon1301, halon2402, hcfc22, hcfc141b, hcfc142b, ccl4, ch2cl2, chcl3, ch3br, ch3ccl3, ch3cl, c2f6, c3f8, c4f10, c5f12, c6f14, c7f16, c8f18, cc4f8, cf4, nf3, sf6, so2f2`. Variable names are lowercase; the CMIP6 annual-mean file stored SCREAMING_CASE `CO2`/`CH4`/…

### 2.3 Ozone — `FZJ-CMIP-ozone-{1-2,2-0}`

| File | Dimensions | Key Variables | Bands/Levels | Time axis |
|---|---|---|---|---|
| `ozone/vmro3_..._gn_200001-202212.nc` (historical chunk) | `time=276, plev=66, lat=96, lon=144` | `vmro3(time, plev, lat, lon)` [mol mol⁻¹], `plev_bnds(plev, bnds)` | 66 pressure levels 1000 hPa → 1e-4 hPa | `"days since 1850-01-01"`, `noleap`, `frequency=mon` |
| `ozone/vmro3_..._gn_185001-185012-clim.nc` (climatology) | `time=12, plev=66, lat=96, lon=144` | Same plus `time_climatology(time, bnds)` | Same | `"days since 1850-01-01"`, `365_day`, `frequency=monC` |
| `ozone/zmta_..._gn_185001-202212.nc` (zonal-mean T_a) | `time=2076, plev=66, lat=96` | `zmta(time, plev, lat)` [K] | Same | `"days since 1850-01-01"`, `frequency=mon` |

Historical span is split across six chunk files: `182901-184912`, `185001-189912`, `190001-194912`, `195001-199912`, `200001-202212` (and earlier `1-2` version).  Units are `days since 1850-01-01`, so CMIP7 **no longer ships the `months since 1850-01-01` axis** that needed a fix in CMIP6.

### 2.4 SST & sea-ice — `PCMDI-AMIP-1-1-10`

| File | Dimensions | Key Variables | Notes | Time axis |
|---|---|---|---|---|
| `tos_..._gn_187001-202212.nc` | `time=1836, lat=180, lon=360` | `tos(time, lat, lon)` [**degC**] | 1° regular lat-lon, `lat=-89.5…89.5`, `lon=0.5…359.5` | `"days since 1870-01-01"`, `gregorian`, `frequency=mon` |
| `tosbcs_..._gn_187001-202212.nc` | `time=1836, lat=180, lon=360` | `tosbcs(time, lat, lon)` [degC], `cell_methods = time: point` | Taylor-smoothed boundary-condition SST (mid-month point sample) | same |
| `siconc_..._gn_187001-202212.nc` | `time=1836, lat=180, lon=360` | `siconc(time, lat, lon)` [**%**] | same grid | same |
| `siconcbcs_..._gn_187001-202212.nc` | `time=1836, lat=180, lon=360` | `siconcbcs(time, lat, lon)` [%], `cell_methods = time: point` | Taylor-smoothed BCs counterpart | same |
| `areacello_..._gn.nc` / `sftof_..._gn.nc` | `lat=180, lon=360` | `areacello` [m²], `sftof` [%] | Grid cell area and ocean fraction | static |

### 2.5 Aerosol properties — `UOEXETER-CMIP-2-2-1`

Seven optical/size properties per file; both time-series and 12-month
climatology variants.

| File | Dimensions | Variable | Units | Wavelength axis |
|---|---|---|---|---|
| `aerosol/ext_..._gnz_185001-202112-clim.nc` | `time=12, lat=36, height=70, wavelength=41` | `ext(time, lat, height, wavelength)` | `m-1` (volume extinction coefficient) | 1.6e-7 … 1e-4 m (41 bins, SW→LW) |
| `aerosol/asy_..._gnz_185001-202112-clim.nc` | same | `asy(time, lat, height, wavelength)` | — (asymmetry parameter) | same |
| `aerosol/ssa_..._gnz_185001-202112-clim.nc` | same | `ssa(time, lat, height, wavelength)` | — (single-scattering albedo) | same |
| `aerosol/nd_..._gnz_185001-202112-clim.nc` | `time=12, lat=36, height=70` | `nd(time, lat, height)` | `cm-3` (H₂SO₄ number density) | — |
| `aerosol/reff_..._gnz_185001-202112-clim.nc` | same | `reff(time, lat, height)` | effective radius | — |
| `aerosol/sad_..._gnz_185001-202112-clim.nc` | same | `sad(time, lat, height)` | surface area density | — |
| `aerosol/vd_..._gnz_185001-202112-clim.nc` | same | `vd(time, lat, height)` | volume density | — |

Height grid: 70 layers, 5000 m → 39 500 m above sea level (same altitude set as CMIP6 volcanic file — hence this **is** the CMIP7 stratospheric aerosol product; full-time-series files (`...175001-202312.nc`) also exist at ~2.7 GB each for `ext/asy/ssa`.

### 2.6 Emissions — CEDS (anthropogenic) + DRES BB4CMIP7 (biomass burning)

| File pattern | Dimensions | Variable | Units | Notes |
|---|---|---|---|---|
| `emissions/<SP>-em-anthro_..._CEDS-..._gn_YYYY.nc` | `time, sector=8, lat=360, lon=720` | `<SP>_em_anthro(time, sector, lat, lon)` | `kg m-2 s-1` | 0.5° grid, sector axis 0…7 (AGR/ENE/IND/TRA/RCO/SLV/WST/SHP); `"days since 1750-01-01"` calendar `365_day`; chunked 50-yr or 1-yr `gr` files |
| `emissions/<SP>-em-AIR-anthro_..._gn_YYYY.nc` | `time, level=25, lat, lon` | `<SP>_em_AIR_anthro(time, level, lat, lon)` | `kg m-2 s-1` | `level(level)` km, 0.305 … 14.945 km; aircraft emissions |
| `emissions/<SP>-em-SOLID-BIOFUEL-anthro_..._supplemental_gn_YYYY.nc` | `time, sector=8, lat, lon` | `<SP>_em_SOLID_BIOFUEL_anthro` | `kg m-2 s-1` | Supplemental residential solid-biofuel product |
| `emissions/<SP>_..._DRES-CMIP-BB4CMIP7-2-1_gn_YYYY.nc` | `time, latitude=720, longitude=1440` | `<SP>(time, latitude, longitude)` | `kg m-2 s-1` | 0.25° grid (note: `latitude`/`longitude` — not `lat`/`lon`); `"days since 1900-01-01"` `noleap` |
| `emissions/<SP>percentage<TYPE>_..._DRES-..._gn_YYYY.nc` | `time, latitude, longitude` | `<SP>percentage<TYPE>(...)` | `%` | Per-fire-type attribution of each burning species (AGRI/BORF/DEFO/PEAT/SAVA/TEMF) |
| `emissions/areacella_..._CEDS-..._gn.nc` | `lat=360, lon=720` | `areacella(lat, lon)` | m² | CEDS grid cell area (static) |
| `emissions/gridcellarea_..._DRES-..._gn.nc` | `latitude=720, longitude=1440` | `gridcellarea(latitude, longitude)` | m² | BB4CMIP7 grid cell area (static) |

Species set (CEDS-anthro, incomplete list): `BC, CH4, CO, CO2, NH3, NMVOC, NOx, OC, SO2` plus AIR and SOLID-BIOFUEL variants.
Species set (BB4CMIP7): `BC, OC, SO2, NH3, NMVOC, NOx, CO, CO2, CH4, C10H16, C2H2, C2H4, C2H4O, C2H5OH, C2H6, C2H6S, C3H6, C3H6O, C3H8, C5H8, C6H6, C7H8, …` (speciated VOCs).

### 2.7 Atmospheric state — `ImperialCollege-3-0` (C4MIP)

| File | Dimensions | Variable | Units | Time axis |
|---|---|---|---|---|
| `atmospheric_state/delta13co2_..._gm_1700-2023.nc` | `time=324` | `delta13co2(time)` | `1` (‰ after multiplying by 1000) | `"days since 1700-01-01"`, `365_day`, `frequency=yr` |
| `atmospheric_state/Delta14co2_..._gz_1700-2023.nc` | `time=324, lat=4` | `Delta14co2(time, lat)` | `1` (‰) | same; `lat = -60, -20, 20, 60` (zonal bands) |

### 2.8 Land state — `UofMD-landState-3-1-{1,2}` (LUH3)

| File | Dimensions | Variables (selected) | Units | Time axis |
|---|---|---|---|---|
| `land_state/multiple-states_..._gn_0850-2024.nc` | `time=1175, lat=720, lon=1440` | `primf, primn, secdf, secdn, urban, c3ann, c4ann, c3per, c4per, c3nfx, pastr, range, secmb, secma, pltns` | `1` (grid fraction) or `kg m-2` or `years` | `"days since 0850-01-01"`, `noleap`, `frequency=yr` |
| `land_state/multiple-management_..._gn_0850-2024.nc` | same | `addtc, combf, cpbf1_c3ann, …` (many management layers) | various | same |
| `land_state/multiple-transitions_..._gn_0850-2023.nc` | `time=1174, lat=720, lon=1440` | `primf_to_secdn, primf_to_urban, …` | `1 yr-1` | same |
| `land_state/multiple-static_..._gn.nc` | `lat=720, lon=1440` | `ptbio, fstnf, carea, icwtr, ccode` | `kg m-2` / `1` / `km2` / `1` / `1` | static |

## 3. Compatibility assessment vs. `run_amip.py`

### ✅ Correct or recoverable matches

- **Ozone, CMIP7 historical (`vmro3_..._gn_YYYY.nc`) → `--ozone-forcing external --ozone-file ...`**
  - `vmro3` is in `_OZONE_VARNAMES` (`external.py:714`), auto-detected.
  - Time axis `days since 1850-01-01` with calendar `noleap` — `_to_days_float` handles numeric + `days since` trivially (`external.py:92-105`), so there is no `months since` parsing pitfall like the CMIP6 file had.
  - Shape `(time, plev, lat, lon)` with `plev` in hPa is reduced zonally and axis-swapped to `(time, lat, plev)` by `_load_monthly_zonal_with_levels` (`external.py:278-296`); hPa → Pa via the `units="hPa"` branch (`external.py:270-271`).
  - `ntime=276 > 12` → `_interp_time_linear` preserves interannual evolution (`external.py:787-790`).
  - **One practical limitation**: the ingestion opens a single path. The historical span is sharded across 5 or 6 50-year chunks (`185001-189912`, …). Running `> 2000` needs either `xarray.open_mfdataset` pre-concat or concatenating the chunk that covers the simulation window; the loader does not glob.

- **Ozone, CMIP7 climatology (`vmro3_..._gn_185001-185012-clim.nc`) → same flags**
  - `ntime=12` → `_interp_monthly_cyclic` branch (`external.py:786`). Works without change.

- **Solar CMIP7 TSI-only (`solar/multiple_..._gn_185001-202312.nc`) → `--solar-source file --solar-file ... --solar-tsi-var tsi`**
  - Lowercase `tsi(time)` is the default (`SolarConfig.tsi_var="tsi"`, `external.py:1085`), so no CLI override is needed — cleaner than CMIP6 which required `--solar-tsi-var TSI`.
  - Time units `days since 1850-01-01` match `_REF_YEAR=1850` exactly (`external.py:1106`).

- **SST/SIC CMIP7 (`tos`, `siconc`) → `--forcing-path <tos-file> --sic-path <siconc-file>` with custom vars**
  - Requires explicit CLI flags because neither the COBE nor HadISST preset matches:
    - `--sst-var tos --sst-offset 273.15` (file is °C)
    - `--sic-var siconc --sic-scale 0.01` (file is %)
    - `--sic-path ...siconc_...nc` (SIC is in a separate file)
  - Shape `(1836, 180, 360)` with 1° regular grid and `lat=-89.5…89.5`, `lon=0.5…359.5` is consumed by `load_amip_forcing` → `RegularGridInterpolator` (`amip.py:417-427`).
  - Time axis `days since 1870-01-01`, calendar `gregorian`, numeric float64 — handled by `_time_coord_to_days` numeric branch (`amip.py:164-172`).
  - Consider using `tosbcs`/`siconcbcs` instead: these are the Taylor-smoothed mid-month boundary-condition fields (`cell_methods = time: point`) and are the recommended AMIP inputs per PCMDI AMIP spec (the non-`bcs` variants are monthly means that can under-represent the seasonal cycle when sampled daily).

### ⚠️ Mismatches / bugs

- **GHG — CMIP7 drops the combined annual-mean file used by the loader**
  - `--ghg-forcing annual_file` expects one NetCDF with **all five** variables `CO2, CH4, N2O, CFC_11, CFC_12` (`external.py:547`) plus time in `"year as %Y.%f"` (`external.py:531-532`).
  - CMIP7 ships **one file per species**, lowercase variable name matching the filename (`co2`, `ch4`, …), `time` in `"days since 1850-01-01"`, monthly. The underscore form `CFC_11` does not exist — it is `cfc11`/`cfc12`.
  - `--ghg-forcing file` also fails: it expects `co2_ppmv, ch4_ppbv, n2o_ppbv` **in one file** as 1-D `(time,)` arrays (`external.py:592`). Split files + lowercase names without unit suffixes break this too.
  - **Fix needed**: new `source="annual_file_cmip7"` (or similar) that opens *N* paths (one per species) or a single pre-concatenated file, with explicit `co2/ch4/n2o/cfc11/cfc12` variable names and `days since <year>` time parsing. Unit scaling also changes: CMIP6 stored mole fractions with native scale factors (1e-6/1e-9/1e-12); CMIP7 stores already-scaled values in `ppm`/`ppb`/`ppt` → drop the per-species multiplier in `get_ghg_at_time` for this path.
  - Note: the `gm` files are global-monthly-mean; for zonal GHG variations use the `gnz` (12 zonal bands) or `gr1z` (2 hemispheres) variants — neither shape matches the current 1-D `(time,)` expectation.

- **Aerosol — CMIP7 has no `aod` variable**
  - `_load_monthly_zonal(path, "aod")` is hard-coded in `get_aerosol_at_time` (`external.py:985`). CMIP7 UOEXETER files carry `ext` / `asy` / `ssa` / `nd` / `reff` / `sad` / `vd` — *not* AOD.
  - Shape is `(time=12, lat=36, height=70, wavelength=41)`. `ext` is a 4-D **volume extinction coefficient** in m⁻¹. To produce a column AOD one must integrate over `height` and pick (or interpolate to) 550 nm from the 41-bin wavelength axis.
  - Fire-path: first `ValueError: Variable 'aod' not found in <file>` at `external.py:187`.
  - **Fix needed**: auto-detect CMIP7 aerosol files by presence of `ext` + `height` + `wavelength` dims (analogous to `_is_cmip6_volcanic_file`), then `(a)` vertically integrate `ext` × `Δz`, `(b)` select the 550 nm bin from `wavelength`, `(c)` return `(mid_days, lat, aod_column)`. Alternatively (preferred for RRTMG-SW) keep `wavelength` as an axis, remap to 14 RRTMG-SW band weights, and thread the band dimension through the aerosol forcing path the same way the spectral-band-aware Kinne plan would.

- **Volcanic — CMIP7 has no separate volcanic file**
  - CMIP6 shipped `bc_aeropt_cmip6_volc_lw_b16_sw_b14_*.nc` with `ext_sun`/`ext_earth` + `altitude` — detected by `_is_cmip6_volcanic_file` (`external.py:807-819`).
  - CMIP7 folds stratospheric aerosol into the UOEXETER aerosolProperties product (same `ext`/`height`/`wavelength` layout as tropospheric aerosol). Neither `ext_sun` nor `altitude` are present.
  - `_is_cmip6_volcanic_file` returns `False` → dispatch falls through to `_load_monthly_zonal(path, "aod")` → same "no `aod` variable" failure as the main aerosol file.
  - **Fix needed**: extend the CMIP6 volcanic loader (or add a sibling) to recognise `ext` + `height` + `wavelength` and select the 550 nm (or RRTMG-SW) band before vertical integration. The 70-layer `height` axis in CMIP7 (5 → 39.5 km above sea level) matches the CMIP6 altitude grid exactly — algorithmically the vertical integration step is identical.

- **Solar spectral — CMIP7 stores SSI on a nm-wavelength axis, not on RRTMG bands**
  - `spectral_file` path expects `(time, nspec)` with `nspec ∈ {14 RRTMG-SW bands, 112 g-points, 224 g-points}` (`external.py:1129-1133`).
  - CMIP7 provides `ssi(time, wlen)` with `wlen=3890` bins from 10.5 nm to 99 975 nm in `W m⁻² nm⁻¹`, i.e. a fine-resolution spectrum covering UV → far-IR, *not* a per-band solar fraction.
  - Passing this file to `--solar-source spectral_file --solar-spectral-var ssi` will:
    1. Pass the `spec.ndim==2` and `spec.shape[0]==ntime` checks (`external.py:336-341`), then
    2. Hit `spec_series.shape[1]=3890`, which is ≥ 50, so the band-expansion branch is skipped,
    3. Try to feed a 3890-long per-g-point fraction into the RRTMG-SW solver that expects 112 g-points — shape mismatch downstream.
  - **Fix needed**: new CMIP7 path that integrates `ssi(time, wlen)` × `wlenbinsize` over each of the 14 RRTMG-SW band intervals (available in the lookup table used by `_expand_bands_to_gpoints`), yielding `(time, 14)` band-integrated power. Divide by total TSI for per-band fractions, then expand to g-points via the existing `_expand_bands_to_gpoints`. The file carries `tsi(time)` directly, so the overall scale is unambiguous.

- **Multi-chunk concatenation**
  - Most CMIP7 categories (ozone, emissions, land state) are physically split into 50-year chunks on disk.  Every loader in `external.py` / `amip.py` opens exactly one file via `xarray.open_dataset` / `_open_forcing_dataset` — no `open_mfdataset`, no globbing.
  - **Operational workaround**: pre-concatenate the chunks covering the simulation window with `cdo cat` / `ncrcat` / `xarray.open_mfdataset` before passing `--ozone-file`, `--aerosol-file`, etc. This was acceptable for CMIP6 (one historical file) but becomes a per-simulation step in CMIP7.

### ℹ️ Categories with no loader today

CMIP7 introduces several categories that have no corresponding `ExternalForcingConfig` sub-config in `external.py`:

- **`emissions/` (CEDS + BB4CMIP7)** — 0.25°/0.5° 3-D `(time, lat, lon)` fluxes in `kg m⁻² s⁻¹`, with an extra `sector` axis for anthropogenic sources and a `level` axis for aircraft. legoESM does not run interactive chemistry or online aerosol, so these are currently out of scope. If a chemistry coupling is added later they become the natural emission boundary condition.
- **`atmospheric_state/` (ImperialCollege C4MIP)** — δ¹³C and Δ¹⁴C of atmospheric CO₂, yearly. Only relevant for isotope-enabled carbon-cycle runs.
- **`land_state/` (UofMD LUH3)** — yearly 0.25° land-use states, management layers, and transitions; multi-GB per variable. A dynamic-vegetation or land-use-change driver would consume these; the current AMIP run path does not.
- **`emissions/areacella_*.nc` / `emissions/gridcellarea_*.nc`** and **`sst_and_seaice/areacello_*.nc` / `sftof_*.nc`** — grid cell area and ocean fraction masks that are only relevant for flux conversions. Not currently read by any loader.

## 4. Required CLI invocation for CMIP7 (today)

Given the compatibility state above, only these flags point at real CMIP7 files without code changes:

```bash
scripts/run/run_amip.py \
  --forcing-path  /work/bd1179/CMIP7_forcings_raw/sst_and_seaice/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc \
  --sic-path      /work/bd1179/CMIP7_forcings_raw/sst_and_seaice/siconc_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc \
  --sst-var       tos --sst-offset 273.15 \
  --sic-var       siconc --sic-scale 0.01 \
  --ozone-forcing external \
  --ozone-file    /work/bd1179/CMIP7_forcings_raw/ozone/<pre-concatenated-vmro3-covering-run-window>.nc \
  --solar-source  file \
  --solar-file    /work/bd1179/CMIP7_forcings_raw/solar/multiple_input4MIPs_solar_CMIP_SOLARIS-HEPPA-CMIP-4-6_gn_185001-202312.nc \
  --solar-tsi-var tsi
```

Dropped relative to CMIP6 until loaders are updated:
- `--ghg-forcing annual_file --ghg-file …` (use `--ghg-forcing constant` with per-period fixed concentrations in the meantime).
- `--aerosol-forcing external --aerosol-file …` (use `--aerosol-forcing constant` / `use_reference_if_missing=True`).
- `--solar-source spectral_file` (use broadband `file` only; spectral requires a new band-integration loader).
- `--volcanic-aerosol-file …` (none).

## 5. Recommended code changes (in order of priority)

1. **`external.py` — CMIP7 GHG loader**
   Add `source="per_species_files"` (or similar) to `GHGConfig` that accepts a mapping `{co2: path, ch4: path, …}` (or a directory + glob) and reads lowercase per-species variables with `days since …` monthly time axes.  Keep the existing `annual_file` path for CMIP6.
2. **`external.py` — CMIP7 aerosol loader**
   Auto-detect `ext + height + wavelength` dims (as `_is_cmip6_volcanic_file` does for `ext_sun + altitude`), integrate over height, pick the 550 nm bin from `wavelength`, return `(mid_days, lat, aod_column)` so the rest of `get_aerosol_at_time` is unchanged.  Optional band-aware variant: keep `wavelength` → remap to RRTMG-SW 14 bands and thread through as a band axis.
3. **`external.py` — CMIP7 volcanic support**
   Same detection as aerosol; `altitude` is replaced by `height` (both in m or km). Integration logic is otherwise identical to `_load_volcanic_cmip6` (`external.py:822-900`).
4. **`external.py` — CMIP7 spectral solar loader**
   Integrate `ssi(time, wlen)` × `wlenbinsize` over RRTMG-SW band wavelength intervals → `(time, 14)` per-band power; optionally normalize by `tsi(time)` and feed the existing `_expand_bands_to_gpoints` path.
5. **Multi-file concatenation**
   Add an `open_forcing_dataset_mf` helper that accepts a list of paths or a glob and concatenates along `time` — used by ozone, per-species GHG, and large emission products.
6. **`amip.py` — CMIP7 SST/SIC preset**
   `get_amip_preset("cmip7")` returning `sst_var="tos"`, `sic_var="siconc"`, `sst_offset=273.15`, `sic_scale=0.01`, and recommending the `_bcs` variants when available. Avoids the 4-flag incantation above.
7. **Documentation in `scripts/run/run_amip.py` help strings**
   Point at the CMIP7 layout and warn that `--ghg-forcing external` and `--aerosol-forcing external` are not wired for CMIP7 yet.

## 6. Glossary additions (CMIP7-specific)

- **`gm` / `gnz` / `gr1z` grid labels** — `input4MIPs` CMOR grid mnemonics.
  - `gm`: global-mean (1-D time series).
  - `gnz`: native zonal (non-regridded latitude bands; CMIP7 CR GHGs use 12 bands at ±82.5° …).
  - `gr1z`: regridded to 1-D zonal mean with 2 cells (hemispheric mean) — used as a compact summary.
  - `gn`: native grid (0.5°, 0.25°, 1°, or model native depending on product).
- **`frequency=monC`** — monthly climatology (12-record cyclic), as opposed to `mon` (monthly means, one record per calendar month).
- **`PCMDI-AMIP-1-1-10`** — PCMDI AMIP reference SST/SIC. CMIP7 version renames variables from `SST_cpl`/`ice_cov` (CMIP6 ICAR-derived COBE-SST2) to CMOR-compliant `tos`/`siconc` (`tos` = "Temperature of Ocean Surface", °C; `siconc` = "Sea-Ice Concentration", %). The `_bcs` suffix marks the Taylor et al. (2000) mid-month boundary-condition variant.
- **LUH3** (`UofMD-landState-3-1-*`) — successor to LUH2 used in CMIP6; re-gridded to 0.25° and extended to 850–2024. States / management / transitions split across three files keeps each under ~10 GB.
- **BB4CMIP7** (`DRES-CMIP-BB4CMIP7-2-1`) — CMIP7 biomass-burning emissions at 0.25° with per-fire-type attribution (AGRI/BORF/DEFO/PEAT/SAVA/TEMF) carried as separate `…percentage<TYPE>` variables.
- **CEDS-2025-04-18** — CMIP7 anthropogenic emission inventory at 0.5°, 8-sector `(AGR, ENE, IND, TRA, RCO, SLV, WST, SHP)`.

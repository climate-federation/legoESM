# AMIP Forcing Files — `scripts/run/run_amip.py`

## 1. Expected schema (from loader code)

| Forcing File | Variables | Dimensions/Shape | Bands/Levels | Code Line/Notes |
|---|---|---|---|---|
| SST/SIC — `--forcing-path` (COBE preset) | `SST_cpl` [K], `ice_cov` [%], `time`, `lat`, `lon` | `(ntime, nlat, nlon)` regular lat-lon; regridded to `(ntime, 6, n, n)` CS or `(ntime, n_lat, n_lon)` Gaussian | Surface only (no vertical, no bands) | `amip.py:111-118` preset; `amip.py:287-391` loader (`load_amip_forcing`); `sic_scale=0.01` (% → fraction), `sst_offset=0.0` |
| SST/SIC — `--forcing-path` (HadISST preset) | `sst` [°C], `sic` [fraction], `time`, `lat`, `lon` | `(ntime, nlat, nlon)` | Surface only | `amip.py:119-126`; `sst_offset=+273.15` (°C → K), `sic_scale=1.0` |
| SST/SIC — `--forcing-path` (custom) | `--sst-var`/`--sic-var` (user-set), `time`, `lat`, `lon` | `(ntime, nlat, nlon)` or `(nlat, nlon)` → broadcast to `(1, nlat, nlon)` | Surface only | `model_driver.py:340-347`; `amip.py:299-311` |
| SST/SIC — `--forcing-path` (ICON unstructured) | `<sst_var>`, `<sic_var>`, `clon` [rad], `clat` [rad], `time` | `(ntime, ncell)` unstructured; KD-tree nearest-neighbour regridded to target | Surface only | `amip.py:134-220` `_load_icon_unstructured`; detected by presence of `cell` dim + `clon`/`clat` |
| SIC split file — `--sic-path` (ICON) | `<sic_var>` only | `(ntime, ncell)` | Surface only | `amip.py:163-169`; opened separately when `sic_path` non-empty |
| Ozone — `--ozone-file` (when `--ozone-forcing external`) | auto-detected: `ozone` / `vmro3` / `o3` / `O3` / `tro3` [mol/mol], `time`, `lat`, optional `plev`/`level`/`lev` | `(time=12, lat[, plev])`; CMIP6 `(time, plev, lat)` auto-swapped → `(time, lat, plev)` | Vertical levels: `nlev_src` from `plev` (Pa; hPa auto-converted if `units ∈ {hPa, mb,…}` or max<1500) | `external.py:637-707` `get_ozone_at_time`; `_detect_ozone_varname`, `_load_monthly_zonal_with_levels`; cyclic monthly interp 365.0 d (noleap model clock; 2026-07-21) |
| GHG annual — `--ghg-file` (when `--ghg-forcing external`) | `CO2` [1e-6], `CH4` [1e-9], `N2O` [1e-9], `CFC_11` [1e-12], `CFC_12` [1e-12], `time` (fractional year, "year as %Y.%f") | `(time, lat=1, lon=1)` → squeezed to `(time,)`; 5 separate 1-D series | Global-mean scalar; no bands/levels | `external.py:446-485` `_load_ghg_annual_file`; `external.py:519-535` `get_ghg_at_time` annual_file branch; day→year via `start_year + day/365.0` (noleap model clock; 2026-07-21) |
| Solar TSI — `--solar-file` (when `--solar-source file`) | `tsi` (or user `--solar-tsi-var`, e.g. `TSI`; case-insensitive lookup), `time` (days) | `(time,)` 1-D | Broadband scalar; no bands | `external.py:891-913`; `_load_timeseries`; abs-day ref = 1850-01-01 (`abs_day = (start_year−1850)*365.25 + day`) |
| Solar spectral — `--solar-file` (when `--solar-source spectral_file`) | `tsi` (optional), `solar_fraction_by_gpt` (or `--solar-spectral-var`), `time` | `time` `(N,)`; spectral `(time, nspec)` where `nspec ∈ {14 bands, 112/224 gpts}` | Bands: 14 RRTMG-SW → expanded to 112 g-points via `bnd_limits_gpt` when `nspec<50` | `external.py:265-305` `_load_time_gpt`; `external.py:915-941`; `_expand_bands_to_gpoints` uses `_DEFAULT_SW_GAS` LUT |
| Aerosol climatology — `--aerosol-file` (when `--aerosol-forcing external`) | `aod` [dimensionless, 550 nm], `time`, `lat`/`latitude`, optional `lon`, optional band axis (e.g. `lnwl`) | `(time=12, lat)`; Kinne-style `(time, lnwl, lat, lon)` → zonal+band-averaged/summed to `(time, lat)` | Bands collapsed (summed to total column AOD after zonal mean); no vertical | `external.py:142-196` `_load_monthly_zonal`; `external.py:765-811`; trailing dims summed post-interp (`issue #178`) |
| Volcanic aerosol — `--volcanic-aerosol-file` | `aod`, `time`, `lat` | `(time, lat)` (same schema as aerosol) | Same as aerosol (multi-dim trailing summed) | `external.py:813-844`; scaled by `--volcanic-aerosol-scale`, added to baseline AOD, clipped ≥0 |

## 2. Actual file structures on HPC (MPI-M `/pool/data/ICON/grids/public/mpim`)

| File | Dimensions | Key Variables | Bands/Levels | Time axis |
|---|---|---|---|---|
| `common/solar_radiation/swflux_14band_cmip6_1850-2299-v3.2.nc` | `time=5400, numwl=14` | `TSI` (time,) [W/m²], `SSI` (time, numwl) [W/m²], `SSI_frac` (time, numwl), `year`, `month`, `lb`/`ub` (numwl) [nm] | 14 RRTMG-SW bands | `"days since 1850-01-01 00:00:00"`, float64 |
| `independent/greenhouse_gases/greenhouse_historical_plus.nc` | `time=2021, lat=1, lon=1` | `CO2` [1e-6], `CH4` [1e-9], `N2O` [1e-9], `CFC_11` [1e-12], `CFC_12` [1e-12] | Global scalar | `"year as %Y.%f"`, float64 |
| `common/aerosol_kinne/aeropt_kinne_sw_b14_fin_*_rast.nc` (fine mode, SW) | `time=12, lnwl=14, lat=180, lon=360, lev=40` | `aod/asy/ssa` (time, lnwl, lat, lon) [f32], `z_aer_fine_mo` (time, lev, lat, lon), `fp1`/`fp4` (time, lat, lon), `zbot`/`ztop` (lev, lat, lon), `asl` (lat, lon) | 14 SW bands, 40 altitude layers | `"days since 2000-01-01 00:00"`, monthly mid-days 14…349 |
| `common/aerosol_kinne/aeropt_kinne_sw_b14_coa_rast.nc` (coarse, SW) | `time=12, lnwl=14, lat=180, lon=360, lev=40` | `aod/asy/ssa`, `z_aer_coarse_mo`, `zbot`/`ztop`, `asl` | 14 SW bands, 40 layers | Same as above |
| `common/aerosol_kinne/aeropt_kinne_lw_b16_coa_rast.nc` (coarse, LW) | `time=12, lnwl=16, lat=180, lon=360, lev=40` | `aod/asy/ssa`, `z_aer_coarse_mo`, `zbot`/`ztop`, `asl` | **16 LW bands** (matches RRTMG-LW), 40 layers | Same |
| `common/aerosol_volcanic_cmip6/bc_aeropt_cmip6_volc_lw_b16_sw_b14_*.nc` | `solar_bands=14, terrestrial_bands=16, latitude=36, altitude=70, month=12` | `ext_sun`/`omega_sun`/`g_sun` (solar_bands, latitude, altitude, month) [1/km]; `ext_earth`/`omega_earth`/`g_earth` (terrestrial_bands, latitude, altitude, month); `wl1_sun`/`wl2_sun`, `wl1_earth`/`wl2_earth` | 14 SW + 16 LW bands, **70 altitude levels (5 → 39.5 km)** | Dim `month=12`; altitude in km; per-year file |
| `common/ozone_cmip6_forcing/historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_*.nc` | `time=600, plev=66, lat=96, lon=144, bound=2` | `vmro3` (time, plev, lat, lon) [mol mol⁻¹]; `bounds_plev/bounds_lat/bounds_lon` | 66 pressure levels (1000 hPa → 1e-4 hPa) | `"months since 1850-01-01 00:00"`, 1850-01 → 1899-12 |

## 3. Compatibility assessment vs. `run_amip.py`

### ✅ Correct matches

- **`greenhouse_historical_plus.nc` → `--ghg-forcing external --ghg-file ...`**
  - Exact match for `_load_ghg_annual_file` (`external.py:446-485`). Time units `"year as %Y.%f"` and `(time, lat=1, lon=1)` layout are explicitly supported. Unit scales (1e-6 / 1e-9 / 1e-12) are consumed correctly by `get_ghg_at_time` (`external.py:519-535`).

- **`swflux_14band_cmip6_*.nc` → `--solar-source spectral_file`** (with correct var flags)
  - `time=5400` units `"days since 1850-01-01"` match `_REF_YEAR=1850` in `get_solar_forcing_at_time` (`external.py:906-907`).
  - Spectral dim `numwl=14 < 50` → auto-expansion via `_expand_bands_to_gpoints` → 112 g-points (`external.py:929-933`). ✓
  - **Required flags**:
    - `--solar-tsi-var TSI` (uppercase — see caveat below)
    - `--solar-spectral-var SSI_frac` (default is `solar_fraction_by_gpt`, not present in file)

- **Kinne `aeropt_kinne_sw_b14_*_rast.nc` / `lw_b16_coa_rast.nc` → `--aerosol-file`**
  - `aod` variable present with `(time=12, lnwl, lat=180, lon=360)` layout.
  - `_load_monthly_zonal` averages over `lon` then collapses `lnwl` → `(time=12, lat=180)` (`external.py:177-196`). ✓
  - No `time.units` attribute, but numeric mid-days `14…349` go through `_to_days_float` → cyclic interp `day % 365.0` works correctly.

### ⚠️ Mismatches / bugs

- **Ozone time axis — unit string not parsed** (`external.py:48-78`, `_to_days_float`)
  - File units: `"months since 1850-01-01"`, values `0, 1, 2, …, 599`.
  - `_to_days_float` only handles `datetime64` / `cftime.datetime` / numeric — on a numeric array it just casts `.astype(np.float64)`, **ignoring the unit string**. So `month=1` is treated as `day=1`.
  - `_interp_monthly_cyclic` then does `day_mod = day % 365.0` on this mis-scaled axis and against mid-days in `[0, 599]` → wrong month selected for every query.
  - Additional mismatch: the loader assumes 12-month climatology; this file has 600 months of time-varying ozone. Cyclic interpolation discards all interannual evolution.
  - **Fix needed**: detect `"months since"` in `units` attr and convert (e.g., × 30.4375) inside `_to_days_float`; or collapse the file to a climatology before passing to `--ozone-file`; or add a non-cyclic time-varying ozone code path.

- **Volcanic aerosol — loader will crash** (`external.py:156-158`, `_load_monthly_zonal`)
  - File has **no `aod` variable**. It stores per-band extinction `ext_sun` / `ext_earth` in `[1/km]`, distributed over 70 altitude levels.
  - `_load_monthly_zonal(path, "aod")` raises `ValueError: Variable 'aod' not found`.
  - Dim name is `month` (not `time`), lat is `latitude` — `_load_monthly_zonal` handles `latitude` but falls back to synthetic monthly days for `time`, so dim-name aliasing alone would not fail; the missing `aod` var is the hard blocker.
  - Physical mismatch: `ext_sun` is not a column AOD — it must be vertically integrated `∫ ext · dz` over the `altitude` axis and a single 550 nm band selected (or each of the 14 SW / 16 LW bands carried independently).
  - **Fix needed**: dedicated volcanic loader that (a) reads `ext_sun`/`ext_earth`, (b) multiplies by layer thickness (from `altitude` coord, km → m), (c) picks or averages over bands, (d) does lat interp on `latitude=36`.

- **`_load_time_gpt` TSI-var lookup is case-sensitive** (`external.py:297`)
  - Uses `if tsi_var in ds.data_vars` (strict match), unlike `_load_timeseries` which builds a lowercase `varname_map` (`external.py:105-108`).
  - The MPI-M solar file uses `TSI` (uppercase); without `--solar-tsi-var TSI` on the CLI the spectral_file path silently falls back to `config.S_0` (loses time-varying TSI).
  - **Fix needed**: mirror the case-insensitive lookup from `_load_timeseries` in `_load_time_gpt`.

### ℹ️ Observations (not bugs)

- Kinne files carry far more information than the loader uses: per-band `aod/asy/ssa`, vertical distribution `z_aer_{fine,coarse}_mo` (time, 40, 180, 360), fine vs. coarse mode separation, and Twomey indirect-effect factors `fp1`/`fp4`. The current `run_amip.py` path only consumes column-integrated, band-averaged `aod` — a significant simplification relative to a full Kinne implementation.
- SW (`lnwl=14`) and LW (`lnwl=16`) Kinne files are treated identically by `_load_monthly_zonal` (all non-(time, lat) axes collapsed). There is no SW/LW-aware path in `run_amip.py`.
- The solar file also stores `year` and `month` as integer variables, plus band edges `lb`/`ub` in nm — useful for verification but not consumed by the loader.

## 4. Glossary — SW/LW bands and vertical layers

### RRTMG spectral bands

These are the spectral discretization used by **RRTMG** (Rapid Radiative Transfer Model for GCMs, Iacono et al. 2008), the broadband radiation scheme that legoESM's `rrtmg` path targets. RRTMG splits the electromagnetic spectrum into a fixed set of bands; within each band the gaseous absorption is further resolved by *g-points* using the correlated-k method.

**RRTMG-SW: 14 shortwave bands** — covers the *incoming* solar spectrum.
- Range: ~0.2–12 µm (820–50 000 cm⁻¹), spanning UV → visible → near-IR.
- 14 bands, numbered 16–29 in AER's canonical numbering.
- Each band holds 8–16 g-points → **112 g-points total**. `_expand_bands_to_gpoints` (`external.py:946-976`) turns 14 per-band fractions into 112 per-g-point weights by repeating each band's value over its g-point range, using `bnd_limits_gpt` from the RRTMG-SW LUT.
- Band edges isolate specific absorber features (e.g. O₃ Hartley/Huggins UV, H₂O windows, near-IR CO₂).

**RRTMG-LW: 16 longwave bands** — covers the *outgoing* thermal infrared spectrum.
- Range: ~3.1–1000 µm (10–3250 cm⁻¹).
- 16 bands, numbered 1–16; 140 g-points total (standard config).
- Band edges isolate major LW absorbers: H₂O rotation, CO₂ 15 µm (key band for CO₂ forcing), O₃ 9.6 µm, CH₄/N₂O bands, 8–12 µm atmospheric window.

### "Matches RRTMG-LW" / "matches RRTMG-SW"

A compatibility statement about file layout:

- `aeropt_kinne_lw_b16_coa_rast.nc` stores `aod/asy/ssa` on a `lnwl=16` axis.
- Those 16 slots are aligned **1-to-1 with RRTMG-LW's 16 bands**, in the same order and with the same wavelength edges. Band `lnwl=3` in the file *is* RRTMG-LW band 3.
- Consequence: the radiation scheme can feed `aod[t, k, :, :]` for band *k* directly into the RRTMG-LW solver — no spectral remapping or averaging needed.
- Same logic applies to SW Kinne files (`lnwl=14` ↔ RRTMG-SW 14 bands).
- The CMIP6 volcanic file uses `solar_bands=14` + `terrestrial_bands=16` — same band counts, different (but also RRTMG-compatible) naming.

### G-points (correlated-k quadrature points)

**G-points** (also written "g-points" or "gpt") are the fundamental computational unit of the **correlated-k distribution (CKD) method**, the technique RRTMG uses to replace brute-force line-by-line radiative transfer with something a GCM can afford.

**tl;dr**: g-points are *"Gauss quadrature points in sorted-absorption-coefficient space"*. One g-point = one `(k, weight)` pair that represents some fraction of a band's spectral interval, chosen so that fluxes come out right despite the spiky underlying absorption.

**In RRTMG numbers**:

| Band set | Bands | G-points | Speedup vs. LBL |
|---|---|---|---|
| RRTMG-SW | 14 | 112 | ~10 000× |
| RRTMG-LW | 16 | 140 | ~10 000× |

Per-band g-point counts vary (some bands have 2, some have 16) — that's why `bnd_limits_gpt` in the RRTMG-SW lookup table is a `(14, 2)` array giving the `[start, end]` g-point index for each band. `_expand_bands_to_gpoints` (`external.py:946-976`) uses that exact table to broadcast a per-band solar fraction to its constituent g-points.

#### The problem they solve

Gas absorption spectra are wildly jagged. Within even a "narrow" band like 15 µm CO₂, the absorption coefficient `k(ν)` oscillates over **~5 orders of magnitude** across thousands of individual spectral lines. To compute flux accurately you'd have to integrate at thousands of wavenumber points per band — ~10⁶ evaluations for the whole spectrum. That's line-by-line (LBL), fine for reference calculations, impossible for a GCM doing this every column, every radiation step.

#### The CKD trick

Within one band, the optical depth `τ(ν) = k(ν)·u` depends only on the *distribution* of `k` values, not on which wavenumber they came from (assuming pressure/temperature don't vary across the band — the key approximation).

So you:

1. Take the spiky `k(ν)` in a band.
2. **Sort it** from smallest to largest. Call the result the **cumulative distribution** `k(g)`, where `g ∈ [0, 1]` is the cumulative probability.
3. `k(g)` is a **smooth, monotonic function** — it can be integrated with a handful of Gauss quadrature points.

Each quadrature point is a **g-point**. It carries:

- a representative absorption coefficient `k_gpt` (tabulated vs. p, T, and absorber amount in the RRTMG LUT),
- a quadrature weight `w_gpt`.

The band flux is then `F_band = Σ_gpt w_gpt · F(k_gpt)`, summed over maybe 8–16 g-points instead of thousands of wavenumbers.

#### "Correlated" — what correlates?

The approximation is that the ordering of `k(ν)` values stays roughly the same across different pressures and temperatures — the strong-absorbing wavenumbers at the surface are still the strong-absorbing ones at the stratopause. That lets RRTMG use one `g(ν)` mapping per band for the whole column. When multiple gases overlap in one band, RRTMG handles the correlation via lookup tables parameterised by `η` (a mixing parameter).

#### Why `run_amip.py` cares

The CMIP6 solar file stores per-band fractions (`SSI_frac`, shape `(time, 14)`). But RRTMG's radiation solver operates at g-point resolution — it wants a `(112,)` array of fractions, one per g-point, summing to 1. Two ways to get there:

1. **Per-band input** → expand: each band's single fraction is repeated over all g-points in that band. `run_amip.py`'s auto-path when `nspec < 50` (`external.py:929-933`).
2. **Per-g-point input** → use directly: file already stores `solar_fraction_by_gpt` with shape `(time, 112)`.

Both end up as a `(112,)` weight vector fed into the RRTMG-SW solver so that band-integrated fluxes are weighted by the current solar spectrum (important for solar-cycle and Maunder-Minimum-type experiments).

### "40 layers" in Kinne files

`lev=40` is a **vertical discretization of the aerosol column into 40 altitude bins**, independent of the host model's vertical grid:

- `zbot_abs` / `ztop_abs` (`(lev=40,)`, m): absolute altitudes of each layer's bottom/top above sea level. Fixed vertical grid, identical everywhere.
- `zbot` / `ztop` (`(lev=40, lat=180, lon=360)`, m above ground): same layering but referenced to local surface altitude `asl`. Over mountains, layer 1 starts higher in absolute altitude.
- `z_aer_fine_mo` / `z_aer_coarse_mo` (`(time=12, lev=40, lat, lon)`, 1/m): **vertical profile of the aerosol distribution** — fraction of column AOD per metre of height in each layer. `Σ z_aer · (ztop − zbot) = 1` over `lev` (i.e. 100% of the column).
- Usage: the radiation scheme multiplies `aod[time, band, lat, lon]` (column total) by `z_aer · Δz` to redistribute AOD onto the host model's vertical levels.

**File layout summary**: "how much AOD per band (14 SW or 16 LW) × where in the column (40 layers) × which season (12 months) × where on Earth (180 × 360)."

**Current `run_amip.py` behaviour**: `_load_monthly_zonal` collapses over `lon`, `lnwl`, *and* `lev` to return only `(time=12, lat)` total column AOD. It discards all band- and height-resolved information that Kinne was specifically built to provide.

## 5. Recommended code changes

1. **`external.py:_to_days_float`** — inspect the `units` attribute for `"months since"` / `"hours since"` / `"seconds since"` and convert to days before returning.
2. **`external.py:get_ozone_at_time`** — add a non-cyclic, calendar-aware path for time-varying ozone (e.g. CMIP6 input4MIPs), or document that files longer than one year must be pre-reduced to a 12-month climatology.
3. **New volcanic loader** — dedicated code path for CMIP6 volcanic files (`ext_sun`, `ext_earth`, altitude in km, `month`, `latitude`), returning per-band column AOD after vertical integration and 550 nm band selection.
4. **`external.py:_load_time_gpt` (line 297)** — case-insensitive `tsi_var` lookup consistent with `_load_timeseries`.
5. **`run_amip.py` help strings** — document MPI-M CMIP6 solar requirement: `--solar-tsi-var TSI --solar-spectral-var SSI_frac`.

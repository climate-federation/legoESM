# AMIP — true CMIP6 AMIP deck for legoESM

Live trace of work on the **AMIP branch** to bring `scripts/run_amip.py`
from a single-file (SST/SIC) experiment into a proper CMIP6 AMIP deck
with **prescribed SST + SIC + transient GHG + ozone + solar (TSI/spectral) +
aerosol (Kinne-style) + volcanic** forcing, validated for physical realism
and conservation across all four grid types (cubed-sphere, gaussian,
latlon, voronoi).

The CMIP6 AMIP protocol is a 1979–2014 (extended to 2021 in CMIP7)
atmosphere-only run with:

| Forcing          | CMIP6 source                                       |
|------------------|----------------------------------------------------|
| SST + SIC        | PCMDI input4MIPs (`amipbc_sic_*` and `amipbc_sst_*`) — monthly, 1°×1° |
| Greenhouse gases | input4MIPs `greenhouse_historical_plus.nc` — annual, global-mean |
| Ozone            | input4MIPs `vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0` — monthly, plev × lat |
| Solar (TSI + spectral) | input4MIPs `swflux_14band_cmip6_1850-2299` — daily, 14 SW bands |
| Aerosol (tropospheric) | MPI-M Kinne dataset `aeropt_kinne_{sw_b14,lw_b16}_*_rast.nc` — monthly, lat × lon × lev |
| Volcanic stratospheric | `bc_aeropt_cmip6_volc_lw_b16_sw_b14_<year>.nc` — per-band per-altitude |

## Status (current iteration)

| Component | Loader | Generator | Driver wiring | Test | Notes |
|-----------|--------|-----------|---------------|------|-------|
| SST/SIC (lat-lon) | ✅ | ✅ | ✅ | unit | `amip.py:load_amip_forcing` |
| SST/SIC (ICON unstructured) | ✅ | — | ✅ | unit | KD-tree NN regrid |
| GHG (annual_file) | ✅ | ✅ | ✅ | unit | CMIP6 `(time, lat=1, lon=1)` |
| Ozone (climatology) | ✅ | ✅ | ✅ | unit | monthly cyclic |
| Ozone (input4MIPs time-varying) | ✅ | ✅ | ✅ | unit | non-cyclic dispatch |
| Solar TSI (file) | ✅ | ✅ | ✅ | unit | case-insensitive var lookup |
| Solar spectral (CMIP6 14-band) | ✅ | ✅ | ✅ | unit | bands → 112 g-pts auto-expand |
| Aerosol Kinne | ✅ | ✅ | ✅ | unit | zonal-mean (band/lev collapsed) |
| Volcanic CMIP6 bc_aeropt | ✅ | ✅ | ✅ | unit | `_load_volcanic_cmip6` integral |
| Driver gating (gray vs RRTMG) | ✅ | — | ✅ | unit | `test_driver_forcing_dispatch` |

## Iteration log

### Iter 1 — Baseline audit and AMIP.md scaffold

- Branched off `main` to `AMIP`.
- Audited `scripts/run_amip.py`, `src/legoesm/forcing/{amip,external,experiments}.py`, and existing tests.
- Confirmed the single-file SST/SIC loader is in place and unit-tested.
- Confirmed `external.py` already implements every CMIP6 forcing channel from
  the AMIP deck (GHG annual, ozone monthly cyclic + non-cyclic, solar TSI +
  spectral 14-band, Kinne aerosol, CMIP6 volcanic ext_sun integration).
- The MPI-M HPC paths in `amip_forcing_files.md` are unreachable on this host
  → must generate **synthetic CMIP6-shape** forcing files with the same
  schemas to drive a self-contained AMIP deck on the local filesystem.
- The `EXPERIMENT_TEMPLATES["amip"]` entry has `forcing_type="fixed"` and
  hard-codes 1979 GHG concentrations — this is **wrong** for the CMIP AMIP
  protocol (which is transient through 1979–2014). Will fix in iter 2.

### Iter 2 — Self-contained AMIP CMIP6 deck infrastructure

**New code**

- `scripts/generate_amip_forcing.py` — synthetic-but-physical forcing-file
  generator producing the canonical 6-file deck:
  - SST/SIC (HadISST schema, monthly, 1979–2014, area-weighted global mean
    292.6 K, +0.18 K/decade warming trend, polar sea-ice).
  - GHG annual file (CMIP6 `(time, lat=1, lon=1)` with `time.units = "year as %Y.%f"`,
    CO2 336.8 → 397.6 ppm, CH4/N2O/CFC-11/CFC-12 trajectories from NOAA AGGI).
  - Ozone monthly climatology (3D, 30 plev × 36 lat) with column 258–385 DU
    and an Antarctic spring ozone-hole proxy.
  - Solar daily file (TSI 1360.4–1361.6 W/m², 11-yr Schwabe cycle, 14 SW band
    fractions with UV variability).
  - Aerosol Kinne-style monthly zonal AOD (550 nm, 0.02–0.25 with hemispheric
    seasonality) + volcanic time-series (background + El Chichón 1982 +
    Pinatubo 1991, peak AOD 0.18).
- `scripts/run_amip_cmip6_deck.py` — orchestrating wrapper that auto-generates
  the deck if missing, fixes the canonical RRTMG + Sundqvist + Kessler + SBM
  + Louis stack, wires every external-forcing flag, supports all four grid
  types, and exposes a `--dry-run` that prints the `run_amip.py` invocation.
- `tests/unit/test_amip_cmip6_deck.py` — 13 unit tests covering every channel
  (SST area-weighted mean band, freezing point, sea-ice extent, GHG annual
  file load + transient evolution + CFC presence, ozone climatology load on
  3D grid, aerosol AOD band, Pinatubo signal, solar TSI + 14-band → 112-gpt
  expansion, AMIP-template-is-transient regression).

**Code fixes**

- `src/legoesm/forcing/experiments.py`:
  - `EXPERIMENT_TEMPLATES["amip"]` `forcing_type` changed from
    `"fixed"` → `"transient"` (CMIP6 protocol mandates time-varying GHGs;
    the model_driver only applies the transient override when the template
    is transient).
  - `_GHG_HISTORICAL` table extended with 1979, 1990, 2010, 2014, 2021 entries
    so AMIP-period GHG interpolation hits real anchor years (was 1850/1900/
    1950/1980/2000/2014 only — gappy across the AMIP window).
  - `_GHG_TABLES["amip"] = _GHG_HISTORICAL` so `ghg_at_year("amip", 1990)`
    returns the historical value rather than the constant base.
  - Updated docstrings to reflect the new transient AMIP contract.
- `src/legoesm/forcing/external.py:_interp_vertical`:
  - **Bug fix**: handle descending source-pressure axes. CMIP6 ozone files
    (UReading-CCMI-1-0) ship `plev` as `1000 → 0.1 hPa` (descending), but
    the previous code passed it straight to `np.searchsorted`, whose
    ascending-xp contract was silently violated, producing essentially-zero
    interpolated ozone at all model levels (caught by
    `TestOzone::test_climatology_loads`).  Fix: detect descending input and
    flip both `log_p_src` and the field axis before the lerp, mirroring the
    handling already in place for descending `lat_src` in
    `_interp_zonal_to_grid`.

**Validation**

- All 13 new AMIP-deck unit tests pass.
- All 112 pre-existing forcing tests pass (`test_external_forcing.py`,
  `test_driver_forcing_dispatch.py`, `test_amip_config.py`,
  `test_corrections.py`) — no regressions from the loader or template fixes.


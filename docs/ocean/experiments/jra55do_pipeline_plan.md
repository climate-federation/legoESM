# JRA55-do Forcing Pipeline (Tropical OMIP Item 2)

**Status**: scoping
**Date**: 2026-05-02
**Parent docs**: `tropical_omip_plan.md` Item 2; `omip_protocol_review.md`
Gap 3 (forcing source).

This is the scoping doc for **Item 2** of the tropical OMIP plan: reading
JRA55-do v1.4+ corrected forcing, regridding to model grid, pre-processing
to noleap, and producing a single Zarr cache the driver consumes per-step.

## 1. What already exists (audit summary)

A repo survey found the following directly reusable infrastructure:

| Component | File | Reuse |
|---|---|---|
| **Local Zarr cache pattern** (download → normalize → cache) | `training/era5_to_state.py`, `ml/data/era5_loader.py` | Direct template |
| **Zarr/NetCDF dispatch** (`decode_times=False`, format-agnostic open) | `forcing/external.py:_open_forcing_dataset` | Use as-is |
| **Variable-alias normalization** (CMIP names → repo names) | `_WB2_VAR_ALIASES` in `era5_loader.py` | Pattern copy |
| **Linear-in-time forcing dispatch** (`get_forcing_at_time(forcing, day)`) | `forcing/amip.py` | Pattern copy |
| **AMIP-style driver structure** | `scripts/run/run_amip.py` | Pattern copy |
| **`FreshwaterForcing` namedtuple + `model.step(freshwater=...)`** | `ocean/freshwater.py`, `ocean_model_latlon_cgrid.py:364,482,806` | Use as-is |
| **`AtmToSurface` 2D struct + `ocean_tile_response`** | `coupler/coupling_fields.py`, `coupler/coupler.py` | Use as-is |

Three hard gaps the audit surfaced:

1. **No conservative regridding anywhere in the repo.** The existing
   `grids/regridding.py` is KD-tree inverse-distance — fine for SST, T, q
   but **not** flux-conservative for precipitation, runoff, or wind stress.
   For 1° → 1° regular lat-lon → regular lat-lon, this is a tractable
   ~100 LOC area-weighted overlap routine. xESMF/ESMF dependency is **not**
   needed for the regular-grid case.
2. **Calendar hardcoding.** `forcing/time_utils.py` (30 lines) is
   `day % 365`-only. No mapping between simulation day and absolute
   (year, day-of-year). OMIP-2 needs an absolute reference (1958-01-01) so
   that cycling at end-of-cycle restarts at the correct seasonal phase.
3. **No analytic-vs-data wind path.**
   `ocean/physics/surface_forcing/wind_profiles.py` is closed-form only
   (cosine-latitude, single/double-gyre). For OMIP, wind enters via
   `AtmToSurface.u_lowest, v_lowest` populated by the *driver*, not by
   `wind_profiles.py`. So this is not a gap in the bulk-flux machinery; it's
   a driver-level question (Item 4 territory).

## 2. JRA55-do v1.4+ "corrected" variable schema

NCAR distributes JRA55-do at https://rda.ucar.edu/datasets/d639000/ . The
"corrected" flavour incorporates LY09 bias adjustments (per OMIP §2.1).
Variables (CMOR names):

| Var | Long name | Units | Cadence | Native grid | Used for |
|---|---|---|---|---|---|
| `uas` | 10 m eastward wind | m/s | **3-hourly** | TL319 (~0.5°) | `AtmToSurface.u_lowest` |
| `vas` | 10 m northward wind | m/s | **3-hourly** | TL319 | `AtmToSurface.v_lowest` |
| `tas` | 2 m air temperature | K | 6-hourly | TL319 | `AtmToSurface.T_lowest` |
| `huss` | 2 m specific humidity | kg/kg | 6-hourly | TL319 | `AtmToSurface.q_lowest` |
| `psl` | sea-level pressure | Pa | 6-hourly | TL319 | `AtmToSurface.p_surface` |
| `rsds` | surface SW down | W/m² | 3-hourly | TL319 | `AtmToSurface.sw_down` |
| `rlds` | surface LW down | W/m² | 3-hourly | TL319 | `AtmToSurface.lw_down` |
| `prra` | rainfall flux | kg/m²/s | 3-hourly | TL319 | `AtmToSurface.precip_total` (liquid part) |
| `prsn` | snowfall flux | kg/m²/s | 3-hourly | TL319 | `AtmToSurface.precip_snow` |
| `friver` | river runoff | kg/m²/s | **daily** | 0.25° land grid | `FreshwaterForcing.runoff` |

Density `rho_lowest` is reconstructed in the driver from `psl`, `tas`, `huss`
via the ideal-gas EOS. `cos_zenith` is computed from absolute date + lat/lon
analytically.

## 3. Implementation plan

### File layout

```
src/legoesm/forcing/
    jra55_do.py               # NEW. ~500 LOC. Self-contained module.
    time_utils.py             # EXTEND. ~+50 LOC.
src/legoesm/grids/
    conservative_regrid.py    # NEW. ~150 LOC. Area-weighted 2D overlap.
scripts/
    prepare_omip_forcing.py   # NEW. ~100 LOC thin CLI.
tests/forcing/
    test_jra55_do.py          # NEW.
    test_conservative_regrid.py  # NEW.
    test_time_utils_omip.py   # NEW.
```

Single self-contained `jra55_do.py` mirrors `amip.py`'s pattern (one file,
all loading/regridding/dispatch). Keeps the new code a single audit unit.

### Phase 1 — Calendar extension (~½ day)

Extend `forcing/time_utils.py`:

```python
def date_to_day(year, month, day_of_month, ref_year=1958, calendar="noleap"):
    """Map (Y, M, D) → simulation days since (ref_year, 1, 1)."""

def day_to_date(day, ref_year=1958, calendar="noleap"):
    """Inverse: simulation day → (Y, M, D, hour)."""

def cos_solar_zenith(lat, lon, day, calendar="noleap"):
    """Solar zenith angle from absolute date — for AtmToSurface.cos_zenith."""
```

`calendar="noleap"` is the only required value for tropical OMIP; the API
admits `"gregorian"` for future use. This module stays small and pure-NumPy
(no JAX) — used at Python level by the driver.

**Tests**: round-trip `(Y,M,D) → day → (Y,M,D)`, leap-day handling
(Feb 29 → Mar 1 in noleap), edge cases at year boundaries.

### Phase 2 — Conservative regridding (~1 day)

`src/legoesm/grids/conservative_regrid.py`:

```python
def compute_overlap_weights_lonlat(
    src_lat_edges, src_lon_edges,
    dst_lat_edges, dst_lon_edges,
) -> RegridWeights:
    """Compute area-weighted overlap weights between two regular lat-lon
    grids. Returns sparse weight indices + values for JIT-safe apply.

    Algorithm: for each (dst_lat, dst_lon) cell, find src cells whose
    edges overlap; weight = (overlap area on sphere) / (dst cell area).
    Conservative: sum of weights per dst cell = 1 to machine precision.
    """

def apply_regrid(field, weights) -> jnp.ndarray:
    """Apply pre-computed weights. JIT-safe."""
```

For 0.5° → 1° (TL319 → 1° regular), each target cell receives 4 source
cells. For 0.25° → 1° (friver land grid → 1°), each target receives 16.
Weight computation is one-time at startup; apply is a matmul.

**Tests**:
- Constant field regrids to itself (conservation gates this).
- Linear field regrids exactly.
- Sum of weights per dst cell = 1 to 1e-14.
- Total integral conserved (mass-conservation test).

### Phase 3 — JRA55-do loader and pre-processor (~3 days)

`src/legoesm/forcing/jra55_do.py`:

```python
@dataclass(frozen=True)
class JRA55DoConfig:
    zarr_root: str               # GCS or local path to JRA55-do "corrected"
    years: tuple[int, int]       # e.g., (1958, 2018) for OMIP-2
    target_grid: LatLonGrid      # model 1° grid
    cache_dir: str               # local Zarr cache
    drop_leap_days: bool = True  # noleap pre-processing

def ensure_local_jra55_cache(config: JRA55DoConfig) -> Path:
    """Download/regrid JRA55-do once; output single consolidated Zarr."""

def load_jra55_slice(cache_path: Path, day: float) -> JRA55Slice:
    """Linear-in-time interpolation between adjacent forcing records.
    Returns a NamedTuple of 2D fields ready to load into AtmToSurface."""
```

Operations inside `ensure_local_jra55_cache`:

1. Open source Zarr (GCS via gcsfs, or local) with `decode_times=False`.
2. Apply variable aliases (CMOR → repo names).
3. **Drop leap days** (Feb 29 across years 1960, 1964, ..., 2016) — single
   `xarray.sel(time=~is_leap_day)` operation.
4. Regrid each variable:
   - `uas, vas, tas, huss, psl` (TL319, 0.5°) → 1° regular: conservative
   - `rsds, rlds` → 1° regular: conservative (radiation flux)
   - `prra, prsn` → 1° regular: conservative (precip flux — must conserve
     global integral)
   - `friver` (0.25° land) → 1° ocean coastal: conservative + coastal
     redistribution (uniform-coastal-by-band per
     `tropical_omip_plan.md` §4 Item 2 quick-fix)
5. Unit checks (precip in kg/m²/s, T in K, etc.) — log any out-of-band values.
6. Write single consolidated Zarr `<cache_dir>/jra55_do_v14_omip2_1deg_noleap.zarr`
   with explicit chunking: `{"time": 1460, "lat": 180, "lon": 360}`
   (1460 = year of 6-hourly records).
7. Total cache size: ~30 GB.

The runtime `load_jra55_slice` is JIT-clean (no Python control flow on
traced values), takes a `day: float` (simulation day since 1958-01-01,
noleap), and returns a `JRA55Slice` NamedTuple. The driver's per-step
glue then builds `AtmToSurface` and `FreshwaterForcing` from the slice.

### Phase 4 — Driver-side glue (deferred to Item 4)

The pieces that bridge `JRA55Slice` to `AtmToSurface` and
`FreshwaterForcing` are **driver-level glue** and live in
`scripts/run_omip_ocean.py` (Item 4), not here. Specifically:

```python
def jra55_to_atm_surface(slice: JRA55Slice, lat, lon, day) -> AtmToSurface:
    return AtmToSurface(
        u_lowest=slice.uas, v_lowest=slice.vas,
        T_lowest=slice.tas, q_lowest=slice.huss,
        p_surface=slice.psl,
        p_lowest=slice.psl,                          # 2 m ≈ surface
        rho_lowest=slice.psl / (R_d * slice.tas
                                * (1 + 0.61 * slice.huss)),  # virtual T
        sw_down=slice.rsds, lw_down=slice.rlds,
        precip_total=slice.prra, precip_snow=slice.prsn,
        cos_zenith=cos_solar_zenith(lat, lon, day),
        co2_ppmv=jnp.array(400.0),                   # static for OMIP
        has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0),
    )

def jra55_to_freshwater(slice, lhflx, dt) -> FreshwaterForcing:
    evap = lhflx / constants.L_v
    return FreshwaterForcing(
        precip=slice.prra + slice.prsn,
        evap=evap,
        runoff=slice.friver_redistributed,
        ice_fw=jnp.zeros_like(slice.prra),           # no ice in tropical OMIP
    )
```

Mentioned here for completeness but not built in Item 2.

### Phase 5 — `scripts/data/prepare_omip_forcing.py` (~½ day)

Thin CLI wrapper:

```bash
python scripts/data/prepare_omip_forcing.py \
    --source gs://noresm-jra55do/v1.4/corrected \
    --years 1958 2018 \
    --target-grid latlon_1deg \
    --cache-dir /scratch/legoESM/jra55_do_omip2 \
    --noleap
```

Calls `ensure_local_jra55_cache(config)`. One-time run before any tropical
OMIP integration. Verbose logging of regrid weights, conservation
diagnostics, leap-day drops, and final cache size.

## 4. Effort breakdown

| Phase | Effort | Deliverable |
|---|---|---|
| 1. Calendar extension | ½ d | `time_utils.py` + tests |
| 2. Conservative regridding | 1 d | `conservative_regrid.py` + tests |
| 3. JRA55-do loader + cache builder | 3 d | `jra55_do.py` + tests |
| 5. CLI wrapper | ½ d | `prepare_omip_forcing.py` |
| **Total** | **5 d** | (Phase 4 is in Item 4) |

Parent plan budget for Item 2: 1 week. **On budget.**

## 5. Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Conservative regridding subtly wrong** — sphere-area overlap is easy to get wrong at the pole. Symptom: global-mean precip not conserved across regrid. | Test with a constant field (must regrid to itself), and a linear lat field (must regrid exactly). Add a runtime assertion: sum of regridded precip × area = sum of source × area to 1e-10. |
| 2 | **JRA55-do v1.4+ "corrected" provenance** — different distributors (NCAR, JAMSTEC, ESGF) ship slightly different post-processing. | Pin to the NCAR RDA d639000 v1.4+ distribution; document the SHA256 of the Zarr in the run README. |
| 3 | **Cache regeneration on every run** — naive caching re-downloads every time. | Include `(zarr_root, years, target_grid_hash, version)` in the cache filename; skip rebuild if cache exists and tag matches. |
| 4 | **friver land grid alignment** — JRA55-do `friver` is on a 0.25° land grid that doesn't align with our ocean coastline. Naive regrid puts runoff on land cells. | The "uniform-coastal-by-band" quick-fix from `tropical_omip_plan.md` redistributes friver to the nearest ocean coastal cells per latitude band, conserving the band-total flux. Implement in `jra55_do.py` after the conservative regrid step. |
| 5 | **Time-axis alignment** — winds (3-hourly) and T,q (6-hourly) on different grids. | Resample to a common 3-hourly axis at cache-build time (interpolate T, q linearly between 6-hourly samples). Document this. |

## 6. What this does and does not produce

**This pipeline produces**: a single consolidated Zarr cache at
`<cache_dir>/jra55_do_v14_omip2_1deg_noleap.zarr` containing all 10
variables on a common 1° regular lat-lon grid, 3-hourly time axis,
noleap calendar, 1958–2018 (60 yr × 8 records/day × 365 days = 175,200
time steps). ~30 GB.

**This pipeline does not produce**: the driver. The driver (Item 4)
consumes the cache via `load_jra55_slice` per step. The bridge between
`JRA55Slice` and `AtmToSurface` / `FreshwaterForcing` is a few lines of
glue that goes in the driver, not in `jra55_do.py`.

## 7. Test plan

Tests under `tests/forcing/`:

- `test_time_utils_omip`: round-trip dates, leap-year handling, cycling
  across year boundary.
- `test_conservative_regrid`: constant field self-regrids, linear field
  regrids exactly, mass conservation to 1e-14, weight-row-sum = 1.
- `test_jra55_do_cache_build`: small synthetic JRA55-do dataset (3 days,
  4×8 grid), verify cache structure, leap-day drop, variable aliasing,
  unit consistency.
- `test_jra55_do_slice_interpolation`: load slice at `day=10.5`, verify
  linear interpolation between records 10.0 and 11.0; AD-clean
  (`jax.grad` through interpolation produces finite gradients).
- `test_jra55_do_atm_surface_conversion`: end-to-end glue from
  `JRA55Slice` → `AtmToSurface`, verify field shapes, dtypes,
  consistency with downstream `ocean_tile_response`.

## 8. Next action

Phase 1 (calendar) is the smallest discrete piece — half a day, no dependencies, immediately useful. Recommend starting there once user signs off on this plan.

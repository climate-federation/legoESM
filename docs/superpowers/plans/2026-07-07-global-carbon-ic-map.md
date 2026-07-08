# Global Land-Carbon IC Map (Stage A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a global, gridded `CarbonState` initial-condition map ("finidat") by sampling the (PFT × climate) space, spinning each archetype to a verified semi-analytic equilibrium, and cover-weight-mapping each grid cell.

**Architecture:** Three pure, independently-testable core functions in `land.carbon.global_init` (build archetypes → equilibrate via the existing semi-analytic spin-up → cover-weighted map to the grid), fed by a climate-feature forcing generator (`land.climate_forcing`, which `lmip_forcing` is refactored onto) and a climatology→features reducer. A `scripts/data` driver wires the real CLM5/ERA5/HWSD data; a `scripts/validate` validator confirms drift→0 on a cell sample.

**Tech Stack:** JAX (jnp, vmap, lax.scan), Equinox-style NamedTuples, existing legoESM land model (`step_multilayer_land`, `analytic_slow_pool_equilibrium`), scikit-learn-free k-means (implement in numpy — no new deps).

## Global Constraints

- Depends on `legoesm.land.carbon.spinup.analytic_slow_pool_equilibrium` + `SlowPoolFluxes` (branch `land/carbon-global-init`, off the PR#823 spin-up).
- No new third-party deps (k-means in numpy). Constants only from `legoesm.constants`; saturation only from `legoesm.thermo`; no hardcoded physical constants (CI ratchets `tests/test_no_inline_physics_coeffs.py`, `tests/test_no_hardcoded_constants.py`). `land/` is scanned by the inline-coeff ratchet — move empirical floats to `_UPPER_SNAKE` module constants or `# coeff-ok:`.
- Every new `.py` gets a direct unit test (CLAUDE.md). `JAX_ENABLE_X64=1` for numerics.
- Ginsburg login-node policy: NO multi-core / >10 s / JAX-compile on the login node. Run pytest + drivers via `sbatch --account=glab` (env: `PYTHONPATH=$(ls -d packages/*/ | tr '\n' ':')`, `JAX_PLATFORMS=cpu`, python `/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python`). py_compile is login-safe.
- Baseline 1°; keep every interface resolution-agnostic (functions take `(ncell, …)` arrays, never a hardcoded grid).
- Reuse `make_soil_grid`, `SOIL_TEXTURE_VG`, `CLM5_PFT_NAMES`/`_CLM5_PFT_TABLE_RAW`/`PARAM_NAMES`, `init_carbon_state`, `reconstruct_carbon_diagnostics`. No duplicated numerics.

---

## File Structure

- `packages/land/legoesm/land/climate_forcing.py` — NEW. Climate-feature single-column forcing generator.
- `packages/land/legoesm/land/lmip_forcing.py` — MODIFY. Refactor onto `climate_forcing` (latitude preset).
- `packages/land/legoesm/land/carbon/climate_features.py` — NEW. Reduce a monthly climatology to the 5 per-cell features.
- `packages/land/legoesm/land/carbon/global_init.py` — NEW. `ArchetypeTable`, `build_archetypes`, `equilibrate_archetypes`, `map_to_grid`.
- `scripts/data/build_global_carbon_ic.py` — NEW. Driver: load real maps → build → equilibrate → map → write finidat.
- `scripts/validate/global_carbon_ic_map.py` — NEW. Drift/realism validator on a cell sample.
- Tests: `tests/land/unit/test_climate_forcing.py`, `tests/land/unit/test_climate_features.py`, `tests/land/unit/test_global_init.py`, `tests/land/validation/test_global_carbon_ic_map.py`.

---

## Task 1: Climate-feature forcing generator

**Files:**
- Create: `packages/land/legoesm/land/climate_forcing.py`
- Modify: `packages/land/legoesm/land/lmip_forcing.py`
- Test: `tests/land/unit/test_climate_forcing.py`

**Interfaces:**
- Produces: `make_climatological_forcing(mat_k, t_seasonal_amp_k, sw_mean_w, precip_rate, doy, hour, *, t_diurnal_amp_k=_T_DIURNAL_AMP_K, dtype=jnp.float64) -> AtmToSurface` (shape `(1,)` fields). `mat_k` mean-annual T [K]; `t_seasonal_amp_k` half-amplitude of the annual T cycle [K]; `sw_mean_w` mean-annual down-SW [W/m²] scaling the diurnal insolation shape; `precip_rate` [kg/m²/s]; `doy∈[0,365)`, `hour∈[0,24)` may be traced.
- `lmip_forcing.make_synthetic_lmip_forcing` keeps its signature but computes `(mat, t_seasonal, sw_mean)` from latitude then delegates.

- [ ] **Step 1: Write the failing test**

```python
# tests/land/unit/test_climate_forcing.py
from __future__ import annotations
import jax, jax.numpy as jnp
import numpy.testing as npt
jax.config.update("jax_enable_x64", True)
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.climate_forcing import make_climatological_forcing

def test_warm_climate_has_higher_T_than_cold():
    warm = make_climatological_forcing(300.0, 5.0, 250.0, 2e-5, jnp.asarray(200.0), jnp.asarray(12.0))
    cold = make_climatological_forcing(270.0, 5.0, 250.0, 2e-5, jnp.asarray(200.0), jnp.asarray(12.0))
    assert float(warm.T_lowest[0]) > float(cold.T_lowest[0])

def test_seasonal_amplitude_controls_summer_winter_gap():
    jul = make_climatological_forcing(285.0, 15.0, 250.0, 2e-5, jnp.asarray(200.0), jnp.asarray(6.0))
    jan = make_climatological_forcing(285.0, 15.0, 250.0, 2e-5, jnp.asarray(15.0), jnp.asarray(6.0))
    assert float(jul.T_lowest[0]) - float(jan.T_lowest[0]) > 20.0

def test_noon_shortwave_scales_with_sw_mean():
    lo = make_climatological_forcing(290.0, 5.0, 100.0, 2e-5, jnp.asarray(200.0), jnp.asarray(12.0))
    hi = make_climatological_forcing(290.0, 5.0, 300.0, 2e-5, jnp.asarray(200.0), jnp.asarray(12.0))
    assert float(hi.sw_down[0]) > float(lo.sw_down[0])
    assert float(lo.sw_down[0]) >= 0.0

def test_returns_atmtosurface_shape1_and_traceable():
    f = make_climatological_forcing(290.0, 5.0, 250.0, 2e-5, jnp.asarray(180.0), jnp.asarray(12.0))
    assert isinstance(f, AtmToSurface) and f.T_lowest.shape == (1,)
    out = jax.jit(lambda d, h: make_climatological_forcing(290.0, 5.0, 250.0, 2e-5, d, h).sw_down)(
        jnp.asarray(180.0), jnp.asarray(12.0))
    assert jnp.all(jnp.isfinite(out))

def test_lmip_latitude_preset_unchanged():
    # Regression: the refactored latitude wrapper matches the old T/precip logic.
    from legoesm.land.lmip_forcing import make_synthetic_lmip_forcing
    f = make_synthetic_lmip_forcing(45.5 * jnp.pi / 180.0, 0.0, jnp.asarray(200.0), jnp.asarray(12.0))
    # 45.5N July: T_atm ~ 280 K per the documented latitude baseline (+/- a few K).
    assert 273.0 < float(f.T_lowest[0]) < 288.0
```

- [ ] **Step 2: Run test to verify it fails**

Run (login-safe compile check first, then via sbatch): `sbatch` a job running `pytest tests/land/unit/test_climate_forcing.py -q`
Expected: FAIL — `ModuleNotFoundError: legoesm.land.climate_forcing`.

- [ ] **Step 3: Write minimal implementation**

Move the T/precip/humidity/solar body of `lmip_forcing.make_synthetic_lmip_forcing` into `climate_forcing.make_climatological_forcing`, parameterised by climate features. Keep the module-level `_UPPER_SNAKE` constants (diurnal amp, snow threshold, RH, pressures, winds, CO2) here; the solar-noon *shape* comes from `hour`/`doy` and is scaled so the daily-mean matches `sw_mean_w`.

```python
# packages/land/legoesm/land/climate_forcing.py  (sketch — reuse lmip_forcing constants)
from __future__ import annotations
import jax.numpy as jnp
from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.thermo import saturation_mixing_ratio

_T_DIURNAL_AMP_K = 3.0
_T_SEASONAL_PEAK_DAY = 200.0
_YEAR_DAYS = 365.0
_HOURS_PER_DAY = 24.0
_DIURNAL_PEAK_HOUR = 14.0
_LW_EFF_EMISSIVITY = 0.75
_RH_FRACTION = 0.6
_SNOW_RAIN_THRESHOLD_K = 275.0
_P_SURFACE_PA = 1.0e5
_P_LOWEST_PA = 9.5e4
_RHO_LOWEST = 1.2
_U_LOWEST = 3.0
_V_LOWEST = 2.0
_CO2_PPMV = 412.0
# Peak-to-mean insolation shape factor for a sinusoidal daytime half-cycle.
_SW_PEAK_OVER_MEAN = jnp.pi   # noon peak of a rectified-sine day ≈ pi * daily-mean

def make_climatological_forcing(mat_k, t_seasonal_amp_k, sw_mean_w, precip_rate,
                                doy, hour, *, t_diurnal_amp_k=_T_DIURNAL_AMP_K,
                                dtype=jnp.float64) -> AtmToSurface:
    local_hour = hour
    # Daytime insolation shape: rectified sine peaking at solar noon, scaled so
    # its daily mean equals sw_mean_w.
    day_phase = jnp.cos(2.0 * jnp.pi * (local_hour - 12.0) / _HOURS_PER_DAY)
    sw = jnp.maximum(_SW_PEAK_OVER_MEAN * sw_mean_w * day_phase, 0.0)
    sw_down = jnp.asarray([sw], dtype=dtype)
    T_season = t_seasonal_amp_k * jnp.cos(2.0 * jnp.pi * (doy - _T_SEASONAL_PEAK_DAY) / _YEAR_DAYS)
    T_diurnal = t_diurnal_amp_k * jnp.cos(2.0 * jnp.pi * (local_hour - _DIURNAL_PEAK_HOUR) / _HOURS_PER_DAY)
    T_atm = jnp.asarray([mat_k + T_season + T_diurnal], dtype=dtype)
    lw_down = jnp.asarray([_LW_EFF_EMISSIVITY * constants.sigma_sb * T_atm[0] ** 4], dtype=dtype)
    p_sfc = jnp.asarray([_P_SURFACE_PA], dtype=dtype)
    q_atm = (_RH_FRACTION * saturation_mixing_ratio(T_atm, p_sfc)).astype(dtype)
    precip_total = jnp.asarray([precip_rate], dtype=dtype)
    precip_snow = jnp.where(T_atm < _SNOW_RAIN_THRESHOLD_K, precip_total, jnp.zeros(1, dtype=dtype))
    return AtmToSurface(
        sw_down=sw_down, lw_down=lw_down, precip_total=precip_total, precip_snow=precip_snow,
        T_lowest=T_atm, q_lowest=q_atm, u_lowest=jnp.asarray([_U_LOWEST], dtype=dtype),
        v_lowest=jnp.asarray([_V_LOWEST], dtype=dtype), p_lowest=jnp.asarray([_P_LOWEST_PA], dtype=dtype),
        p_surface=p_sfc, rho_lowest=jnp.asarray([_RHO_LOWEST], dtype=dtype),
        cos_zenith=jnp.asarray([jnp.maximum(day_phase, 0.0)], dtype=dtype),
        co2_ppmv=jnp.asarray([_CO2_PPMV], dtype=dtype),
        has_radiation=jnp.ones(1, dtype=dtype), has_precipitation=jnp.ones(1, dtype=dtype))
```

Then refactor `lmip_forcing.make_synthetic_lmip_forcing` to compute the latitude preset and delegate (keep its own solar-geometry `cos_sza` for `sw_mean`):

```python
# lmip_forcing.py — body becomes:
#   mat = _T_BASE_EQUATOR_K - _T_BASE_POLE_DROP_K*abs(lat_rad)/half_pi
#   t_seasonal = _T_SEASONAL_POLE_AMP_K*abs(lat_rad)/half_pi
#   sw_mean = constants.S_0 * <daily-mean cos_sza(lat, day)>   # keep existing solar geometry
#   return make_climatological_forcing(mat, t_seasonal, sw_mean, precip_rate, day, hour, dtype=dtype)
```

Keep `make_synthetic_lmip_forcing`'s public signature and `_DEFAULT_PRECIP_RATE`.

- [ ] **Step 4: Run tests to verify they pass**

Run (sbatch): `pytest tests/land/unit/test_climate_forcing.py tests/land/unit/test_lmip_forcing.py -q`
Expected: PASS (including the unchanged `test_lmip_forcing.py`).

- [ ] **Step 5: Commit**

```bash
git add packages/land/legoesm/land/climate_forcing.py packages/land/legoesm/land/lmip_forcing.py tests/land/unit/test_climate_forcing.py
git commit -m "feat(land): climate-feature forcing generator; refactor lmip_forcing onto it"
```

---

## Task 2: Climatology → per-cell climate features

**Files:**
- Create: `packages/land/legoesm/land/carbon/climate_features.py`
- Test: `tests/land/unit/test_climate_features.py`

**Interfaces:**
- Produces: `ClimateFeatures = NamedTuple(mat_k, map_yr, t_seasonal_amp_k, aridity, sw_mean_w)` each `(ncell,)`; and `reduce_climatology_to_features(monthly_t_k, monthly_precip_rate, monthly_sw_w, monthly_netrad_w) -> ClimateFeatures` where each `monthly_*` is `(ncell, 12)`. `map_yr = mean(precip_rate)*seconds_per_year`; `t_seasonal_amp_k = 0.5*(max−min monthly T)`; `aridity = MAP / PET` with `PET = priestley_taylor(mean netrad)`; `sw_mean_w = mean(monthly_sw)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/land/unit/test_climate_features.py
import numpy as np, numpy.testing as npt
from legoesm.land.carbon.climate_features import reduce_climatology_to_features

def _const_month(v, n=3): return np.full((n, 12), float(v))

def test_mat_is_annual_mean_T():
    t = np.tile(np.linspace(280, 300, 12), (2, 1))
    f = reduce_climatology_to_features(t, _const_month(2e-5, 2), _const_month(250, 2), _const_month(100, 2))
    npt.assert_allclose(f.mat_k, t.mean(axis=1), rtol=1e-9)

def test_map_is_annual_precip():
    f = reduce_climatology_to_features(_const_month(290), _const_month(2e-5), _const_month(250), _const_month(100))
    npt.assert_allclose(f.map_yr, 2e-5 * 365.0 * 86400.0, rtol=1e-6)

def test_seasonality_half_range():
    t = np.zeros((1, 12)); t[0, 0] = 270.0; t[0, 6] = 300.0; t[0, 1:6] = 285; t[0, 7:] = 285
    f = reduce_climatology_to_features(t, _const_month(2e-5, 1), _const_month(250, 1), _const_month(100, 1))
    npt.assert_allclose(f.t_seasonal_amp_k, 15.0, rtol=1e-9)

def test_aridity_wet_gt_dry():
    wet = reduce_climatology_to_features(_const_month(290), _const_month(1e-4), _const_month(250), _const_month(80))
    dry = reduce_climatology_to_features(_const_month(290), _const_month(1e-6), _const_month(250), _const_month(80))
    assert float(wet.aridity[0]) > float(dry.aridity[0])
```

- [ ] **Step 2: Run to verify it fails** — Run (sbatch) `pytest tests/land/unit/test_climate_features.py -q`; Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# packages/land/legoesm/land/carbon/climate_features.py
from __future__ import annotations
from typing import NamedTuple
import numpy as np
from legoesm import constants

_SECONDS_PER_YEAR = 365.0 * 86400.0
_PT_ALPHA = 1.26            # Priestley-Taylor coefficient [-]
_PT_GAMMA_OVER_S = 0.5      # (gamma/(s+gamma)) ~ 0.5 mid-latitude proxy [-]
_PET_MIN = 1e-9            # coeff-ok: divide-by-zero floor on PET

class ClimateFeatures(NamedTuple):
    mat_k: np.ndarray
    map_yr: np.ndarray
    t_seasonal_amp_k: np.ndarray
    aridity: np.ndarray
    sw_mean_w: np.ndarray

def reduce_climatology_to_features(monthly_t_k, monthly_precip_rate, monthly_sw_w,
                                   monthly_netrad_w) -> ClimateFeatures:
    monthly_t_k = np.asarray(monthly_t_k); monthly_precip_rate = np.asarray(monthly_precip_rate)
    monthly_sw_w = np.asarray(monthly_sw_w); monthly_netrad_w = np.asarray(monthly_netrad_w)
    mat = monthly_t_k.mean(axis=1)
    map_yr = monthly_precip_rate.mean(axis=1) * _SECONDS_PER_YEAR
    t_seasonal = 0.5 * (monthly_t_k.max(axis=1) - monthly_t_k.min(axis=1))
    # PET [kg/m2/yr] from net radiation via Priestley-Taylor (energy/L_v -> mass).
    pet = _PT_ALPHA * _PT_GAMMA_OVER_S * np.maximum(monthly_netrad_w.mean(axis=1), 0.0) \
        / constants.L_v * _SECONDS_PER_YEAR
    aridity = map_yr / np.maximum(pet, _PET_MIN)
    sw_mean = monthly_sw_w.mean(axis=1)
    return ClimateFeatures(mat, map_yr, t_seasonal, aridity, sw_mean)
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/land/unit/test_climate_features.py -q`; Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add packages/land/legoesm/land/carbon/climate_features.py tests/land/unit/test_climate_features.py
git commit -m "feat(land): reduce monthly climatology to per-cell climate features"
```

---

## Task 3: Archetype builder (clustering)

**Files:**
- Create: `packages/land/legoesm/land/carbon/global_init.py`
- Test: `tests/land/unit/test_global_init.py`

**Interfaces:**
- Consumes: `ClimateFeatures` (Task 2).
- Produces:
  - `ArchetypeTable = NamedTuple(pft_id (n_arch,), mat_k, map_yr, t_seasonal_amp_k, aridity, sw_mean_w (each (n_arch,)), soil_class (n_arch,) object array of texture-key strings)`.
  - `build_archetypes(pft_weights (ncell,n_pft), features: ClimateFeatures, soil_class (ncell,) str, land_mask (ncell,) bool, *, k_per_pft=12, w_min=0.05, seed=0) -> (ArchetypeTable, cell_archetype_id (ncell,n_pft) int, cell_archetype_weight (ncell,n_pft) float)`. `cell_archetype_id[c,p] = -1` where PFT p has weight < `w_min` in cell c; else the archetype index. `cell_archetype_weight[c,p]` = the cover weight (0 where id=-1).

- [ ] **Step 1: Write the failing test**

```python
# tests/land/unit/test_global_init.py  (clustering portion)
import numpy as np, numpy.testing as npt
from legoesm.land.carbon.climate_features import ClimateFeatures
from legoesm.land.carbon.global_init import build_archetypes

def _feats(mat, mapyr, seas, arid, sw):
    a = lambda v: np.asarray(v, float)
    return ClimateFeatures(a(mat), a(mapyr), a(seas), a(arid), a(sw))

def test_two_pft_two_climate_makes_expected_archetypes():
    # 4 cells: PFT0 in warm+cold, PFT1 in warm+cold.
    ncell, npft = 4, 2
    w = np.zeros((ncell, npft)); w[0, 0] = w[1, 0] = 1.0; w[2, 1] = w[3, 1] = 1.0
    feats = _feats([300, 270, 300, 270], [2000]*4, [3, 12, 3, 12], [1.5]*4, [250]*4)
    soil = np.array(["loam"] * ncell)
    tab, cid, cw = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=1, w_min=0.05, seed=0)
    # k=1 per PFT -> 2 archetypes (one per PFT), each averaging its climate.
    assert tab.pft_id.shape == (2,)
    assert set(np.unique(tab.pft_id)) == {0, 1}
    # Every occupied (cell,pft) maps to a valid archetype; empties are -1.
    assert (cid[w >= 0.05] >= 0).all() and (cid[w < 0.05] == -1).all()
    # Cover weights reproduce the input cover.
    npt.assert_allclose(cw, w, rtol=0, atol=0)

def test_determinism_same_seed():
    ncell, npft = 20, 3
    rng = np.random.default_rng(1); w = rng.random((ncell, npft)); w /= w.sum(1, keepdims=True)
    feats = _feats(rng.uniform(270, 305, ncell), rng.uniform(200, 3000, ncell),
                   rng.uniform(2, 15, ncell), rng.uniform(0.2, 3, ncell), rng.uniform(120, 300, ncell))
    soil = np.array(["loam"] * ncell)
    a = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=3, seed=7)
    b = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=3, seed=7)
    npt.assert_array_equal(a[1], b[1]); npt.assert_allclose(a[0].mat_k, b[0].mat_k)
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/land/unit/test_global_init.py -q -k build_archetypes or archetypes`; Expected: FAIL (module missing).

- [ ] **Step 3: Write minimal implementation** (numpy k-means, seeded; standardise features per PFT before clustering)

```python
# packages/land/legoesm/land/carbon/global_init.py  (Task 3 portion)
from __future__ import annotations
from typing import NamedTuple
import numpy as np
from legoesm.land.carbon.climate_features import ClimateFeatures

_FEATURE_FIELDS = ("mat_k", "map_yr", "t_seasonal_amp_k", "aridity", "sw_mean_w")

class ArchetypeTable(NamedTuple):
    pft_id: np.ndarray          # (n_arch,) int
    mat_k: np.ndarray
    map_yr: np.ndarray
    t_seasonal_amp_k: np.ndarray
    aridity: np.ndarray
    sw_mean_w: np.ndarray
    soil_class: np.ndarray      # (n_arch,) str

def _kmeans(x, k, seed, n_iter=50):
    """Lloyd k-means on standardised (m, d) features; deterministic. Returns
    (labels (m,), centroids (k, d)) in standardised space."""
    rng = np.random.default_rng(seed)
    m = x.shape[0]; k = min(k, m)
    idx = rng.permutation(m)[:k]; cent = x[idx].copy()
    labels = np.zeros(m, int)
    for _ in range(n_iter):
        d = ((x[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels) and _ > 0:
            labels = new; break
        labels = new
        for j in range(k):
            sel = labels == j
            if sel.any(): cent[j] = x[sel].mean(0)
    return labels, cent

def build_archetypes(pft_weights, features, soil_class, land_mask, *,
                     k_per_pft=12, w_min=0.05, seed=0):
    pft_weights = np.asarray(pft_weights); ncell, npft = pft_weights.shape
    feat = np.stack([np.asarray(getattr(features, f)) for f in _FEATURE_FIELDS], axis=1)  # (ncell, 5)
    # Standardise globally so per-PFT clusters share a metric.
    mu = feat.mean(0); sd = feat.std(0); sd = np.where(sd < 1e-9, 1.0, sd)  # coeff-ok: std floor
    feat_std = (feat - mu) / sd
    soil_class = np.asarray(soil_class, dtype=object)
    cell_id = np.full((ncell, npft), -1, int)
    cell_w = np.where(pft_weights >= w_min, pft_weights, 0.0)
    at = {f: [] for f in ("pft_id", *_FEATURE_FIELDS, "soil_class")}
    next_arch = 0
    for p in range(npft):
        occ = np.where((pft_weights[:, p] >= w_min) & np.asarray(land_mask))[0]
        if occ.size == 0:
            continue
        labels, _ = _kmeans(feat_std[occ], k_per_pft, seed + p)
        for j in np.unique(labels):
            members = occ[labels == j]
            cell_id[members, p] = next_arch
            at["pft_id"].append(p)
            for fi, fname in enumerate(_FEATURE_FIELDS):
                at[fname].append(float(feat[members, fi].mean()))
            # modal soil texture in the cluster
            vals, cnts = np.unique(soil_class[members], return_counts=True)
            at["soil_class"].append(str(vals[cnts.argmax()]))
            next_arch += 1
    table = ArchetypeTable(
        pft_id=np.asarray(at["pft_id"], int),
        mat_k=np.asarray(at["mat_k"]), map_yr=np.asarray(at["map_yr"]),
        t_seasonal_amp_k=np.asarray(at["t_seasonal_amp_k"]),
        aridity=np.asarray(at["aridity"]), sw_mean_w=np.asarray(at["sw_mean_w"]),
        soil_class=np.asarray(at["soil_class"], dtype=object))
    return table, cell_id, cell_w
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/land/unit/test_global_init.py -q -k archetypes`; Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add packages/land/legoesm/land/carbon/global_init.py tests/land/unit/test_global_init.py
git commit -m "feat(land): archetype builder (per-PFT climate k-means)"
```

---

## Task 4: Archetype equilibration (semi-analytic, batched)

**Files:**
- Modify: `packages/land/legoesm/land/carbon/global_init.py`
- Test: `tests/land/unit/test_global_init.py` (add), `tests/land/validation/test_global_carbon_ic_map.py` (compute smoke)

**Interfaces:**
- Consumes: `ArchetypeTable` (Task 3); `map_yr → precip_rate = map_yr/seconds_per_year`; PFT params from `_CLM5_PFT_TABLE_RAW`/`PARAM_NAMES`; soil from `SOIL_TEXTURE_VG`.
- Produces: `equilibrate_archetypes(table: ArchetypeTable, *, n_spinup=200, n_verify=40, dt=3600.0, n_layers=10, soil_depth=3.0) -> (CarbonState (n_arch,), dict[str, np.ndarray])`. The dict carries per-archetype `gpp, npp, som_kgC, biomass_kgC, drift_frac_per_yr` for QC.

- [ ] **Step 1: Write the failing test** (a 2-archetype run: finite equilibria, allocation closes, small drift)

```python
# tests/land/validation/test_global_carbon_ic_map.py  (Task 4 portion)
import numpy as np
from legoesm.land.carbon.global_init import ArchetypeTable, equilibrate_archetypes
from legoesm.land.carbon.config import CarbonState

def test_equilibrate_two_archetypes():
    # PFT 4 = broadleaf_evergreen_tropical, PFT 7 = broadleaf_deciduous_temperate.
    tab = ArchetypeTable(
        pft_id=np.array([4, 7]),
        mat_k=np.array([298.0, 283.0]), map_yr=np.array([2000.0, 800.0]),
        t_seasonal_amp_k=np.array([3.0, 12.0]), aridity=np.array([2.0, 1.0]),
        sw_mean_w=np.array([230.0, 180.0]),
        soil_class=np.array(["clay_loam", "loam"], dtype=object))
    eq, qc = equilibrate_archetypes(tab, n_spinup=20, n_verify=6, dt=7200.0, n_layers=6, soil_depth=2.0)
    assert isinstance(eq, CarbonState) and eq.C_som.shape == (2,)
    assert np.all(np.isfinite(np.asarray(eq.C_som)))
    assert (np.asarray(qc["gpp"]) >= 0).all()
    # Tropical archetype fixes more C than the temperate one.
    assert float(np.asarray(eq.C_wood)[0]) > 0.0
```

- [ ] **Step 2: Run to verify fail** — sbatch `pytest tests/land/validation/test_global_carbon_ic_map.py -q -k equilibrate`; Expected: FAIL (`equilibrate_archetypes` missing).

- [ ] **Step 3: Write minimal implementation** — build one `MultiLayerLandConfig` per archetype (PFT params + soil), a climate-forcing closure per archetype, run the 3-phase semi-analytic `run_pixel`-style loop **batched over archetypes** (ncol = n_arch via the vectorized `step_multilayer_land`). Reuse `analytic_slow_pool_equilibrium` + `reconstruct_carbon_diagnostics`. Factor the shared spin-up loop so this and the validator's `run_pixel` do not duplicate it — extract `legoesm.land.carbon.spinup.run_semi_analytic_spinup(step_fn, carbon0, state0, forcing_fn, n_spinup, n_verify, steps_per_year, dt, cwd_humification_eff)` in this task and have both callers use it.

```python
# global_init.py (Task 4 portion) — key structure (pseudocode-tight)
import jax, jax.numpy as jnp
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.richards import RichardsConfig
from legoesm.land.soil_texture import SOIL_TEXTURE_VG
from legoesm.land.surface_params import CLM5_PFT_NAMES, _CLM5_PFT_TABLE_RAW, PARAM_NAMES
from legoesm.land.carbon.config import CarbonConfig, CarbonState
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.carbon.spinup import (SlowPoolFluxes, analytic_slow_pool_equilibrium,
                                        run_semi_analytic_spinup)
from legoesm.land.carbon_diagnostics import reconstruct_carbon_diagnostics
from legoesm.land.climate_forcing import make_climatological_forcing
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state, root_zone_beta_soil

_SECONDS_PER_YEAR = 365.0 * 86400.0
_U_MIN = 1.0

def _is_woody(pft_name):  # trees/shrubs woody; grass/crop herbaceous
    return not any(t in pft_name for t in ("grass", "crop"))
```

The archetype config uses per-archetype PFT/soil; because `MultiLayerLandConfig` sub-configs carry scalars, run each archetype as its own column but **batch the carbon-pool state** across archetypes and loop configs where a config array is required (or, simplest correct v1: `vmap` the single-archetype spin-up over the archetype axis with the config supplied via `land_params` arrays — reuse `LandSurfaceParams` for per-archetype Vc_max25/LCMA/root_depth/theta_wp/fc and one shared hydraulics-per-soil-class group). Document whichever path; the test above (n_arch=2) gates it. Persist `qc` from the verify-segment means.

- [ ] **Step 4: Run to verify pass** — sbatch `pytest tests/land/validation/test_global_carbon_ic_map.py -q -k equilibrate` + `tests/land/unit/test_carbon_spinup.py`; Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add packages/land/legoesm/land/carbon/global_init.py packages/land/legoesm/land/carbon/spinup.py tests/land/validation/test_global_carbon_ic_map.py
git commit -m "feat(land): batched archetype semi-analytic equilibration"
```

---

## Task 5: Global mapper (cover-weighted pool mix)

**Files:**
- Modify: `packages/land/legoesm/land/carbon/global_init.py`
- Test: `tests/land/unit/test_global_init.py` (add)

**Interfaces:**
- Consumes: `cell_archetype_id (ncell,n_pft)`, `cell_archetype_weight (ncell,n_pft)` (Task 3), `archetype_equilibria: CarbonState (n_arch,)` (Task 4).
- Produces: `map_to_grid(cell_archetype_id, cell_archetype_weight, archetype_equilibria) -> CarbonState (ncell,)`. `pools[c] = Σ_p w[c,p] · eq[id[c,p]]` (id=-1 contributes 0). Renormalise by vegetated fraction is NOT applied — bare fraction legitimately holds less carbon.

- [ ] **Step 1: Write the failing test**

```python
# tests/land/unit/test_global_init.py (mapper portion)
import numpy as np, numpy.testing as npt, jax.numpy as jnp
from legoesm.land.carbon.config import CarbonState
from legoesm.land.carbon.global_init import map_to_grid

def _cs(vals):  # vals: (n_arch,) per pool identical for simplicity
    a = lambda: jnp.asarray(vals, float)
    return CarbonState(a(), a(), a(), a(), a(), a())

def test_single_pft_cell_equals_archetype():
    eq = _cs([10.0, 20.0])
    cid = np.array([[0, -1], [1, -1]]); cw = np.array([[1.0, 0.0], [1.0, 0.0]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som), [10.0, 20.0], rtol=1e-9)

def test_mixed_cell_is_cover_weighted_mix():
    eq = _cs([10.0, 30.0])
    cid = np.array([[0, 1]]); cw = np.array([[0.25, 0.75]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som), [0.25 * 10 + 0.75 * 30], rtol=1e-9)

def test_absent_pft_contributes_zero_and_pools_nonneg():
    eq = _cs([10.0, 30.0])
    cid = np.array([[-1, 1]]); cw = np.array([[0.0, 0.5]])
    out = map_to_grid(cid, cw, eq)
    npt.assert_allclose(np.asarray(out.C_som), [0.5 * 30], rtol=1e-9)
    assert (np.asarray(out.C_som) >= 0).all()
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/land/unit/test_global_init.py -q -k map_to_grid or mapper`; Expected FAIL.
- [ ] **Step 3: Write minimal implementation**

```python
# global_init.py (Task 5 portion)
def map_to_grid(cell_archetype_id, cell_archetype_weight, archetype_equilibria):
    cid = np.asarray(cell_archetype_id); cw = np.asarray(cell_archetype_weight, float)
    safe = np.where(cid >= 0, cid, 0)                    # gather index (masked below)
    mask = (cid >= 0).astype(float) * cw                 # (ncell, n_pft)
    fields = {}
    for f in archetype_equilibria._fields:
        vals = np.asarray(getattr(archetype_equilibria, f))    # (n_arch,)
        gathered = vals[safe]                                   # (ncell, n_pft)
        fields[f] = jnp.asarray((gathered * mask).sum(axis=1))  # (ncell,)
    return archetype_equilibria.__class__(**fields)
```

- [ ] **Step 4: Run to verify pass** — `pytest tests/land/unit/test_global_init.py -q`; Expected PASS.
- [ ] **Step 5: Commit**

```bash
git add packages/land/legoesm/land/carbon/global_init.py tests/land/unit/test_global_init.py
git commit -m "feat(land): cover-weighted map of archetype equilibria to the grid"
```

---

## Task 6: Driver — build the global finidat from real data

**Files:**
- Create: `scripts/data/build_global_carbon_ic.py`
- Modify: `tests/land/unit/test_global_init.py` (add a driver-on-synthetic-arrays test)

**Interfaces:**
- Consumes: Tasks 2–5. Loads CLM5 PFT weights (`land.surface_data.sources.clm5_surfdata`), ERA5/AMIP monthly climatology (`tools.forcing.amip`), HWSD soil texture class (`land.surface_data.sources.hwsd2`), regridded to a common `(ncell,)` land vector at the target resolution.
- Produces: writes `<out>/global_carbon_ic.npz` (per-pool `(ncell,)` + lat/lon + PFT ids) and `<out>/archetypes.npz` (the `ArchetypeTable` + equilibria + QC). CLI: `--resolution-deg 1.0 --k-per-pft 12 --n-spinup 200 --n-verify 40 --dt 3600 --output results/global_carbon_ic`.

- [ ] **Step 1: Write the failing test** (drive the *core* on synthetic arrays end-to-end — no data files)

```python
# tests/land/unit/test_global_init.py (driver-core portion)
import numpy as np
from legoesm.land.carbon.climate_features import reduce_climatology_to_features
from legoesm.land.carbon.global_init import build_archetypes, equilibrate_archetypes, map_to_grid

def test_core_pipeline_synthetic_world():
    # 3 cells, 2 PFTs (tropical=4, temperate=7), simple climatology.
    ncell, npft = 3, 17
    w = np.zeros((ncell, npft)); w[0, 4] = 1.0; w[1, 7] = 1.0; w[2, 4] = 0.5; w[2, 7] = 0.5
    t = np.stack([np.full(12, 298.0), np.full(12, 283.0), np.full(12, 290.0)])
    pr = np.stack([np.full(12, 6e-5), np.full(12, 2.5e-5), np.full(12, 4e-5)])
    sw = np.full((ncell, 12), 220.0); nr = np.full((ncell, 12), 90.0)
    feats = reduce_climatology_to_features(t, pr, sw, nr)
    soil = np.array(["clay_loam", "loam", "loam"])
    tab, cid, cw = build_archetypes(w, feats, soil, np.ones(ncell, bool), k_per_pft=1, seed=0)
    eq, qc = equilibrate_archetypes(tab, n_spinup=15, n_verify=5, dt=7200.0, n_layers=6, soil_depth=2.0)
    grid = map_to_grid(cid, cw, eq)
    assert grid.C_som.shape == (ncell,)
    assert np.all(np.isfinite(np.asarray(grid.C_som)))
    # Mixed cell 2 is between the two pure cells for total ecosystem C.
    tot = lambda i: sum(float(np.asarray(getattr(grid, f))[i]) for f in grid._fields)
    assert min(tot(0), tot(1)) - 1.0 <= tot(2) <= max(tot(0), tot(1)) + 1.0
```

- [ ] **Step 2: Run to verify fail** — sbatch `pytest tests/land/unit/test_global_init.py -q -k core_pipeline`; Expected FAIL until Tasks 3–5 exist (they do) — this test passes once they're in, so it actually gates the *integration*; keep it.
- [ ] **Step 3: Write the driver** — a `main(argv)` that: loads the three real maps + regrids to `(ncell,)`; `reduce_climatology_to_features`; `build_archetypes`; `equilibrate_archetypes`; `map_to_grid`; writes the two `.npz`. Guard each loader import at function scope. Print a QC summary (n_archetypes, per-biome SOC/biomass ranges). Provide `--dry-run-synthetic` that fabricates a tiny world so the driver is testable without data files.
- [ ] **Step 4: Run to verify pass** — sbatch `pytest tests/land/unit/test_global_init.py -q` and `python scripts/data/build_global_carbon_ic.py --dry-run-synthetic --output /tmp/gcic` (writes npz).
- [ ] **Step 5: Commit**

```bash
git add scripts/data/build_global_carbon_ic.py tests/land/unit/test_global_init.py
git commit -m "feat(data): global carbon IC driver (archetypes -> finidat)"
```

---

## Task 7: Validator — drift→0 on a cell sample

**Files:**
- Create: `scripts/validate/global_carbon_ic_map.py`
- Modify: `tests/land/validation/test_global_carbon_ic_map.py` (add)

**Interfaces:**
- Consumes: the finidat `.npz` (Task 6), the per-cell configs (rebuilt from PFT+soil+climate).
- Produces: `assess_ic_map(finidat, sample_idx, n_years, dt) -> dict` with per-pool `drift_frac_per_yr`, `cold_start_drift_frac_per_yr` (control), coverage stats. CLI writes a Markdown report + JSON.

- [ ] **Step 1: Write the failing test** (pure-logic drift metric on synthetic trajectories — no model)

```python
# tests/land/validation/test_global_carbon_ic_map.py (validator-logic portion)
import numpy as np
import importlib.util, sys
from pathlib import Path
_V = Path(__file__).resolve().parents[3] / "scripts" / "validate" / "global_carbon_ic_map.py"
def _load():
    s = importlib.util.spec_from_file_location("gcicv", _V); m = importlib.util.module_from_spec(s)
    sys.modules["gcicv"] = m; s.loader.exec_module(m); return m

def test_drift_metric_zero_for_flat_series():
    v = _load()
    flat = np.ones((5,)) * 100.0
    assert abs(v.drift_frac_per_yr(flat)) < 1e-9

def test_drift_metric_detects_decline():
    v = _load()
    series = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
    assert v.drift_frac_per_yr(series) < 0.0
```

- [ ] **Step 2: Run to verify fail** — sbatch `pytest tests/land/validation/test_global_carbon_ic_map.py -q -k drift_metric`; Expected FAIL.
- [ ] **Step 3: Write the validator** — `drift_frac_per_yr(series)` = `(series[-1]-series[0])/(len-1)/max(|series[-1]|,eps)`; `assess_ic_map` draws a stratified sample (by PFT/climate bin), rebuilds each sampled cell's config, runs the coupled land+carbon `n_years` from (a) the mapped IC and (b) a cold IC (`init_carbon_state` defaults), reports both drifts + coverage; `main` writes report.md + JSON + a bar figure (mapped vs cold drift). Reuse the batched column step (ncol = sample size).
- [ ] **Step 4: Run to verify pass** — sbatch `pytest tests/land/validation/test_global_carbon_ic_map.py -q`; then a compute smoke: `python scripts/validate/global_carbon_ic_map.py --finidat /tmp/gcic/global_carbon_ic.npz --sample 8 --years 3` and eyeball mapped-drift ≪ cold-drift.
- [ ] **Step 5: Commit**

```bash
git add scripts/validate/global_carbon_ic_map.py tests/land/validation/test_global_carbon_ic_map.py
git commit -m "feat(validate): global carbon IC map drift/realism validator"
```

---

## Task 8: Full-scale 1° build + codex review + docs

**Files:** Modify `docs/land/carbon_equilibrium_audit.md` (link the global-init capability); no new source.

- [ ] **Step 1:** sbatch the real 1° build (`scripts/data/build_global_carbon_ic.py --resolution-deg 1.0 --output results/global_carbon_ic`) on a compute node; confirm it writes the finidat + archetype QC and the per-biome SOC/biomass ranges look sane.
- [ ] **Step 2:** sbatch the validator on the real finidat (stratified sample ~200 cells, 5 yr); confirm mapped-drift ≪ cold-drift (the map IS at equilibrium). Send the drift figure to the user.
- [ ] **Step 3:** Run the guardrail ratchets (`tests/test_no_inline_physics_coeffs.py`, `tests/test_no_hardcoded_constants.py`, `tests/test_param_specs.py`) — green.
- [ ] **Step 4:** codex adversarial review (`codex exec --sandbox read-only "..." < /dev/null`) on `global_init.py`, `climate_forcing.py`, `climate_features.py`, the spin-up refactor; fix findings; re-review until clean.
- [ ] **Step 5:** Commit docs; push branch `land/carbon-global-init`; open a PR (base main) titled "feat(land): global carbon IC map (Stage A)". Then Stage B (modular training) gets its own spec.

---

## Self-Review

- **Spec coverage:** archetype builder (T3), climate axis/features (T2), climate forcing (T1), equilibration reuse (T4), cover-weighted mapper (T5), driver/finidat (T6), drift/realism validator (T7), full 1° + codex + PR (T8). Resolution-agnostic — all core fns take `(ncell,…)`. Stage B explicitly deferred. ✓
- **Placeholders:** the one soft spot is Task 4's per-archetype-config batching (config-loop vs `land_params` `vmap`); the plan names both concrete paths and the n_arch=2 gating test — the implementer picks one and the test enforces correctness. No "TODO/handle edge cases".
- **Type consistency:** `ArchetypeTable` fields (T3) consumed verbatim in T4/T6; `CarbonState (n_arch,)` from T4 consumed by `map_to_grid` (T5); `cell_archetype_id/weight (ncell,n_pft)` consistent T3→T5; `ClimateFeatures` fields consistent T2→T3. `run_semi_analytic_spinup` introduced in T4 and reused by the validator's run_pixel — added to `spinup.py`. ✓

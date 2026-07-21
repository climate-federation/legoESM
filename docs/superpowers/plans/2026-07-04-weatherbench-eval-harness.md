# WeatherBench Forecast Campaign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the net-new WeatherBench-2 forecast-scoring harness (Stage 0) that turns a trained legoESM spectral dycore into a WB2 lead-time scorecard (RMSE + ACC + bias on headline pressure-level + surface variables vs ERA5), then run the swap-and-train campaign (Stages 1–3) on top of it.

**Architecture:** Reuse the AIMIP spectral trainer (`_train_spectral_loop`), ERA5/WB2 Zarr ingestion (`era5_to_state.py`), spectral rollout (`spectral_rollout`), and the correct-but-orphaned metric/plot primitives in `evaluations/`. Add (a) pure-JAX headline diagnostics (Z500, T850, Q700, U/V@level, MSLP, T2m, 10m wind, precip) that map a rolled-out `SpectralHydrostaticState` to WB2 pressure-level/surface fields, (b) a scorer that inits from ERA5, rolls out to each lead, regrids to the WB2 1.5° common grid, and scores RMSE/ACC/bias, (c) baselines + SOTA overlay, (d) a SLURM-launched CLI.

**Tech Stack:** Python 3, JAX (x64 for spectral), Equinox, xarray/gcsfs (ERA5+climatology Zarr), scipy (offline regrid), matplotlib.

## Global Constraints

- **Login-node policy (HARD):** all `pytest` / JAX-compiling / `mpirun` / training commands run ONLY on a compute node via `srun --account=glab --pty ...` or `sbatch --account=glab`. The login node is for `git`, `grep`, `Read`, config edits only. Every "Run: pytest ..." step below is implicitly `srun --account=glab --time=0:20:00 --pty env JAX_ENABLE_X64=1 .venv/bin/python -m pytest ...`.
- **x64:** spectral + all headline diagnostics use `JAX_ENABLE_X64=1` (geopotential/log-p integration needs float64).
- **Constants/thermo:** import from `legoesm.constants` (`g`, `R_d`, `R_v`, `epsilon`, `T_freeze`, `kappa`) and `legoesm.thermo`; virtual temperature from `atmosphere.physics._shared`. NO literals `9.80616/287.0/461.51/0.622/273.15`.
- **No duplicate numerics:** reuse `evaluations/metrics.py` (rmse/acc/bias/compute_scorecard), `evaluations/visualize.py` (plotters), `grids/vertical.py::compute_geopotential`, `core/bulk_flux.py` (`psi_m`/`psi_h`/`compute_most_fluxes`). Do NOT re-derive.
- **Every new `.py` gets a direct unit test.** Factory/dispatch on unknown → `raise`.
- **Sign-convention gate:** every hydrostatic/reduction term carries a coordinate-convention comment and a sign walk (Φ up; MSLP ≥ p_s for land above sea level).
- **Codex adversarial review MANDATORY** before declaring any stage done (`/codex:adversarial-review --wait` → fix → `/codex:review --wait` → repeat to clean or 30 iter). Codex companion crashes here → use `codex exec --sandbox read-only`.
- **Worktree:** all campaign code lands in a pinned worktree off `main` (`legoESM_wbforecast`, branch `wb-forecast-campaign`) created via `superpowers:using-git-worktrees` at execution start — avoids the shared-checkout long-run hazard. Commit + push every task.
- **Home for the harness:** the top-level `evaluations/` package (canonical WB2 home; `metrics.py`/`visualize.py`/`weatherbench.py` already live there). CLI drivers go in `scripts/validate/`, sbatch wrappers in `scripts/cluster/`.

---

## STAGE 0 — WeatherBench eval harness (net-new foundation)

### Task 1: Log-pressure interpolation to a target level

**Files:**
- Create: `evaluations/headline_diagnostics.py`
- Test: `tests/evaluations/test_headline_diagnostics.py`

**Interfaces:**
- Produces: `interp_to_pressure_level(field, p_model, p_target_pa) -> jax.Array` — `field` shape `(..., nlev)`, `p_model` shape `(..., nlev)` (Pa, monotonically increasing downward or upward), `p_target_pa` scalar float. Returns shape `(...)`. Linear in `ln(p)`; extrapolation clamped to the nearest level (no NaN). Differentiable.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluations/test_headline_diagnostics.py
import jax.numpy as jnp
import numpy as np
from evaluations.headline_diagnostics import interp_to_pressure_level

def test_interp_logp_recovers_linear_in_lnp():
    # field that is exactly linear in ln(p): f = a + b ln(p)
    p = jnp.array([1000.0, 700.0, 500.0, 300.0, 100.0]) * 100.0  # Pa
    a, b = 250.0, -7.5
    f = a + b * jnp.log(p)
    got = interp_to_pressure_level(f, p, 50000.0)  # 500 hPa
    expected = a + b * np.log(50000.0)
    assert np.isclose(float(got), float(expected), atol=1e-9)

def test_interp_clamps_above_top():
    p = jnp.array([1000.0, 500.0, 100.0]) * 100.0
    f = jnp.array([10.0, 5.0, 1.0])
    # target below the top level pressure (50 hPa) -> clamp to top value
    got = interp_to_pressure_level(f, p, 5000.0)
    assert np.isfinite(float(got))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluations/test_headline_diagnostics.py::test_interp_logp_recovers_linear_in_lnp -v`
Expected: FAIL — `ImportError: cannot import name 'interp_to_pressure_level'`.

- [ ] **Step 3: Write minimal implementation**

```python
# evaluations/headline_diagnostics.py
"""WeatherBench-2 headline-variable diagnostics from model state.

Pure-JAX, differentiable. Coordinate convention: geopotential Φ increases
UPWARD; pressure DECREASES upward. All pressures in Pa, all temperatures in K.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from legoesm import constants


def interp_to_pressure_level(field, p_model, p_target_pa):
    """Linear-in-ln(p) interpolation of `field` to `p_target_pa`.

    field, p_model: (..., nlev). Extrapolation is clamped to the nearest
    bounding level (no NaN); p_model need not be pre-sorted along nlev.
    """
    lnp = jnp.log(p_model)
    lnpt = jnp.log(jnp.asarray(p_target_pa, dtype=lnp.dtype))
    # distance in ln(p); pick the two levels bracketing the target
    # Use searchsorted-free weighting so it stays vmap/grad friendly.
    below = lnp <= lnpt            # levels at higher pressure than target
    above = lnp > lnpt
    # nearest level below (largest lnp among `below`) and above (smallest lnp among `above`)
    neg_inf = jnp.array(-jnp.inf, dtype=lnp.dtype)
    pos_inf = jnp.array(jnp.inf, dtype=lnp.dtype)
    lnp_lo = jnp.max(jnp.where(below, lnp, neg_inf), axis=-1)
    lnp_hi = jnp.min(jnp.where(above, lnp, pos_inf), axis=-1)
    f_lo = jnp.take_along_axis(field, jnp.argmax(jnp.where(below, lnp, neg_inf), axis=-1)[..., None], axis=-1)[..., 0]
    f_hi = jnp.take_along_axis(field, jnp.argmin(jnp.where(above, lnp, pos_inf), axis=-1)[..., None], axis=-1)[..., 0]
    have_both = jnp.isfinite(lnp_lo) & jnp.isfinite(lnp_hi)
    w = jnp.where(have_both, (lnpt - lnp_lo) / jnp.where(lnp_hi != lnp_lo, lnp_hi - lnp_lo, 1.0), 0.0)
    interp = f_lo + w * (f_hi - f_lo)
    # clamp: if target above top -> f at min-pressure level; if below bottom -> f at max-pressure level
    f_top = jnp.take_along_axis(field, jnp.argmin(p_model, axis=-1)[..., None], axis=-1)[..., 0]
    f_bot = jnp.take_along_axis(field, jnp.argmax(p_model, axis=-1)[..., None], axis=-1)[..., 0]
    only_above = (~jnp.isfinite(lnp_lo)) & jnp.isfinite(lnp_hi)  # target above top
    only_below = jnp.isfinite(lnp_lo) & (~jnp.isfinite(lnp_hi))  # target below bottom
    out = jnp.where(have_both, interp, jnp.where(only_above, f_top, jnp.where(only_below, f_bot, f_lo)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluations/test_headline_diagnostics.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add evaluations/headline_diagnostics.py tests/evaluations/test_headline_diagnostics.py
git commit -m "feat(wb): log-p interpolation to pressure level for headline diagnostics"
```

---

### Task 2: Geopotential height at a pressure level (Z500)

**Files:**
- Modify: `evaluations/headline_diagnostics.py`
- Modify: `tests/evaluations/test_headline_diagnostics.py`

**Interfaces:**
- Consumes: `compute_geopotential(T, p_s, sigma_coord, phis)` (`grids/vertical.py:190`, Simmons-Burridge, returns Φ at full levels `(..., nlev)`); virtual-temperature helper from `atmosphere.physics._shared`; `interp_to_pressure_level` (Task 1).
- Produces: `geopotential_height_at(T, q, p_s, phis, sigma_coord, p_target_pa) -> jax.Array` — Z (geopotential height, metres) at the target level, shape `(...)`. Uses virtual temperature so Z500 matches ERA5.

- [ ] **Step 1: Write the failing test**

```python
def test_z500_isothermal_atmosphere():
    # Isothermal T_v = 250 K, flat surface (phis=0, p_s=1000 hPa).
    # Analytic: z(p) = (R_d T / g) ln(p_s / p). At 500 hPa:
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.grids.vertical import SigmaCoordinate  # confirm import path at build
    from evaluations.headline_diagnostics import geopotential_height_at
    nlev = 20
    sigma = SigmaCoordinate.uniform(nlev)  # confirm constructor at build time
    T = jnp.full((4, 8, nlev), 250.0)
    q = jnp.zeros((4, 8, nlev))
    p_s = jnp.full((4, 8), 100000.0)
    phis = jnp.zeros((4, 8))
    z500 = geopotential_height_at(T, q, p_s, phis, sigma, 50000.0)
    expected = (constants.R_d * 250.0 / constants.g) * jnp.log(100000.0 / 50000.0)
    assert float(jnp.max(jnp.abs(z500 - expected))) < 25.0  # within 25 m (discretisation)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluations/test_headline_diagnostics.py::test_z500_isothermal_atmosphere -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write minimal implementation** (append to `headline_diagnostics.py`)

```python
def _virtual_temperature(T, q):
    # T_v = T (1 + (R_v/R_d - 1) q) = T (1 + (1/epsilon - 1) q)
    return T * (1.0 + (1.0 / constants.epsilon - 1.0) * q)

def geopotential_height_at(T, q, p_s, phis, sigma_coord, p_target_pa):
    """Geopotential HEIGHT [m] at p_target_pa. Convention: Φ increases upward."""
    from legoesm.grids.vertical import compute_geopotential
    T_v = _virtual_temperature(T, q)
    phi = compute_geopotential(T_v, p_s, sigma_coord, phis)          # (..., nlev), J/kg
    p_model = sigma_coord.full_levels * p_s[..., None]               # confirm attr name at build
    phi_at = interp_to_pressure_level(phi, p_model, p_target_pa)
    return phi_at / constants.g                                      # geopotential height [m]
```

Build-time note: confirm `SigmaCoordinate.full_levels` (or `sigma_full`) attribute + `.uniform` constructor by reading `grids/vertical.py`; adjust `p_model` accordingly (pure-sigma `σ·p_s`; if hybrid, use `a_k + b_k·p_s`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluations/test_headline_diagnostics.py::test_z500_isothermal_atmosphere -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluations/headline_diagnostics.py tests/evaluations/test_headline_diagnostics.py
git commit -m "feat(wb): geopotential height at pressure level (Z500) via Simmons-Burridge + virtual T"
```

---

### Task 3: Mean sea-level pressure reduction

**Files:**
- Modify: `evaluations/headline_diagnostics.py`, `tests/evaluations/test_headline_diagnostics.py`

**Interfaces:**
- Produces: `mean_sea_level_pressure(p_s, T_lowest, phis, *, lapse_rate_k_per_m=0.0065) -> jax.Array` — MSLP [Pa], shape `(...)`. Standard hydrostatic reduction with a fixed lapse rate; `z_s = phis/g`. Convention: MSLP ≥ p_s where z_s > 0.

- [ ] **Step 1: Write the failing test**

```python
def test_mslp_reduces_upward_and_identity_at_sea_level():
    import jax.numpy as jnp
    from evaluations.headline_diagnostics import mean_sea_level_pressure
    from legoesm import constants
    p_s = jnp.array([90000.0, 100000.0])
    T_low = jnp.array([280.0, 288.0])
    phis = jnp.array([1000.0 * constants.g, 0.0])   # 1000 m, and sea level
    mslp = mean_sea_level_pressure(p_s, T_low, phis)
    assert float(mslp[1]) == float(p_s[1])          # sea level -> identity
    assert float(mslp[0]) > float(p_s[0])           # elevated -> reduced up to higher p
```

- [ ] **Step 2: Run** → FAIL (ImportError).

- [ ] **Step 3: Implement** (append)

```python
def mean_sea_level_pressure(p_s, T_lowest, phis, *, lapse_rate_k_per_m=0.0065):
    """MSLP [Pa]. p_msl = p_s * (1 + Γ z_s / T0)^(g/(R_d Γ)), T0 = T_lowest + Γ z_s/2.
    Convention: z_s = phis/g (m); reduction raises pressure for z_s>0."""
    z_s = phis / constants.g
    gamma = lapse_rate_k_per_m
    T0 = T_lowest + 0.5 * gamma * z_s               # mean-layer temperature
    exponent = constants.g / (constants.R_d * gamma)
    ratio = 1.0 + gamma * z_s / jnp.maximum(T0, 1.0)
    return p_s * ratio ** exponent
```

Sign walk (required): `z_s>0` → `ratio>1` → `p_msl>p_s` ✓; `z_s=0` → `ratio=1` → identity ✓. `lapse_rate_k_per_m` is a documented published constant (ICAO standard 6.5 K/km) — keep as a kwarg default Name, not an inline literal in a body (satisfies the inline-coeff ratchet; add `# coeff-ok: ICAO standard atmosphere lapse rate` if flagged).

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(wb): mean sea-level pressure reduction`.

---

### Task 4: Screen-level T2m and 10 m wind (MOST)

**Files:**
- Modify: `evaluations/headline_diagnostics.py`, `tests/evaluations/test_headline_diagnostics.py`

**Interfaces:**
- Consumes: `core/bulk_flux.py::psi_m(zeta)`, `psi_h(zeta)`, `compute_most_fluxes(...)` (read the exact signature at build time — Step 0).
- Produces: `screen_level_t2m(T_sfc, T_lowest, z_lowest, obukhov_L, z0h, *, z_ref=2.0) -> jax.Array`; `wind_10m(u_lowest, v_lowest, z_lowest, obukhov_L, z0m, *, z_ref=10.0) -> jax.Array`. Standard MOST interpolation between the surface and the lowest model level.

- [ ] **Step 0 (read):** `grep -n "def compute_most_fluxes\|def psi_m\|def psi_h" packages/core/legoesm/core/bulk_flux.py` and read lines 147–260 to confirm the stability-function signatures and what `compute_most_fluxes` returns (u_star, theta_star, L). Reuse those; do not re-implement psi_m/psi_h.

- [ ] **Step 1: Write the failing test**

```python
def test_t2m_between_surface_and_lowest_level_neutral():
    import jax.numpy as jnp
    from evaluations.headline_diagnostics import screen_level_t2m
    # Neutral (L -> inf): log interpolation. 2 m value between T_sfc and T_lowest.
    T_sfc = jnp.array([300.0]); T_low = jnp.array([295.0])
    z_low = jnp.array([40.0]); L = jnp.array([1e12]); z0h = jnp.array([0.01])
    t2 = screen_level_t2m(T_sfc, T_low, z_low, L, z0h)
    assert float(T_low[0]) < float(t2[0]) < float(T_sfc[0])
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** (append; neutral-log baseline + MOST stability correction using the reused `psi_h`/`psi_m`)

```python
def _log_frac(z_ref, z_low, z0, psi_fn, L):
    from legoesm.core.bulk_flux import psi_m, psi_h  # noqa: F401 (selected below)
    num = jnp.log(z_ref / z0) - psi_fn(z_ref / L)
    den = jnp.log(z_low / z0) - psi_fn(z_low / L)
    return num / jnp.maximum(den, 1e-6)

def screen_level_t2m(T_sfc, T_lowest, z_lowest, obukhov_L, z0h, *, z_ref=2.0):
    """T at z_ref via MOST: T2 = T_sfc + (T_low - T_sfc) * f_h. Convention: z up."""
    from legoesm.core.bulk_flux import psi_h
    f = _log_frac(z_ref, z_lowest, z0h, psi_h, obukhov_L)
    return T_sfc + (T_lowest - T_sfc) * f

def wind_10m(u_lowest, v_lowest, z_lowest, obukhov_L, z0m, *, z_ref=10.0):
    from legoesm.core.bulk_flux import psi_m
    f = _log_frac(z_ref, z_lowest, z0m, psi_m, obukhov_L)
    return jnp.sqrt(u_lowest**2 + v_lowest**2) * f
```

Build-time note: confirm `psi_m`/`psi_h` sign convention (some codes define ψ with the opposite sign); the test above only checks monotonic bracketing — add a stable/unstable sign-consistency assert once the convention is confirmed against `compute_most_fluxes`.

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(wb): MOST screen-level T2m + 10m wind diagnostics`.

---

### Task 5: Assemble headline fields from a rolled-out spectral state

**Files:**
- Create: `evaluations/wb_forecast.py`
- Test: `tests/evaluations/test_wb_forecast.py`

**Interfaces:**
- Consumes: the grid-space extraction of `(T, u, v, q_v, p_s, phis, T_sfc, z0, L)` from a `SpectralHydrostaticState` — reuse the exact extraction used by `run_aimip.py::_evaluate_variant` / `neural_gcm_spectral._spectral_state_loss_components` (Step 0: read those to find the state→grid accessor). Task 1–4 diagnostics.
- Produces: `diagnose_headline_fields(state, grid, sigma_coord) -> dict[str, jax.Array]` keyed `"z500","t850","q700","u850","v850","u250","v250","t2m","u10","v10","mslp"`, each a `(nlat, nlon)` array on the model grid.

- [ ] **Step 1: Write the failing test** — build a tiny synthetic `SpectralHydrostaticState` (or the minimal duck-typed struct the accessor needs) at T21/8lev, call `diagnose_headline_fields`, assert the dict has all keys and finite `(nlat,nlon)` arrays with physical ranges (`z500` ∈ [4500, 6200] m for the isothermal-ish column, `mslp` ∈ [90000, 105000] Pa).

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** `diagnose_headline_fields` — extract grid fields via the reused accessor, then call `geopotential_height_at(..., 50000.0)`, `interp_to_pressure_level` for T850/Q700/U,V@levels, `mean_sea_level_pressure`, `screen_level_t2m`, `wind_10m`. Assemble the dict.

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(wb): assemble WB2 headline fields from spectral state`.

---

### Task 6: Regrid model grid → WB2 1.5° common grid

**Files:**
- Modify: `evaluations/wb_forecast.py`, `tests/evaluations/test_wb_forecast.py`

**Interfaces:**
- Produces: `regrid_to_wb2(field, src_lat, src_lon, *, conservative=False) -> (np.ndarray, wb2_lat, wb2_lon)` — bilinear (or conservative for precip) to the WB2 1.5° grid (121×240). Offline numpy/scipy (eval is not JIT'd); reuse the scipy `RegularGridInterpolator` pattern already in `era5_to_state.regrid_latlon_to_gaussian`.

- [ ] **Step 1: Test** — regrid a constant field → constant preserved; regrid a smooth `cos(lat)` field → max error < 1e-3; output shape `(121, 240)`.
- [ ] **Step 2: Run** → FAIL. **Step 3:** implement (factor the interpolation helper from `era5_to_state`; do not duplicate). **Step 4:** PASS. **Step 5:** commit `feat(wb): regrid model fields to WB2 1.5° common grid`.

---

### Task 7: Score forecast (RMSE + ACC + bias) into a WB2 scorecard

**Files:**
- Modify: `evaluations/wb_forecast.py`, `tests/evaluations/test_wb_forecast.py`

**Interfaces:**
- Consumes: `evaluations/metrics.py::rmse(pred, target, weights, mask=None)`, `acc(pred, target, climatology, weights)`, `bias(pred, target, weights)`, `compute_scorecard`.
- Produces: `score_forecast(pred_fields, verif_fields, clim_fields, wb2_lat_weights) -> dict[(var, lead) -> {"rmse","acc","bias"}]`. Wires the ACC that `weatherbench.py` currently loads-then-ignores.

- [ ] **Step 1: Test** — identical pred==verif → RMSE 0, bias 0, ACC 1 (given non-degenerate anomalies); pred==climatology while verif≠clim → ACC ≈ 0. Assert against hand-computed values.
- [ ] **Step 2:** FAIL. **Step 3:** implement calling the metric primitives with cosine-latitude weights for the 1.5° grid. **Step 4:** PASS. **Step 5:** commit `feat(wb): WB2 RMSE+ACC+bias scorer (wire ACC)`.

---

### Task 8: Baselines — climatology, persistence, SOTA table

**Files:**
- Create: `evaluations/baselines.py`, `tests/evaluations/test_baselines.py`
- Create: `evaluations/data/wb2_sota_headline.csv` (committed small table of published WB2 headline RMSE at 1/3/5/7/10 d for IFS-HRES, GraphCast, Pangu, GenCast, NeuralGCM, climatology, persistence)

**Interfaces:**
- Produces: `climatology_forecast(clim_ds, init_date, leads)`, `persistence_forecast(ic_fields, leads)`, `load_sota_headline(csv_path) -> dict[model -> dict[(var,lead) -> rmse]]`.

- [ ] **Step 1: Test** — persistence at lead 0 == IC; SOTA CSV loads with expected models+leads; climatology forecast is date-independent in lead. **Step 2:** FAIL. **Step 3:** implement (persistence trivial; climatology from WB2 climatology Zarr already loaded by `era5_loader.create_climatology_dataset`; CSV parse). **Step 4:** PASS. **Step 5:** commit `feat(wb): climatology + persistence baselines + WB2 SOTA table`.

---

### Task 9: Scorer orchestrator (ERA5 IC → rollout → score, averaged over init dates)

**Files:**
- Modify: `evaluations/wb_forecast.py`, `tests/evaluations/test_wb_forecast.py`

**Interfaces:**
- Consumes: `era5_to_state.load_era5_ic` + the spectral IC builder used by `run_aimip._evaluate_variant` (Step 0: reuse, don't rebuild); `spectral_rollout(initial_state, physics_fn, grid, sigma_coord, pe_config, dt, n_steps, ...)`; Tasks 5–7.
- Produces: `run_wb_forecast_eval(physics_fn, grid, sigma_coord, pe_config, dt, init_dates, leads_hours, era5_cfg, clim_cfg, out_dir) -> scorecard_dict` — for each init date: build IC from ERA5, roll out to each lead (accumulate rollout so leads are cumulative steps, not restarts), diagnose headline fields, regrid, score vs ERA5 verification at the valid time; average metrics over init dates; write `wb_scorecard.json`.

- [ ] **Step 1: Test (T21 smoke, tiny)** — 1 init date, leads `(6, 24)`, an untrained default physics_fn; assert the returned scorecard has all headline `(var, lead)` keys with finite RMSE and ACC∈[-1,1], and `wb_scorecard.json` is written. Mark `@pytest.mark.slow`; runs under `srun` (compiles the spectral rollout).
- [ ] **Step 2:** FAIL. **Step 3:** implement the init-date loop reusing the AIMIP IC builder + `spectral_rollout`; leads are cumulative (`n_steps = lead_hours*3600/dt`). **Step 4:** run under `srun --account=glab --time=0:30:00`. PASS. **Step 5:** commit `feat(wb): WB2 forecast scorer orchestrator (ERA5 IC -> rollout -> scorecard)`.

---

### Task 10: CLI + SLURM wrapper

**Files:**
- Create: `scripts/validate/run_weatherbench_eval.py`
- Create: `scripts/cluster/wb_forecast/wb_eval.sbatch`
- Test: `tests/unit/test_run_weatherbench_eval_cli.py`

**Interfaces:**
- Produces: CLI `run_weatherbench_eval.py --checkpoint <epoch.eqx> --config <wb_era5.yaml> --test-year 2020 --init-cadence-hours 12 --leads 6,24,72,120,240 --out <dir> [--baselines] [--sota evaluations/data/wb2_sota_headline.csv]`. Loads the trained physics_fn from the checkpoint (reuse the AIMIP checkpoint loader), calls `run_wb_forecast_eval`, then `visualize.plot_rmse_vs_leadtime` / `plot_acc_vs_leadtime` / `plot_scorecard` with the SOTA overlay.

- [ ] **Step 1: Test** — argparse round-trip: `build_eval_config_from_args(["--checkpoint","x.eqx","--config","c.yaml","--leads","6,24"])` returns leads `(6,24)`, test_year 2020 default; unknown `--leads abc` → `SystemExit`. **Step 2:** FAIL. **Step 3:** implement argparse + `main()` wiring; sbatch requests 1 GPU, `--account=glab`, `JAX_ENABLE_X64=1`, `PYTHONPATH=packages/*`. **Step 4:** PASS (CLI-parse test runs on login node, <5 s, no JAX import at module top — guard the heavy imports inside `main`). **Step 5:** commit `feat(wb): WeatherBench eval CLI + SLURM wrapper`.

---

### Task 11: Stage-0 end-to-end smoke + codex gate

- [ ] **Step 1:** `sbatch scripts/cluster/wb_forecast/wb_eval.sbatch` with an existing AIMIP T21/T63 checkpoint (from a prior ACE2 run) + `--baselines`; confirm it writes `wb_scorecard.json` + PNGs with finite RMSE/ACC curves and the SOTA overlay renders.
- [ ] **Step 2:** Regenerate + send the RMSE-vs-lead PNG (standing preference).
- [ ] **Step 3:** Codex adversarial review of the whole `evaluations/` diff (`codex exec --sandbox read-only` per the local-crash workaround) → fix flagged → re-review to clean or 30 iter. Report verdict.
- [ ] **Step 4:** Commit + push; open/refresh the campaign PR.

**Stage-0 deliverable:** a working, tested WB2 scorer that turns any trained legoESM spectral checkpoint into a headline RMSE+ACC scorecard vs ERA5 with a SOTA overlay. This is independently useful software.

---

## STAGE 1 — OAT swap + train (T63) — *detailed tasks authored after Stage 0 lands*

Depends on Stage-0's finalized `run_wb_forecast_eval` API + the checkpoint loader. Task outline (each becomes a full TDD/orchestration task in the follow-on plan):

- **T1.1** `config/wb/wb_era5.yaml` base (T63, dycore, ERA5 train/eval windows, multi-step forecast loss block `multi_step_hours:[6,12,18,24]`, MSE+CRPS weights, `residual_normalize`). Test: config loads + `validate_strict` passes.
- **T1.2** Per-family `combo_*/` overlays (13 arms + 5 `none` controls) generated by a `run_wb_sweep_stage1.py` cloned from `run_aimip_classical_sweep_stage1.py`. Test: every generated `(scheme, grid)` is a real matrix case (catalog-backed, per CLAUDE.md).
- **T1.3** Wire the Stage-0 scorer as the per-arm evaluation in the sweep runner (replace the single-lead AIMIP scorer). Test: runner emits a per-arm `wb_scorecard.json`.
- **T1.4** SLURM array launch (`_wb_sweep_stage1_runner.sbatch`); aggregate `wb_sweep_scorecard.{json,png}` (clone `aimip_sweep_scorecard.py`); pick winner/family by composite headline RMSE+ACC at 1/3/5 d.
- Gate: codex review on harness + config generator; physics-contract/dispatch/constants ratchets (no new physics numerics).

## STAGE 2 — Assemble + joint-train (T63 → T106)

- **T2.1** `config/wb/wb_best.yaml` = winning scheme per family; joint-train ALL trainable params under the multi-step forecast loss (multi-family generalization of one AIMIP variant).
- **T2.2** Climate-stat guardrail: track ACE2 climatology/spectral diagnostic each epoch; flag/abort on material climate-drift regression.
- **T2.3** Escalate the best config to T106 (reuse `config/aimip/t106/` machinery).
- Gate: codex review; conservation + physics-contract gates; SLURM (72 h as needed).

## STAGE 3 — Final scorecard vs SOTA

- **T3.1** Run best T63 + T106 through `run_weatherbench_eval.py` on the full 2020 test year (12 h init cadence).
- **T3.2** Overlay WB2 SOTA + climatology/persistence anchors; emit headline RMSE/ACC-vs-lead curves (Z500,T850,T2m,Q700,MSLP,10m-wind) + scorecard heatmap.
- **T3.3** Results doc under `docs/` with the honest §2 positioning; send headline PNG.
- Gate: codex review of the results doc + figures; final PR.

---

## Self-Review

- **Spec coverage:** Stage-0 tasks cover the spec §4 Stage-0 items — SegmentCarry→WB2 bridge (T5,T9), headline diagnostics incl. surface vars (T2–T5), ACC wiring (T7), baselines+SOTA (T8), CLI (T10), smoke+codex (T11). Stages 1–3 map to spec §4 Stage 1/2/3. Reuse ledger (spec §5) is enforced by the Global Constraints + per-task "reuse, don't re-derive" notes. Risks (spec §6): BPTT (Stage 1 loss config short unroll), diagnostic signs (T2–T4 analytic tests + sign walks), compute (SLURM constraint), climate guardrail (T2.2).
- **Placeholder scan:** Stage-0 steps carry real code or a concrete read-then-implement instruction with named reused signatures. Stages 1–3 are explicitly labeled "detailed tasks authored after Stage 0 lands" (they depend on Stage-0's finalized API + empirical winners) — this is intentional decomposition, not a placeholder.
- **Type consistency:** `interp_to_pressure_level`, `geopotential_height_at`, `mean_sea_level_pressure`, `screen_level_t2m`, `wind_10m`, `diagnose_headline_fields`, `regrid_to_wb2`, `score_forecast`, `run_wb_forecast_eval` names are used consistently across tasks. Metric primitives use the confirmed `evaluations/metrics.py` signatures.

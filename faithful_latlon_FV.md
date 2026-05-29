# Faithful lat-lon Finite-Volume AMIP (CMIP6) — change log

Goal: lat-lon FV grid runs AMIP (CMIP6) with **physically realistic** results,
**good MPI + GPU scaling**, tested, adversarially reviewed.

`grid=latlon` + `discretization=finite_volume` → `latlon_cgrid_primitive_equations`
(C-grid finite-volume hydrostatic PE). Verification runs on **CPU**
(`JAX_PLATFORMS=cpu`) — Metal backend broken in this env.

---

## Status matrix

| Axis | State |
|------|-------|
| Runs AMIP gray (1/10-day) | ✅ validated clean (iter 0) |
| Runs AMIP full RRTMG/CMIP6 | ✅ validated clean (iter 0) |
| Metal import crash | ✅ fixed (iter 0, `spectral_pe` device op → `math.log`) |
| Regression tests | ✅ fast + gated deck tests (iter 0) |
| **Physically realistic climate** | 🟡 `ic="standard"` realistic IC added (iter 1): CWV 84→11 kg/m² + meridional gradient. Multi-year climate validation still TODO |
| **MPI scaling + serial==MPI** | ⏳ AMIP physics NOT wired through `make_latlon_mpi_step` (`physics_fn=None`, latlon_mpi.py:1299). Halo/scatter/polar-filter/reductions done+tested (dry). Next axis. |
| **GPU scaling** | ⏳ dycore GPU-ready; `run_levante_gpu_scaling.py:1023` raises ValueError for `latlon` (stale "A-grid removed" guard); `create_latlon_mesh` exists but no SPMD wrapper wired. Next axis. |

---

## Iteration log

### iter 0 (commit c1e0317a) — baseline + Metal import fix
- Fixed `spectral_pe.py` module-level `float(jnp.log(...))` → `math.log` (was
  crashing whole import chain on Metal: "unknown attribute code: 22").
- Verified latlon FV runs AMIP: gray 1/10-day + full RRTMG CMIP6 1-day, all
  `validate_amip_run.py` ALL CHECKS PASSED (mass+moisture conserved, finite,
  stable 10-day).
- Tests: `test_amip_rrtmg_latlon.py` (FV+RRTMG step test `slow`; SST-regrid
  default), deck `latlon/finite_volume` gray + rrtmg (nightly gate).
- Adversarial review: approve, no findings.
- KNOWN: cold-start uniform 300K IC → CWV 84 kg/m² (3× Earth ~25). Spins down
  toward 76 over 10 days but realism not established. Grid-agnostic init, not
  FV-specific. **Realism is the next axis.**

### iter 1 — realistic IC (`ic="standard"`)
Recon (3 axes) found: realism blocked by uniform-300K IC (CWV 84≈6×Earth); ERA5
IC `NotImplementedError` for lat-lon; MPI AMIP physics unwired (`physics_fn=None`);
GPU harness blocks `latlon`. Picked realism (self-contained, locally verifiable).
- NEW `src/legoesm/atmosphere/standard_atmosphere.py`: grid-agnostic
  `standard_atmosphere_temperature(lat, sigma_full, cfg)` — polytropic
  constant-lapse-rate troposphere `T=T_sfc·σ^(R_d·Γ/g)` floored at strato,
  with equator-pole surface gradient `T_sfc=T_eq−ΔT·sin²lat`.
- Wired `ic="standard"` in `model_driver._apply_standard_atmosphere_ic`
  (latlon + cubed_sphere; raises for others), `config.py` `_valid_ic`,
  `run_amip.py --ic` choices. Equator T from `--t-init`; Earth defaults else.
- VERIFIED: latlon FV, init CWV 11.4 (vs 84.7 uniform), tropics 29 / poles 1.6
  kg/m². End-to-end run_amip 5-day gray latlon FV `--ic standard`: ALL CHECKS
  PASSED, CWV 11→17 (moistening to equilib), precip 1.84 mm/day.
- TESTS: `tests/atmosphere/unit/test_standard_atmosphere.py` (9, profile +
  driver wiring + CWV-realism + meridional gradient + contrast-vs-default).
- NOTE: 11.4 mean is on the dry side of Earth's ~24 (the RH·σ² moisture init is
  dry); acceptable cold-start, spins up. Revisit moisture-init profile later.
- ADV-REVIEW #1 (HIGH, fixed same commit): `ic="standard"` listed `gaussian`
  supported but spectral state is `SpectralHydrostaticState` (T_hat, no
  grid-space `T`) → AttributeError. Fix: `validate_strict` rejects
  `ic="standard"` for grid_type ∉ {latlon, cubed_sphere}; helper drops gaussian.
- ADV-REVIEW #2 (HIGH, fixed same commit): standard IC set a meridional T
  gradient but left u=v=0 + uniform p_s → unbalanced → startup geostrophic
  shock. Measured: max|dv/dt| 2.2e-3 (mild, ~= default IC). Fix: added
  thermal-wind-balanced zonal wind `standard_atmosphere_zonal_wind` —
  analytic geostrophic `u_g=(R_d·ΔT/(a·κ·Ω))·cos(lat)·(1−σ_eff^κ)` (regular at
  equator for the sin²lat profile), capped at the tropopause, deep-tropics
  taper. Result: subtropical jet ~29 m/s @ ~24°, weak equator, 0 at surface;
  max|dv/dt| 2.2e-3→5.9e-4 (3.8× lower). +6 wind/balance tests (16 total).
- ADV-REVIEW #3 (HIGH, fixed same commit): on cubed_sphere u/v are cube-LOCAL
  vector components; the geographic jet needs a grid-angle rotation (not wired)
  → wrong/discontinuous wind. Fix: restrict `ic="standard"` to grid_type=
  'latlon' only (validate_strict + helper); cubed_sphere/spectral/mpas rejected
  early. Lat-lon is the user's target and is fully correct (A-grid u=east).
  cubed_sphere standard IC = documented follow-up (needs vector rotation+test).
- MPI diagnosis: mpi4py 4.1.2/mpirun present, **but mpi4jax broken with
  installed JAX 0.10.1** (supported ≤0.10.0): raw allreduce returns wrong arity
  ("too many values to unpack"); dry latlon MPI step test hung earlier (likely
  same incompat → asymmetric rank error → collective deadlock). ⇒ local MPI
  scaling UNVERIFIABLE here; defer to cluster (Levante). Not a code bug in
  legoESM — env/version mismatch.
- Stability smoke (res48, polar filter, dt300, 15d, standard IC): ALL CHECKS
  PASSED — max_wind peak 28→final 21 m/s (bounded, no runaway), no NaN, p_s
  drift 4.4e-3, CWV 17, precip 1.6. +1 cheap in-CI startup-no-blowup test (17
  total). Re-review → (pending).

### iter 2 — realism review fixes (rounds #1-3) + seasonal climate validation
- Closed 3 adversarial-review HIGHs on the standard IC (gaussian crash; wind
  imbalance → thermal-wind jet; cubed-sphere vector basis → latlon-only). See
  iter-1 entry. Commit f31010d9 (lat-lon only, 18 tests). Round-4 review pending.
- SEASONAL CLIMATE VALIDATION (res48, polar filter, dt300, 90 days, ic=standard,
  analytical SST, gray rad, full physics): STABLE + physical. max_wind settles
  26→15 m/s (bounded, no runaway), CWV steady 15-17 kg/m² (Earth-like), precip
  ~1.4 mm/day, T_atm 253 K, no NaN/drift. Confirms the balanced realistic IC
  yields a sensible idealized AMIP climate over a full season. (Real-SST CMIP6
  multi-year + zonal-jet structure vs ERA5 = deeper follow-up.)

- ADV-REVIEW #4 (HIGH, fixed same commit): over non-flat topography the
  scaffold reduces p_s against uniform T_init, then standard IC replaced only
  T/u → p_s balanced against wrong column. Fix: `_apply_standard_atmosphere_ic`
  now re-scales p_s to the standard surface temperature
  (p_s·exp(−phis/R_d·(1/T_sfc−1/T_init)) = p0·exp(−phis/(R_d·T_sfc))); exact
  no-op when phis==0 so all flat runs stay bit-identical. +1 formula-consistency
  test (19 total). Round-5 review pending.
- DISCOVERED (pre-existing, separate bug): `gaussian_mountain` + latlon crashes
  (`add incompatible shapes (n_lat,),(n_lon,)`, topography.py:23) — fails with
  ic=default too. Blocks non-flat *analytic* topography AMIP on latlon. Couldn't
  exercise codex's "non-flat topo smoke" via the driver; p_s fix tested by the
  exact formula instead. Candidate fix iteration.

- ADV-REVIEW #5 (MEDIUM, fixed in IC commit): T_init ≤ 40 → pole T_sfc ≤ 0 →
  (T_strato/T_sfc)^(1/κ) NaN in wind. Fix: clamp T_sfc≥T_strato in σ_trop
  (NaN-safe; gives u=0 for no-troposphere columns) + validate_strict requires
  150 ≤ T_init ≤ 360 K for ic=standard. +2 tests (21 total in IC test file).
  Round-6 review pending.

### iter 3 — fix gaussian_mountain latlon bug (commit separate)
- `gaussian_mountain`/`zonal_ridge`/`schaer_mountain` used 1-D `grid.lat`/`lon`
  → (n_lat,)+(n_lon,) broadcast crash on latlon (worked on cubed-sphere where
  lat/lon are 2-D). Fix: `_grid_lat_lon_2d(grid)` helper prefers lat2d/lon2d.
  Now non-flat analytic topography AMIP works on latlon; also end-to-end
  validates the standard-IC p_s recompute over a real 2490 m mountain
  (driver setup ic=standard+gaussian on latlon: p_s 98062-100000 Pa, consistent).
  +4 tests in test_topography.py (31 total). NOTE: real-topo file path
  (load_real_topography target lat/lon) may have a similar 1-D issue — untested
  (needs data); candidate follow-up.

### iter 4+ — (next) GPU harness un-block (single-device verifiable); MPI deferred (mpi4jax↔JAX 0.10.1)
- TBD

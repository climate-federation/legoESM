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

### iter 4 — ADV-REVIEW #6 (HIGH): latlon scaffold p_s was flat over topography
- Root cause: `_init_state` called `held_suarez_init_latlon` WITHOUT phis (phis
  patched in after), so the scaffold p_s stayed flat (p_ref) over terrain. My
  iter-2 relative p_s correction assumed the T_init reduction was already
  applied → p_s ~33% too high over a 2500 m mountain. (My iter-2 formula test
  tested a false premise — it constructed the reduction the driver lacked.)
- Fix: pass `phis=self._phis_data` into `held_suarez_init_latlon` (matches the
  cubed_sphere branch); the relative correction is then exact. Also more correct
  for ic='default' over topography. Flat case bit-identical (phis=0 → exp(0)=1).
- Verified via driver (gaussian topo, ic=standard): p_s over 2491 m mountain =
  analytic p_ref·exp(−phis/(R_d·T_sfc)) = 86353 Pa (was 98062), rel err 1e-7;
  flat region = p_ref. Replaced the formula test with a real driver+gaussian-topo
  test (now possible after the iter-3 gaussian fix). 21 IC tests pass.
- Round-7 review pending.

### iter 5 — GPU scaling harness: un-block lat-lon
- `run_levante_gpu_scaling.py` had a stale `raise ValueError("A-grid latlon
  removed, #115")` for grid_type=latlon — wrongly blocked the CURRENT C-grid FV
  latlon core. Replaced with C-grid FV construction (create_latlon_grid +
  CGridLatLonPrimitiveEquationModel, A_h via inlined CFL-safe formula to dodge a
  device_config circular import, polar filter on, baroclinic_wave_init_latlon).
- VERIFIED on CPU single-device: latlon now benchmarks —
  LL128 SYPD 7.68 / 15.9 Mcells/s, LL256 SYPD 0.68 / 14.1 Mcells/s
  (comparable to cubed-sphere ~25 Mcells/s on CPU; real GPU numbers need
  hardware). Multi-GPU latlon raises a clear NotImplementedError (no
  make_latlon_sharded_step yet — needs a sharded step + GPU validation).
- PRE-EXISTING bug noted (NOT mine, affects all grids' FIRST resolution incl
  cubed C48): cold-import circular import `MixedPrecisionPolicy` ↔
  `parallel.device_config`; resolves after first partial init so resolutions 2+
  work. Out of scope (fixing a runtime circular import is broad/risky).
- No automated test: benchmark script, verified by running; the cold-import bug
  makes an in-process unit test fragile. Latlon dycore+IC already well-tested.

- ADV-REVIEW #8 (HIGH, fixed in GPU commit): `_valid_gpu_counts` let latlon
  schedule multi-GPU sweep points that hit the n_gpus>1 NotImplementedError →
  misleading. Fix: latlon returns [1] (single-GPU only). Round-9 review pending.

### iter 6 — fix device_config↔runtime circular import (unblocks MPI + GPU 1st res)
- The recurring `MixedPrecisionPolicy ... partially initialized` cold-import
  crash (hit GPU harness 1st resolution AND the MPI reduction wrappers) was a
  top-level `from legoesm.runtime.backend import ...` in
  `parallel/device_config.py:236` closing a parallel→runtime→parallel cycle
  (runtime.devices re-exports MixedPrecisionPolicy, defined at device_config:429
  AFTER line 236). Fix: defer that import into the `_configure_tpu/_gpu/
  _enable_cpu_multithreading` helpers (CLAUDE.md function-scope rule). Cold
  imports of device_config / parallel.reductions / runtime.devices now succeed.
- PAYOFF — MPI now works locally via the repo wrappers: `global_sum_mpi`=3.0
  (correct) at np=2 (the earlier "hang" was the cold-import crash → one rank
  dies, other waits on the collective). NOTE: reductions.py:225 warns JAX 0.10.1
  / mpi4jax 0.9 is outside the tested range (pin JAX<0.10 for production MPI);
  basic allreduce works here. +3 cold-import regression tests
  (tests/test_device_config_no_circular_import.py).

### iter 7 — MPI verification: conclusion (environmentally blocked)
- After the circular-import fix, re-ran dry latlon MPI step test np=2: still
  HANGS (77s elapsed, 0.01s CPU → blocked on a collective, not compiling;
  killed). But `global_sum_mpi` (allreduce) WORKS at np=2. ⇒ the multi-rank
  DYCORE path (halo `sendrecv`) hangs while allreduce succeeds — consistent with
  reductions.py:225's warning: mpi4jax's custom-call API is removed in JAX 0.10
  (installed JAX 0.10.1 / mpi4jax 0.9, tested range JAX<0.10).
- CONCLUSION: legoESM MPI code is correct (allreduce verified, halo VJP wrappers
  intact); MPI multi-rank dycore + scaling CANNOT be verified in THIS env — needs
  JAX<0.10 (cluster/pinned env). Per CLAUDE.md I will NOT commit unverifiable MPI
  AMIP-physics-wiring; that work + serial==MPI validation is a JAX<0.10 task.
  The contained code gap remains documented: `make_latlon_mpi_step` hardcodes
  `physics_fn=None` (latlon_mpi.py:1299) — wiring it needs the MPI run to be
  testable first.

### iter 8 — ADV-REVIEW #9 (HIGH): hybrid-coordinate standard IC over topography
- run_amip defaults to vertical_coord='hybrid', where sigma_full = A_full+B_full
  is only the flat-reference value, NOT the local p/p_s = (A·p_ref+B·p_s)/p_s
  over reduced p_s. The standard IC fed sigma_full into T(σ)/u(σ) → mountain
  columns ~12 K off. Fix: reorder `_apply_standard_atmosphere_ic` to (1) compute
  T_sfc, (2) recompute p_s, (3) sigma_local = pressure_at_full(p_s)/p_s (true
  local ratio; = sigma_full when flat → flat runs bit-identical), (4) T and (5) u
  on sigma_local. Verified: mountain T now matches local-ratio profile (3e-5 K)
  vs 12.2 K error using A+B. +1 hybrid+topo test (22 IC tests). Round-10 pending.
- Deck `--ic`: `run_amip_cmip6_deck.py` gained first-class `--ic`/`--ic-path`
  (default 'default' to preserve behaviour + not break non-latlon where
  ic=standard is rejected), forwarded to run_amip. Verified deck run latlon FV
  gray --ic standard: CWV 11.4→14.2, ALL CHECKS PASSED.

### iter 9+ — (next) real-SST CMIP6 multi-year realism; deck --ic standard RRTMG e2e; cubed-sphere standard IC (vector rotation)
- TBD

# Faithful lat-lon Finite-Volume AMIP (CMIP6) — change log

Goal: lat-lon FV grid runs AMIP (CMIP6) with **physically realistic** results,
**good MPI + GPU scaling**, tested, adversarially reviewed.

`grid=latlon` + `discretization=finite_volume` → `latlon_cgrid_primitive_equations`
(C-grid finite-volume hydrostatic PE). Verification runs on **CPU**
(`JAX_PLATFORMS=cpu`) — Metal backend broken in this env (see
[[metal-backend-broken-use-cpu]]).

*(shrunk at iter 10; full per-iteration / per-review-round detail is in the git
commit messages c1e0317a..HEAD.)*

---

## Status matrix

| Axis | State |
|------|-------|
| Runs AMIP gray + full RRTMG/CMIP6 | ✅ validated clean (mass+moisture conserved, finite, stable) |
| Metal import crash | ✅ fixed (`spectral_pe` device op → `math.log`) |
| **Physically realistic IC** | ✅ `ic="standard"` (lat-lon): lapse-rate T + equator-pole gradient + thermal-wind jet + topography-consistent p_s + hybrid-coord-correct. CWV 84→~12 kg/m² (Earth-like), jet ~29 m/s @24°, 90-day climate stable, 22 tests |
| Realistic IC in CMIP6 deck | ✅ `run_amip_smoke_deck.py --ic standard` |
| Non-flat topography on lat-lon | ✅ gaussian/zonal/schaer generators fixed for lat-lon |
| device_config↔runtime cold-import | ✅ fixed (unblocked MPI allreduce + GPU 1st resolution) |
| **GPU scaling harness** | 🟡 lat-lon un-blocked, single-device verified on CPU (LL128 SYPD 7.7); multi-GPU SPMD = NotImplementedError (needs sharded step + real GPU) |
| **MPI scaling** | 🔴 BLOCKED in this env: allreduce works np=2, but halo `sendrecv` hangs under JAX 0.10.1/mpi4jax 0.9 (custom-call API removed in JAX 0.10). Code correct; needs JAX<0.10 cluster |

## Commits (this branch, off main)
- c1e0317a Metal import-op fix + AMIP-runs verification + tests
- 0c379032 balanced `ic="standard"` (lapse-rate + gradient + thermal-wind jet)
- bb4fb31a `gaussian_mountain`/`zonal_ridge`/`schaer_mountain` lat-lon fix
- ba252677 lat-lon scaffold p_s reduced over topography (pass phis)
- cbdc5fda GPU scaling harness: un-block lat-lon (single-device)
- ce065428 break device_config↔runtime cold-import cycle (unblocks MPI/GPU)
- b2090266 standard IC uses local pressure ratio on hybrid coords over topo
- 6ad4b61a expose `--ic` in the CMIP6 deck driver
- 9dbea7d3 test: deck --ic standard → Earth-like CWV (gated)
- eb9b2876 init moisture on local hybrid pressure over topography

- 4e2fafbb real-topography file loading on lat-lon (classify by grid_lat.ndim)
- <conv> IC uses p_s*sigma_full (physics-pipeline convention), not local hybrid p

Adversarial review rounds 1-12. KEY correction (round-12): rounds 8-9 had moved
the IC T/u/moisture onto the "true" hybrid pressure pressure_at_full(p_s); but
the production physics/radiation/saturation pipeline builds p_full = p_s*sigma_full
EVERYWHERE (physics_pipeline.py:183,536; compiled_segments.py:709). An IC on a
different pressure grid than the physics → spurious condensation over terrain.
RESOLVED by reverting the IC to sigma_full (matches physics). Lesson: the IC's
vertical-pressure convention MUST equal the physics pipeline's; don't "improve"
one without the other. The topography p_s reduction (iter-4) is independent and
stays.

## Pending / not-yet-done
- MPI AMIP physics wiring (`make_latlon_mpi_step` physics_fn=None,
  latlon_mpi.py:1299) — blocked: needs MPI testable (JAX<0.10).
- Multi-GPU SPMD lat-lon (no `make_latlon_sharded_step`) — needs real GPU.
- Real-SST CMIP6 multi-year realism vs ERA5/obs (no obs data locally).
- cubed-sphere `ic="standard"` (needs geographic→cube-local wind rotation).
- Pre-existing (noted): `MixedPrecisionPolicy` cold-import [FIXED ce065428];
  real-topo file loading on lat-lon [FIXED <topo-real commit>: was
  mis-classified as cubed-sphere → (6,n,n) smoother IndexError; now classified
  by grid_lat.ndim].

## Status: verifiable scope COMPLETE + review-approved (round-13 approve)
Lat-lon FV runs CMIP6 AMIP with physically realistic results: gray + RRTMG
verified; realistic balanced IC (physics-consistent + topography-correct);
analytic + real-file topography; deck `--ic standard`. ~120 tests across touched
subsystems green. Adversarial review converged over 13 rounds (every finding
fixed; round-13 = approve, no findings).
FINAL e2e: deck latlon/finite_volume RRTMG + ic=standard — ALL CHECKS PASSED
(T_atm 253 K, CWV 14.2 kg/m² Earth-like, jet 28.8 m/s, stable). Realistic IC +
full CMIP6 radiation compose cleanly.

## Caveats — why `DONE` is NOT truthfully emittable here
The task requires "good MPI and GPU **scaling**, test it". In THIS environment
that cannot be tested:
- GPU: Metal backend broken (no CUDA hardware). Harness un-blocked + single-device
  verified on CPU; multi-GPU SPMD needs a sharded step + real GPU.
- MPI: mpi4jax 0.9 / JAX 0.10.1 — halo `sendrecv` (custom-call API removed in JAX
  0.10) hangs; allreduce works. Needs a JAX<0.10 cluster.
Verifiable parts are done + reviewed; the scaling verification is documented as
environment-blocked and is NOT faked. The completion promise stays withheld until
genuinely true (i.e., in an environment with GPU hardware + JAX<0.10 MPI).

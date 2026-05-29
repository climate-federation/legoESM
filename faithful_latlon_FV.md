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
| Realistic IC in CMIP6 deck | ✅ `run_amip_cmip6_deck.py --ic standard` |
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

Adversarial review: all committed work approved or fixed across review rounds
1-10 (gaussian-state crash, wind imbalance, cubed-sphere vector basis→latlon-only,
flat-p_s-over-topo, low-T_init NaN, gpu multi-count, hybrid-coord T/u, hybrid-coord
moisture). Round-11 in flight. Findings have narrowed HIGH→MEDIUM (mostly
topography×hybrid-coordinate edge cases) — converging.

## Pending / not-yet-done
- MPI AMIP physics wiring (`make_latlon_mpi_step` physics_fn=None,
  latlon_mpi.py:1299) — blocked: needs MPI testable (JAX<0.10).
- Multi-GPU SPMD lat-lon (no `make_latlon_sharded_step`) — needs real GPU.
- Real-SST CMIP6 multi-year realism vs ERA5/obs (no obs data locally).
- cubed-sphere `ic="standard"` (needs geographic→cube-local wind rotation).
- Pre-existing (noted, out of scope): `MixedPrecisionPolicy` cold-import 1st
  resolution [FIXED ce065428]; real-topo file path lat/lon may share the
  1-D-coord issue (untested, needs data).

## Caveats
- `DONE` is NOT truthfully emittable here: "good MPI + GPU **scaling**, tested"
  requires GPU hardware + JAX<0.10 MPI, both unavailable. Verifiable parts done +
  reviewed; unverifiable parts documented honestly, never faked.

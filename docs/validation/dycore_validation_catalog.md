# Dycore Validation Test Catalog

**Reference**: Owen Hughes, *"How to validate a 3D spherical dynamical core"*,
Tutorial, University of Michigan, 30 April 2026
(`Downloads/validating_dycores.pdf`).

This catalog tracks legoESM's coverage of the canonical idealized test set that
any 3D spherical dynamical core should pass before being trusted as the
dynamical core of an ESM. Tests are grouped by the Hughes (2026) section.

**Status legend**

| Symbol | Meaning |
| --- | --- |
| ✅ | Wired into legoESM (matrix script + unit test) |
| ⚠️ | Partial coverage — different DCMIP vintage, single grid, or via integration only |
| ❌ | Missing |

The matrix script's `--family` filter selects subsets:
`hughes` selects the full Hughes 2026 set, individual flags
(`sw / hydro / nh / moist / climate / tracer / dcmip2008 / dcmip2012 /
dcmip2016`) select sub-families.

Run with `JAX_ENABLE_X64=1 python scripts/matrix/run_atmosphere_test_matrix.py
--quick --family <name>`.

---

## §2 — Tracer transport (prescribed-wind passive advection)

| ID | Test | Grids | Status | Module |
| --- | --- | --- | --- | --- |
| L12-2D | Lauritzen 2012 standard 2D suite (gaussian, slotted-cylinder, correlated tracers) | all | ⚠️ M3 | TBD `tests/test_cases/lauritzen2012/gaussian_advection.py` |
| DCMIP12-1-1 | 3D deformational flow | all | ✅ | `tests/test_cases/dcmip_transport.py` (test_num=11) |
| DCMIP12-1-2 | Hadley-like meridional circulation | all | ✅ | `tests/test_cases/dcmip_transport.py` (test_num=12) |
| DCMIP12-1-3 | Thin clouds over orography | all | ✅ | `tests/test_cases/dcmip_transport.py` (test_num=13) |
| DCMIP08-T | JW rotated baroclinic with passive DCMIP-2008 tracers (operator-splitting) | all | ❌ M3 | TBD `tests/test_cases/dcmip2012/jw_dynamical_tracers.py` |

## §3 — Hydrostatic dry 3D

| ID | Test | Grids | Status | Module |
| --- | --- | --- | --- | --- |
| JW06-BW | Classic Jablonowski-Williamson baroclinic wave | all | ✅ | `tests/test_cases/baroclinic_wave.py` |
| JW06-BS | Un-rotated JW **base-state control** (`baroclinic_steady`, σ, perturbation OFF) — diagnostic control for #1028: any jet decay here is the dycore failing to hold the balanced state, since no wave exists to consume it | all | ✅ | `tests/test_cases/baroclinic_wave.py` |
| JW06-RS | Rotated JW steady-state (α=π/4) | all | ✅ M1 | `tests/test_cases/dcmip2008/jablonowski_rotated.py` |
| JW06-RB | Rotated JW baroclinic wave (α=π/4) | all | ✅ M1 | `tests/test_cases/dcmip2008/jablonowski_rotated.py` |
| DCMIP12-2-0-0 | Atmospheric rest state with steep hydrostatic topography | all | ✅ M1 | `tests/test_cases/dcmip2012/rest_state_topography.py` |
| DCMIP08-6-0 | Small-amplitude Rossby-Haurwitz wave (3D, isothermal background) | all | ✅ M1.b | `tests/test_cases/dcmip2008/rossby_haurwitz_6_0.py` |
| DCMIP08-5-0 | Mountain-induced Rossby wave (3D zonal flow over isolated mountain) | all | ✅ M1.b | `tests/test_cases/dcmip2008/mountain_rossby_5_0.py` |
| DCMIP08-3-1 | Gravity wave on non-rotating Earth (isothermal) | all | ✅ M1.b (canonical: matrix runner builds grid with `omega=0`) | `tests/test_cases/dcmip2008/gravity_wave_3_1.py` |
| DCMIP08-3-2 | Inertio-gravity wave on rotating planet | all | ✅ M1.b | `tests/test_cases/dcmip2008/inertio_gravity_3_2.py` |

## §4 — Non-hydrostatic dry 3D

| ID | Test | Grids | Status | Module |
| --- | --- | --- | --- | --- |
| DCMIP25-TC1 | Mountain-triggered gravity waves (Schaer-mountain variant) | NH | ⚠️ | `tests/test_cases/dcmip2025/test_case_1.py` |
| DCMIP12-NH-2-0-1 | Atmospheric rest state with steep Schaer (NH) topography | NH | ❌ M2 | TBD `tests/test_cases/dcmip2012/rest_state_schaer_nh.py` |
| DCMIP12-2-1 | Schaer-mountain orographic GW without shear | NH | ❌ M2 | TBD `tests/test_cases/dcmip2012/schaer_gw_no_shear.py` |
| DCMIP12-2-2 | Schaer-mountain orographic GW with shear | NH | ❌ M2 | TBD `tests/test_cases/dcmip2012/schaer_gw_with_shear.py` |
| DCMIP12-3-1 | Non-orographic gravity waves | NH | ❌ M2 | TBD `tests/test_cases/dcmip2012/non_orographic_gw.py` |
| Klemp15-PGW | Quasi-planar reduced-radius orographic GW | NH | ❌ M2 | TBD `tests/test_cases/klemp2015/orographic_gw_planar.py` |

## §5 — Dynamical tracer tests

| ID | Test | Grids | Status | Module |
| --- | --- | --- | --- | --- |
| DCMIP12-4-1 | EPV / θ tracer for dry baroclinic wave (discrete EPV conservation diagnosis) | all | ❌ M3 | TBD `tests/test_cases/dcmip2012/jw_dynamical_tracers.py` + `src/legoesm/diagnostics/epv.py` |

## §6 — Moist 3D

| ID | Test | Grids | Status | Module |
| --- | --- | --- | --- | --- |
| DCMIP25-TC2 | Moist baroclinic wave (DCMIP 2025 vintage) | NH | ⚠️ | `tests/test_cases/dcmip2025/test_case_2.py` |
| DCMIP25-TC3 | Squall-line / supercell (DCMIP 2025 vintage) | NH | ⚠️ | `tests/test_cases/dcmip2025/test_case_3.py` |
| DCMIP16-1-1 | Moist baroclinic wave (Reed-Jablonowski simple physics, no topography) | all | ❌ M3 | TBD `tests/test_cases/dcmip2016/moist_baroclinic_1_1.py` |
| DCMIP16-1-2 | Moist baroclinic wave with toy chemistry | all | ❌ M3 | TBD `tests/test_cases/dcmip2016/moist_baroclinic_chemistry_1_2.py` |
| HJ23-MBW | Hughes & Jablonowski 2023 moist baroclinic with topography | all | ❌ M3 | TBD `tests/test_cases/dcmip2016/moist_baroclinic_topo_1_3.py` |
| DCMIP16-2-0 | Supercell with idealized physics (Klemp small-Earth) | NH | ❌ M3 | TBD `tests/test_cases/dcmip2016/supercell_2_0.py` |
| DCMIP16-3-0 | Tropical cyclone (Reed & Jablonowski) | NH | ❌ M3 | TBD `tests/test_cases/dcmip2016/tropical_cyclone_3_0.py` |
| Klemp15-SC | Klemp 2015 supercell variant (reduced-radius) | NH | ❌ M3 | TBD `tests/test_cases/klemp2015/supercell.py` |

## §7 — Climate-timescale benchmarks

| ID | Test | Grids | Status | Module |
| --- | --- | --- | --- | --- |
| HS94 | Held-Suarez (1994) — Newtonian relaxation + Rayleigh friction | all | ✅ | `src/legoesm/atmosphere/held_suarez.py` |
| HS-Topo | Held-Suarez with idealized Gaussian-mountain topography | all | ✅ M1 | `src/legoesm/atmosphere/idealized/held_suarez_topo.py` |
| WS09-SP | Small-planet Held-Suarez (Wedi & Smolarkiewicz 2009, X=125) | all | ⚠️ M1.b | helpers at `src/legoesm/atmosphere/idealized/small_planet.py`; matrix-script runner wiring deferred to a follow-up commit |
| TJ16-MHS | Moist Held-Suarez (Thatcher & Jablonowski 2016) | all | ⚠️ stability fixture only | TBD `src/legoesm/atmosphere/idealized/moist_held_suarez.py`; existing fixture: `tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py` |
| F06-AQUA | Frierson gray-radiation aquaplanet (2006) | all | ❌ M1.b/M3 | TBD `src/legoesm/atmosphere/idealized/frierson_gray.py` |

## Williamson (1992) shallow-water completion (implied prerequisites)

| ID | Test | Grids | Status | Module |
| --- | --- | --- | --- | --- |
| W1 | Cosine bell advection (solid-body rotation, β=π/4) | all | ✅ | `tests/test_cases/cosine_bell.py` (alias W1) |
| W2 | Steady-state nonlinear zonal geostrophic flow | all | ✅ | `tests/atmosphere/shallow_water/test_cases/williamson.py` |
| W3 | Steady-state nonlinear zonal geostrophic flow with compact support (forced) | all | ❌ M1.b | TBD `tests/test_cases/williamson_extended.py::williamson3_*` |
| W4 | Forced nonlinear flow with translating low | all | ❌ M1.b | TBD `tests/test_cases/williamson_extended.py::williamson4_*` |
| W5 | Zonal flow over isolated mountain | all | ✅ | `tests/atmosphere/shallow_water/test_cases/williamson.py` |
| W6 | Rossby-Haurwitz wave (wavenumber 4) | all | ✅ M1 | `tests/test_cases/williamson_extended.py::williamson6_*` |

---

## Milestone roadmap

The full Hughes-tutorial catalog is being delivered in **three milestones**.

### M1 — Hydrostatic-dry & climate-idealized (in progress)

Lands the catalog and a foundational batch of new tests using existing
hydrostatic dycores (no new physics packages). M1.a (delivered in this PR):

- `docs/validation/dycore_validation_catalog.md` (this file)
- `src/legoesm/atmosphere/idealized/topography.py` — shared mountain
  constructors (Williamson-5 cone, Gaussian, DCMIP §2-0-0 cosine-bell);
  `tests/test_cases/_terrain_helpers.py` shim re-exports them for
  backward compatibility
- `tests/test_cases/williamson_extended.py` — W6 Rossby-Haurwitz init
  for cubed-sphere, lat-lon, MPAS, and spectral grids; matrix-script
  wiring is active for MPAS + spectral (cube + lat-lon deferred to M1.b)
- `tests/test_cases/dcmip2008/jablonowski_rotated.py` — DCMIP 2008
  §4-1 (rotated steady) + §4-2 (rotated baroclinic), all four grids,
  α=π/4 default
- `tests/test_cases/dcmip2012/rest_state_topography.py` — DCMIP 2012
  §2-0-0 atmosphere-at-rest with topography, all four grids
- `src/legoesm/atmosphere/idealized/held_suarez_topo.py` — Held-Suarez
  forcing wrapped over a DCMIP §2-0-0 ridged cosine-bell mountain,
  all four grids
- `scripts/matrix/run_atmosphere_test_matrix.py` — new `--family` filter,
  `family` field on `TestCase`, `_CASE_TO_FAMILY` lookup, ~16 new
  TestCase rows wired through the existing `run_baroclinic`,
  `run_held_suarez`, and `run_shallow_water` runners
- `tests/atmosphere/{shallow_water,hydrostatic}/unit/test_*.py` — 30
  direct-import sanity tests covering all of the above (all green)
- `README.md` — links the catalog under "Validation Infrastructure"

M1.b (delivered in a second commit on this PR):

- `tests/test_cases/dcmip2008/_shared.py` — isothermal-hydrostatic
  state builders for all four grids (factored out so the four
  DCMIP-2008 modules below can stay short)
- `tests/test_cases/dcmip2008/gravity_wave_3_1.py` — DCMIP §3-1
  gravity wave on (notionally) non-rotating sphere
- `tests/test_cases/dcmip2008/inertio_gravity_3_2.py` — DCMIP §3-2
- `tests/test_cases/dcmip2008/mountain_rossby_5_0.py` — DCMIP §5-0
- `tests/test_cases/dcmip2008/rossby_haurwitz_6_0.py` — DCMIP §6-0
- `src/legoesm/atmosphere/idealized/small_planet.py` — small-planet
  grid factory wrappers (Wedi & Smolarkiewicz 2009)
- 4 new TestCase rows × 4 grids per case = 16 rows wired into
  `run_baroclinic` via a new `_build_dcmip2008_state` dispatcher
- 21 direct-import unit tests in
  `tests/atmosphere/hydrostatic/unit/test_dcmip2008_dry_3d.py`
- Catalog updated to mark all four DCMIP-2008 dry-3D entries ✅

Still deferred (small follow-up):

- `tests/test_cases/williamson_extended.py::williamson3_*`, `williamson4_*`
- W6 matrix-script wiring for cubed-sphere (FV3 D-grid) and
  lat-lon C-grid (needs analytic edge-/face-midpoint wind init for
  non-zonally-symmetric flow)
- Small-planet HS matrix-runner wiring (the IC + grid factory are
  delivered; wiring needs a small extension to ``run_held_suarez``
  to plumb the small-planet grid factories)
- `src/legoesm/atmosphere/idealized/frierson_gray.py` (dry + moist)
- Progression-suite extension (`tests/validation/run_dycore_progression_suite.py`)

### M2 — Non-hydrostatic dry 3D (next PR)

- DCMIP 2012 NH Schaer suite (rest-state, with shear, without shear)
- DCMIP 2012 non-orographic gravity waves
- Klemp 2015 quasi-planar reduced-radius variant
- New runner `run_dcmip_nh_dry`

### M3 — Moist & tracer (final PR)

- New idealized-physics packages: Reed-Jablonowski simple physics,
  Thatcher-Jablonowski moist HS, Klemp supercell, Frierson moist
- DCMIP 2016 moist suite (1-1, 1-2, Hughes-Jablonowski 2023, supercell, TC)
- Lauritzen 2012 standard 2D advection suite
- DCMIP 2012 4-1 EPV/θ dynamical tracer test
- Discrete-EPV diagnostic in `src/legoesm/diagnostics/epv.py`

---

## Reference list

The Hughes (2026) tutorial cites:

1. Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber,
   P. N. (1992). A standard test set for numerical approximations to the
   shallow water equations in spherical geometry. *J. Comput. Phys.*,
   102, 211–224.
2. Held, I. M., & Suarez, M. J. (1994). A proposal for the intercomparison
   of the dynamical cores of atmospheric general circulation models.
   *Bull. Amer. Meteor. Soc.*, 75(10), 1825–1830.
3. Frierson, D. M. W., Held, I. M., & Zurita-Gotor, P. (2006). A
   gray-radiation aquaplanet moist GCM. Part I. *J. Atmos. Sci.*, 63(10),
   2548–2566.
4. Jablonowski, C., & Williamson, D. L. (2006). A baroclinic instability
   test case for atmospheric model dynamical cores. *QJRMS*, 132, 2943–2975.
5. Wedi, N. P., & Smolarkiewicz, P. K. (2009). A framework for testing
   global non-hydrostatic models. *QJRMS*, 135(639), 469–484.
6. Lauritzen, P. H., Skamarock, W. C., Prather, M., & Taylor, M. (2012).
   A standard test case suite for two-dimensional linear transport on the
   sphere. *Geosci. Model Dev.*, 5(3), 887–901.
7. Klemp, J., Skamarock, W., & Park, S.-H. (2015). Idealized global
   non-hydrostatic atmospheric test cases on a reduced-radius sphere.
   *J. Adv. Model. Earth Syst.*, 7(3), 1155–1177.
8. Thatcher, D. R., & Jablonowski, C. (2016). A moist aquaplanet variant
   of the Held-Suarez test for atmospheric model dynamical cores.
   *Geosci. Model Dev.*, 9(4), 1263–1292.
9. Hughes, O. K., & Jablonowski, C. (2023). A mountain-induced moist
   baroclinic wave test case for atmospheric general circulation models.
   *Geosci. Model Dev.*, 16(22), 6805–6831.
10. DCMIP 2008 test case description (rotated JW, gravity waves,
    Rossby-Haurwitz, mountain-induced Rossby).
11. DCMIP 2012 test case description (deformational tracer, Hadley
    advection, rest with topography, Schaer mountain GW, non-orographic
    GW, EPV/θ tracers).
12. DCMIP 2016 test case description (moist baroclinic wave, supercell,
    tropical cyclone).

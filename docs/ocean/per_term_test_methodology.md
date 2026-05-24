# Per-term test methodology

Phase 1 of the Adcroft follow-up plan
(`docs/ocean/adcroft_followups.md`) establishes a per-term validation
harness for the ocean dycore.  This doc explains the methodology — the
*shape* of a per-term test, the convergence-rate convention, the
toggle factory, and the gradient-correctness variant — so that every
test landed under `tests/ocean/unit/test_per_term_*.py` looks the same
and the next-developer-to-add-a-test does not have to re-derive the
pattern.

---

## 1. What a per-term test is

A per-term test exercises **one term** of the dycore against a closed-
form solution.  Everything else is turned off, so the only thing the
test can fail on is that one term.  This is the dycore equivalent of
unit testing.

The reviewer triangulation in `adcroft_review_consensus.md` Item 6
(consensus 4/4) is the rationale.  In short: integrated tests (gyre,
ACC, Eady) cannot localise a failure to a single operator and cannot
measure scheme order; per-term tests can do both.

## 2. The toggle factory

`src/legoesm/ocean/experiments/test_configs.py` exposes
`make_test_config(term, grid="latlon", **overrides)`.  Every test
should construct its config via this factory rather than via
hand-rolled `LatLonCGridOceanConfig(disable_pgf=True, ...)` calls,
because:

1. The factory sets `test_mode=True` explicitly.  Production paths
   should never accept a config with `test_mode=True`; the test mode
   is meant to be **loud in the audit trail**
   (ocean-model-expert review,
   `adcroft_review_consensus.md` §3 Item 6).
2. The factory is the single place that knows which `disable_*` flags
   map to which test recipe.  When a new test is added that needs a
   new isolation pattern, the factory grows one new `term` literal —
   the dispatch is centralised, not scattered through test files.

Supported `term` literals:

| `term`                     | Recipe                                  | Disables                                  |
|----------------------------|-----------------------------------------|-------------------------------------------|
| `"coriolis_only"`          | #1 — inertial oscillation               | pgf, momentum_advection, tracer_advection, drag |
| `"tracer_advection_only"`  | #2 — 1D cosine bell convergence         | coriolis, pgf, momentum_advection, drag   |
| `"gravity_wave"`           | #5 — linear SW gravity wave dispersion  | coriolis, momentum_advection, tracer_advection, drag |

When you add a new per-term test, also add the matching `term`
literal in `_disable_flags_for_term`.

## 3. The disable flags

Both `LatLonCGridOceanConfig` and `MPASOceanConfig` carry six new
fields (Phase 1A of the plan):

```
test_mode: bool = False                  # explicit "this is a test config" signal
disable_coriolis: bool = False
disable_pgf: bool = False
disable_momentum_advection: bool = False
disable_tracer_advection: bool = False
disable_drag: bool = False
```

Production code paths leave all of these `False`.  When a flag is
`True`, the corresponding term contributes zero in the relevant
tendency / step function:

| Flag                          | Lat-lon gate (file / line area)                            | Status |
|-------------------------------|------------------------------------------------------------|--------|
| `disable_pgf`                 | `ocean_pe_latlon_cgrid.py:1047–1057` (zeroes `dp_dx`, `dp_dy`) | wired  |
| `disable_momentum_advection`  | same; plus PV-flux block ~`:1213` (skipped)                  | wired  |
| `disable_drag`                | `ocean_pe_latlon_cgrid.py:1534` (drag block conditional)    | wired  |
| `disable_coriolis`            | `ocean_model_latlon_cgrid.py:744` (skip Matsuno step)       | wired  |
| `disable_tracer_advection`    | (not wired yet — gates land alongside Recipe #2)            | planned |

MPAS gates are not wired yet; they land when MPAS-specific per-term
tests are added (planned for follow-up).

The gates are **Python `if` on static bool capture**, *not*
`jnp.where`.  The disable flag is part of the config NamedTuple
identity, so JIT caches the branch.  Using `jnp.where` would trace
both branches at compile time and waste compute; the CLAUDE.md JAX
rule under "JAX Engineering Rules" enforces this explicitly.

## 4. Convergence rate, not absolute L2

Every per-term test that has a closed-form reference reports the
empirical **convergence rate** across a resolution sweep and asserts
that the rate is within tolerance of the scheme's formal order.

Why rate, not absolute L2:

- **Absolute L2 thresholds drift** with grid choice, numerical noise,
  and minor implementation changes.  Tightening the threshold catches
  regressions but produces false failures on benign refactors;
  loosening it lets real regressions through.
- **Rate is the invariant.**  A 2nd-order scheme produces rate ≈ 2 no
  matter what the absolute error happens to be.  A failing rate is a
  much sharper bug signal than a failing threshold.
- **The reviewer triangulation** all three persona-driven reviewers
  picked convergence rate as the right pass criterion
  (`adcroft_review_consensus.md` §3 Item 6 cross-cutting
  refinements).

The helper `tests/ocean/unit/_helpers.py:convergence_rate` returns
one rate per pair of successive resolutions.
`assert_convergence_rate_at_least(errors, resolutions, expected_order,
tolerance=0.15)` is the standard assertion: every pairwise rate must
be at least `expected_order - tolerance`.

Pairwise rather than aggregate: a single bad refinement (e.g.,
limiter activation at one resolution) shows up as a degraded *pair*
rather than averaging out.

### Rate targets per scheme (1D smooth advection benchmark, Recipe #2)

Empirical results from `test_per_term_tracer_advection.py` on a
Gaussian IC + SSP-RK3 time integration + short-translation diagnostic
(1/8 revolution to avoid numerical-diffusion saturation):

| Scheme        | Formal order | Empirical L2 rate | Test threshold |
|---------------|:------------:|:-----------------:|----------------|
| `upwind`      | 1            | ≈ 1.0             | rate ≥ 0.80    |
| `tvd` (Van Leer) | 2         | ≈ 2.0             | rate ≥ 1.70    |
| `dst3`        | 3            | **≈ 1.0** *(see below)* | **not asserted** |
| `ppm`         | 3            | (under investigation — overflowed on smooth Gaussian, periodic; tracked as follow-up) | not asserted |
| `weno5`       | 5            | (capped to ≲ 3 by SSP-RK3 anyway; not asserted) | not asserted |
| `weno7`       | 7            | (same)            | not asserted   |
| `som`         | 3 (moment-preserving) | (different methodology; needs separate test) | not asserted |

**Why DST-3 doesn't show 3rd order in L2 on this test.** Our DST-3
implementation (`src/legoesm/ocean/advection.py:89`) applies a Van Leer
limiter that activates wherever the gradient ratio
``r = δ_uu / δ`` flips sign — i.e., at every smooth extremum.  At
that point the scheme reduces to first-order upwind locally, and the
L2 norm picks up the 1st-order error region.  The empirical rate over
``n_lon ∈ {64, 128, 256, 512}`` is ≈ 1.0.

DST-3 *is* better than upwind in absolute terms at the same resolution
(L2 ratio ≈ 0.4 at ``n_lon = 128``), but the convergence-rate
diagnostic does not reveal its higher-order nature with this IC + the
production limiter.

Future work (Phase 5 follow-up):
- Test DST-3 on an IC without smooth extrema (e.g. linear ramp on a
  closed domain) to see clean 3rd-order behaviour.
- Add a *peak-position phase-error* diagnostic that is insensitive to
  amplitude clipping.
- Investigate PPM's overflow on the smooth Gaussian periodic setup
  (likely a partial-cell / pole interaction).

For **non-smooth** initial conditions (top-hat / discontinuity) every
scheme drops to rate 1 near the front.  Monotone schemes (tvd, ppm_fct,
dst3 with limiters, weno*) are additionally checked for **no new
extrema** (a hard pass/fail) and mass conservation to machine
precision.

## 5. Gradient correctness (1D — gates Phase 4 AD work)

Every per-term test ships with a `jax.grad` variant that compares the
autodiff gradient against a finite-difference reference on a summary
diagnostic.  This is the per-term AD regression coverage — by the
time the Item 1 metric refactor lands (Phase 4), every wired term has
a gradient test that gates the refactor.

The pattern:

```python
def _summary(initial_state) -> float:
    final = run_test(initial_state)
    return jnp.linalg.norm(final.<diagnostic>)

g_ad = jax.grad(_summary)(initial_state)
g_fd = finite_difference(_summary, initial_state, eps=1e-6)
np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5)
```

`finite_difference` uses central differences and a step calibrated to
the relevant field magnitude.  Mismatch indicates either an AD bug
in the operator (e.g., a hard `jnp.clip` that breaks gradient flow)
or — more commonly — a finite-difference step that's too small for
single-precision arithmetic (run with `JAX_ENABLE_X64=1`).

## 6. Test naming and location

- File:   `tests/ocean/unit/test_per_term_<term>.py`
- Class:  `class TestPerTerm<Term>:`
- Methods:
  - `test_convergence_rate_<scheme>` for each tracer/dycore scheme,
  - `test_gradient_correctness_<scheme>` for AD coverage,
  - `test_invariants_<diagnostic>` for closed-form-vs-actual checks
    (e.g., circle radius for inertial oscillation, mass conservation
    for advection).

Each file imports the toggle factory and the convergence helper from
`_helpers`.  No production code is imported from a test file beyond
what's exported by the factory + the chosen dycore step function.

## 7. What this methodology is *not*

- Not a replacement for the integrated test matrix (gyre, ACC, Eady,
  …).  Those are the *system* tests; per-term tests are the *unit*
  tests.  Both layers needed.
- Not a substitute for the Beckmann–Haidvogel seamount test (Phase
  2A, Recipe #10).  The seamount test is the canonical PGF
  benchmark and needs its own treatment because PGF over varying
  bathymetry has no clean closed form.
- Not a substitute for visual inspection of cubed-sphere v-wind
  snapshots (CLAUDE.md "Validation Rules").  Edge artifacts on
  cubed-sphere are detected by eye, not by L2.

## 8. Status

| Phase 1 component                                | State |
|--------------------------------------------------|-------|
| Disable flags on `LatLonCGridOceanConfig` / `MPASOceanConfig` | landed (Phase 1A) |
| Lat-lon gates (PGF, momentum advection, drag, Coriolis) | landed                     |
| Lat-lon tracer-advection gate                    | planned with Recipe #2 |
| MPAS gates                                        | planned with first MPAS-specific per-term test |
| `make_test_config` factory                       | landed (Phase 1A) |
| `convergence_rate` helper                        | landed (Phase 1B) |
| This methodology doc                             | landed (Phase 1B) |
| Recipe #1 inertial-oscillation test              | planned (Phase 1C.1) |
| Recipe #2 1D advection convergence test          | planned (Phase 1C.2) |
| Recipe #5 linear gravity-wave test               | planned (Phase 1C.3) |
| Gradient-correctness variants                    | planned (Phase 1D) |
| Bit-for-bit regression infrastructure             | planned (Phase 1E)  |

Progress log: `docs/ocean/adcroft_followups.md` § Progress log.

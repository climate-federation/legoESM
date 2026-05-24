# Dycore-tester independent review

*Auditor: Claude (dycore-tester persona). Method: read-only, lens =
validation infrastructure, per-term isolation tests, convergence-rate
gaps, benchmark coverage.*

---

## Headline

From a dycore-tester perspective, the ocean's validation infrastructure
is one rung below where the implementation already sits. Per-term unit
tests against closed-form solutions — inertial oscillation, 1D periodic
tracer advection with order-of-accuracy convergence, 1D Gaussian
diffusion, linear gravity wave, Beckmann–Haidvogel seamount — are
**entirely absent** from `tests/ocean/`. The matrix runs only
integrated cases (rest state, IGW, gyres, Eady, ACC), which means we
cannot localise a failure to one term and we have **no measured
convergence rates** for any operator. Williamson SW infrastructure
already exists for the atmosphere; the `nlev=1` ocean path is real and
clean — the wiring is the work. The single most damaging gap is the
absence of a 1D pure-advection convergence harness: we ship 9 tracer
schemes (`upwind` → `weno7`) without a single test that demonstrates
any of them actually reaches its advertised order. Everything else in
this review descends from that observation.

---

## Per-item assessment

### Item 1 — Coordinate-baked operators

**Validation hook.** How do we *test* the metric-driven refactor works?

The proposed `QuadrilateralCGridMetrics` container refactor is a
*correctness-preserving* change, so the dycore-tester strategy is:

1. **Bit-for-bit lat-lon regression.** Run the entire ocean test
   matrix (61 cases, 31 unique) before and after the refactor on
   regular lat-lon; assert state checksums equal at every output step.
   This requires per-step `state.tree_map(jnp.bitwise_xor, ...)` or
   equivalent deterministic checksum. If checksums diverge, the
   refactor is bugged. The dossier says MOM6 supergrid metrics are
   "consistent by construction" — the test is whether legoESM's
   refactor *is* consistent.

2. **Vector-calculus identities on the new metric container.** Extend
   `tests/unit/test_vector_calculus_identities.py` (already exists for
   the cubed-sphere D-grid) to run on the lat-lon `QuadrilateralCGridMetrics`
   path. The identities to test (per the existing CS test):
   - $\nabla \cdot (\nabla \times \mathbf{v}) = 0$ to machine precision.
   - $\nabla \times (\nabla \phi) = 0$ to machine precision.
   - Discrete Stokes' theorem: $\oint \mathbf{u} \cdot d\mathbf{l} =
     \int (\nabla \times \mathbf{u}) \cdot d\mathbf{A}$ to a single
     floating-point ULP.
   - Discrete divergence theorem analogue.

3. **Williamson-2 cross-coordinate test.** Adding Mercator as a
   metric-provider variant: run Williamson-2 (recipe #6) on regular
   lat-lon, then on Mercator with $\alpha = \pi/4$ — the **same flow
   in different metric coordinates**. The L2(η) errors must match to
   within scheme-order accuracy. Mismatch implies the metric provider
   has a bug, not the operator.

4. **Cubed-sphere panel-boundary visual.** Per CLAUDE.md "Visual
   verification for spatial/grid artifacts" rule: when the refactor
   touches CS operators, **Williamson-2 v-wind snapshot** is the
   reliable detector. Error norms can improve while artifacts shift.

**Audit grounding.** `latlon_cgrid_operators.py` has 127 references to
`cos`, `R_earth`, `radius` (per audit). The dycore-tester position:
**these tests must exist before the refactor, not after**, so the
refactor has something to be bit-for-bit against. Otherwise the only
gate is "code looks cleaner", which is not falsifiable.

### Item 2 — Layered + coordinate-invariant (2a SW limit, 2b PGF)

**2a — SW test matrix that should exist.**

Audit confirmed Williamson SW infrastructure already lives at:
- `tests/unit/test_williamson2_cdgrid.py`
- `tests/test_cases/williamson*.py`
- `tests/atmosphere/shallow_water/test_cases/williamson_mpas.py`

Wired to atmosphere only. The dycore tester demands the **same five
cases** wired to the ocean dycore at `nlev=1`:

| case          | what it tests                                  | pass criterion                                                            |
|---------------|------------------------------------------------|---------------------------------------------------------------------------|
| Williamson-1  | tracer advection convergence on the sphere     | scheme-order convergence rate (see Item 5)                                |
| Williamson-2  | steady-state geostrophic flow; metric handling | L2(h)/H $\lesssim$ 1e-3 at 1° lat-lon day 5; convergence rate 2           |
| Williamson-5  | flow over a mountain (orographic PGF)          | compare to NCAR spectral reference Jakob-Chien 1995 Fig. 7                |
| Galewsky      | barotropic instability symmetry, edge effects  | symmetric vortex pair at day 6; no panel imprint on cubed sphere          |
| Rossby–Haurwitz | nonlinear vorticity dynamics                 | wavenumber-4 pattern preserved for 14 days                                |

Adapter strategy: write a thin shim in `tests/ocean/unit/sw_adapter.py`
that exposes the SW state `(h, u, v)` from the ocean dycore at
`nlev=1` and feeds it through the existing Williamson harness. The
adapter is ~50 LOC; the SW tests are reused, not rewritten.

**Cubed-sphere variant.** Once the CS ocean stabilises (per
`project_cgrid_cubesphere_ocean` and the instability project), the
SW matrix on CS becomes the **single sharpest** edge-effect detector
the project has. Galewsky on CS at C48 will reveal cube imprint that
no integrated ocean test catches.

**2b — Test that confirms coord invariance.**

The PGF schemes (z* `"adcroft"`, density-Jacobian `"smc03"`,
Adcroft–Hallberg–Harrison `"ahh08"`) are coord-invariant *in principle*.
The dycore-tester confirmation is **a single benchmark — the seamount
test (Item 7, recipe #10) — run on each PGF scheme**. A genuinely
coord-invariant PGF has $\max|u|$ on the seamount test that does not
depend on the vertical coordinate choice. We don't currently exercise
that comparison because:
- We don't have the seamount test.
- We don't have `nlev=1` SW comparison against `nlev=20` z* PGF on the
  *same physical setup*.

**Adcroft & Hallberg (2006) invariance test (dycore-tester extension).**
Set up two runs of the seamount test:
1. z* coordinate, $nlev=20$, AHH08 PGF.
2. σ-coordinate (if we add it; or z-coord) on the same problem.

If the PGF is truly coord-invariant, both runs should produce
indistinguishable $\max|u|(t)$ within machine precision (modulo
remapping at the column level). This is the cleanest possible test for
coord-invariance — Alistair did not name it explicitly but it follows
directly from his principle.

### Item 3 — Per-term time stepping

**Test family that proves each term's stepper is right.**

Per-term tests *are* per-term-stepper tests. The two layers cannot be
separated. Specifically:

| term                  | stepper test                                          | recipe |
|-----------------------|-------------------------------------------------------|--------|
| Coriolis              | inertial oscillation; KE drift vs `dt`                 | #1     |
| Tracer advection      | 1D cosine bell, convergence in space at fixed CFL      | #2     |
| Tracer advection (dt) | cosine bell, convergence in time at fixed dx           | new — vary `dt` $\in \{T/4096, T/2048, T/1024, T/512\}$, expect rate equal to outer-integrator order |
| Diffusion             | 1D Gaussian spread; convergence in time + space        | #3     |
| Linear gravity wave   | dispersion test; phase error vs `dt`                   | #5     |
| Barotropic mode-split | gravity wave with split + unsplit, compare phase       | new — verify split adds < 1 % phase error                                                          |
| Implicit vertical mix | analytical implicit-diffusion solution                 | KPP unit test #13                                                                                  |

**The "time-convergence" twin of every space-convergence test.** Audit
showed the current outer-integrator infrastructure
(`src/legoesm/timestepping/dispatch.py`) is not wired into the ocean
dycores. **This is testable.** Take the inertial-oscillation parcel
test and vary `dt` while holding `dx` fixed; the convergence-in-time
rate exposes the actual time scheme used. If the rate is 2 we are
seeing Matsuno (forward–backward); if 1 we are seeing Euler.

**Differentiability of implicit Coriolis (future).** Audit notes that
AD through implicit Coriolis should work via implicit function theorem.
Recipe #14 (AD consistency) covers this: finite-difference gradient on
the inertial-oscillation parcel test, compared against `jax.grad`. If
implicit Coriolis breaks AD, the recipe-1 setup catches it without any
new test.

**Worksheet §4 deprecation hook.** The dycore tester reinforces the
audit recommendation: replace single-axis "outer integrator" with a
per-term table. The right gating is a CI test
`tests/ocean/unit/test_per_term_steppers.py` that asserts
`get_stepper_config(grid="latlon")` returns a dict with one entry per
term, each pointing to a documented stepper. The schema, not the
choices, is what we lock in.

### Item 4 — Energy-conserving Coriolis

**Inertial-oscillation recipe (full).** See recipe #1 in the table.

The disambiguation Alistair owes us (spatial PV-flux vs temporal scheme
vs both) does not block test construction: the inertial-oscillation
test discriminates between them.

- *Spatial alone*: in the parcel test on `nlev=1`, the spatial scheme
  collapses to a $2\times2$ matrix multiply on $(u,v)$ — no PV flux
  to evaluate. The spatial discretisation only matters at higher
  resolution with curvature. Williamson-2 at multiple resolutions is
  the spatial test (recipe #6).
- *Temporal alone*: the parcel test isolates the temporal scheme.
  Matsuno has KE error $\sim (f\,dt)^2$ per inertial period; CN
  conserves KE exactly. The KE drift diagnostic distinguishes them
  cleanly.

**Concrete dycore-tester prediction.** Today both grids use Matsuno
(audit-confirmed, contradicting dossier's "Heun on MPAS"). On the
inertial-oscillation recipe at $f\,dt = 0.036$ (i.e., `dt=360 s`,
`f=1e-4`), expect:
- Matsuno: KE drift $\sim 10^{-3}$ per inertial period.
- CN-Coriolis (future): KE drift $< 10^{-12}$.

If we land the test now, every later commit touching the Coriolis
substep gets a sharp pass/fail signal.

**Spatial EC verification on lat-lon.** The Sadourny EC variant is
asserted in `latlon_cgrid_operators.py:392`; the test for it is
*discrete global KE conservation* on a free-decay run with all
diffusion off (Smag=0, Leith=0, hyperdiff=0, drag=0). On a Galewsky
jet at day 0–6: $|KE(t)-KE(0)|/KE(0) < 10^{-10}$ per day under EC; an
EN scheme drifts at $\sim 10^{-6}$/day. **This test currently does
not exist.**

**MPAS PV-flux change.** Audit finding: `pv_scheme="enstrophy"` is the
MPAS default. Switching to `"energy"` is a one-line config change but
the right gating test is *exactly the same KE-conservation diagnostic*
on the Galewsky jet ported to MPAS — verifies the switch actually does
what the name says.

### Item 5 — Tracer advection

**1D advection convergence rate recipe.** See recipe #2 in the table.

**The full test suite the dycore tester demands** (all on `nlev=1`,
single-tracer, constant velocity, periodic BC):

1. **1D cosine bell, smooth.** $nx \in \{64, 128, 256, 512\}$, one
   period of translation. Report L1, L2, Linf and `log2` convergence
   rate. Pass criteria:
   - `upwind`: rate $\ge 0.95$ (within 5 % of 1).
   - `tvd`: rate $\ge 1.9$.
   - `dst3`: rate $\ge 2.85$.
   - `ppm`: rate $\ge 2.85$.
   - `weno5`: rate $\ge 4.7$.
   - `weno7`: rate $\ge 6.5$.
   - `som`: rate $\ge 2.9$ (3rd-order moment-preserving).

2. **1D square wave, discontinuous.** Same setup, replace cosine bell
   with a top-hat. Pass criteria:
   - All schemes drop to rate 1 near the discontinuity.
   - Monotone schemes (tvd, ppm_fct, dst3 with limiters, weno*): no new
     extrema. **No new extrema** is a hard pass/fail.
   - Mass conservation to machine precision (1e-14) for all schemes.

3. **2D solid-body rotation (Williamson-1 cosine bell).** Williamson
   recipe #4 in the table. Same convergence-rate assertions.

4. **CFL sensitivity.** Per dossier note "DST-3 reaches full benefit at
   CFL > 0.1". Run at CFL $\in \{0.01, 0.05, 0.1, 0.5\}$ and verify
   the rate at CFL=0.5 is at least as good as at CFL=0.05. If rate
   *degrades* at higher CFL, the time-stepping coupling has a bug.

5. **Williamson-1 on cubed-sphere (if/when we add CS ocean).** The 2D
   rotation test on the sphere — exercises panel-boundary halo
   exchange for the tracer scheme.

**Audit-confirmed default mismatch (carry-over from `review_claude_audit.md`).**
- Lat-lon default is `"tvd"` (2nd order), not `"dst3"`. The convergence
  test reveals this immediately.
- MPAS default is `"upwind"` (1st order). Same.

The dycore tester position: **the production default must match what
the convergence test recommends**. If we ship `"tvd"` as default and the
convergence test shows `"dst3"` is 1.5–2 decades better at same cost, the
default is wrong.

**Limiter / differentiability sanity check.** TVD and PPM limiters
introduce $C^0$ but not $C^1$ behaviour. The dycore-tester gradient
check (recipe #14) at a smooth IC will pass; at a near-discontinuous
IC the gradient may be undefined at the limiter switch. This is a
measure-zero issue but worth flagging in the test docstring so it
isn't mistaken for a bug.

### Item 6 — Per-term tests (central item)

**Validation hook.** This is the gating item: every other item collapses
to "we cannot tell if it's right" without per-term isolation tests.

**Inventory of what exists vs what is needed.**

Existing isolation knobs (audit-confirmed):
- `gm_redi: object = None` disables GM/Redi.
- `smag_C`, `leith_C` set to 0 disable those closures.
- `apvm_dt = 0` disables APVM (MPAS).
- `maxvel_barotropic = 0.0` disables MAXVEL clip.
- KPP/convection: togglable via subconfig.

Missing isolation knobs (must be designed before tests are buildable):
- `enable_coriolis: bool` — currently no way to zero out Coriolis cleanly
  on either grid. Forward–backward substep runs unconditionally.
- `enable_pgf: bool` — no `pgf_scheme="zero"` shortcut in lat-lon's
  `_valid_pgf` set (MPAS does have `"zero"` per `ocean_pe_mpas.py:436`).
- `enable_momentum_advection: bool` — momentum advection is baked into
  the dycore step.
- `enable_tracer_advection: bool` — exists implicitly through
  `tracer_advection` strings but no `"zero"` option.
- `enable_drag: bool` — bottom drag is unconditional.

Existing tests (`tests/ocean/`):
- `unit/`: low-level numerical kernels; no closed-form-vs-step tests.
- `validation/`: integrated cases.
- `distributed/`: MPI correctness.
- **Zero** Williamson SW tests on the ocean dycore.
- **Zero** inertial-oscillation tests.
- **Zero** 1D pure-advection convergence tests (despite shipping 9
  tracer schemes from upwind to weno7).
- **Zero** 1D diffusion convergence tests.
- **Zero** flow-past-a-bump tests.

**What needs to happen, in order.**
1. Land the `EnableFlags` NamedTuple inside both
   `LatLonCGridOceanConfig` and `MPASOceanConfig`. Defaults preserve
   current behaviour (all True).
2. Plumb the flags through `_step_impl` (lat-lon) and the MPAS step
   function with Python `if` gating (per JAX rule: static bool capture,
   not `jnp.where`).
3. Write a `make_test_config(term="coriolis_only", grid="latlon")`
   factory returning the minimal config for each per-term recipe.
4. Land the unit tests in order: inertial oscillation (#1) → 1D
   advection (#2) → 1D diffusion (#3) → Williamson-2 ocean (#6) →
   seamount (#10). Each test runs at multiple resolutions and asserts
   the convergence rate, **not just an absolute error threshold**.
5. Standardise a `convergence_rate(L2_list, n_list)` helper in
   `tests/ocean/unit/_helpers.py` so every test reports the same number.

**Why convergence rate, not just L2.** Absolute-error thresholds drift
with grid choice and noise. Convergence rate is the *invariant* — a 2nd-
order scheme produces rate 2 regardless of starting error. Failing
rate is a sharper bug signal than failing threshold.

### Item 7 — Flow past a bump (central item)

**Validation hook.** Beckmann–Haidvogel (1993) is the *only* standardised
benchmark for PGF discretisation quality over varying bathymetry. The
test exists in the literature for 33 years; legoESM ships AHH08, SMC03,
and the standard z* PGF without ever running it.

**The gap.** Audit confirmed `rest_state_stratified_with_land` uses
scalar `H_max = 5500.0` (`experiments/rest_state.py`); the "with land"
variant only adds land masking via `effective_land_lat`. **No
topographic variation.** No PGF cancellation error is measured.
`acc_channel` has a Gaussian ridge but it's a forced run, not a rest
test.

**Recipe.** See expanded recipe #10 in the table above. The canonical
numbers (recap):
- Domain 320 km × 320 km, $H_0=4500$ m + Gaussian seamount $h_s=4000$ m
  $\times \exp(-r^2/L^2)$, $L=25$ km.
- $N^2=10^{-4}$ s$^{-2}$ linear T stratification, constant S.
- $\Delta x \in \{10, 5, 2.5\}$ km, $nlev=20$, z* layers.
- 180 days, no forcing.
- Expected $\max|u|$ at day 180:
  - `pgf_scheme="adcroft"` (z*): O(1 cm/s).
  - `pgf_scheme="smc03"`: O(1 mm/s).
  - `pgf_scheme="ahh08"`: smaller still (per `pgf_ahh08.py` docstring
    quoting Adcroft–Hallberg–Harrison 2008).

**Cross-grid comparison.** Run on MPAS with `pgf_scheme="ahh08"` vs
`pgf_scheme="centered"` (current MPAS default). Expectation: AHH08
$\max|u|$ should beat centred by ≥ 1 decade. If not, the AHH08
implementation has a bug — the seamount test is the canonical detector.

**Why this also unlocks Item 2b.** Coordinate-invariant PGF (Item 2b)
requires *measuring* PGF quality, not just shipping multiple schemes.
The seamount test is that measurement. Without it, "switch to AHH08"
is a vibe; with it, it's a falsifiable decision.

**Alistair-implied vs dycore-tester-demanded extensions.**
- Alistair recommended the rotating variant. The dycore tester adds
  the non-rotating ($f=0$) variant: isolates the PGF from the Coriolis
  feedback that can mask cancellation errors via geostrophic
  adjustment.
- Adds a *convergence-rate* requirement: $\max|u|$ should decrease at a
  scheme-dependent rate with $\Delta x$. Beckmann–Haidvogel didn't
  emphasise this; the dycore tester does, because a bug-free PGF must
  converge.
- Adds AD consistency check (recipe #14) at the seamount setup —
  gradient through the PGF over varying bathymetry is a non-trivial
  AD probe.

---

## Canonical per-term / per-mode test recipe table

Density-first table. "Effort" = S (≤1 wk), M (1–3 wk), L (>3 wk).
"Location" column: `unit/` = pytest CI gate, `matrix/` = visual +
diagnostic outputs in test matrix, `both` = both. Convergence-rate
column gives the rate to expect for a smooth field; first-order is the
expected behaviour at limiter activation or non-smooth features. AD
target = `jax.grad` finite-difference agreement at the indicated digit
count.

| #  | term / mode                       | isolation knob (current state)                                                                                                | IC + BC                                                                                                  | diagnostic                                          | closed-form / target                                                            | convergence rate                            | location          | effort | notes                                       |
|----|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------|-------------------|--------|---------------------------------------------|
| 1  | Coriolis + time stepper           | needs new flag: `enable_pgf=False`, `enable_advection=False`, `gm_redi=None`, `smag_C=0`, `leith_C=0`, `enable_drag=False`     | f-plane (40 km × 40 km doubly-periodic 1-cell-deep), `u₀=(0.1, 0)` m/s, `f=1e-4`, no eta gradient        | parcel trajectory (`u(t), v(t)`); KE(t)             | $u=u_0\cos(ft)$, $v=-u_0\sin(ft)$; period $T_f=2\pi/f≈17.45$ h; ΔKE=0           | 2nd Matsuno; 2nd CN-Coriolis (KE exact)     | unit + matrix     | S      | also gates Item 4                           |
| 2  | 1D periodic tracer advection      | use shallow-water `nlev=1`, set η constant, `u` constant; only tracer module active                                           | 1° lat-lon equatorial strip (or 1D channel), cosine-bell tracer, `u=10 m/s`, periodic BC                 | L1, L2, Linf vs translated bell; shape preservation | $q(x,t)=q_0(x-ut)$; mass conserved to 1e-14; L2 → 0 at scheme order             | upwind 1; tvd 2; dst3 3; ppm 3; weno5 5     | unit (CI gate)    | S      | run at nx=64, 128, 256, 512                 |
| 3  | 1D Gaussian diffusion             | turn off advection + Coriolis + PGF; Smag/Leith off; set `kappa_h>0` only                                                     | 1D channel, $T(x,0)=\exp(-x^2/(2\sigma_0^2))$, $\sigma_0=50$ km, no flow                                 | width $\sigma^2(t)$ vs analytic                     | $\sigma^2(t)=\sigma_0^2+2\kappa t$; mass conserved                              | 2 (centred Laplacian)                       | unit              | S      | verifies Smag/Leith Laplacian operator      |
| 4  | Williamson-1 cosine bell (SW)     | `nlev=1` ocean, `pgf_scheme="zero"`, prescribed solid-body rotation `u`                                                       | cosine bell at $(\lambda_0,\phi_0)=(3\pi/2,0)$ radius $R/3$, height 1000 m, rotation rate $\alpha$       | L1/L2/Linf vs rotated bell after 12 days            | bell returns to start unchanged                                                 | scheme order (see #2)                       | unit + matrix     | M      | reuse `tests/test_cases/williamson1*`       |
| 5  | Linear SW gravity wave            | `nlev=1`, $f=0$, small-amplitude η pulse, no nonlinear advection                                                              | doubly-periodic 1000 km × 1000 km, $H=4000$ m, $\eta=10^{-3}\cos(kx)$ m, $u=0$                           | dispersion $\omega(k)$, energy conservation         | $\omega^2=gHk^2$; phase speed $c=\sqrt{gH}≈198$ m/s                             | 2 (C-grid FB)                               | unit              | S      | with Coriolis: Poincaré $\omega^2=f^2+gHk^2$ |
| 6  | Williamson-2 steady geostrophic   | `nlev=1`, solid-body rotation balanced by Coriolis + free-surface gradient; full operator stack except advection sub-canceled | Williamson (1992) §3 setup, $u_0=2\pi R/(12 \text{ days})$, $\alpha=0$ or $\pi/4$                        | L2(h), L2(u) vs analytical                          | exact steady state; lat-lon L2(h) < 1e-3 after 5 days at 1° expected            | 2                                           | unit + matrix     | M      | also probes Item 1 metric correctness       |
| 7  | Williamson-5 mountain             | `nlev=1`, isolated mountain $h_s$, zonal mean flow                                                                            | $u_0=20$ m/s, mountain $h_s=2000$ m centred at $(3\pi/2,\pi/6)$, radius $\pi/9$                          | energy + enstrophy time series; field at day 15     | NCAR spectral reference (Jakob-Chien et al. 1995 Fig. 7)                        | n/a (compare to reference)                  | matrix            | M      | reveals orographic-PGF error                |
| 8  | Galewsky barotropic instability   | `nlev=1`, perturbed mid-lat jet, full SW                                                                                      | Galewsky (2004) §3, $u_{\max}=80$ m/s, $\eta$-perturbation amplitude $\hat h=120$ m                      | ζ at day 6 (symmetry, panel imprint, grid noise)    | clean symmetric vortex pair; no panel imprint on CS, no cube wavenumber-4       | n/a (visual + harmonic)                     | matrix            | M      | best edge-effect detector on CS             |
| 9  | Rest-state preservation, flat-bot | full stack on; flat bathymetry $H=$ const; $T(z), S(z)$ stratified; $u=v=0$                                                   | 4° × 4° basin or globe; `Vaisala_freq`-like profile; no wind, no flux                                    | max($|u|$), drift in $T$, $S$ vs $t$                | $\max|u| < 10^{-9}$ m/s after 30 days; T drift $< 10^{-6}$ K/day                | n/a (must be machine-precision)             | unit + matrix     | S      | baseline already exists                     |
| 10 | Beckmann–Haidvogel seamount       | full stack; **Gaussian seamount bathymetry**; $u=v=0$ initial                                                                 | 320 km × 320 km, $H_0=4500$ m, seamount $h_s=4000\exp(-(r/L)^2)$, $L=25$ km; $N^2=10^{-4}$ s$^{-2}$; rot | $\max|u|(t)$ over 180 days                          | $\max|u| < 1$ cm/s for z*; $< 1$ mm/s for SMC03; AHH08 best (per `pgf_ahh08.py`)| n/a (compare PGF schemes)                   | matrix (+ unit)   | M      | **single biggest gap**                      |
| 11 | Geostrophic balance steady        | full stack; impose meridional $\eta$ slope; Matsuno Coriolis                                                                  | 1000 km × 1000 km β-plane; $\eta_y=10^{-6}$, $u=-g\eta_y/f$                                              | drift in $u$, mass conservation                     | exact balance; $u$ stays at $g\eta_y/f$ within Matsuno phase error              | 2                                           | unit              | S      | sanity check on Coriolis+PGF coupling       |
| 12 | Linear baroclinic Rossby wave     | β-plane, 2-layer SW, small perturbation                                                                                       | 2-layer 4000 km × 2000 km zonal channel, β=2e-11, $g'=0.02$ m/s², perturb interface as `cos(kx)`         | phase speed $c$; wavelength $\lambda$               | $c=-\beta R_d^2$ with $R_d=\sqrt{g'H_1H_2/(H_1+H_2)}/f$; $c≈-1.5$ cm/s          | 2                                           | matrix            | M      | catches β-term and layer-coupling bugs      |
| 13 | Mixed-layer deepening (KPP)       | 1-column run; no advection; surface buoyancy loss only                                                                        | column, $N^2=10^{-4}$, $B_0=10^{-7}$ m²/s³ surface forcing                                               | MLD$(t)$ vs Kraus–Turner                            | $h(t) = (2 B_0 t / N^2)^{1/2}$                                                  | n/a (closure quality)                       | unit              | M      | direct KPP test                             |
| 14 | AD consistency, per term          | finite-difference vs `jax.grad` on each `_step_impl` term                                                                     | small periodic domain; small `dt`; one outer step                                                        | `\|grad_AD - grad_FD\|_∞ / \|grad_FD\|_∞`           | < 1e-5 (float64) per active term                                                | n/a                                         | unit              | M      | also gates implicit-Coriolis future work    |

**Implementation-ready expansions for the top 3.**

**Recipe 1 — Inertial oscillation (Coriolis isolation).**
- Domain: 40 km × 40 km doubly-periodic f-plane on **`nlev=1`** lat-lon
  C-grid, `nx=ny=8` (this is a parcel test, not a resolution test).
- Bathymetry: flat $H=10$ m (anything; just needs to exist).
- Coriolis: $f=10^{-4}$ s$^{-1}$ ⇒ $T_f=2\pi/f≈62831.85$ s ≈ 17.45 h.
- IC: $u=0.1$ m/s, $v=0$, $\eta=0$, $T,S$ uniform.
- Timestep: `dt = 360 s` (≈ $T_f/175$) and `dt = 90 s` for convergence.
- Run: $4 T_f \approx 251327$ s (4 inertial periods).
- Target diagnostics:
  - Trajectory error: $\sqrt{(u_n - u_0\cos(f t_n))^2 + (v_n + u_0\sin(f t_n))^2}$.
  - KE drift: $|KE(t) - KE(0)|/KE(0)$.
  - Period error: locate first sign-change of $v$, compare to $T_f/4$.
- Pass criterion:
  - Matsuno: trajectory L2 amplitude error $\le c \cdot (f\,dt)^2$ with $c \approx O(1)$;
    after 4 periods, expect KE drift $\lesssim 10^{-4}$ at `dt=360 s`,
    $\lesssim 10^{-5}$ at `dt=90 s` (≈ 4× reduction is the dycore-tester
    sanity check on 2nd-order behaviour).
  - CN-Coriolis (future): KE drift must be $< 10^{-12}$ — exact discrete
    conservation is what motivates the prototype.
- Blocker: requires `enable_pgf=False`, `enable_advection=False`,
  `enable_drag=False` flags that **do not currently exist** in
  `LatLonCGridOceanConfig`. Build the toggles first.

**Recipe 6 — Williamson-2 ocean (steady geostrophic on the sphere).**
- Wire `tests/test_cases/williamson2*` to the **ocean dycore at
  `nlev=1`** via a thin adapter that maps the SW state `(h, u, v)` to
  `(η + H, u, v, T_const, S_const)` and back.
- Resolutions: 4°, 2°, 1°, 0.5° lat-lon; report L2(η), L2(u), L2(v) at
  day 5.
- Two flow orientations: $\alpha=0$ (zonal), $\alpha=\pi/4$ (diagonal —
  exercises off-pole metric handling).
- Reference values (lat-lon): at 1° expect L2(h)/H ≲ 1e-3 after 5 days
  if metric handling and PGF are clean. **Failure mode that should be
  flagged loudly:** L2 grows with resolution or stays flat — implies
  metric error (Item 1) or PGF bug (Item 2b). Convergence rate target =
  2 between successive resolutions (compute as `log2(L2[n]/L2[n+1])`).
- Cubed-sphere variant: same setup at C24, C48, C96. Inspect the
  v-wind field for cube-imprint (Galewsky-style edge artifacts) —
  visual + spherical-harmonic decomposition on wavenumber 4.

**Recipe 10 — Beckmann–Haidvogel seamount (PGF over topography).**
- Geometry: 320 km × 320 km doubly-periodic or wall-bounded f-plane
  patch; $H_0=4500$ m flat bottom + Gaussian seamount centred:
  $h_s(x,y) = h_0 \exp\!\big(-((x-x_c)^2 + (y-y_c)^2)/L^2\big)$ with
  $h_0=4000$ m, $L=25$ km, so seamount peaks at $z=-500$ m.
- Resolutions: $\Delta x=10$, $5$, $2.5$ km. Vertical: $nlev=20$ with z*
  layers ($\Delta z \approx 50$ m near surface, $\Delta z \approx 500$
  m near bottom).
- Stratification: $T(z) = T_0 + \Delta T \cdot z/H_0$ with $T_0=5$ °C,
  $\Delta T=15$ °C (linear T, constant S) giving
  $N^2 \approx 10^{-4}$ s$^{-2}$ — matches Beckmann–Haidvogel 1993 §3.
- Initial condition: $u=v=0$, $\eta=0$, hydrostatic $p$. **No wind, no
  surface flux, no heat flux. No relaxation.**
- $f = 10^{-4}$ s$^{-1}$ (rotating variant); also run $f=0$
  (non-rotating, isolates PGF more cleanly per Item 7 standing question
  #3).
- Time horizon: 180 days. Output $\max|u|$ daily.
- Target numbers (per `pgf_ahh08.py` docstring on ETOPO+ico4, scaled
  expectations on this seamount):
  - `pgf_scheme="adcroft"` (z* standard): $\max|u| \sim 1$ cm/s after
    180 days.
  - `pgf_scheme="smc03"`: $\max|u| \lesssim 1$ mm/s.
  - `pgf_scheme="ahh08"` (requires `eos="wright"`): $\max|u|$ should be
    smaller still.
- Pass criterion: 100× spread between schemes is expected; if all three
  produce $\max|u| \sim$ cm/s, the implementations are likely broken.
- Convergence: $\max|u|$ should *decrease* with resolution for
  scheme-correct PGF, *increase* for buggy PGF (steeper bathymetry
  gradient amplifies the cancellation error).
- Cross-grid: same setup on MPAS (icosahedral variant; need bathymetry
  override hook). MPAS density-Jacobian PGF expected to produce
  smallest $\max|u|$.


---

## What's most urgent

**Single most-damaging absence: the 1D tracer-advection convergence
harness (recipe #2).**

Reasoning. The ocean ships nine tracer schemes — `upwind`, `tvd`,
`ppm`, `ppm_fct`, `dst3`, `dst3_multidim`, `som`, `weno5`, `weno7` —
and *not one* has a CI test that measures its convergence rate against
the analytical translated cosine bell. The dossier asserts DST-3 is
the default; the audit found the actual default is `tvd` for lat-lon
and `upwind` for MPAS. Without recipe #2 we cannot tell:
- which scheme is actually selected at runtime;
- whether the selected scheme reaches its advertised order;
- whether the selected scheme is the right default;
- whether any of the nine implementations is bug-free.

Cost is small (S effort): `nlev=1`, 1D periodic channel, one tracer,
constant velocity, four resolutions, periodic BC. The full recipe is
about 80 LOC of test code plus the `make_test_config(term="tracer_only")`
factory from Item 6.

Payoff is large: gates the default choice (Item 5), gates the
per-term framework (Item 6), demonstrates the convergence-rate
methodology that the entire test programme depends on, and produces
the first quantitative number the dycore tester can actually quote.
The audit calls out a "capability vs default gap" across Items 2b, 4,
and 5 — recipe #2 is the smallest test that turns *any* of those
defaults into a falsifiable claim, and it does so for the cheapest
test in the entire matrix.

Land it first. The seamount test (recipe #10) is more important
scientifically; recipe #2 unblocks the methodology.


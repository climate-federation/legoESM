# Independent code audit — legoESM ocean against the Adcroft items

**Auditor:** Claude (in-conversation).
**Method:** read-only code exploration of `src/legoesm/ocean/`, `src/legoesm/timestepping/`, `scripts/ocean_test_matrix/`, `tests/ocean/`, plus selected design docs. Quantitative summaries where possible.
**Scope:** the 7 items in `docs/ocean/adcroft_followups.md`, with a bias toward findings that *contradict or refine* what the dossier says — i.e., things the synthesis should cross-check.

---

## Headline findings (TL;DR)

1. **The dossier (`docs/legoesm_ocean_model.md`) materially misrepresents several current defaults.** In particular: the dossier says MPAS Coriolis uses Heun; in fact both lat-lon and MPAS use forward–backward (Matsuno) Coriolis. The dossier implies AL81 is the MPAS PV-flux default; in fact the default is `pv_scheme="enstrophy"` and AL81 is not even in the dispatch. The dossier says the tracer default is DST-3; in fact the lat-lon default is `"tvd"` (2nd-order) and the MPAS default is `"upwind"` (1st-order).

2. **Several Adcroft-aligned schemes already exist in source but are not the defaults.** AHH08 (Adcroft–Hallberg–Harrison 2008) and SMC03 (Shchepetkin–McWilliams 2003) PGF schemes are both implemented (`pgf_ahh08.py`, `pgf_smc03.py`) and selectable via `pgf_scheme`. PPM tracer advection is selectable (`tracer_advection in {"ppm", "ppm_fct"}`). DST-3 is selectable. Sadourny EC Coriolis is what lat-lon uses today. *The architectural compliance with Alistair's framework is significantly better than the dossier suggested — but the defaults trail the capabilities.*

3. **Item 1 (coord-baked operators) is real and severe on lat-lon.** `latlon_cgrid_operators.py` (2810 lines) has 127 references to `cos`, `R_earth`, and `radius`. Operators compute `R · cos(lat) · dlon` inline. Cubed-sphere operators are similar by inspection. Metric-driven refactor is unambiguous work.

4. **The validation gaps are large.** Williamson SW tests exist in the repo but are wired to atmosphere, not ocean. No per-term unit tests (inertial oscillation, 1D advection, 1D diffusion). No flow-past-a-bump / seamount rest-state test. `rest_state_stratified_with_land` has uniform `H_max = 5500`; it does not exercise topographic PGF errors.

---

## Per-item findings

### Item 1 — Coordinate baked-in vs metric-driven

**Quantitative finding.** `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py` is **2810 lines** with **127 references** to `cos`, `R_earth`, `radius`. Inline metric construction patterns (lines 199, 343, 658) like `R * cos(lat_face) * dlon`. There is no `MetricBundle` abstraction; operators reach into `grid.radius` and compute `cos(lat)` per call.

**Implication.** Item 1's framing is fully validated. The refactor work is concrete: extract the metric arrays at grid-construction time, pass them as explicit operator arguments, delete all inline `cos`/`R_earth`/`radius` lookups inside the operators. Mercator and cubed-sphere panel coordinates then become metric-provider changes, not operator rewrites.

**Code pointer.** Start at `compute_div_*`, `compute_grad_*`, `compute_curl_*` in `latlon_cgrid_operators.py` — these are the canonical operators and account for most of the inline metric usage.

---

### Item 2a — `nlev=1` orthogonality

**Quantitative finding.** Zero hits for `nlev == 1`, `n_levels == 1`, or similar patterns in `src/legoesm/ocean/`. The dycore is genuinely layered; `nlev=1` is *not* a special case in the code.

**Williamson SW infrastructure.** Present in repo at `tests/unit/test_williamson2_cdgrid.py`, `tests/test_cases/williamson*.py`, `tests/atmosphere/shallow_water/test_cases/williamson_mpas.py`. **Not** present under `tests/ocean/`. The harness exists but only the atmosphere uses it.

**Implication.** Item 2a's "wire SW tests on `nlev=1` ocean path" is feasible and self-contained — the test harness is already in the codebase, just not pointed at the ocean dycore.

---

### Item 2b — Coordinate-invariant PGF

**Quantitative finding.** The codebase contains:

- `src/legoesm/ocean/dynamics/pgf_smc03.py` — Shchepetkin–McWilliams (2003) density-Jacobian PGF.
- `src/legoesm/ocean/dynamics/pgf_ahh08.py` — Adcroft–Hallberg–Harrison (2008) finite-volume PGF.

Both are dispatchable via the `pgf_scheme` config field.

**Lat-lon** (`ocean_pe_latlon_cgrid.py:507`): valid set `{"adcroft", "smc03"}`, default `"adcroft"`. AHH08 not available here.

**MPAS** (`ocean_pe_mpas.py:436`): valid set `{"centered", "adcroft", "smc03", "ahh08", "zero"}`, default `"centered"`.

**Accuracy ordering** (per docstring at `pgf_ahh08.py:61`): on ETOPO+ico4, `"adcroft"` ≈ 1.4 × 10⁻⁶, `"smc03"` ≈ 8.6 × 10⁻⁸, AHH08 better still. AHH08 also requires `eos="wright"` (`ocean_pe_mpas.py:333`).

**Implication.** Item 2b is *closer to done* than the follow-up doc suggests. The infrastructure is in place. The action items are: (i) add AHH08 to lat-lon's `_valid_pgf` set, (ii) switch the production defaults from `"adcroft"`/`"centered"` to a coord-invariant scheme (likely `"ahh08"` where available, `"smc03"` otherwise), (iii) actually *measure* the spurious-current improvement on a seamount test (which is Item 7).

---

### Item 3 — Per-term time stepping

**Quantitative finding.** Lat-lon `_step_impl` (`ocean_model_latlon_cgrid.py:617`) and MPAS step function both have explicit forward–backward (Matsuno) Coriolis substep — `_forward_backward_coriolis_3d` (line 274) and `_forward_backward_coriolis_mpas_3d` (line 56). Per-term stepping is *already happening* — the per-grid `_step_impl` function embeds the per-term scheme.

**Mismatch with dossier.** Dossier §11 said MPAS uses Heun for Coriolis. Source disagrees: it uses Matsuno, the same family lat-lon uses. The MPAS Heun framing appears to be wrong.

**Mismatch with worksheet.** Worksheet §4 presents "outer integrator" as a single axis (Euler / Heun / SSP-RK3 / RK4). The reality is: outer step is forward Euler in the lat-lon `_step_impl`, with the Coriolis substep being Matsuno, the barotropic mode being its own subcycle, and tracer advection having its own sub-stepper. Worksheet §4 is over-simplified.

**Outer integrator infrastructure exists.** `src/legoesm/timestepping/dispatch.py` provides `dispatch_integrator(state, tendency_fn, dt, integrator_name)` with SSP-RK3, SSP-RK34, RK4 available. But the ocean dycores do not call into it — they use their own `_step_impl`. This is a fork in the architecture.

**Implication.** Item 3's per-term audit is more about *documenting and exposing* what is already there than refactoring from scratch. The synthesis should highlight that the "outer integrator" axis on the worksheet should be deprecated and replaced by a per-term table — most of which is already implicit in the code.

---

### Item 4 — Coriolis: energy-conserving

**Quantitative finding (spatial).** Lat-lon: explicit comment at `latlon_cgrid_operators.py:392` — "We use the Sadourny (1975) energy-conserving" form. Already EC. MPAS: `MPASOceanConfig.pv_scheme: str = "enstrophy"` (default), `mpas_config.py:124`. Options: `{"energy", "enstrophy", "mixed"}`. **AL81 is not in the dispatch** (despite dossier §8 listing it).

**Quantitative finding (temporal).** Both grids use forward–backward (Matsuno). This is *partially* energy-conserving but not exactly so. Exact discrete energy conservation for the rotation operator requires Crank–Nicolson / implicit midpoint.

**Implication.**
- Lat-lon is already aligned with Alistair's "lean energy-conserving" — spatial EC + Matsuno. A move to CN-Coriolis is a marginal refinement.
- **MPAS default needs to change.** `pv_scheme="enstrophy"` is *enstrophy-*conserving, the opposite of Alistair's recommendation. Switching the default to `pv_scheme="energy"` is a one-line config change with potentially large downstream impact (and worth empirical validation on the existing matrix).
- The dossier's claim that AL81 is implemented is wrong; if we want AL81 on MPAS we have to add it.

---

### Item 5 — Tracer advection: 3rd-order TVD

**Quantitative finding.** Lat-lon (`state.py:569`): default `tracer_advection: str = "tvd"`. Full options: `{"upwind", "tvd", "ppm_fct", "ppm", "dst3", "dst3_multidim", "som", "weno5", "weno7"}`. MPAS (`mpas_config.py:268`): default `tracer_advection: str = "upwind"`.

**Mismatch with dossier and worksheet.** Both document DST-3 as the default. The actual lat-lon default is 2nd-order TVD; the actual MPAS default is 1st-order upwind. PPM is available on lat-lon (dossier didn't list it).

**Implication.** Alistair's "3rd-order TVD as first choice" is *not* currently the default on either grid. Concrete action: change lat-lon default to `"dst3"` (or `"ppm"` for direct MOM6 comparison), change MPAS default to a 3rd-order option (needs checking what is wired for MPAS specifically — `"upwind"` is the only one we know runs there by default; need to confirm DST-3/PPM availability on Voronoi).

---

### Item 6 — Per-term tests and term toggles

**Term-toggle inventory.** Partial coverage:
- GM/Redi: `gm_redi: object = None` (`None` disables) — `state.py:570`, `mpas_config.py:253`.
- Smagorinsky, Leith: coefficient $> 0$ enables.
- MAXVEL clip: `maxvel_barotropic = 0.0` disables.
- APVM: `apvm_dt = 0` disables (`mpas_config.py:126`).
- KPP, convection: presumed togglable via subconfig (not verified in this audit).

What is **not** clearly togglable: Coriolis, PGF, horizontal advection (you can't trivially "turn off the PGF" the way you can disable GM/Redi). Item 6's term-toggle API needs design work.

**Test inventory.** `tests/ocean/` has subdirs `unit/`, `validation/`, `distributed/`. No top-level Williamson tests for ocean. No inertial-oscillation test. No 1D pure-advection convergence test for the ocean tracer schemes. No seamount/bump rest-state test.

**Implication.** Item 6's test-harness work is real and large. The framework for "isolate one term" requires extending the config dataclasses with explicit `enabled` flags for the un-togglable terms, plus a `make_test_config(term=...)` factory.

---

### Item 7 — Flow past a bump

**Quantitative finding.** `src/legoesm/ocean/experiments/rest_state.py` uses scalar `H_max: float = 5500.0` as the bathymetry. The "with land" variant adds land masking via `effective_land_lat`, not topographic variation. The current `rest_state_stratified_with_land` test therefore does *not* exercise PGF errors over varying bathymetry.

**Bathymetry tests we do have.** `acc_channel` has a Gaussian ridge (`experiments/acc_channel.py:140`) but it is a forced run, not a rest-state preservation test. `lock_exchange` and `overflow` have sloping bottoms but are dynamic tests by design.

**Implication.** Item 7's gap is confirmed: there is no Beckmann–Haidvogel / Shchepetkin–McWilliams seamount-in-rest-stratification test in the matrix. Adding one is straightforward — it would reuse the `rest_state.py` initial-condition machinery with a Gaussian bump bathymetry override and is the right benchmark for choosing the production PGF default (Item 2b).

---

## Cross-cutting observations

1. **Capability vs default gap.** Across Items 2b, 4, 5, the better scheme is already implemented but not selected by default. The shortest path to Adcroft-alignment is often a `default = "X"` change, not a new implementation.

2. **Dossier accuracy debt.** The dossier we sent into the conversation overstated several alignments (AL81 PV-flux, DST-3 default, Heun Coriolis on MPAS). For the Alistair follow-up communication, an erratum is warranted.

3. **Per-term framework is half there.** The `_step_impl` per-grid functions already do per-term stepping (forward–backward for Coriolis, sub-stepping for barotropic, etc.) but it is not exposed as a config axis. The `src/legoesm/timestepping/dispatch.py` outer-integrator module exists but is not wired into the ocean dycores. These should converge.

4. **Architectural debt is concentrated in the operators.** Item 1's coord-baked operators are the single largest piece of work implied by Alistair's framework. It also unlocks Item 2b (a coord-invariant PGF needs metric-driven operators) and clarifies Item 5 of the broader grid story (Mercator, cubed-sphere, tripolar all become metric-provider variants).

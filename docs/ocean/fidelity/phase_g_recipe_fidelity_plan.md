# Phase G — Recipe Fidelity

## Status

**Infrastructure complete; first Veros acceptance run pending Veros install.**
Branch ``matching_Veros_oracle`` (10 commits ahead of ``main``) carries the
full Phase G implementation; 113 new tests passing; zero regressions in the
pre-existing 79 advection / mixing / GM-Redi / momentum-diagnostics tests.

What landed (session 2026-05-28):

- **G.0a — scheme wires + constants override.** Centralized TVD limiters
  (closes a "no duplicate numerics" violation), wired
  ``tracer_advection="superbee"``, ``A_h_cos_power`` config field,
  verified implicit vertical-friction wiring, recipe-level
  ``override_constants(**VEROS_CONSTANTS)`` mechanism with 10-site
  snapshot registry.
- **G.0b — tendency probe harness.** ``probe_latlon_cgrid`` for
  per-process tendencies, region masks, per-region metrics
  (L1/L2/L∞/sign-match/pattern-corr).
- **G.0c — ACC recipe + Veros↔legoESM state bridge + comparison driver.**
  ``build_acc_recipe()`` matches ``veros.setups.acc.acc.ACCSetup``
  parameter-for-parameter. ``veros_snapshot_to_legoesm_state``
  handles halo strip / time-level selection / vertical reversal /
  axis transpose. ``scripts/validate/ocean_fidelity/compare_tendencies_acc.py``
  drives the end-to-end comparison.
- **G.1a — TKE closure (Gaspar 1990 / Burchard 2002)** as
  ``vertical_mixing="tke"``, wired into the implicit-mixing path.
- **G.1b — Veros nonlin3 EOS** as ``eos="veros_nonlin3"`` (bit-for-bit
  parity to ``veros/core/density/nonlinear_eq3.py``). UNESCO 1980 EOS
  also landed as ``eos="unesco80"`` for general-purpose use.
- **G.1c — Redi taper width** (``GMRediConfig.taper_width_frac``
  exposing Veros's ``iso_dslope / iso_slopec``).
- **G.x — leapfrog + AB2 + Robert-Asselin** standalone scheme module
  (full outer-integrator dispatch wiring deferred — see below).

**Two corrections from reading Veros source** (Veros cloned at
``/home/dbalwada/veros``):

1. **Veros is C-grid, not B-grid.** Verified against
   ``veros/variables.py``: ``U_GRID = ("xu", "yt", "zt")``,
   ``V_GRID = ("xt", "yu", "zt")``. The "B-grid parallel core"
   item is **no longer required** — staggering matches legoESM's
   lat-lon C-grid exactly.
2. **Veros's ``eq_of_state_type=3`` is NOT Jackett-McDougall 1995.**
   It's a quadratic-in-T polynomial with ``betaS=0`` (no salinity
   dependency). Verified against ``veros/core/density/nonlinear_eq3.py``
   and ported as ``eos="veros_nonlin3"`` with bit-for-bit parity.

See ``docs/ocean/fidelity/phase_g_veros_recipe_audit.md`` for the
per-dimension status table.

**Remaining work — single user-driven step blocks the first
acceptance run:**

```bash
.venv/bin/pip install -e /home/dbalwada/veros
.venv/bin/python scripts/validate/ocean_fidelity/compare_tendencies_acc.py \
    --write-report docs/ocean/fidelity/veros_acc_tendency_comparison.md
```

(The auto-mode classifier blocked the pip install during the session
— external-source code execution is a manual approval step.)

After the first run lands, expect iteration: real comparisons surface
real bugs and recipe gaps. The harness is designed for that loop.

## Motivation

Phases A–F established **tier-3 fidelity**: free-run bulk statistics (``T_min``,
``u_mean``, ``ke_mean``, AMOC, ACC) from legoESM agree with Veros / NEMO references
at 5% tolerance over short integrations. The "bulletproof" claim holds for every
fidelity-comparison metric the present harness can drive on a laptop (44 / 44).

The residual vulnerability is **interpretive**. On ``eady_uniform`` the Veros
comparison passes with ``u_mean`` 32% off and ``u_abs_max`` 20% off (both PASS because
of the abs-floor rule for near-zero references). From the bulk-metric harness alone we
cannot tell whether the cause is:

1. A bug in legoESM (sign error, missing term, wrong normalization, wrong constant).
2. A *scheme* difference (legoESM defaults to centered-2nd-order tracer advection
   where Veros uses upwind; different Coriolis form; different EOS coefficients).
3. Genuine sensitivity to an initial-condition transient, with no actionable signal.

Free-run statistics smear all three together. **Tendency-level comparison localizes
them.** Given an identical ocean state, legoESM and the reference model should produce
the same tendency for each isolated process (tracer advection, momentum advection,
PGF, Coriolis, mixing, EOS-derived density) within the discretization-truncation
error of whichever scheme is coarser. Disagreement at that level is either a bug or a
recipe mismatch — both actionable. Phase G adds this **tier-2** comparison layer.

## The fidelity tier model

| Tier | What is compared | Sensitivity | Cost | Status |
|------|------------------|-------------|------|--------|
| 1 | Analytic / semi-analytic benchmark | Discretization order, grid metric | Low | Partial (IGW; geostrophic adjustment WIP per commit ``5f4d18cf``) |
| 2 | Per-tendency on frozen reference state | Implementation correctness | Medium | **Phase G** |
| 3 | Bulk free-run statistics | Integrated agreement | Medium | Phases A–F ("bulletproof", 5% gate) |
| 4 | Climatology / CMIP ensemble overlap | Configuration realism at decadal+ | High | Phase F skeleton (``docs/ocean/long_runs/results_*_skeleton.md``) |

Tiers are complementary, not redundant. A model can pass tier 3 while failing tier 2
(compensating bugs that cancel in integrated statistics); pass tier 2 while failing
tier 3 (correct schemes wired into the wrong time integrator or boundary handling);
pass tier 4 while failing tier 1 (gross discretization errors hidden by ensemble
spread). **Phase G fills the tier-2 hole.**

## The recipe framing

A framing change accompanies Phase G. The legoESM vision is that **any production
ocean model is reachable as a configuration of legoESM, given the right recipe of
choices** (EOS, advection, Coriolis, mixing closure, time integrator, grid,
constants). What legoESM additionally enables is *combinations* of choices that no
single existing model exposes.

Under this framing, "Veros comparison" is not a one-off validation effort. It is the
**acceptance test for legoESM-Veros mode** — a specific dispatch of choices that
becomes a durable artifact in the repo:

```
configs/recipes/
├── README.md                   # how recipes encode reference-model choices
├── veros_4deg_global.yaml      # legoESM-Veros recipe
├── mom6_omip.yaml              # legoESM-MOM6 recipe
└── mitgcm_aimwc.yaml           # legoESM-MITgcm recipe
```

Each recipe pins every scientific knob legoESM exposes to the reference's canonical
value. Tier-2 tendency comparison on a snapshot from the reference model becomes the
**acceptance gate for the recipe**: if every isolated process tendency matches within
discretization error on the same state, the recipe is correct and the legoESM mode
faithfully reproduces the reference.

Three downstream consequences:

1. **Recipe diffs are scientifically interesting.**
   ``diff veros_4deg_global.yaml mom6_omip.yaml`` is a comparable enumeration of where
   two mature ocean models disagree — same codebase, same harness, decisions made
   explicit. That is a publication, not just a test fixture.
2. **Novel paths become quantifiable.** Mixed recipes
   (e.g. MOM6 PGF + Veros tracer advection + MPAS Voronoi grid + adjoint-friendly time
   integrator) can be characterized relative to the validated reference recipes; the
   deltas have physical interpretation rather than "legoESM said something".
3. **The "why not just use MOM6" question has a clean answer.** legoESM-MOM6-mode
   plus autodiff plus GPU plus surgical recipe modification is something MOM6 itself
   cannot give you. The recipe acceptance gate is the proof that the underlying
   physics is faithful.

## Phase G plumbing

The tier-2 harness needs five components. Each is described as a deliverable.

### Reference snapshot ingestion

- Input: NetCDF / HDF5 ocean state from the reference model.
- Output: legoESM state pytree on a matching grid.
- **Constraint**: avoid grid projection in the first pass. Run legoESM on the
  reference's native grid (lat-lon C-grid at the reference's resolution) so no
  interpolation is required. Cross-grid comparison (reference on lat-lon → legoESM on
  cube / MPAS) is a later sub-phase, to keep interpolation noise out of the initial
  tendency-error budget.
- Module: new ``src/legoesm/ocean/fidelity/ingest_<ref>.py`` per reference, plus a
  small shared schema.

### Per-term tendency dumps from the reference

Each mature model can be configured to emit per-process tendencies as diagnostics:

- **Veros** — tendency arrays are accessible in Python from the state object
  (e.g. ``vs.du_mix``, ``vs.dtemp_hmix``, ``vs.flux_top``); easiest first target.
- **MOM6** — ``MOM_diagnostics`` exposes per-term ``du/dt`` and ``dT/dt`` outputs
  including ``PGF_u``, ``Cor_u``, ``visc_rem_u``, ``T_advection_xy``.
- **MITgcm** — ``diagnostics_pkg`` provides per-package per-term output with similar
  granularity.

One-time configuration effort per reference. Documented as a sub-doc per reference
under ``docs/ocean/fidelity/phase_g_<ref>_diagnostics.md``.

### Isolated tendency calls in legoESM

Existing legoESM tendency modules already expose per-process functions usable without
the timestepping loop:

- ``ocean/dynamics/ocean_tendency_common.py`` —
  ``iterate_eos_and_pressure_anomaly``, ``apply_sponge_tracer_relaxation``,
  ``apply_freshwater_virtual_salt_top``.
- ``ocean/dynamics/barotropic_common.py`` — ``compute_filter_weights``, ``bebt_blend``.
- Per-scheme modules under ``ocean/dynamics/`` (advection schemes, PGF variants) and
  ``ocean/physics/`` (vertical mixing, convection, bottom drag).

Phase G adds a thin harness — ``ocean/fidelity/tendency_probe.py`` — that takes a
frozen state pytree plus a recipe and returns a dict keyed by process name. No
timestepping, no scan loop, no donation. JIT-compiled per (recipe, snapshot) pair so
the comparison itself stays cheap.

### Recipe specification

- Format: YAML, mirroring ``configs/`` driver-config conventions.
- Required keys per recipe:
  - EOS choice + coefficients (``rho_0``, ``alpha_T``, ``beta_S``, EOS variant).
  - Momentum advection scheme.
  - Tracer advection scheme.
  - Coriolis form (energy-conserving, enstrophy-conserving, EEN, Hollingsworth-corrected).
  - Vertical mixing closure + parameters.
  - Time integrator (recorded but not exercised in tier-2).
  - Grid type + resolution + bathymetry source.
  - Constants overrides (must reference ``legoesm.constants`` entries, not literals).
- Validator: every recipe must round-trip through ``ExperimentConfig.validate_strict``
  (cf. dispatch-discipline rules in ``CLAUDE.md``) before being accepted as a Phase G
  recipe. Unknown scheme literals must raise.

### Comparison metrics

A single global L2 hides spatially-localized bugs. Per process, the harness reports:

- **Per-region L1 / L2 / L∞** — interior, lateral boundary, equator (|lat| < 5°),
  mixed-layer, abyssal.
- **Per-level error norms** — diagnose vertical-coordinate-dependent bugs.
- **Sign-match map** — fraction of grid cells where
  ``sign(legoESM) == sign(reference)``.
- **Pattern correlation** — global and per-region.
- **Histogram of normalized residual** — ``(legoESM − ref) / |ref|`` clipped to a
  finite range, used to catch a small tail of large errors that pattern correlation
  smooths over.

Output per (recipe, snapshot) pair: one ``.md`` report + per-process PNGs, mirroring
the ``bulletproof_run_*.md`` format from Phases A–F.

## Phased rollout

### G.0 — First Veros ACC acceptance run

Anchored on **Veros's built-in setup gallery** (``veros.setups.*``),
specifically ``veros.setups.acc.acc.ACCSetup`` (30 × 42 × 15
re-entrant channel, the smallest canonical Veros gallery case).

**G.0a — scheme prerequisites (DONE).**
1. ✅ ``tracer_advection="superbee"`` wired on lat-lon C-grid + MPAS;
   limiters centralized in ``ocean/dynamics/_flux_limiters.py``.
2. ✅ Recipe-level constants override via
   ``override_constants(**VEROS_CONSTANTS)``; 10-site snapshot
   registry covers ``constants.{g, rho_ocean, c_sw}`` and the eos /
   coupler aliases.
3. ✅ ``A_h_cos_power: int = 1`` on ``LatLonCGridOceanConfig``
   exposing Veros's ``hor_friction_cosPower``.
4. ✅ Implicit vertical-friction wiring verified
   (``implicit_vertical_mixing=True`` routes correctly).

**G.0b — tendency probe harness (DONE).**
``probe_latlon_cgrid`` + ``build_region_masks`` +
``per_region_metrics`` + ``compare_probe_results``. Momentum closure
test mirrors the production diagnostic-closure invariant.

**G.0c — ACC recipe + Veros↔legoESM state bridge + driver (DONE).**
``build_acc_recipe()`` matches ACCSetup parameter-for-parameter
(verified by reading Veros source).
``veros_snapshot_to_legoesm_state`` strips Veros's halos, selects
the τ time level, reverses the vertical axis, transposes (x,y)
↔ (lat,lon).
``scripts/validate/ocean_fidelity/compare_tendencies_acc.py`` orchestrates
the comparison and emits a Markdown report.

**Acceptance gate (per-process, on a frozen ACC snapshot):**
- Pattern correlation > 0.99 on each tendency.
- Per-region L2 within 2× scheme truncation error.
- Sign-match > 0.999 in the interior.

**Goal: validate the harness, not the model.** The first comparison
will surface a legoESM bug or an unaccounted-for recipe mismatch
— that's the point. The harness is now ready to drive that loop.

### G.1 — Full Veros recipe across the canonical gallery

- **G.1a — TKE closure (DONE).** Ported as ``vertical_mixing="tke"``
  with Gaspar 1990 / Burchard 2002 closure equations, configurable
  via ``TKEConfig`` (matches Veros's ``c_k``, ``c_eps``, ``alpha_tke``,
  ``mxl_min``, ``tke_mxl_choice``, ``kappaM_min``, ``kappaH_min``).
  Wired into ``compute_vertical_K_profiles`` via the implicit-solver
  path. Prognostic TKE state-pytree wiring (carry across timesteps)
  is the documented follow-up; current implementation uses Mode B
  iterated diagnostic which produces K profiles within a few percent
  of full prognostic equilibrium for typical ocean conditions.
- **G.1b — Veros nonlin3 EOS (DONE).** Veros's
  ``eq_of_state_type=3`` is a quadratic-in-T polynomial with
  ``betaS=0`` (not JM95 — corrected from the original audit draft
  via direct read of ``veros/core/density/nonlinear_eq3.py``).
  Ported as ``eos="veros_nonlin3"`` with bit-for-bit parity verified
  by direct formula comparison. UNESCO 1980 EOS also landed as
  ``eos="unesco80"`` (general-purpose nonlinear seawater EOS,
  separate from Veros parity).
- **G.1c — Redi taper width (DONE).** ``GMRediConfig.taper_width_frac``
  exposes Veros's ``iso_dslope / iso_slopec`` ratio. Default 0.1
  preserves legoESM's pre-2026 behavior bit-exactly; Veros ACC's
  ``iso_slopec=0.01, iso_dslope=0.005`` maps to ``0.5``.
- **G.1d — Re-run ``legoesm-veros-acc`` with all tendencies (PENDING
  Veros install).** Driver is ready (``compare_tendencies_acc.py``);
  the run produces per-process L1/L2/L∞/sign-match/pattern-corr
  metrics on every tendency in ``LatLonProbeResult`` against Veros's
  ``du_*`` / ``dv_*`` / ``dtemp_*`` / ``dsalt_*`` diagnostic arrays.
- **G.1e — Extend to ``legoesm-veros-global-4deg``** wrapping
  ``GlobalFourDegreeSetup``. Recipe builder analogous to
  ``build_acc_recipe()`` — straightforward once ACC is green.

**Outcome:** ``configs/recipes/veros_acc.yaml`` (and later
``veros_global_4deg.yaml``) become versioned, durable artifacts. The
repo can claim "legoESM contains Veros's canonical-gallery setups
as faithful configurations" *and* point at the per-process
acceptance metrics that prove it.

### G.2 — MOM6 recipe

- New ingest from MOM6 NetCDF output + ``MOM_diagnostics`` per-term arrays.
- New recipe ``configs/recipes/mom6_omip.yaml``.
- Acceptance: same per-process gates on a MOM6 OMIP snapshot.
- **Expected to surface gaps** in legoESM's scheme menu (e.g. specific PGF
  formulations, particular mixing closures, particular MOM6 EOS variants). Each gap
  is a feature-work item, not throwaway scaffolding. The recipe doc tracks them.

### G.3 — MITgcm recipe

- Same pattern. Phase G machinery is by now a template.
- Output: ``configs/recipes/mitgcm_aimwc.yaml``.

### G.4 — Mixed recipes (research output)

- Once G.1–G.3 are green, run combinations such as MOM6 PGF + Veros tracer advection +
  MPAS grid + adjoint-friendly time integrator.
- Characterize deltas relative to the validated parent recipes.
- This is where the legoESM differentiator becomes publishable research, not
  internal QA.

## Open questions

1. **Snapshot timing.** Frozen states from a reference's transient
   (e.g. first day) vs. quasi-equilibrium (after spin-up) probe
   different parts of the operator. The driver currently runs ACC
   for ``DT_MOM_S`` (4800 s, one Veros dt_mom step) — short enough
   that the snapshot is essentially the IC plus one step. Once
   the first comparison lands, the protocol for picking
   acceptance-run length needs definition.
2. **Implicit-vs-explicit vertical mixing.** Veros uses implicit
   vertical viscosity and diffusivity. legoESM's TKE port (G.1a)
   computes K profiles via the implicit-solver fallback path —
   matched. A snapshot's "vertical-mixing tendency" comparison is
   well-defined as long as both sides report the K profiles at
   matching interfaces.
3. **Time-integrator coupling.** Veros uses leapfrog + AB2 +
   Robert-Asselin; legoESM uses SSP-RK3 by default. The leapfrog
   scheme was implemented standalone
   (``timestepping/leapfrog_ab2.py``, deleted 2026-10 as unwired; recover from git e86e0b86d) but not wired into legoESM's
   outer-integrator dispatch yet (would require
   ``SegmentCarry`` extension with a τ-1 carry field).
   Tier-2 compares un-integrated tendencies, so this does NOT
   block G.0 acceptance. It matters for tier-3 free-run match —
   tracked as a follow-up.
4. **Recipe completeness.** EKE (Eden & Greatbatch 2008) is enabled
   in Veros ACC. **UPDATE (2026-05-29):** legoESM now HAS the
   Eden-Greatbatch prognostic-EKE closure (`lateral_mixing/eke.py` +
   `GMRediConfig.eke`); its `kappa_GM=c_k·L·√E` formula + ACC params
   were verified against Veros's `K_gm` to machine precision (gate E9;
   see the strategy doc §8 EKE ledger). It is built + truth-tier
   verified (E1–E8) but NOT yet flipped on in the recipe: legoESM's
   mixing length `L` (Visbeck Rossby radius) does not yet match Veros's
   `eke_len` (which adds the eddy-dependent Rhines limiting), so the
   prognostic kappa_GM would be ~25× too large. Adoption is deferred
   behind the `eke_len` mixing-length variant (§8 "next must-build").
   The current recipe uses constant GM kappa (≈Veros's effective GM).
5. **Cube and MPAS grids.** Phase G is lat-lon C-grid only for now
   (ACC uses lat-lon natively). Cube + MPAS recipes are a later
   sub-phase that needs the lat-lon → cube interpolation question
   settled.
6. **Coupled fluxes.** Phase G is ocean-only. Air-sea fluxes from
   a fixed atmosphere snapshot are tractable; full coupled-system
   recipes are a later phase.

## Answered questions (during this session)

- **Veros's horizontal staggering** — C-grid (verified
  ``veros/variables.py``). The originally-conjectured B-grid
  delta does not exist.
- **Veros's nonlinear EOS family** — ``eq_of_state_type=3`` is
  a quadratic-in-T polynomial with ``betaS=0``, NOT
  Jackett-McDougall 1995. Bit-for-bit ported as
  ``eos="veros_nonlin3"``.
- **Veros tendency-variable names for the bridge** — ``du_cor``,
  ``du_mix``, ``du_adv`` (momentum); ``dtemp_{hmix,vmix,iso}`` and
  ``dsalt_{hmix,vmix,iso}`` (tracer). Verified against
  ``veros/variables.py``.

## Relation to existing work

- **Predecessor**: Phases A–F (``docs/ocean/fidelity/bulletproof_summary.md``). Phase G
  strengthens, does not replace, the bulk-metric gate.
- **Adjacent**: ``docs/dev-notes/ocean_validation_improvement_plan.md`` predates the recipe
  framing; it remains as historical context and is not modified here.
- **Existing harness reused**: ``scripts/validate/ocean_fidelity/compare_legoesm_vs_veros.py``,
  ``scripts/validate/ocean_fidelity/compare_legoesm_cube_vs_latlon.py``, and
  ``scripts/validate/ocean_fidelity/run_comparison.py`` provide the reference-driver
  scaffolding that G.0 will extend (snapshot dump-and-load) and G.1 will reuse
  end-to-end.
- **Existing modules called**: ``ocean_tendency_common.py``,
  ``barotropic_common.py``, the scheme modules under ``ocean/dynamics/`` and
  ``ocean/physics/``.

## Deferred work

- **Prognostic TKE state**. ``tke`` field on
  ``LatLonCGridOceanState`` + state-update wiring. Current
  implementation uses Mode B diagnostic iteration which produces
  K profiles within ~few % of full prognostic equilibrium —
  adequate for the first ACC acceptance run; the prognostic
  upgrade tightens the tier-2 match.
- **Leapfrog outer-integrator dispatch.** Standalone scheme was
  delivered (``timestepping/leapfrog_ab2.py``, deleted 2026-10 as unwired; recover from git e86e0b86d). Wiring it as
  ``outer_integrator="leapfrog_ab2"`` requires extending
  ``SegmentCarry`` with a τ-1 carry field. Tier-2 unaffected;
  needed for tier-3 free-run match.
- **EKE closure (Eden & Greatbatch 2008).** **DONE (2026-05-29):**
  built as a canonical block (`lateral_mixing/eke.py` + `GMRediConfig.eke`),
  truth-tier verified (E1–E8), closure form + ACC params oracle-verified
  vs Veros `K_gm` to machine precision (E9). Remaining: the `eke_len`
  mixing-length variant (Rhines limiting) before recipe adoption — see
  the strategy doc §8 EKE ledger.
- **Veros gallery beyond ACC.** ``GlobalFourDegreeSetup``,
  ``global_1deg``, ``global_flexible``, ``north_atlantic``,
  ``wave_propagation``. Each is a new recipe builder analogous
  to ``build_acc_recipe()`` once the ACC pattern is proven.
- **MOM6 + MITgcm recipes** (originally numbered G.2 / G.3 below).
  Phase G machinery is now general enough — the per-reference
  work is the per-model state bridge + per-term diagnostic
  extraction.
- **Higher-level docs reorganization** (``ocean_experiments/`` +
  ``ocean_fidelity/`` + ``ocean_long_runs/`` have overlaps).
  Intentionally deferred; Phase G is a *next layer*, not a
  reorganization.
- **Atmosphere / coupled-system recipe framing.** Phase G is
  ocean-scoped. Whether the same approach is applied to the
  atmosphere is a downstream question.

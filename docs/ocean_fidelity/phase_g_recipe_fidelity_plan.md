# Phase G — Recipe Fidelity (planning)

## Status

**Planning.** Phases A–F are complete (see ``docs/ocean_fidelity/bulletproof_summary.md``).
This document captures the framing and a phased rollout for the next layer of ocean
fidelity work. Implementation begins after this doc is reviewed and accepted.

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
| 4 | Climatology / CMIP ensemble overlap | Configuration realism at decadal+ | High | Phase F skeleton (``docs/ocean_long_runs/results_*_skeleton.md``) |

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
under ``docs/ocean_fidelity/phase_g_<ref>_diagnostics.md``.

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

### G.0 — First concrete step: tracer advection vs Veros ACC gallery setup

Anchored on **Veros's built-in setup gallery** (``veros.setups.*``), not on
custom legoESM ports. The legoESM-Veros recipe is defined by matching
Veros's *canonical configurations* — that is what "running Veros" means in
practice.

- Recipe: ``legoesm-veros-acc`` wrapping
  ``veros.setups.acc.acc.ACCSetup`` (30 × 42 × 15 re-entrant channel,
  pyOM2-derived, the smallest canonical Veros gallery case). Already
  wrapped at ``legoesm.ocean.fidelity.veros_configs.acc_channel``.
- Snapshot: short ACC run via the existing Veros runner.
- Process: tracer advection — **superbee**, the Veros canonical (Veros
  enables ``enable_superbee_advection=True``).
- Prerequisites that block the run (see audit doc for details):
  1. Wire ``tracer_advection="superbee"`` in legoESM (the ``"tvd"``
     dispatch currently calls Van Leer, not Sweby — though
     ``_sweby_limiter`` exists in ``ocean/advection.py``).
  2. Recipe-level constants override (``g``, ``R_earth``, ``rho_ocean``,
     ``omega`` differ between Veros and legoESM at the 0.02–0.1% level).
  3. ``A_h_cos_power`` field in ``HarmonicMixingConfig`` for cos(lat)
     scaling.
  4. Verify implicit vertical-friction dispatch wiring.
- Acceptance:
  - Pattern correlation > 0.99 on ``dT/dt`` due to advection.
  - Per-region L2 within 2× scheme truncation error.
  - Sign-match > 0.999 in the interior.
- Goal: **validate the harness, not the model.** The first comparison
  will surface either a legoESM bug or an unaccounted-for recipe
  mismatch; the first round is about confirming the harness measures
  what we think it measures.

### G.1 — Full Veros recipe across the canonical gallery

After G.0 lands:

- **G.1a — Port Veros's TKE closure** to legoESM as
  ``vertical_mixing="tke"``. ACC turns on TKE; without this the
  mixing-tendency comparison on ACC must fail. ~400–600 LOC + tests.
- **G.1b — Port Jackett-McDougall 1995 EOS** as ``eos="jm95"``. Needed
  for any gallery setup that enables ``eq_of_state_type=3``. ~80 LOC.
- **G.1c — Veros Redi taper parameter mapping** in ``gm_redi.py`` so
  ``iso_dslope`` / ``iso_slopec`` / ``iso_steep`` pin cleanly.
- **G.1d — Re-run ``legoesm-veros-acc`` with all tendencies**.
  Acceptance: per-process gates pass on momentum advection, PGF,
  Coriolis (with C-grid stencil delta documented), GM/Redi, vertical
  mixing (TKE), lateral friction.
- **G.1e — Add ``legoesm-veros-global-4deg``** wrapping
  ``GlobalFourDegreeSetup``. Full-domain version of the same
  acceptance gates.
- Outcome: ``configs/recipes/veros_acc.yaml`` and
  ``configs/recipes/veros_global_4deg.yaml`` become versioned, durable
  artifacts. The repo can now claim "legoESM contains Veros's
  canonical-gallery setups as faithful configurations."

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

1. **Snapshot timing.** Frozen states from a reference's transient (e.g. first day)
   vs. quasi-equilibrium (after spin-up) probe different parts of the operator. Tier-2
   probably needs both. Protocol needs definition before G.0.
2. **Implicit-vs-explicit vertical mixing.** Veros uses implicit vertical viscosity
   and diffusivity; legoESM uses explicit by default. A snapshot's "vertical-mixing
   tendency" is well-defined for explicit, more subtle for implicit. Tier-2 for
   mixing may need a residual-budget formulation rather than direct tendency-array
   comparison.
3. **Time-integrator coupling.** Strictly, "tendency at time t" is well-defined;
   "tendency *as applied by the integrator*" differs (RK3 vs leapfrog vs split forward
   Euler). G.0 compares un-integrated tendencies. A separate tier-2.5
   (integrator-applied tendency) is worth distinguishing if it turns out integrators
   are a major source of free-run divergence.
4. **Recipe completeness.** Some Veros / MOM6 choices may not have a legoESM
   implementation yet. Each gap becomes a feature-work item under Phase G; the recipe
   YAML tracks them with explicit ``status: missing`` entries until closed.
5. **Cube and MPAS grids.** ``bulletproof_summary`` notes ``lock_exchange`` still
   diverges on the global cube (face-seam PGF mode). Phase G on the cube requires
   either the lat-lon → cube interpolation question (and its noise budget) settled,
   or deferring cube Phase G until the cube PGF face-seam fix lands. MPAS is in
   better shape (eady_uniform + dino already pass at 5% bulk).
6. **What about coupled fluxes?** Phase G is ocean-only. Air-sea fluxes from a fixed
   atmosphere snapshot are tractable; full coupled-system recipes are a later phase.

## Relation to existing work

- **Predecessor**: Phases A–F (``docs/ocean_fidelity/bulletproof_summary.md``). Phase G
  strengthens, does not replace, the bulk-metric gate.
- **Adjacent**: ``docs/ocean_validation_improvement_plan.md`` predates the recipe
  framing; it remains as historical context and is not modified here.
- **Existing harness reused**: ``scripts/ocean_fidelity/compare_legoesm_vs_veros.py``,
  ``scripts/ocean_fidelity/compare_legoesm_cube_vs_latlon.py``, and
  ``scripts/ocean_fidelity/run_comparison.py`` provide the reference-driver
  scaffolding that G.0 will extend (snapshot dump-and-load) and G.1 will reuse
  end-to-end.
- **Existing modules called**: ``ocean_tendency_common.py``,
  ``barotropic_common.py``, the scheme modules under ``ocean/dynamics/`` and
  ``ocean/physics/``.

## Deferred work

- A higher-level reorganization of ``docs/`` (the ``ocean_experiments/`` +
  ``ocean_fidelity/`` + ``ocean_long_runs/`` split has overlaps). Phase G is
  intentionally a *next layer*, not a reorganization.
- Atmospheric and coupled-system recipe framing. Phase G is ocean-scoped. Whether the
  same approach is applied to the atmosphere is a downstream question.
- Veros recipe **audit** — concrete listing of which Veros scheme choices are already
  implementable in legoESM vs. where the gaps are. Comes as a follow-up artifact
  under ``docs/ocean_fidelity/phase_g_veros_recipe_audit.md`` once this planning doc
  is accepted, so the plan is reviewable independently of the gap accounting.

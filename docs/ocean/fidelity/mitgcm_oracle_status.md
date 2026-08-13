# MITgcm oracle — program status & scoreboard

**Status (2026-06-18):** MITgcm is the **2nd ocean oracle** (after Veros). **Full
tutorial-suite survey done** (15 MITgcm tutorials): **10 are out of scope** for the
hydrostatic-ocean recipe path (3 nonhydrostatic — `deep_convection`, `plume_on_slope`,
`rotating_tank`; 1 atmospheric — `held_suarez_cs`; 1 pressure-coordinate — `global_oce_in_p`;
3 BGC/offline — `cfc_offline`, `dic_adjoffline`, `global_oce_biogeo`; 2 adjoint/optim —
`global_oce_optim`, `tracer_adjsens`). **5 are in-scope ocean-dynamics cases; 4 now have
recipes** (`barotropic_gyre`, `baroclinic_gyre`, `advection_in_gyre`, `reentrant_channel`),
the 5th (`global_oce_latlon`) scoped as buildable (~3–5 d glue). Plus `front_relax` (a
verification case, not a tutorial). The program surfaced a class of latent **C-grid metric
bugs** root-caused and fixed (iterations 7–8), a no-slip zero-length-face bug, a 2×
background-`K_v` double-count in two recipes, and a broader code-duplication audit (#514–#519).

Branch: `feat/mitgcm-oracle` (off `main`). Companion deep-dive:
`docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md`. Strategy/doctrine:
`docs/ocean/fidelity/oracle_recipe_strategy.md`.

---

## Approach

An MITgcm case is reproduced as a **pure-config legoESM recipe** that selects shared
canonical blocks (EOS, advection/viscosity dispatch, integrator, barotropic solver) — never
a bespoke solver (per the oracle-recipe doctrine). MITgcm is offline (Fortran): a reference
run is loaded from its dumps / `%MON` monitor output; the harness (readers, state bridge,
runner) lives outside the model. Built locally via a conda `mitgcm-build` (gfortran) env.

Fidelity tiers: (0) construction/forcing bit-exactness, (1) 10-step free-surface pattern,
(2) per-term momentum-tendency match, (3) monitor-statistics / equilibrium match.

---

## Cases & fidelity achieved

| case | type | grid | reference | result |
|---|---|---|---|---|
| `tutorial_barotropic_gyre` | single-layer wind-driven gyre | Cartesian β-plane 62×62×1 | on-box MITgcm (built), 75k-step + `%MON` | 10-step eta pattern **0.9997**; per-term tendencies match; **equilibrium LAMINAR** |
| `front_relax` | baroclinic front geostrophic adjustment | Cartesian f-plane channel 32×1×15 | shipped `results/output.txt` `%MON` (no rebuild) | 20-step `eta_max` **1%**, `uvel_max` **3%** |
| `tutorial_baroclinic_gyre` | wind+buoyancy double gyre | **spherical** 62×62×15 (lat 15–75N) | on-box rebuild, `%MON` + **field + momentum-tendency dumps** | 10-step `eta_max` **0.9%**, `uvel_max` **3%**, `vvel_max` **10%**→~1% (transient); field max\|u\| **1–5%** row-by-row, eta corr **0.9989**; **tendency tier**: total `du/dt` corr **0.987** / `dv/dt` **0.9996**, wind input corr **0.987** |
| `tutorial_advection_in_gyre` | passive-tracer advection (DST3) | Cartesian β-plane 60×60×1 | on-box rebuild, field dumps | tracer pattern corr **1.0000**, mass exact, signal-rel L2 **0.68%**; `dst3_multidim`≡MITgcm `advScheme=80` (isolates the advection scheme on prescribed velocity) |
| `tutorial_reentrant_channel` | ACC channel + GM/Redi + sponge | Cartesian β-plane 20×40×49 (reentrant-x) | on-box rebuild (gmredi+rbcs), `%MON` + field | 10-step T corr **1.0000**, eta corr **0.9999**, u corr **0.9973**; `eta_max` 1.7%, `uvel_max` 2.3%, `theta` 0.1% |

### Barotropic gyre — two tiers

- **Equilibrium oracle (bit-exact):** `GyreFaithfulModel` (`ocean/fidelity/mitgcm_gyre_faithful.py`)
  — a numpy/scipy unsplit MITgcm-faithful stepper reproducing MITgcm's laminar
  `|u|max≈0.031, |v|max≈0.084` at 1× **and** 2× resolution. Committed + tested. (It is a
  bespoke harness stepper; folding its "unsplit explicit-Coriolis → implicit free surface"
  sequencing into the canonical dispatch and deleting it is tracked in **#519**.)
- **Production model:** the canonical `LatLonCGridOceanModel` (recipe `build_gyre_recipe`,
  MITgcm-faithful config: `explicit_ab2` face-f Coriolis + centered flux-form advection +
  θ=1 implicit free surface) now spins up **laminar**, equilibrating at `|u|max≈0.027`
  (within ~13% of 0.031) / `|v|max≈0.078` — vs the prior **turbulent** 0.066→runaway. The
  residual ~13% is the production CG-solve vs the faithful direct solve + ~1% operator diffs.

### Front_relax (baroclinic)

Validates the production `implicit_cn` free surface on **stratified** geostrophic
adjustment — `eta_max` 1%, `uvel_max` 3% vs MITgcm. Confirms `implicit_cn` is the same
solver at `Nr=1` (gyre) and `Nr=many` (front); no separate single-layer solver is needed.
v1 fidelity caveats (documented in the recipe): uniform `dy` + 15 active levels (MITgcm
refines `dy` ~30% and carries 10 sub-bathymetry partial cells); `vvel_max` 0.04 vs 0.009
is an adjustment/time-scheme transient, not the viscosity.

---

## Key numerics delivered (and bugs root-caused along the way)

- **Iteration 7 — cos-lat metric inconsistency (the gyre's turbulent overshoot).** The
  β-plane grid set `cos_lat≡1` but stored pseudo-`lat=y/radius`; `divergence_cgrid`
  recomputed an ~1.8%-short v-face length while `gradient_x_cgrid` used `cos_lat=1`,
  breaking grad/div adjointness → non-conservative implicit free surface → laminar→turbulent.
  Fixed with an opt-in `cartesian_pseudo_lat` (lat=0) + the MITgcm-faithful recipe config.
- **Iteration 8 — `curl_vertex_cgrid` vertex-area collapse (one root, three symptoms).**
  The curl recomputes the dual-cell area as `R²·dlon·|Δsin(lat)|`, which collapses to 0
  when `lat≡0`; the pole guard then divided vorticity by 1.0 → ~1e8× blow-up of vorticity /
  vector-Laplacian / biharmonic. Fixed by reading the stored `grid.area_q`. This **one fix**
  resolved the biharmonic instability, the `explicit_substep` f-plane instability, and made
  `cartesian_pseudo_lat` safe for every solver. Bit-unchanged on spherical grids.
- **Component biharmonic** (`flux_divergence_bilaplacian_cgrid`): MITgcm's `viscA4` is a
  per-component `∇⁴` (`useStrainTensionVisc=.FALSE.`), not the vector `grad(div)−curl(curl)`
  form — added as the faithful, stable operator and wired into the `flux_divergence` family.
- **No-slip side-drag zero-length-face guard** (`no_slip_sidedrag_cgrid`): on a spherical grid
  the polar boundary v-faces have `dx_v = R·cos(lat_v)·dlon → 0`, so the unguarded `1/dx_v²`
  was `inf` and `inf·0` (mask) → NaN, blowing up any no-slip run on a non-Cartesian grid. Guarded
  with `where(Δ>0, 1/Δ², 0)`; bit-identical on the beta-plane (uniform metrics).

These fixes are on `feat/mitgcm-oracle` with non-vacuous regression tests; the full spherical
operator/vorticity suites pass byte-for-byte unchanged.

---

## Long-timescale verification (the necessary-not-sufficient test)

A 1-year run (26280 steps, 2600× the 10-step match) of `tutorial_baroclinic_gyre`,
legoESM vs an on-box MITgcm 1-yr run (`scripts/tmp/_bgyre_longrun_{lego,compare}.py`):

- **eta + T HOLD all year** — eta pattern corr **0.999** (amplitude within 2%) and T
  pattern corr **0.99** (range to 0.07°C) at every month. The gyre circulation + thermocline
  climate match.
- **`u` develops a 2Δx CHECKERBOARD null mode** — invisible at step 10 (signed-u matched 3%),
  dominant by week 4. Raw `u` alternates sign cell-to-cell (zig-zag ratio **1.6–1.8** vs
  MITgcm's smooth **0.1–0.4**); `|u|max` ~45% high and growing, worst toward the cold northern
  boundary (~0.3 m/s of pure grid noise). It is AGEOSTROPHIC (eta doesn't see it).
- **Root-cause investigation (30-day zig-zag sweeps, `_bgyre_*` probes):** it is a C-grid
  **velocity (u AND v) null mode** — `T` and `eta` stay smooth (zig-zag ~0.1, like MITgcm), only
  the velocity carries it. Systematically isolated:
  - **EXCITER = the surface-T restoring.** Wind-only (no restoring) is smooth (zig-zag **0.07**);
    adding the restoring → **1.39**. The restoring maintains a meridional T-gradient → a
    *baroclinic* hydrostatic pressure gradient (invisible to eta, hence eta corr stays 0.999)
    that feeds the velocity null mode. Convection is NOT it (removing it → 1.66, worse).
  - **DAMPER = raw Laplacian only.** A_h×10 kills it (1.39→**0.09**). Ruled OUT as fixes:
    viscosity-operator form (`vector_laplacian` 1.37 ≈ `flux_divergence` 1.39), Coriolis scheme
    (`matsuno_split` 1.27, **energy-conserving** 1.26 — no help), momentum advection (`upwind`
    1.33), and biharmonic at faithful magnitudes (B_h=1e14 → 0.82 partial, ≫ MITgcm which uses
    none).
  - **The puzzle:** MITgcm is smooth with the SAME `viscAh=5000` + `viscAr=1e-2` + same grid +
    same restoring (its `data` sets *only* those two viscosities). So legoESM's Laplacian is
    effectively ~10× weaker ON THE 2Δx MODE than MITgcm's `del2` at the same nominal coefficient
    — a stencil wavenumber-response difference, not a missing term.
  - **NOTE (corrected):** an earlier "biharmonic-not-wired" suspicion was WRONG — `B_h` is wired
    correctly (the `Bh_bilap` diagnostic scales linearly with `B_h`); B_h=1e10 was simply ~1000×
    too small (biharmonic needs `B_h ~ A_h·Δx² ≈ 1e13–1e14` to compete with the Laplacian).
  - **Next:** measure the legoESM Laplacian's damping rate on a pure 2Δx velocity mode vs the
    analytic `A_h·(π/Δx)²` and vs MITgcm's `del2` — find why it under-damps the grid mode, then
    the minimal faithful fix (match MITgcm's effective grid-scale viscous response). LESSON:
    short-run tendency match (corr 0.987) is necessary but NOT sufficient; the null-mode growth
    is a long-timescale-only signal — exactly the barotropic-gyre marginal-stability lesson.

## Open items / next

1. **`tutorial_baroclinic_gyre` — RESOLVED at the field tier.** The earlier "~28% high
   `uvel`" was a **test convention bug**, not a model error: MITgcm's `%MON dynstat_*_max`
   monitors are *signed* maxima, but the test compared legoESM's `max|u|` against them. u's
   largest-magnitude value is a *negative* excursion (−0.0229), so `max|u|` ≠ signed max.
   On-box field dumps (`scripts/tmp/_bgyre_field_compare.py`, tile-assembled from the
   rebuild) confirm: legoESM matches MITgcm's `max|u|` to **5%** globally and **1–5%
   row-by-row at every latitude**, signed `uvel_max` to **3%**, `eta_max` to **0.9%**, with
   an **eta pattern correlation of 0.9989**. The test now compares signed-max to signed-max
   (eta < 3%, u < 6%, v < 15%). This is a genuine field-tier oracle match. The residual
   `vvel_max` ~10% at step 10 is a **startup transient**: a 200-step convergence test
   against MITgcm field dumps (`scripts/tmp/_bgyre_vratio_convergence.py`) shows the v.max
   ratio decay 1.097→1.013→0.996→0.987 (steps 10/50/100/200) with the v pattern correlation
   pinned at 0.999 — it self-corrects to ~1%, not a structural error. **No structural bug
   remains in this case.**
2. **Per-term tendency tier — DONE for `tutorial_baroclinic_gyre`.** Using MITgcm's
   DIAGNOSTICS_PKG momentum dumps (`TOTUTEND`, `Um_Advec`, `Um_Cori`, `Um_dPhiX`, `Um_Diss`,
   `USidDrag`, `Um_Ext`; `scripts/tmp/_bgyre_tendency_tier.py`). MITgcm's flux-form split
   (`Um_Advec`/`Um_Cori` separate) vs legoESM's vector-invariant diagnostic (`KE_PGF`+
   `vortcor`, planetary Coriolis applied in the step function) makes a term-by-term
   advection/Coriolis comparison decomposition-dependent, so we match the two
   **decomposition-independent** quantities: (a) the **total** tendency — MITgcm
   `TOTUTEND ≡ Δu/Δt` vs legoESM by one-step finite-difference — `du/dt` **corr 0.987**
   (RMS ratio 0.958), `dv/dt` **corr 0.9996** (RMS ratio 1.001); and (b) the **wind** term,
   isolated path-independently by differencing (legoESM routes wind through the implicit-vmix
   surface BC, MITgcm through the explicit `Um_Ext` body force) — depth-integrated momentum
   input **corr 0.987**, peak **identical** (`9.99e-5 = τ/ρ₀` for both). Tendency tier for
   `front_relax` / `barotropic_gyre` (the latter already had per-term calibration via the
   faithful stepper) is the remaining extension.
3. **front_relax full fidelity** — variable `dy` + 25-level partial-cell vertical; close the
   `vvel_max` transient.
4. **Production-model exactness** — close the gyre's residual ~13% (production CG solve vs
   direct) only via faithful numerics, not tuning; the bit-exact 0.031 stays the
   `GyreFaithfulModel`.
5. **Architectural follow-ups** (from the duplication audit): **#514** (curl/divergence
   metric, the seed), **#515** (unify C-grid metric handling — operators read stored fields),
   **#516** (incompatible `cos(lat_v)` discretizations), **#517** (dycore primitive
   duplication), **#518** (physics duplicate-numerics), **#519** (fidelity/IC debt incl.
   promoting the unsplit barotropic option and deleting `GyreFaithfulModel`).

---

## Artifacts

- Recipes: `ocean/fidelity/mitgcm_{barotropic_gyre,front_relax,baroclinic_gyre}_recipe.py`.
- Faithful stepper: `ocean/fidelity/mitgcm_gyre_faithful.py`.
- Harness: `ocean/fidelity/mitgcm_{io,runner,monitor,state_bridge}.py`.
- Tests: `tests/ocean/fidelity/test_mitgcm_{barotropic_gyre,front_relax,baroclinic_gyre,gyre_faithful}_recipe.py`,
  `tests/grids/test_beta_plane_cgrid.py`, `tests/ocean/unit/test_no_slip_sidedrag.py`.
- MITgcm builds (gfortran conda env `mitgcm-build`): `/tmp/mitgcm_gyre/{build_diag,build_2x,run_*}`;
  reference source tree `/swot/SUM01/spencer/MITgcm/verification/`.

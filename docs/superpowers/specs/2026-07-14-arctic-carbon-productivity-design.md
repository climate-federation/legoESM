# Arctic / high-latitude carbon productivity rescue — design

**Date:** 2026-07-14
**Branch:** `land/arctic-carbon-productivity` (off `land/carbon-global-init` @ `d4c5f5115`)
**Component:** `packages/land/legoesm/land/carbon/` (DifferLand prognostic carbon)
**Status:** design approved (both mechanisms) — pending spec review, then implementation plan.

## Problem

The science-grade global land-carbon IC (real ERA5 climate + CLM5 cover, built
2026-07-14, drift 0.025 %/yr = at equilibrium) is realistic for temperate and
tropical biomes but **collapses at high latitude**: area-weighted global SOC
**3.47 kgC/m²** vs observed ~9.5, with the deficit concentrated in the Arctic
band (66–90°N SOC **0.61 kgC/m²** / biomass 0.19 vs observed 20–100+). Per-PFT
equilibria: `needleleaf_deciduous_boreal` (larch) is **dead — 0 biomass / 0 SOC
across all 12 archetypes**; `c3_arctic_grass` biomass 0.19; `needleleaf_evergreen_boreal`
partially collapses (several archetypes → 0). High-latitude soils are the largest
terrestrial carbon store, so this is the single largest remaining gap to a good
global carbon IC.

## Root cause (code-level, verified)

`carbon_cycle.py:662–666`:

```
R_maint = (r_maint_fol·C_fol + r_maint_root·C_root + r_maint_wood·C_wood)
          · exp(Q10_exp·(T − T_ref_ra))      # T_ref_ra = 298.15 K
```

Two structural gaps drive the death spiral:

1. **`R_maint` is never gated by substrate.** It demands the full biomass-proportional
   amount every step regardless of the labile reserve `C_lab`. Over a boreal winter
   (GPP ≈ 0, `exp(Q10·(T−298K))` small but nonzero), root+wood maintenance respiration
   slowly drains `C_lab`; the deficit cascade (`711–772`, order labile→wood→root→foliage)
   then eats structural pools until biomass → 0.

2. **No cold dormancy.** `compute_phenology` (`136–237`) has only evergreen (continuous
   turnover) and DALEC-deciduous (Gaussian leaf-fall *timing*) modes. Deciduous foliage
   still respires year-round; GPP's `f_T` factor weakens but never zeros below freeze.
   `is_evergreen("needleleaf_deciduous_boreal")` → False, so larch is "deciduous" but
   still pays foliar maintenance respiration through the frozen season it should be leafless.

Precedent: `_freeze_modifier` (`carbon_cycle.py:267`) already freeze-gates SOM
decomposition — the module has an established pattern for temperature gating.

## Approved approach: two opt-in, PFT-scoped mechanisms

Both default **off** (byte-identical to current production via a static Python
feature gate on a bool — the sanctioned JAX feature-gating exception, not `jnp.where`).
Both engineered so that, even when **on**, a healthy well-watered plant (ample `C_lab`,
not cold-deciduous) is unaffected — so temperate/tropical equilibria do not regress.

### Mechanism 1 — NSC-gated maintenance respiration (the load-bearing fix)

Multiply `R_maint` by a smooth substrate-limitation factor:

```
R_maint_eff = R_maint · f_nsc
f_nsc = r_maint_floor_frac + (1 − r_maint_floor_frac) · smoothstep(C_lab / C_lab_ref)
C_lab_ref = nsc_ref_labile_frac · (C_fol + C_root + C_wood)   # labile / live-biomass scale
```

- **Reference is live biomass, NOT foliage.** Tying `C_lab_ref` to `C_fol` alone would
  fail in the exact case being fixed: a winter-leafless plant has `C_fol → 0`, so a
  foliage-scaled reference collapses (`C_lab/ref → large → f_nsc → 1`) and the gate never
  engages while root+wood respiration keeps draining. Scaling to persistent live biomass
  `(C_fol + C_root + C_wood)` keeps the reference finite through the leafless season, so the
  gate throttles the root+wood drain — the actual death driver. `nsc_ref_labile_frac`
  defaults **below** typical healthy NSC (~0.02) so a healthy plant sits at `C_lab ≥ C_lab_ref`
  (f_nsc = 1, no temperate/tropical regression); the no-regression gate empirically confirms
  and tier-2-tunes it.
- `f_nsc → 1` when `C_lab ≥ C_lab_ref` (ample reserve → healthy plants unaffected).
- `f_nsc → r_maint_floor_frac` (small floor > 0, basal metabolism) as `C_lab → 0`.
- `smoothstep` = C¹ Hermite (`3x²−2x³` on a clamped `[0,1]` argument) — differentiable,
  no Python control flow on traced values.
- **Physical basis:** respiratory downregulation under carbon starvation (Atkin &
  Tjoelker 2003). The NSC/labile pool is the substrate signal.
- **Effect:** in winter the plant *hibernates* — respiration throttles as reserves
  deplete instead of cannibalizing structural carbon — then refills `C_lab` from the
  short summer GPP window. This alone likely breaks the spiral; it also protects
  `needleleaf_evergreen_boreal` and arctic grass, which are not cold-deciduous.

Applies to **all** `R_maint` terms (foliar + root + wood): the winter root+wood drain
is the actual death driver, so gating must cover them.

### Mechanism 2 — cold-deciduous freeze dormancy (phenological correctness)

For cold-deciduous PFTs, a dormancy factor gates foliar carbon fluxes:

```
d = sigmoid((T − freeze_dormancy_threshold_K) / dormancy_transition_width_K)   # ∈ (0,1)
GPP_eff        = GPP · d                     # no photosynthesis when frozen/leafless
R_maint_fol_eff = r_maint_fol·C_fol·temp_factor · d    # no foliar respiration when dormant
```

- Root/wood `R_maint` is **not** dormancy-gated (roots persist; they are handled by
  Mechanism 1's NSC gate). Only the **foliar** term is zeroed — the leaf-drop behavior.
- Reproduces the larch/tundra strategy: shed foliage in winter to avoid its respiration
  cost. `d` is a smooth sigmoid → differentiable.
- **PFT-scoped** via a new classifier `is_cold_deciduous(pft_name)` in `surface_params.py`,
  True for `needleleaf_deciduous_boreal`, `c3_arctic_grass`, `broadleaf_deciduous_boreal_shrub`.
  Evergreen and temperate/tropical PFTs get `d ≡ 1` (untouched).

### Coordination / gate composition

Mechanism 1 is primary (stops the winter root+wood drain that kills the plant).
Mechanism 2 is the phenological refinement (correct larch leaf-drop). Implemented and
tested **independently** (M1 first, M2 second) but shipped together per the approved scope.

Gate composition (explicit, no ambiguity):
- Dormancy `d` is active only when `cold_deciduous_dormancy` (master flag) **and** the
  archetype's `cold_deciduous` (PFT trait) are both true; otherwise `d ≡ 1`.
- The two gates are independent physical effects, so when both mechanisms are on the
  **foliar** `R_maint` term carries **both** factors (`r_maint_fol·C_fol·temp·f_nsc·d`);
  **root/wood** `R_maint` carry `·f_nsc` only (no dormancy — roots persist); foliar **GPP**
  carries `·d` only. Each factor ∈ (0,1], so composition never increases a flux.

## Config & parameter specs (`config.py` `CarbonConfig`)

New fields (each float gets a `__param_spec__` entry; `__physics_contract__` updated):

| field | type | default | units | tier | note |
|-------|------|---------|-------|------|------|
| `nsc_gated_respiration` | bool | `False` | — | static gate | Mechanism 1 on/off |
| `nsc_ref_labile_frac` | float | 0.02 | — | 2 | `C_lab_ref = this · (C_fol+C_root+C_wood)` |
| `r_maint_floor_frac` | float | 0.10 | — | 2 | `f_nsc` floor at `C_lab→0` |
| `cold_deciduous_dormancy` | bool | `False` | — | static gate | Mechanism 2 on/off |
| `cold_deciduous` | bool | `False` | — | static (per-PFT) | set from `is_cold_deciduous` |
| `freeze_dormancy_threshold_K` | float | 273.15 | K | 2 | `T_dorm` |
| `dormancy_transition_width_K` | float | 2.0 | K | 0 (excluded) | sigmoid width (numerics) |

Defaults reference `legoesm.constants` where applicable (`freeze_dormancy_threshold_K`
defaults to `constants.T_freeze`). Booleans are not spec-eligible (static gates).

## Invariants (must hold)

- **Conservation:** gating a respiration *flux* retains the un-respired carbon in its
  source pool; NPP = GPP − R_auto rises consistently and the pool update uses the same
  gated `R_maint`, so `Σ ΔC = −NEE·dt` still closes to machine precision. Verified by an
  analytic column conservation test with each flag on.
- **Sign convention:** `R_maint` is a loss (plant → atmosphere, positive-out). Both gates
  *reduce* the loss (less respiration → less death) — the correct direction. Each gate term
  gets an inline sign-convention comment; a unit test asserts gated `R_maint ≤` ungated.
- **Differentiability:** `smoothstep` and `sigmoid` are C¹; no `jnp.where` on the feature
  flags (static Python `if`), no Python control flow on traced values. Gradients flow.
- **No silent behavior change:** flags off → byte-identical (static gate); a regression
  test asserts bit-identical pools vs current `HEAD` with defaults.

## PFT scoping / wiring

- New `is_cold_deciduous(pft_name)` in `surface_params.py` (sibling of `is_evergreen`,
  `303–313`).
- Archetype grouping in `global_init.py:394–412` extends `(is_woody, is_evergreen,
  soil_class)` → `(is_woody, is_evergreen, is_cold_deciduous, soil_class)` so cold-deciduous
  archetypes spin with `cold_deciduous=True`. (Group count rises modestly; compile cache
  version bumps.)
- The coupled-run path already threads per-archetype config; `cold_deciduous` rides the
  same CarbonConfig as `evergreen`.

## Tests (TDD — write before the body)

Unit (`tests/land/unit/test_carbon_cycle.py`):
1. `test_nsc_gate_throttles_at_low_labile` — `f_nsc(C_lab=0)=r_maint_floor_frac`,
   `f_nsc(C_lab≫ref)=1`, monotone in between.
2. `test_nsc_gate_conserves` — analytic column, flag on, `Σ ΔC = −NEE·dt`.
3. `test_nsc_gate_off_is_byte_identical` — flag off ⇒ bit-identical to current.
4. `test_cold_deciduous_zeros_foliar_fluxes_below_freeze` — GPP·d and foliar R_maint → 0
   as `T ≪ T_dorm` for a cold-deciduous config; `d→1` above.
5. `test_cold_deciduous_off_and_non_cd_pft_byte_identical`.
6. `test_gated_rmaint_le_ungated` (sign / monotonicity).
7. `test_starved_column_survives_with_nsc_gate` — a boreal-winter forcing column that
   dies (biomass→0) with the gate OFF **retains** biomass with it ON (the death-spiral
   regression, provably non-vacuous).

Contract: update `__physics_contract__` + `__param_spec__`; `tests/test_param_specs.py`
and `tests/test_no_inline_physics_coeffs.py` must stay green (new coeffs → config fields).

## No-regression + realism gate

1. Flags off → byte-identical (test 3/5 + a full-column equality test).
2. `scripts/validate/land_carbon_equilibrium.py` (7 climate pixels incl. boreal + tundra),
   flags on: temperate-forest / tropical / savanna / temperate-grass equilibria unchanged
   within tolerance; **boreal + tundra reach nonzero equilibrium** (currently decline).
3. Rebuild the global carbon IC (`build_global_carbon_ic.sbatch`, flags on, equilibrium
   cache version bumped) → arctic-band SOC lifts toward observed; temperate/tropical
   per-PFT ranges unchanged. Compare **only** flags-on-vs-flags-off on the identical
   climate/grid/metric (controlled comparison — no protocol drift).
4. Codex adversarial review (`codex exec --sandbox read-only` on a compute node) until clean.

## CLI (`run_lmip.py`) — per the config-field-needs-a-flag rule

`--nsc-gated-respiration` / `--no-...`, `--cold-deciduous-dormancy` / `--no-...`,
`--nsc-ref-labile-to-fol`, `--r-maint-floor-frac`, `--freeze-dormancy-threshold-k`
(float tunables also reachable via `--params carbon.<field>`); round-trip tests in
`tests/unit/test_run_lmip_cli.py`.

## Files touched

- `packages/land/legoesm/land/carbon/carbon_cycle.py` — `f_nsc` + dormancy `d`; gate
  `R_maint` (all terms) and foliar GPP/R_maint; helpers `_nsc_respiration_factor`,
  `_cold_deciduous_dormancy_factor` (siblings of `_freeze_modifier`).
- `packages/land/legoesm/land/carbon/config.py` — new fields + `__param_spec__` + contract.
- `packages/land/legoesm/land/surface_params.py` — `is_cold_deciduous`.
- `packages/land/legoesm/land/carbon/global_init.py` — grouping key + cache version bump.
- `scripts/run/run_lmip.py` — CLI flags + config wiring.
- Tests as above.

## Out of scope / follow-ups

- Photoperiod-triggered phenology (larch also cues on daylength, not only T) — T-only
  dormancy is the first cut.
- Nitrogen limitation of high-latitude productivity — separate lever.
- Coupled-run ingestion of the rebuilt (flags-on) finidat is already supported
  (`d4c5f5115`, `run_coupled --carbon-ic`); the equilibrium cache version bump propagates.
- Merging the whole pipeline to `main` (stranded stack) — tracked separately.

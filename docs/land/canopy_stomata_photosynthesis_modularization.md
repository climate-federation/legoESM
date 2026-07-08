# Land surface scheme de-duplication: audit and modularization plan

**Audience:** Pierre (and anyone touching the land surface schemes)
**Status:** design proposal — no code changed yet
**Scope:** identify and de-duplicate numerics shared across the land surface stacks — the simpler
AMIP energy-balance path (`SimpleSEB`, via `land/carbon/stomata.py`), the two-leaf canopy module
(`land/canopy/`), the slab and multilayer drivers, and the coupler. The primary, deepest target is
the **leaf physics** (stomatal conductance + photosynthesis, Sections 1–6, with a full phased plan
and test methodology). The document then audits **soil components** (Section 7) and **radiation +
turbulent fluxes including the atmosphere/coupler interface** (Section 8), reusing the same
verification methodology. Goal: one source of truth per numerical kernel, **without changing any
results** where the dedup is mechanical, flagging any physics unification as an explicit decision.
Do not disrupt the multilayer or CLM-ML canopy.

---

## 1. Motivation

legoESM's design doctrine is modular physics with a single source of truth for each numerical
kernel (CLAUDE.md: *"Shared utilities — never re-derive"*, *"No duplicate numerics across
dycores/physics/grids/tests"*). Today two land surface stacks re-implement the same leaf
equations:

- the **SimpleSEB** column path Pierre developed for AMIP/AIMIP — a bulk-flux surface energy
  balance whose transpiration/GPP come from `land/carbon/stomata.py`; and
- the **two-leaf canopy** (`TwoLeafCanopyConfig`) — a sunlit/shaded leaf energy-balance Newton
  closure whose leaf physics live in `land/canopy/`.

The two stacks are *legitimately different surface schemes* (bulk single-skin vs resolved leaf
energy balance), and that is fine — they are meant to be interchangeable plug-ins behind the same
`SurfaceFluxOutput`. The problem is that they **re-derive the same empirical leaf kernels**
instead of sharing them, which is exactly the duplication the doctrine forbids. This document
identifies precisely what is duplicated, what only *looks* duplicated, and a phased, test-gated
path to a single source of truth.

Guiding constraints:
1. **No result change** for either scheme in Phase 1 (the safe, mechanical dedup).
2. **Do not touch** the multilayer land driver dispatch, the CLM-ML external wrapper, or the
   `SurfaceFluxOutput` contract.
3. Any physics *unification* (as opposed to mechanical dedup) is called out explicitly and left
   as a decision, not folded in silently.

---

## 2. Duplication map

Verified by reading the code on current `main` + the EC-site branch.

| Kernel | Location A (two-leaf canopy) | Location B (SimpleSEB / AMIP) | Classification |
|---|---|---|---|
| **Ball-Berry `gs`** | `canopy/stomatal.py:32` `ball_berry_gs(An, RH, Cs, m, b0)` | `carbon/stomata.py:318` `ball_berry_gs(A, RH, Cs, config)` | **TRUE duplication** — identical math, forked signature |
| **Medlyn `gs`** | `canopy/stomatal.py:61` `medlyn_gs(An, VPD, Cs, g1, g0)` | `carbon/stomata.py:336` `medlyn_gs(A, VPD, Cs, config)` | **TRUE duplication** — identical math, forked signature |
| **Farquhar photosynthesis** | `canopy/photosynthesis.py` `c3_photosynthesis` + `c4_photosynthesis` + `photosynthesis` (FvCB C3 + Collatz C4, sunlit/shaded) | `carbon/stomata.py:225` `farquhar_photosynthesis` (Farquhar C3-only, big-leaf) | **Different models, also numerically divergent** (different gas constant, `Ha`, acclimation, co-limitation form) — a physics decision, not a mechanical dedup (see Phase 2/3) |
| **Jarvis `gs`** | — | `carbon/stomata.py:356` `jarvis_gs` | **Distinct** — CO2-independent, one copy only, leave as-is |
| **Coupled A–Ci–gs solve** | inside `canopy/solver.py` Newton closure | `carbon/stomata.py:416` `solve_coupled_farquhar_ci` | **Distinct** — different closure strategies |
| **Empirical LUE GPP** | — | `carbon/carbon_cycle.py:54` `compute_gpp` | **Distinct** — fallback only |
| Aerodynamics / soil evaporation / energy balance | canopy: within-canopy resistances + Newton leaf-EB + Sellers-1992 soil `r_ss` | SimpleSEB: `core.bulk_flux` MOST/COARE3 single skin + bucket `β·q_sat` | **Distinct physics** — do NOT merge |

### The true duplication, side by side

`canopy/stomatal.py` (explicit slope/intercept, because the two-leaf canopy needs per-leaf-class
`m_C3/m_C4/b0_C3/b0_C4`):

```python
def ball_berry_gs(An, RH, Cs, m, b0):
    A_pos = jnp.maximum(An, 0.0); Cs_safe = jnp.maximum(Cs, 1.0)
    return jnp.maximum(b0 + m * A_pos * RH / Cs_safe, b0)
```

`carbon/stomata.py` (unpacks a scalar `StomataConfig`):

```python
def ball_berry_gs(A, RH, Cs, config):
    A_pos = jnp.maximum(A, 0.0); Cs_safe = jnp.maximum(Cs, 1.0)
    return jnp.maximum(config.g0 + config.g1_bb * A_pos * RH / Cs_safe, config.g0)
```

These bodies are character-identical modulo where `(slope, intercept)` come from. Medlyn is the
same story, including the shared `_DIFFUSIVITY_RATIO_H2O_CO2 = 1.6` prefactor and the `0.05 kPa`
VPD floor, both currently declared once per file. This is the fork documented in
`canopy/stomatal.py:8-19` — deliberate, but it duplicates the math.

### What is NOT duplication (do not "fix")

- **Jarvis** exists once. It is a different model (multiplicative environmental factors,
  CO2-independent) and is the carbon-off fallback for SimpleSEB. Leave it.
- **The two Farquhar codes are different models**, not copies: `canopy/photosynthesis.py` is the
  canonical FvCB C3 **plus** Collatz C4 with sunlit/shaded scaling, Kattge–Knorr (2007) peaked
  Arrhenius acclimation and Tjoelker/Atkin dark respiration; `carbon/stomata.py` is C3-only,
  big-leaf, and runs C4 PFTs through C3 kinetics as a documented approximation. They share the
  *form* of the FvCB primitives (Γ\*, Kc/Ko, the Wc/Wj co-limitation) but diverge in the actual
  numbers — different universal gas constant (`8.314` vs `constants.R_universal`), different `Ha`,
  Kattge–Knorr acclimation vs simple Arrhenius, and different co-limitation forms — so they are
  not even bit-for-bit at the primitive level. Unifying them is a *physics decision*, not a
  mechanical dedup — see Phase 2/3.
- **Aerodynamics, soil evaporation, and the energy-balance closure differ by design** between a
  bulk single-skin scheme and a resolved leaf energy balance. Not a target.

---

## 3. Modularization approach

Follow the pattern the ocean dynamics already uses to share numerics across otherwise-distinct
schemes — `ocean/dynamics/ocean_tendency_common.py` and `barotropic_common.py`:

- **Pure, pytree-friendly functions**, no config objects and no module-global state.
- **Inject anything scheme-specific as a callable** (the land analogue already exists: the
  two-leaf canopy takes `soil_thermal_fn` as an injected callable; the ocean helpers take
  `fill_fn`/`eos_fn`).
- **Dispatch stays in the caller** — the `isinstance(config.surface_scheme, ...)` chain in
  `multilayer_land.py` and `slab_land.py` is untouched.
- **`SurfaceFluxOutput` (`surface_scheme/base.py`) stays the untouched, scheme-agnostic return
  type** — the whole point is that downstream (snow, Richards, soil thermal, carbon, TileResponse)
  never branches on scheme.

**Placement — a neutral shared kernel module.** Put the pure kernels in a new
`packages/land/legoesm/land/stomatal_kernels.py` (config-free), imported by **both**
`canopy/stomatal.py` and `carbon/stomata.py`. A neutral module mirrors the ocean `_common.py`
idiom and keeps the kernels free of cross-subpackage coupling. (For the record, the *current*
direction is already one-way `carbon → canopy`: `carbon/stomata.py:49` imports `SIFConfig` from
`canopy/sif.py`, while `canopy` does not import `carbon`. So a `canopy`-local kernel imported by
`carbon` would in fact also be cycle-free today — the neutral module is preferred for cleanliness
and to avoid *adding* more cross-subpackage coupling, not because it is the only cycle-safe
option. The reverse — a `carbon`-local kernel imported by `canopy` — is the placement that *would*
risk a cycle, and the neutral module sidesteps it.) Each existing entry point stays where it
is and becomes a thin adapter that unpacks its own parameter source and delegates:

```python
# land/stomatal_kernels.py  — single source of truth, config-free, pure
_DIFFUSIVITY_RATIO_H2O_CO2 = 1.6
_VPD_FLOOR_KPA = 0.05

def ball_berry_gs(A, RH, Cs, slope, intercept):
    A_pos = jnp.maximum(A, 0.0); Cs_safe = jnp.maximum(Cs, 1.0)
    return jnp.maximum(intercept + slope * A_pos * RH / Cs_safe, intercept)

def medlyn_gs(A, VPD_kPa, Cs, g1, g0):
    A_pos = jnp.maximum(A, 0.0); Cs_safe = jnp.maximum(Cs, 1.0)
    VPD = jnp.maximum(VPD_kPa, _VPD_FLOOR_KPA)
    return jnp.maximum(
        g0 + _DIFFUSIVITY_RATIO_H2O_CO2 * (1.0 + g1 / jnp.sqrt(VPD)) * A_pos / Cs_safe, g0)
```

```python
# canopy/stomatal.py — canopy keeps its per-leaf-class explicit-arg names, delegates
from legoesm.land.stomatal_kernels import ball_berry_gs as _bb, medlyn_gs as _med
def ball_berry_gs(An, RH, Cs, m, b0):  return _bb(An, RH, Cs, m, b0)
def medlyn_gs(An, VPD_kPa, Cs, g1, g0): return _med(An, VPD_kPa, Cs, g1, g0)
```

```python
# carbon/stomata.py — SimpleSEB keeps its config signature, unpacks then delegates
from legoesm.land.stomatal_kernels import ball_berry_gs as _bb, medlyn_gs as _med
def ball_berry_gs(A, RH, Cs, config):  return _bb(A, RH, Cs, config.g1_bb, config.g0)
def medlyn_gs(A, VPD_kPa, Cs, config): return _med(A, VPD_kPa, Cs, config.g1_med, config.g0)
```

The two public wrappers are *interface adapters* (they translate a parameter source to the
kernel), which is legitimate under the doctrine — the banned pattern is a re-export-only
`X_utils.py`, not a real adapter, and CLAUDE.md already blesses this kind of caller-specific
branching for `land/stomata_utils.py`.

---

## 4. Phased plan (risk-ordered)

### Phase 1 — stomatal-kernel dedup (low risk, do first)

Extract Ball-Berry + Medlyn into `land/stomatal_kernels.py`; make `canopy/stomatal.py` and
`carbon/stomata.py` delegate as above. This is a **pure-function relocation**: the arithmetic is
unchanged, so both schemes must produce **bit-for-bit identical** `SurfaceFluxOutput`. Move the
shared constants (`_DIFFUSIVITY_RATIO_H2O_CO2`, the VPD floor) into the kernel module so they too
exist once. Outcome: one copy of the two gs equations, zero result change.

### Phase 2 — audit the two Farquhar codes for genuinely shared primitives (expect little)

Audit `canopy/photosynthesis.py` against `carbon/stomata.py:farquhar_photosynthesis` for
biochemical primitives that are *numerically identical*. **Expect the audit to find that most are
not** — the two paths already diverge in their kinetic constants and forms, so "shared primitives
waiting to be factored" overstates the opportunity:

- **Different universal gas constant.** `canopy/photosynthesis.py` hardcodes `_R = 8.314`;
  `carbon/stomata.py` uses `constants.R_universal = 8.314462618`. So even Γ\* — which otherwise
  shares `_GS25 = 42.75` / `_HA_GS = 37830` — differs at ~1e-5 relative and *fails* a bit-for-bit
  gate. (Reconciling this on `constants.R_universal` is itself a small result change.)
- **Vcmax temperature response is genuinely different physics, not a shared primitive.** Canopy
  uses Kattge–Knorr (2007) peaked Arrhenius + growth-temperature acclimation (`Ha=72000`); carbon
  uses simple Arrhenius (`Ha_Vc=65330`, no acclimation). Do **not** treat
  `vcmax_temperature_response` as a canonical primitive to factor toward — it is divergent.
- **Co-limitation forms differ** — a single sqrt smooth-min in carbon vs two-stage smaller-root
  quadratics in canopy.

Consequence: under the "only bit-for-bit passes move" rule, Phase 2 realistically factors **little
to nothing** unless the constants are first reconciled — and reconciling them *changes results*,
which is Phase 3, not Phase 2. The honest read is that the stomatal-kernel dedup (Phase 1) is the
clean mechanical win, and the two Farquhar codes are better described as **two deliberately
different photosynthesis models** than as copies. If a shared kinetics module is still wanted, its
only safe Phase-2 contents are the handful of truly closed-form identical helpers (e.g. a
Michaelis–Menten Kc/Ko term) — audit first, factor only what passes, keep the two canopy-scaling
wrappers (two-leaf sunlit/shaded C3+C4 vs big-leaf C3) distinct.

### Phase 3 — converge SimpleSEB photosynthesis? (physics decision — Pierre's call)

The remaining question is whether SimpleSEB's C3-only big-leaf photosynthesis should adopt the
canonical FvCB C3+C4 two-leaf kinetics. This **changes AMIP results** and therefore requires full
AMIP re-validation (Section 5). It may be intentional to keep SimpleSEB as a deliberately simpler,
cheaper AMIP model. This is flagged as a decision, not a mechanical dedup, and is out of scope for
the safe refactor.

---

## 5. Test & verification plan

The refactor is "correct" iff it changes no numbers where it claims to. The oracle and gates:

### 5.1 Primary oracle — `SurfaceFluxOutput` equivalence

Pin a small set of representative fixed inputs and assert **field-by-field equality** of the
`SurfaceFluxOutput` returned by `compute_two_leaf_canopy_fluxes` and `compute_simple_seb_fluxes`
before vs after the refactor:
- Phase 1: expect **bit-for-bit** (`jnp.array_equal`) — it is a pure relocation.
- Phase 2: expect equality to ~1e-10 for the primitives that were audited as identical; any
  field that moves is a bug in the extraction.

This is the cheapest, sharpest regression net and should be added as a unit test in
`tests/land/unit/`.

### 5.2 New dedup-enforcing test (mirror the ocean precedent)

Add `tests/land/unit/test_no_canopy_stomata_duplication.py`, modeled exactly on
`tests/ocean/unit/test_no_scheme_duplication.py` (substring-based, not AST):
- **presence**: `land/stomatal_kernels.py` exists and defines `ball_berry_gs` / `medlyn_gs`;
- **delegation**: `canopy/stomatal.py` and `carbon/stomata.py` both import + call the kernels;
- **anti-inline**: the old inline fragment (e.g. `g1_bb * A_pos * RH`) no longer appears in the
  consumers, so re-introducing a copy goes red and the message points at the offending file.

### 5.3 Must-pass CI ratchets (a land physics move trips these if done wrong)

- `tests/test_no_inline_physics_coeffs.py` — the moved `1.6` / `0.05` must live in a module-level
  `_UPPER_SNAKE` constant or carry `# coeff-ok:` (land budget is a shrink-only baseline).
- `tests/test_no_hardcoded_constants.py` — no bare physical-constant literals.
- `tests/test_param_specs.py` — land `*Config`s are fully specced; if any `*Config` field moves
  modules, its `__param_spec__` entry moves with it (spec key must equal the field name).
- `tests/test_dispatch_hardening.py` — the land unknown-scheme `raise ValueError` guards
  (`simple_seb.py`, `slab_land.py:step_land`, `carbon/carbon_cycle.py:step_carbon`) are a
  grow-only baseline; relocating a dispatcher must update it.
- `tests/test_no_private_cross_imports.py` — if the split imports a helper across the new module
  boundary, **promote** it (drop the underscore + re-export), do not import a `_`-name.

### 5.4 Carry the conventions on any moved code

- **`__param_spec__`** (machine-enforced for land) travels with its `*Config`.
- **`__physics_contract__`** (doctrine; currently NOT machine-enforced for land) should be added
  to `land/stomatal_kernels.py` and carried with any moved physics body. Optionally, extend
  `_is_physics_py` in `tests/test_physics_contracts.py` to cover
  `packages/land/legoesm/land/canopy/` to make it enforced — a nice follow-up, out of scope here.

### 5.5 End-to-end land + AMIP regression (especially before Phase 3)

- `tests/land/unit/` canopy + stomatal + photosynthesis units (`test_canopy_stomatal.py`,
  `test_canopy_photosynthesis.py`, `test_stomata.py`, `test_canopy_le_cap.py`,
  `test_canopy_solver_grad.py`).
- `tests/land/test_land_stability.py` — 30-day slab + multilayer budgets (energy/water closure,
  carbon positivity, realistic GPP) before and after.
- `tests/land/integration/test_ec_site_run.py` — offline EC-site skill unchanged.
- `scripts/validate/validate_land_era5.py` — diff the AMIP-land-vs-ERA5 metrics on both trees
  (the direct AMIP-land climatology check; its internals are unit-tested by
  `tests/land/test_validate_land_era5.py`).
- `tests/integration/test_amip_deck_smoke.py` + `scripts/validate/validate_amip_run.py` — a
  finite-state, end-to-end AMIP run through the land tile.

For Phase 1 all of the above should be **unchanged**; for Phase 3 they are the acceptance battery
for a deliberate result change.

---

## 6. Non-goals and risks

- **Non-goal:** merging the aerodynamics, soil-evaporation, or energy-balance closures — these are
  genuinely different physics between a bulk single-skin scheme and a resolved leaf energy
  balance, and interchangeability is already provided by `SurfaceFluxOutput`.
- **Non-goal:** touching the multilayer land dispatch or the CLM-ML external wrapper.
- **Leave in place:** `jarvis_gs`, `solve_coupled_farquhar_ci`, and the LUE `compute_gpp`.
- **Risk to watch:** import cycles (mitigated by the neutral kernel module), and the Phase 2
  temptation to unify primitives that are only *approximately* the same — anything that fails the
  bit-for-bit audit is Phase 3 physics, not Phase 2 dedup.

**Leaf-physics bottom line:** Phase 1 is a safe, mechanical, bit-for-bit dedup of the two identical
stomatal kernels and should land first behind the `SurfaceFluxOutput` oracle + the new dedup test.
Phase 2 realistically factors little (the Farquhar primitives are already numerically divergent).
Phase 3 — whether SimpleSEB adopts the canonical C3+C4 photosynthesis — is a physics choice for
Pierre, gated by the AMIP battery.

---

## 7. Soil components

The **heavy soil numerics are already single-source** — the doctrine holds where it matters most.
What remains is thin/orchestration duplication. The same verification methodology (Section 5)
applies: `SurfaceFluxOutput` field-by-field equivalence + the guardrail ratchets, and a substring
dedup test where a helper is extracted.

### 7.1 Already shared (do not touch)

Retention curves + K(θ) + ψ(θ) (`soil_hydraulics.py`), the Richards solve (`richards.py`, single
caller `multilayer_land.py:754`), soil-thermal diffusion + heat capacity + Johansen conductivity
(`soil_thermal.py`), the tridiagonal Thomas solver (`core/timestepping/tridiagonal.py`, repo-wide),
and bucket runoff (`bucket_hydrology.py:partition_bucket_runoff`, already de-duplicated between
slab and the coupler). SimpleSEB re-implements none of these — it consumes a caller-supplied
`beta_soil` and returns `G_soil` + a Robin surface conductance.

### 7.2 Real duplication (all thin / orchestration, not the core solvers)

| # | What | Where | Value | Note |
|---|---|---|---|---|
| S1 | **slab re-implements the SimpleSEB energy balance inline** | `slab_land.py:step_land` (~110–130 lines) does **not** call `compute_simple_seb_fluxes`; only `multilayer_land.py:535` uses the factored version | **High** | slab carries an *older, less-hardened fork* — missing `_MAX_LAND_EXCHANGE_COEFF` and `_LAND_CONDENSATION_FLOOR_W` that `simple_seb.py` has. Fix = route slab through `compute_simple_seb_fluxes` (it already takes `beta_soil` as a caller arg). |
| S2 | **`root_zone_moisture_stress` vs `root_zone_beta_soil`** | `multilayer_land.py:87` and `:179` | Medium | identical β_root / w_frac_rz / β_soil; the `#823` docstring already flags them numerically identical. Fix = one delegates to the other. |
| S3 | **Kelvin pore humidity `h_r`** | `multilayer_land.py:439-440` ≡ `:713-714` | Low (cheap) | byte-identical `exp(min(ψ·g/(R_v·T),0))` in the canopy and SimpleSEB branches. Fix = a 2-line helper (natural home `soil_hydraulics.py`). |
| S4 | **snow sublimation / soil-evap partition wrapper** | `slab_land.py:236`, `:560`, `multilayer_land.py:662` | Medium | ~30-line partition glue ×3 (the melt/age math underneath is already shared via `snow_budget.update_snow`). |
| S5 | **bucket β ramp** `β_min+(1-β_min)·clip(W/W_max,0,1)` | `slab_land.py:138/337/646` + coupler `physics_pipeline.py:370` | Low | cross-package one-liner; a good companion helper to the already-shared `partition_bucket_runoff`. |

### 7.3 Legitimately distinct (do NOT merge)

- Slab **lumped explicit** thermal (`C_soil·d_soil`, 0-D) vs the multilayer **implicit column**
  (θ-dependent Johansen) — different discretizations of different models.
- **Bucket hydrology** vs the **Richards column** — different water models.
- The two **soil-evaporation models**: bucket `β = S_top^exp · h_r` (multilayer SimpleSEB) vs
  Sellers-1992 `r_ss` + Sakaguchi-Zeng litter + Kelvin (canopy). Distinct by design — but the
  divergence is **asymmetric**: the **multilayer** SimpleSEB path already *has* the Kelvin `h_r`
  term (that is exactly the S3 duplication at `multilayer_land.py:712-718`) and lacks only the
  Sellers dry-surface-layer resistance the canopy path (and CLM5) have; the **slab / pipeline** SEBs
  (`slab_land.py`, `_step_slab_land`) lack Kelvin entirely. Giving SimpleSEB a shared **bare-soil**
  evaporation-efficiency kernel would be a *fidelity/consistency* improvement — it **changes
  results** (Phase-3-style, needs AMIP re-validation), and it must expose the **bare-soil**
  efficiency only, because SimpleSEB currently throttles *total* latent heat and would otherwise
  wrongly suppress transpiration too.

### 7.4 The plug-in contract

**Multilayer honors it** — all three surface schemes converge to one `SurfaceFluxOutput`, and the
entire soil post-processing (snow, sublimation partition, Richards, soil-thermal with the Robin
BC, carbon, TileResponse) runs **once**, scheme-agnostic (only two correct feature-gate `isinstance`
branches remain). **Slab does not** — it carries two separate hand-written post-flux bodies and its
SimpleSEB branch never builds a `SurfaceFluxOutput`. Routing slab through the shared contract (S1)
is the single highest-value soil-side cleanup.

---

## 8. Radiation and turbulent fluxes (land, coupler, atmosphere)

This is the **best-modularized** area — the shared kernels already dominate, verified by a
cross-component audit (land, ocean, sea ice, coupler, atmosphere).

### 8.1 Turbulent surface fluxes — single-source, clean split

`core/bulk_flux.py` (`compute_most_fluxes` MOST/COARE3/Large-Yeager, `simple_bulk_fluxes`, the
`psi_m`/`psi_h` stability functions, `large_yeager_neutral_cd`) is the **genuine single source** for
bulk transfer across land + sea ice + coupler + standalone-ocean forcing + the atmosphere surface
layer. No land/ice/coupler/atmosphere-PBL path runs its own MOST iteration — roughness/scheme
differences (COARE3 over ocean, `z0_ice`, `L_s` for ice sublimation) are *configuration of the
kernel*. The atmosphere↔coupler split is clean: PBL schemes consume `compute_surface_fluxes` or a
driver-injected flux tuple; the coupler owns the one air-sea/land/ice exchange
(`_tiled_surface_flux`) with an explicit double-count guard.

Separate MOST loops that are **legitimately distinct** (reuse core primitives, own solver): the
canopy multi-source resistance MOST (`canopy/stability.py`, a different formulation), the NEMO
bit-parity `ocean/bulk_flux_omip.py` (fidelity target), and the LES wall model. **Minor true dups
worth tidying:** the canopy's unstable-branch ψ functions (`canopy/stability.py:117,127`) are
byte-identical to core's and could import from `core/bulk_flux.py`; and the inline
constant-coefficient SH/LH in `physics_pipeline.py:1194-1196` / `bulk_formulas.py:140-144` re-spell
`simple_bulk_fluxes` (const-coeff placeholders, harmless).

### 8.2 Surface radiation — kernel is the source, a few stray re-implementations

`core/surface_energy.py:surface_radiation_fluxes` (`sw_net = (1-α)·sw_down`; `lw_net = ε·lw_down −
εσT⁴`; `lw_up`) is used by the coupled tiles (slab_land, multilayer_land, simple_seb, sea-ice
thermo, lake). `surface_albedo.py` is the single land-albedo source (no duplicated blend math). The
atmosphere↔surface split is clean (column RT takes surface-supplied α/ε/T, produces down-fluxes;
surface produces net + `lw_up`; the coupler blends flux-conservingly via shared helpers).

**Stray full-formula re-implementations that should route through the kernel** (the actionable
radiation dedup): `ocean/simple_ocean.py:215-217/275-277`, `ocean/simple_ocean_mpas.py:111,157`, and
`coupler/driver/physics_pipeline.py:472,478-479` (the pipeline land SEB) compute `(1-α)·sw` and
`ε·lw_down − εσT⁴` inline instead of calling `surface_radiation_fluxes`. The many partial
`εσT⁴+(1-ε)·lw_down` `lw_up` expressions (coupler ocean/ice/lake tiles, accumulator, tile blend) are
**intentional and kernel-consistent** — leave them or hoist a tiny `lw_up(T,ε,lw_down)` helper.
Legitimately distinct σT⁴ uses (atmosphere gray/RRTMGP Planck + surface BC, canopy multi-leaf RT,
snow-band sub-grid RT, ice multi-band SW, the analytic `4εσT³` Jacobians) are not duplication.

### 8.3 Relevance to the land dedup

Most of Section 8 is outside the land surface schemes (ocean/coupler), so it is lower priority for
the land refactor. The **land-relevant** items are small and, as pure relocations, bit-for-bit:
(a) the pipeline land-SEB radiation re-implementation (`physics_pipeline.py:472-479`) → route through
`surface_radiation_fluxes`; (b) the canopy unstable-ψ sub-expression → import from `core/bulk_flux.py`.

---

## 9. Impact on AMIP results and site-level validation

The two validation targets exercise **disjoint code paths**, which is what makes most of this plan
safe:

- **Site-level (EC-site, `run_ec_site.py`)** = multilayer soil + **two-leaf canopy**
  (`TwoLeafCanopyConfig`) → `canopy/stomatal.py`, `canopy/photosynthesis.py`, the canopy soil-evap
  (Sellers + Kelvin), the multilayer soil kernels.
- **AMIP** (`run_amip.py` → `ModelDriver` → `PhysicsPipeline`) = a land tile whose
  slab-vs-multilayer choice is the `use_multilayer_land` config flag (**not** the
  `component_factory.py` `LandComplexity` rung — that drives the standalone/matrix builder, not the
  AMIP pipeline). The two AMIP land tiles are *different code*: **multilayer-AMIP** runs
  `step_multilayer_land` with **SimpleSEB** (`SimpleSEBConfig`, from `land_surface_scheme="simple_seb"`,
  mapped in `model_driver.py`) → `carbon/stomata.py` + the multilayer SimpleSEB bare-soil throttle
  (which *does* include the Kelvin `h_r`, see S3). **Slab-AMIP** runs
  `PhysicsPipeline._step_slab_land` (`physics_pipeline.py:443`) — a separate inline SEB that uses
  neither `SimpleSEBConfig` nor `compute_simple_seb_fluxes` nor `carbon/stomata.py` (fixed
  `beta_land` bucket ramp only). `land/slab_land.py:step_land` (the S1 target) is yet *another* slab
  SEB, used by the LMIP/SCM drivers and the coupled-ESM `coupler/coupler.py` tile — **not** the AMIP
  `ModelDriver` slab tile.

Because the two paths are disjoint at the leaf level, and the two-leaf canopy is the *richer* side,
**every result-changing refactor flows one direction — bringing the SimpleSEB/slab family up to the
canopy/CLM5 physics — so it touches the AMIP and/or standalone-/coupled-slab land paths but NOT the
site-level validation.**

### 9.1 Zero-impact (bit-for-bit; proven by the `SurfaceFluxOutput` oracle)

Change **neither** AMIP **nor** site-level results — pure relocations of identical math, gated by
field-by-field `SurfaceFluxOutput` equality:
- **Phase 1** stomatal-kernel dedup (both paths);
- soil **S2** (root-zone β), **S3** (Kelvin h_r), **S4** (snow partition), **S5** (bucket ramp);
- the §8.3 radiation/turbulent tidy-ups — *if* the inline formula matches the kernel exactly (if it
  does not, that is a latent inconsistency to surface, not a silent change).

Gate: the oracle passes bit-for-bit, and `test_land_stability.py`, `test_ec_site_run.py`,
`test_amip_deck_smoke.py` are unchanged.

### 9.2 Result-changing on the AMIP / standalone-slab side (site-level unaffected)

Bring the SimpleSEB/slab family toward canonical physics — result-changing on the AMIP and/or the
standalone/coupled-slab land paths (S1 is the standalone/coupled-slab one, not AMIP; see its bullet),
each gated by the relevant battery, leaving EC-site untouched:
- **Soil S1** (route `slab_land.py:step_land` through `compute_simple_seb_fluxes`): adds the
  exchange-coeff ceiling + condensation floor `slab_land.py` lacks. NOTE this is the
  standalone-LMIP/SCM and coupled-ESM (`coupler/coupler.py`) slab tile — **not** the AMIP
  `ModelDriver` slab tile, which is the separate inline `_step_slab_land` fork (that would need its
  own equivalent hardening). Effectively a hardening/bug-fix, expect small cold-start-only
  differences; multilayer-AMIP is unaffected either way.
- **Soil-evap fidelity** (give SimpleSEB the bare-soil Sellers dry-surface-layer resistance — the
  multilayer SimpleSEB already has the Kelvin `h_r`): changes AMIP
  latent heat / near-surface T. Pierre's own sweep bounds the sign (strengthening the throttle warms
  JJA +1.0→+2.5 K and barely moves soil moisture 0.215→0.219). It is **fidelity, not the fix for the
  dry/warm bias** (that is a water-budget issue, out of scope here), and must expose the **bare-soil**
  efficiency only — SimpleSEB currently throttles *total* LE, so a naive stronger resistance would
  wrongly suppress transpiration too.
- **Photosynthesis Phase 3** (SimpleSEB → canonical C3+C4): changes AMIP GPP and, via stomata, latent
  heat. A deliberate physics upgrade.

### 9.3 How to measure the AMIP effect

Run the same config on both trees and diff:
1. `scripts/validate/validate_land_era5.py` — the direct AMIP-land-vs-ERA5 climatology (2 m T,
   latent/sensible heat, soil moisture) — the primary acceptance number;
2. `tests/integration/test_amip_deck_smoke.py` + `scripts/validate/validate_amip_run.py` —
   finite-state end-to-end;
3. `tests/land/test_land_stability.py` — 30-day energy/water-budget closure (slab and multilayer).

For a result-changing step, "no regression" = ERA5 metrics within run-to-run/tuning noise (or
improved); for Phase 1 and the §9.1 set, it means **identical**.

### 9.4 Bottom line for the two targets

- **Site-level (EC-site) validation is insulated** — every part of this plan that touches the
  two-leaf path is bit-for-bit, so the EC-site figures/skill do not move.
- **AMIP moves only for the deliberate SimpleSEB upgrades** (soil-evap Sellers fidelity and
  photosynthesis convergence on the multilayer-AMIP `SimpleSEB` tile; plus, *if* applied to the
  AMIP `_step_slab_land` fork, the S1-style exchange-coeff/condensation hardening — S1 as scoped
  above targets `slab_land.py`, the LMIP/coupled-slab tile, not the AMIP pipeline), each gated by
  the AMIP-vs-ERA5 battery; the mechanical dedups (Phase 1, S2–S5, the radiation/turbulent
  tidy-ups) leave AMIP identical.

---

## 10. Priority summary (whole audit)

| Rank | Item | §  | Result change? | Effort |
|---|---|---|---|---|
| 1 | Stomatal-kernel dedup (Ball-Berry + Medlyn) | Phase 1 | **None** (bit-for-bit) | Low |
| 2 | `slab_land.py` → `compute_simple_seb_fluxes` | S1 | LMIP/coupled-slab (hardening; NOT the AMIP `_step_slab_land` tile) | Medium |
| 3 | `root_zone_moisture_stress`/`root_zone_beta_soil` merge | S2 | **None** | Low |
| 4 | Kelvin `h_r` helper | S3 | **None** | Trivial |
| 5 | Snow sublimation-partition helper | S4 | **None** | Medium |
| 6 | Pipeline-SEB radiation → kernel; canopy-ψ → core | §8.3 | **None** (verify) | Low |
| 7 | Bucket-β-ramp helper | S5 | **None** | Trivial |
| — | Soil-evap fidelity (SimpleSEB Sellers bare-soil resistance; Kelvin already present on multilayer SimpleSEB) | 7.3 | **AMIP** (decision) | Medium |
| — | SimpleSEB → canonical C3+C4 photosynthesis | Phase 3 | **AMIP** (decision) | High |

Ranks 1–7 are safe, mostly bit-for-bit de-duplications (start with #1). The two unranked items are
physics decisions for Pierre, each gated by the AMIP-vs-ERA5 battery; neither affects site-level
validation.

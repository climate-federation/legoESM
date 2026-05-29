# Oracle-Recipe Verification Strategy (ocean)

> **Status:** living doc — iterate freely. Scope: **ocean component only** for now;
> a full-legoESM expansion is gated on this pattern proving out (see §9).
> **Owner:** Dhruv. **Started:** 2026-05-28. **Branch of record:** `matching_Veros_oracle`.
> Companion docs: `phase_g_recipe_fidelity_plan.md`, `phase_g_veros_recipe_audit.md`,
> `veros_acc_tendency_comparison.md`. Always-loaded rule summary lives in
> `CLAUDE.md` → "Oracle-Recipe Fidelity" (see §10 for the proposed block).

---

## 1. The problem and the vision

**Problem.** When a legoESM simulation doesn't match expectations, three causes are
confounded and indistinguishable from the output alone:

1. a **code bug**,
2. a different **component-wiring** choice (which blocks were assembled), or
3. different **parameter / constant** settings.

This confound is the central obstacle to confidence in a modular ("lego") model.

**Vision.** Because the blocks can be assembled many ways, designate configurations
that replicate an existing production model *exactly* — same wiring, same parameters,
same constants, same numerics. Configured identically, legoESM should produce the
**same answers**: not bitwise (float-op order differs), but **per-timestep tendencies
match to tolerance**, and in long runs **statistics match**. Any residual divergence
on such a config is then unambiguously a **bug** — the confound collapses.

This verifies *some* combinations against production-model **oracles**, enabling rapid,
high-confidence progress. Novel research combinations (no oracle) are developed more
slowly, but inherit confidence from **block-level** and **coupling-contract** verification
(see §3, rule D).

**Verification ≠ validation.** Oracle-matching is pure *verification* (are we solving the
equations right?). It is **not** *validation* against nature (are these the right
equations?). "Matches Veros" ≠ "correct" — see §3 rule E on inherited bugs.

**This strategy is ADDITIVE — legoESM's existing testing stays first-class.** legoESM
already has a large, mature verification suite, and it must keep growing and stay green:

- The **ocean unit-test suite** (hundreds of tests under `tests/ocean/`), the **ocean test
  matrix** (~39 cases across cubed-sphere / lat-lon / MPAS), **conservation & budget
  diagnostics**, **MPI / distributed parity** tests, and the **analytic benchmark** suite
  (Williamson, Stommel/Munk, Eady, IGW, geostrophic adjustment, lock-exchange).
- The **visual-verification discipline** (CLAUDE.md): passing tests + error norms are
  *necessary but not sufficient* for grid artifacts (cube imprint, halo seams, edge noise)
  — those are caught only by inspecting fields (e.g. Williamson-2 v-wind). Oracle work does
  not relax this.
- The **`FidelityCase` tiers 0–8** harness and golden-master artifact flow already in place.

The oracle-recipe layer **slots into** this — equivariance tests are tier-0/1, oracle
tendency-matches are tier-3 — it does **not** supplant unit tests, the matrix, conservation
checks, or visual verification. In fact, most existing tests *are* the truth-based tiers
(0–2) that the doctrine deems **more authoritative** than oracle-matching (§2), so "keep
doing the existing testing well" is the load-bearing part of this plan, not an afterthought.
Concretely: every new block keeps its direct unit test (CLAUDE.md code-hygiene); the matrix
stays green; visual verification stays mandatory; and the §6 de-mirroring refactor must
leave the **full suite green at every step** (G-C1…G-C5 each end green).

---

## 2. The verification ladder

Verification is **tiered**, cheapest/most-authoritative first. The existing
`ocean/fidelity/registry.py` (`FidelityCase(tier 0–8, metric_fn, reference_fn,
ci_marker)`) already implements this scaffolding, including `reference_fn` that may be
**analytic / literature / external** and `ci_marker ∈ {fast, nightly, manual_only}`.
The harness "never re-integrates the model: it consumes artifacts emitted by
`run_ocean_test_matrix.py`" — i.e. **golden-master**: comparison against a committed
reference runs in CI without the external oracle installed.

| Tier | Check | Reference | Oracle needed? | CI |
|------|-------|-----------|----------------|-----|
| 0–1 | Conservation; **equivariance / convention-invariance**; rest-state PGF≈0; reversibility | invariant (truth) | none | every PR |
| 2 | Analytic / manufactured solution (Stommel/Munk transport, Eady growth rate, geostrophic adjustment, IGW dispersion, Kelvin wave speed) | closed-form (truth) | none | every PR |
| 3 | Per-process **tendency match** vs reference model on a frozen state | external model (frozen golden artifact) | to *produce* golden; not to *check* | every PR (vs golden) |
| 4 | Short bounded-time **trajectory** match | external model | yes | nightly/manual |
| 5–8 | Long-run **statistics** match | external model / obs | yes | nightly/manual |

**Principles for using the ladder:**

- **Push verification UP the ladder.** Prefer truth-based tiers (0–2) over model-matching
  wherever a closed-form or invariant exists. They are cheaper, run every PR, need no
  external model, and verify *correctness* rather than *imitation*.
- **Model-oracles (tier 3+) earn their keep only where the upper tiers cannot reach** —
  i.e. coupled multi-process behavior with no closed form ("PE + KPP + GM/Redi + bulk
  flux, wired together").
- **Freeze the golden.** Once a tier-3 match is achieved, commit the oracle's reference
  tendencies as an artifact so CI compares against it without the oracle installed. A
  `manual_only` job **refreshes** the golden against the live upstream model on a
  declared cadence, or the frozen reference silently drifts from upstream.
- **The long-run statistical tier (5–8) is the weakest and most expensive.** Treat it as
  a coarse sanity gate, not a precision instrument — same-model runs differ in statistics
  across compiler/decomposition/resolution. Do not over-invest wall-clock there expecting
  bug-finding power.
- **Defense-in-depth against inherited bugs:** a tier-3 oracle match is only trusted for a
  block that *also* clears tiers 0–2. The truth tiers catch bugs inherited *from* the
  oracle; the oracle catches wiring/convention bugs the truth tiers miss. A bug survives
  only in the intersection.

---

### 2.1 Execution surfaces (CI vs deployed)

Oracle-matching is **not** monolithically "out of CI" — it factors into a *cheap CI half*
(compare legoESM against a committed golden) and an *expensive deployed half* (run the live
oracle to produce/refresh that golden). Where each tier runs:

| Tier | What | Execution surface |
|------|------|-------------------|
| 0–2 + clarity guards | conservation, equivariance, analytic/MMS, structural-modularity | **CI, every PR** — deterministic, no oracle |
| 3 — frozen-state tendency match | legoESM probe on a committed frozen state vs **committed oracle golden** + committed tolerance | **CI-able** (fast/nightly) — the cheap comparison half |
| (golden refresh) | run **live oracle** (Veros) to regenerate the reference | **deployed on demand** + scheduled cadence |
| 4 — short trajectory match | real legoESM integration vs oracle trajectory | nightly → mostly **deployed** |
| 5–8 — long statistical match | multi-year runs of both models | **deployed only** (GPU box), never CI |

This maps directly onto the existing `ocean/fidelity/` harness: the `ci_marker ∈ {fast,
nightly, manual_only}` field; the design rule *"the fidelity layer never re-integrates the
model — it consumes artifacts"*; and heavy optional deps (Veros, copernicusmarine) imported
only when the relevant submodule is touched. The live driver
`scripts/ocean_fidelity/compare_tendencies_acc.py` is the `manual_only`/deployed piece; the
metric-vs-committed-tolerance fixtures are the `fast` piece.

**Obligation that comes with moving the live oracle out of CI:** the committed golden can go
**stale** vs upstream Veros — if Veros later fixes a bug, the frozen reference still encodes
the old behavior and CI happily verifies against an outdated truth. So "deployed when needed"
**must include a *scheduled* golden refresh** (a `manual_only`/cron job on the GPU box), not
purely ad-hoc — otherwise you trade compute cost for a silent-staleness risk. Track the
refresh cadence per oracle in the §8 ledger.

---

## 3. Doctrine (the rules)

**A. A recipe is pure config over shared canonical blocks.** A recipe that reproduces a
production model is built by *selecting* canonical blocks and pinning parameters/constants
— never a bespoke `veros_*` solver. Numerics added to match an oracle land in the
**canonical module** (`eos.py`, the advection/limiter dispatch, `vertical_mixing/`, the
integrator dispatch) as **selectable options**, reachable by any config.
*Status: `veros_acc_recipe.py` already complies — it builds a stock
`LatLonCGridOceanConfig` + `OceanPhysicsConfig`.*

**B. Mimicry-only glue stays in the fidelity harness, never the model.** Decision test:
> *Would a user pursuing a different goal ever select this?*

- **Yes** → canonical shared block (Robert-Asselin filter, superbee, nonlin2 EOS, TKE,
  cos(lat) viscosity). Matching the oracle then verifies real production code.
- **No** — it exists only to mimic this one oracle (halo strip, axis transpose,
  time-level handling, wall padding) → it lives in the **bridge / harness**
  (`veros_state_bridge.py`, `tendency_probe.py`), outside the model.

**C. Conventions are handled only at I/O boundaries (the bridge), never as model-core
flags.** Physics is invariant to conventions (see §4). The model core works in **one
canonical representation**; only the bridge re-encodes. No `vertical_index_direction`
toggle on a solver. Genuine *physics* choices get config flags; *conventions* never do.

**D. Verify at block granularity + a coupling-contract test, so confidence transfers to
novel combos.** Oracle-matching a whole config only verifies that config. To make a novel
(un-oracled) combination trustworthy, verify each **block** in isolation (against
truth/oracle) and verify the **coupling contract** (interface shapes, flux signs,
conservation across the coupler) independently. A novel combo = verified blocks +
verified contract. *Caveat:* a block verified at one regime/parameter set is not
automatically correct at another (KPP matched at ACC params may be wrong at the equator);
block tests should span regimes.

**E. The oracle is not truth.** Matching a model reproduces its bugs, quirky conventions,
and compensating errors. Tier-3 matches must sit *under* truth tiers (§2). Watch for
compensating-error false greens (a block bug cancelling a convention mismatch).

**F. Constants are config, not module-global monkey-patches.** No `override_constants`-style
global mutation in shippable paths. `ConstantsConfig` defaults reference
`legoesm.constants`; base constants only; derived constants (κ, ε) recomputed from the
base, never read stale. (See §6.)

**G. Per-oracle discipline — declare before you chase.** Before starting an oracle match,
write down: the **tolerance** per field per region, the **list of conventions** you are
replicating, the explicit **non-goals** (differences that are expected/acceptable), and a
**timebox**. Without this, residual-chasing (e.g. Phase G's 0.5% density residual) becomes
an open-ended research project that consumes the velocity the strategy was meant to create.

**H. Gap → block: a missing capability revealed by an oracle goes into the canonical module,
never a `veros_*` clone.** When matching an oracle exposes that legoESM lacks a feature, the
fix is a new **config-selectable option in the canonical module**, gated on the truth tiers
(§2), not on the oracle match. First **classify** the gap:
- **Missing *method*** (a real, reusable algorithm — e.g. flux-form momentum advection) →
  add the block (`momentum_advection="flux_form"`).
- **Missing *variant* of a block we have** (e.g. Veros's isopycnal discretization vs our
  triads) → extend the existing block with an option; don't fork it.
- **Convention, not physics** (§4 φ-test) → bridge only; add nothing to the model.
- **Oracle bug / legacy quirk** → do NOT bake into the canonical block; if a recipe must
  reproduce it, that is a clearly-marked recipe-level compat flag.

A new block is a first-class lego: it must clear the truth tiers (conservation, equivariance,
idealized cases) **independently** — an oracle match (tier 3) alone is the lowest-trust
evidence — pass the modularity tester, carry a direct unit test, and join the supported
matrix. Prioritize by **real need** (a recipe's fidelity target, or a researcher), not by
completeness: "legoESM has every block" is the destination, not a licence to preemptively
clone every oracle closure (that breeds untested-but-live code). Track each gap in the
per-oracle **missing-blocks ledger** (§8).

**I. One concept, one name, one implementation.** Oracles name the same quantity differently
(Veros `kappaM` ≈ legoESM `A_v`; `r_bot` = `bottom_drag_r`; `K_gm_0` = `kappa_GM`); left
unmanaged this breeds parallel vocabularies and copy-paste numerics that "differ only in
parameters." Two defences:
- The **concept registry** (`ocean/fidelity/concept_registry.py`) is the canonical
  cross-oracle map: concept → canonical legoESM name + aliases + per-oracle names + units.
  Bridges/recipes translate oracle vocabularies **through it**, so a synonym is recognised,
  not re-coined.
- A **two-layer auditor** (§9) enforces it: a deterministic CI guard (registry consistency +
  an alias ratchet that only shrinks) plus an agent-based semantic pass
  (lego-modularity-tester **dimension 10**) that finds same-thing-different-name functions and
  "differs-only-in-parameters, generalise" duplication no text scan can reach.

Before adding any numeric helper, search for an existing one (CLAUDE.md pre-impl rule) **and
consult the registry**; if the thing exists under another name, extend or rename — never
re-implement.

---

## 4. Convention-invariance: criterion + equivariance tests

**Decidable criterion.** A difference between two models is a **convention** iff there
exists a bijective re-encoding φ (flip vertical index, transpose axes, change pressure
units, relabel face ownership, halo width) such that

> **physics(φ(x)) = φ(physics(x))**  to truncation tolerance.

- **Convention** (equivariant under φ) → handle in the **bridge only**; transform away;
  never replicate inside the model.
- **Physics/numerics choice** (not equivariant) → **replicate as config** (EOS form,
  advection scheme, integrator, grid type, land-mask inequality).

The criterion also catches **mislabeled conventions**: the strict `x > 1.0` land mask in
`build_acc_land_mask` *looks* like a convention but changes which cells are wet → it is
**not** a physics-preserving bijection → it is geometry/physics and must be **replicated**,
not transformed.

**Equivariance tests (tier-0/1, oracle-free, CI).** Operationalize the criterion as
symmetry tests. These push convention-bugs **down** from tier-3 (oracle, expensive, needs
the external model) to tier-1 (cheap, every PR):

| Test | φ | Catches |
|------|---|---------|
| EOS unit-invariance | pressure in Pa ↔ dbar ↔ depth-m | the Pa↔depth-m bug already fixed in `6569a22f` |
| Vertical-flip equivariance | k=0-surface ↔ k=0-bottom | bridge/model index-direction errors; hydrostatic cumsum order |
| Axis-transpose equivariance | (lat,lon) ↔ (lon,lat) | bridge transpose errors |
| Halo-width invariance | 1-cell ↔ 2-cell halo | halo-dependent leakage |

Generalize the **existing pattern**: `test_tripole_fold.py` already does round-trip /
involution checks (a convention-invariance test for the fold). Lift it into a systematic
equivariance tier across the ocean blocks.

**Caveat — float non-associativity is diagnostic, not a flaw.** Even a pure convention
(e.g. vertical flip reorders the hydrostatic `cumsum`) is not bitwise-invariant in float
arithmetic. So equivariance holds *to tolerance*. A tight-tolerance equivariance test
therefore **partitions** a discrepancy into "benign float-order" (≈ machine ε; ~1e-7 in
float32) vs "real asymmetry" (bug). For Phase G specifically: a vertical-flip equivariance
test would tell us whether the 0.5% density residual is a convention-handling bug in the
bridge or a genuine numerics difference — **without another Veros run**.

---

## 5. Recipes as first-class, user-invocable configs

Oracle recipes graduate from `fidelity/` fixtures to **shipped configs** users can import
and run — "legoESM-as-Veros, validated" — gaining legoESM's superpowers (architecture-
agnostic, differentiable, GPU/TPU/MPI) on a fidelity-verified path. Honest scope: this
holds **on the verified recipe path**, not arbitrary assemblies, and is not bitwise repro.

> **DECISION (2026-05-28): strict, verifiability-first.** No users are waiting, so we do
> not ship a loose "Veros-style defaults" recipe early. The recipe earns its
> "configured-identically-to-Veros, validated" status only when §6 (`ConstantsConfig`,
> G-C1…G-C5) **and** a committed tier-3 golden are done. Get the model verifiably amazing
> first; ship after. The entire graduation checklist below is therefore a **hard gate**.

**Graduation checklist (per recipe):**

- [ ] Kill the constants monkey-patch — correctness must not depend on a `with
      override_constants(...)` context (see §6). **Blocker, not cleanup.**
- [ ] Move the recipe (pure config bundle) out of the `fidelity/` namespace (which carries
      heavy optional deps: Veros, copernicusmarine) into a user-facing home
      (`ocean/recipes/`).
- [ ] Keep its `FidelityCase` registration in `fidelity/` as a permanent **regression
      guard** (tier-3 vs committed golden).
- [ ] Establish a `manual_only` **golden-refresh cadence** against upstream Veros.
- [ ] Direct unit test that the recipe **constructs** and round-trips through the public
      config API (per CLAUDE.md code-hygiene).

---

## 6. Implementation plan — ocean-scoped `ConstantsConfig`

*Grounded in the 2026-05-28 recon of constant-access sites and the config/recipe surface.*

### 6.1 What goes in it

Exactly **5 base constants**: `g`, `Omega`, `R_earth`, `rho_0`, `c_sw`. Defaults
reference `legoesm.constants`. **EOS coefficients are NOT included** — they already live
in `VerosNonlin2Config` / `VerosNonlin3Config` / `LinearEOSConfig`.

```python
# ocean/constants_config.py  (new)
class ConstantsConfig(NamedTuple):
    g:        float = constants.g            # 9.80665  — traced (tendency)
    rho_0:    float = constants.rho_ocean    # 1025.0   — traced (tendency)
    c_sw:     float = constants.c_sw         # 3991.0   — traced (tendency)
    Omega:    float = constants.Omega        # 7.292e-5 — static (mesh/Coriolis at init)
    R_earth:  float = constants.R_earth      # 6.371e6  — static (grid metrics at init)
```

### 6.2 Static vs traced — resolved by *where each is consumed*

A NamedTuple of Python floats is static-by-default and serves **both** roles, because the
split maps onto the consumption site (recon-confirmed: none flow into array shapes or
static control flow):

- **Static-init** — consumed at grid/mesh **construction, outside JIT** → stay concrete.
  - `R_earth`: `deg_to_m`/`_degtom`, bathymetry scaling, experiment IC setup, grid metrics.
  - `Omega`: main Coriolis `f` is precomputed into `grid`/`mesh.fEdge` at init.
- **Traced-physics** — consumed **inside jitted tendencies** → as pytree leaves become
  traced ⇒ **differentiable** (opt-in: pass a JAX scalar to take ∂/∂constant).
  - `g`: buoyancy fluxes in KPP / TKE / k_profiles; hydrostatic pressure; N².
  - `rho_0`, `c_sw`: heat/salt/freshwater flux→tendency conversions (coupler, forcing).
  - `Omega`: the optional GM-Visbeck path recomputes `f` from latitude at tendency time.

### 6.3 The real work is **de-mirroring**, not threading

The PE dynamical core already threads `config.g`/`config.rho_0` cleanly
(`iterate_eos_and_pressure_anomaly(rho_0, g, ...)` in `ocean_tendency_common.py`). The
leaks that the monkey-patch papers over are:

1. **Module-level mirror bindings** (the `_CONSTANT_SHADOWS` targets) — replace each with
   a read from the threaded `ConstantsConfig`:
   - `rho_0`: `eos.rho_0`, `diagnostics._RHO_0`, `shortwave._RHO_0_DEFAULT`,
     `mpas_physics.rho_0_ref`, `_gm_redi_common._RHO_0_DEFAULT`, `gm_redi_mpas._RHO_0`,
     `gm_redi_latlon_cgrid._RHO_0`, `prescribed.rho_0_ref`, `bulk_formulas.rho_0_ref`
   - `c_sw`: `eos.c_sw`, `shortwave._C_SW_DEFAULT`, `prescribed.c_sw`, `bulk_formulas.c_sw`
2. **Inline `constants.X` reads in tendencies** — replace with config reads:
   - `constants.g`: `vertical_mixing/{integration,mpas_integration,k_profiles,tke}.py`
     buoyancy fluxes; GM Visbeck.
   - `constants.rho_ocean` / `constants.c_sw`: `coupler/{runoff_apply, ice_shelf_apply,
     omip2_applicator}.py`, `forcing/sss_restoring.py`.
3. **One cross-boundary leak** outside `ocean/`: `legoesm.coupler.bulk_flux.G` (the `g`
   shadow). Out of ocean scope — wrap at the ocean/coupler seam, or defer to the
   full-ESM phase. (This is exactly the kind of seam that makes full-ESM its own phase.)

### 6.4 Staged steps (each ends green; minimal local diffs)

- **G-C1** — Define `ConstantsConfig`; add `constants: ConstantsConfig = ConstantsConfig()`
  to `LatLonCGridOceanConfig` (and thread into `OceanPhysicsConfig`/`make_ocean_physics`).
  Defaults = `constants.*` ⇒ **zero behavior change**, full suite green. Add direct unit test.
- **G-C2** — De-mirror the `ocean/` `rho_0`/`c_sw` module bindings (1) + inline tendency
  reads (2) in physics. Each site reads from threaded `ConstantsConfig`.
- **G-C3** — Thread `ConstantsConfig` into `ocean/coupler/*` and `forcing/*` flux→tendency
  conversions; de-mirror their reads.
- **G-C4** — Migrate the recipe: replace `with override_constants(**VEROS_CONSTANTS):
  build_acc_recipe()` with `ConstantsConfig(**VEROS_CONSTANTS)` injected into the config
  (and into `acc_A_h`/`_degtom`). Delete `recipe_constants.py` monkey-patch (or keep
  `override_constants` as a deprecated test shim for one cycle).
- **G-C5 (guard)** — Audit test (extends the existing constant-discipline audit): fails if
  any `ocean/` tendency/physics/coupler path reintroduces a direct
  `constants.{g,rho_ocean,c_sw}` read or a module-level mirror binding.

### 6.5 Splits to get right (or it breaks)

- **Base vs derived:** only base constants in config. κ = R_d/c_pd, ε = R_d/R_v etc. must
  be **recomputed from the config base**, never read from a now-stale module global. The
  `_CONSTANT_SHADOWS` registry was hand-maintaining this dependency graph; config makes it
  explicit, but the recompute must be implemented.
- **Split-brain during migration:** while some code reads the global and some reads config,
  a block can silently use the Earth default while the recipe set Veros's `rho_0`. Mitigate
  with config-default-references-`constants` (single source) + the G-C5 audit.

---

## Code-clarity & decomposition modularity (verification enabler) — sequence BEFORE §7

**This is not standalone cleanup — it is a prerequisite for the verification tiers.** You
cannot assert `physics(φ(x)) = φ(physics(x))` on "the PGF stage", nor write the
block-granularity tests of §3 rule D, when that stage is 80 LOC buried at offset ~1085
inside a single 1299-line function. So decomposition is sequenced **ahead of** the
equivariance tier (§7). (Full assessment: legoESM's operators/recipe/docs are *cleaner than
Veros*; the solver orchestration + config surface are the debt.)

**Make it a RUNNABLE auditor, not a one-off refactor:**

1. **New `lego-modularity-tester` dimension — "structural / decomposition modularity"
   (static, read-only).** Runtime swap-modularity (the agent's dimensions 1–8) is necessary
   but not sufficient: a monolithic fused solver is a *structural* modularity failure — you
   can't isolate, test, equivariance-test, or swap a stage you can't address. The agent
   flags: monolithic tendency/step functions over a LOC ceiling (~400) not decomposed into
   named substage helpers (known: `latlon_cgrid_ocean_baroclinic_tendencies` ≈1299,
   `_step_impl` ≈572); solver entry points below a docstring floor; config NamedTuples
   >~25 fields with no grouping (`LatLonCGridOceanConfig`=45); two-source-of-truth params
   (`A_h` in dynamics *and* physics config); deprecated-but-live config (physics-level
   bottom drag); buried mode-switch booleans (`implicit_vertical_mixing`). For each, it
   reports whether decomposing/grouping unblocks a stage-level unit/equivariance test.
   *(Added to `.claude/agents/lego-modularity-tester.md` as dimension 9.)*

2. **Deterministic CI guard** (cheap, every-PR; the audit-enforced style CLAUDE.md already
   uses) for the mechanizable subset: a test that fails on (a) any ocean tendency/step
   function over the LOC ceiling without an allow-list entry, (b) a physical-parameter name
   defined in >1 config, (c) a deprecated-but-live config field. Locks the gains so the
   monolith can't silently regrow.

3. **First application (not the whole track):** decompose
   `latlon_cgrid_ocean_baroclinic_tendencies` (1299 LOC) into ~12 named pure substage
   functions (~100 LOC reorg, **same numerics**, operators unchanged) — then each substage
   gets the §7 equivariance tests. Also: group/section the 45-field `LatLonCGridOceanConfig`
   (+ class docstring with minimal-vs-production example), and resolve the 3 footguns
   (single source for `A_h`; remove/clearly-gate deprecated physics bottom drag; validate
   `eos`/`eos_linear` coupling per dispatch discipline).

`/slopbuster` remains the periodic qualitative code-quality pass; dimension 9 is the
structural-modularity-specific, verification-aligned audit.

## 7. Equivariance-test tier (implementation)

> **Status 2026-05-28: first tests landed** — `tests/ocean/unit/test_equivariance.py`
> (5 tests, all green): Pa↔depth round-trip, nonlin2 pressure-unit pinning, EOS point-wise
> vertical-flip equivariance (wright + veros_nonlin2), and the hydrostatic-pressure
> cumsum-order diagnostic. **Diagnostic result (residual partitioned, oracle-free):**
> (c-order) cumsum/flip summation order → Δρ ≈ 2.5e-15 kg/m³ (~10¹⁵× below the ~5 kg/m³
> residual) — **exonerated** (tested); (c-depth) legoESM's actual-ρ `∫ρg dz` vs Veros's
> geometric-depth `abs(zt)` pressure argument → bounded at ≈ **0.05 kg/m³** on an ACC column
> (≤11 m depth offset) — too small; (b) wall-row zero-padding feeds **T=0,S=0** to the EOS →
> **−27 kg/m³** per contaminated cell — the only candidate with the right magnitude.
> **Prime suspect: wall-row padding** (memo cause b). Fix: mask wall rows in `land_mask` /
> interpolate them so the EOS isn't evaluated on T=0,S=0. *Next equivariance increment:* a
> bridge round-trip test (oracle-free) asserting the bridged state has no T=0,S=0 wet cells.

- Add a `tests/ocean/unit/test_equivariance.py` (or register tier-0/1 `FidelityCase`s) for
  the φ-transforms in §4: EOS unit-invariance, vertical-flip, axis-transpose, halo-width.
- Each asserts `physics(φ(x)) ≈ φ(physics(x))` to a declared tolerance; tolerance chosen so
  benign float-order passes and real asymmetry fails (and report which).
- Generalize the `test_tripole_fold.py` round-trip/involution helpers rather than
  duplicate (per CLAUDE.md anti-duplication).
- **Near-term payoff:** run the vertical-flip + EOS-unit equivariance tests against the
  Phase G ACC frozen state to partition the 0.5% density residual (bridge convention bug
  vs genuine numerics) before spending another Veros run.

---

## 8. Open questions, risks, non-goals

**Resolved**
- Recipe public contract: **strict** "configured-identically-to-Veros" — *decided
  2026-05-28* (§5). §6 + a committed golden are a hard gate before the recipe ships its
  validated form. Verifiability-first; no early loose-defaults release.

**Open questions**
- Which oracles graduate to shipped recipes (Veros ACC first; then MOM6/MITgcm)?
- Oracle authority vs tractability: Veros is easy to match but low-authority; MOM6/NEMO are
  authoritative but entangled (ALE remapping resists clean decomposition).

**Risks**
- Entangled oracles may not be reconstructable by pure composition → that failure is
  diagnostic: either the decomposition is incomplete, or the model is **statistics-only**
  (tier 5–8), never config-level (tier 3). Decide per oracle; do not assume.
- Golden drift if the `manual_only` refresh cadence lapses.
- Convention ledger rot — replicated conventions masquerading as physics over time.

**Non-goals**
- Bitwise reproducibility.
- "Matches Veros" treated as "validated against nature."
- Verifying arbitrary novel combos via oracles (those rely on §3 rule D).

**Per-oracle ledger (maintain one table per oracle):** field-by-region tolerance |
conventions replicated | declared non-goals | timebox | golden-refresh cadence.

### Veros ACC ledger (2026-05-28)
- **Density (rho):** interior/boundary/equator/ML L2 ≈ 0.037 kg/m³, corr 1.0000 — PASS.
  Residual is the rho-vs-geometric-depth pressure-argument difference (bounded ≤0.05) +
  time-level; declared non-goal to drive lower.
- **Coriolis (du/dv_cor):** interior corr 0.96 / 0.98, weighted-sign 0.98 — PASS. Validates
  grid alignment + bridge + face→centre interpolation.
- **Momentum advection (du/dv_adv):** corr 0.58 / −0.34 — **documented formulation delta,
  NOT a bug.** legoESM is vector-invariant (advection = ζ×u + ∇KE + w∂u/∂z, with ∇KE lumped
  into `pgf_ke`); Veros is flux-form (∇·(uu)). The two cannot be cleanly mapped per-process.
- **Momentum mixing (du/dv_mix):** corr ~0 — **documented structural delta, NOT a bug.**
  The ACC recipe uses `implicit_vertical_mixing=True`, so legoESM produces no *explicit*
  vertical-viscosity tendency to compare against Veros's `du_mix`.
- **Tracer per-process (iso = GM/Redi):** CORRECTED 2026-05-29. The earlier "inconclusive /
  uniform-S" note was wrong: GM/Redi was **mis-wired and inactive** (set only in
  physics.lateral_mixing; the lat-lon model reads the top-level config.gm_redi). Fixed
  (c5abe950) + the probe now computes GM/Redi directly (`gm_redi_tracer_tendency_latlon`).
  With GM/Redi active: **T_iso L2 1.5e-9, corr 0.17** — GM/Redi IS now compared, but
  legoESM's `gm_redi_latlon_cgrid` vs Veros's isoneutral scheme correlate poorly (a genuine
  GM/Redi formulation/taper/triad difference — the most implementation-divergent param;
  signal is also tiny at 10 days). Documented delta under the default acceptance approach;
  candidate for deeper verification (integral/budget) if GM/Redi fidelity must be tightened.
  **S_iso ≈ 0 is physically correct** (uniform S=35 → no isopycnal salt flux; Veros's is ~0
  too). **vmix** stays out (implicit in legoESM → no explicit tendency).
- **OPEN DECISION (verification-philosophy, owner = user):** tier-2 frozen-state PER-PROCESS
  comparison on ACC carries clean validating signal ONLY in density (0.037, corr 1.0) and
  Coriolis (0.96–0.98). Momentum advection (vector-invariant vs flux-form), momentum mixing
  (implicit), and tracer mixing (uniform-S/implicit/weak) are all documented deltas or
  no-signal. So the literal Q2 gate (`corr>0.95 per process`) is the wrong bar. Options:
  (a) accept density+Coriolis as the validating subset, document the rest (current);
  (b) verify momentum/mixing via *total* tendency or an energy/enstrophy/tracer-variance
  budget instead of per-process; (c) enrich the recipe (tracer-active IC, explicit mixing)
  and/or expose legoESM's KE-gradient + implicit-mixing effective tendencies in the probe.
  Governs future MOM6/MITgcm matching. This is the recurring crux of per-process tier-2.

### Veros ACC missing-blocks ledger (per doctrine rule H)

Each documented delta is classified as **add** (new canonical block) / **extend** (variant of
an existing block) / **bridge** (convention) / **accept** (legitimate, leave) / **defer**.

| Gap (legoESM vs Veros ACC) | Class | Response | Priority |
|---|---|---|---|
| **Flux-form momentum advection** — legoESM lat-lon C-grid only has `vector_invariant`/`weno5`/`weno7`; Veros/MOM6/MITgcm use flux-form ∇·(uu). Drives the `du_adv` corr 0.58 delta. | Missing **method** | **Add** `momentum_advection="flux_form"` (+ `momentum_flux_scheme`) to the canonical dycore; reuse `divergence_cgrid` + the existing face-interp/limiter schemes. Gate on conservation + Stommel/Munk gyre + zero-velocity + equivariance + grad (NOT just oracle-match). See `flux_form_momentum_scope.md`. | **High** — broadly useful; closes the momentum delta |
| **Veros isopycnal/GM-Redi discretization** — we have GM/Redi (`slope_scheme` = triads/centered); Veros's isoneutral scheme differs (T_iso corr 0.17). | Missing **variant** | **Extend** `gm_redi` with a Veros-discretization option, *or* (if our triads are the better method) keep ours and **accept** the delta with a documented justification. Decide on need. | Medium |
| **EKE closure** — Veros eddy-kinetic-energy parameterization; no legoESM equivalent. | Missing **method** | **Add on demand** only when a recipe/researcher needs it; not preemptive. | Low |
| **IDEMIX** — internal-wave energy/mixing; disabled in the ACC adapter anyway. | Missing **method** | **Defer.** | Low |
| **Implicit vs explicit vertical mixing** for ACC (`du_mix` corr ~0). | — (config, already exists) | **Accept** — both paths exist; the recipe selects implicit. Not a gap. | — |
| **Density / EOS / hydrostatic pressure** (corr 1.0). | — | **Accept** — already matches (shared discretization). | — |

### Cross-oracle naming (concept registry seed)

Synonyms found in the 2026-05-29 survey, now in `ocean/fidelity/concept_registry.py`:
Veros `kappaM`/`kappaH` ≈ legoESM `A_v`/`K_v`; `r_bot` = `bottom_drag_r`; `K_gm_0`/`K_iso_0`
= `kappa_GM`/`kappa_Redi`; `nz` = `n_levels`. Internal debt tracked there too (`nlev` vs
`n_levels`, `T_sfc` vs `T_surface`, `c_sw` vs `c_ocean`/`c_p`, `bottom_drag_coeff`, the
`tau_relax` days-vs-seconds and `C_water`/`c_water` unit hazards). Actionable item flagged:
**DINO hardcodes `c_p=3991.86`** instead of pinning via `ConstantsConfig` — fix in a cleanup PR.

---

## 9. Concept registry & duplication auditor (doctrine rule I)

The risk doctrine rule I addresses: as we match more oracles, the same physical thing arrives
under many names, and near-identical numerics accumulate ("differs only in parameters"). Two
layers, because the problem is part mechanical and part semantic.

**Layer 1 — deterministic CI (every PR, no oracle install, no false positives):**
- **`ocean/fidelity/concept_registry.py`** — the canonical map: concept → canonical legoESM
  name + aliases (debt) + per-oracle names + units + status (`canonical` / `unit-hazard` /
  `entrenched` / `fidelity-scoped`). The single source bridges, recipes, and the auditor read.
- **`tests/ocean/unit/test_concept_registry.py`** — (a) registry internal consistency (no
  canonical doubles as an alias, oracle keys known); (b) an **alias ratchet**: an
  `ratchet=True` alias may only appear in files in a committed `ALIAS_BASELINE` — a new file
  using it fails the gate (debt only shrinks, like `LOC_ALLOW_LIST`); (c) a non-vacuous
  detector check. Seeded with `bottom_drag_coeff` (→ `bottom_drag_r`). Entrenched debt (`nlev`,
  `T_sfc`) and unit hazards (`tau_relax`, `C_water`/`c_water`) are **documented but not
  auto-ratcheted** — an identifier scan can't judge units, and a 3566-site gate is impractical;
  those are dedicated-cleanup-PR / dimension-10 items.

This layer composes with the existing guards rather than replacing them:
`test_no_scheme_duplication.py` (shared-block use), `test_clarity_guards.py` (LOC ceiling),
`test_constants_audit.py` (no re-mirrored constants), `test_config_footguns.py` (single-source
config fields). The gap they share — **semantic** equivalence under different names — is layer 2.

**Layer 2 — agent-based semantic audit (on-demand / nightly): lego-modularity-tester
dimension 10.** What no text/AST-pattern scan can reach:
- **synonym functions** — two functions computing the same thing under different names
  (`apply_sponge` vs `restore_tracers`);
- **parameter-only duplication** — helpers that differ only in arg names / indexing / defaults
  and should be one generic function (the survey already found: sponge-γ
  `compute_sponge_gamma_latlon`/`_mpas`; DM95 taper; bottom-drag padding; vertical-mixing vmap
  stacking — all `generalizable=True`);
- **cross-oracle naming gaps** — an oracle concept not yet in the registry → propose a registry
  entry;
- **orphaned decompositions** — extracted helpers re-duplicated elsewhere.
The agent reports candidates with a generalisation suggestion + effort; a human approves the
merge and, where appropriate, adds the new shared helper + a deterministic guard so the
specific duplication can't recur.

**Division of labour:** layer 1 stops *known* synonyms/duplication from spreading (fast,
deterministic, ratcheted); layer 2 *discovers* new ones (semantic, adversarial). Findings from
layer 2 graduate into layer 1 (a new registry alias + baseline, or a new shared helper + a
`test_no_scheme_duplication` entry).

---

## 10. Expansion path to full legoESM

The ocean is the pilot. **Gate full-ESM expansion on the ocean pattern proving successful**
(a shipped, monkey-patch-free, fidelity-verified ACC recipe + the equivariance tier in CI).
On success, generalize: a model-agnostic `ConstantsConfig` threaded through atmosphere /
land / coupler (the repo-wide ~71-constant surface; the `legoesm.coupler.bulk_flux.G` seam
is the first cross-component case), and a model-agnostic recipe/registry layout. Until then,
keep this doctrine **ocean-scoped**.

---

## 11. Proposed `CLAUDE.md` rules block (pending approval)

Add under the ocean rules, with a one-line pointer to this doc. Kept short on purpose —
CLAUDE.md is always-loaded.

```markdown
## Oracle-Recipe Fidelity (ocean; see docs/ocean_fidelity/oracle_recipe_strategy.md)
- Recipe = pure config over shared canonical blocks. Reproducing a production model
  (e.g. Veros ACC) MUST select canonical blocks + pin params/constants — never a bespoke
  `veros_*` solver. Oracle-matching numerics land in the canonical module (eos.py,
  advection/limiter dispatch, vertical_mixing/, integrator dispatch) as selectable options.
- Mimicry-only glue → fidelity harness, never the model. Test: "would a user with a
  different goal ever select this?" Yes → canonical block. No (halo strip, axis transpose,
  time-level handling) → bridge/harness only.
- Conventions handled only at I/O boundaries (the bridge), never as model-core flags.
  A difference is a convention iff a bijective φ gives physics(φ(x))=φ(physics(x)) to tol
  → transform in bridge; else it's a physics choice → config. Verify via equivariance
  tests (tier-0/1, oracle-free, CI). A "convention" that changes the wet domain / answers
  is physics — replicate it, don't transform it.
- Constants are config (ConstantsConfig), not module-global monkey-patches. No
  override_constants-style global mutation in shippable paths. Defaults reference
  legoesm.constants. Base constants only; derived (κ,ε) recomputed from base.
- Oracle match (tier 3) is only trusted for a block that also clears the truth tiers
  (0–2: conservation, equivariance, analytic/MMS). Truth tiers catch oracle-inherited
  bugs; the oracle catches wiring bugs.
```
```

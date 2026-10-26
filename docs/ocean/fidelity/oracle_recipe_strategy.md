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
  - **Machine-enforced (#388 Ask#4):** `legoesm.ocean.fidelity.precedence.evaluate_precedence`
    turns this into a checked verdict — any *failing* truth tier (0–2) LOCKS the oracle
    tiers (≥3) so their match no longer counts. The scorecard generator
    `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` reports the verdict and
    exits non-zero on a violation (oracle present + truth-tier failure).

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
`scripts/validate/ocean_fidelity/compare_tendencies_acc.py` is the `manual_only`/deployed piece; the
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

**Apples-to-apples scoping (which blocks an experiment actually requires).** For a true
apples-to-apples match on a *given experiment*, the must-build set is the blocks the oracle
**activates for that experiment**, minus what legoESM already has — **not** everything the
oracle *could* do. A block the oracle has but runs with **off** in this experiment (e.g. Veros
ACC sets `enable_idemix=False`) is **not** built: run both models with it off — that is already
the oracle's own config — and the comparison stays honest. So: read the oracle's experiment
settings, list its *active* blocks, and build only the active ones legoESM lacks. (Veros ACC:
active = neutral/skew diffusion (GM/Redi), TKE, **EKE**, bottom + horizontal + implicit-vert
friction, nonlinear EOS type-3, and **flux-form momentum**; inactive = IDEMIX, explicit-vert
friction, quadratic drag. The must-build list was EKE and flux-form momentum — **both now built**
(flux-form adopted in the recipe; EKE closure built + oracle-verified, adoption pending the
`eke_len` mixing-length variant — see §8); IDEMIX etc. are *not* needed.) Note this also applies to
*paper* oracles: an experiment that replicates a paper pins the paper's parameter values via
config (e.g. DINO's `c_p=3991.86`, `rho_0=1026.0` from Kamm et al. 2025) — that is the doctrine
working correctly (oracle value, via config), **not** constant-discipline debt to "fix" toward
`legoesm.constants`.

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

### Veros ACC missing-blocks ledger (per doctrine rule H; verified against the Veros setup 2026-05-29)

Must-build = blocks **active in Veros's ACC setup** that legoESM lacks (apples-to-apples scoping,
rule H). Veros ACC active blocks (from `veros/setups/acc/acc.py`): neutral + skew diffusion
(GM/Redi), TKE, **EKE** (`enable_eke=True`), bottom/horizontal/implicit-vert friction, nonlinear
EOS type-3, **flux-form momentum**. Inactive (run both with off): IDEMIX (`enable_idemix=False`),
explicit-vert friction, quadratic drag. Each delta is classified **add** / **extend** / **bridge**
/ **accept** / **defer**.

| Gap (legoESM vs Veros ACC) | Active in ACC? | Class | Response | Priority |
|---|---|---|---|---|
| **Flux-form momentum advection** — Veros uses flux-form ∇·(uu); legoESM had only `vector_invariant`/`weno`. | **yes** | Missing **method** | **DONE (built + adopted).** Added `momentum_advection="flux_form"` (+ `momentum_flux_scheme` upwind/centered) to the canonical dycore (reuses `divergence_cgrid` + face-interp); verified on truth tiers F1–F8 (bit-identity, conservation machine-eps, zero-velocity, uniform-flow, differentiability, gyre stability). Adopted `flux_form`/`centered` in `build_acc_model_config`. **Oracle (developed-flow) result: du_adv corr 0.584→0.654, dv_adv −0.336→0.715 vs Veros** — flux-form matches Veros's flux-form far better (apples-to-apples validated). See `flux_form_*` docs. | **DONE** |
| **EKE closure** — Veros eddy-kinetic-energy parameterization (`enable_eke=True`, `eke_c_k=0.4`, superbee advection + isopycnal diffusion); no legoESM equivalent. | **yes** | Missing **method** | **DONE (built + oracle-verified + ADOPTED).** (Adoption was unblocked by the `eke_len` Rhines variant — see the next row.) Added the Eden-Greatbatch prognostic-EKE closure as a canonical block: `ocean/physics/lateral_mixing/eke.py` (2-D depth-integrated `E`; `kappa_GM=c_k·L·√E`; semi-implicit positivity; conservative transport) + `GMRediConfig.eke` + step coupling + restart. Truth tiers E1–E8 green (budget closure machine-eps, positivity-by-construction, differentiability, idealized-channel bounded spin-up, regression lock). **Oracle (E9, developed-flow Veros ACC, enable_eke): legoESM's `eke_kappa_gm` reproduces Veros's `K_gm` to MACHINE PRECISION (corr=1.0, max_rel_err=0 on 19560 wet cells) when fed Veros's own `eke`+`eke_len`** — the closure form + ACC params (c_k=0.4, c_eps=0.5, k_max=1e4, lmin=100, superbee, isopycnal diffusion, all verified against `veros/setups/acc/acc.py`) match exactly. See the EKE build docs + `eke_build_progress.md`. | **DONE** |
| **EKE mixing length `eke_len`** — Veros: `max(lmin, min(eke_cross·L_rossby, eke_crhin·L_rhines))` with the eddy-energy-dependent **Rhines scale** `L_rhines=√(√E/β)` + `eke_cross=2` (developed ACC: ~8 km mean, 47 km max). legoESM's EKE `L` reuses the Visbeck first-baroclinic Rossby radius (~200 km, saturated) — ~25× larger. | **yes** (part of EKE) | Missing **variant** | **DONE (built + adopted).** Added the Rhines-limited mixing length as the selectable `EKEConfig.mixing_length_scheme="rhines"` (default `"rossby"` keeps the Visbeck length bit-identical): `eke_rhines_length=√(√E/β)`, `eke_deformation_radius=min(c₁/|f|, √(c₁/2β))` with `c₁=∫N dz/π` (reuses the shared `_eady_growth_and_length` N — no duplicate numerics), `eke_len_composite=max(lmin, min(eke_cross·L_def, eke_crhin·L_rhines))`; `β=2Ω cosφ/R` analytic from config constants. Truth tiers L1–L3 green (formula/dispatch unit tests, default bit-identical incl. E8 + decomposition goldens, differentiability finite+nonzero, dry-column NaN-safe). **Oracle (L4, developed-flow Veros ACC): fed Veros's own N²/f/β/eke, legoESM reproduces Veros `L_rossby` (5.8e-16), `L_rhines` (0.0), `eke_len` (0.0) AND `K_gm` (0.0) to MACHINE PRECISION** — Veros `eke_len` mean 8.3 km / max 47.3 km, the Rhines scale IS the limiter vs the 56.9 km mean deformation radius. ADOPTED in `ACC_GM_REDI_CONFIG` (`eke=EKEConfig(mixing_length_scheme="rhines", eke_cross=2.0, eke_crhin=1.0)`; initial `eke` field seeded so the scan-based free-run carries a constant pytree). See `eke_len_build_progress.md` + `eke_len_build_spec.md`. | **DONE** |
| **EKE isopycnal diffusion (`K_iso = K_gm`)** — Veros ACC sets `enable_eke_isopycnal_diffusion=True` so the Redi *tracer* diffusivity follows the prognostic GM coefficient (`K_iso = K_gm`, ~0.3 m²/s cold-start; `veros/core/eke.py:74-75`). legoESM holds `kappa_Redi=1000` constant — only the GM *skew* term is EKE-driven, so during a cold start legoESM `K_iso=1000 ≫` Veros `~0.3` (a ~3000× mismatch). | **yes** (part of EKE) | Missing **method** | **Build:** make `kappa_Redi` array-capable through the GM/Redi triad+centered flux builders (must stay bit-identical on the constant-kappa default — golden-locked core numerics), add an `EKEConfig` iso-diffusion flag (Veros `enable_eke_isopycnal_diffusion`), and have the step pass `kappa_redi_override = kappa_gm_override` when active. Then regenerate the rhines step golden + re-measure the ACC free-run (the 30-day transport/KE gap interpretation depends on it). Surfaced by the eke_len **L5+L6 review (Issue 1)**. | **DONE (built + adopted).** Made `kappa_Redi` array-capable through the GM/Redi triad + centered flux builders (scalar default bit-identical — decomposition + 3 EKE goldens hold; gate R1, 125 tests) via the same face-interpolation as `kappa_GM`; added `EKEConfig.isopycnal_diffusion` (Veros `enable_eke_isopycnal_diffusion`, default False); the step passes `kappa_redi_override = kappa_gm_override` when set, so K_iso = K_gm and the `(kappa_Redi-kappa_GM)` horizontal off-diagonal cancels exactly as in Veros. ADOPTED in `ACC_GM_REDI_CONFIG` (`isopycnal_diffusion=True`). **Oracle (R3): Veros's own K_iso == K_gm to machine precision (max_rel_err 0.0, mean 33 m²/s); legoESM reproduces it transitively** (K_iso = kappa_redi_override = kappa_gm_override, the prognostic kappa that L4 matched to Veros K_gm machine-exact). **Measure-first (R3, EKE+Redi-on 30-day free-run): the ACC-transport/KE gap is UNCHANGED (+70.3%/+233.2%)** — with the WHOLE EKE/GM-Redi closure now apples-to-apples, the 30-day gap is conclusively the **integrator** (forward-Euler vs Veros AB2 — see the next row) + spin-up, NOT the eddy closure. Locked by the `rhines_kiso` step golden. | **DONE** |
| **Outer time integrator** (audit gap #7) — legoESM forward-Euler / split-explicit vs Veros. **CORRECTION (2026-05-29, from Veros source): this Veros version uses Adams-Bashforth-2, NOT leapfrog+Robert-Asselin** — tracers `temp[taup1]=temp[tau]+dt_tracer·((1.5+ε)·dtemp[tau]−(0.5+ε)·dtemp[taum1])` (`thermodynamics.py`), AB2 momentum (`solve_stream.py`), separate `dt_tracer=43200`/`dt_mom=4800`, and the AB2 ε-offset (`AB_eps=0.1`) — NOT an Asselin filter — suppresses the computational mode. The audit doc's "leapfrog+AB2+Robert-Asselin" was wrong. With the WHOLE EKE/GM-Redi closure now apples-to-apples (machine-exact), the measure-first 30-day ACC gap (+70%/+233%) is conclusively dominated by THIS. | **yes** | Missing **method** | **Build the AB2 OUTER scheme** reusing `timestepping/leapfrog_ab2.py:ab2_step` (deleted 2026-10 as unwired; recover from git e86e0b86d) + the inner `tracer_time_integrator="ab2"` pattern: a τ-1 *tendency* carry, separate `dt_tracer`/`dt_mom`, and the implicit vertical mixing applied ONCE per step (NOT doubled). **A leapfrog+Robert-Asselin attempt (gates I1+I2) was built, measured + reviewed, then REVERTED (commit `d1648f8e`)** — it was (a) the wrong scheme and (b) UNSTABLE: the extract-from-FE wrapper (`X^{n+1}=X^{n-1}+2(X_FE-X^n)`) could not separate advection from diffusion, so it leapfrogged the implicit vertical mixing (doubling the backward-Euler increment, `2·Δ(dt)` vs `Δ(2dt)`) → the free-run blew up at 3 days; the 40-step unit test masked a 230× computational-mode growth. The measure-first + adversarial-review process caught the misdirected build before it shipped. | **AB2 BUILT (gate A1) — but NOT the 30-day lever.** `outer_integrator="ab2"` (extract-from-FE: AB2 the explicit FE increment `X^{n+1}=X^n+(1.5+ε)·ΔX^n−(0.5+ε)·ΔX^{n-1}`; the barotropic mode is kept from the split-explicit free-surface solve, un-AB2'd; the baroclinic momentum deviation is AB2'd). **STABLE** — 8 unit tests + a 300-step channel + the real-ACC **30-day free-run** (where the leapfrog blew up at 3 days); default `forward_euler` bit-identical (36 tests). AB2 applies the implicit-mixing increment ~1× (not the leapfrog's fatal 2×). **Measure-first: the gap is UNCHANGED (+70.3%/+233.4%, same as forward-Euler)** ⇒ the integrator SCHEME (FE vs AB2) is NOT the 30-day lever. CAVEAT (adversarial-review #2): this AB2s the TOTAL increment (incl. the once-applied implicit vertical mixing) rather than Veros's faithful explicit-AB2 + implicit-once, and uses a single dt (not `dt_tracer≠dt_mom`). Consequence: vertical-mixing stability is CONDITIONAL (threshold `dt·K_v·4/dz²_min ≲ a few`; a stiff-K_v channel blows up at ~200 steps, while forward-Euler — implicit-once — is unconditionally stable). Safe for mild mixing (ACC's TKE: `dt·λ≈-0.2`); `step` GUARDS against `outer_integrator="ab2"` + convective adjustment, and a stiff-K_v test locks the regression. Full fidelity (explicit/implicit split + dt split) is the documented refinement. **NEXT lever (NOT the integrator scheme): the energetics.** A **1-year free-run** shows the gap is PERSISTENT, not spin-up — transport +58% (was +70% @30d), total KE **+443%** (was +233%), max|u| **+124%** (was −14%): legoESM's ACC becomes ~5× more energetic than Veros while the thermal mean state matches (vol-mean T +0.5%). The EKE/GM eddy damping is still cold at 1 year (multi-year equilibration) in BOTH models, so it isn't yet arresting the jet. Candidates: (a) the **barotropic formulation** (Veros rigid-lid/streamfunction `solve_stream.py` vs legoESM split-explicit free surface — but matching the rigid-lid would REGRESS legoESM's modern free surface, so likely an accept-the-delta with justification, not a must-build); (b) an **eddy/momentum dissipation deficit** (legoESM accumulating KE faster — bottom drag / lateral viscosity / barotropic-mode damping); (c) the `dt_tracer≠dt_mom` split. A barotropic-vs-baroclinic KE decomposition (where the excess energy lives) is the bounded next diagnostic. **DISSIPATION AUDIT (2026-05-29) — found a bottom-drag mis-mapping AND, by fixing it, isolated the genuine deeper difference: the BAROTROPIC FORMULATION.** (1) The recipe copied Veros's `r_bot=1e-5` into `bottom_drag_r=1e-5`, but Veros applies it as a RATE on the bottom cell (`du/dt=-r_bot·u`, NO ÷dz; `friction.py:284-289`) while legoESM uses the stress form `du/dt=-r·u/h_bot` — so the identical value made legoESM's drag ~`h_bot`≈276× TOO WEAK. The FAITHFUL mapping is `R_BOT=r_bot·h_bot≈2.76e-3` (adopted). (2) BUT the faithful drag does NOT reconcile the transport — it OVER-damps it: @30d it looked like a clean fix (KE +233%→+27%, transport +70%→−22%), but **@1yr the faithful drag gives transport −83% (18 vs 110 Sv) / KE −29%**, vs the mis-mapped `r=1e-5`'s +58% / +443%. So legoESM's barotropic transport OVER-RESPONDS to bottom drag vs Veros (same bottom-cell rate → 18 Sv legoESM vs 110 Sv Veros) ⇒ the genuine deeper difference is the **barotropic formulation** (legoESM split-explicit free surface vs Veros rigid-lid/streamfunction); the old `r=1e-5` had masked it via COMPENSATING errors (under-drag ≈ cancelled the over-responsive barotropic). We keep the FAITHFUL drag (match Veros's scheme, not tune to the metric); the large transport residual is the barotropic-formulation delta. **The barotropic formulation (free-surface vs rigid-lid) is the clearly-isolated remaining ACC gap.** **RIGID-LID BUILT + VALIDATED (2026-05-29) — hypothesis CONFIRMED.** A full general rigid-lid streamfunction solver was added as `barotropic_solver="rigid_lid"` (faithful port of Veros `solve_stream.py`: the weighted vertex Laplacian `∇·((1/H)∇)ψ = curl_vertex(velocity_recovery(ψ))` reusing the audited C-grid stencils; AD-safe symmetric-CG elliptic solve (on the SPD form `−A_vertex·L`, Jacobi-preconditioned — adversarial review caught a BiCG-STAB adjoint-VJP NaN that broke `jax.grad`, and CG's symmetric `custom_linear_solve` adjoint fixed it; gradient now matches finite-difference); the multiply-connected island machinery — flood-fill, `psin` basis, `line_psin` coupling — with the channel-transport line integral via discrete Stokes; AB2 ψ-stepping; Coriolis added to the streamfunction RHS). **1-year ACC: transport O(100 Sv) — 78–115 Sv across solver round-off, chaotically variable at 1 yr (see below) — versus the free-surface's collapsed 18 Sv (−83%) at the same faithful bottom drag** (Veros 110.6 Sv). The rigid lid RESTORES a Veros-magnitude ACC, *confirming* that the barotropic formulation (free-surface over-responding to bottom drag) was the dominant control on the ACC transport. NB the precise value is NOT pinned: the two solvers (the original BiCG-STAB and the shipped CG, see below) agree bit-for-bit through the 30-day spinup (both 150.6 Sv) and diverge only as turbulence develops — the flow is over-energetic (KE ≈19× Veros) so instantaneous transport varies chaotically; a tight quantitative match needs the dissipation fix below + a time-mean transport, not a 1-yr snapshot. SECONDARY (KE excess — INVESTIGATED 2026-05-29, prompted by "are the dissipation schemes actually matched?"). The rigid-lid KE looked ≈19× Veros at 1 yr. Audit: the dissipation **coefficients MATCH** (A_h=2.2e5 harmonic cos¹, centered advection, B_h=0, bottom drag, GM/Redi, TKE). The "19×" turned out to be (a) PARTLY a **premature 1-yr comparison** — Veros's KE *also* climbs all year (3.1e15@30d → 8.6e15@90d → 1.4e16@180d → 2.1e16@365d, never plateauing; both models are pre-equilibrium under the cold prognostic-EKE/GM sink), so the equilibrium gap is overstated; and (b) a GENUINE but milder over-energisation — legoESM is 4.6×@30d → 8.5×@90d → 19×@365d, i.e. more energetic *early* (before eddies dominate) and *widening*, tracking the ACC **spin-up overshoot** (legoESM hits 150 Sv by day 30 vs Veros ~35; the developing eddies then amplify the gap). It is **NOT a numerics bug**: the operator has no checkerboard nullmode (cb has 127× larger response than a smooth mode); the momentum-advection scheme is not the source (flux-form and the energy-conserving vector-invariant grow identically); and free-decay tests show the stratification FLATTENS as KE grows (⇒ physical baroclinic APE→KE, not a spurious source). My earlier "missing barotropic dissipation / add a damper" was WRONG (tune-to-metric; the free surface's lower KE was its *numerical* substep filter + a weak 18 Sv ACC, not validated dissipation). **ENERGY BUDGET (both models, 2026-05-29) — root cause localized.** Built a KE + APE(=PE−RPE, sorted, rigorous) + wind-power diagnostic and ran legoESM rigid-lid vs bridged Veros at 30/90/180 d. (1) **Wind RULED OUT** — P_wind is comparable, Veros even higher (1.65e11 vs 1.11e11 W @180d); legoESM is more energetic *despite less* wind (corrects the "spin-up→more wind" guess). (2) Energy is **buoyancy-driven, not wind** (d(KE+APE)/dt ≈ 2.3e12 W ≫ P_wind ≈ 1.1e11 W). (3) Energy is **APE-dominated** (APE/KE ≈ 30–100) and **legoESM holds ≈5× more APE than Veros** (2.0e19 vs 4.1e18 J @180 d). **The restoring is NOT the differentiator** (verified, user-prompted): the restoring *scheme* is identical and its measured APE generation G is comparable — Veros's is even *higher* (5.95e11 vs 2.87e11 W @180 d) — yet Veros holds 5× *less* APE. So the differentiator is the **APE drainage, not the generation**: Veros drains its APE effectively, legoESM doesn't. ⇒ legoESM runs hot because its **APE isn't drained**: the sink is the **GM eddy-induced isopycnal flattening**, weak under the *cold-started prognostic EKE* (the mean front is bounded by the same T\* in both — the 5× is *eddy* APE governed by the GM sink), amplified by legoESM's stronger ACC. (Caveat: the *absolute* budget doesn't fully close — legoESM's d(KE+APE)/dt exceeds wind + the surface-only G probe; the *comparison* is robust, full closure needs the explicit conversion + dissipation terms.) **ADIABATIC CONSERVATION TESTS (closed-budget certainty).** (A) Uniform-T/S adiabatic core (no forcing/dissipation/stratification): **KE is conserved** for both advection schemes (~92%, flat) — no spurious momentum/advection source (the vector-invariant advection is a *tested* Arakawa-Lamb-81 energy+enstrophy-conserving scheme). (B) **RPE flat to 6 digits** adiabatically — no spurious mixing. (C) the adiabatic KE growth is ~4% of the forced KE growth ⇒ **the KE excess is ~96% physical, not a spurious KE source.** ⇒ **HIGH-CONFIDENCE: the KE excess is the GM/EKE eddy-sink spin-up regime (physics), not a numerical artifact.** ONE bounded loose end: the *stratified* adiabatic run doesn't conserve **KE+PE** (PE grows; in-situ pressure doesn't fix it) — *either* a real **Adcroft-PGF energy non-conservation** (the PGF is built for partial-cell accuracy, not designed/tested EC — a *general* dycore property, same PGF in free-surface runs) *or* my continuous PE ≠ the discrete-PGF-consistent PE. Minor for KE; closing it needs the discrete-Adcroft-consistent APE — a focused dycore follow-up, decoupled from the ACC result. **ZONAL MOMENTUM BUDGET (both models, robust — clean state integrals, no APE/PGF issues).** Sanity: wind input 3.16e12 N identical (flow-independent), Coriolis ~0 (mass conservation). (1) **Flat-bottom Munk balance** in both: bottom drag is only ~10–30% of the wind; ~70% is removed by the **lateral wall viscous stress** (the large Munk A_h=2.2e5 absorbs the wind at the side walls — no form stress on a flat bottom). (2) **Vertical structure differs:** legoESM's ACC is **more barotropic** (u_bot/u_surf ≈ 0.3–0.5) vs Veros's **surface-intensified** (≈ 0.13–0.22) — legoESM's wind momentum penetrates deeper ⇒ higher bottom velocity ⇒ more bottom drag + more total momentum. This is the **same story as the energy budget from an independent diagnostic**: legoESM's more-energetic eddies (weak GM under cold EKE) transfer wind momentum downward (interfacial form stress) ⇒ a more barotropic ACC. Distinct from the barotropic *formulation* (the rigid-lid matched the transport *magnitude*); this is the vertical *distribution*, set by the eddy field. An eddy-parameterisation spin-up/equilibration regime — NOT wind, NOT numerics (schemes match, physical APE→KE). The diagnostic is a script; promote to a tested `budgets.py` module + add the explicit APE→KE conversion term for a permanent closed budget if wanted. 13 direct unit tests; full free-surface path bit-unchanged (default solver). | **RIGID-LID BUILT + VALIDATED: restores an O(100 Sv) ACC (free-surface collapsed to 18 Sv) ⇒ barotropic-formulation hypothesis CONFIRMED; gradient-NaN fixed (symmetric CG, adversarial review). KE excess INVESTIGATED: dissipation schemes MATCH; the "19×@1yr" is partly a premature comparison (Veros KE also climbs all year, both pre-equilibrium under cold GM) + a genuine milder over-energisation (4.6×@30d→19×@365d). ENERGY BUDGET (both models) localized the root: wind RULED OUT (Veros's P_wind is higher); energy is buoyancy-driven (restoring→APE); legoESM builds ≈5× more APE ⇒ its APE isn't drained fast enough = the GM eddy-flattening sink set by the cold prognostic EKE (front oversteepens), amplified by the stronger ACC. NOT a numerics bug, NOT missing dissipation, NOT the wind. Transport result stands.** |
| **Veros isopycnal/GM-Redi discretization** — we have GM/Redi (`slope_scheme` = triads/centered); Veros's isoneutral scheme differs (T_iso corr 0.17). | yes (have variant) | Missing **variant** | **DONE (root-caused + built, `31313850`): Veros-faithful NEUTRAL-density isoneutral slopes.** legoESM built slopes from IN-SITU density, whose vertical gradient carries compressibility (∂ρ/∂p·∂p/∂z ≈ 4.5e-3 kg/m⁴) ⇒ \|∂_zρ\| ~4× too steep ⇒ slopes 4× too flat ⇒ K_33 ∝ S² collapsed 10–25× interior / 10⁴× near-surface. Veros uses locally-referenced NEUTRAL gradients `drdT·∇T + drdS·∇S` (`isoneutral.py:40-41`). Added `eos.eos_density_derivatives` (jax.grad of any EOS at fixed p) + `GMRediConfig.slope_density="in_situ"\|"neutral"` (default in_situ BIT-IDENTICAL; ACC recipe opts in). KEY: Veros never CLIPS the slope (DM95 taper only) — the neutral path is unclipped (clipping would saturate the taper and inflate K_33 ~10×). Truth tiers hold (bit-identity Δ=0, Redi cancellation 1e-17, conservation 1e-17, AD=FD). **T_iso 0.17→0.37 (Veros-K_iso-fed = the meaningful metric); K_33 now ~2× of Veros vs 20–100× too small.** Same in-situ-vs-locally-referenced class as the convection-N² fix (`dfc8318b`). NB the constant-kappa frozen-state probe reads 0.077 — a KAPPA confound (constant 1000 ≠ Veros's variable K_iso), not a regression. **kr-sum refinement DONE (`37112b46`, 2026-06-09):** every triad now pairs its reference cell's EOS derivatives in numerator AND denominator (W-face B-triads + u/v-face above-faces; Veros kr/ki loops, term-for-term 0.0). Measured inert on the uniform ACC (T_iso 0.3665→0.3666, as predicted — bites at variable metrics/tripolar) but it FIXED a real adiabaticity defect: per-triad Redi cancellation 3.5e-11 → 8.9e-26 (the above-face triads had mixed cells between numerator and denominator). Remaining (low): exact per-corner metric factors (dxu/dxt·cost weights — inert on uniform grids); the `min(x,−eps)` vs Veros `min(0,x)−eps` floor = equivalent regularization (~1e-6 interior), documented. | **DONE** |
| **IDEMIX** — internal-wave energy/mixing. | **no** (`enable_idemix=False`) | n/a | **Defer** — Veros runs ACC without it, so neither model needs it for apples-to-apples. | — |
| **Implicit vs explicit vertical mixing** (`du_mix` corr ~0). | implicit | — (config exists) | **Accept** — both paths exist; recipe selects implicit. Not a gap. | — |
| **Density / EOS / hydrostatic pressure** (corr 1.0). | yes | — | **Accept** — already matches (shared discretization). | — |

### Veros ACC ledger — closing rows (2026-06-03..09)

The dycore-numerics-matching program is COMPLETE: every Veros dycore option is
imported as a config-selectable, default-off-bit-identical option. Rows below
close out the items the 2026-05-29 table left open (each physics-validator
signed off; probe evidence under `.physics-validator/`).

| Item | Result |
|---|---|
| **Flux-divergence lateral viscosity** (`8075f152`) | Veros's componentwise `∇·(A_h∇u)` (`harmonic_friction`) added as `lateral_viscosity_operator="flux_divergence"` (default `vector_laplacian` bit-identical; ACC opts in), verified term-for-term MACHINE-0 vs Veros + operator-matched K_diss_h. The ralph loop caught a real `1/cos(φ)` bug in the v-flux divergence the build's own tests missed. **Measure: friction FORM is definitively NOT the barotropic-split lever** (flux_div u_bot/u_surf 0.489 / MOC 223 Sv vs vector 0.488 / 218; Veros 0.21 / 26). |
| **Centered 2nd-order tracer advection** (`0dc649ae`) | Veros's `_adv_flux_2nd` as `tracer_advection="centered"` (reuses the flux-form divergence machinery); STABLE in ACC; recipe opts in. Supersedes the earlier accept-the-delta TVD note; measured: not a climate lever (less-diffusive advection runs the abyss WARMER). |
| **#44 faithful AB2: explicit-AB2 + implicit-vmix-once** (`51755c40`) | The AB2 caveat row above is CLOSED: AB2 now extrapolates only the EXPLICIT increment and applies implicit vertical mixing ONCE after (Veros `thermodynamics.py`/`solve_stream.py`) ⇒ unconditionally stable in the vertical, convection guard REMOVED. FE path bit-identical (7e-15). |
| **#44 dt_mom≠dt_tracer asynchronous stepping** (`8c878228`) | `dt_mom_ratio` (default 1.0 bit-identical; ACC=9 = 43200/4800). dt_mom drives momentum+barotropic+implicit friction; dt_tracer drives tracers. Requires rigid_lid (free surface leaks O((dt_tracer−dt_mom)·∂h/∂t) tracer mass — validator-measured 1.3e-6 vs rigid-lid 4.6e-16). 11/11 validator probes PASS. NB the handoff's "momentum subcycle" framing was WRONG — Veros runs momentum ONCE per dt_tracer clock step (`veros.py:306`). |
| **Rung-6 oracle tendency-match (integrator)** | Single faithful step on a bridged 20-day developed Veros state (AB2 history seeded, rigid lid, bridge round-trip 0.0): **the integrator arithmetic adds NO error** beyond tier-2 operator + placement deltas. Surfaced the surface-forcing placement gap (next row). Momentum at this seeding fidelity: see the momentum-fair row. |
| **Implicit surface TRACER forcing** (`1e8ec9e1`) | Rung 6 localized a 1.6× over-application: legoESM applied T*/q_net/shortwave in the EXPLICIT tendency (⇒ AB2-extrapolated ×1.6); Veros adds `forc_temp_surface` to the backward-Euler vmix RHS at weight 1.0 (`thermodynamics.py:276`). Added `surface_forcing_implicit` (default off bit-identical; ACC opts in): the withheld surface tracer forcing enters the implicit solve input. **Realized T L2 3.85→1.69** (restoring-off floor 1.23; residual = restoring-source/state diff, not placement). Wind stays explicit (Veros AB2s it too). |
| **Momentum total fair-comparison** (`.physics-validator/momentum_fair/`) | The momentum DYCORE MATCHES Veros operator-by-operator: explicit operators corr 0.95/0.91 (u/v, after fixing a Veros stale-time-level capture), implicit vertical-friction operator corr 0.99/0.96 vs `du_mix`, Coriolis 0.96/0.98, flux-form advection, PGF. The tier-2 `du_adv`/`du_mix` rows are COMPARISON ARTIFACTS (decomposition / implicit-not-explicit), as suspected. The realized-increment residual is the friction-PLACEMENT form — closed by the additive-placement option (next row). Rigid-lid psi seeded gauge-clean from the bridged transport (recover-vs-transport corr 0.997); the realized increment is ~90% baroclinic. |
| **Layer-1 completeness audit + closure-energy campaign** (2026-06-09/10, user-driven: "chaos explains weather, not climate") | 3-agent inventory (Veros ACC active terms / legoESM faithful stack / provenance+probe-coverage) found the closure ENERGY CYCLE unmatched: TKE was DIAGNOSTIC-only (quasi-steady Mode B) vs Veros's prognostic carry; TKE missing eke_diss_iw/K_diss_bot/P_diss_adv/P_diss_nonlin sources; EKE lateral diffusion constant-k_iso vs Veros 0.5·max(500,ΣK_gm); EKE vertical advection deferred; EKE source missing −P_diss_iso. **Veros-side ablations (the decisive cheap experiments, `.physics-validator/veros_ablation/`): ALL of these are climate-INERT in Veros over 10 yr** (H1 ratios 0.995–1.005; H2+H3+H6 combined ≈1.000). **Forcing fields EXONERATED at machine precision** (realized wind acceleration ratio 1.000000; T* piston equivalence 5e-23; u*³ form identical). **H5 refresh: the residual is REAL on the full faithful stack** (10-yr vs 10-yr, same metrics): KE 5.4×, EKE 19.8×, transport 1.52×, u_bot/u_surf 2.42×, abyss +2.8 °C. |
| **Prognostic TKE** (`2bfa841a`, 2026-06-10) | Veros-faithful prognostic TKE carry (state.tke, one backward-Euler step/model step at dt_tke=dt_mom) + energy-recycling sources (eke_diss_iw exact to `eke.py:163`; K_diss_bot via the shared K_diss_h mapping from legoESM's own drag) + ACC recipe opt-in, wired through forward-Euler, AB2, and AB2+additive paths. Validator SHIP (default-off byte-identical on 4 paths; exactly-once advance; kappa-timing e_new-vs-tke[tau] equilibrium-equivalent at machine precision, documented). **ATTRIBUTION (10-yr, GPU): moves the residual the right way but does not close it** — transport 1.52→1.35, KE 5.41→4.51, EKE 19.8→17.4, u_surf 2.06→1.88, u_bot/u_surf 2.42→2.31 (vs Veros). P_diss_adv/P_diss_nonlin deferred (documented). |
| **Residual status after the campaign** | Every per-term comparison is matched (machine on operators, 0.9+ on totals), all closure-structure holes are built or measured-inert, forcing is exact — and the 10-yr equilibrium still diverges ~4.5× in KE. REMAINING never-isolated candidates (ranked): (1) the rigid-lid ψ-solve / barotropic-coupling OUTPUTS on identical RHS (never snapshot-compared; includes Veros's dpsi extrapolation + vertical-mean removal/re-add sequence); (2) spherical METRIC/curvature terms in the momentum budget (Veros `momentum.py:43-73` tantr terms — coriolis tier-2 corr 0.96/0.98 bounds but does not isolate them); (3) isolated vertical momentum advection; (4) realized bottom-drag tendency; (5) the T_iso 0.37 pattern (APE drainage). If those also clear, the eddy-mean-equilibration attribution finally stands on closed ground. |
| **Additive momentum vertical-friction placement** (2026-06-09, the LAST per-tendency placement fix) | Veros adds `du_mix` (backward-Euler friction increment evaluated on u^n, `core/friction.py`) ADDITIVELY alongside the AB2-extrapolated explicit tendency at weight 1.0 (`solve_stream.py:173-196`, verified against source); legoESM applied friction SEQUENTIALLY (backward-Euler on the AB2 state). At equilibrium AB2(du)≈−du_mix, so the O(dt²·A_v) placement delta dominates the realized increment. Added `momentum_friction_additive` (default off; requires ab2 + implicit vmix, rejected otherwise; tracers stay sequential = Veros's own tracer placement). **Oracle (native_probe.py): realized baroclinic u-increment corr 0.11→0.28, L2 ratio 4.8→2.3** toward the reconstruction ceiling 0.44/1.57 (the remaining gap is the −0.6·du_p AB2-history term, dominated by bridge-seeding fidelity, not placement; v stays noisy 0.03–0.19 in all variants — the handoff's momentum-seeding caveat). Physics-validator SHIP 7/7: default-off byte-identical to pristine HEAD (worktree diff = 0.0), increment depth-mean ≤1e-10·scale (barotropic untouched), heat drift 1.8e-15 both modes, dt_mom scaling exact (8.9996/9), stable at A_v=10 m²/s, AD=FD 8.8e-11. 10-day faithful-stack smoke stable (ACC 13.45 Sv, max\|u\|=0.048). |

### Cross-oracle naming (concept registry seed)

Synonyms found in the 2026-05-29 survey, now in `ocean/fidelity/concept_registry.py`:
Veros `kappaM`/`kappaH` ≈ legoESM `A_v`/`K_v`; `r_bot` = `bottom_drag_r`; `K_gm_0` = `kappa_GM`,
`K_iso_0` = `kappa_Redi` (constant fallback ONLY — **stale for EKE-on ACC**: with
`enable_eke_isopycnal_diffusion=True` Veros ignores `K_iso_0` and sets `K_iso = K_gm`, the prognostic
kappa — see the EKE-isopycnal-diffusion ledger row; legoESM does not yet reproduce this);
`nz` = `n_levels`. Internal debt tracked there too (`nlev` vs
`n_levels`, `T_sfc` vs `T_surface`, `c_sw` vs `c_ocean`/`c_p`, `bottom_drag_coeff`, the
`tau_relax` days-vs-seconds and `C_water`/`c_water` unit hazards). **Correction (2026-05-29):**
DINO's `c_p=3991.86` / `rho_0=1026.0` (and NeverWorld2-lite's inherited copies) are **intentional
paper-fidelity values** (Kamm et al. 2025), correctly carried as config-field defaults — **not**
constant-discipline debt; they must NOT be "fixed" toward `legoesm.constants` (that would break
paper replication). The fidelity-side Veros constants in `veros_configs/eady_uniform.py` now
reference `VEROS_CONSTANTS_CONFIG` (value-identical, done). Remaining low-priority items
(`isomip_plus` `g_val=9.81`/`rho_sw=1028`, `held_larichev` `rho_0=1024`) have unclear intent →
audit before touching, do not silently change.

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
## Oracle-Recipe Fidelity (ocean; see docs/ocean/fidelity/oracle_recipe_strategy.md)
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

# Arctic / High-Latitude Carbon Productivity Rescue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the boreal/tundra carbon death spiral (`needleleaf_deciduous_boreal` dead, Arctic SOC 0.61 vs obs 20–100+ kgC/m²) by adding two opt-in, default-off, PFT-scoped mechanisms to DifferLand prognostic carbon, lifting high-latitude pools toward observed without regressing the working temperate/tropical equilibria.

**Architecture:** Two smooth, differentiable multipliers on the maintenance-respiration and GPP fluxes in `carbon_cycle.py`, each behind a static Python feature-gate bool (default `False` → byte-identical). (1) `f_nsc(C_lab, live-biomass)` throttles **all** `R_maint` under labile-reserve depletion; (2) `d(T)` sigmoid zeros **foliar** GPP and `R_maint` for cold-deciduous PFTs below a freeze threshold. New `CarbonConfig` fields + `__param_spec__`; `is_cold_deciduous` classifier; archetype grouping + equilibrium-cache-version threading; `run_lmip` CLI. Validated column-first (7 pixels) then global rebuild, flags-on-vs-off.

**Tech Stack:** JAX (jnp, `jax.nn.sigmoid`), Equinox-free NamedTuple config, pytest. Compute on Ginsburg compute nodes via `sbatch` (login-node policy) with the jn2808 conda python.

## Global Constraints

- **Differentiable:** gates use `smoothstep`/`jax.nn.sigmoid`; **no** `jnp.where`/Python control flow on **traced** values. Feature flags are **static Python bools** → plain `if` (the repo's feature-gating exception).
- **Opt-in / byte-identical:** every new mechanism defaults **off**; with flags off the numerics are bit-identical to the pre-change code (`f_nsc ≡ 1.0`, `d ≡ 1.0` literals).
- **Carbon-conserving:** gating a respiration flux retains the un-respired C in-pool; `R_maint_day` (gated) is the single value feeding `R_auto`/`NPP`/the deficit cascade/NEE. `Σ ΔC = −NEE·dt` to machine precision.
- **Sign convention (carbon, positive-out):** `R_maint` is a LOSS (plant→atmosphere). Both gates **reduce** the loss; a test asserts gated `R_maint ≤` ungated.
- **Constants:** no literal `273.15` — use `legoesm.constants.T_freeze` (CI ratchet `test_no_hardcoded_constants`).
- **Param hygiene:** every new `:float` `CarbonConfig` field gets a `__param_spec__` entry (`params` if tunable, `excluded` with a reason if numerics); bools are not spec-eligible. New empirical coeffs live in config fields, not function bodies (`test_no_inline_physics_coeffs`).
- **PFT scope for M2:** `is_cold_deciduous` = `needleleaf_deciduous_boreal`, `c3_arctic_grass`, `broadleaf_deciduous_boreal_shrub`. The already-healthy `broadleaf_deciduous_boreal` is deliberately excluded.
- **CLI:** each new user-tunable `CarbonConfig` field gets a `run_lmip.py` flag + round-trip test; float tunables also reachable via `--params land.carbon.<field>`.
- **Compute:** never on the login node. `sbatch` on a compute node; `PYTHONPATH=$(ls -d packages/*/ | sed 's:/$::' | tr '\n' ':')`, `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu`, python `/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python`.
- **Worktree/branch:** `/burg-archive/glab/users/pg2328/lc_rebase`, branch `land/arctic-carbon-productivity`.

---

### Task 1: `is_cold_deciduous` PFT classifier

**Files:**
- Modify: `packages/land/legoesm/land/surface_params.py:303-313` (add after `is_evergreen`)
- Test: `tests/land/unit/test_surface_params.py`

**Interfaces:**
- Produces: `is_cold_deciduous(pft_name: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/land/unit/test_surface_params.py
def test_is_cold_deciduous_selects_larch_arctic_grass_and_boreal_shrub():
    from legoesm.land.surface_params import is_cold_deciduous
    assert is_cold_deciduous("needleleaf_deciduous_boreal")
    assert is_cold_deciduous("c3_arctic_grass")
    assert is_cold_deciduous("broadleaf_deciduous_boreal_shrub")
    # NOT the already-healthy boreal broadleaf tree, nor evergreen/temperate/tropical:
    assert not is_cold_deciduous("broadleaf_deciduous_boreal")
    assert not is_cold_deciduous("needleleaf_evergreen_boreal")
    assert not is_cold_deciduous("broadleaf_deciduous_temperate")
    assert not is_cold_deciduous("c3_grass")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.../python -m pytest tests/land/unit/test_surface_params.py::test_is_cold_deciduous_selects_larch_arctic_grass_and_boreal_shrub -v`
Expected: FAIL (ImportError: cannot import name 'is_cold_deciduous').

- [ ] **Step 3: Write minimal implementation**

```python
# surface_params.py, after is_evergreen (line ~313)
_COLD_DECIDUOUS_PFTS = frozenset({
    "needleleaf_deciduous_boreal",
    "c3_arctic_grass",
    "broadleaf_deciduous_boreal_shrub",
})


def is_cold_deciduous(pft_name: str) -> bool:
    """True for cold-deciduous / winter-dormant high-latitude PFTs.

    These PFTs (larch ``needleleaf_deciduous_boreal``, arctic graminoids
    ``c3_arctic_grass``, ``broadleaf_deciduous_boreal_shrub``) shed or
    metabolically shut down their foliage over the frozen season, so both
    canopy GPP and foliar maintenance respiration should stop when frozen
    (``carbon.carbon_cycle`` Mechanism 2, gated by
    ``CarbonConfig.cold_deciduous`` + ``cold_deciduous_dormancy``).  An
    EXACT-name membership set, not a substring match: the already-productive
    ``broadleaf_deciduous_boreal`` tree is intentionally excluded (a substring
    on "deciduous_boreal" would wrongly include it).
    """
    return pft_name in _COLD_DECIDUOUS_PFTS
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /burg-archive/glab/users/pg2328/lc_rebase
git add packages/land/legoesm/land/surface_params.py tests/land/unit/test_surface_params.py
git commit -m "feat(land): is_cold_deciduous PFT classifier (larch/arctic-grass/boreal-shrub)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3Xx8yToULd6EGpdBEpGn3"
```

---

### Task 2: NSC-gated maintenance-respiration factor + config fields

**Files:**
- Modify: `packages/land/legoesm/land/carbon/config.py` (`CarbonConfig` fields + `__param_spec__`)
- Modify: `packages/land/legoesm/land/carbon/carbon_cycle.py` (helper `_nsc_respiration_factor`, near `_freeze_modifier` ~267)
- Test: `tests/land/unit/test_carbon_cycle.py`

**Interfaces:**
- Consumes: `CarbonConfig.nsc_ref_labile_frac`, `.r_maint_floor_frac`, `.nsc_gated_respiration`
- Produces: `_nsc_respiration_factor(C_lab, C_fol, C_root, C_wood, config) -> jnp.ndarray` ∈ [floor, 1]

- [ ] **Step 1: Add the config fields** (`config.py`, in `CarbonConfig`, after the `woody` block ~128)

```python
    # --- High-latitude productivity rescue (opt-in, default-off, PFT-scoped) ---
    # Mechanism 1 — NSC/substrate-gated maintenance respiration (Atkin &
    # Tjoelker 2003): throttle R_maint as the labile reserve depletes relative
    # to live biomass, so a dormant plant downregulates respiration instead of
    # cannibalising structural pools to death.  Default OFF -> byte-identical.
    nsc_gated_respiration: bool = False
    nsc_ref_labile_frac: float = 0.02   # C_lab_ref = this*(C_fol+C_root+C_wood) [-]
    r_maint_floor_frac: float = 0.10    # f_nsc floor as C_lab -> 0 [-]
```

- [ ] **Step 2: Add the `__param_spec__` entries** (`config.py`, in `CarbonConfig["params"]`)

```python
            "nsc_ref_labile_frac": {"units": "1", "bounds": (0.005, 0.1), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "labile/NSC as a fraction of live biomass at which maintenance respiration is unthrottled; respiratory downregulation under substrate limitation (Atkin & Tjoelker 2003)", "shape": None},
            "r_maint_floor_frac": {"units": "1", "bounds": (0.0, 0.3), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "basal maintenance-respiration floor retained under full NSC depletion (Atkin & Tjoelker 2003)", "shape": None},
```

- [ ] **Step 3: Write the failing test**

```python
# tests/land/unit/test_carbon_cycle.py
def test_nsc_respiration_factor_throttles_and_saturates():
    import jax.numpy as jnp
    from legoesm.land.carbon.carbon_cycle import _nsc_respiration_factor
    from legoesm.land.carbon.config import CarbonConfig
    cfg = CarbonConfig(scheme="differland", nsc_gated_respiration=True,
                       nsc_ref_labile_frac=0.02, r_maint_floor_frac=0.10)
    C_fol, C_root, C_wood = 100.0, 100.0, 800.0
    ref = 0.02 * (C_fol + C_root + C_wood)      # = 20 gC/m2
    f0 = _nsc_respiration_factor(jnp.asarray(0.0), jnp.asarray(C_fol),
                                 jnp.asarray(C_root), jnp.asarray(C_wood), cfg)
    fref = _nsc_respiration_factor(jnp.asarray(ref), jnp.asarray(C_fol),
                                   jnp.asarray(C_root), jnp.asarray(C_wood), cfg)
    fhi = _nsc_respiration_factor(jnp.asarray(10 * ref), jnp.asarray(C_fol),
                                  jnp.asarray(C_root), jnp.asarray(C_wood), cfg)
    fmid = _nsc_respiration_factor(jnp.asarray(ref / 2), jnp.asarray(C_fol),
                                   jnp.asarray(C_root), jnp.asarray(C_wood), cfg)
    assert abs(float(f0) - 0.10) < 1e-6            # floor at empty reserve
    assert abs(float(fref) - 1.0) < 1e-6           # saturates at the reference
    assert abs(float(fhi) - 1.0) < 1e-6            # stays 1 above it
    assert 0.10 < float(fmid) < 1.0                # monotone between
    # winter-leafless robustness: C_fol=0, reserve still gates off root+wood
    fwin = _nsc_respiration_factor(jnp.asarray(0.0), jnp.asarray(0.0),
                                   jnp.asarray(C_root), jnp.asarray(C_wood), cfg)
    assert abs(float(fwin) - 0.10) < 1e-6
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.../python -m pytest tests/land/unit/test_carbon_cycle.py::test_nsc_respiration_factor_throttles_and_saturates -v`
Expected: FAIL (cannot import `_nsc_respiration_factor`).

- [ ] **Step 5: Write the helper** (`carbon_cycle.py`, after `_freeze_modifier` ~ line 305)

```python
def _nsc_respiration_factor(
    C_lab: jnp.ndarray,
    C_fol: jnp.ndarray,
    C_root: jnp.ndarray,
    C_wood: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Substrate (NSC) limitation of maintenance respiration, ``f_nsc`` in
    ``[r_maint_floor_frac, 1]``.

    Respiratory downregulation under carbon starvation (Atkin & Tjoelker 2003):
    when the labile / non-structural-carbon reserve ``C_lab`` is depleted
    relative to a fraction of LIVE BIOMASS, maintenance respiration throttles
    toward a small basal floor instead of demanding the full biomass-proportional
    amount and cannibalising structural pools to death::

        C_lab_ref = nsc_ref_labile_frac * (C_fol + C_root + C_wood)
        f_nsc = r_maint_floor_frac
                + (1 - r_maint_floor_frac) * smoothstep(C_lab / C_lab_ref)

    ``smoothstep`` is the C1 Hermite ``x^2 (3 - 2x)`` on a ``[0, 1]``-clamped
    argument -> differentiable; ``f_nsc -> 1`` for ample reserve
    (``C_lab >= C_lab_ref``; healthy plants unaffected -> temperate/tropical
    no-regression) and ``-> r_maint_floor_frac`` as ``C_lab -> 0``.

    The reference scales with LIVE BIOMASS, not foliage: a winter-leafless plant
    has ``C_fol -> 0``, so a foliage-only reference would collapse and leave the
    root+wood winter drain ungated (the exact death-spiral case).  ``C_root`` /
    ``C_wood`` persist through the leafless season and keep the reference finite.
    A ``1e-10`` epsilon guards ``ref`` so a fully bare column yields the floor,
    not ``0/0``.
    """
    ref = config.nsc_ref_labile_frac * (C_fol + C_root + C_wood)
    x = jnp.clip(C_lab / jnp.maximum(ref, 1e-10), 0.0, 1.0)
    smoothstep = x * x * (3.0 - 2.0 * x)
    floor = config.r_maint_floor_frac
    return floor + (1.0 - floor) * smoothstep
```

- [ ] **Step 6: Run test to verify it passes** — Run as Step 4. Expected: PASS.

- [ ] **Step 7: Guard the param-hygiene ratchets**

Run: `.../python -m pytest tests/test_param_specs.py tests/test_no_inline_physics_coeffs.py -q`
Expected: PASS (new floats specced; `0.02`/`0.10`/`3.0`/`2.0` are exempt math or config-sourced — if `test_no_inline_physics_coeffs` flags the smoothstep `3.0`/`2.0`, they are exempt math constants; add `# coeff-ok: C1 Hermite smoothstep` only if the ratchet requires).

- [ ] **Step 8: Commit**

```bash
git add packages/land/legoesm/land/carbon/config.py packages/land/legoesm/land/carbon/carbon_cycle.py tests/land/unit/test_carbon_cycle.py
git commit -m "feat(carbon): NSC/substrate-gated maintenance-respiration factor (opt-in)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3Xx8yToULd6EGpdBEpGn3"
```

---

### Task 3: Cold-deciduous freeze-dormancy factor + config fields

**Files:**
- Modify: `packages/land/legoesm/land/carbon/config.py` (fields + spec; add `from legoesm import constants`)
- Modify: `packages/land/legoesm/land/carbon/carbon_cycle.py` (helper `_cold_deciduous_dormancy_factor`)
- Test: `tests/land/unit/test_carbon_cycle.py`

**Interfaces:**
- Consumes: `CarbonConfig.freeze_dormancy_threshold_K`, `.dormancy_transition_width_K`, `.cold_deciduous`, `.cold_deciduous_dormancy`
- Produces: `_cold_deciduous_dormancy_factor(T, config) -> jnp.ndarray` ∈ (0, 1]

- [ ] **Step 1: Add config fields** (`config.py`, right after the Task-2 fields). Ensure `from legoesm import constants` is imported at module top.

```python
    # Mechanism 2 — cold-deciduous freeze dormancy: for cold-deciduous PFTs,
    # zero foliar GPP + foliar R_maint below a freeze threshold (larch/tundra
    # leaf-drop metabolism).  Default OFF -> byte-identical.
    cold_deciduous_dormancy: bool = False   # master enable
    cold_deciduous: bool = False            # per-PFT trait (from is_cold_deciduous)
    freeze_dormancy_threshold_K: float = constants.T_freeze  # dormancy below this T [K]
    dormancy_transition_width_K: float = 2.0                 # sigmoid half-width [K]
```

- [ ] **Step 2: Add `__param_spec__` entries** — tunable threshold in `params`, numerics width in `excluded`:

```python
# in CarbonConfig["params"]:
            "freeze_dormancy_threshold_K": {"units": "K", "bounds": (263.0, 278.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "cold-deciduous winter-dormancy onset temperature (~0 degC); larch/tundra phenology", "shape": None},
# in CarbonConfig["excluded"]:
            "dormancy_transition_width_K": "numerics: cold-deciduous dormancy sigmoid half-width [K]",
```

- [ ] **Step 3: Write the failing test**

```python
def test_cold_deciduous_dormancy_factor_zeros_when_frozen():
    import jax.numpy as jnp
    from legoesm.land.carbon.carbon_cycle import _cold_deciduous_dormancy_factor
    from legoesm.land.carbon.config import CarbonConfig
    from legoesm import constants
    cfg = CarbonConfig(scheme="differland", cold_deciduous_dormancy=True,
                       cold_deciduous=True, dormancy_transition_width_K=2.0)
    d_cold = _cold_deciduous_dormancy_factor(
        jnp.asarray(constants.T_freeze - 10.0), cfg)
    d_thr = _cold_deciduous_dormancy_factor(jnp.asarray(constants.T_freeze), cfg)
    d_warm = _cold_deciduous_dormancy_factor(
        jnp.asarray(constants.T_freeze + 10.0), cfg)
    assert float(d_cold) < 0.01          # frozen -> dormant (no foliar flux)
    assert abs(float(d_thr) - 0.5) < 1e-6  # sigmoid midpoint at the threshold
    assert float(d_warm) > 0.99          # warm -> active
```

- [ ] **Step 4: Run to verify it fails** — Expected: FAIL (import error).

- [ ] **Step 5: Write the helper** (`carbon_cycle.py`, after `_nsc_respiration_factor`)

```python
def _cold_deciduous_dormancy_factor(
    T: jnp.ndarray,
    config: CarbonConfig,
) -> jnp.ndarray:
    """Cold-deciduous winter-dormancy factor ``d`` in ``(0, 1]``.

    Smooth freeze-onset sigmoid ``d = sigmoid((T - freeze_dormancy_threshold_K)
    / dormancy_transition_width_K)`` -> ``1`` for warm (active canopy), ``-> 0``
    when frozen (leaves shed / metabolically dormant).  Callers multiply it onto
    the FOLIAR GPP and FOLIAR maintenance-respiration terms so a cold-deciduous
    PFT neither photosynthesises nor pays foliar respiration through the frozen
    season (larch/tundra strategy).  Differentiable; the ``cold_deciduous`` /
    ``cold_deciduous_dormancy`` STATIC gates are applied by the caller (feature
    gating), so this returns the smooth factor unconditionally.
    """
    z = (T - config.freeze_dormancy_threshold_K) / config.dormancy_transition_width_K
    return jax.nn.sigmoid(z)
```

- [ ] **Step 6: Run to verify it passes.** Then `tests/test_param_specs.py -q` PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/land/legoesm/land/carbon/config.py packages/land/legoesm/land/carbon/carbon_cycle.py tests/land/unit/test_carbon_cycle.py
git commit -m "feat(carbon): cold-deciduous freeze-dormancy factor (opt-in)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3Xx8yToULd6EGpdBEpGn3"
```

---

### Task 4: Wire both gates into GPP + R_maint (the integration)

**Files:**
- Modify: `packages/land/legoesm/land/carbon/carbon_cycle.py` — `compute_gpp` (75-119) and the `R_maint_day` block (661-666)
- Test: `tests/land/unit/test_carbon_cycle.py`

**Interfaces:**
- Consumes: `_nsc_respiration_factor`, `_cold_deciduous_dormancy_factor` (Tasks 2/3)
- Produces: gated `compute_gpp`; gated `R_maint_day` inside `step_carbon_differland`

- [ ] **Step 1: Gate `compute_gpp`** — replace the final return (line 119):

```python
    # Cold-deciduous freeze dormancy (Mechanism 2): zero foliar GPP when the
    # canopy is frozen/dormant.  STATIC gates -> plain Python if (feature gating,
    # not a traced jnp.where); d == 1.0 when off => byte-identical.
    if config.cold_deciduous_dormancy and config.cold_deciduous:
        d = _cold_deciduous_dormancy_factor(T, config)
    else:
        d = 1.0
    return jnp.maximum(config.epsilon * APAR * f_T * f_CO2 * beta, 0.0) * d
```

- [ ] **Step 2: Gate `R_maint_day`** — replace lines 662-666 with the split, both factors:

```python
    # NSC/substrate gate (Mechanism 1) throttles ALL maintenance terms; the
    # cold-deciduous dormancy gate (Mechanism 2) additionally zeros the FOLIAR
    # term.  Both are static-flag feature gates; f_nsc == 1.0 and d == 1.0 when
    # off => byte-identical to the ungated (r_maint_fol*C_fol + ...)·temp_factor.
    if config.nsc_gated_respiration:
        f_nsc = _nsc_respiration_factor(
            state.C_lab, state.C_fol, state.C_root, state.C_wood, config)
    else:
        f_nsc = 1.0
    if config.cold_deciduous_dormancy and config.cold_deciduous:
        d = _cold_deciduous_dormancy_factor(T, config)
    else:
        d = 1.0
    # Sign convention (carbon, positive-out): R_maint is a LOSS plant->atmosphere;
    # f_nsc, d in (0,1] REDUCE the loss (survival), never increase it.
    R_maint_day = (
        config.r_maint_fol * state.C_fol * f_nsc * d
        + config.r_maint_root * state.C_root * f_nsc
        + config.r_maint_wood * state.C_wood * f_nsc
    ) * temp_factor_ra  # gC/m2/day
```

- [ ] **Step 3: Write the byte-identical + monotonicity + conservation tests**

```python
def _demo_step(cfg):
    """One differland step on a fixed warm column; returns (new_state, co2_flux)."""
    import jax.numpy as jnp
    from legoesm.land.carbon.carbon_cycle import step_carbon_differland
    from legoesm.land.carbon.config import CarbonState
    state = CarbonState(
        C_lab=jnp.asarray(50.0), C_fol=jnp.asarray(120.0), C_root=jnp.asarray(150.0),
        C_wood=jnp.asarray(9000.0), C_lit=jnp.asarray(200.0),
        C_som_active=jnp.asarray(300.0), C_som_slow=jnp.asarray(3000.0),
        C_som_passive=jnp.asarray(12000.0))
    return step_carbon_differland(
        state, sw_down=jnp.asarray(250.0), T=jnp.asarray(295.0),
        precip=jnp.asarray(3e-5), co2_ppmv=jnp.asarray(400.0), beta=jnp.asarray(0.8),
        doy=180.0, lat=jnp.asarray(0.7), dt=3600.0, config=cfg)

def test_gates_off_are_byte_identical_to_param_changes():
    # Gate flags OFF: perturbing the new params must not change the step at all.
    from legoesm.land.carbon.config import CarbonConfig
    import jax
    base = CarbonConfig(scheme="differland")                 # all gates default off
    pert = base._replace(nsc_ref_labile_frac=0.5, r_maint_floor_frac=0.9,
                         freeze_dormancy_threshold_K=250.0, dormancy_transition_width_K=8.0)
    a, _ = _demo_step(base); b, _ = _demo_step(pert)
    for fa, fb in zip(a, b):
        assert float(jax.numpy.max(jax.numpy.abs(fa - fb))) == 0.0

def test_nsc_gate_reduces_rmaint_and_conserves():
    # Gate ON at a depleted labile: NPP rises (less respiration) and the step
    # conserves carbon (Σ ΔC == -co2_flux over dt, kgC vs kgCO2 mass conversion).
    import jax.numpy as jnp
    from legoesm.land.carbon.config import CarbonConfig, CarbonState
    from legoesm.land.carbon.carbon_cycle import step_carbon_differland
    depleted = CarbonState(
        C_lab=jnp.asarray(1.0), C_fol=jnp.asarray(120.0), C_root=jnp.asarray(150.0),
        C_wood=jnp.asarray(9000.0), C_lit=jnp.asarray(200.0),
        C_som_active=jnp.asarray(300.0), C_som_slow=jnp.asarray(3000.0),
        C_som_passive=jnp.asarray(12000.0))
    common = dict(sw_down=jnp.asarray(0.0), T=jnp.asarray(280.0), precip=jnp.asarray(1e-6),
                  co2_ppmv=jnp.asarray(400.0), beta=jnp.asarray(0.1), doy=15.0,
                  lat=jnp.asarray(1.1), dt=3600.0)
    off = CarbonConfig(scheme="differland", nsc_gated_respiration=False)
    on = CarbonConfig(scheme="differland", nsc_gated_respiration=True,
                      nsc_ref_labile_frac=0.02, r_maint_floor_frac=0.10)
    s_off, co2_off = step_carbon_differland(depleted, config=off, **common)
    s_on, co2_on = step_carbon_differland(depleted, config=on, **common)
    tot = lambda s: sum(float(getattr(s, f)) for f in s._fields)
    # gate on retains more carbon (less respired) than gate off:
    assert tot(s_on) > tot(s_off)
    # conservation: total plant+soil C change == -(net CO2-C) flux * dt (gate on).
    # co2_flux is kgCO2/m2/s, positive up; ΔC is gC/m2. 44/12 = CO2/C mass ratio.
    dC = tot(s_on) - tot(depleted)
    co2_C = float(co2_on) * (12.0 / 44.0) * 1e3 * 3600.0   # kgCO2/m2/s -> gC/m2 over dt
    assert abs(dC + co2_C) < 1e-3 * max(1.0, abs(dC))
```

*(Note: confirm `CarbonState` field names + `step_carbon_differland` signature against the module before running; adjust the `tot`/flux mass conversion to the module's actual `co2_flux` units — the test asserts the invariant, tune the constant to the real units.)*

- [ ] **Step 4: Run the tests** — Expected: PASS (byte-identical, retention, conservation).

- [ ] **Step 5: Write the death-spiral regression (starved-column survival)**

```python
def test_starved_boreal_column_survives_only_with_nsc_gate():
    """A cold, dark, low-beta forcing integrated for a boreal-winter-length run
    drives live biomass -> ~0 with the gate OFF (death spiral) but RETAINS
    biomass with it ON.  Provably non-vacuous: fails if the gate does nothing."""
    import jax, jax.numpy as jnp
    from legoesm.land.carbon.config import CarbonConfig, CarbonState
    from legoesm.land.carbon.carbon_cycle import step_carbon_differland
    def run(cfg, n=2000):
        s = CarbonState(
            C_lab=jnp.asarray(20.0), C_fol=jnp.asarray(80.0), C_root=jnp.asarray(120.0),
            C_wood=jnp.asarray(4000.0), C_lit=jnp.asarray(100.0),
            C_som_active=jnp.asarray(200.0), C_som_slow=jnp.asarray(2000.0),
            C_som_passive=jnp.asarray(8000.0))
        for i in range(n):
            s, _ = step_carbon_differland(
                s, sw_down=jnp.asarray(20.0), T=jnp.asarray(268.0),
                precip=jnp.asarray(1e-6), co2_ppmv=jnp.asarray(400.0),
                beta=jnp.asarray(0.05), doy=float(15 + i) % 365.0,
                lat=jnp.asarray(1.1), dt=3600.0, config=cfg)
        return float(s.C_fol + s.C_root + s.C_wood)
    live_off = run(CarbonConfig(scheme="differland", nsc_gated_respiration=False))
    live_on = run(CarbonConfig(scheme="differland", nsc_gated_respiration=True,
                               nsc_ref_labile_frac=0.02, r_maint_floor_frac=0.10))
    assert live_on > 10.0 * max(live_off, 1e-6)   # gate rescues the column
```

- [ ] **Step 6: Run it** — Expected: PASS (gate ON retains ≫ gate OFF). If the loop is slow, wrap the body in `jax.lax.scan` per the module's existing forward integrator; keep the assertion.

- [ ] **Step 7: Full carbon-cycle test module + differentiability smoke**

```bash
.../python -m pytest tests/land/unit/test_carbon_cycle.py -q
JAX_ENABLE_X64=1 .../python -c "import jax, jax.numpy as jnp; from legoesm.land.carbon.carbon_cycle import _nsc_respiration_factor; from legoesm.land.carbon.config import CarbonConfig; g=jax.grad(lambda cl:_nsc_respiration_factor(cl,jnp.asarray(1.),jnp.asarray(1.),jnp.asarray(1.),CarbonConfig(scheme='differland',nsc_gated_respiration=True))); print('grad', float(g(jnp.asarray(0.5))))"
```
Expected: all pass; finite gradient printed (differentiable).

- [ ] **Step 8: Commit**

```bash
git add packages/land/legoesm/land/carbon/carbon_cycle.py tests/land/unit/test_carbon_cycle.py
git commit -m "feat(carbon): gate GPP + R_maint by NSC reserve and cold-deciduous dormancy

Fixes the high-latitude death spiral (opt-in, default-off byte-identical,
carbon-conserving, differentiable). Starved boreal column survives with the
NSC gate on; dies with it off.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3Xx8yToULd6EGpdBEpGn3"
```

---

### Task 5: Physics contract + `run_lmip` CLI flags + round-trip tests

**Files:**
- Modify: `packages/land/legoesm/land/carbon/carbon_cycle.py` (`__physics_contract__` — locate with `grep -n __physics_contract__`)
- Modify: `scripts/run/run_lmip.py` (CLI flags + config wiring)
- Test: `tests/unit/test_run_lmip_cli.py`

**Interfaces:**
- Consumes: the four tunables + two master gates on `CarbonConfig`
- Produces: `--nsc-gated-respiration/--no-nsc-gated-respiration`, `--cold-deciduous-dormancy/--no-...`, `--nsc-ref-labile-frac`, `--r-maint-floor-frac`, `--freeze-dormancy-threshold-k` flags

- [ ] **Step 1: Extend `__physics_contract__`** — add to the module contract's notes/`conserves` that the opt-in NSC + cold-deciduous gates reduce (never increase) R_maint/GPP and conserve carbon. Run `.../python -m pytest tests/test_physics_contracts.py -q` → PASS.

- [ ] **Step 2: Write the failing CLI round-trip test**

```python
# tests/unit/test_run_lmip_cli.py
def test_carbon_arctic_rescue_flags_round_trip():
    from scripts.run.run_lmip import build_arg_parser, build_config_from_args  # match actual names
    args = build_arg_parser().parse_args([
        "--nsc-gated-respiration", "--cold-deciduous-dormancy",
        "--nsc-ref-labile-frac", "0.03", "--r-maint-floor-frac", "0.15",
        "--freeze-dormancy-threshold-k", "271.0"])
    cfg = build_config_from_args(args)
    cc = cfg.carbon  # the CarbonConfig on the assembled land config
    assert cc.nsc_gated_respiration is True
    assert cc.cold_deciduous_dormancy is True
    assert abs(cc.nsc_ref_labile_frac - 0.03) < 1e-9
    assert abs(cc.r_maint_floor_frac - 0.15) < 1e-9
    assert abs(cc.freeze_dormancy_threshold_K - 271.0) < 1e-9
```

- [ ] **Step 3: Run to verify it fails** — Expected: FAIL (unknown args / attributes).

- [ ] **Step 4: Add the flags + wiring** in `run_lmip.py` (mirror the existing `--carbon-woody`/`--cwd-humification-eff` flags: `argparse` `BooleanOptionalAction` for the two gates, `type=float` for the three tunables; thread into the `CarbonConfig(...)` builder). Use `constants.T_freeze` as the `--freeze-dormancy-threshold-k` default.

- [ ] **Step 5: Run to verify it passes** — Expected: PASS. Also `tests/unit/test_params_reachability_audit.py -q` PASS (the three floats reachable via `--params land.carbon.*`).

- [ ] **Step 6: Commit**

```bash
git add packages/land/legoesm/land/carbon/carbon_cycle.py scripts/run/run_lmip.py tests/unit/test_run_lmip_cli.py
git commit -m "feat(lmip): CLI flags for the arctic carbon-productivity gates + physics contract

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3Xx8yToULd6EGpdBEpGn3"
```

---

### Task 6: Column no-regression + realism gate (7-pixel, compute node)

**Files:**
- Create: `scripts/cluster/land_carbon/validate_arctic_rescue.sbatch`
- Use: `scripts/validate/land_carbon_equilibrium.py`

**Interfaces:** consumes the Task-5 CLI; produces a scorecard comparing flags-off vs flags-on at the 7 climate pixels.

- [ ] **Step 1: Write the sbatch** (mirror `run_equilibrium.sbatch`; two runs — flags off then on — to the same pixels, same dt/years/output layout, so the ONLY change is the flags)

```bash
#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=arctic_rescue_val
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/lc_rebase/scripts/tmp/%x_%j.out
set -uo pipefail
WORKDIR=/burg-archive/glab/users/pg2328/lc_rebase; cd "$WORKDIR"
export PYTHONPATH=$(ls -d "$WORKDIR"/packages/*/ | sed 's:/$::' | tr '\n' ':')
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=1"
PYJ=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== OFF (baseline) ==="; /usr/bin/time -v $PYJ scripts/validate/land_carbon_equilibrium.py \
  --years 120 --dt 3600 --n-layers 10 --soil-depth 3.0 --output results/arctic_rescue_off 2>&1
echo "=== ON (nsc + cold-deciduous) ==="; /usr/bin/time -v $PYJ scripts/validate/land_carbon_equilibrium.py \
  --years 120 --dt 3600 --n-layers 10 --soil-depth 3.0 --nsc-gated-respiration \
  --cold-deciduous-dormancy --output results/arctic_rescue_on 2>&1
echo "EXIT=$? done: $(date)"
```
*(If `land_carbon_equilibrium.py` lacks these flags, add the same pass-through there — same-PR CLI rule — before this task.)*

- [ ] **Step 2: Submit** `sbatch scripts/cluster/land_carbon/validate_arctic_rescue.sbatch`; poll `squeue`. Do NOT run on the login node.

- [ ] **Step 3: Read the two scorecards.** Gate criteria:
  - Temperate-forest / tropical / savanna / temperate-grass equilibria **unchanged** between off and on (within a few %).
  - Boreal + tundra pixels: **nonzero** live biomass + SOC with flags ON (they decline/→0 with OFF).
  If temperate/tropical moved materially, lower `nsc_ref_labile_frac` (gate biting healthy plants) and re-run.

- [ ] **Step 4: Commit the sbatch + a short findings note** under `docs/land/` (basename-referenced).

---

### Task 7: Global-init grouping + master-flag threading + cache bump

**Files:**
- Modify: `packages/land/legoesm/land/carbon/global_init.py:404,408-412` (grouping key + config) and `equilibrate_archetypes`/`iter_archetype_batches` signatures (thread the two master flags)
- Modify: `scripts/data/build_global_carbon_ic.py` (bump `_EQUILIBRIUM_CACHE_VERSION`; add `--nsc-gated-respiration`/`--cold-deciduous-dormancy` flags threaded into `equilibrate_archetypes`)
- Test: `tests/land/unit/test_global_init.py`

**Interfaces:** produces cold-deciduous-aware archetype grouping and flag-threaded equilibria.

- [ ] **Step 1: Failing test** — `is_cold_deciduous` splits a larch archetype into its own group and, with `cold_deciduous_dormancy=True`, its config carries `cold_deciduous=True`:

```python
def test_grouping_sets_cold_deciduous_trait():
    # build a tiny ArchetypeTable containing needleleaf_deciduous_boreal + a temperate
    # broadleaf, call iter_archetype_batches(..., cold_deciduous_dormancy=True), and
    # assert the larch group's CarbonConfig.cold_deciduous is True, the other's False.
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Extend the grouping** (`global_init.py`):

```python
        key = (is_woody(pft_name), is_evergreen(pft_name),
               is_cold_deciduous(pft_name), str(soil_class[a]))
    ...
    for (group_woody, group_evergreen, group_cold_deciduous, soil), members in groups.items():
        ...
        carbon_cfg = CarbonConfig(
            scheme="differland", woody=group_woody, evergreen=group_evergreen,
            cold_deciduous=group_cold_deciduous,
            nsc_gated_respiration=nsc_gated_respiration,          # threaded kwarg
            cold_deciduous_dormancy=cold_deciduous_dormancy,     # threaded kwarg
            C_lab_init=_C_LAB_SEED, C_fol_init=_C_FOL_SEED, C_root_init=_C_ROOT_SEED,
            C_wood_init=_C_WOOD_SEED, C_lit_init=_C_LIT_SEED, C_som_init=_C_SOM_SEED)
```
Add `nsc_gated_respiration: bool = False, cold_deciduous_dormancy: bool = False` kwargs to `iter_archetype_batches` and `equilibrate_archetypes`, default False (byte-identical). Import `is_cold_deciduous`.

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Build driver** — in `build_global_carbon_ic.py`: add `--nsc-gated-respiration`/`--cold-deciduous-dormancy` argparse flags, pass to `equilibrate_archetypes(...)`, and **bump** `_EQUILIBRIUM_CACHE_VERSION` `"v4" -> "v5"` with a comment ("+ opt-in NSC/cold-deciduous productivity gates change the high-lat equilibria when enabled"). Extend the `_equilibrium_cache_key` docstring's params caveat to note the two flags now affect the equilibrium.

- [ ] **Step 6: Test + commit**

```bash
.../python -m pytest tests/land/unit/test_global_init.py -q
git add packages/land/legoesm/land/carbon/global_init.py scripts/data/build_global_carbon_ic.py tests/land/unit/test_global_init.py
git commit -m "feat(carbon): cold-deciduous archetype grouping + thread productivity gates into the global IC build

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3Xx8yToULd6EGpdBEpGn3"
```

---

### Task 8: Global IC rebuild (flags-on) + codex review + controlled comparison

**Files:** uses `scripts/cluster/land_carbon/build_global_carbon_ic.sbatch` (a flags-on variant)

- [ ] **Step 1: Codex adversarial review** (compute node, read-only) over the branch diff `d4c5f5115..HEAD`:

```bash
# on a compute node (srun --pty ... or an sbatch wrapper), NOT the login node:
codex exec --sandbox read-only "Adversarially review the arctic carbon-productivity fix
(NSC-gated R_maint + cold-deciduous dormancy) in packages/land/legoesm/land/carbon/:
check carbon conservation, sign conventions, differentiability, the byte-identical-when-off
guarantee, and the winter-leafless reference robustness."
```
Iterate fixes until clean (the iterate-with-codex loop).

- [ ] **Step 2: Flags-on global rebuild** — copy `build_global_carbon_ic.sbatch` → `build_global_carbon_ic_arctic.sbatch` adding `--nsc-gated-respiration --cold-deciduous-dormancy` and a distinct `--output results/global_carbon_ic_arctic` + `--equilibrium-cache-dir .carbon_equilibrium_cache` (v5 key → clean recompute). `sbatch` it (CPU, ~7–12 h).

- [ ] **Step 3: Controlled comparison** — with the **identical** climatology/grid/spin config, compare `results/global_carbon_ic/` (flags-off, v4) vs `results/global_carbon_ic_arctic/` (flags-on, v5): per-PFT SOC/biomass (expect `needleleaf_deciduous_boreal` > 0, arctic band SOC ↑), area-weighted global SOC toward 9.5, and temperate/tropical per-PFT ranges unchanged. Rerun the drift validator (`validate_global_carbon_ic.sbatch` pointed at the arctic archetypes) → still PASS (< 5 %/yr). Report numbers next to configs (no protocol drift).

- [ ] **Step 4: Finalize** — write a findings note under `docs/land/`, update memory, and (separately) decide the merge-to-main path for the whole stranded stack.

---

## Self-Review

**Spec coverage:** M1 NSC gate → Tasks 2,4; M2 cold-deciduous → Tasks 1,3,4; config+specs → Tasks 2,3; classifier → Task 1; grouping+cache → Task 7; CLI → Task 5; conservation/sign/diff/byte-identical → Task 4 tests; no-regression → Task 6; global rebuild+codex → Task 8. All spec sections covered.

**Placeholder scan:** the two flagged "confirm against the module" notes (CarbonState field names in Task 4; `land_carbon_equilibrium.py` flag pass-through in Task 6) are verification instructions, not placeholders — each has a concrete fallback action. No TBD/TODO.

**Type consistency:** `_nsc_respiration_factor(C_lab, C_fol, C_root, C_wood, config)` and `_cold_deciduous_dormancy_factor(T, config)` signatures identical across Tasks 2/3/4. Field names `nsc_gated_respiration`, `nsc_ref_labile_frac`, `r_maint_floor_frac`, `cold_deciduous_dormancy`, `cold_deciduous`, `freeze_dormancy_threshold_K`, `dormancy_transition_width_K` consistent across config, tests, CLI, grouping.

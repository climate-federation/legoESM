# Cold-deciduous leaf-carbon resorption — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Add carbon-conserving leaf-carbon resorption (C_fol→C_lab at abscission) so cold-climate deciduous stands refill their labile reserve instead of collapsing to biomass 0, default-off/byte-identical.

**Architecture:** One new `CarbonConfig.leaf_c_resorption_frac` field gates a fractional split of the existing leaf-fall flux in `step_carbon_differland`: the shed `leaf_litter` is partitioned into a resorbed part (→C_lab) and a litter part (→C_lit), instead of all→C_lit. Reuse the shipped `cold_deciduous_dormancy` gate alongside it (no new code). Efficacy proven on the carbon-balance diagnostic before wider integration.

**Tech Stack:** JAX, DifferLand DALEC-based prognostic carbon, pytest.

## Global Constraints

- Compute (pytest / JAX) runs ONLY via sbatch on Ginsburg compute nodes (`--account=glab`), never the login node. Python: `/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python`, `PYTHONPATH=$(ls -d packages/*/…)` glob.
- Carbon-conserving: `Σ ΔC == −NEE·dt` at machine precision must still hold.
- Differentiable: static-bool/fraction only, no Python control flow on traced values, no new `jnp.where` on data.
- `leaf_c_resorption_frac` default `0.0` ⇒ byte-identical to current.
- Every new config field: `__param_spec__` entry. Codex adversarial review before done.
- Worktree: `/burg-archive/glab/users/pg2328/legoESM_resorption`, branch `land/cold-deciduous-resorption`.

---

### Task 1: Resorption flux + config field + unit tests

**Files:**
- Modify: `packages/land/legoesm/land/carbon/config.py` (field ~line 322, `__param_spec__` ~line 12)
- Modify: `packages/land/legoesm/land/carbon/carbon_cycle.py` (`step_carbon_differland`: after line 804; lines 936, 1021, 1032)
- Test: `tests/land/unit/test_carbon_cycle.py`

**Interfaces:**
- Produces: `CarbonConfig.leaf_c_resorption_frac: float = 0.0`. Read only inside `step_carbon_differland`.

- [ ] **Step 1: Write failing tests** in `tests/land/unit/test_carbon_cycle.py` (new class `TestLeafResorption`):

```python
class TestLeafResorption(unittest.TestCase):
    def _common(self):
        st = _make_carbon_state(shape=(3,), C_fol=200.0, C_lab=50.0, C_lit=100.0)
        # winter-ish leaf-fall day so lff>0; NH deciduous
        return st, dict(T=jnp.full(3, 283.0), sw_down=jnp.full(3, 200.0),
                        co2_ppmv=jnp.full(3, 400.0), beta=jnp.ones(3),
                        precip=jnp.full(3, 2e-5), lat=jnp.zeros(3),
                        doy=jnp.full(3, 280.0), dt=3600.0)

    def test_off_is_byte_identical(self):
        st, kw = self._common()
        s0, _f0 = step_carbon_differland(st, config=_make_carbon_config(leaf_c_resorption_frac=0.0), **kw)
        s_base, _fb = step_carbon_differland(st, config=_make_carbon_config(), **kw)
        for p in s0._fields:
            self.assertTrue(jnp.allclose(getattr(s0, p), getattr(s_base, p), atol=0, rtol=0), p)

    def test_resorption_moves_fol_carbon_to_labile_not_litter(self):
        st, kw = self._common()
        s_on, _ = step_carbon_differland(st, config=_make_carbon_config(leaf_c_resorption_frac=0.3), **kw)
        s_off, _ = step_carbon_differland(st, config=_make_carbon_config(leaf_c_resorption_frac=0.0), **kw)
        # C_fol loss identical (resorption does not change the C_fol sink)
        self.assertTrue(jnp.allclose(s_on.C_fol, s_off.C_fol, atol=1e-9))
        # C_lab higher, C_lit lower, by the SAME amount (conserving transfer)
        self.assertTrue(jnp.all(s_on.C_lab > s_off.C_lab))
        self.assertTrue(jnp.all(s_on.C_lit < s_off.C_lit))
        gained = s_on.C_lab - s_off.C_lab
        lost = s_off.C_lit - s_on.C_lit
        self.assertTrue(jnp.allclose(gained, lost, rtol=1e-6), "resorbed C_lab gain != C_lit loss")

    def test_monotonic_in_fraction(self):
        st, kw = self._common()
        labs = [step_carbon_differland(st, config=_make_carbon_config(leaf_c_resorption_frac=f), **kw)[0].C_lab
                for f in (0.0, 0.1, 0.3)]
        self.assertTrue(jnp.all(labs[1] >= labs[0]) and jnp.all(labs[2] >= labs[1]))

    def test_conserves_with_resorption(self):
        st, kw = self._common()
        cfg = _make_carbon_config(leaf_c_resorption_frac=0.3)
        new, nee, _d = step_carbon_differland(st, config=cfg, return_diagnostics=True, **kw)
        dC = sum(getattr(new, p) - getattr(st, p) for p in st._fields)
        self.assertTrue(jnp.allclose(dC, -nee * kw["dt"], atol=1e-6),
                        "Σ ΔC == −NEE·dt violated with resorption")
```

- [ ] **Step 2: Run tests, verify they FAIL** (via sbatch — see Global Constraints):
Run: `pytest tests/land/unit/test_carbon_cycle.py::TestLeafResorption -q`
Expected: FAIL (`TypeError: unexpected keyword 'leaf_c_resorption_frac'`).

- [ ] **Step 3: Add config field** in `config.py`, in `CarbonConfig` right after `leaf_fall_period` (~line 322):

```python
    leaf_c_resorption_frac: float = 0.0  # frac of shed foliage C resorbed C_fol->C_lab at abscission [-]
```

And a `__param_spec__` entry (in the dict at line 12, next to the phenology closures):

```python
            "leaf_c_resorption_frac": {"units": "1", "bounds": (0.0, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "fraction of shed foliage carbon resorbed into the labile reserve before abscission (deciduous leaf-C/N recovery; Aerts 1996; Vergutz et al. 2012)", "shape": None},
```

- [ ] **Step 4: Implement the flux split** in `carbon_cycle.py`. After line 804 (`leaf_litter = ...`), add:

```python
    # Leaf-carbon resorption at abscission: a fraction of the shed foliage C is
    # resorbed into the labile reserve (C_fol -> C_lab) rather than lost to
    # litter, as deciduous plants recover leaf C/N before leaf-fall (Aerts 1996;
    # Vergutz 2012).  Carbon-conserving: C_fol still loses the FULL leaf_litter;
    # the shed flux is partitioned between C_lab (resorbed) and C_lit (litter).
    # f == 0 (default) => leaf_resorb == 0, leaf_to_lit == leaf_litter =>
    # every downstream term is byte-identical to the pre-resorption code.
    leaf_resorb = config.leaf_c_resorption_frac * leaf_litter  # gC/m2/day -> C_lab
    leaf_to_lit = leaf_litter - leaf_resorb                    # gC/m2/day -> C_lit
```

Change line 936 (`lit_net_avail`) `leaf_litter` → `leaf_to_lit`:
```python
        state.C_lit + (leaf_to_lit + root_litter) * dt_days, 0.0,
```
Change line 1021 (C_lab) to add `+ leaf_resorb`:
```python
            state.C_lab + (A_lab - lab_release - lab_deficit_draw + leaf_resorb) * dt_days),
```
Change line 1032 (C_lit) `leaf_litter` → `leaf_to_lit`:
```python
            state.C_lit + (leaf_to_lit + root_litter
                           - R_het_lit - lit_to_som) * dt_days),
```
Leave C_fol (line 1023) and `NEE_day` UNCHANGED (resorption is a live-pool-internal transfer; C_fol still loses the full `leaf_litter`, and no atmosphere flux changes).

- [ ] **Step 5: Run tests, verify PASS** (sbatch):
Run: `pytest tests/land/unit/test_carbon_cycle.py::TestLeafResorption -q`
Expected: 4 passed. Also run the existing conservation class to confirm no break: `pytest tests/land/unit/test_carbon_cycle.py -q` → all pass.

- [ ] **Step 6: Commit**
```bash
git add packages/land/legoesm/land/carbon/config.py packages/land/legoesm/land/carbon/carbon_cycle.py tests/land/unit/test_carbon_cycle.py
git commit  # feat(carbon): leaf-carbon resorption at abscission (opt-in, conserving)
```

---

### Task 2: Efficacy gate — does resorption + dormancy revive larch's cells?

**Files:**
- Modify: `scripts/tmp/larch_carbon_balance_diag.py` (copy from lc_rebase; add resorption variants)

**Interfaces:**
- Consumes: `CarbonConfig.leaf_c_resorption_frac` (Task 1). The diagnostic passes it via `iter_archetype_batches` overrides / a config field on the batch config.

- [ ] **Step 1:** Copy `scripts/tmp/larch_carbon_balance_diag.py` from the lc_rebase worktree into this worktree; add variants that set `leaf_c_resorption_frac ∈ {0.0, 0.2, 0.35}` together with `cold_deciduous_dormancy=True`, integrating the mat 266/271/277/282 ladder. Because `iter_archetype_batches` builds the group config, thread the fraction by overriding `batch.config.carbon` via `_replace(leaf_c_resorption_frac=f)` on the returned batch config before `make_archetype_step_fn` (mirror how the diagnostic already toggles the gates).

- [ ] **Step 2:** sbatch the diagnostic (CPU, ~10min). Read the trajectory.

- [ ] **Step 3: GO/NO-GO.** PASS iff larch at mat 265–277 reaches sustained LAImax > 0 and NPP ≥ 0 (nonzero equilibrium biomass) at some `f ≤ 0.4` with dormancy on. If PASS → record the working `f`, proceed to Task 3. If FAIL → STOP, report to the user; the fallback (cold-acclimated GPP) is a separate design, not part of this plan.

- [ ] **Step 4: Commit** the diagnostic extension (`scripts/tmp` is gitignored; instead record the verdict + working `f` in the spec doc and commit the doc update).

---

### Task 3: CLI integration (only if Task 2 PASSES)

**Files:**
- Modify: `scripts/run/run_lmip.py` (flag + builder wiring)
- Modify: `scripts/data/build_global_carbon_ic.py` (flag + cache-key hash)
- Test: `tests/land/unit/test_run_lmip_cli.py` (round-trip)

- [ ] **Step 1:** Write a failing round-trip test in `tests/land/unit/test_run_lmip_cli.py`: parsing `--leaf-c-resorption-frac 0.2` sets `CarbonConfig.leaf_c_resorption_frac == 0.2` on the built config. Run (sbatch) → FAIL.
- [ ] **Step 2:** Add `--leaf-c-resorption-frac` (type float, default `CarbonConfig().leaf_c_resorption_frac`) to `run_lmip.py` argparse + wire into the config builder; add the same flag to `build_global_carbon_ic.py` and hash it into `_equilibrium_cache_key` next to the gate flags (bump `_EQUILIBRIUM_CACHE_VERSION`).
- [ ] **Step 3:** Run the round-trip test (sbatch) → PASS.
- [ ] **Step 4: Commit** (`feat(carbon): --leaf-c-resorption-frac CLI + cache-key`).

---

### Task 4: No-regression + codex review

- [ ] **Step 1:** sbatch a controlled archetype A/B on THIS tree: rebuild archetypes with `leaf_c_resorption_frac=<working f>` + `cold_deciduous_dormancy` on vs off (same clim/surf, one variable). Assert temperate_deciduous / tropical biomass Δ within tolerance (≤5%) AND boreal/larch archetypes reach nonzero. Record the table.
- [ ] **Step 2:** Codex adversarial review (`codex exec --sandbox read-only`) on the diff; fix flagged; re-review until clean or 30 iter.
- [ ] **Step 3:** Push branch, open PR (climate-federation, same-repo), verify head SHA + files, merge (user gate for the merge). Ship opt-in / default-off; note the default-on proposal as a follow-up gated on the no-regression table.

---

## Self-review

- **Spec coverage:** Mechanism-1 flux → Task 1. Config+spec → Task 1. Conservation/sign/byte-identical/monotonicity tests → Task 1. Efficacy gate → Task 2. CLI+cache → Task 3. No-regression + codex → Task 4. Dormancy reuse → Task 2/4 (config toggle, no new code). All spec sections covered.
- **Placeholder scan:** none — all steps carry real code / exact commands.
- **Type consistency:** `leaf_c_resorption_frac` (float) used identically in config, flux, tests, CLI. `leaf_resorb`/`leaf_to_lit` defined once, consumed at 936/1021/1032.

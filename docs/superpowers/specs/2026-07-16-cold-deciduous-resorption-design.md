# Cold-climate deciduous viability: leaf-carbon resorption + dormancy — design

**Status:** approved design, pre-implementation.
**Goal:** make cold-climate deciduous trees (larch `needleleaf_deciduous_boreal`,
and the coldest boreal-broadleaf cells) equilibrate to nonzero carbon instead of
collapsing to biomass 0, without regressing the temperate/tropical equilibria.

## Problem

The science-grade global carbon IC has a high-latitude deficit
(`docs/land/arctic_carbon_residual_audit.md`): larch equilibrates to biomass 0 /
GPP 0 at every climate, and even boreal-broadleaf dies at its coldest cells. The
audit ruled out leaf economics and established that it is a model-wide deciduous
cold-tolerance limit, not a per-PFT parameter bug.

### Diagnostic evidence (job 9055372, raw forward integration, no analytic reset)

A seeded established stand (C_fol=150, C_wood=3000, C_lab=100 gC/m²) was
integrated 40 yr across a mean-annual-temperature ladder (mat 266/271/277/282 K,
seasonal amplitude fixed), for four gate variants. The mechanism is unambiguous:

- **The binding constraint is labile-reserve (C_lab) depletion.** C_lab drains
  from the seed (100) to **0 within 2–5 years**; only then does leaf-out fail
  (LAImax → 0 → GPP → 0 → biomass bleeds down). C_lab drains because each spring
  leaf-out spends it while the labile re-fill from NPP (`a_lab`) cannot keep up
  in a short cold season — and **leaf-fall dumps 100% of C_fol to litter (no
  resorption), so the reserve can never recover the leaf carbon.**
- The existing dormancy + NSC gates (variant D) lift viability to **mat ≈ 280 K
  only** (mat 282 survives: sustained LAImax 0.64, NPP +137 at yr 39). At mat
  266–277 — larch's actual cells — even dormancy + NSC collapses. So dormancy
  alone is insufficient; a reserve-*refill* lever is missing.

## Design

Two mechanisms attacking the C_lab budget: **resorption refills** the reserve,
**dormancy protects** it.

### Mechanism 1 — leaf-carbon resorption at abscission (new)

Physically universal: deciduous plants resorb ~10–30 % of leaf carbon (and ~50 %
of N) into storage before abscission (Aerts 1996; Vergutz et al. 2012). The model
currently discards all of it.

In `step_carbon_differland` (`carbon/carbon_cycle.py`), the leaf-fall flux
`leaf_litter = C_fol * _effective_rate(lff, dt_days)` [gC/m²/day] is split:

- current: `C_fol −= leaf_litter·dt`, `C_lit += leaf_litter·dt`
- new:     `C_fol −= leaf_litter·dt` (unchanged);
           `C_lab += f·leaf_litter·dt`;  `C_lit += (1−f)·leaf_litter·dt`

with `f = config.leaf_c_resorption_frac`.

- **Carbon-conserving by construction:** the C_fol sink is unchanged; the shed
  flux is partitioned between C_lab and C_lit, summing to `leaf_litter`
  (`−L + f·L + (1−f)·L = 0`). The closed-column `Σ ΔC = −NEE·dt` identity is
  preserved and must be asserted by a unit test.
- **Differentiable:** a fractional split of an existing flux; no Python control
  flow on traced values, no new `jnp.where`.
- **Config:** `CarbonConfig.leaf_c_resorption_frac: float = 0.0` → **byte-
  identical when off** (f=0 ⇒ the new terms vanish, C_lit split reduces to the
  current all-to-litter). `__param_spec__`: units dimensionless, bounds
  `(0.0, 0.5)`, `tunable_tier` extended (2), transform sigmoid, category
  phenology, reference "Aerts 1996; Vergutz 2012".
- **Scope:** applies to the leaf-fall flux for all PFTs (physically correct
  everywhere), but only materially affects deciduous PFTs (evergreen `lff` is a
  small continuous turnover). Because it is carbon-conserving and defaults off,
  it cannot silently perturb any equilibrium; the no-regression gate quantifies
  its default-on effect on temperate/tropical.

### Mechanism 2 — whole-plant cold-deciduous dormancy (reuse existing)

Reuse the shipped opt-in `cold_deciduous_dormancy` gate (zeroes foliar GPP and
whole-plant R_maint when frozen, `config.cold_deciduous_dormancy AND
config.cold_deciduous`). No new code; it is enabled *together* with resorption in
the build/run config so the reserve is both refilled (resorption) and protected
(dormancy). PFT-scoped via `is_cold_deciduous`.

## Staged validation (de-risks — resorption may still be insufficient)

1. **Efficacy gate FIRST.** Implement the flux + unit tests (conservation, sign
   `0 ≤ resorbed ≤ shed`, byte-identical-when-0, monotonicity: more f ⇒ more
   C_lab), then add resorption to the carbon-balance diagnostic
   (`scripts/tmp/larch_carbon_balance_diag.py`) and confirm it **revives larch's
   actual cells (mat 265–277)** — sustained LAImax > 0, NPP ≥ 0, nonzero
   equilibrium biomass. This is a hard go/no-go before wider integration. If
   short, iterate `f` (up to ~0.4) and/or combine with a cold-GPP adjustment
   (separate design) before proceeding.
2. **Integrate (only if the gate passes).** CLI flags on the drivers that expose
   `CarbonConfig` tunables — `run_lmip.py` (`--leaf-c-resorption-frac`) and
   `scripts/data/build_global_carbon_ic.py` (same flag, hashed into the
   equilibrium-cache key next to the gate flags) — each with a round-trip CLI
   test. No-regression on the temperate/tropical archetypes (rebuild archetypes
   with resorption on vs off; temperate/tropical biomass Δ within tolerance,
   controlled comparison on the SAME tree). Codex adversarial review.
3. **Ship opt-in / default-off.** Then, only if the full global build shows no
   temperate/tropical regression AND a real arctic gain, propose flipping the
   default on in a follow-up.

## Files

- `packages/land/legoesm/land/carbon/carbon_cycle.py` — the resorption flux split
  in `step_carbon_differland`, reading `config.leaf_c_resorption_frac`.
- `packages/land/legoesm/land/carbon/config.py` — the field + `__param_spec__`.
- `tests/land/unit/test_carbon_cycle.py` — conservation / sign / byte-identical /
  monotonicity unit tests (tendencies, not just integration).
- `scripts/run/run_lmip.py`, `scripts/data/build_global_carbon_ic.py` — CLI flag +
  wiring + cache-key hash; round-trip tests in the corresponding
  `tests/.../test_run_*_cli.py`.
- `scripts/tmp/larch_carbon_balance_diag.py` — extend with the resorption variant
  (efficacy gate).

## Acceptance criteria

- Unit tests: carbon conservation residual ≈ 0 with f > 0; `resorbed = f·shed`
  exactly; byte-identical outputs when f = 0; C_lab monotonic in f.
- Efficacy: larch mat 265–277 reaches nonzero sustained equilibrium (LAImax > 0,
  NPP ≥ 0) with resorption + dormancy.
- No-regression: temperate/tropical archetype biomass unchanged within tolerance
  under the default-off setting (trivially) and quantified under default-on.
- Codex adversarial review clean.

## Risks / open questions

- **Resorption may be insufficient at the coldest cells** (like dormancy alone).
  Mitigated by the staged efficacy gate — we learn this before integrating, and
  the fallback (cold-acclimated GPP) is a separate, larger design not undertaken
  here unless the gate forces it.
- **Default-on regression risk** for temperate/tropical deciduous (they gain a
  reserve buffer). Mitigated by keeping default-off and quantifying before any
  flip.
- Resorption fraction magnitude: literature spans ~10–30 % C; treat as a tunable
  (`tier` extended) with a conservative enabled default (~0.2) chosen from the
  efficacy sweep.

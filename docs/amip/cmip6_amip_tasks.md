# Fully-Physical CMIP6 AMIP — Task Ledger

Companion to [`cmip6_amip_plan.md`](cmip6_amip_plan.md). Status: ✅ done · 🔄 in progress ·
⬜ todo · 🧊 blocked. Keep this in sync as work lands.

---

## P0 — Reference config
- ✅ Rebuild `amip_production.yaml` to Pierre's 6 directives (latlon24/L20, stable-BL MOST,
  multilayer land + `beta_soil`, McFarlane GWD, ERA5 IC + CMIP6 forcing).
- ✅ `--surface-stability-scheme` CLI flag + round-trip test (directive 5).
- ✅ Land over-evaporation fix (β_soil throttling, #833 merged).
- ✅ Baseline stability shakedown (30-day latlon24, no drift).

## P1 — Robustly-stable baseline
- ✅ Physics-toggle stability harness (`physics_toggle_screen.py`) + tests.
- ✅ Tier-1 5-day screen (nothing in the reference set destabilizes baseline).
- ✅ Top-of-atmosphere sponge #836 → PR #846 (config-gated; fixes anvil-thinning blow-ups).
- 🔄 Understand the day-8-9 localized "non-finite winds" instability under cloud-reducing
  perturbations (subgrid-autoconversion cases still blow with the sponge on).
- ⬜ Exit criterion: reference baseline + intended calibration levers run **30 days** drift-free.

## P2 — Cloud + precip calibration (the hard part)
- ✅ Numerical skill gate `amip_skill_score.py` vs CERES/GPCP/ERA5 + tests.
- ✅ Tier-2 CLI sweeps (cloud-albedo + precip-efficiency) — **verdict: rsut insensitive to all CLI
  levers (~228); precip stuck ~0.8 vs 2.8**. Coarse CLI phase exhausted.
- 🔄 **P2a — root-cause the precip=0.8 weak hydrological cycle**: decompose precip (convective vs
  large-scale), check surface evaporation (hfls/evap — land β #833 + drier-init 0.25 over-throttle?),
  verify the precip diagnostic sums all sources. ← CURRENT
- ⬜ P2b — expose the identified lever (likely Bechtold `precip_efficiency` as a top-level
  `ExperimentConfig` scalar + CLI flag; it is NOT `--params`-reachable in AMIP) + round-trip test.
- ⬜ P2c — SCM-RCE tuning of the responsive params → inject via `--params`.
- ⬜ P2d — differentiable gradient calibration (cloud↔precip coupled) vs CERES/GPCP.

## P3 — Validation at latlon24
- ⬜ 30-day tuned run scored by the skill gate (target: fail-count « 2/9; albedo/precip physical).
- ⬜ Visual regression (cube imprint N/A on latlon; check spatial precip/cloud fields).
- ⬜ Conservation (mass/energy/moisture-fixer scale ≈ 1).

## P4 — CMIP6 campaign scale-up
- ⬜ Confirm merged 1979-2014 CMIP6 forcing reader (Pinatubo/El Chichón present — staged).
- ⬜ Higher-res / longer pilot (decade) → full 36-yr chain.
- ⬜ Multi-GPU/node SPMD at production res (#693 solved via #755).
- ⬜ CMOR Amon output + ClimateEval scorecard.

## Track B — structural fixes (own PRs)
- ✅ Sponge #836 → PR #846.
- ⬜ Non-orographic GWD #834 (combined orographic + non-oro).
- ⬜ Radiation cadence dt-invariance (`rad_update_steps` step-count vs wall-time).
- ⬜ `clear_sky_diag` no-op #843 (rsutcs/rlutcs never produced).

## Open GitHub items
- PR #846 (sponge, closes #836) · issue #843 (clear_sky_diag) · issue #834 (non-oro GWD).
- Synced from main: #839 (dq_r, closed #832), #840 (`--conv-cloud-condensate`), #835→#838.

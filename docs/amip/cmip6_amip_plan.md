# Fully-Physical CMIP6-Protocol AMIP — Plan

**GOAL.** A fully physical, CMIP6-protocol AMIP simulation in legoESM — real CMIP6 external
forcings + ERA5 initial conditions + our agreed reference parameters (Pierre's 6 directives) +
the real multilayer-Richards land model — that reaches **acceptable numerical skill** and can be
scaled to the multi-decade CMIP6 campaign. `lat-lon n_lat24 / L20` is **step 1** (cheap, fast);
higher-res / longer runs come after.

Companion task ledger: [`cmip6_amip_tasks.md`](cmip6_amip_tasks.md).

---

## Reference configuration (Pierre's 6 directives — authoritative)

| # | Directive | Setting | State |
|---|---|---|---|
| 1 | Grid: lat-lon or MPAS, low-res | `latlon` `resolution:24` `L20` `latlon_cgrid`, `dt 600→60` (pole clamp) | done |
| 2 | Cloud scheme: **calibrate** | Sundqvist + Morrison; **over-reflective, in calibration** | **open (P2)** |
| 3 | GWD parameterizations correct | McFarlane orographic; non-oro pending (#834) | done / #834 |
| 4 | Land: LMIP-calibrated, not Jarvis | multilayer Richards + `beta_soil`, `land_stomatal_beta:false` | done |
| 5 | Surface: new stable-BL MOST | `surface_stability_scheme: beljaars_holtslag1991` | done |
| 6 | Stomata on (via `beta_soil`) | on | done |

Single source of truth: `config/amip/amip_production.yaml` (physics) + `config/amip/amip_production.sh`
(machine paths). Skill gate: `scripts/validate/amip_skill_score.py` vs CERES-EBAF / GPCP / ERA5.

---

## Phases

### P0 — Reference config (DONE)
Rebuilt to the 6 directives; committed; baseline **stable** at latlon24 (30-day shakedown, no drift).

### P1 — Lock a robustly-stable baseline (IN PROGRESS)
Baseline is stable, but **cloud-reducing perturbations destabilize** at day 7-9 (localized
"non-finite winds", modest max_v). Structural finding: the hydrostatic latlon-cgrid had **no
top sponge** (#836, PR #846) — it decisively fixes the anvil-thinning blow-ups but not the
subgrid-autoconversion ones. **Exit criterion:** the reference baseline + the intended calibration
levers run 30 days drift-free at latlon24.

### P2 — Calibrate cloud + precip to acceptable skill (IN PROGRESS — the hard part)
**Current diagnosis (2026-07-07):** over-reflection (rsut ~228, albedo 0.65 vs 0.29) is
**insensitive to every CLI cloud/precip lever** (subgrid autoconversion, q_c, anvil condensate all
leave rsut ~223-229). The real anomaly is **precip ~0.8 vs 2.8 mm/day everywhere** — the whole
hydrological cycle is ~3× too weak; cloud water condenses but barely precipitates, so it piles up.
- **P2a — root-cause the weak hydrological cycle** (precip 0.8): decompose precip (convective vs
  large-scale), check surface evaporation (hfls/evap — the land β_soil #833 + drier-init 0.25 may
  over-throttle), verify the precip diagnostic sums all sources.
- **P2b — pull the identified lever.** Candidates: Bechtold `precip_efficiency` (NOT `--params`-
  reachable in AMIP — needs a top-level `ExperimentConfig` scalar + CLI flag, cf. Pierre's #840),
  ice-phase autoconversion, surface-evaporation retune, cloud-optical mapping.
- **P2c — tune properly** once the responsive params are known: SCM-RCE
  (`run_scm_rce_convection_tuning.py`) then differentiable gradient calibration
  (`train_scm_rce_params.py` + `param_collector.build_trainable_params`) of the coupled
  cloud↔precip params against CERES/GPCP.

### P3 — Validate at latlon24
30-day run scored by `amip_skill_score.py` (tas/pr/rlut/rsut/hfls/prw/clt/netTOA vs obs) + visual
regression + conservation. **Acceptable skill** = substantial drop from the 2/9 fail count, albedo
and precip physical, no drift.

### P4 — Scale to the CMIP6 campaign
Higher-res / longer / merged 1979-2014 CMIP6 forcing (built, reader-verified). Pilot decade → full
36-yr. Multi-GPU/node SPMD (#693, solved via #755). CMOR Amon output.

---

## Track B — structural fixes (parallel, own PRs)
- **Top sponge #836** → PR #846 (config-gated, default off; corrected honest body).
- **Non-orographic GWD #834** — combined orographic + non-oro (single-select today).
- **Radiation cadence** — `rad_update_steps` is a step count; the pole-cell dt-clamp fires RT ~10×
  more than intended → wall-time-based cadence.
- **`clear_sky_diag` no-op #843** — rsutcs/rlutcs never produced (blocks CRE decomposition).

## Process (throughout)
- Every bug → **draft the GitHub issue, show the user before posting**; land validated work as
  focused PRs; **sync `origin/main` before each phase** (Pierre works the same issues — he fixed
  #835/#839/#840; re-verify imports after every merge).
- **Controlled comparison**: change ONE variable, hold the eval protocol byte-identical; report the
  full config next to every number; never compare across drifted setups.

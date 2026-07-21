# legoESM WeatherBench Forecast Campaign — Design

**Date:** 2026-07-04
**Branch/worktree:** dedicated worktree off `main` (`legoESM_wbforecast`, branch `wb-forecast-campaign`) — avoids the shared-checkout long-run hazard (concurrent sessions mutating model code mid-run).
**Author:** Pierre Gentine (with Claude)

## 1. Goal

Demonstrate that legoESM produces skillful deterministic weather forecasts, using a strategy analogous to the AIMIP intercomparison: **swap one physics parameterization family at a time** (microphysics, clouds, turbulence/PBL, convection, gravity-wave drag), **train the swapped scheme's parameters on ERA5 under the WeatherBench protocol** (multi-step forecast-error objective), then **assess a WeatherBench-2 lead-time scorecard** (RMSE + ACC vs ERA5) and position legoESM against SOTA models reported in WeatherBench / WeatherBenchX.

## 2. Honest scientific framing (locks the claim)

The WeatherBench-2 headline board is led by pure-ML emulators (GraphCast, GenCast, Pangu) that beat operational IFS at coarse resolution. A physics-based GCM at ~1–2° **will not** beat GraphCast on day-3–5 Z500 RMSE — that is not the winnable game and we will not claim it.

The defensible, winnable claim is the **NeuralGCM lane**: a *differentiable* physics core whose parameterizations are *trained on ERA5* becomes a competitive **physics/hybrid** forecaster. The demonstration is:
1. Training physics parameters on the forecast objective measurably improves short-to-medium-range skill and reduces systematic bias vs the untrained default.
2. The best swapped+trained legoESM configuration sits credibly in the physics/hybrid tier of the WB2 board (near IFS / NeuralGCM-physics), with an explicit, honest gap to pure-ML at medium range.
3. The swap-and-train ablation reveals *which physics choices matter* for forecast skill — a scientific result independent of the absolute ranking.

Expected outcome (stated up front to avoid overclaiming): physics tuning most improves **1–3 day skill** and **systematic biases / climate drift**; day-5–10 skill is dominated by dycore + resolution and will move only modestly. This is the NeuralGCM finding and we reproduce its *shape*.

## 3. Decisions (locked defaults; revisit with user if desired)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| Grid/res | Which dycore + resolution | **Staged spectral T63 → T106** | Reuse the ACE2/AIMIP spectral infra verbatim; T63 single-GPU + proven **full BPTT**; escalate only the winners to T106 (~1.1°, head-to-head configs already exist). Scoring regrids to the WB2 1.5° common grid regardless. |
| Sweep | Schemes per family | **Curated 2–3 per family** (~13 arms) + 5 per-family `none` controls | Covers the meaningful physics spread; tractable on SLURM. Full ~40-scheme menu deferred as an optional extension. |
| Loss | Training objective | **Multi-step forecast RMSE + CRPS** (GenCast-style BPTT), with **climate-stat drift tracked as a guardrail diagnostic** (not initially in the loss) | This *is* the WeatherBench protocol (forecast skill at lead times); CRPS avoids RMSE double-penalty/blurring; the climate guardrail respects the project's physical-consistency priority (forecast-tuning must not wreck the climate). |
| SOTA | Comparison scope | **Full WB2 board, honest lane** | Overlay WB2 published headline scores (IFS HRES/ENS, GraphCast, Pangu, GenCast, NeuralGCM, climatology, persistence); position legoESM in the physics/hybrid tier; be explicit pure-ML leads the headline RMSE. |

**Curated scheme menu (each arm perturbs exactly ONE family from the AIMIP ACE2 classical winner reference):**

| Family | Schemes (arms) | Trainable params (approx, via `__param_spec__`) |
|--------|----------------|--------------------------------------------------|
| Microphysics | `kessler`, `morrison`, `thompson` | 5 / 61 / 33 |
| Clouds | `sundqvist`, `xu_randall` | shared `CloudConfig` ~20 |
| Turbulence/PBL | `louis`, `ysu`, `edmf` | 6 / 10 / 8 |
| Convection | `tiedtke`, `bechtold`, `edmf` | 18 / 18 / 6 |
| Gravity-wave drag | `mcfarlane`, `hines` | 9 / 4 |
| Controls | family = `none` (×5) | — |

Reference (fixed) config for OAT = the AIMIP ACE2 classical winner set (edmf/louis/mcfarlane/sundqvist/xu_randall); each arm changes one family string + trains that family's params, holding the rest fixed.

## 4. Architecture — three stages on one foundation

### Stage 0 — WeatherBench eval harness (net-new foundation, MUST be first)

The existing `evaluations/weatherbench.py` is an orphaned skeleton: it scores only a grid-space `model(x, grid)` (SFNO-style), loads a climatology but **never computes ACC**, has **no surface headline vars**, no CLI, no SegmentCarry bridge, and no SOTA baselines. The metric primitives (`evaluations/metrics.py::rmse/acc/bias/compute_scorecard`) and plotters (`evaluations/visualize.py`) are correct and reused verbatim.

Build (all in `evaluations/`, the canonical top-level WB2 home; reuse metrics/visualize, do **not** re-derive):

1. **SegmentCarry → WB2 scorer bridge** (`evaluations/wb_forecast.py`, new, tested). Given a trained `physics_fn` + spectral dycore + grid + sigma: init `SegmentCarry` from ERA5 at each init date (reuse `era5_to_state.era5_to_spectral_carry`), roll out via `spectral_rollout` to each lead, diagnose headline vars, regrid to the WB2 1.5° common grid, score RMSE+ACC+bias vs ERA5 verification and WB2 climatology. Emits a `(var, level, lead)` scorecard consumed by the existing `compute_scorecard`/`plot_*`.
2. **Headline diagnostics** (`evaluations/headline_diagnostics.py`, new, tested — pre-impl grep done: factor from `grids/vertical.py` geopotential + `dephy_scm.py::_interp_profile_to_pressure`; do not duplicate):
   - **Z500 / T850 / Q700 / U,V@level**: hydrostatic geopotential integration Φ(σ) from surface (Φ_s = phis) upward, then log-pressure interpolation to the requested level. Sign convention documented at the term (Φ increases upward; hydrostatic ∂Φ/∂ln p = −R_d T_v).
   - **MSLP**: standard sea-level reduction from p_s using T and phis.
   - **T2m / 10m wind**: surface-layer (MOST) screen-level diagnostic from the lowest model level + surface (reuse the existing `SurfaceLayerConfig` similarity functions; grep before writing).
   - **Total precipitation**: accumulate `PhysicsOutput.precip` over the WB2 accumulation window (field confirmed `.precip`).
   - All diagnostics JAX-pure and differentiable (log-p interp differentiable).
3. **ACC wiring**: compute anomaly correlation vs the WB2 ERA5 climatology (1990–2019) already loaded by the loader; wire the imported-but-unused `acc`.
4. **Baselines + SOTA**: (a) compute our own **climatology** and **persistence** forecasts from ERA5 (cheap anchors); (b) fetch/import WB2 published headline scores for IFS HRES/ENS, GraphCast, Pangu, GenCast, NeuralGCM from the WB2 results Zarr/CSV; fall back to a small committed CSV of paper headline numbers if network is unavailable on the compute node.
5. **CLI**: `scripts/validate/run_weatherbench_eval.py` — given a checkpoint + config, run the WB2 protocol on the 2020 test year (init dates every 12h, subsample allowed), emit `wb_scorecard.{json,png}` + lead-time RMSE/ACC curves. SLURM sbatch wrapper under `scripts/cluster/`.

**Gate:** unit tests for every new module (diagnostics against analytic columns + a known ERA5 snapshot; ACC against a hand-computed case; regrid conservation for precip). A T21 smoke run of the full scorer end-to-end. Codex adversarial review before Stage 1.

### Stage 1 — OAT swap + train (T63)

Clone the AIMIP sweep-matrix generator pattern (`run_aimip_classical_sweep_stage1.py` + `config/aimip/sweep/stage1/manifest.json`) into a **WeatherBench** variant:
- `config/wb/` tree: base `wb_era5.yaml` (T63, dycore, ERA5 windows, **multi-step forecast loss** block: `multi_step_hours: [6,12,18,24]`, MSE+CRPS weights, `residual_normalize`), per-family `combo_*/` overlays (one scheme string swap each), sbatch array runner.
- Training reuses `_train_spectral_loop` unchanged (full BPTT over the ≤24 h unroll). Each arm trains its one family's `__param_spec__` params via `apply_param_overrides` inside the loss (traced leaves; production leaves stay static).
- Each arm is scored by the **Stage-0 WB2 scorer** on the held-out test window (not the single-lead AIMIP scorer) → per-arm `(var, level, lead)` scorecard.
- Aggregate → pick the winner scheme per family (composite over headline RMSE+ACC at 1/3/5 d).

**Gate per arm:** the physics-contract / dispatch-hardening / constants ratchets already apply (no new physics numerics — only swap + train). Codex adversarial review on the harness + config generator. SLURM array for the runs. Note training horizon (≤48 h unroll) vs eval horizon (10 d) explicitly.

### Stage 2 — Assemble + joint-train (T63, then T106 for the winner)

- Combine the winning scheme per family into one `wb_best.yaml`; **joint-train all trainable params** across families under the same multi-step forecast loss (this is the multi-family generalization of a single AIMIP variant).
- Track the climate-stat guardrail (ACE2 climatology/spectral diagnostic) alongside forecast loss; abort/flag if climate drift worsens materially.
- Escalate the single best config to **T106** (reuse `config/aimip/t106/` head-to-head machinery) for the headline result.

**Gate:** codex adversarial review; conservation + physics-contract gates; SLURM (72 h walltime budget as needed).

### Stage 3 — Final scorecard vs SOTA

- Run the best T63 and T106 models through the Stage-0 WB2 eval on the full 2020 test year.
- Overlay WB2 published SOTA + our climatology/persistence anchors.
- Emit the headline deliverables: RMSE-vs-lead and ACC-vs-lead curves (Z500, T850, T2m, Q700, MSLP, 10m wind), the scorecard heatmap, and a short results doc under `docs/` with the honest positioning from §2.
- Regenerate + send the headline PNG (standing preference).

## 5. Reuse / no-duplication ledger (CLAUDE.md compliance)

- **Metrics**: `evaluations/metrics.py` (rmse/acc/bias/compute_scorecard) + `ml/loss.py` (latitude_weighted_rmse, area_weighted_afcrps) — reused, not re-derived.
- **Loss**: `training/losses.py::multi_step_rollout_loss` + `neural_gcm_spectral.py::_spectral_state_loss_components` — reused; **no new loss numerics**.
- **ERA5**: `era5_to_state.py` (IC + verification), WB2 Zarr + climatology loaders — reused.
- **Training loop**: `_train_spectral_loop` + `make_aimip_classical_spectral_physics` swap point — reused; new campaign = new configs + a WB scorer, not a new trainer.
- **Sweep matrix**: `run_aimip_classical_sweep_stage1.py` pattern — cloned to `config/wb/`, not reinvented.
- **Constants/thermo**: all diagnostics use `legoesm.constants` + `legoesm.thermo`; hydrostatic/geopotential factored from `grids/vertical.py`.
- **New `.py` gets a direct test** (diagnostics, scorer, CLI). Scheme/factory dispatch already raises on unknown.

## 6. Risks & mitigations

- **BPTT instability** at T63 over the rollout → keep the training unroll short (≤24–48 h), full BPTT on spectral only (latlon is truncated-BPTT-only, hence spectral choice); reuse the thermo AD clamps that fixed the prior latlon grad-NaN.
- **Diagnostic sign/level errors** (Z500 hydrostatic integration, MSLP reduction) → analytic-column + known-ERA5-snapshot unit tests; mandatory sign-convention walk per CLAUDE.md.
- **Overclaiming vs SOTA** → §2 framing is the deliverable's spine; the scorecard shows the honest gap.
- **Compute** → everything runs on SLURM (`--account=glab`, up to 72 h); login node only for build/git/grep. Pinned worktree for reproducibility.
- **Physics-tuning wrecks climate** → climate-stat guardrail diagnostic in Stage 2; abort criterion.

## 7. Deliverables

1. `evaluations/wb_forecast.py`, `evaluations/headline_diagnostics.py` (+ tests) — the reusable WB2 scorer for `SegmentCarry` dycores, with ACC + surface headline vars.
2. `scripts/validate/run_weatherbench_eval.py` + `scripts/cluster/` sbatch — the WB2 eval CLI.
3. `config/wb/` — base + per-family combo + suite configs; sweep-array generator.
4. Stage-1 per-scheme WB2 scorecards; Stage-2 assembled+joint-trained best model (T63 + T106).
5. Stage-3 headline scorecard vs SOTA + a results doc under `docs/` with honest positioning.
6. Codex adversarial review run + verdict reported at each stage.

## 8. Out of scope (YAGNI)

- Pure-ML emulator arms (col_nn / SFNO-physics exist but are not the brief; optional later add-on).
- 0.25° resolution (WB2 also runs 1.5°; our comparison is at 1.5°).
- Ensemble/probabilistic headline beyond the CRPS training term (deterministic scorecard is the demonstration).
- Full ~40-scheme menu (curated subset first; extend if signal warrants).

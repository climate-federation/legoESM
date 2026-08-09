# U-Cast learning strategy vs ours — extracted recipe, ranked gap, what landed

Source: Cachay, Watson-Parris & Yu, *U-Cast: A Surprisingly Simple and
Efficient Frontier Probabilistic AI Weather Forecaster*, arXiv:2604.09041
(code: https://github.com/Rose-STL-Lab/u-cast). Companion to
[[ace2_learning_strategy_gap]], which does the same for ACE2.

The paper's own summary of its recipe: *"a probabilistic forecaster built on a
standard U-Net backbone trained with a simple recipe: deterministic
pre-training on Mean Absolute Error followed by short probabilistic
fine-tuning on the Continuous Ranked Probability Score (CRPS) using Monte
Carlo Dropout for stochasticity"* — matching or exceeding GenCast and IFS ENS
at 1.5° while *"reducing training compute by over 10x compared to leading
CRPS-based models"*, in **under 12 H200 GPU-days**.

That last number is the reason this paper matters to us specifically: our
constraint has never been architecture, it has been data volume and compute.
A frontier probabilistic recipe that costs 12 GPU-days is inside our budget.

## U-Cast, exactly as published

| Item | U-Cast |
|---|---|
| Architecture | standard U-Net (DhariwalUNet), 1.5° (a 1° variant also trained) |
| Inputs / outputs | **two past snapshots** -> residual w.r.t. the last input; 172 in (2 x 83 + 6 forcings), 83 out (6 vars x 13 levels + 5 surface) |
| Step | **12 h**; 60-step autoregressive rollout = 15 days |
| Data | ERA5 1979-2019 |
| Stage 1 | **spatially weighted MAE** (latitude-dependent area weights), **100 epochs** |
| Stage 2 | **unbiased CRPS** (Zamo & Naveau 2018), ensemble **M=2** in training, **8 epochs** |
| Optimizer | **Muon** (peak LR 3e-3, wd 0.1) + AdamW for 1-D params (3e-4, wd 0.03); linear warmup 1500 steps then cosine; batch 48 |
| Stage-2 optimizer | Muon 7e-3 / AdamW 7e-5, same schedule shape |
| EMA | decay **0.9999**, evaluated |
| Stochasticity | **MC-Dropout, p=0.10**, standard dropout layers, **active during training AND inference** |
| Inference ensemble | K=4 deep-ensemble members x N stochastic rollouts each |
| Loss weights | stage 2 uses GenCast's per-variable weights |

### Their ablations (what each ingredient is worth, in THEIR setup)

| Ingredient | Effect |
|---|---|
| Muon vs AdamW | AdamW degrades z500 CRPS by **up to 15 %** (1-3 day range) |
| Dropout rate 5 % / 10 % / 15 % | **~1 % CRPS** across the whole range — not a tuned quantity |
| Training M=2 vs M=4 | M=4 gives "only marginal CRPS gains" |
| adaLN vs MC-Dropout | adaLN improves spread-skill but **significantly degrades CRPS** |
| MAE-pretrain -> CRPS vs CRPS from scratch | curriculum is **3.4 % better CRPS** and converges in **3x fewer steps** |

## Ours before this change (`sfno_full`, T63 tier)

| Item | ours |
|---|---|
| Architecture | SFNO, embed 64 / 4 blocks, 34 channels, 68.3M params |
| Inputs | **one** snapshot -> next state; 6 h step |
| Loss | pure MSE (`config/wb/loss_presets/ace2.yaml`), residual-normalized, 2 x 6 h |
| Optimizer | AdamW 1e-4 in the shipped maxdata arm (`run_aimip` itself defaults to `muon`; the ACE2-copying suites override it back to AdamW) |
| EMA | 0.999 |
| Stochasticity | **none in the model.** Ensembles came from (a) 8 independently trained seeds, (b) IC perturbation — documented in its own source as "a SENSITIVITY probe, not a calibrated analysis-error ensemble" |
| Probabilistic scoring | **none.** The WB2 scorecard collapsed any ensemble to its mean and reported RMSE/ACC/bias |

## The gap, ranked

1. **No stochasticity in the network at all.** Every "ensemble" we have costs
   either N full training runs (the 8-seed mx8) or relies on an uncalibrated
   IC perturbation. U-Cast gets an unlimited ensemble from ONE network for the
   price of N forward passes. **CLOSED** — see below.
2. **No probabilistic score.** We could not have *detected* a CRPS improvement
   if we had made one: the scorecard had no CRPS and no spread-skill column,
   so an under-dispersed ensemble and a calibrated one looked identical.
   Building the instrument had to precede any CRPS training.
   **CLOSED** — see below.
3. **MSE, not MAE, in stage 1.** The M=1 limit of fair-CRPS *is* the MAE, so
   MAE-pretraining is the matched initialisation for a CRPS fine-tune. It is
   NOT a deterministic-RMSE improvement — MSE is the RMSE-optimal
   deterministic loss — so this must be measured, not assumed.
   **AVAILABLE, not yet run** (`config/wb/loss_presets/ucast.yaml`).
4. **AdamW, not Muon.** Their single largest optimizer effect (up to 15 % z500
   CRPS). `create_optimizer` already ships `muon` and `muon_partitioned`, and
   `run_aimip` already defaults to `muon`; the shipped arms opted out to copy
   ACE2. **One config line, unmeasured in this campaign.**
5. **No stage-2 CRPS fine-tune.** The pieces exist
   (`LossConfig.w_afcrps`, `ml.loss.area_weighted_afcrps`,
   `neural_gcm_spectral.ensemble_afcrps_loss`) but nothing calls them from
   `_train_sfno_full_loop`. **STILL OPEN — the largest remaining item.**
6. **One input snapshot, not two.** Two past states give the network a finite
   difference, i.e. a tendency it currently has to infer. Costs a channel-spec
   change (34 -> 68 input channels) and a loader change. **OPEN.**
7. **EMA 0.999 vs 0.9999.** Smallest item, and 0.9999 may be too slow at our
   step counts — an EMA horizon of 10^4 updates against our few-thousand-update
   epochs would lag badly. Not obviously portable. **OPEN.**

## What landed

### 1. MC-Dropout in the SFNO (`ml/sfno_block.py`, `ml/sfno.py`)

`SFNOConfig.dropout` (default 0.0). Dropout sits inside each block's MLP,
after the hidden activation and before the output projection — the same
position U-Cast's residual block uses (`...SiLU -> dropout -> conv`), i.e. on
the residual BRANCH, never on the identity path, so the skip connection stays
exact. `SFNO.__call__(x, grid, key=None)`: `None` runs the layer in inference
mode.

Two properties are asserted because both are load-bearing:

* **Enabling dropout does not change the deterministic forecast.** Same
  weights, no key -> bit-identical output. This is what lets an arm turn the
  knob on without invalidating the deterministic scorecard protocol.
* **The rate is a STATIC equinox field.** `eqx.nn.Dropout` keeps `p` and
  `inference` as ORDINARY fields, so holding one as a submodule adds two
  leaves per block and breaks `tree_deserialise_leaves` against every
  checkpoint written before the knob existed. Measured, not reasoned:
  loading `epoch_0059.eqx` into the submodule version raised
  `TreePathError at blocks[0].dropout.p`. With the static field, the same
  checkpoint loads into both a p=0 and a p=0.1 skeleton and reproduces its
  forecast exactly.

`SFNOPrimitiveEquationModel.ensemble_step(state, key, n_members)` mirrors the
existing `UCastPrimitiveEquationModel.ensemble_step` (vmap over per-member
keys) and routes every member through the SAME `_apply_postprocess` chain
(filter -> mass -> clip) as `step`. It refuses `dropout == 0`, where every
member is identical and the CRPS silently degenerates to the MAE.

### 2. Dropout ACTIVE during training

U-Cast's dropout is on during training, not only at inference — MC-Dropout
members drawn from a network that never saw a mask are not the recipe. The
first version of this change threaded a key through the model and the
evaluator but NOT through `_rollout_segment`, so `sfno_dropout` would have
trained a dropout-free network that the evaluator then sampled. Caught by
adversarial review.

Now: `_rollout_segment(..., key)` folds the scan index for a **fresh mask
every macro step**; `_chained_loss` folds per segment; the epoch loop derives
`fold_in(fold_in(fold_in(base, dp_rank), epoch), sample_idx)`. `dp_rank` is
folded FIRST because every rank walks the same `(epoch, sample_idx)` counters
over its own shard — without it all ranks apply the identical mask to
different samples, correlating exactly the gradients the data-parallel mean
exists to decorrelate. The pmap step takes a per-DEVICE key (`in_axes=0`).

The test is a real 2-sample training run, not an `inspect.getsource` probe:
same init, same data, same seed, `sfno_dropout` 0.0 vs 0.2 -> the weights must
differ (and 0.0 twice must be bit-identical). A source assertion would have
passed against the broken version.

### 3. The probabilistic instrument (`evaluations/wb_forecast.py`, `wb_orchestrator.py`)

`score_ensemble_forecast` returns, per field:

* `crps` — area-weighted almost-fair CRPS, from
  `legoesm.ml.loss.almost_fair_crps` (the same estimator the training loss
  uses, so train and eval cannot drift);
* `spread` — area-weighted RMS of the `ddof=1` ensemble standard deviation;
* `spread_skill` — `sqrt((M+1)/M) * spread / rmse(ensemble mean)`. ~1
  calibrated, <1 under-dispersed.

The area weighting and masking are copied from `evaluations.metrics.rmse` so a
CRPS and an RMSE on the same row integrate over exactly the same cells. The
first version summed an un-broadcast `(n_lat, 1)` weight column and inflated
every score by `n_lon`; the "identical members must reduce to the plain MAE"
test caught it.

`run_wb_forecast_eval` accepts a member LIST from `rollout_fn`, diagnoses each
member, and means the FIELDS — which also removes the mean-then-diagnose error
the nonlinear z500/mslp diagnostics carried. The list check deliberately
rejects a tuple: a spectral state IS a NamedTuple, so `isinstance(rolled,
tuple)` silently unpacked one state into its fields and called them members
(caught by the existing orchestrator tests).

### 4. Eval CLI

`--mc-dropout-members N` (K checkpoints x N draws each) and `--probabilistic`.
A fresh dropout key per macro step; the base key varies per
(checkpoint, draw, case) and is held FIXED across leads within a case so a
member is a coherent trajectory rather than a re-randomised field per lead.

`--probabilistic` also switches the ensemble mean from mean-then-diagnose to
diagnose-then-mean. That is the more correct construct but a DIFFERENT one, so
it is opt-in and its help text says a deterministic number from this mode is
not comparable to one without it.

`--mc-dropout-members` on a checkpoint whose suite resolves `sfno_dropout=0`
is refused: every member would be the same forecast.

### 5. Architecture sidecar

`train_sfno_full_spectral` writes `sfno_arch.json` next to the checkpoints;
the evaluator refuses a mismatch. Needed because `dropout` is static and
therefore NOT serialised: a p=0.1 checkpoint deserialises cleanly into a p=0.0
skeleton and silently produces a deterministic "ensemble" (or the reverse, a
mis-calibrated one). Every other architecture field would fail loudly on a
leaf-shape mismatch; this one would not. The sidecar is never overwritten — a
resume that changes the architecture raises rather than rewriting the record
under already-trained weights.

### 6. Two pre-existing defects fixed in passing

Both found by adversarial review of the above, both unrelated to U-Cast:

* **`_spectral_state_loss_components`: the q MAE term ignored
  `residual_normalize`**, dividing by 1.0 while T/u/v/ps divided by their
  residual scales. Under a residual-normalized MAE preset that suppresses the
  humidity term by `1/q_resid_scale` ~ **2000x** — humidity effectively
  unsupervised while the config advertises an equal weight. No shipped preset
  set `w_crps_q > 0`, so nothing already measured changes.
* **`run_aimip`'s in-run `sfno_full` evaluation rolled with `clip_q=False` and
  NO spectral filter**, while training uses `clip_q=True` and the suite's
  filter. It scored a different model from the one trained, in the
  configuration whose measured failure mode is 969 m/s winds by macro step 4
  and NaN at step 12. The standalone WB2 evaluator already read both off
  `pe_config`; this lane was missed when the filter landed.

## Arm 1: `config/aimip/scale/suite_sfno_full_ucast_dropout.yaml`

Controlled against `suite_sfno_full_maxdata.yaml` — 2,160 train days, ace2 MSE
preset, AdamW 1e-4, EMA 0.999, 60 epochs, identical eval windows — changing
exactly one field, `sfno_dropout: 0.0 -> 0.1`. A contract test asserts the
single-variable property mechanically and that the variant overlay (merged
LAST, and which has silently shadowed `cfg_overrides` before) does not contain
the key.

Dropout is chosen as the FIRST of the three levers because it is the only one
that unlocks a capability we do not have — a calibrated ensemble from one
network instead of eight training runs — and because it is simultaneously a
regulariser, so it is measurable on the deterministic scorecard too.

Falsifiable before the run: CONFIRMS if deterministic RMSE stays within ~5 %
of baseline AND spread-skill moves materially toward 1; REFUTES if
deterministic RMSE degrades >10 %, or spread-skill does not move (dropout adds
no useful dispersion and the ensemble story needs stage 2 instead). No cheaper
offline test exists — no checkpoint in this repo was trained with dropout.

## Baseline measurement: the mx8 ensemble is UNDER-DISPERSED everywhere

First use of the new instrument, on EXISTING checkpoints — no training. 8 seed
members (`aimip_ace2_sfno_full_t63_allyears` + the 7 `mx4_seed*` EMA
checkpoints), `--probabilistic`, eval year 2017, 6 inits at 144 h stride,
WB2 1.5°, no IC perturbation, no dropout (these nets have none).

| field | lead | RMSE | CRPS | spread | spread-skill |
|---|---|---|---|---|---|
| z500 [m] | 24 h | 32.54 | 15.88 | 17.39 | **0.569** |
| z500 | 72 h | 59.25 | 28.07 | 34.41 | **0.617** |
| t850 [K] | 24 h | 1.99 | 1.075 | 0.922 | **0.491** |
| t850 | 72 h | 2.785 | 1.438 | 1.706 | **0.650** |
| q700 [kg/kg] | 24 h | 1.248e-3 | 6.667e-4 | 6.286e-4 | **0.535** |
| u850 [m/s] | 24 h | 3.794 | 2.084 | 2.059 | **0.576** |
| v250 [m/s] | 24 h | 6.613 | 3.588 | 3.535 | **0.569** |
| mslp [Pa] | 24 h | 464.6 | 215.1 | 175.1 | **0.400** |
| t2m [K] | 24 h | 4.815 | 3.767 | 0.996 | **0.220** |

**CONFIRMED** (direction): every field at every lead is under-dispersed —
spread-skill 0.22-0.70 where 1.0 is calibrated. Consistent across 14 fields x
2 leads, so this is not a noisy single number. Spread-skill improves with lead
(0.57 -> 0.62 for z500), the usual pattern.

**PLAUSIBLE** (levels): 6 inits from 2017-01-01 at 144 h stride is ~30 days of
one season. The exact ratios will move with a full-year sample.

Two consequences:

* **This is precisely the failure mode U-Cast's recipe targets**, and it is
  the first time we could see it — the deterministic scorecard has no column
  in which an under-dispersed ensemble looks different from a calibrated one.
* **Eight independently TRAINED seeds buy only 0.57.** More seeds cannot close
  that gap (spread-skill grows like sqrt of nothing useful when the members
  share a systematic error), and each one costs a full training run. That is
  the argument for the MC-Dropout + CRPS route rather than more deep-ensemble
  members.
* **t2m at 0.22 will NOT be fixed by any ensemble method.** 8 seeds disagree
  by 1.0 K about a field they get wrong by 4.8 K — they are wrong together.
  That is a structural/bias error; do not expect the dropout arm to move it.

PROTOCOL WARNING, do not misread: the RMSE column here is **not** comparable
to `results/bench_mx4/mx8_fullyear.json` (z500 24 h = 30.88 m). That run used
60 inits over the full year AND the mean-then-diagnose reduction; this one
uses 6 inits and diagnose-then-mean. Two variables differ. The table above is
a baseline for the dropout arm under an IDENTICAL protocol, nothing else.

### Instrument controls run before quoting the above

1. Right quantity — CRPS and spread-skill are the quantities the recipe
   optimises, not a proxy.
2. Same transform both sides — CRPS, spread and RMSE all come from the SAME
   diagnosed member fields, same WB2 grid, same above-ground mask, same
   cos-lat weights (lifted from `metrics.rmse`).
3. Same time/config — all three metrics from ONE run, one protocol.
4. Metric proven on a known answer — identical members reduce to exactly the
   MAE (3.0 on the synthetic fixture); a calibrated Gaussian ensemble scores
   spread-skill in (0.8, 1.3); a deliberately under-dispersed one scores
   < 0.3; masked cells are excluded. All asserted in
   `tests/unit/test_ucast_learning_strategy.py`.
5. Reduction supports the claim — per-field, per-lead area-weighted
   aggregates averaged over 6 cases; the DIRECTION is claimed, the levels are
   labelled provisional.

## Second-pass ingredient audit (2026-08-02, both PDFs text-extracted)

Re-read both recipes from the papers themselves (pypdf on the arXiv PDFs;
ArchesWeatherGen arXiv:2412.12971 == Sci. Adv. 10.1126/sciadv.adx2372) and
diffed every ingredient against our lane. Verbatim anchors:

* U-Cast stage 1: "cosine decay with peak learning rates of 3e-3 (Muon) and
  3e-4 (AdamW), and weight decay of 0.1 and 0.03"; stage 2 "8 epochs ...
  learning rates of 7e-3 (Muon) and 7e-5 (AdamW)"; "effective batch size of
  48 and a linear warmup of 1500 steps"; "10% dropout ... EMA of model
  weights (decay 0.9999)"; "same variable-specific loss weights as in (Price
  et al., 2024)" [GenCast]; input = "a window of two past input snapshots",
  predicting the residual w.r.t. the last.
* ArchesWeatherGen: Phase 1 MSE 250k steps; Phase 2 "recent past
  fine-tuning" on 2007-2018 for 50k steps ("forecasting models have a
  [distribution shift] ... less constrained in the past"); Phase 3 optional
  multi-step FT 20k steps (length-2 rollouts) — NOT used for the ensemble
  model; loss = latitude weights x "the same coefficients as GraphCast" per
  variable; generative head: flow matching on Mx4-normalized residuals,
  gamma_s = sqrt(SNR) rebalancing, sigmoid-of-normal timestep sampling, 25
  Euler steps, rho = 1.05 noise scaling, OOD fine-tune 60k steps on 2019.

### Ingredient matrix

| # | Ingredient | U-Cast | ArchesGen | ours (2026-08-02) |
|---|---|---|---|---|
| 1 | MC-Dropout in net, active in training | yes (10%) | — | **ARM RUNNING** (26625679) |
| 2 | MAE stage-1 loss | yes | — (MSE) | **ARM RUNNING** (26625681) |
| 3 | Muon(2-D)+AdamW(1-D) | yes | — | **ARM RUNNING** (26625680), but see #4 |
| 4 | Per-GROUP LR + wd (Muon 3e-3/0.1 vs AdamW 3e-4/0.03) | yes | — | **MISSING** — `_muon_partitioned_optimizer` passes ONE schedule + ONE wd to both branches |
| 5 | Short CRPS fine-tune, ensemble M=2 in the graph | yes (8 ep) | — | **MISSING** (machinery exists, unwired) — top priority |
| 6 | Per-variable (GenCast/GraphCast) loss weights incl. per-LEVEL | yes | yes | **MISSING** — `LossConfig` has no per-level vector; all three references (ACE2 too) use it |
| 7 | Two past input snapshots | yes | yes (x_{t-δ}, x_t) | **MISSING** — channel spec + loader change |
| 8 | Recent-past fine-tuning (2007-2018, ~20% extra steps) | — | yes | **MISSING, NEW FINDING** — cheap: resume maxdata ckpt, fine-tune on the recent third of the window; our 1979-2014-uniform sampling has exactly the distribution shift they correct |
| 9 | Multi-step (rollout) fine-tuning | not needed for ensembles | optional, skipped for gen. model | have (GC arm) — deprioritise per both papers |
| 10 | EMA 0.9999 evaluated | yes | yes (EMA used) | have 0.999; 0.9999 averages ~10^4 updates — LONGER than one of our 60-epoch runs (~5.2k updates), would underweight late training. Keep 0.999 unless step count grows |
| 11 | Residual prediction w.r.t. last input | yes | yes (residual head) | sfno_full trains `residual_prediction=False` (loss residual-normalized, net emits full state) — flag as a candidate arm, untested |
| 12 | Deep ensemble x stochastic members (KxN) | K=4 x N | Mx4 mean | **HAVE** (eval: `--member` x `--mc-dropout-members`) |
| 13 | Probabilistic scorecard (fair CRPS, spread-skill) | yes | yes (fCRPS etc.) | **HAVE** (2026-08-01) |
| 14 | Residual flow-matching generative head + OOD FT + rho=1.05 | — | yes | **MISSING** — the ArchesGen alternative to #5; assessed HIGH-value in the ArchesWeatherGen section above |
| 15 | IC perturbation for dispersion | — | (GenCast EDA-style) | **HAVE** (`--ensemble-ic-noise`, `--ensemble-noise-scale` = their rho) |

Ranked additions to the plan (beyond the three running arms):

1. **#5 CRPS fine-tune** — unchanged top priority, now with U-Cast's exact
   budget: ~8% of stage-1 epochs, M=2, LR ~2x stage-1 (Muon) / 0.25x (AdamW).
2. **#6 per-level loss weights** — the ONLY ingredient all three reference
   models share that we cannot even express. `LossConfig` needs a per-level
   weight vector (z500 x10 / t850 x5 analogue).
3. **#8 recent-past fine-tune** — new, cheap (one suite, resume + window
   subset), directly matched to our 36-year uniform sampling.
4. **#4 per-group Muon LR/wd** — one function (`_muon_partitioned_optimizer`
   grows `adamw_lr_scale` + separate wd). Do AFTER the muon arm reports:
   if Muon-at-1e-4 already helps, the split is the next controlled step.
5. **#7 two past snapshots**, then **#11 residual head**, then **#14
   generative head** (big; only after CRPS fine-tune is measured).

## Parameter accessibility — the CLUBB standard, enforced (2026-08-02)

User directive: every parameterization parameter accessible for tuning, like
CLUBB (whose 48 closure constants are all namelist-tunable). Status before:
the TRAINING collector (`build_trainable_params`) already reached all 686
tier-1/2 registry params — but the RUN-driver `--params` route reached only
265: run_amip's atm route was a hand-verified flat-scalar map, and the
shrink-only reachability baseline carried **421** unreachable params,
*including all 48 CLUBBParams and CLUBBLite* (the exact "like CLUBB" case).

Closed by ONE generic route instead of 421 hand-wirings:
`run_config_yaml.apply_params_to_pipeline` routes qualified `--params`
overrides into the BUILT pipeline's scheme configs (`convection_config` /
`micro_config` / `turbulence_config` / `gwd_config`) POST-setup, PRE-run —
the same sanctioned mutation point as the AIMIP-trained-config injection,
verified consumed at trace time (`config=self.<x>_config` at
physics_pipeline.py:804/1279/1507/1594). run_amip splits a calibration file:
scalar-mapped params pre-setup (unchanged), everything else post-setup.
Validation parity with the class-router path: unknown name, non-numeric,
out-of-bounds, or scheme-not-built all `SystemExit`.

Baseline: **421 -> 106**. Newly reachable: all CLUBB (52), every convection
family (Bechtold's full 12, AhmedNeelin, ZM, EDMF...), all microphysics
(Morrison/Thompson/P3/SBM/SDM...), all GWD. The remaining 106 are classes no
driver config tree carries: atm.rad + atm.clouds built INLINE from flat
scalars (12+8), ocean.eke (20) / ocean.sf (11) absent from run_omip's tree,
land.canopy (11) etc. — each still listed per class in the regenerated
baseline. The audit now probes the production bundle+router over EVERY
registered scheme selection, so a new scheme's params are audited
automatically.

Also fixed in passing: `cloud_partial_coverage_optics` sat in the scalar map
but had been dropped from the registry (#1280 exclusion drift) — the
map-contract test KeyError'd on it; removed with the documented NOTE pattern.

## Stage 2 landed (2026-08-02): the ensemble-CRPS fine-tune

Ingredient #5, the top-ranked gap, is now implemented — built while the three
stage-1 arms train, so it is ready for the dropout arm's checkpoint.

`ensemble_crps_loss_components` scores M MC-Dropout rollouts with the
almost-fair CRPS. Two deliberate reuses so nothing can drift: the pointwise
estimator is `ml.loss.almost_fair_crps` (the SAME function
`evaluations.wb_forecast.score_ensemble_forecast` scores with), and the
area/level weighting plus the residual>full>none scale selection are copied
from `_spectral_state_loss_components`, so a stage-2 CRPS is commensurate with
the stage-1 MAE. Per-variable weights come from the `w_crps_*` family, whose
M=1 limit IS that MAE — the `ucast` preset carries over unchanged.

Config: `crps_finetune_epochs` (0 = off) + `crps_ensemble_size` (default 2,
U-Cast's M). The last N epochs switch objective; the flag rides in the phase
spec so the per-phase JIT cache keys on it (the CRPS graph is M vmapped
rollouts and must not reuse the deterministic compilation).

Every guard here refuses a SILENT no-op rather than a crash — each was a real
defect found in review, not a hypothetical:

* `sfno_dropout == 0` -> every member identical, CRPS degenerates to the MAE;
* `crps_ensemble_size < 2` -> same degeneracy (and an explicit `0` must not be
  silently defaulted to 2, which an `or 2` did);
* `crps_finetune_epochs >= n_epochs` -> not a fine-tune of a pre-trained model;
* all `w_crps_*` zero -> constant-zero loss, exactly-zero gradient, a
  fine-tune that trains nothing;
* early stopping is SUPPRESSED whenever stage 2 is configured — otherwise a
  stage-1 plateau ends the run before the CRPS epochs and emits a
  deterministic checkpoint labelled as a CRPS run;
* a non-`sfno_full` variant with the knob set is REFUSED in `run_aimip`: those
  dispatch to `_train_spectral_loop`, which never reads it.

Two bugs the review caught that would have trained silently-wrong models:

1. **Horizon mismatch.** The non-curriculum path first rolled
   `segment_steps[-1]` while scoring `target_carry[-1]`: `segment_steps` are
   INCREMENTAL, so with leads [6, 12] it advanced 6 h and compared against the
   12 h target. The fix does NOT use `sum(segment_steps)` either — those are
   per-increment ROUNDED counts, so their sum is only the final lead when
   every increment is on the dt_sfno grid. It derives
   `round(final_lead_h*3600/dt_sfno)` and refuses an off-grid final lead. This
   is strictly stronger: leads [6, 12] at dt_sfno=4 h have off-grid increments
   (1.5 -> [2,2] = 16 h) but an exact final lead (3 steps), and now train
   CORRECTLY rather than being rejected. Guard scoped to non-curriculum runs,
   which supply their own exact per-phase step count.
2. **Dead knob.** `run_aimip._build_spectral_config` did not forward either
   field, so a suite setting `crps_finetune_epochs` ran pure stage 1 and
   logged nothing — the same failure mode as the overlay-shadowed
   `sfno_embed_dim` in 2026-07. Now forwarded, with a round-trip test through
   the real builder.

Non-vacuity: the end-to-end test runs 3 epochs with the last on CRPS and
asserts the weights differ from the stage-2-off run at the same seed; and the
estimator test asserts a bracketing ensemble beats its own collapsed mean —
with the truth as a DRAW from the predictive distribution, not its mean (the
first version put the target on the collapsed forecast, where CRPS 0 is the
correct answer and the test was asserting the estimator broken when it was
right — the same calibration trap as the spread-skill fixture).

Review: 3 codex rounds (5 + 4 + 1 findings), all closed.

NOT yet run: stage 2 needs a dropout-trained stage-1 checkpoint, which arm 1
produces tonight. No suite ships it yet — that is the next arm, and it is a
FINE-TUNE (resume from arm 1, ~8 % more epochs), not a fresh run.

## RESULTS: the three stage-1 arms, measured (2026-08-03)

Protocol held FIXED on every row: `epoch_0057` (the last epoch all four arms
share), EMA weights, eval year 2017, 60 inits at 144 h stride, WB2 1.5 deg,
area-weighted RMSE. Each arm differs from the baseline in exactly ONE
`cfg_overrides` field, enforced by a contract test.

z500 RMSE [m] (persistence floor in brackets):

| arm | 24 h | 72 h | 120 h | 240 h |
|---|---|---|---|---|
| persistence | 61.65 | 94.78 | 105.6 | 113.9 |
| baseline (adamw, MSE, p=0) | 21.72 | 46.57 | 71.89 | 106.3 |
| dropout 0.1 | 18.41 | 43.27 | 68.35 | **119.4** |
| **muon_partitioned** | **17.16** | **41.91** | **67.57** | **103.7** |
| mae preset | **40.8** | 55.51 | 76.1 | 105.8 |

Cells beating the baseline, over all 16 fields x 4 leads:
**muon 60/64, mae 50/64, dropout 35/64.**

### 1. Muon wins outright — and it was HANDICAPPED

Better at every lead on z500, and on 60 of 64 field-lead cells. U-Cast's
ablation transfers: they report AdamW costing "up to 15 %" z500 CRPS; we
measure -21 % RMSE at 24 h from the optimizer swap alone.

Crucially this ran at the BASELINE lr 1e-4 for both branches, deliberately,
to keep the arm single-variable. U-Cast runs Muon at 3e-3 with the AdamW
1-D group at 3e-4 — a 10x ratio our `_muon_partitioned_optimizer` could not
express (one schedule, one weight decay for both). That is now implemented
(`muon_lr_scale` / `adamw_lr_scale` / `*_weight_decay_scale`), so the measured
-21 % is a FLOOR, not the recipe's result.

### 2. Dropout helps short leads, breaks day 10

-15 % at 24 h but **+12 % at 240 h**, where it drops below the persistence
floor. The degradation is systematic, not noise: t850 +43 %, t2m +42 %,
q700 +16 %, mslp +15 % at 240 h. A regulariser that trades long-lead
stability for short-lead accuracy.

### 3. MAE is not uniformly bad — it is bad for z500 and mslp specifically

+88 % z500 and +62 % mslp at 24 h, yet 50/64 cells still beat the baseline
and t850 / q700 / winds / t2m are comparable-to-better. The damage is
concentrated in the two fields that are smooth hydrostatic integrals of the
state. PLAUSIBLE mechanism (median-vs-mean matters most where the error
distribution is closest to Gaussian and the field is large-scale); NOT
confirmed — no perturbation test was run.

This was the pre-registered expectation ("MSE targets the conditional MEAN
and is the RMSE-optimal deterministic loss ... a large RMSE IMPROVEMENT would
be a surprise"), so it does not disqualify MAE from its actual job: being the
matched initialisation for a CRPS fine-tune. Whether MAE-init -> stage 2
beats MSE-init -> stage 2 is UNTESTED and is the only question that settles
it.

## RESULT: MC-Dropout alone does NOT give a calibrated ensemble

8 MC-Dropout members from ONE net vs 8 independently trained seeds, both
`--probabilistic`, 60 inits, identical leads and reduction:

| spread-skill (1.0 = calibrated) | 24 h | 72 h | 120 h | 240 h |
|---|---|---|---|---|
| MC-Dropout x8 (one net) | 0.229 | 0.155 | 0.180 | 0.426 |
| 8 seeds | 0.582 | 0.621 | 0.750 | 0.909 |

Pre-registered REFUTE condition ("spread-skill stays <= the 8-seed value")
is met by 3-5x, at every lead and every field.

The data-tier confound STRENGTHENS this: the dropout arm is the 2,160-day
model, the seed control the 288-day one, so the dropout net has lower RMSE —
which would RAISE its spread-skill at equal spread. It is lower anyway, so
the actual spread must be far smaller. (The CRPS and RMSE columns of that
same comparison ARE confounded by data volume and no conclusion is drawn
from them.)

This is the recipe read correctly, not a failure of it: in U-Cast dropout is
the stochasticity SOURCE and the CRPS fine-tune is what teaches the network
to use it — "the model learns to leverage dropout masks". Stage 1 alone
leaves dropout as a mere regulariser. **Stage 2 is REQUIRED, not optional**,
and its target is now a pre-registered number: beat 0.58 / 0.62 / 0.75 / 0.91.

### Consequence for the next arm

Stage 2 needs `sfno_dropout > 0` (the ensemble source), and only the dropout
arm has it — the muon arm, the best stage-1 model, has p=0. The next arm is
therefore **muon + dropout together** as a deliberate NEW BASELINE (labelled
as such, never compared term-by-term against the single-knob rows), followed
by the short CRPS fine-tune from that checkpoint.

## Still open, in priority order

1. **Stage-2 ensemble-CRPS fine-tune** in `_train_sfno_full_loop` (M=2, ~8 % of
   the pre-training epochs, per U-Cast). This is the actual probabilistic
   training step; everything above is the scaffolding it needs.
2. **Muon arm** — one config line, their largest optimizer effect, unmeasured
   here.
3. **MAE arm** — `loss_preset: ucast`, ready to run.
4. **Two input snapshots** — channel-spec + loader change.

## HANDOFF (2026-08-05) — state at context compression

### Simulations: 6 arms trained + scored, nothing running

z500 RMSE [m], ALL at epoch_0057, EMA, eval 2017 / 60 inits / 144 h stride /
WB2 1.5 deg. Scorecards in `results/bench_ucast/<arm>_ep0057_ema.json`.

| arm | 24 h | 72 h | 120 h | 240 h | cells>base |
|---|---|---|---|---|---|
| persistence | 61.65 | 94.78 | 105.6 | 113.9 | — |
| baseline (maxdata: adamw, MSE, p=0) | 21.72 | 46.57 | 71.89 | 106.3 | — |
| dropout 0.1 | 18.41 | 43.27 | 68.35 | 119.4 | 35/64 |
| muon_partitioned | 17.16 | 41.91 | 67.57 | 103.7 | 60/64 |
| mae preset | 40.8 | 55.51 | 76.1 | 105.8 | 50/64 |
| muondrop (muon+dropout) | 18.88 | 44.08 | 69.02 | 107.4 | 57/64 |
| **muonlr (muon_lr_scale 10)** | **16.35** | **39.65** | **65.79** | **100.2** | **64/64** |

Verdicts vs their PRE-REGISTERED criteria:
* muon CONFIRMED; muonlr CONFIRMED (the LR RATIO is worth another -4.7 % at
  24 h on top of the optimizer swap; cumulative -24.7 % vs baseline).
* dropout CONFIRMED short-lead, REFUTED at day 10 (below persistence).
* mae REFUTED for z500/mslp (+88 %/+62 % at 24 h), neutral-to-better on
  t850/q700/winds/t2m — as predicted, MSE is the RMSE-optimal loss.
* muondrop REFUTED: 18.88 vs muon's 17.16, so the two do NOT compose;
  dropout costs ~half of Muon's gain. Muon DOES cure dropout's day-10
  collapse (119.4 -> 107.4).

### MC-Dropout does NOT give a calibrated ensemble (the decisive result)

spread-skill (1.0 = calibrated), 8 MC-dropout members from ONE net vs 8 seeds,
both --probabilistic, 60 inits:

| | 24 h | 72 h | 120 h | 240 h |
|---|---|---|---|---|
| MC-dropout x8 | 0.229 | 0.155 | 0.180 | 0.426 |
| 8 seeds | 0.582 | 0.621 | 0.750 | 0.909 |

REFUTE condition met by 3-5x. The data-tier confound STRENGTHENS it (the
dropout net has lower RMSE, which would RAISE spread-skill at equal spread).
Read correctly this is the recipe, not a failure of it: dropout is the
stochasticity SOURCE, the CRPS fine-tune teaches the model to use it.
**Stage 2 is REQUIRED, not optional.**

### NEXT RUN (decided, not submitted)

Stage 2 CRPS fine-tune resuming IN PLACE from muondrop:
`n_epochs: 65`, `crps_finetune_epochs: 5` (epochs 60-64 switch objective).
Pre-registered target: beat spread-skill 0.58 / 0.62 / 0.75 / 0.91.
Base is muondrop (18.88) not muonlr (16.35) because stage 2 requires
`sfno_dropout > 0` and muonlr has p=0 — a known ~15 % deterministic cost,
accepted because stage 2 answers a CALIBRATION question.

### Infra fixes that paid off (keep)

* `AIMIP_EVAL_TRAIN_MAX_DAYS=300` gate — the post-training TRAIN-period eval
  reloaded all 9,504 snapshots (~4.2 h) BEFORE params.eqx was written; it
  drained 3 arms + maxdata. Gate fired in production; both round-2 arms
  completed with params.eqx.
* `_env.sh` guards: missing LEGOESM_REPO and a jax-less PY now fail at source
  time (two arm submissions died 7-10 s in without them).
* `sbatch --export` splits on commas: `ONLY=a,b` arrives as `ONLY=a`. One job
  per arm.
* Arm cost: ~4.3 h ERA5 load + 60 epochs @ ~400 s = 2 links. Scoring ~2 h/arm.

### Scheme-agnostic trained params — WIRED BUT DISABLED

`aimip_trainable_schemes: "extended"` builds an `AIMIPTrainableBundle`
(legacy `AIMIPClassicalParams` + spec-driven `TrainablePhysicsParams` for the
ACTIVE schemes). Keys derived from the BUILT CONFIG TREE via
`aimip_active_scheme_keys` / `aimip_scheme_keys_for` — generalising the
pre-existing `run_scm_les_turbulence_tuning._scheme_keys_for` (the mechanism
was NEVER lost, only turbulence-only and SCM-only). Gradient test PROVES
non-zero grads on CLUBB and Bechtold leaves. edmf+louis = 46 leaves,
bechtold+clubb = 102.

**OFF in `config/aimip/clubb_bechtold/variant_classical.yaml`** — codex round 2
found 3 HIGH defects. MUST fix before enabling:
1. Generic overrides splice OVER fields the legacy `to_<scheme>_config` set,
   DISCONNECTING the 46 legacy leaves (zero gradient). Enabling silently
   trades 46 working leaves for 80 new ones. DESIGN DECISION NEEDED: when both
   cover a field, the spec leaf should win AND the legacy ParamConstraint be
   dropped so exactly one thing trains it — but that changes legacy checkpoint
   layout, so it is the user's call.
2. `run_aimip` resumes into a bare `AIMIPClassicalParams` BEFORE the wrap;
   Equinox accepts a prefix template, so a bundle checkpoint restores the
   classical prefix and REINITIALISES `schemes` every chain link.
3. Composite GWD (`"mcfarlane+hines"`, the executor splits on `+`) and the
   `atm.sdm` / `atm.fastsbm` microphysics namespaces derive NO keys.
Plus MEDIUM: nested active configs (E3SM CAM orographic/frontal/beres,
radiation ozone/cloud) skipped; `spatial_lr_scale` silently ignored because
`getattr(model, "spatial_surface")` is False on the bundle.

The CLUBB+Bechtold arm is still usable as-is: legacy leaves train as today,
CLUBB/Bechtold at scheme defaults, and CLUBB's 48 constants are calibratable
offline via `--params`.

### Issue #1464 (WB neural_gcm degrades 12/16 fields) — investigated, NOT posted

Codex PARTIALLY CONFIRMED my hypothesis and REFUTED two parts. RETRACTED:
"no momentum sink at all" (dycore hyperdiffusion 1e15 + spectral filter 0.01
act on vor/div) and the zero-`lnps` differential (classical radiation/
convection/turbulence ALSO return zero lnps; p_s evolves via dycore
continuity in both arms). CONFIRMED: the adapter returns identically zero
vor/div AND uses ONLY the T and q_v heads, discarding the network's u/v,
precipitation, condensate and flux heads; classical Smagorinsky IS active with
surface stress. z500 improves because it is diagnosed hydrostatically from
T/q/p_s — exactly what the adapter corrects — which my mechanism cannot
explain and should not be stretched to. Cheapest test (eval-only, existing
checkpoints): score native / +Smagorinsky-momentum-only / +full Smagorinsky,
auditing the injected tendency shows a real negative low-level KE tendency
first.

### Uncommitted, and a concurrent session already cost us once

A `pre-main-sync 2026-08-04_1305` stash from another session REVERTED
`run_config_yaml.py` (deleting `apply_params_to_pipeline` while keeping its
callers) — recovered, codex-CLEAN. Everything here is still uncommitted.

## ITERATION 2026-08-08 — the three arms are not close, and the reason is mass

### The 3-arm table that started this (8 inits x 24 h, 2017, WB2 1.5 deg)

z500 RMSE [m] / area-weighted bias, and the same for mslp [Pa]:

| arm | z500 24 h | z500 240 h | mslp 24 h | mslp 240 h |
|---|---|---|---|---|
| persistence | 63.5 | 119.5 | — | — |
| ERA5 annual-mean climatology | 122.8 | 134.0 | — | — |
| classical | 95.8 / -38.2 | 266.8 / -215.6 | -202 | **-1.64e3** |
| column_nn | 90.9 / -7.6 | 214.9 / -84.2 | -227 | **-1.63e3** |
| sfno_full (allyears) | 38.9 / +0.3 | 103.6 / +5.2 | -56 | -84 |

Published references in geopotential HEIGHT metres (the WB2 CSV is m^2/s^2;
divide by g before comparing): GraphCast 4.1 / 74.6, IFS-HRES 4.8 / 81.8,
NeuralGCM 3.9 / 76.6 at 24 h / 240 h.

**Both dycore arms are BELOW PERSISTENCE at every lead.** classical is also
below the annual-mean climatology from day 5 on.

### What the per-field breakdown says (this is the finding)

The WIND fields are nearly identical across all three arms — u850 at 24 h is
4.80 / 4.70 / 4.28 m/s for classical / column_nn / sfno_full, and the same
holds at 700, 500 and 250 hPa and for u10/v10. The arms separate almost
entirely in the MASS fields: z500, mslp, t850, t2m.

classical's day-10 z500 error is 65% BIAS (-215.6 m of a 266.8 m RMSE), and a
-16.4 hPa surface-pressure drop alone displaces z500 by roughly
(16.4/1013) x 5500 ~ 90 m. The bias is cos-lat AREA-WEIGHTED — checked in
`evaluations/wb_forecast.py:233,257`, which builds `weights = cos(deg2rad(lat))`
and passes them to `metrics.bias` — so this is not the unweighted-mean trap.

### Mechanism, PLAUSIBLE, with the discriminating test now running

`SpectralPEConfig.fix_mass` and `anchor_mass_to_initial` both default to False
(`spectral_pe.py:296-297`) and `run_aimip._build_spectral_config` never passed
either one, so the dry-mass anchor has been OFF on every AIMIP arm ever trained
or scored — verified by building the real config for all three arms. The arm
that does NOT drift is the one with no dycore: `sfno_full`'s physics explicitly
zeroes the (l=0,m=0) coefficient of its dlnps/dt ("DRY-MASS constraint",
`neural_gcm_spectral.py:764-774`).

This is PLAUSIBLE, not confirmed. mslp is a REDUCED pressure, so a cold bias
over topography inflates it — though the scale is wrong for that to be the
whole story (a -5 K bias at 1500 m elevation moves mslp by ~0.6 hPa, not 16).
The A/B under way is eval-only and controlled: the SAME column_nn checkpoint,
the SAME 8x24 h protocol, `suite_curriculum_v2_massfix.yaml` differing from
`suite_curriculum_v2.yaml` in exactly the two anchor flags (asserted by
`tests/unit/test_aimip_mass_anchor_knob.py`).

CONFIRMS: mslp bias at 240 h collapses toward sfno_full's -84 Pa.
REFUTES: bias unchanged -> not a global-mass mode; measure area-weighted p_s
directly before proposing anything else.

Caveat to carry: anchoring the FORECAST while the model was TRAINED unanchored
is a train/eval mismatch. If it confirms, the fix belongs in training.

### Also found: the published classical checkpoint no longer loads

`results/aimip_ace2curr_most_long/classical/params.eqx` fails
`eqx.tree_deserialise_leaves` on the current checkout — the eval skeleton wants
`spatial_surface.fields['Cd_neutral'].coeffs` of shape (13,) and the file holds
a scalar (job 26792652). The spatial-surface field became a 13-coefficient
basis after that arm trained. The 95.78 m row above therefore cannot be
reproduced on today's code; `suite_classical_v3.yaml` retrains the identical
recipe into a clean dir purely to get a loadable classical checkpoint.

### Submitted this iteration

| job | what | why |
|---|---|---|
| 26792653/54 | column_nn + sfno_muonlr rescored at 60 inits x 144 h | the U-Cast table and the 3-arm table were on DIFFERENT protocols; that is a confound, not a result |
| 26792691 | sfno stage-2 CRPS fine-tune (`suite_sfno_full_ucast_crps.yaml`) | the decided-but-never-submitted run from the 2026-08-05 handoff |
| 26792703 | column_nn + Muon @ 10:1 LR ratio (`suite_curriculum_v2_muonlr.yaml`) | one variable; worth -24.7% on sfno_full, and column_nn's MLP is the leaf class Muon is for |
| 26792704 | column_nn at 288 train days (`suite_curriculum_allyears.yaml`) | one variable; data is the only lever that ever moved an arm >few % (sfno 89.5 -> 38.9 m) |
| 26792713->16 | classical retrain, current code (`suite_classical_v3.yaml`) | the old checkpoint does not load |
| 26792784 | the mass-anchor A/B | above |

Muon is deliberately NOT tried on classical: `muon_partitioned` routes 1-D
leaves to the AdamW branch, and classical's 124 knobs are scalars, so
`muon_lr_scale` would be an exact no-op there.

### Independent strategy review (GLM-5.2), filtered

Useful: (a) rank the mass anchor first — matches the measurement; (b) the
cheapest sfno lever is a SECOND input snapshot (t-6 h), which doubles input
channels for negligible FLOPs and is what U-Cast itself feeds; (c) unbalanced
ERA5 initial conditions fed to a spectral dycore radiate spurious gravity
waves and would show up at 24 h — untested here, cheap to test with a digital
filter on the IC.

REJECTED: its answer on our MAE result read "MAE" as Masked Autoencoding and
explained the z500 degradation as a masking artifact. In U-Cast and in our arm
MAE is Mean Absolute Error. The explanation is void; the measurement stands
(MSE is the RMSE-optimal loss, so an L1 stage-1 costing z500 is expected).

### RETRACTION + the real fix (same day, 2026-08-08)

**I was wrong earlier in this section.** Setting `fix_mass` in the config did
NOT enable the anchor for the AIMIP arms. `_apply_mass_fixer` lives only on
`SpectralPrimitiveEquationModel`, and AIMIP — training AND both evaluators —
integrates through the FUNCTIONAL `neural_gcm_spectral.spectral_rollout`, which
never called it. The first A/B job would have produced a byte-copy of the
baseline and been read as "the anchor does not help". Caught by codex, not by
me: exactly the PROVE-THE-PATH-EXECUTES gate.

The fix now lives where the forecast runs. `spectral_pe.py` gained two free
functions — `global_dry_mass(grid, lnps_hat)` and
`anchor_lnps_to_mass(grid, state, target_mass)` — and the class methods
delegate to them (no second copy of the mass integral). `spectral_rollout`
captures its own initial-state mass and re-anchors after the filter chain.
Two details that matter:

* `spectral_rollout` has TWO step bodies and the arms take DIFFERENT ones:
  classical is rad-gated (`aimip_rad_update_interval` 36), column_nn is not.
  Both are wired, both are tested.
* the target is captured per rollout, i.e. per forecast case, so it cannot go
  stale across WB2 cases the way the class-side `_target_mass` does (it is only
  ever set when `None`).

Tests: `tests/unit/test_spectral_rollout_mass_anchor.py` — mass held to <1e-3
ppm in both bodies, off-by-default byte-identical, jit+grad safe, and a
NON-VACUITY case proving the unanchored rollout really does leak (the spectral
filter damps non-mean `lnps` modes, which moves ∫exp(lnps)dA even though its
l=0 multiplier is 1 — codex's point, and a leak the anchor closes).

Two pre-existing failures confirmed NOT mine by re-running with the change
stashed: `test_anchor_mass_api::test_anchored_step_supports_jax_grad_spectral_pe`
and the two `TestMidEpochResume` cases.

### Supporting evidence found while fixing it

`sfno_pe.SFNOPrimitiveEquationModel` has `correct_mass: bool = True` by
default and applies `correct_dry_air_mass` post-step. So the ONE arm that does
not drift is the one arm with a mass corrector switched on. That is the
asymmetry, stated in code rather than inferred.

The mslp-reduction alternative is REFUTED by arithmetic: the reduction is
`p_s (1 + Γz_s/T0)^(g/RΓ)` (`headline_diagnostics.py:115`), so a COLD bias makes
mslp HIGHER, not lower — +109 Pa at 500 m elevation and +350 Pa at 1500 m for
-5 K, and exactly 0 over ocean. Observed bias is -1.64e3 Pa with the opposite
sign, so the reduction cannot produce it and the true p_s deficit is if
anything LARGER than 16 hPa.

### Known, accepted: stage 2 restarts the optimizer schedule

The CRPS fine-tune resumes MODEL leaves only; the SFNO loop builds a fresh
optimizer state and a fresh warmup+cosine sized for all 65 epochs
(`neural_gcm_spectral.py:4318`). Resuming at epoch 60 therefore runs stage 2 at
~peak LR (1e-4) rather than the stage-1 cosine tail. NOT designed, but it
points the same way U-Cast does — their stage 2 uses a HIGHER LR than stage 1
(7e-3 vs 3e-3) — so the run stands and the LR is documented rather than
silently attributed to the CRPS objective.

### Open question now under measurement: is the sfno error made in step 1?

sfno_full (muonlr) is 4.0x worse than GraphCast at 24 h (16.35 vs 4.05 m) but
only 1.34x worse at 240 h (100.2 vs 74.6 m). A deficit concentrated at SHORT
lead is the signature of missing tendency information — the model sees ONE
snapshot and must infer d/dt from spatial structure, where U-Cast and GraphCast
both feed TWO past snapshots. Job 26793246 scores the muonlr arm at leads
6/12/24/72 to see how much of the 24 h error already exists at 6 h. Neither
`channel_packing` nor `ucast_pe` has any history support today, so a two-input
arm is a real multi-file change and is NOT started until this says it is worth
it.

### column_nn: Muon does NOT transfer, and the arm is UNDERFITTING

Matched epoch-for-epoch AND lead-for-lead (v2 baseline
`logs/wb2/wb2_cnn_26505703.out`, Muon arm `uni_train_26792703.out`; identical
curriculum, identical 20-day data, the ONLY difference being
`aimip_optimizer: muon_partitioned` + `muon_lr_scale: 10`):

| epoch (lead) | v2 AdamW | muon 10:1 |
|---|---|---|
| 0 (6 h) | 24.184 | 24.247 |
| 1 (6 h) | 24.102 | 24.134 |
| 2 (12 h) | 25.496 | 25.531 |
| 3 (12 h) | 25.374 | 25.446 |
| 4 (12 h) | 25.363 | 25.418 |
| 5 (24 h) | 29.319 | 29.572 |

Muon is worse at EVERY matched point, by 0.2-0.9 %. PLAUSIBLE, not final —
this is training loss at 6 of 11 epochs, and the WB2 scorecard is the verdict.
But the direction is consistent and the sign is not what the sfno_full result
(-24.7 % on z500) predicted. Plausible reason: Muon's orthogonalised update is
a large-2-D-matrix method, and the column MLP is 216k parameters of small
matrices applied per column.

The more useful reading of that table is what BOTH columns do: each curriculum
phase plateaus within two epochs and then oscillates (v2 at 120 h goes 85.02 ->
83.25 -> 84.12). The TRAINING loss is stuck. That is underfitting, so more
held-out data is not the first thing column_nn needs.

**Scale context nobody had written down: column_nn is 216,614 parameters
against sfno_full's 68.3M — 315x fewer — and the two are being asked to land in
the same place on the scorecard.**

New arm (26793606, `suite_curriculum_v2_wide.yaml`): `nn_hidden_dim` 256->1024,
`nn_n_layers` 4->6, ~6.3M params, nothing else changed. Near-free in wall clock
because each step is dominated by the dycore + RRTMGP, not the MLP.
CONFIRMS: training loss at matched epoch/lead below v2's AND z500 24 h below
90.9 m. REFUTES: loss unchanged -> the plateau is optimization
(LR / epochs-per-phase), not capacity.

### The stopping criterion needs stating: ACE2 is not scored on WB2 RMSE

"Close to ACE/ACE2" and "close to the WB2 leaderboard" are two DIFFERENT
targets and this campaign has been measuring only the second.

ACE2's own headline is a CLIMATE score, not a forecast one — its checkpoints
are selected on "channel-mean global RMSE of TIME-MEANS from eight 5-year
inference runs, re-selected across 4 seeds using twelve 5-year runs"
([[ace2_learning_strategy_gap]] line 24), i.e. climatology fidelity and
multi-year drift. It publishes no z500 24 h deterministic RMSE. GraphCast,
IFS-HRES, NeuralGCM and Pangu — the rows in
`config/wb/sota/wb2_headline_rmse.csv` — are the ones a WB2 forecast scorecard
can be compared against.

So the honest reading of where the arms stand is: the WB2 scorecard answers
"how far from GraphCast/IFS", and a separate AIMIP climate scorecard (5-year
time-means + drift) is what answers "how far from ACE2". Both are in scope for
this campaign; only the first has been run.

### The LAST checkpoint is not the best one: epoch 57 -> 59 costs 24 % on z500

First 60-init scorecard back (`results/bench_wb2_p60/sfno_full_muonlr_scorecard.json`).
Controlled pair — same arm, same weights family (EMA), same protocol (2017,
60 inits, 144 h stride, WB2 1.5 deg); the ONLY difference is which epoch:

| checkpoint | 24 h | 72 h | 120 h | 240 h |
|---|---|---|---|---|
| epoch_0057_ema (the U-Cast table row) | 16.35 | 39.65 | 65.79 | 100.16 |
| params_ema = epoch_0059_ema (the FINAL) | 20.26 | 43.00 | 67.18 | 101.10 |

Verified `params_ema.eqx` is byte-identical to `epoch_0059_ema.eqx`, so this is
two epochs of real degradation: **+24 % at 24 h**, shrinking to +0.9 % at 240 h.

Three consequences:

1. **Every arm's headline uses its FINAL checkpoint** — `bench_wb2_levante.sbatch`
   prefers `params_ema.eqx` — so all of them may be past their best, and the
   short leads are where it costs most.
2. `aimip_patience: 10` early stopping never fired, i.e. the training
   VALIDATION LOSS did not degrade while WB2 z500 at 24 h got 24 % worse. The
   objective and the scorecard disagree about which epoch is best.
3. This is the same problem ACE2 solved by selecting checkpoints on a climate
   score computed from inference runs rather than on validation loss
   ([[ace2_learning_strategy_gap]]). We take the last epoch.

NOT actionable by scoring 2017 and picking the winner — 2017 is the WB2 TEST
year and choosing on it is leakage. A selection rule has to run on the suites'
own `eval_windows` (2015/2016 in the maxdata tier). Left as the next
checkpoint-selection item, deliberately not bolted on here.

Bias check on the same file, for the mass story: sfno_full's z500 bias is
-2.0 -> -3.3 m and its mslp bias -58.7 -> -61.7 Pa across 24->240 h, i.e. FLAT.
The dycore arms went -202 -> -1640 Pa on the same metric. The contrast is not a
protocol artifact.

## MASS ANCHOR: CONFIRMED (job 26793199, 2026-08-08)

Same checkpoint (`results/aimip_ace2curr_v2/column_nn/params.eqx`), same
protocol (2017, 8 inits, 24 h stride, WB2 1.5 deg), the ONLY difference being
`fix_mass` + `anchor_mass_to_initial`:

| field / lead | anchor OFF | anchor ON |
|---|---|---|
| mslp bias 24 h | -227 Pa | -62.8 Pa |
| **mslp bias 240 h** | **-1.63e3 Pa** | **-41.9 Pa** |
| mslp RMSE 240 h | 2.46e3 Pa | 1.87e3 Pa (-24 %) |
| z500 RMSE 240 h | 214.9 m | 202.0 m (-6.1 %) |
| z500 bias 240 h | -84.2 m | +37.9 m |
| t850 RMSE 240 h | 6.80 K | 6.50 K (-4.4 %) |
| u850 / v850 / t2m, all leads | — | unchanged |

The mslp bias is not merely smaller, it is **flat with lead** (-63/-41/-41/-42
Pa at 24/72/120/240 h) instead of secular. That is the signature of the drift
being removed rather than damped, and it beats the pre-registered CONFIRM
target (sfno_full's -84 Pa). Winds and t2m are untouched, exactly as a
correction to the `lnps` global mean should be.

So the chain holds end to end: the two arms sharing the spectral dycore lost
global dry mass, `spectral_rollout` had no fixer, and restoring ∫p_s dA each
step removes the bias and takes 24 % off mslp RMSE and 6 % off z500 RMSE at day
10 with ZERO retraining.

z500's bias flips -84.2 -> +37.9 m: anchoring p_s removes the mass term and
leaves the THICKNESS term, which is positive. Not over-correction of the
anchor — a different, smaller error now visible underneath.

### What was launched on the back of it

The A/B anchored a model TRAINED unanchored, i.e. physics that had learned to
live with a drifting p_s. Both dycore arms are now retraining under the
constraint (`suite_curriculum_v2_anchored.yaml` job 26794955,
`suite_classical_anchored.yaml` job 26794956).

Two pending arms were CANCELLED to make room, on evidence rather than
impatience: `cnn_allyears` (the 288-day data tier — column_nn's training loss
is stuck, so it is underfitting and more held-out data is not its constraint)
and `cls_v3` (the unanchored classical retrain — superseded by the anchored
twin, and the unanchored reference already exists as the published 95.78 m row).

### Codex round 2 found exactly one thing, and it was real

`spectral_amip_rollout` — the PRESCRIBED-SST lane used by
`run_aimip_amip_inference.py` and `run_aimip_amip_finetune.py` — has its own
step body and was still bypassing the anchor, and neither runner forwarded the
two flags into `SpectralPEConfig`. Fixed in all three places, and the test file
now covers that body too (7 tests). Everything else in the round-2 checklist
(numerics of the sqrt(4pi) correction, jit/grad safety, the class refactor
being behaviour-preserving, the suite A/B pairs, the plot race) came back clean.

## THE ONE-PROTOCOL TABLE (2026-08-08) — and what it reframes

z500 RMSE [m], eval 2017, 60 inits, 144 h stride, WB2 1.5 deg. SOTA rows are
`config/wb/sota/wb2_headline_rmse.csv` divided by g (the CSV is m^2/s^2; our
`z500` is geopotential HEIGHT).

| row | 24 h | 72 h | 120 h | 240 h |
|---|---|---|---|---|
| GraphCast | 4.06 | 12.66 | 27.97 | 74.64 |
| NeuralGCM | 3.87 | 11.80 | 27.24 | 76.63 |
| IFS-HRES | 4.81 | 13.99 | 31.15 | 81.86 |
| **sfno_full muonlr (epoch 57)** | **16.35** | **39.65** | **65.79** | **100.16** |
| sfno_full muonlr (final epoch) | 20.26 | 43.00 | 67.18 | 101.10 |
| persistence | 61.65 | 94.78 | 105.65 | 113.88 |
| **column_nn v2** | **89.25** | 119.77 | 148.75 | 205.16 |
| ERA5 annual-mean climatology | 107.94 | 107.17 | 107.53 | 107.04 |
| classical most_long (8-init protocol) | 95.78 | 146.46 | 186.85 | 266.84 |

Read it in two halves.

**sfno_full is a real forecast model.** It beats persistence at every lead, is
4.0x GraphCast at 24 h and only 1.34x at 240 h. The deficit is concentrated at
SHORT lead.

**Both dycore arms are worse than persistence at every lead**, and worse than
the annual-mean climatology from day 5. column_nn's mslp bias still runs
-218 -> -1591 Pa here (this checkpoint was scored unanchored).

### The reframing: the physics package is NOT what separates the dycore arms

classical carries RRTMGP, Bechtold, CLUBB, McFarlane, Sundqvist and 124 tuned
knobs. column_nn carries a 216k-parameter MLP and nothing else. At 24 h they
score **95.8 m and 89.3 m** — within 7 % of each other, and both ~5.5x worse
than sfno_full. Two completely different physics packages landing in the same
place points at what they SHARE: the spectral dycore and how an ERA5 state
enters it.

PLAUSIBLE, and the cheap discriminator is error-onset timing rather than
another training arm. If the ~90 m is already most-there at 6 h, the error is
made immediately — initial-condition imbalance radiating gravity waves, or a
dycore/coordinate mismatch — and no amount of physics tuning touches it. If it
grows smoothly from small, it is accumulating model error and the physics does
matter. Jobs 26793246 (sfno_full) and 26795248 (column_nn) score leads
6/12/24 h for exactly this.

This is the question that decides where the next GPU-hours go, so nothing
further is launched for the dycore arms until it answers.

## ERROR ONSET: sfno_full's gap is made in the FIRST STEP (job 26793246)

z500 RMSE [m], muonlr arm, 2017, 8 inits, leads 6/12/24/72 h:

| | 6 h | 12 h | 24 h | 72 h |
|---|---|---|---|---|
| sfno_full muonlr | 11.6 | 15.0 | 20.5 | 44.8 |
| persistence | 25.4 | 39.8 | 63.5 | 92.3 |

**57 % of the 24 h error is already present at 6 h — a SINGLE model step.** One
6 h step costs 11.6 m, while GraphCast's entire 24 h forecast is 4.06 m: our
one-step error alone is 2.9x their day-1 error. The deficit is single-step
ACCURACY, not error accumulation, and the 240 h numbers (1.34x GraphCast) say
the long-lead behaviour is already respectable.

That kills a class of levers. More training epochs, longer rollout curricula,
better long-lead regularisation and the CRPS calibration work all act on
accumulation. The binding constraint is what one step can resolve.

### What that promotes, with the reason it fits the measurement

1. **A SECOND INPUT SNAPSHOT (t-6 h).** We feed ONE state, so the network must
   infer d/dt from spatial structure alone; U-Cast feeds two past snapshots and
   GraphCast feeds two. This is precisely a single-step-accuracy lever, and it
   is cheap: input channels double, FLOPs barely move (the cost is in the
   spectral blocks, not the input projection). Independently ranked first per
   GPU-hour by GLM-5.2 before this measurement existed.
2. **Vertical resolution, 8 levels vs GraphCast's 37.** Also a single-step
   lever — baroclinic structure the model cannot represent is error it makes
   immediately. Much more expensive (data pipeline, channel spec, full
   retrain), so it is second.

Neither `legoesm.ml.channel_packing` nor `ucast_pe.py` has any history support
today, so (1) is a genuine multi-file change: packing, SFNO `in_channels`, the
training pair loader (needs t-6 h as well as t and t+6 h), the rollout carry,
and the eval. New arch = new checkpoint layout, so it lands as a new arm dir
rather than a resume.

### Two-snapshot input, STAGE 1 landed (model side)

`SFNOPrimitiveEquationConfig.history_steps` (default **0** = every existing arm
byte-identical). At 1, `step(state, dt, prev_states=(prev,))` packs the past
state with the SAME `pack_pe_state` and concatenates it ahead of the current one
along the channel axis, oldest first.

Decisions worth recording:
* **Normalisation statistics are TILED, not recomputed.** The two halves are the
  same physical variables at different times, so one set of per-channel moments
  keeps them on a common scale and a finite difference between corresponding
  channels stays meaningful.
* **`in_channels` is CHECKED, not inferred** — `history_steps=1` demands
  `2 x (4*nlev+2)` and raises otherwise. A merely compatible-looking width would
  train fine while reading the wrong channels.
* **Refused in `hybrid_tendencies` mode.** That path evaluates the network at RK
  STAGE states, which have no history; reusing the macro-step history at every
  stage makes the integrator inconsistent.

`tests/unit/test_sfno_pe_history_input.py`, 7 tests, including the non-vacuity
one that matters: two DIFFERENT pasts with the SAME current state must give
different forecasts, so a history tensor that gets concatenated and then ignored
fails. Gradient flow to the history channels is asserted too — without it
training cannot use them. Existing sfno_pe consumers: 31 passed, 1 skipped.

STILL TO DO before an arm can train (stage 2): the training pair loader must
emit t-6 h alongside t and t+6 h (`load_training_data` builds from
`carries[base + d]`, so the previous snapshot is `base + d - 1` and `d` must
start at 1 to avoid crossing a window boundary), the sfno_full training loop
must carry the previous state through its chained rollout, `run_aimip` must
forward `sfno_history_steps` and size `in_channels`, and the WB2 evaluator must
supply a real t-6 h state for the first step.

### CAPACITY REFUTED for column_nn (job 26793606) — and that closes the argument

`nn_hidden_dim` 256->1024, `nn_n_layers` 4->6, 216,614 -> ~6.3M parameters
(**29x**), everything else held at v2. Training loss at matched epoch AND lead:

| epoch (lead) | v2 (216k) | wide (6.3M) |
|---|---|---|
| 4 (12 h) | 25.363 | 25.355 |
| 5 (24 h) | 29.319 | 29.332 |
| 6 (24 h) | 29.267 | 29.247 |
| 7 (72 h) | 57.188 | 57.022 |

**29x the parameters moves the loss by under 0.3 %.** My pre-registered REFUTE
condition, met exactly. column_nn is not capacity-limited.

Put the three column_nn interventions together — this is now converging
evidence, not a single result:

| intervention | change | effect on training loss |
|---|---|---|
| optimizer: AdamW -> Muon @ 10:1 | the largest sfno_full win (-24.7 % z500) | +0.2 to +0.9 % (worse) |
| capacity: 216k -> 6.3M params | 29x | <0.3 % |
| physics package: 216k MLP vs RRTMGP+Bechtold+CLUBB+124 knobs | total replacement | 89.3 vs 95.8 m z500 at 24 h (7 %) |

Every PHYSICS-side lever is a no-op, including replacing the physics wholesale.
That is what it looks like when the loss is set by something else, and the only
thing all these arms share is the spectral dycore and the path an ERA5 state
takes into it. The reframing two sections up is now CONFIRMED by triangulation
rather than plausible.

Consequence for where GPU-hours go: stop tuning dycore-arm physics. The open
question is the dycore path itself, and job 26795248 (column_nn at leads
6/12/24 h) is the measurement that localises it in time.

### Checkpoint-selection scan, first point (job 26794644)

On the VALIDATION year 2015 (not the 2017 test year), 8 inits:
`epoch_0045_ema` gives z500 20.11 / 45.61 / 71.89 / 106.19 m. Remaining epochs
50/53/55/57/59 still running; the point of the scan is a selection rule that
does not touch 2017.

### Two-snapshot input, STAGE 2: the pairing helper

`build_history_pairs(ic_states, ic_times, history_steps, era5_cadence_hours)`
-> `(keep_indices, prev_states)`, prev tuples OLDEST FIRST.

Two decisions that are the whole point of the function:

* **Derived from what `load_training_data` ALREADY returns, not re-fetched.**
  Extending each window backwards by one snapshot would change the window list,
  and the ERA5 window cache is KEYED on that list — every cache on disk would be
  invalidated and hours re-downloaded, to gain ~5 % more samples (a 5-day window
  is 20 snapshots at 6 h cadence). Dropping the first sample of each window is
  the cheaper trade by a wide margin.
* **Contiguity comes from TIMESTAMPS, never list adjacency.** The maxdata tier is
  12 monthly windows x 36 years, so consecutive entries in `ic_states` cross a
  window boundary at every window edge, where the list-predecessor is a YEAR
  away. Pairing across it feeds the network a bogus t-6 h and trains without
  error. `ic_times` is therefore REQUIRED and its absence raises rather than
  falling back to adjacency.

`tests/unit/test_history_pairs.py`, 6 tests: two windows a year apart drop TWO
samples not one, multi-step history is asserted oldest-first, a 12 h gap at a
declared 6 h cadence drops the sample, missing times raise, and the cadence is
honoured as a parameter (the same data at the wrong declared cadence keeps
nothing).

Remaining for a trainable arm (stage 3): thread `prev_states` through the
sfno_full chained-rollout loss (it calls `model_wrapper.step(s, dt_sfno,
key=...)` at `neural_gcm_spectral.py:4406`, which is confirmed to be the
executed path for this variant), forward `sfno_history_steps` from `run_aimip`
while sizing `in_channels` to `(1+h) x (4*nlev+2)`, and give the WB2 evaluator a
real t-6 h state for the first step.

## RETRACTION + THE REAL LAT-LON BREAK (2026-08-08, user-directed)

**Retracted: "stop tuning dycore-arm physics".** That over-read three null
results. The arms ARE trainable; the spectral numbers are just high. The
measurements stand (Muon +0.2-0.9 % worse, 29x capacity <0.3 %, physics swap
7 %) but "physics tuning is pointless" does not follow from them.

User's actual report: they were trainable before and now are not. That is TRUE,
and it is the LAT-LON path, not the spectral one:

```
TypeError: train_neural_gcm() got an unexpected keyword argument 'rollout_hours'
TypeError: train_physics_params() got an unexpected keyword argument 'rollout_hours'
```

Both lat-lon AIMIP variants died at the trainer call. The spectral arms kept
working because they go through a DIFFERENT trainer
(`neural_gcm_spectral.train_*_spectral`), which is why the WB2 spectral numbers
never moved and I could not see a regression in them.

`training_driver.train_physics_params` / `train_neural_gcm` had lost
`rollout_hours` plus the physics kwargs (`microphysics`, `rad_update_steps`,
`rad_stop_gradient`) that `run_aimip_latlon.py` passes. Restored, and threaded
to where they matter:

* `rollout_hours` now reaches `single_day_rollout(..., hours=...)` inside
  `_build_train_step`. **This is the part that had to be right**: that helper
  defaults to `hours=24` and the lat-lon driver loads 6 h targets, so merely
  accepting-and-ignoring the kwarg would have scored a 24 h forecast against a
  6 h target and trained quietly on the wrong thing — worse than the TypeError.
* `**segment_kwargs` forwarded to `build_training_segment`, so the tuned
  rollout is built with the caller's microphysics/radiation settings rather
  than defaults.
* `rad_stop_gradient` reaches `physics_pipeline.build_step_unified`.

**Second defect, same run: the driver exited 0 while BOTH variants failed.** It
wrote a scorecard whose every entry was an `error` string and returned success —
a batch job would have reported a clean run. The per-variant try/except is
right (one arm's crash must not discard the other's results); swallowing it into
the exit status is not. Now returns 1 and names the failed variants.

`tests/unit/test_training_driver_rollout_hours.py`, 5 tests, asserting on
`_build_train_step` — the function that RUNS the rollout — not on the public
wrappers that merely forward.

### MPAS: not wired, but every piece already exists

There is no MPAS AIMIP driver. There does not need to be a new script:

* `era5_to_mpas_carry` already exists (`era5_to_state.py:1466`) — full
  edge-normal wind projection via `angleEdge`, vertical interpolation, phis
  smoothing.
* `ModelDriver` already accepts `mpas` as a discretization (the AMIP lane runs
  it).
* `train_physics_params` / `train_neural_gcm` are already GRID-AGNOSTIC — they
  take `model` and `grid` as arguments and never construct either.

So MPAS is a `--grid {latlon_cgrid,mpas}` flag on `run_aimip_latlon.py`
dispatching two things: the `--discretization` handed to `build_latlon_config`,
and `era5_to_latlon_carry` vs `era5_to_mpas_carry`. A third near-duplicate
driver would be the copy-paste this repo forbids.

Note for later: `aimip_grid` in the spectral configs is a DEAD key — nothing in
`run_aimip.py` reads it, and the grid is hardcoded `create_gaussian_grid` at two
sites. It should either be honoured or removed; right now it advertises a choice
that does not exist.

## THREE RESULTS (2026-08-08, later)

### 1. Checkpoint selection on the VALIDATION year — ~20 % of z500 for free

Scan of the muonlr arm, validation year 2015, 8 inits (job 26794644), z500 RMSE
[m]:

| epoch | 24 h | 72 h | 120 h | 240 h |
|---|---|---|---|---|
| 45 | 20.11 | 45.61 | 71.89 | 106.19 |
| 50 | 20.92 | 48.37 | 76.39 | 113.07 |
| 53 | 19.52 | 45.70 | 71.01 | 105.08 |
| 55 | 17.81 | 43.70 | 68.22 | 101.12 |
| **57** | **17.41** | **43.01** | **67.65** | **97.64** |
| 59 (FINAL — what every scorecard used) | 21.65 | 45.45 | 69.36 | 97.94 |

Epoch 57 wins on 2015, and the SAME ordering holds on the held-out 2017 test
year (16.35 vs 20.26). So the validation year picks the right checkpoint without
ever touching the test year — a legitimate, leakage-free rule worth **~20 %** of
z500 RMSE, on models we have already trained.

Applied: `params_selected_ema.eqx` + a `params_selected_ema.json` recording the
chosen epoch and the evidence, and `bench_wb2_levante.sbatch` now prefers it over
`params_ema.eqx`.

### 2. column_nn makes ~90 % of its day-1 error in the FIRST 6 HOURS

2017, 8 inits (job 26795248), z500 RMSE [m]:

| | 6 h | 12 h | 24 h |
|---|---|---|---|
| column_nn | **82.20** | 83.70 | 90.87 |
| persistence | 25.44 | 39.83 | 63.52 |
| sfno_full (no dycore) | 11.6 | 15.0 | 20.5 |

After ONE 6-hour step the hybrid is already **3.2x worse than doing nothing**,
and 90 % of its 24 h error is present at 6 h. The bias at 6 h is +1.27 m, i.e.
essentially zero — so this is not a systematic offset accumulating, it is a
large, immediate, unbiased error. Same ERA5 initial conditions and the same
evaluator give the pure-ML arm 11.6 m.

That is the signature of initialisation shock — an ERA5 state entering the
spectral dycore unbalanced and radiating gravity waves — not of physics error.
It also explains cleanly why every physics-side lever was a null result: they
all act on a term that is not the dominant one.

### 3. Training WITH the mass anchor is worth far more than anchoring at eval

column_nn retrained under the constraint (job 26794955) vs the v2 baseline, same
curriculum, matched epoch AND lead:

| epoch (lead) | v2 baseline | anchored |
|---|---|---|
| 7 (72 h) | 57.19 | 49.84 (-12.9 %) |
| 8 (72 h) | 55.18 | 47.49 (-13.9 %) |
| 9 (120 h) | 85.02 | 63.38 (-25.5 %) |
| 10 (120 h) | 83.25 | **61.67 (-25.9 %)** |

Eval-only anchoring bought -6 % z500 at 240 h; TRAINING under the constraint
buys **-26 %** of the 120 h loss. The train/eval mismatch was real: physics that
learned to live with a drifting surface pressure is not the same physics as one
trained without the drift. WB2 scorecard for this arm submitted (job 26801540),
at leads 6/12/24/72/120/240 so the onset curve comes with it.

## LAT-LON FIXED, MPAS WIRED (2026-08-08)

### Second break, found only by re-running the smoke

Restoring `rollout_hours` got column_nn training on lat-lon (1 epoch, 12.9 s,
loss 1800.1, eval T_column RMSE 1.32 K). classical then hit a DIFFERENT error:

```
TypeError: build_segment_fn() got multiple values for keyword argument 'microphysics'
```

`trainable.to_segment_kwargs()` and the driver's `segment_kwargs` OVERLAP — both
carry `microphysics` — and two `**` of the same key is a TypeError at the call.
Fixed by merging, with the TRAINED value second so it wins: it is the quantity
being optimised, and letting the caller's static default override it would
silently cut that parameter out of the rollout it is supposed to be tuning.

Worth noting how this was found: the FIRST fix looked complete (column_nn
trained, exit code honest) and the second failure was only visible because the
smoke was re-run end to end rather than trusted.

### MPAS: one flag, not a third driver

`run_aimip_latlon.py --grid {latlon,mpas}`. The carry-based trainers take
`model` and `grid` as ARGUMENTS and never construct either, so MPAS needed
exactly two dispatches:

* `build_latlon_config`: `--grid-type mpas --discretization mpas` (both already
  valid choices in `run_amip`'s parser).
* the carry converter: `era5_to_mpas_carry` instead of `era5_to_latlon_carry`.

**Stated limitation rather than a silent one.** The two converters do not take
the same arguments: the lat-lon one accepts `microphysics` / `turbulence`, which
size the tracer slots the physics pipeline expects; `era5_to_mpas_carry` has no
such parameters and builds the edge-normal wind state plus scalar cell fields
only. Passing them is a TypeError; DROPPING them silently is worse — a run
configured with microphysics would start from a carry with nowhere to put
condensate. So `--grid mpas` with anything other than
`--microphysics none --turbulence none` raises with the reason and the fix.
Extending `era5_to_mpas_carry` to build tracer slots is the follow-up that lifts
it.

Also: `aimip_grid` in the SPECTRAL configs remains a dead key — nothing in
`run_aimip.py` reads it, and the grid is hardcoded `create_gaussian_grid` at two
sites. It advertises a choice that does not exist and should be honoured or
removed.

### The anchor helps CLASSICAL too (job 26794956, in flight)

Matched epoch AND lead against the unanchored `most_long` baseline (same
12/24/72/120 curriculum):

| epoch (lead) | baseline | anchored |
|---|---|---|
| 4 (24 h) | 21.76 | 20.90 (-4.0 %) |
| 5 (24 h) | 21.72 | 20.86 (-4.0 %) |
| 6 (72 h) | 43.14 | **35.90 (-16.8 %)** |

Same shape as column_nn (-13 % at 72 h, -26 % at 120 h): small at short lead,
large at long lead, which is what a correction to a SECULAR drift must look
like. Two independent physics packages, same sign, same lead-dependence.

### A THIRD break in the same call chain, and the worst of the three

The classical smoke still failed after the merge fix, and the duplicate was not
where I patched it. `build_training_segment` passed four settings BY NAME right
next to `**extra_kwargs`:

```python
rad_update_steps=1, microphysics="none", fix_moisture=False, fix_mass=False, ..., **extra_kwargs
```

so any caller supplying one collided. That is the crash. The quiet half is
worse: when nobody supplied them — which is every caller that is not the lat-lon
driver — **every carry-based training rollout ran with no microphysics and no
mass fixer regardless of the experiment's configuration.** The model being tuned
was not the model being configured. Now defaults in a dict that caller values
override.

Method note: three separate defects sat in one call chain, and each was only
exposed by re-running the smoke end to end after the previous fix. Two of the
three were invisible to the unit tests that existed.

## LAT-LON: BOTH VARIANTS TRAIN AGAIN (2026-08-08)

Smoke, CPU, n_lat=32, 1 epoch:

| variant | train loss | seconds | T_column RMSE | p_s RMSE |
|---|---|---|---|---|
| classical | 133.00 | 1253.1 | 1.376 K | 530.9 Pa |
| column_nn | 1800.11 | 6.8 | 1.317 K | 530.7 Pa |

Four defects, all in one call chain, each exposed only by re-running the smoke
end to end after the previous fix:

1. `train_physics_params` / `train_neural_gcm` had lost `rollout_hours` and the
   physics kwargs -> immediate TypeError, nothing trained. (The user's report.)
2. `to_segment_kwargs()` and the caller's kwargs OVERLAP on `microphysics`;
   double-splatting is a TypeError. Merged, trained value wins.
3. `build_training_segment` passed four settings by name NEXT TO
   `**extra_kwargs` -> same TypeError for any caller supplying one, and — the
   quiet half — every carry-based training rollout ran with `microphysics="none"`
   and `fix_mass=False` regardless of configuration. Now overridable defaults.
4. **`multi_step_rollout_loss` had no production caller at all.** Its docstring
   says "shared by every AIMIP trainer so the rollout+loss is defined ONCE";
   `_build_train_step` instead inlined `single_day_rollout` + `combined_loss`,
   so `loss_config.multi_step_hours` was SILENTLY IGNORED — the lat-lon driver
   built tuple-of-lead targets and handed them to a loss that only ever ran one
   rollout. `_build_train_step` now routes through the shared function, which
   also makes `rollout_hours` correct on both the single- and multi-step paths.

(One of the four was mine: `_to_carry` closed over `args` from an enclosing
scope it does not have; `grid_kind` is now an explicit parameter of
`load_window_pairs`.)

Defects 3 and 4 are the interesting ones: neither produced a crash in the
default configuration, both changed what model was actually being trained, and
neither was visible to the existing unit tests.

## THE MEASUREMENT THAT REFRAMES THE LEADERBOARD COMPARISON (2026-08-08)

`scripts/validate/wb2_representation_floor.py` (committed probe, job run on CPU,
2017, 6 inits): the lead-0 error of the ERA5 -> spectral model state -> WB2 1.5
deg round trip, with **no model integration at all** — same diagnosis, same
regrid, same cos-lat weights, same masks the scorecard uses.

| field | FLOOR (no model) | sfno_full @6 h | column_nn @6 h | GraphCast @24 h |
|---|---|---|---|---|
| z500 [m] | **5.97** | 11.6 | 82.2 | **4.06** |
| t850 [K] | 1.00 | 1.20 | 1.61 | 0.529 |
| mslp [Pa] | 260 | 275 | 1.04e3 | — |
| t2m [K] | **4.63** | — | — | — |

### RETRACTION: our numbers are not comparable to the WB2 leaderboard

**GraphCast's published 24 h z500 (4.06 m) is BELOW our pipeline's own lead-0
floor (5.97 m).** Our scoring path cannot represent a GraphCast-quality forecast
even with a perfect model. Every "4x worse than GraphCast" statement in this
document — including the one that motivated the two-snapshot work — compares a
37-level 0.25 deg model scored natively against our 8-level T63 pipeline. That
is a confound, not a result. The leaderboard rows stay useful as an ORDER OF
MAGNITUDE, not as a target our arms can be subtracted from.

The floor is set by what the pipeline throws away: 8 sigma levels against ERA5's
pressure levels, the T63 truncation, and a hydrostatic z500 diagnosis. Lowering
it means more vertical levels — which is also the second single-step lever
already identified.

### What the floor does and does not excuse

Splitting each 6 h error in quadrature against the floor:

* **z500, column_nn: the floor is 0.5 % of its MSE.** Its 82 m is ~82 m of real
  model error. The dycore first-step problem is CONFIRMED, not an artifact.
* **z500, sfno_full: the floor is 26 % of its 6 h MSE** (model-only ~9.9 m of
  11.6 m). Real, but the model still dominates — the single-step reading
  survives for sfno, in weakened form.
* **mslp, sfno_full: the floor is 89 % of its 6 h MSE.** Its mslp score at short
  lead is very nearly the pipeline scoring itself.
* **t2m: the floor is 4.63 K against arm scores of 4.97-6.2 K at 24 h.** That
  column has been close to meaningless — and note `wb_forecast.py` already
  documents t2m as the LOWEST-MODEL-LEVEL air temperature "proxy", not a real
  2 m diagnostic.

### Method note, and why this should have been first

This is the "validate the instrument before quoting its number" rule, and I
quoted leaderboard gaps all day without it. The probe costs one CPU run and no
GPU. It should have been the first measurement of the campaign, not the
fifteenth.

### Decomposing the floor: VERTICAL levels are worth ~2x the horizontal truncation

Same probe, 2017, 4 inits, sweeping the two things the pipeline throws away.
Lead-0 RMSE:

| levels | trunc | z500 [m] | t850 [K] | u850 | u250 | mslp [Pa] | q700 |
|---|---|---|---|---|---|---|---|
| 8 | T63 | 5.99 | 1.004 | 1.73 | 3.62 | 262 | 5.23e-4 |
| 8 | T106 | 5.53 | 0.890 | 1.35 | 3.51 | 267 | 3.57e-4 |
| 16 | T63 | 5.11 | 0.755 | 1.55 | 2.25 | 290 | 6.08e-4 |
| 16 | T106 | **4.33** | **0.563** | **1.06** | **1.90** | 297 | 4.79e-4 |

z500 floor, from the 8/T63 baseline:
* **8 -> 16 levels: -14.6 %** (vertical)
* **T63 -> T106: -7.7 %** (horizontal)
* both: -27.8 %

**Vertical resolution is worth about twice the horizontal truncation**, and the
split is sharper for the fields that live on vertical structure: u250 -38 % and
t850 -25 % from levels alone, against -3 % and -11 % from truncation. That is
what an 8-level column does to a jet.

Two things this does NOT say:

* It is a FLOOR, not a model error. A 16-level model also has more to learn, so
  the training cost is real and this measurement does not predict the trained
  skill.
* **Even 16 levels AND T106 leaves the floor at 4.33 m, still above GraphCast's
  published 24 h score of 4.06 m.** Doubling both resolutions does not buy a
  pipeline that can represent a leaderboard-quality forecast. The comparison
  needs the floor stated next to it, permanently.

Two fields go the WRONG way with more levels — mslp 262 -> 290 Pa and q700
5.2e-4 -> 6.1e-4 — which is worth understanding before any 16-level arm is
trained, since mslp is the field the mass anchor was fixing. Not chased here;
flagged.

Ranking for the arms as it now stands: vertical levels for the floor AND
(per the single-step argument) for the model; the two-snapshot input for
sfno_full's remaining single-step error; the mass anchor already banked.

## CODEX ROUND 3: I WAS WRONG ABOUT MPAS (2026-08-08)

**RETRACTED: "MPAS is wired, one flag, not a third driver."** It was not, and I
never ran it. I reasoned from "the trainers take model+grid as arguments,
therefore the grid is a two-value swap" and shipped the claim without a smoke
test. Same failure as the mass anchor earlier today — PROVE THE PATH EXECUTES —
and I made it twice in one session.

What actually blocks MPAS, verified in the source:

1. `--resolution` is a LATITUDE COUNT in this driver and an SCVT SUBDIVISION
   LEVEL for MPAS (the builder caps it at 10). `--n-lat 72` (or 32 under
   `--smoke`) does not construct a mesh at all — it fails before any data loads.
2. `era5_to_mpas_carry` returns an `MPASCarry`, which is NOT a `SegmentCarry`
   and carries none of the `held_*` radiative-flux fields this driver's loss and
   evaluation read.
3. The SST regrid, `build_training_segment` and the eval accumulators all index
   `grid.lat` / `grid.lon` / `cos_lat` and a cell-centred `v`, none of which an
   SCVT mesh provides.

Refusing microphysics/turbulence — my "honest limitation" — addressed NONE of
these. `--grid mpas` now exits with the three blockers named. A loud refusal is
worth more than a flag that looks supported; an MPAS AIMIP lane needs its own
carry + forcing + eval path first, which is a real piece of work, not a flag.

### Also fixed from round 3

* **Residual skip + history was silently wrong.** SFNO's ACE-style residual adds
  `x_in[..., :out_channels]` — the FIRST block of the input. I packed the past
  FIRST, so a residual state-update network (`residual_prediction` defaults
  True) would have predicted `past_state + correction`: a 6 h-stale baseline,
  no error anywhere. Current state now occupies the first block; a test gives
  the past a large offset and asserts the output does not follow it.
* **I broke an existing test.** `test_train_step_builds_segment_once` passed
  `loss_config=None`, which only worked while the trainer inlined its own
  rollout and never read the config. Fixed the fixture (real `LossConfig`, and
  its `single_day_rollout` fake now accepts `hours=`). Verified by stashing my
  change that the OTHER two failures in that file are pre-existing.

### Codex confirmations worth keeping

* The four `build_training_segment` defaults break no existing caller — it
  enumerated all of them (`scale_build.py:437,453,508`,
  `training_driver.py:465,529,609`, `run_aimip_latlon.py:465,490`).
* Multi-step is now coherent: `multi_step_rollout_loss` sorts leads and consumes
  tuple targets in the same order; `rollout_hours` is correctly the
  single-target path only.
* Tiling the normalisation statistics across history copies is correct.
* `build_history_pairs` indexing and boundary handling are correct — its only
  problem is that it is still dead code (stage 3 remains).
* The AMIP mass anchor placement is right and both AMIP scripts forward the
  flags. No defect found there.

Fair criticism I am recording rather than arguing with: the new rollout-hours
tests are mostly SOURCE-TEXT assertions, not executed forwarding.

### Source-text tests replaced by EXECUTED ones (codex round-3 criticism, accepted)

Codex was right that the rollout-hours tests asserted on source text. That is a
tripwire, not a proof: it passes if someone renames a variable while breaking
the behaviour, and it cannot see whether the value ARRIVES. Three executed tests
added, calling the real `train_neural_gcm` with a REAL tiny `NeuralPhysics`
(a duck-typed stub does not survive `make_neural_step_unified`, which reads
`nlev` off it — and a fake that dodges the real call path would defeat the
point) and observing what the rollout is handed:

* `rollout_hours=6.0` in -> the rollout is called with `hours=6.0`, not 24.
* `microphysics="kessler"`, `rad_update_steps=7` -> both arrive at
  `build_segment_fn`, overriding the defaults, while `fix_mass` keeps its
  default.
* omitting everything -> `microphysics="none"`, `rad_update_steps=1`,
  `hours=24.0` (the non-vacuity partner: the override test would pass trivially
  if the defaults happened to equal the overrides).

NON-VACUITY VERIFIED by breaking the forwarding on purpose (hardcoding
`rollout_hours=24.0` at the call site): the executed test fails, and the file
returns to 12 passed once restored. 12 tests total.

### Two-snapshot input, STAGE 3 (partial) — and why it stops here

Landed: `NeuralGCMSpectralConfig.sfno_history_steps` (default 0), architecture
sizing `in_channels = (1 + history) x spec.n_channels`, `history_steps` onto the
emulator config, `run_aimip` forwarding the suite key, and a history-carrying
`_rollout_segment` whose scan carry is `(previous, current)`.

One design point worth stating: after the first macro step the "previous" state
is the model's OWN prior prediction, not ERA5. That is the honest autoregressive
setup — at inference nothing else exists — so only step 0 takes a real observed
predecessor.

NOT landed: the three sfno_full loss functions (`_chained_loss`, the CRPS
member loss, `_curriculum_loss`) and the sample loop — including the pmap batch
path and `stage_sample` — do not thread a predecessor. Wiring that means
touching the data-parallel batching, which I am not able to verify end to end
here without running the full sfno training.

**So the feature is GATED, not half-live.** At the default 0 everything is
byte-identical; above 0 the run RAISES at `_rollout_segment` ("no prev_state was
supplied") because no call site passes one. A partial version that quietly
defaulted the predecessor to the current state would feed a zero tendency and
train a different model than the config describes — the exact failure class this
whole day has been about. `tests/unit/test_sfno_pe_history_input.py` pins the
gate AND asserts it becomes invalid the moment a call site starts passing
`prev_state`, so the test fails loudly when someone finishes the wiring rather
than silently passing on stale assumptions.

Correction to the round-3 report: `residual_prediction=False` in the sfno_full
arch (`neural_gcm_spectral.py:3996`), so codex's residual-skip defect never
affected THIS arm. The channel-order fix still stands — the class is a generic
API and `residual_prediction` defaults True elsewhere.

## column_nn's FIRST-STEP ERROR IS SURFACE PRESSURE (2026-08-08)

Derived from measurements already in hand — the anchored column_nn scorecard at
lead 6 h and the representation floor — with the floor removed in quadrature:

| field | total @6 h | floor | model-only |
|---|---|---|---|
| z500 | 82.27 m | 5.97 | **82.05 m** |
| mslp | 1044 Pa | 260 | **1011 Pa** |
| t850 | 1.608 K | 1.00 | 1.258 K |
| u850 | 2.98 m/s | 1.72 | 2.44 |
| v850 | 3.09 m/s | 1.60 | 2.64 |

The eval diagnoses z500 hydrostatically, so a surface-pressure error displaces
it by ``(R_d T / g)(dp/p)``. Putting the model-only mslp error through that:

* 1011 Pa -> **82.9 m** of z500 displacement
* measured model-only z500 error: **82.1 m**

The temperature route is refuted: reproducing 82 m from thickness alone needs a
column-mean T error of **4.04 K**, and the measured model-only t850 error is
**1.26 K** — three times too small.

**So essentially the whole first-step z500 error is the surface-pressure error.**
Not temperature, not the winds (2.4-2.6 m/s model error, in family with
sfno_full).

Labelled honestly: the ratio is CONFIRMED to dominate, PLAUSIBLE at exactly
100 %. The relation is not a coincidence — it is the same hydrostatic dependence
the evaluator uses to produce z500 — but RMSEs do not combine as a strict linear
propagation, and the displacement is nominal-T dependent (74 m at 250 K, 89 m at
300 K), so "consistent within the temperature uncertainty" is the correct claim.

### Why the mass anchor did NOT help at 6 h, and what the real target is

The anchor restores the GLOBAL MEAN of ``int p_s dA``. The 6 h error is a
SPATIAL PATTERN error in p_s — the anchored and unanchored arms score the same
82 m at 6 h while diverging by 24 % at 240 h. Both facts are consistent: a
secular global drift is what the anchor removes, and it is a long-lead term.

So the target for the dycore arms is now specific: **the spatial surface-pressure
error the dycore develops within its first 6 hours.** That is a different
question from anything tried today (optimizer, capacity, data volume, physics
package, mass anchor), and it is the one worth the next GPU-hours.

### How large the first-step pressure error is: worse than CLIMATOLOGY

Same anchored column_nn scorecard, mslp RMSE [Pa]:

| | 6 h | 12 h | 24 h | 72 h | 120 h | 240 h |
|---|---|---|---|---|---|---|
| model | **1044** | 1050 | 1147 | 1402 | 1594 | 1902 |
| persistence | 366 | 463 | 674 | 923 | 1029 | 1104 |
| ERA5 climatology | **811** | 793 | 790 | 798 | 819 | 892 |

**After ONE 6-hour step the dycore's surface-pressure field is worse than the
annual-mean climatology (1044 vs 811 Pa), and 2.9x worse than not forecasting at
all (366 Pa).** For scale: persistence's 366 Pa is close to the 260 Pa
representation floor, i.e. the real 6 h change in the field is only ~257 Pa,
while the model injects ~1011 Pa on top of it.

That is not a tuning-scale error. One macro step puts more error into p_s than
the field's entire climatological spread, and — via the hydrostatic z500
diagnosis — that single quantity accounts for the whole 82 m z500 error.

### The next measurement, and it is already queued

The anchored CLASSICAL arm (job 26794956) will be the first loadable classical
checkpoint on current code. Scoring it at leads 6/12/24/72/120/240 tests
directly whether the 6 h pressure error is physics-INDEPENDENT: classical and
column_nn share the dycore and the ERA5 initial state but have completely
different physics (RRTMGP + Bechtold + CLUBB + 124 tuned knobs vs a 216k MLP).

* both ~1000 Pa at 6 h -> the dycore / initialisation path owns it, and no
  physics work of any kind will move it;
* materially different -> physics drives the first step after all, and the
  three null results from today (optimizer, capacity, data) need re-reading.

No new compute is needed to set this up — the arm is already training, and the
scorecard just needs the short leads included.

## MASS ANCHOR IN TRAINING: -25 to -30 % AT 5 DAYS, ON BOTH ARMS

Matched epoch AND lead, one flag changed, both dycore arms (jobs 26794955 /
26794956):

| arm / lead / epoch | baseline | anchored | delta |
|---|---|---|---|
| **classical @120 h ep9** | 65.33 | **45.46** | **-30.4 %** |
| column_nn @120 h ep9 | 85.02 | 63.38 | -25.5 % |
| column_nn @120 h ep10 | 83.25 | 61.67 | -25.9 % |
| classical @72 h ep7 | 43.08 | 35.84 | -16.8 % |
| column_nn @72 h ep8 | 55.18 | 47.49 | -13.9 % |

Provenance for the classical baseline: its scorecard's
`train_loss_history` is `[65.28289942933154]`, matching the epoch-10 line in its
log exactly, which fixes 65.326233 as its epoch 9 — the row above.

Two completely different physics packages — RRTMGP + Bechtold + CLUBB with 124
tuned knobs, and a 216k-parameter MLP — improve by the same 25-30 % from the
same one-line constraint, with the gain growing with lead (14-17 % at 72 h,
25-30 % at 120 h). That lead-dependence is the signature of removing a SECULAR
drift, and it is the strongest confirmation of the mechanism so far: the effect
is a property of the shared dycore, not of either physics package.

This is the largest single improvement of the campaign, and unlike every other
lever tried today it came from a conservation defect rather than a
hyperparameter.

### RESOLVED: "more levels makes pressure worse" is a DIAGNOSTIC artifact

Flagged earlier and worth resolving, because surface pressure turned out to be
the field that matters most: the floor sweep showed mslp getting WORSE with more
levels (262 -> 290 Pa at T63, 8 -> 16 levels). Read the code rather than
guessing:

* `_apply_phis_hydrostatic_adjustment` computes
  `p_s_corrected = p_s * exp(delta_phis / (R_d T_sfc))`. It takes `sigma`, but
  uses it ONLY for the hybrid floor (`if is_hybrid`). These suites run
  `vertical_coord: sigma`, so **p_s is identical at 8 and 16 levels.**
* The mslp DIAGNOSTIC is `mean_sea_level_pressure(p_s, T[..., -1], phis)`, and
  the lowest full-level sigma moves 0.9381 -> 0.9691 between 8 and 16 levels.
  A different lowest-level temperature changes the reduction over topography —
  and it is exactly zero over ocean, where `z_s = 0` makes the reduction the
  identity.

So the surface-pressure FIELD does not degrade with resolution; only its
sea-level REDUCTION does, through the temperature it borrows. **The
recommendation to add vertical levels is not compromised for the field that
matters most.** CONFIRMED by code read, not inferred.

A cleaner check would compare p_s directly instead of mslp; p_s is not in the
WB2 headline set, which is why the reduction is being scored in the first place.

# THE ANSWER: THE FIRST-STEP ERROR IS PHYSICS-INDEPENDENT (2026-08-09)

Job 26807330. Two arms, SAME spectral dycore, SAME ERA5 initial states, SAME
evaluator and protocol (2017, 8 inits, 24 h stride), COMPLETELY different
physics:

* **classical** — RRTMGP radiation, Bechtold convection, CLUBB turbulence,
  McFarlane gravity-wave drag, Sundqvist microphysics, 124 tuned parameters
* **column_nn** — a 216k-parameter per-column MLP and nothing else

Model-only error at lead 6 h (representation floor removed in quadrature):

| | classical | column_nn | ratio |
|---|---|---|---|
| mslp | 967 Pa | 1011 Pa | **0.96** |
| z500 | 82.7 m | 82.1 m | **1.01** |

**They agree to 1-4 %.** Replacing the entire physics package — every scheme,
every tuned constant — changes the first-step error by essentially nothing.

## This explains every null result of the campaign

| lever tried | effect |
|---|---|
| optimizer AdamW -> Muon @ 10:1 | +0.2 to +0.9 % (worse) |
| capacity 216k -> 6.3M (29x) | <0.3 % |
| physics package, total replacement | 1-4 % at 6 h |

All three act on PHYSICS, and physics does not set this error. They were not
weak levers badly applied; they were levers on the wrong term.

## But physics DOES matter at long lead

The arms diverge exactly where they should:

| lead | 6 h | 24 h | 120 h | 240 h |
|---|---|---|---|---|
| mslp classical | 1001 | 1059 | 1291 | **1393** |
| mslp column_nn | 1044 | 1147 | 1594 | **1902** |
| t850 classical | 1.51 | 2.66 | 6.61 | **8.55** |
| t850 column_nn | 1.61 | 3.03 | 5.40 | **6.75** |

Identical at step 1, then separating — and in opposite directions by field
(classical better on pressure, column_nn better on temperature). So the
campaign has two distinct problems, not one:

1. **A first-step error owned by the dycore / initialisation path.** ~1000 Pa of
   surface pressure, physics-independent, and via the hydrostatic diagnosis it
   is the ENTIRE 82 m z500 error. Worse than the annual-mean climatology
   (811 Pa) and 2.7x worse than not forecasting (366 Pa).
2. **Long-lead behaviour, which physics and the mass anchor both move.** The
   anchor is worth -25 to -30 % at 120 h on both arms; physics choice is worth
   tens of percent at 240 h.

Claim: **CONFIRMED**. Controlled — one variable (the physics package), everything
else held, with the predicted outcome. Caveat stated: classical was scored at
epoch 9 and column_nn at its final epoch, so the two are not at identical
training maturity; the agreement to 1-4 % DESPITE that strengthens the result
rather than weakening it.

## Where the next GPU-hours go

Not into physics. Into the first 6 hours of the dycore's surface pressure:
initial-condition balance, the ERA5 -> spectral p_s reconciliation over
topography, and the dycore's continuity/filter treatment of `lnps`. Vertical
levels remain the resolution lever (worth ~2x the horizontal truncation, and the
apparent mslp penalty is a reduction artifact, not a field degradation).

## IT IS AN INITIALISATION SHOCK, NOT ACCUMULATING ERROR (2026-08-09)

Growth rate of the mslp error, per HOUR of forecast (same scorecards):

| | first 6 h | 6-12 h | 12-24 h | 24-72 h | 72-120 h | 120-240 h |
|---|---|---|---|---|---|---|
| classical | **166.8** | 1.3 | 4.2 | 3.0 | 1.8 | 0.9 |
| column_nn | **174.0** | 1.0 | 8.1 | 5.3 | 4.0 | 2.6 |

**The error accumulates 130-170x faster in the first six hours than in the next
six** — and both arms do it at the same rate, within 4 %.

That is not model error accruing. It is a STEP: the model jumps to a state
displaced from the ERA5 analysis by ~10 hPa RMS almost immediately, then evolves
at an ordinary rate from there. The displacement does not recover — the error
stays near 1000 Pa for a full day (1001 -> 1009 -> 1059 Pa at 6/12/24 h).

Combined with the lead-0 floor of 260 Pa, the sequence is unambiguous:

* t=0: the ERA5-derived initial state is GOOD (260 Pa, near the representation
  floor).
* within 6 h: the dycore has moved p_s ~1000 Pa away from it, physics-
  independently.
* thereafter: normal slow growth.

This is the classic NWP initialisation-shock signature — the analysis is not
balanced with respect to THIS model's dynamics, so fast modes adjust it onto the
model's own attractor in the first hours. And because z500 is diagnosed
hydrostatically, that single displacement IS the entire 82 m z500 error.

### What follows

The standard remedy is to balance the initial state against the model before
integrating — digital-filter initialisation or normal-mode initialisation — so
the fast adjustment happens BEFORE t=0 rather than being scored as forecast
error. That is the concrete next implementation, and it is squarely a
dycore/initialisation task, not a physics or training-recipe one.

Two cheaper things worth measuring first, both of which sharpen the target:
* a high-cadence (per-model-step) time series of area-weighted RMS
  `p_s(t) - p_s(0)` through the first 6 h, to see whether the adjustment is
  monotonic or an oscillation that settles — this distinguishes a balance
  problem from a systematic orography/mass mismatch;
* the same split over ocean vs land, since the phis reconciliation
  (`p_s * exp(delta_phis/(R_d T_sfc))`) only acts where topography was smoothed.

Neither needs a training run.

## RETRACTION: my spin-up probe measures EVOLUTION, not error

I built `scripts/validate/aimip_pressure_spinup.py` to test whether the ~1000 Pa
6 h pressure error is a purely dynamical initialisation shock. **It does not
answer that, and I designed it wrong.**

It records the area-weighted RMS of `p_s(t) - p_s(0)` in a zero-physics run.
That is displacement from the initial state — which the real atmosphere also
does. The true 6 h change is ~258 Pa (persistence error at 6 h, floor removed),
and the zero-physics dycore moves **313 Pa** in 6 h. Same order. The probe
therefore cannot separate correct evolution from wrong evolution; only a
comparison against the ERA5 verification can, and that is what the scorecard
already does.

This is the "is the metric measuring what its name says" gate, failed by me
after writing the gate into the probe's own docstring.

### What the run DOES establish, and these stand

* **No land/ocean asymmetry: 311.9 Pa land vs 314.6 Pa ocean at 6 h**, and equal
  at every step from 10 minutes on. The `p_s * exp(delta_phis/(R_d T_sfc))`
  reconciliation to smoothed topography acts ONLY where topography was smoothed,
  so it cannot produce a signal that is identical over ocean. **That mechanism
  is REFUTED** — which is a real result, just not the one I was after.
* **Global-mean p_s is conserved to +0.15 Pa over 6 h** in a free run,
  independently confirming the dry-mass anchor works outside the training loop.
* **The evolution is not anomalously large** — 313 Pa of dynamical motion
  against a true change of ~258 Pa. The dycore is not flinging the state around,
  so gross over-activity is not the explanation either.
* The curve is **monotonic**, not a ringing oscillation that settles: still
  climbing 25 Pa/h at hour 6. A classic gravity-wave adjustment would have rung
  and decayed within an hour or two, so the simple Lamb-wave-shock story does
  not fit what the dycore actually does.

### Where that leaves the diagnosis

The initialisation-shock reading rested on the 130-170x growth-rate step in the
SCORECARD, and that measurement is untouched — it compares against ERA5 at each
lead. What is now refuted is one candidate MECHANISM (orography reconciliation)
and weakened is another (ringing gravity-wave adjustment).

The next probe must compare the 6 h forecast against the ERA5 verification and
split the error into amplitude versus phase/pattern, rather than measuring
displacement from t=0. That is a different, more careful instrument, and it is
the honest next step rather than another guess at the mechanism.

## THE DAMAGE IS TO THE PRESSURE FIELD'S PATTERN, AND ONLY IT

Anomaly correlation (1.0 = perfect pattern) at lead 6 h, same scorecards, same
cases — no new run, the metric was already there:

| field | classical | column_nn | persistence |
|---|---|---|---|
| t850 | 0.971 | 0.966 | 0.969 |
| z500 | 0.828 | 0.831 | 0.978 |
| **mslp** | **0.564** | **0.547** | **0.898** |

Read down the column: after ONE 6-hour step the model's TEMPERATURE pattern is
as good as persistence (0.971 vs 0.969), while its SURFACE-PRESSURE pattern has
collapsed from 0.898 to 0.56. z500 sits in between, exactly as it must, being
diagnosed hydrostatically from both.

So the 6 h error is not an amplitude error and not a uniform shock: **the
spatial pattern of p_s is being destroyed while temperature and winds keep
theirs.** (Winds were already shown to be in family with sfno_full —
2.4-2.6 m/s model-only at 6 h.)

### This narrows the mechanism sharply

A generic unbalanced initial condition radiates gravity waves, which corrupt
winds AND pressure together. The winds are fine. So "the analysis is unbalanced"
in its simple form does not fit either — consistent with the zero-physics probe,
where the adjustment was monotonic rather than a ringing wave.

What can corrupt p_s specifically while leaving T and u/v intact is the
**surface-pressure tendency itself** — the continuity term
`dlnps/dt = -int (div(V) + V . grad lnps) dsigma`. An error in that vertical
integral, its weights, or the divergence it consumes would produce exactly this
signature: prognostic winds and temperatures evolve correctly, and the single
diagnostic they feed develops a wrong pattern.

Labelled **PLAUSIBLE** — this is a localisation from a signature, not a
measurement of the term. The next step is to read `spectral_pe_tendencies`'
`dlnps_dt_grid` construction and check the sigma-integration weights against the
continuity equation, then verify with a manufactured solution where the exact
`dlnps/dt` is known. That is a code-plus-unit-test task, not a GPU one.

What is CONFIRMED here: the 6 h damage is a pattern failure localised to p_s,
temperature and winds are unaffected, and persistence beats the model on that
pattern by 0.90 vs 0.56.

## THE MODEL BEATS PERSISTENCE ON WINDS AND TEMPERATURE — AND ONLY MASS IS BROKEN

Anomaly correlation, anchored classical vs persistence, leads 6/12/24/72 h:

| field | model | persistence |
|---|---|---|
| v850 | 0.869 / 0.814 / 0.691 / 0.327 | 0.840 / 0.641 / 0.345 / 0.205 |
| v500 | 0.937 / 0.901 / 0.816 / 0.358 | 0.867 / 0.686 / 0.393 / 0.217 |
| u250 | 0.952 / 0.940 / 0.900 / 0.594 | 0.944 / 0.865 / 0.723 / 0.537 |
| t850 | 0.971 / 0.948 / 0.919 / 0.726 | 0.969 / 0.931 / 0.878 / 0.771 |
| **mslp** | **0.564** / 0.539 / 0.484 / 0.326 | **0.898** / 0.834 / 0.647 / 0.348 |
| **z500** | **0.828** / 0.822 / 0.791 / 0.636 | **0.978** / 0.946 / 0.865 / 0.720 |

**The dycore beats persistence on every wind component at nearly every lead —
v500 at 24 h is 0.816 against 0.393, more than double.** It matches or beats
persistence on temperature. It is a working forecast model.

It is uniquely, badly worse on the two MASS fields. That reverses the earlier
gloomy reading: the problem is not a broken dycore, it is something specific to
surface pressure and the geopotential diagnosed from it.

### Three mechanisms tested and REFUTED

1. **Orography reconciliation** (`p_s * exp(delta_phis/(R_d T_sfc))`) — refuted
   by the zero-physics probe's land/ocean symmetry (311.9 vs 314.6 Pa).
2. **The continuity term.** Read against the equation: sigma-coordinate
   continuity integrates to `dlnps/dt = -int(D + V.grad lnps) dsigma / (1-sigma_top)`,
   and the code computes exactly that (`_compute_sigma_dot_gaussian` returns
   `sum(div_3d * dsigma)`, divided by `sigma_range`). Checked the one place a
   quiet factor error could hide — `sum(dsigma)` vs `1 - sigma_half[0]` — and
   the ratio is 1.000000000000 at both 8 and 16 levels. **Correct.**
3. **Orography decaying during the rollout.** `apply_spectral_filter_to_state`
   touches `vor_hat`, `div_hat`, `T_hat`, `lnps_hat` — not `phis_hat`. Verified
   by a 36-step zero-physics rollout with a perturbed orography: RMS change 0,
   arrays bit-identical. **Refuted.**

### The hypothesis that now fits everything — PLAUSIBLE, untested

ACC is an anomaly correlation against CLIMATOLOGY, and the scorecard's
climatology is ERA5's. If the model's own preferred p_s base state differs from
ERA5's by a fixed spatial pattern, then:

* mslp/z500 ACC collapses (the anomaly is contaminated by a constant offset),
* winds and temperature are untouched (no such base-state offset),
* the error appears within 6 h and then barely grows (1001 -> 1009 -> 1059 Pa),
* the lead-0 floor is small, because at t=0 p_s is still ERA5's own.

That is a systematic BASE-STATE offset, not a forecast-skill failure — and it
would be correctable rather than fundamental.

**The test is cheap and specific:** compute the 6 h mslp error FIELD for several
cases and measure what fraction of its variance is explained by the case-mean
error pattern. A high fraction means a fixed offset; a low fraction means
genuine per-case forecast error. That is the next probe, and unlike the last one
it compares against ERA5 rather than against t=0.

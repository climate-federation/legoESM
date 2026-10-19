# SWG online learning, round 7: training-driver receipt

Date: 2026-10-14. Base: `d63e2b9f4` (round 6 merged). Work copy
`claude-swgonl-r7-driver`; every change is uncommitted (the operator
commits). CPU only, fp64. Scope: the round-6 codex review's defects 1 (no
training driver; the shared optimiser's defaults are not the template's) and
5 (launch prerequisites; stale design note). **No training was launched.**

Commands below run from the work copy with
`PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`
and `python -m pytest -q -p no:cacheprovider <node id>`. A "reversion" undoes
one fix in a scratch copy of its file, never in the work copy (round 6's
mechanism): table `mutants.py`, runner `mutrun.py`, driver `reversions.py`,
results `reversions.jsonl`; measurement scripts `m01_*` to `m07_*`; citation
checker `check_citations.py`; all in
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_r7/driver/impl/`.

## What changed

- `scripts/run/run_swg_online_training.py` (new). `train` runs one network
  seed of one configuration file, resumably (`run_swg_online_training.py:660`);
  `check-data` runs the record through every split and window length
  (`run_swg_online_training.py:359`); `profile` measures the cost
  (`run_swg_online_training.py:865`).
- `config/swg_online/exp1_template_ring_excluded.json`,
  `exp1_template_ring_weighted.json`, `exp1_symmetric_ring_weighted.json`
  (new): one committed configuration per candidate.
- `packages/ml/legoesm/training/swg_online.py`: the strict optimiser wrapper
  (`swg_online.py:1467`, `swg_online.py:1487`) and the network scales
  (`swg_online.py:369`, `swg_online.py:381`, `swg_online.py:439`). Nothing
  round 6 fixed was touched.
- `tests/ml/test_run_swg_online_training.py` (new, 39 tests) and four new
  tests in `tests/ml/test_swg_online.py`, whose real-data seam test now builds
  the optimiser from the strict settings.
- `docs/ocean/fidelity/testcases/nemo_testcases_l1_swg_online_design.md`:
  rewritten to what is now true; prerequisites and open decisions listed.

## Finding dispositions

| Finding | Disposition | Evidence | Reversion that makes its test fail |
|---|---|---|---|
| Codex 1: no training driver consumes the dataset, updates the network, or records the resolved experiment | Fixed: `train` (`run_swg_online_training.py:660`) records every resolved choice in `run_manifest.json` (`run_swg_online_training.py:574`); the batch loss is Eq. 7 over the contract's batch (`run_swg_online_training.py:391`) | `test_run_swg_online_training.py:384` two window lengths, six steps, gates, manifest; `test_run_swg_online_training.py:418` the logged first loss equals Eq. 7 recomputed window by window outside the driver (relative `1e-9`) | `drv_wall_ring_hardcoded_weighted` (logged 0.0365 against 0.0060), `drv_objective_hardcoded_symmetric` (0.0078 against 0.0060) |
| Codex 1: the shared optimiser's defaults (warm-up, cosine, clipping, Optax epsilon 1e-8) are not the template's | Fixed: `SWGOptimizerSettings` has no defaults (`swg_online.py:1467`); `create_swg_optimizer` refuses a `TrainingConfig` (`swg_online.py:1498`), any schedule but `constant` (`swg_online.py:1504`), any clipping (`swg_online.py:1507`) | `test_swg_online.py:1208` every unset field raises `TypeError`, a `TrainingConfig` and the factory's alternatives are refused; `test_swg_online.py:1229` six steps with gradient norms 1e-7 to 1e3 equal the explicit Keras-convention factory call word for word, and step one is the hand value | `opt_accepts_any_settings`, `opt_settings_default_clipping`, `opt_schedule_unchecked`, `opt_factory_warmup_cosine`, `opt_factory_clipping`, `opt_factory_optax_epsilon` |
| Codex 5: objective unresolved | Owner decision, kept open with no default: no start without a configuration file (`run_swg_online_training.py:951`), every key required (`run_swg_online_training.py:138`), one configuration per candidate | `test_run_swg_online_training.py:67` the three candidates follow the template procedure; `test_run_swg_online_training.py:125` 18 plants (missing, extra, unknown choices) raise; `test_run_swg_online_training.py:134` no configuration, no start, nothing written | `drv_config_extra_keys_allowed`, `drv_objective_unchecked`, `drv_three_seeds_not_required` |
| Codex 5: wall-ring treatment unresolved | Same; the configured ring reaches the loss | `test_run_swg_online_training.py:418` (its configuration uses `excluded`) | `drv_wall_ring_hardcoded_weighted` |
| Codex 5: Eq. 6 network scales unresolved | Implemented as the shallow-water analogue, every factor from the deck (`swg_online.py:381`); measured on ten real states (below); owner to confirm | `test_swg_online.py:1266` each scale against its formula, moved-deck plants, non-uniform depth refused; `test_swg_online.py:1306` the report measures what the network reads; `test_run_swg_online_training.py:623` scaled inputs of order one on real starts | `scales_full_width_L`, `scales_table1_rho`, `scales_velocity_without_dx`, `scales_ssh_without_f0`, `scales_potential_without_depth`, `scales_depth_not_uniform`, `report_potential_unscaled`, `report_template_input_unscaled`, `report_scales_unchecked` |
| Codex 5: optimiser convention unresolved | Every configuration states Keras AdamW, epsilon 1e-7 in the Keras convention, constant rate, no clipping (the task's instruction) | `test_run_swg_online_training.py:67` pins the values; `test_swg_online.py:1229` | `opt_factory_optax_epsilon` |
| Codex 5: 3.125 km adequacy | Open, outside this task's files: the operator's FAIL is an instrument defect in the adequacy tool (next section); SSH, vorticity and every year-1 statistic pass | `round6/reference/adequacy.json`; `nemo_testcase_l1_swg_online_reference_adequacy.py:102-103` | none (not this task's code) |
| Codex 5: every split and window on the completed record | The record is not complete. `check-data` (`run_swg_online_training.py:359`) runs every split at every window length on demand; `train` validates the whole contract before a run directory exists (`run_swg_online_training.py:703`); run here on the six complete real chunks | `test_run_swg_online_training.py:268` every split at every window length, held-out batches on the fixed lattice, an uncovered split raises; `test_run_swg_online_training.py:623` real chunks through the command line | `drv_check_data_skips_test_split`, `drv_check_data_first_window_only` |
| Codex 5: design note stale | Rewritten to the current state | design note | none (prose) |
| Codex 3, producer side: a free-run manifest need not identify a checkpoint or seed | Every free-run manifest names the run, seed, window length, checkpoint digest, objective and contract digest (`run_swg_online_training.py:806`); refusing duplicates is the scorer owner's | `test_run_swg_online_training.py:545` | `drv_free_run_without_checkpoint_digest` |
| Codex 4, producer side: the sampling label is trusted and missing metadata read as snapshot | Every free-run day file carries the snapshot label the harness requires (`run_swg_online_training.py:541`), read back through the harness's own loader; the free-run deck is the double-gyre driver's dg50 deck and passes the scorer's deck binding (`run_swg_online_training.py:287`) | `test_run_swg_online_training.py:545`; `test_run_swg_online_training.py:573` | `drv_free_run_unlabelled`, `drv_free_run_namelist_unresolved` |
| Task: start gates | 1% initial-acceleration bar on the first batch's ten starts (`run_swg_online_training.py:758`); no-inert gate at the first step (`run_swg_online_training.py:774`); finite loss and gradient at every step (`run_swg_online_training.py:772`) | `test_run_swg_online_training.py:447` seed 0 (5.0%) stops before any update; `test_run_swg_online_training.py:466` NaN loss, infinite gradient and all-zero gradient stop the run | `drv_gate_not_enforced`, `drv_finite_unchecked`, `drv_no_inert_skipped` |
| Task: curriculum | Network carried; optimiser re-initialised at every window length (`run_swg_online_training.py:750`) | `test_run_swg_online_training.py:510` the step count restarts at window two (1, not 5) and window two starts from window one's last network | `drv_optimiser_not_reinitialised`, `drv_network_reset_each_window` |
| Task: checkpoints and resume | Atomic per-epoch checkpoint of network and optimiser state (`run_swg_online_training.py:786`, the repository's `save_checkpoint`); resume checks the digest and loads both (`run_swg_online_training.py:732`) and refuses any changed choice (`run_swg_online_training.py:689`) | `test_run_swg_online_training.py:530` killed while writing checkpoint two, the resumed epoch's checkpoint equals what the killed process was writing word for word; `test_run_swg_online_training.py:589` a changed configuration, `windows_per_call`, commit and an unlisted seed are refused | `drv_resume_ignores_checkpoint`, `drv_resume_reinitialises_optimiser`, `drv_resume_identity_unchecked`, `drv_identity_without_execution`, `drv_identity_without_commit`, `drv_seed_unchecked` |
| Task: data | Pins complete chunks that touch a split (`run_swg_online_training.py:309`, `run_swg_online_training.py:336`); a failed start leaves no run directory, and a start killed after the directory appears leaves its log (`run_swg_online_training.py:714`) | `test_run_swg_online_training.py:238` pending, archived and out-of-split chunks are not pinned, a mislabelled chunk raises; `test_run_swg_online_training.py:290`; `test_run_swg_online_training.py:303` | `drv_contract_reads_pending_chunks`, `drv_contract_pins_every_chunk`, `drv_staging_left_behind`, `drv_log_not_staged` |
| Task: GPU by index | `JAX_PLATFORMS` exactly `cuda` (or `cpu`), `CUDA_VISIBLE_DEVICES` the named GPU, backend agreeing (`run_swg_online_training.py:241`) | `test_run_swg_online_training.py:148` | `drv_gpu_alias_accepted` |
| Task: batch of ten in device calls | Mean of equal parts (`run_swg_online_training.py:431`) | `test_run_swg_online_training.py:486` | `drv_parts_summed_not_averaged` |
| Task: cost | `profile` (`run_swg_online_training.py:865`) | `test_run_swg_online_training.py:673` | `drv_profile_epoch_without_validation` |

Reversions: 43, each run against the tests named for it in `mutants.py`;
all 43 exit 1 with the intended failure (`reversions.jsonl`; for example
`drv_resume_reinitialises_optimiser`: checkpoint digest `7dbc22e4...` against
`1d62bf4c...`). Earlier passes are kept and superseded: in the first, the
mutated driver lacked its sibling `run_swg_double_gyre.py`, so four driver
mutants failed on that missing file rather than at their assertion (the
runner now copies the sibling), and the pending-chunk mutant crashed before
its assertion (it now pins the manifest it reads); a later pass was stopped
to land two driver changes. Only the final pass is reported.

## Measurements

**Eq. 6 network scales** (`m01_eq6_scales.py`, results `m01_eq6_scales.json`).
Derivation in the design note and at `swg_online.py:381`. From the deck
(`beta = 2e-11`, `L = rn_domsiz_m/2 = 2,000 km`, `f0 = 7e-5`,
`tau0 = 0.1 Pa`), the card (`rho0 = 1026`, `g = 9.80665`, `D = 500 m`) and
the closure geometry (`dx = 50 km`):

| Scale | Value |
|---|---|
| `|beta| L` (Eq. 6a, input vorticity) | 4.000e-5 s-1 |
| `alpha_out = |tau0| pi/(rho0 D L)` (Eq. 6b) | 3.062e-13 s-2 |
| velocity `|beta| L dx` | 2.000 m s-1 |
| SSH `|f0| |beta| L dx^2 / g` | 0.7138 m |
| potential `alpha_out D dx^2` | 0.3827 m3 s-2 |

Ten real coarse states: absolute days 731, 749, ..., 893 (every 18 days) from
the complete round-6 chunks, rebalanced as the contract does, the committed
16/4/3 network (7,696 parameters):

| Operator | Scaled SSH rms / max | Scaled u_T rms / max | Scaled v_T rms / max | Template input `zeta/(beta L)` rms / max |
|---|---|---|---|---|
| filtered (`template`) | 0.483 / 2.93 | 0.056 / 0.529 | 0.064 / 0.527 | 0.045 / 0.416 |
| box (`symmetric`) | 0.503 / 3.35 | 0.067 / 0.781 | 0.077 / 0.712 | 0.073 / 0.916 |

| Seed | Initial scaled potential rms / max (filtered) | Its five-point Laplacian rms / max | Initial acceleration, filtered / box (bar 1%) |
|---|---|---|---|
| 20261013 | 0.027 / 0.058 | 0.0049 / 0.042 | 0.241% / 0.294% |
| 20261014 | 0.019 / 0.090 | 0.0065 / 0.056 | 0.352% / 0.431% |
| 20261015 | 0.102 / 0.137 | 0.0245 / 0.209 | 0.861% / 0.862% |

The scaled inputs are of order one where the flow is strong and match the
template's own dimensionless input on the same states, as the derivation
intends. The initial output is of order 0.01--0.1: one percent of the wind
acceleration corresponds to a one-cell scaled-potential difference of about
0.127, so an initial output of order one would fail the round-6 bar. The
same networks with unit scales give 0.64--2.22%; seeds 20261014 and
20261015 fail there.

**Gate on the end-to-end test card** (`m02_small_gate.py`): on the 1,000 km
card the 1% bar is four times tighter in scaled units; only the 16/4/3
network passes, for seeds 2, 4, 7, 10 and 11 of 0..11 (seed 11: 0.46%;
seed 0: 5.0%, used by the failing-gate test).

**End-to-end CPU run** (test card, two window lengths, two epochs each, killed
while writing its second checkpoint and resumed): six optimiser steps; the
validation loss on the same fixed windows drops from 0.5229 to 0.5123 (one
day) and from 1.7223 to 1.6981 (two days); the resumed epoch's checkpoint
equals what the killed process was writing.

**Cost** (`m05_profile_cpu.json`, the driver's `profile`, batch 10, real dg50
card; other users' jobs kept the load average at 11--26; table in the design
note): one optimiser step 188 s (1 day), 390 s (2), 768 s (4), 1,507 s (8);
one validation batch 59, 119, 236, 469 s; peak resident memory 3.8, 4.7,
6.1, 8.4 GiB; compilation 25--43 s per window length. Five epochs cost
96.5--101.9 h at every window length, 396 h per seed. A free-run day costs
4.2 s in the profile and 4.5 s timed inside one run (`m07_free_run_day.py`),
so the four 20-year runs add 34--36 h. Loading and rebalancing cost about
0.5 s per batch on the real chunks (`m06_batch_overhead.py`), about half an
hour per seed. One seed: about 430 h on this CPU. Expected on one V100S:
about 94 h (3.9 days) if the training program gains the 4.6x that the fine
model's compiled forward day gains (GPU 4.49 against CPU 20.73 ms per step;
PLAUSIBLE, a different program); one candidate's three seeds on the two GPUs
about 8 days. The operator's `profile --device cuda` replaces this estimate;
the profile's free-run day was timed afterwards by a different method (see
UNVERIFIED).

## Outside this task's files

- `profile_swg_online.py:92` builds its optimiser with
  `create_swg_optimizer(TrainingConfig(...))`. The strict wrapper refuses a
  `TrainingConfig`, so `run_profile` now raises `TypeError` there at run time
  (its unit test returns before that line). Fix: pass
  `SWGOptimizerSettings(optimizer="adamw", learning_rate=1.0e-3,
  schedule="constant", weight_decay=4.0e-3, adam_b1=0.9, adam_b2=0.999,
  adam_epsilon=1.0e-7, adam_epsilon_convention="keras_epsilon_hat",
  gradient_clipping="none")` and import it instead of `TrainingConfig`. The
  driver's `profile` now measures what that script measured, with the real
  optimiser step.
- `nemo_testcase_l1_swg_online_reference_adequacy.py:102-103`: for rotational
  velocity the arrays are (day, component, wet point) and `axis=1` reduces
  over the two components, so each day's NRMS is 6,400 per-point numbers and
  "first day above 0.1 = 1562" is a flattened index. Reducing over every
  non-day axis (`axis=tuple(range(1, left_flat.ndim))`) and re-running the
  operator's command settles the 3.125 km verdict; SSH (0.36--0.42%) and
  vorticity (0.44--0.50%) already pass, and every year-1 statistic passes.
- The held-out scorer (codex findings 3 and 4): the driver's free-run
  manifests now name the run, seed, window length, checkpoint digest,
  objective and contract digest, and every day file carries the snapshot
  label. Refusing two seed labels on one root or one checkpoint digest, and
  refusing a file without the label instead of defaulting to snapshots, are
  the scorer owner's changes.
- Two ratchet failures exist on the base commit:
  `tests/test_no_private_cross_imports.py` flags `swg_online.py`'s import of
  `_entry_from_state` (base line 29; this round adds a use, not an import),
  and `tests/test_no_hardcoded_constants.py` flags `273.15` in
  `tests/unit/test_jra55_do.py`.

## Decisions for the owner

1. Which candidate trains first: (1) `template`+`excluded`, (2)
   `template`+`weighted` (Eq. 7 as written), (3) `symmetric`. My pick: (1).
   Deciding factor: the first interior ring is 5.0% of the 6,241 weighted
   points; it carries 1.8--3.6% of the no-closure loss under `symmetric`
   but 13.9--19.6% under `template` (round-6 table), three to four times its
   share. That points (PLAUSIBLE, not tested) at the filtered target's wall
   imbalance rather than at dynamics a closure should learn.
2. The symmetric configuration uses `weighted`; `excluded` is one edit.
3. Confirm the Eq. 6 analogue scales, including `rho0 = 1026` from the card
   (NEMO's, which converts the stress) rather than Table 1's 1000 (2.6% in
   the output scale). My pick: confirm.
4. Seeds 20261013--15. Seed 20261015 reached 0.86% of the 1% bar on ten
   real starts and could stop on its own first batch; replacing a seed that
   fails is a recorded choice. My pick: keep them, replace only on a failure.
5. `windows_per_call` on the GPU: the largest of 10, 5, 2, 1 that the
   operator's GPU `profile` fits at 8 days.

## Choices made

ASKED (the task): the data contract and its gap; training on the first 80%
of the 12 years; windows 1, 2, 4, 8 with five epochs, batch 10, offset
redrawn per epoch and window length, shuffled; network carried, optimiser
re-initialised; Keras AdamW at a constant rate through a strict wrapper
that refuses the factory's defaults; objective and wall ring from the
configuration with no default, one configuration per candidate, no start
without a configuration; rebalanced starts; Eq. 6 scales from the deck
(beta, L = half the domain, tau0, rho0, D); at least three seeds, each a
separate resumable run; the no-inert gate at the first step and the 1% bar
on ten real starts; per-epoch atomic checkpoints and resume; the JSON loss
log; a 20-year free run per window length in the harness's format with its
identity; GPU by index with `JAX_PLATFORMS=cuda`; the CPU cost and the GPU
expectation; the CPU end-to-end test.

UNASKED, each offered for revert:
1. The analogue's form: scales set by the coarse spacing `dx`, the SSH
   scale through geostrophy (adds `f0` and `g`), `rho0` from the card.
2. `weighted` for the symmetric candidate.
3. Seeds 20261013--15; one seed drives both the network initialisation and
   the window sampler; the rollout key `PRNGKey(seed)` is inert.
4. Start gates on the first training batch's ten starts; the Eq. 6 report
   recorded there without a bar.
5. Abort on a non-finite loss or gradient element at any step.
6. Logged training loss = each batch at its pre-update parameters;
   validation loss = mean over every validation batch at the epoch's end.
7. The `windows_per_call` execution setting (part of the run identity).
8. Free runs integrate `swg_online_step` (the trained composition), run in
   the training process after each window length, continue after a
   non-finite day, record their own timing, and an interrupted one is
   archived and restarted.
9. The contract pins complete chunks that touch a split (closed day
   intervals); chunks being written are skipped; a complete chunk whose
   label disagrees with its manifest raises.
10. A failed first start leaves no run directory; resume refuses a change of
    configuration, commit, device type or `windows_per_call` (not of the GPU
    index).
11. Native transcendentals on the GPU, libm on the CPU (the existing
    driver's two modes).
12. `train` refuses modified tracked files; `profile` stamps `-dirty`.
13. Configurations as strict-schema JSON in `config/swg_online/`.
14. The CPU end-to-end test's card (1,000 km, one-hour step, gravity
    0.05 m s-2 to keep the RK3 bound) and its seed 11.
15. The `check-data` command, which reads one test batch per window length.
16. The profile's rest-state starts and `floor(days / w)` window counts.

## Tests

Final runs on the final code, one file group at a time:

- `tests/ml/test_run_swg_online_training.py`: `39 passed in 387.82s`.
- `tests/ml/test_swg_online.py`: `54 passed in 120.68s`.
- `tests/ml/test_create_optimizer.py tests/ml/test_profile_swg_online.py`:
  `12 passed in 8.32s`.
- `tests/test_scripts_layout.py tests/test_no_hardcoded_constants.py
  tests/test_no_private_cross_imports.py`: `2 failed, 5425 passed, 2 skipped`;
  both failures are the base commit's (above).
- `tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py`:
  `2 failed, 15 passed`; the gate refuses to stamp the uncommitted tree
  ("3 tracked file(s) modified"), as in round 6.
- `ruff check` on the four changed Python files: `All checks passed!`.
- Citations: `check_citations.py` on this receipt and the design note,
  `CITATIONS 57 checked, 0 failing`; a copy with one citation moved by two
  lines gives `1 failing`. The lane's citation gate maps none of these
  files and refuses an uncommitted tree.

## Review

UNREVIEWED. By instruction no external review tool was run here; the
operator runs one review of the merged round.

## UNVERIFIED

- Nothing ran on a GPU. That `JAX_PLATFORMS=cuda` yields
  `jax.default_backend() == "gpu"` was read in JAX 0.10.0's `xla_bridge.py`
  (`cuda` is an alias of the `gpu` platform) and matches the GPU smoke's
  recorded backend; it was not executed with that setting. GPU memory for
  ten 8-day windows in one call is unknown.
- The GPU wall time is the CPU measurement divided by a ratio measured on a
  different program (the fine model's compiled forward day).
- The CPU timings ran beside other users' jobs (load average 11--26 during
  the profile, up to 120 later).
- The profile timed a free-run day as the difference of a three-day and a
  one-day run; that method was replaced afterwards by a timer inside one run
  because it fails when days take milliseconds. On the real card the new
  method gives 4.5 s against the profile's 4.2 s.
- The full 12-year contract has not been built (record incomplete): six real
  chunks went through the data path. The test year's chunk label is assumed
  to be `dual`, like the 12-year record's.
- Each seed's 1% gate on its own first batch is unknown until the run
  starts; the numbers above are on ten other real starts.
- No lock prevents two processes from using one run directory.
- The scale report's Laplacian equals the template's output only for
  `h = D`; the live thickness differs by the SSH, at most about 0.5% here.
- The Keras version Maddison used is not stated; the Keras-convention update
  is round 6's transcription of Keras 3.6.0.
- The three held-out-gate findings listed as open in the design note are the
  review's; they were not re-measured here.

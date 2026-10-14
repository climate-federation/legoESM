# NEMO testcase L1 SWG online learning — Round 5 receipt

Date: 2026-10-13

Incoming tip: `ba30b61d80dd6ecb26c0693d268fa319d4fb72f4`

Writable clone:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-313834`

## Verdict

**LANDED, NO TRAINING.** The pre-training implementation now has two explicit
objective operators with no default, a continuity-commuting filtered state
restriction, a split-owning content-pinned dataset contract, exact template
AdamW controls, fail-closed physical/native-grid gates, and rotational-velocity
reference adequacy. Only the pre-existing one-step seam test updates parameters.

The scientific selection between the objectives is not guessed. Ten dual
four-day acquisitions must apply the registered 8/10 start-admission rule.
The 3.125 km run was still pending at 50/365 days and its 50 km restriction did
not exist, so reference adequacy is **UNMEASURED**. Those acquisitions and the
schema-v2 multi-year hourly record remain prerequisites to the first training
run.

## Filtered restriction

The new path filters fine T-cell volume and native face transports, never three
independent velocity components. Scalar volume is even at each wall. U is even
at its tangential south/north walls and odd about its zero west/east faces; V
uses the opposite parities. Applying the same separable Gaussian to this
extended C grid makes the finite-volume divergence commute with filtering.
Normal wall transport remains exactly zero, and no mean-restoration patch is
applied. Standard deviation is 8 fine cells = 50 km; exact conservative
restriction follows filtering.

On pinned real day 1095 the resulting coarse divergence commutator is
`2.569395036157115e-18 m s-1` maximum in the interior and
`2.9288017035889553e-18 m s-1` on boundary cells. Both normal wall transports
are bitwise zero and relative total-volume error is `2.5e-16`. For a 64-cell
plane wave the analytic factor is `0.7346029443286334`; the implementation gives
`0.7346029450593634` (relative error `9.95e-10`). The +1 m temporal-continuity
plant and even-normal-reflection plant fail.

The GPU driver's hourly schema is now v2. One fine compiled scan emits t0 and
24 right endpoints/day for both `box` and `filtered`; v1 readers remain
box-only. A synthetic production-path test gives identical final prognostic
words with the sampler off and with both streams on. The GPU acquisition still
must verify the real next-day restart word-for-word. The chunk wrapper skips
completed chunks, archives rather than deletes an interrupted chunk, and does
not set or unset `XLA_PYTHON_CLIENT_MEM_FRACTION`.

## Objective and data contract

Objective selection is mandatory:

- `template`: t0 and target are the fine-grid filtered restriction; model
  vorticity is unfiltered.
- `symmetric`: t0 and target are the exact box restriction; both diagnosed
  vorticities receive the coarse Gaussian. This is explicitly a departure from
  the template.

Both are zero at the stored window start. For a two-dimensional Nyquist
checkerboard the coarse Gaussian's one-axis amplitude is
`0.014386770893472086`, so the analytic symmetric/template squared-loss ratio
is `4.284037960448793e-08`; the real ten-start rises and target top-octave
shares await the dual acquisition. The loss owns the complete Eq. 7 reduction:
interior `dx^2`, the `1e4/(beta^2 L^2)/(4L^2)` scale, and uniform means over
hours and batch windows. Rotational velocity remains an explicit option, off
for experiment one.

The dataset contract pins every source root and manifest digest. It loads
eight-day or longer v2 streams, reconstructs t0 from stored T/U/V at any
offset, enforces train/validation/test bounds separated by at least 720 hours,
and computes normalizers only through its training-split method. Each epoch
redraws a seeded offset, constructs non-overlapping 1/2/4/8-day windows,
shuffles them, and yields batch 10. Validation-hour injection, a split-crossing
window, a test-split normalizer request, an unpinned manifest, and ambiguous
coverage all raise. The real schema-v1 `start_day0731` manifest
`078f9c57d4a8e569303e41b5647eb339252ee1128b239bffa41acdf6efc349c6`
loads all 192 hours as labelled box-only data and is refused for `template`.
At the status check 24 of the operator's v1 eight-day segments were complete;
no partial segment was read.

The planned contract uses the 12-year interval `[day731,day5111)`: 3,504 days
of training `[731,4235]`, a 30-day gap, validation `[4265,5111]`, another
30-day gap, then the independent year `[5141,5506]`. This is 4,745 fine model
days total. At the measured 8 s/day it costs about 10.54 GPU-hours serial or
about 5.27 hours when evenly split across two available GPUs.

## Optimizer, health, and controls

`create_optimizer` now accepts explicitly selected constant versus historical
warmup/cosine scheduling, clipping on/off, and Adam epsilon. Its existing
defaults reproduce the previous update and state bit-for-bit. Experiment one
passes constant `1e-3`, AdamW weight decay `0.004`, beta1/beta2 `0.9/0.999`,
epsilon `1e-7`, and no separate clipping; optimizer state is reinitialized at
each window. Optax 0.2.8's installed `adamw` source was checked, and the first
update matches a hand calculation.

Physical health is now conjunctive with every statistical gate. Non-finite
fields, non-positive thickness, or Courant at least one cannot be rescued by
perfect skill. KE spectra and second-difference noise are computed from native
face velocities; enstrophy is computed directly from native vorticity. A
phase-flipped 0.1 m/s face checkerboard goes through the real scorer and fails.

The recomputed pinned years-2--3 controls are:

| Control | Face-transport RMSE | DST KE L1 | Wet EKE NRMSE | EKE/truth | Native top KE / enstrophy / grid index |
|---|---:|---:|---:|---:|---:|
| `nu=10` | `1.439733e6` | `0.223004` | `0.393673` | `0.732341` | `0.0538625 / 0.435918 / 0.931792` |
| `nu=80` | `2.007138e6` | `0.386407` | `0.773158` | `0.396429` | `0.0275464 / 0.294723 / 0.706924` |

The frozen better-skill and worse-noise bars are therefore the `nu=10` values
for these fields. Collapsed variance, two-cell grid noise, noise removal without
eddy recovery, and eddy recovery without skill all return `pass=false`.
Scorecard:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/control_scorecard.json`,
SHA-256 `6309446a91f63c1e4edb4a720d1b353c6d1fef97ae15fb5636980f6c89c8a59c`.

## Round-4 review disposition

| Review item | Disposition | Evidence: command and decisive output | Scratch reversion that fails |
|---|---|---|---|
| 1. Split-safe dataset/batches | Fixed in code; data acquisition open | Focused loader/contract tests: `2 passed`; real v1 length `192` | `reversion_objective_dataset.log`: leaked validation hour `DID NOT RAISE` |
| 2. Wrong symmetric-only objective and missing batch reduction | Fixed: two required names, template model unfiltered, loss owns `1/N_windows` | Objective/batch tests pass; checker response differs by >100 and mixed batch is exactly half | `reversion_objective_dataset.log`: four tests fail, including equal checker response and wrong batch reduction |
| 3. Physical health ignored | Fixed unconditionally | Gate test supplies non-finite/negative-thickness/Courant-2 health and returns failure | `reversion_health_native_noise.log`: otherwise-perfect unhealthy score passes |
| 4. Native Nyquist noise hidden | Fixed on native U/V through the real scorer | Phase-flipped 0.1 m/s plant exceeds `0.931792` and gate fails | `reversion_health_native_noise.log`: native index forced to zero and plant fails its assertion |
| 5. Optimizer mismatch | Fixed without changing defaults | `9 passed`; hand first update and historical bitwise update | `reversion_optimizer.log` and `reversion_optimizer_epsilon.log`: schedule, fail-closed, and epsilon tests fail |
| 6. Adequacy uses native rather than rotational velocity | Implementation fixed; science result open | Loader calls the rotational reconstruction eight times and exposes only `rotational_velocity` | `reversion_adequacy_rotational.log`: `len(calls) == 0` |
| 7. 30-day composition debt | Carried, not relabelled; user approved proceeding | Maximum remains `5.218048215738236e-15`, first bar violation day 9, bar unchanged | No reversion: this is recorded debt, not a code claim |
| 8. Potential orientation undocumented | Fixed at implementation term and design | Stage suite exercises signs/corners: `20 passed`; convention is east/north clockwise rotation | Existing directional force plants disagree if either sign is reversed |
| Round-5 filtered restriction | Fixed and measured on real data | Real maxima `2.57e-18/2.93e-18 m s-1`; walls exact zero | `reversion_filter.log` and `reversion_real_filtered_continuity.log`: even-normal reflection gives `9.12e-4 m s-1` |
| Round-5 dual sampler/admission | Built; GPU measurement open | Sampler/missing-operator/wrapper selection: `3 passed`; rule is exactly 8/10 | `reversion_driver.log`: three tests fail; `reversion_decision.log`: 7/10 incorrectly exits 0 |

All reversion logs are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/`.
Every test added in this round was exercised against a scratch mutation:
optimizer default/schedule/epsilon (3), objective and batch reduction (2),
dataset and v1 labelling (2), sampler/wrapper (3 including the modified parity
test), filtered restriction (4 synthetic plus 1 real), the 8/10 decision (1),
physical/native gates (2), and rotational adequacy (1). Each named violation
produced a pytest failure; no source tree used for the landed commits was
mutated for this exercise.

## Reference adequacy and OPEN work

At the last read, only
`dg3125_gpu_365d/manifest.pending.json` existed with `status=pending`,
`completed_days=50`, `days=365`, SHA-256
`8de1e2e56e92b53a0ae3d191a118d4884a3d9912479f13a222268a7dff72a464`;
`dg3125_conservative_dg50_365d` was absent. The fail-closed adequacy tool now
compares SSH, vorticity, and preregistered rotational velocity over the early
common path, but no result is claimed.

The dual-start measurement is also open, so neither target top-octave share nor
the real 0.1 m/s checkerboard rise is reported. The analytic ratio above is not
a substitute. Objective selection remains automatic from the registered
8/10 rule; it is not a user decision.

## Exact acquisitions

Run from the clean committed clone. Either physical device may be substituted
when idle; never kill a running job.

1. Generate ten dual-operator four-day segments, then apply the decision rule:

```bash
TRUTH=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_1095d OUT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_10x4d; mkdir -p "$OUT"; for day in 731 771 811 851 891 931 971 1011 1051 1091; do next=$((day + 1)); target=$(printf '%s/start_day%04d' "$OUT" "$day"); test -f "$target/manifest.json" && continue; XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" scripts/run/run_swg_double_gyre_gpu.sh 0 --deck maddison --days 4 --restart "$(printf '%s/day%04d.npz' "$TRUTH" "$day")" --hourly-targets --verify-next "$(printf '%s/day%04d.npz' "$TRUTH" "$next")" --output "$target" || exit; done
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_balanced_start.py --dual-operator --hourly-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_10x4d --start-days 731 771 811 851 891 931 971 1011 1051 1091 --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/dual_operator_admission.json
```

The analyzer exits 0 only when filtered restriction passes at least 8/10 and
selects `template`; exit 1 is the measured selection of `symmetric`.

2. Generate the 12-year record in two resumable 30-day-chunk processes, then
the held-out year after the registered gap. Completed chunks are skipped and
interrupted chunks are preserved as `.interrupted[.N]`. These commands preserve
the caller's memory fraction:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" scripts/run/run_swg_online_hourly_chunks.sh 0 731 2921 30 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_12y dual
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" scripts/run/run_swg_online_hourly_chunks.sh 1 2921 5111 30 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_12y dual
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" scripts/run/run_swg_online_hourly_chunks.sh 0 5141 5506 30 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_test_year dual_test
```

Each command must wait until every requested restart is listed by the truth
manifest. The wrapper and driver refuse an unlisted day.

3. After the one-year sensitivity manifest is complete, restrict and run the
redesigned adequacy gate:

```bash
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python packages/ocean/legoesm/ocean/fidelity/swg_coarse_graining.py --source-deck dg3125 --input /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_gpu_365d --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_conservative_dg50_365d
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_reference_adequacy.py --candidate-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/conservative_truth_dg50_1095d/snapshots --sensitivity-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_conservative_dg50_365d/snapshots --years 1 --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/reference_adequacy.json
```

## Validation and commits

Implementation commits are `a01d68f4f`, `92c359ab2`, and `fdf1ff100`; design
and the initial receipt are `fa08b259e`. The final affected CPU battery is
`93 passed`: coarse-graining/driver `36 passed in 199.03s`, ML optimizer/loss/
dataset `23 passed in 204.85s`, and balanced-start/evaluation/adequacy/RK3-stage
`34 passed in 39.88s`. Targeted Ruff reports `All checks passed!`.
`compileall -q`, `git diff --check`, and shell syntax checks emit no output and
exit zero.

The clean-tree oracle citation gate at `fa08b259e` reports `PASS`, three
citations, zero failures, and all internal self-plants fired. Shifting the
registered first RK3 citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE`. Evidence is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/citation_gate.json`
and `citation_gate_plant.json` beside it.

No training driver, stochastic rollout, or optimization experiment was added.
Stochastic readiness remains interface-only, and the energy-score definition
from Round 4 is unchanged.

## Round 5b repair

The registered double-gyre climate instrument is restored to its branch-point
behaviour. The branch point is
`8f8551c8734947ff5b1489eb169f3cfd4cdfcaf2`, the parent of the first
`swg_online` commit (`23bc6d0cb`) and the last commit before this lane changed
the instrument. The last double-gyre-lane commit on the file was `09671b614`.
The repair is `3461a66d6`.

The five commits named by the operator changed an existing registered caller
as follows:

| Commit | Effect seen by an existing caller before this repair |
|---|---|
| `23bc6d0cb` | Generalised the provider and made the default no-geometry diagnostic add thickness and Courant outputs; the CLI roots also ceased to be argparse-required. |
| `fe9596c60` | Replaced dry-cell zeros by NaNs in five maps and added enstrophy, top-octave, and checkerboard-health outputs to the default metric set. |
| `18862d4cf` | Added the `nu=10`/`nu=80` online control selector; it made no further numerical change to the registered NEMO/legoESM arm. |
| `a9d98be95` | Replaced the registered periodic isotropic KE spectrum by the closed-basin sine-transform spectrum and replaced the checkerboard projection by a second-difference index. |
| `a01d68f4f` | Replaced cell-centred online spectra/noise by native-face versions inside the shared default; callers without a card then failed before producing a report. |

The history also contains interstitial commit `68646355a`, not one of the five
listed in the addendum. It transiently changed the added enstrophy derivation;
`a9d98be95` superseded that implementation, so it leaves no independent
tip-versus-branch-point delta.

`diagnose(..., diagnostics="registered")` is now the unchanged default and
contains the branch-point metric implementation: cell-centred maps, periodic
isotropic KE spectrum, and exactly the original twelve output arrays.
`diagnostics="online"` owns the sine-transform streamfunction spectrum,
direct native-vorticity enstrophy, native-face spectra, native-face
second-difference noise, and physical-health fields. The online evaluation
harness passes that selection explicitly. Both the Python and CLI dispatchers
raise on an unknown selection, and the filtered/coarse CLI arm refuses to run
unless `--diagnostics online` is stated.

The requested full years-2--3 replay was stopped before output creation after
the registered Python tile assembler proved too slow. The authorized year-1
fallback completed against the same raw NEMO and legoESM daily-mean inputs:

```text
PASS year 1, spin-up -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/round5b_registered_replay_y1
metrics_json_exact True
comparison_json_exact True
array_keys_exact True
array_ieee_words_exact True unequal []
metric_rows 24
comparison_rows 12
```

The comparison target is the committed fp64 year-1 report at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round2/climate_year1_daily_mean_fp64/`.
Thus every metric summary and comparison row is JSON-exact, and every stored
floating-point word is identical. The non-fp64 sibling report is deliberately
not the target: its NEMO cell-centred fields were generated under a different
precision policy.

The online controls were independently regenerated through the explicit
online path at `round5/control_scorecard_round5b.json`. The JSON is byte-for-
byte equal to `control_scorecard.json`, including SHA-256
`6309446a91f63c1e4edb4a720d1b353c6d1fef97ae15fb5636980f6c89c8a59c`.
The frozen `nu=10` values remain EKE/truth
`0.7323411189244107`, sine-transform KE error
`0.22300407134605818`, and native noise
`0.9317922221566435`; `nu=80` remains
`0.3964287848321674`, `0.3864065874533908`, and
`0.7069240384788047`, respectively.

The new full-default synthetic JSON pin was exercised against a scratch
reversion. Changing the default selector from `registered` back to `online`
made the pinned test fail with
`GateError: online diagnostics require a card and resolved namelist` and the
terminal line `1 failed in 1.44s`. Reapplying the repair produced
`SCRATCH_REVERT_CLEAN`; the landed focused selection reports
`17 passed in 25.24s`.

The exact full lane battery from `autopilot.sh` collected 283 tests and ended:

```text
282 passed, 1 skipped in 2323.82s (0:38:43)
```

Targeted Ruff reports `All checks passed!`; `git diff --check` emits no output.
No filtered-restriction, objective, dataset-contract, optimizer, or acquisition
code changed in this repair.

## Round 5c GPU path

The operator measured that every cuDNN convolution on the installed V100S,
JAX 0.10.0, and cuDNN 9.21.1.3 path failed at execution.  The cause is not
claimed.  Commit `850672081` therefore removes convolution primitives from all
three programs that the next acquisition lowers: the dual hourly sampler, the
closure model and training objectives, and the cost profiler.  No objective,
dataset, optimizer, or evaluation definition changed.

The closed-C-grid filter now uses one shared array implementation for both the
offline tool and in-scan sampler.  Each Gaussian axis is a weighted sum of
shifted slices after the registered even/odd parity extension.  On the pinned
fine day 1095, the final coarse T/U/V state differs from Round 5's dense 97 by
97 result by `1.1937117960769683e-12 m`,
`2.6645352591003757e-15 m s-1`, and
`1.887379141862766e-15 m s-1`, respectively.  The continuity residual is
`4.459310829938671e-18 m s-1` in interior cells and
`3.2500654186147504e-18 m s-1` on wall cells; normal wall transports are
bitwise zero and relative volume error is `2.5e-16`.  The existing continuity,
wall, volume, and transfer-function plants remain live.  CPU XLA reports the
fine dual sampler's filtered restriction at `30,587,136` temporary bytes,
`9,953,664` argument bytes, and `169,368` output bytes: a 40,710,168-byte
compiled live bound, rather than the failed 28.71-GiB dense request.  The GPU
smoke records the actual device peak before any long acquisition.

`ShiftedConv2d` retains each former Equinox layer's OIHW weight and bias shapes,
key split, uniform initialization, padding, and cross-correlation orientation.
The real five-layer network remains 7,696 parameters and agrees with the former
Equinox network in forward values and parameter gradients to rounding.  Its
ten-state maximum initial closure/wind acceleration ratio remains exactly
`0.007791671811074865`, below the registered 1% bound.  The symmetric
objective's 9 by 9 Gaussian is likewise two shifted 1-D sums and agrees with
the previous dense forward and gradient results to rounding.

The installed JAX source was read rather than assuming inspection APIs:
`jax._src.stages.Lowered.as_text()` supplies StableHLO text,
`Compiled.memory_analysis()` supplies compiler estimates, and
`Device.memory_stats()` supplies platform-dependent live and peak bytes.  A
single fail-closed guard rejects `stablehlo.convolution`, `mhlo.convolution`,
or textual `convolution(`.  Tests lower the real filtered restriction, both
real objectives, the 7,696-parameter model step, and the real dual sampler.
Each also lowers an actual planted `lax.conv_general_dilated`; bypassing the
guard makes all four tests fail with `DID NOT RAISE`.

The committed GPU smoke requires a physical GPU and
`XLA_PYTHON_CLIENT_MEM_FRACTION`, refuses a dirty tree or non-GPU backend, and
runs three fresh workers so peak statistics are per task.  It compares one
pinned fine restart's box and filtered samples between CPU and GPU, then runs a
reverse-mode 24-hour real-network window for each objective and requires every
parameter-gradient leaf to be finite and nonzero.  It prints and stores wall
seconds, compiler memory, and GPU `peak_bytes_in_use`.  GPU execution is
**ACQUISITION_NEEDED**; no GPU was used in this CPU-only round.

A CPU 1-day template forward profile exercised the final compiled path:
objective `2.2992715948960706e-05`, execution `6.758193481713533 s`, and
compiled temporary `64,135,904` bytes.  It was a profile only, with no update.
All eight GPU profiles must follow the measured objective selection and now
take the physical device explicitly.

### Round 5c live reversions

Every new test was run against a scratch mutation; logs are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/`.

| Contract | Scratch reversion | Decisive output |
|---|---|---|
| Lowered filter, symmetric objective, network, and sampler contain no convolution | Make the shared guard inert | `round5c_reversion_guard.log`: `4 failed`; every real `lax.conv_general_dilated` plant reports `DID NOT RAISE` |
| Fine filter preserves the dense result and analytic transfer | Scale every shifted kernel weight by 1.0001 | `round5c_reversion_numerical.log`: transfer `0.7347498729944048` versus `0.7346029443286334`; real T difference becomes `0.10030584733323167 m` |
| Network preserves Equinox initialization and the initial-force bound | Scale the initializer bound by 1.01 | same log: weight words differ and ten-state ratio becomes `0.008210819625129451`; `4 failed` total |
| GPU smoke comparison and memory-fraction refusal fail closed | Set comparison tolerance to 1 and bypass the fraction check | `round5c_reversion_smoke.log`: comparator plant `DID NOT RAISE`; missing-fraction assertion fails; `2 failed` |
| Profiles identify shifted layers and require a matching physical GPU | Restore the old metadata and bypass GPU selection | `round5c_reversion_profile.log`: missing shifted-layer key and missing exception; `2 failed` |

The focused final-state verification is `60 passed in 328.15s (0:05:28)`.
Targeted Ruff reports `All checks passed!`; `compileall -q`, shell syntax,
and `git diff --check` exit zero without output.  The exact full lane battery
and two independent review verdicts are recorded below after they complete.

### Round 5c exact acquisitions

Run these from the clean committed clone, in order.  Physical GPU 0 is explicit
in every GPU command; substitute one other idle physical index consistently if
needed, and never kill a running job.

1. Smoke the real sampler and both 24-hour reverse objectives first:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_gpu_smoke.py 0 --restart /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_1095d/day0731.npz --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/gpu_smoke_gpu0.json
```

2. Archive the ten failed cuDNN attempts, regenerate the ten dual-operator
segments, and apply the registered objective-selection rule.  Analyzer exit 0
selects `template`; exit 1 is a valid measured selection of `symmetric`; any
other exit refuses the sequence.

```bash
TRUTH=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_1095d OUT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_10x4d; mkdir -p "$OUT"; for day in 731 771 811 851 891 931 971 1011 1051 1091; do next=$((day + 1)); target=$(printf '%s/start_day%04d' "$OUT" "$day"); if [[ -f $target/manifest.json ]] && grep -q '"status": "complete"' "$target/manifest.json"; then continue; fi; if [[ -e $target ]]; then archive=$target.cudnn_failed; suffix=0; while [[ -e $archive ]]; do suffix=$((suffix + 1)); archive=$target.cudnn_failed.$suffix; done; mv -- "$target" "$archive"; fi; XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" scripts/run/run_swg_double_gyre_gpu.sh 0 --deck maddison --days 4 --restart "$(printf '%s/day%04d.npz' "$TRUTH" "$day")" --hourly-targets --verify-next "$(printf '%s/day%04d.npz' "$TRUTH" "$next")" --output "$target" || exit; done
set +e; PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_balanced_start.py --dual-operator --hourly-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_10x4d --start-days 731 771 811 851 891 931 971 1011 1051 1091 --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/dual_operator_admission.json; admission_rc=$?; set -e; [[ $admission_rc -eq 0 || $admission_rc -eq 1 ]]
```

3. Profile forward and reverse for 1, 2, 4, and 8 days with the selected
objective:

```bash
ADMISSION=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/dual_operator_admission.json; OBJECTIVE=$(/home/dbalwada/legoESM/.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["selected_objective"])' "$ADMISSION"); for mode in forward reverse; do for days in 1 2 4 8; do env -u JAX_PLATFORMS CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" JAX_ENABLE_X64=1 PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src /home/dbalwada/legoESM/.venv/bin/python scripts/bench/profile_swg_online.py --window-days "$days" --mode "$mode" --objective "$OBJECTIVE" --physical-gpu 0 --output "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/profile/gpu0_${OBJECTIVE}_${mode}_${days}d.json" || exit; done; done
```

4. Generate the resumable dual-operator 12-year record and held-out year:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" scripts/run/run_swg_online_hourly_chunks.sh 0 731 2921 30 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_12y dual
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" scripts/run/run_swg_online_hourly_chunks.sh 1 2921 5111 30 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_12y dual
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:?set GPU memory fraction}" scripts/run/run_swg_online_hourly_chunks.sh 0 5141 5506 30 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round5/hourly_dual_test_year dual_test
```

## OPEN

- Acquire and score the ten dual starts; this selects `template` or `symmetric`.
- Acquire and freeze the schema-v2 12-year train/validation record and held-out
  year, then materialize the contract JSON with every final manifest hash.
- Complete and pass the 3.125 km adequacy gate before treating 6.25 km as truth.
- Build the resumable training driver only in the next round; no training was
  authorized inside this sandbox round.

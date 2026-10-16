# SWG online learning — round 4 receipt

Date: 2026-10-13

Status: **PRE-TRAINING CODE COMPLETE, ACQUISITIONS HELD**. No training was
performed. One content-pinned 24-hour AdamW update is solely the required seam
test. The conservative start, deterministic loss, data lineage, spectral and
evaluation gates, daily-transport path, and decision-tool exits are now
admitted. Training remains held because the one-year 3.125 km reference has
not completed and the available four-day target cache cannot supply the ruled
batch-10 eight-day stage.

Incoming tip: `d2445fb47`. Writable clone:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-261083598`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round4/`.
The binding round-2 and round-3 review logs have SHA-256
`e11d17abf36e9a95160c6f8c9b0f7ef9c57eaa6b6eb97ff5e91d7f3f2361cca9`
and `1cbe41bf41a2c3125982d6030cf33b17e91b15abaedacd762166599e47e5d357`.

## Loss, optimizer, and real data seam

The deterministic objective is Maddison's Eq. 7: the squared error of
sigma-one-coarse-cell Gaussian-filtered relative vorticity, at every hourly
right endpoint, with interior `dx^2` weights and scale
`1e4/(beta^2 L^2)/(4 L^2)/M`. Rotational-velocity terms are explicit but off.
The analytic scale test has a single-cell known answer and proves that a large
velocity error is ignored by the default objective.

The ensemble interface remains, but stochastic readiness is **interface only**.
The energy score now implements Brolly Eqs. 9--10 per lead with pair denominator
`2*S*(S-1)` and a uniform lead average. `S<2` is refused, and the non-identical
ensemble `{truth, truth+1}` has score zero. There is no member rollout and no
stochastic claim.

The ten acquired four-day segments are content-pinned as train days
731/771/811/851/891/931, validation 971/1011, and test 1051/1091. T0 is the
conservative day-N state and targets are the same segment's hours 1--96. Model
and truth call one balanced diagnostic. Training-only variances are
`2.6760339329767915e-12 s-2`, `0.011834588859409795 m2 s-2`, and
`0.013574072201870643 m2 s-2` for vorticity, rotational U, and rotational V.

The real seam test loads pinned day 731, rolls the 7,696-parameter model for 24
hours, evaluates the real loss, checks every trainable leaf has nonzero finite
gradient, and applies one AdamW update. It passes. Initialization seed 20261013
has maximum closure acceleration `1.5188444e-9 m s-2` over all ten real starts,
or `0.00779167` of `tau0/(rho0 H)`, below the registered 1% bar.

Maddison used the Keras AdamW defaults: learning rate `1e-3`, weight decay
`0.004`, beta1/beta2 `0.9/0.999`, epsilon `1e-7`, batch 10, five epochs per
window (paper text lines 464--530). `create_optimizer` expresses AdamW, LR, WD,
and the beta defaults; it necessarily retains Optax epsilon `1e-8`, cosine
scheduling, and its clipping transform. The difference is recorded, not hidden.

## Balanced start and predictability

The pre-registered admission is per start. It requires
`zeta_error_h1 <= 2*zeta_error_h24/24` and
`max(divergent_KE_h1:h6) <= 2*mean(divergent_KE_h25:h48)`. All 10 conservative
starts pass and all 10 raw starts fail. Conservative hour-1 vorticity error is
`1.067e-8` to `1.987e-8 s-1`; its early/late fast-energy ratio is `1.322` to
`1.742`. Truth hour-24 vorticity RMS is `2.160e-6` to `2.340e-6 s-1`. The old
raw-versus-conservative ratio is diagnostic-only. Report SHA-256:
`858d8653d96fe17d4a543bf35043a415f1c63f84ff6edf3b40b526de484d9036`.

Both 120-day balanced twins completed. The registered rule selects the first
lead after which divergence/vorticity stays below 0.07 with at least 20 points.
It gives `10.281568 d` at amplitude `1e-11 s-1` (`R2=0.95698`) and
`13.620549 d` at `1e-10 s-1` (`R2=0.88488`), both over days 2--120. The 32%
spread is retained; no average is promoted. The zero initial SSH of a balanced
perturbation is now correctly reported as an undefined SSH crossing rather
than refused. Report hashes are
`559da524cf1f247c39d4d88136b1991f999c79b4dd0482610500a0aabf5e9522`
and `342eba8ec1a2e065fc526c84bd93656f750157ea84a6c509dbf1baf49b738dac`.

## Conservation, spectra, and held-out controls

New daily-mean runs accumulate instantaneous live `h*u` and `h*v` inside the
compiled day and restrict the mean transports. The filter refuses an archive
that supplies only `h(mean eta)*mean(u)`. A temporal continuity test closes at
`2e-22 m s-1`; its +1 m SSH plant produces at least `1/86400 m s-1` and fails.
The existing three-year daily-mean archive remains explicitly legacy/caveated;
it was not relabelled.

The restarted-next comparator now checks float64 IEEE words. Its signed-zero
plant has zero absolute error, one unequal word, and fails. The closure-force
test fixes the U/V orientation signs, checks all four corners, and shows that a
V-sign reversal creates divergence at least 4 in its manufactured units. The
executed oracle materializes stage 1 at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:204-210`, records stage 2 at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:220-235`, and completes stage 3
at `SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:360,366,376`. The
absent-closure branch remains unchanged.

Periodic FFT spectra are replaced by a DST-I Dirichlet-streamfunction KE
spectrum, matching Maddison's paper-text line 670. Enstrophy is transformed
from vorticity itself. Grid noise is local second-difference RMS divided by
velocity RMS; both checkerboard phases score `2.82843`, so cancellation cannot
hide the plant.

The scorecard accepts arbitrary native `(time,y,x)` T/U/V arrays and uses that
same path for both controls. It checks finite fields, positive thickness, and
Courant below one; computes mean transport from live face transports; reports
raw per-year values and half-range uncertainty; and pins both control and truth
manifests in code.

| Control, years 2--3 | Face-transport RMSE (`m3 s-1`) | DST KE L1 | Wet EKE-map NRMSE | EKE/truth | Top KE / enstrophy / grid noise |
|---|---:|---:|---:|---:|---:|
| `nu=10` | `1.439733e6` | `0.248509` | `0.393673` | `0.732341` | `0.0056396 / 0.0634475 / 0.564443` |
| `nu=80` | `2.007138e6` | `0.331588` | `0.773158` | `0.396429` | `0.0034951 / 0.0431900 / 0.466276` |

Both controls pass physical health. Collapsed variance, both-phase grid noise,
noise removal without EKE recovery, and EKE recovery without skill/noise
improvement all fail. Scorecard SHA-256:
`27c407800282fa3f016b92978d144e6ad996571993a06af143a3f85ad0228dfa`.

## Reference adequacy and profiles

The adequacy tool is redesigned for the queued one-year sensitivity. It scores
days 1--8 from rest using NRMS <=10% and pattern correlation >=0.99 for SSH,
vorticity, and velocity and reports first 1%/10% crossings. Year-1 KE, EKE,
jet, and DST-spectrum scores use bars `max(10%, 6.25-km adjacent-year
variability)`. Inadequacy exits 1. The dg3125 acquisition had a pending
manifest with zero complete days when checked, so no partial file was read and
no adequacy verdict exists.

The corrected CPU profile uses the real network, Maddison loss, uniform
checkpointing, and AdamW plumbing without updating parameters:

| Window | Steps / observations | Forward s / peak KiB | Reverse s / peak KiB |
|---:|---:|---:|---:|
| 1 day | 216 / 24 | `2.323 / 1,140,328` | `7.039 / 1,964,800` |
| 2 days | 432 / 48 | `5.260 / 1,209,832` | `13.845 / 2,139,784` |
| 4 days | 864 / 96 | `8.574 / 1,379,060` | `29.181 / 2,507,060` |
| 8 days | 1,728 / 192 | `17.408 / 1,675,948` | `58.787 / 3,235,716` |

The operator's GPU profile attempt failed before model execution because
`JAX_PLATFORMS=gpu` selected unavailable ROCm; the corrected CUDA command is in
OPEN. The eight CPU JSON hashes are recorded in their evidence directory.

## Review finding dispositions

| Review finding | Disposition | Evidence |
|---|---|---|
| R2-1 spatial continuity missed wrong SSH | Fixed | Temporal continuity API and +1 m plant |
| R2-2 daily means used product of means | Fixed prospectively; legacy archive still labelled | In-scan live transport means; product input refused |
| R2-3 decision tools exited zero on failure | Fixed | Reference, start, and tangent CLI exit tests |
| R2-4 FFT/inferred enstrophy/cancelling checkerboard | Fixed | DST streamfunction, direct zeta, local phase-safe plant |
| R2-5 30-day composition debt | Recorded, not changed or re-raised | Round-3 composition table; no relaxed bar |
| R2-6 numeric equality accepted signed zero | Fixed | IEEE-word comparator and signed-zero plant |
| R2-7 one-year adequacy lacked uncertainty | Fixed | Early path plus three-year 6.25 km variability bars |
| R3-1 energy score equation wrong | Fixed | Per-lead `2S(S-1)` implementation and `{truth,truth+1}` test |
| R3-2 start gate asked the wrong question | Fixed | Pre-registered per-start gate; conservative 10/10, raw 0/10 |
| R3-3 no real batch/lineage/normalizers | Fixed for available cache; 8-day acquisition open | Pinned split, training-only variances, real rollout/update test |
| R3-4 stochastic member rollout absent | Operator ruled interface-only | Correct score interface; no stochastic claim |
| R3-5 evaluation could not score outputs or health | Fixed | Native-array path, frozen hashes/bars, health, uncertainty, live transport |
| R3-6 spectral diagnostics invalid | Fixed | DST/direct-zeta/local-grid-noise tests |
| R3-7 tests could not fail | Fixed | Analytic scale, covariance, +1 m, signed-zero, phase/sign plants |
| R3-8 force sign/corners undocumented | Fixed | Sign/corner/staggering tests and rewritten design note |

## Validation, source fidelity, and commits

The focused battery reports `87 passed in 193.80s`. Ruff reports
`All checks passed!`; `compileall -q` and `git diff --check` emit no output.
Every changed/new Python implementation has direct tests. The training module
is orchestration, not a new `physics/` Config, so no new physics-contract or
`__param_spec__` registration applies.

The source-fidelity basis is unchanged from the three cited oracle ranges
above; the closure is optional around the certified transcription, and absent
closure performs no zero addition. The round-4 force edits did not modify NEMO
sources or card parameters.

An additional independent-review invocation was attempted after the focused
battery. The installed `codex exec` failed before reading the change because
its in-process app-server could not initialize on the sandbox's read-only
filesystem. That is an environment limitation, not a PASS; the two binding
external reviews remain the independent review basis for this round, with all
15 findings dispositioned above.

Round commits before this receipt are `45a6756eb` (pre-registration) and
`a9d98be95` (implementation and tests).

## OPEN

1. The operator-owned dg3125 run is active under
   `round2/reference/dg3125_gpu_365d`; after its complete manifest and
   `dg3125_conservative_dg50_365d` exist, run the fail-closed adequacy command
   in the machine-readable acquisition line below.
2. Acquire 50 eight-day hourly segments at starts 731, 771, ..., 2691 from the
   manifest-listed growing truth. Freeze starts 0--29/30--39/40--49 as
   train/validation/test. This supports batch 10 and every 1/2/4/8-day window.
3. Repeat the eight profiles on an idle GPU with `JAX_PLATFORMS=cuda`; never
   kill the running dg3125 or truth jobs.

Exact operator commands (run only after confirming that physical GPU 0 is
idle) are:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-261083598
for k in $(seq 0 49); do d=$((731 + 40*k)); n=$((d + 1)); scripts/run/run_swg_double_gyre_gpu.sh 0 --deck maddison --days 8 --restart "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth/day$(printf '%04d' "$d").npz" --hourly-targets --verify-next "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth/day$(printf '%04d' "$n").npz" --output "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round4/hourly_targets_8d/start_day$(printf '%04d' "$d")"; done
for mode in forward reverse; do for window in 1 2 4 8; do CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src /home/dbalwada/legoESM/.venv/bin/python scripts/bench/profile_swg_online.py --window-days "$window" --mode "$mode" --output "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round4/profile/gpu0_${mode}_${window}d.json"; done; done
```

After the already-queued dg3125 run and restriction have complete manifests,
run:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-261083598
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_reference_adequacy.py --candidate-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/conservative_truth_dg50_1095d/snapshots --sensitivity-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_conservative_dg50_365d/snapshots --years 1 --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round4/reference/adequacy.json
```

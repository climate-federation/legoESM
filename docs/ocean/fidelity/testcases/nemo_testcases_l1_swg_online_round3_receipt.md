# SWG online learning — round 3 receipt

Date: 2026-10-13

Status: **BUILD COMPLETE, TRAINING HELD**. The differentiable closure seam,
network, balanced loss, 1/2/4/8-day segment runner, optimizer plumbing,
profiles, and held-out gates are implemented and tested. No optimization
update was taken. None of the three 30-day dg50 execution compositions is
bit-identical to NEMO, so phase-4 training awaits the user's explicit
composition-debt decision. The operator's round-2 GPU acquisition queue had
not produced a completed manifest when this receipt was written.

Incoming tip: `acadd2dee`. Writable clone:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-nANtlMUs`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round3/`.

## Coarse execution compositions

The unchanged numerical bar and exact IEEE-word test were applied separately.
The output now records the first reference word, candidate word, and ordered
ULP distance, including signed zero.

| Composition | Exact rows | Rows within `1e-15` | Maximum absolute difference | First numerical debt |
|---|---:|---:|---:|---|
| per-step ladder | 249/690 | 561/690 | `9.4785291e-15` | day 9, `sshn[1,1]`, 16 ULP |
| compiled day | 249/690 | 543/690 | `5.2180482e-15` | day 9, `sshn[1,1]`, 25 ULP |
| flat training segment | 249/690 | 543/690 | `5.2180482e-15` | day 9, `sshn[1,1]`, 25 ULP |

Every composition first differs bitwise on day 1 in
`vtau_before[60,1]`: NEMO stores `+0.0`, while legoESM carries `-0.0`, an
ordered distance of one IEEE word and zero numerical error. On day 9 the
per-step candidate at `sshn[1,1]` is `0.09109582693621130` against NEMO's
`0.09109582693621107`; compiled and flat produce
`0.09109582693621142`. The compiled-day and flat-segment summaries are
identical, refuting the operator's proposed bit-identical composition result.
No bar was relaxed.

The JSON hashes are:

- per-step: `2a06ebbe09ea0a7e897ec339d7fca4c7e6aaf03d3675a146740d4ce14c72b1f5`;
- compiled day: `cb11b8c81c0d8a7bde2dc6a56fc1db763141c161b991c7c38292a6d9c6d86dda`;
- flat segment: `4cd20ca4318f1690d83179c2c7dec63116acc435b4ca2cdbb63d274fdfe70872`.

The stage source still materializes and rotates stage 1 at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:204-210`, records stage 2 at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:220-235`, and completes stage 3
at `SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:360,366,376`. The optional
learned potential is evaluated at each of those three stage boundaries. The
ordinary `None` branch performs no addition, and the existing stage/card tests
remain bit-identical.

## Closure seam and network

The new pure function consumes one native F-point transport-tendency
potential in `m3 s-2`. Its matched differences produce native U/V transport
tendencies, divided by the live face cross-sections to obtain acceleration.
The discrete continuity divergence cancels at roundoff, the boundary-potential
mask gives zero wall-normal force, and both basin-integrated transport-force
components telescope to zero. Random-field tests cover all four claims; a
stage-level absence test proves the certified branch is unchanged.

The Equinox model has five width-3 convolutions, 16 hidden channels, ELU, an
11-cell/550 km receptive field, and 7,696 parameters. Inputs are SSH plus the
explicit noise realization, T-centred native U/V, and wet mask. Uniform U/V
metrics remain static geometry in the potential-to-face transform rather than
indistinguishable CNN channels. Output is one boundary-zero native F
potential. The output layer has no bias: a constant-potential bias would be
annihilated and violate the
no-inert-parameter rule. Noise shares the SSH weights rather than owning an
inactive deterministic-only branch. Maddison's comparison architecture has
seven width-5, 32-filter residual convolutions and 154,857 weights (local paper
text `/home/dbalwada/.claude/jobs/b3690d5d/tmp/maddison2026.txt`, lines
436--442).

Repository search found and reused `legoesm.ml.training.create_optimizer`,
`legoesm.training.inert_params.assert_no_inert`, and the public ocean
vorticity diagnostic. `ml/loss.py` and `ml/channel_packing.py` were inspected;
their spherical tensor conventions do not match this native C-grid balanced
loss, so they were not duplicated or modified. This training orchestration
module is outside a `physics/` package and introduces no physics Config, so the
physics-contract and `__param_spec__` ratchets do not apply.

The one-step closure JVP agrees with centred finite differences at
`rtol=2e-8, atol=2e-12`; the 216-step/24-hour JVP passes at
`rtol=2e-5, atol=2e-10`. Eager and JIT 24-hour objectives are identical. Each
stage is a live discriminator, parameter/noise/key replacements do not
retrace, and both the first-step and profiled reverse gradients pass the
repository no-inert gate.

## Loss, segment, and profile

The balanced diagnostic uses sigma-one-cell reflected Gaussian vorticity and
a discrete Dirichlet-Poisson inversion for rotational U/V. The common loss API
accepts `(members, hours, fields)`: deterministic point loss uses one member;
the multivariate energy score refuses `S < 2`. Training-split variances and wet
T-cell areas normalize all three fields. Raw SSH remains diagnostic-only.

Segment builders admit only 1, 2, 4, and 8 days, emit 24 right-endpoint
observations/day, checkpoint every production step uniformly, keep the
network/noise/key as traced arguments, and close over only the analytically
steady wind. A closure-free four-day build and an independently compiled
eight-day build are bit-identical at all 96 common hourly T/U/V endpoints.
The optimizer is constructed and initialized through the shared factory, but
is never updated.

The real-network CPU profiles include the actual hourly balanced loss:

| Window | Steps | Forward execution / peak RSS | Reverse execution / peak RSS |
|---:|---:|---:|---:|
| 1 day | 216 | 2.154 s / 1,151,968 KiB | 10.584 s / 1,948,928 KiB |
| 2 days | 432 | 4.707 s / 1,232,552 KiB | 14.118 s / 2,152,584 KiB |
| 4 days | 864 | 8.501 s / 1,397,204 KiB | 30.242 s / 2,503,016 KiB |
| 8 days | 1,728 | 17.380 s / 1,700,180 KiB | 62.673 s / 3,248,708 KiB |

The eight profile JSON hashes, in forward 1/2/4/8 then reverse 1/2/4/8 order,
are `98e0ca4870d5785c41fc8f673335a711356fcda0ddae13571d81af165f483474`,
`1681cbeeadd0a8744efac1d0ff00be57a8982f6ee546500b284e9f350da5ba2f`,
`36dd2b5efa11d9bda4f11d95aa5a8698f2651dcfe7c30c3672e3e1d74775751a`,
`e9466bd9455aa5d9ce6480779a832758b38641b4b617640e40c280694d4d2612`,
`dde6bd5c5f63b35618a7dcc85e97ebfcc8e0ae69eb00d1dfb4a13dbfb2910795`,
`17db63ce78979dad6eeb23da4e32ccb63ffc2e0a9659246f2baa183ca8ab8287`,
`f3860bc857de81dea2645904c11c7cc2a7329cb39862f7a75b5b797d852f246d`,
and `fa5c1c601cc0be2171fe401173eab9b715bea058090fc125d19ad4ff5477d89b`.

## Held-out controls and fail plants

The scorecard consumes the content-pinned round-2 year-2/year-3 arrays and
scores wet cells only.

| Control | Mean-transport RMSE (`m2 s-1`) | KE-spectrum relative L1 | Wet EKE-map NRMSE | EKE/truth | Top-octave KE / enstrophy |
|---|---:|---:|---:|---:|---:|
| `nu=10` | 38.3093 | 0.199185 | 0.334802 | 0.732702 | 0.181777 / 0.757835 |
| `nu=80` | 55.1007 | 0.188445 | 0.766893 | 0.396588 | 0.151000 / 0.729424 |

The future median of at least three seeds must beat the better control on all
three skill columns. Every seed must retain 80--120% of truth EKE and stay no
noisier than the worse control. A candidate with perfect skill scores but zero
variance fails only the EKE gate; a perfect-skill candidate with a two-cell
noise metric above both controls fails only spectral health. The scorecard is
`round3/evaluation/controls_years2_3.json`, SHA-256
`162686e7052ce95dfe9f9ce34793bf48515a0fa94a4393e5ff157907486cc686`.

## Scope disposition

| Round-3 requirement | Disposition | Evidence |
|---|---|---|
| Three dg50 compositions | Complete; hypothesis refuted | 30-day table and ULP-localized JSONs |
| F-potential every stage | Complete | Divergence, wall, integral, absence and per-stage tests |
| Equinox CNN with stochastic hook | Complete | Explicit manifest/spec, RNG/noise, 7,696 live parameters |
| Balanced swappable loss | Complete | Point/energy shared API; `S=1` refusal |
| 1/2/4/8 segment runner | Complete | Hourly scan, uniform checkpoint, zero-closure A/B |
| CPU/GPU profile | CPU complete; GPU acquisition open | Eight CPU JSONs and exact command below |
| Both-control evaluation | Complete | Years-2--3 table and content hashes |
| Collapse/grid-noise plants | Complete and both fail | Independent gate-reason fields in scorecard |
| No optimization | Complete | Profile manifests say `training_performed=false` |
| Round-2 queued science acquisitions | Still open | No complete hourly/balanced/reference manifest was present |

## Validation and commits

Every new Python module has a direct test. The final focused suite (network,
profile, evaluation, RK3 stage, and ladder) reports `60 passed in 201.33s`.
The complete SWG-online suite reports `20 passed in 46.32s`. Ruff reports
`All checks passed!`; `compileall -q` and `git diff --check` emit no output.
The citation gate is run from `## Coarse execution compositions` so every new
oracle citation above must be mapped to a real compiled source symbol.

External adversarial review is **UNREVIEWED in this round**. The required
`codex exec` fallback failed before review with `failed to initialize
in-process app-server client: Read-only file system`; the independent Claude
fallback produced no review output and was stopped after six minutes. This is
not claimed as a waiver or a PASS. The operator already assigned the round-2
independent review to round 4; round 4 must also disposition this round-3 diff
before any optimization. Two self-audit findings were fixed meanwhile: the
profile now includes the actual hourly balanced loss, and two exactly
redundant uniform metric CNN channels were removed.

Round commits before the final receipt update are `be6919d4f`, `a63b076d4`,
`78a04b23c`, and `7fb26fe4`.

## OPEN

1. The operator's existing queue remains authoritative for hourly targets,
   balanced starts/twins, and 3.125 km reference adequacy. Do not duplicate it
   and never read a directory without its completed manifest. Queue script and
   log are
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_gpu0_queue.sh` and
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_gpu0_queue.log`;
   expected outputs are under `swgonl_rounds/round2/{hourly_targets,balanced_start,predictability,reference}/`.

2. Run the same real-network profile on the first idle physical GPU, replacing
   `0` with `1` if that is the idle device:

   ```bash
   cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-nANtlMUs
   for mode in forward reverse; do for window in 1 2 4 8; do CUDA_VISIBLE_DEVICES=0 PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=gpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/bench/profile_swg_online.py --window-days "$window" --mode "$mode" --output "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round3/profile/gpu0_${mode}_${window}d.json"; done; done
   ```

3. User decision: may phase 4 train the flat-segment composition with explicit
   composition debt (maximum `5.218e-15` at 30 days), while retaining the exact
   bar and never describing dg50 as bit-identical to NEMO? The recommended
   answer is **yes**: it is the actual training composition, it agrees with the
   compiled day, and its debt is smaller than the per-step path. A no answer
   sends the next round to operand localization instead of optimization.

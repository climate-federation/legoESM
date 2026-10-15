# SWG online-learning design

Date: 2026-10-15

Status: **Round 9: pilot ready, nothing trained.** The driver
(`scripts/run/run_swg_online_training.py`) and one committed configuration
per candidate objective (`config/swg_online/`) exist and are tested on the
CPU. The pilot choices are fixed below. Its CUDA/native free runs are admitted
only after the two no-closure GPU controls pass the mechanical device check.
The user has accepted the recorded dg50 composition debt. The learned term
closes only the resolution gap and may not compensate a legoESM/NEMO gap.

## Fixed physical model and target convention

The training model is the NEMO-certified `dg50` card: 50 km, 400 s,
500 m, full gravity `g=9.80665 m s-2`, and an explicit free surface. The
per-axis gravity-wave Courant number is 0.560190, and the two-dimensional RK3
imaginary argument is 1.58446, below `sqrt(3)`. The training model remains
`nu=10 m2 s-1`; `nu=80 m2 s-1` is a separate grid-Reynolds control, never a
silent replacement.

The stage ladder remains the exact certification authority. A separate 30-day
composition audit found that none of the actual execution compositions is
bit-identical to NEMO. Per-step has 249/690 exact rows, 561/690 within
`1e-15`, and maximum error `9.479e-15`; compiled-day and flat-segment each have
249/690 exact, 543/690 within `1e-15`, and maximum error `5.218e-15`. All first
differ on day 1 at `vtau_before[60,1]` by signed zero (one ordered IEEE word),
then first exceed the numerical bar on day 9 at wall-adjacent `sshn[1,1]`
(16 ULP per-step; 25 ULP compiled/flat). Therefore this document names the
training composition but does not call it NEMO bit-for-bit. The user accepted
this debt on 2026-10-13; the unchanged `1e-15` certification bar still records
it as debt.

The state target is the conservative restriction in `swg_coarse_graining.py`:
sum fine T-cell volumes into each coarse T cell and sum `e2*h*u` or `e1*h*v`
over the fine faces tiling each coarse face, then diagnose coarse SSH and face
velocity using the coarse live QCO thickness. There is no component-mean
restoration. The round-1 independent-state Gaussian remains a labelled
diagnostic and is forbidden as a training target.

There are two named objectives and **no default**; the owner has not chosen.
`symmetric` starts and targets with the exact box restriction and applies a
sigma-one-coarse-cell Gaussian to both model and target vorticity (registered
on F points, odd about the wall vertices). `template` follows Maddison's
data/model asymmetry: start and target from one fine-grid filtered
restriction, model vorticity unfiltered in the loss. Maddison defines the
Gaussian "width (specifically a standard deviation)" as one coarse spacing,
filters instantaneous fine vorticity, and then samples the coarse grid
([Maddison 2026, sections 2.3--2.4](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004883)).
The `template` operator filters the fine finite-volume extensives (volume
even about the walls, normal transport odd, tangential transport even;
standard deviation 50 km = 8 fine cells) and then restricts them exactly, so
continuity, zero normal wall transport and total volume hold without any
restoration.

## Reference adequacy (open)

At 6.25 km the Munk width has 1.27 cells, so a 3.125 km sensitivity run is
the adequacy test: days 1--8 from the common rest state (normalized RMS at
most 10% and pattern correlation at least 0.99 for coarse SSH, relative
vorticity and rotational velocity) and year-1 KE, EKE, jet position and
DST-streamfunction KE spectrum against the 6.25 km run's adjacent-year
difference.

Operator measurement on the round-6 code (`round6/reference/adequacy.json`):
the tool printed `FAIL measured=1 adequate=False`. Year 1 passes every
statistic (KE 1.1%, EKE 9.6%, jet 3.6% of the basin width, spectrum 8.7%,
against bars of 19.6%, 18.6%, 10% and 19.8%). Days 1--8 pass for SSH (NRMS
0.36--0.42%) and vorticity (0.44--0.50%), correlation above 0.99998. Rotational
velocity "fails" only through an instrument defect: its arrays are
(day, component, wet point) and the NRMS reduction runs over the component
axis (`nemo_testcase_l1_swg_online_reference_adequacy.py:102-103`, `axis=1`),
so each day's value is 6,400 per-point numbers and the reported first
crossing "day 1562" is a flattened index. Its pattern correlation is above
0.99998. The corrected rotational-velocity NRMS has not been measured. Two
earlier points stay open: the year-1 bars include the spin-up change from
year 1 to year 2 (an earlier review), and the 3.125 km run covers one year.

## Initialization and hourly observations

The fine 6.25 km truth (`swgdg_rounds/legoesm_runs/dg_gpu_7300d_truth`)
restarts every 30 days into dual-operator hourly chunks (schema v2): one
compiled fine scan stores both coarse states (box and filtered) at t0 and at
24 right endpoints per day, and each chunk must reproduce the uninterrupted
next fine daily state bit for bit (`--verify-next`). The 12-year record
`[day731, day5111)` is written to `round6/hourly_dual_12y` as chunks
`dual_dayAAAA_to_dayBBBB`; the independent test year `[day5141, day5506]`
goes to `round6/hourly_dual_test_year`.

Every training and validation window starts from the stored state at its
start hour with its **sea level rebalanced** (`build_start_rebalancer`):
velocities are never changed; each iteration takes the card's own first-step
acceleration without closure, splits off its closed-wall divergent part
`grad(phi)` and adds `(phi - <phi>)/g` to wet SSH, so volume is unchanged; it
stops when the divergent part's rms is at most `1e-3` of the rotational
part's and raises after 20 updates. Targets are the stored right endpoints of
the same operator 1..24w hours later. Measured on the stored day-731 start:
box converges in 6 iterations, filtered in 5.

Start admission is **truth-relative and selects nothing**
(`nemo_testcase_l1_swg_online_balanced_start.py`). Each rebalanced start
gets a 48-hour no-closure forecast compared with targets of the same
operator; it passes if the hour-1 vorticity error is at most twice the
hour-24 error divided by 24, if its fast (divergent) kinetic energy is at
most twice the same-operator truth's in the worst of hours 1--6 and in the
mean of hours 25--48, and if rebalancing kept the volume (within `1e-12` of
the basin volume). The tool reports both operators; the old rule that
selected `template` when 8 of 10 filtered starts passed is gone. The
admission run on the current record is the operator's.

## Dataset contract

`SWGDatasetContract` owns the data (`legoesm-swg-online-dataset-v1`): split
bounds in absolute hours, one objective (which selects the operator's
fields), batch size 10, and a list of chunk roots each pinned by its
manifest's SHA-256. Every source must be complete, schema v2, fp64,
unperturbed, from the registered fine deck, with a passed next-day
verification and a restart that the one fine truth run lists with the same
digest; day files must hold exactly 24 records with the right absolute
hours. Adjacent chunks must agree at their shared restart hour to `1e-10` of
the field maximum, and each split must be one contiguous pinned record.

Experiment one's splits (days, inclusive bounds as right endpoints): training
`[731, 4235]` (3,504 days, the first 80% of the 12 years, as Maddison),
validation `[4265, 5111]` after a 30-day gap, test `[5141, 5506]` after
another 30-day gap. The test year is pinned in the contract and never read
by training.

Training batches: one starting offset per (seed, window length, epoch),
non-overlapping windows, shuffled, ten per batch, an incomplete last batch
dropped (Maddison redraws the offset and the order at each window length and
epoch). Held-out batches use one fixed chronological lattice from the
split's first hour. Each issued batch is bound to its split by a content
digest. Eq. 7 needs no data normalizers; the contract computes none.

## Closure seam and network

The closure is an Equinox module producing one scalar potential on native
F points. The card applies the matched rotated gradient to form native U/V
accelerations at **every RK3 stage**. The same F-point potential is zero on
the closed boundary, so wall-normal force is zero; the discrete divergence of
the volume-force transport is roundoff zero. If the closure argument is
absent, no addition or reassociation occurs, preserving the certified card
byte-for-byte. Positive U is east and positive V is north; the force is the
clockwise rotated gradient `(dPhi/dy, -dPhi/dx)` divided by the live face
cross-section.

The network reads four channels: SSH (plus the explicit noise input), U and
V averaged to T points, and the wet mask. It has five width-3 convolutions,
16 hidden channels, ELU, an 11-cell (550 km) receptive field and 7,696
parameters, smaller than Maddison's seven width-5, 32-filter residual
layers with 154,857 weights. The final convolution has no bias (a constant
potential is annihilated by the curl); noise enters the SSH channel through
the state weights, so deterministic training has no noise-only leaf.

**Input and output scales: the shallow-water analogue of Maddison Eq. 6**
(`maddison_eq6_network_scales`). The template divides its input vorticity by
`|beta| L` and multiplies its output vorticity tendency by
`alpha_out = |tau0| pi / (rho0 D L)`, `L` half the basin side. This network
reads grid-point SSH and velocity and returns a transport potential, so each
scale is the field whose differences over one coarse spacing `dx` carry the
template's scale: velocity `U = |beta| L dx` (a one-cell difference of `u/U`
is `zeta/(|beta| L)`), SSH `|f0| |beta| L dx^2 / g` (the five-point Laplacian
of the scaled SSH is the geostrophic vorticity over `|beta| L`), potential
`alpha_out D dx^2` (the force's curl is `-lap(Phi)/h`, so minus the
five-point Laplacian of the scaled potential is the template's output for
`h = D`). From the deck: `beta = 2e-11`, `L = 2,000 km`, `dx = 50 km`,
`f0 = 7e-5`, `tau0 = 0.1 Pa`, `D = 500 m`, and the card's `rho0 = 1026`
(NEMO's, which turns the stress into acceleration; Maddison's Table 1 uses
1000) and `g`: `U = 2.0 m s-1`, SSH scale `0.7138 m`, potential scale
`0.3827 m3 s-2`, `|beta| L = 4e-5 s-1`, `alpha_out = 3.062e-13 s-2`.

Measured on ten real starts (absolute days 731 + 18k, rebalanced, both
operators): scaled SSH rms 0.48--0.50 and maximum 2.9--3.4; scaled T-centred
velocity rms 0.056--0.077 and maximum 0.53--0.78; the template's own input
`zeta/(|beta| L)` on the same states rms 0.045--0.073 and maximum 0.42--0.92,
so the analogue gives the network the template's magnitudes. The initial
scaled output is small: potential rms 0.02--0.10 and maximum 0.06--0.15.
Initialization is `PRNGKey(seed)` per network seed.

**Start gates, every run.** On the ten starts of the first training batch,
the initial closure acceleration must be at most 1% of `tau0/(rho0 H)`
(`initial_closure_acceleration_ratio`), and every parameter leaf must have a
finite non-zero gradient at the first optimiser step (the repository's
no-inert rule). With the Eq. 6 scales one percent corresponds to a one-cell
scaled-potential difference of about `L/(100 pi dx) = 0.127`. Measured on the
ten real starts above: seeds 20261013, 20261014 and 20261015 give 0.24%,
0.35% and 0.86% (filtered) and 0.29%, 0.43% and 0.86% (box). The loss and
every gradient element must be finite at every step.

## Loss, rollout and curriculum

The loss is exactly Maddison's Eq. 7: squared vorticity error weighted by
`dx^2` at interior F points, times `1e4/(beta^2 L^2)/(4 L^2)` with `L`
half the deck's square side (`rn_domsiz_m = 2L`, checked against the grid),
averaged over all `M = 24w` hourly right endpoints and all windows. Model and
target pass through the same diagnostic, so the loss is zero at a stored
start. Rotational-velocity terms and data normalizers are not part of it.
Whether the first interior F ring next to the walls is weighted is an owner
decision: `wall_ring` is `weighted` (Eq. 7 as written) or `excluded`, with no
default; under `template` that ring carries 14--20% of the no-closure loss.

Each window runs the certified segment (`build_segment_fn`, rematerialising
every production step) with the closure at every stage, zero stage noise and
zero noise scale. The curriculum is exactly 1, 2, 4 and 8 days, five epochs
each, batch 10. The network is carried from one window length to the next;
the optimiser is re-initialised at every window length. No 16, 32 or 60-day
stage is authorized: 60 days is 12,960 coarse steps, beyond the measured
balanced-vorticity predictability time (10.3 days at `1e-11 s-1`, 13.6 days at
`1e-10 s-1`).

The optimiser is the template's Keras-default AdamW at a constant rate: rate
`1e-3`, weight decay `0.004`, betas `0.9` and `0.999`, epsilon `1e-7` added to
the square root of the uncorrected second moment (Keras' convention), no
clipping. It is built only through the strict wrapper `create_swg_optimizer`
over the shared factory: every field of `SWGOptimizerSettings` is required,
a `TrainingConfig` (whose unset fields carry the factory's warm-up, cosine
decay, clipping and Optax epsilon) is refused, and any schedule but
`constant` or any clipping is refused.

## The training driver

`scripts/run/run_swg_online_training.py train --config <file> --seed <s>
--run-root <dir> --device cpu|cuda [--physical-gpu <k>]
--windows-per-call <n>` runs one network seed of one configuration in
`<dir>/<experiment>/seed<s>`. It refuses to start without a configuration
file, and the file must state every choice with no extra or missing key:
deck, objective, wall ring, network, one or more seeds (the held-out gate
scores three or more), record roots and chunk label, split days, batch size,
curriculum, optimiser and free run (including its device and math policy,
which must be explicitly `cpu`+`libm` or `cuda`+`native`; the pilot selects
the latter).
`windows_per_call` (1, 2, 5 or 10) only splits each batch of ten into equal
device calls to bound memory; the batch loss and gradient are the mean of
the parts.

- A GPU run needs `JAX_PLATFORMS=cuda` (JAX 0.10 refuses `gpu`),
  `CUDA_VISIBLE_DEVICES` equal to `--physical-gpu`, x64, and runs with native
  transcendentals; a CPU run needs `JAX_PLATFORMS=cpu` and uses libm.
  Training refuses a tree with modified tracked files (it stamps the commit).
- First start: the driver pins every complete chunk (a `manifest.json` with
  status complete; chunks still being written are skipped) that touches a
  split into a round-6 contract, validates it, and only then creates the run
  directory with `run_manifest.json` (every resolved choice: configuration
  and its digest, seed and what it seeds, commit, versions, device, card
  namelist, network spec and scales, optimiser, loss constants, contract
  digest, operator, start-state rule, curriculum, gates, free-run deck). The
  contract needs all three splits covered, so training cannot start before
  the test year is written.
- Every epoch: training batches, then the validation loss on every
  validation batch of that window length, then an atomic checkpoint of the
  network and optimiser state, then `training_log.json` with the per-batch
  training losses (each at its pre-update parameters), the validation loss,
  and the checkpoint's digest.
- Resume: rerunning the same command continues after the last logged epoch
  from its checkpoint (digest checked) and refuses if anything in the run
  manifest differs (configuration, commit, device type, windows per call).
- After the last epoch of a window length, a separate process (`free-run
  --config <file> --seed <s> --run-root <dir> --window-days <w>`, round 8)
  runs the coarse model from rest with that window's last checkpoint
  (`swg_online_step`, closure at every stage) for 7,300 days on the device
  and math policy the configuration states. CUDA runs disable XLA autotuning
  and record the device, math library, XLA flags, JAX version and GPU name.
  Every day file holds the end-of-day snapshot beside the mean of every
  post-step state of that day, under the scorer's dual per-file labels; the
  manifest is the one free-run schema of `legoesm.training.swg_online`
  (checkpoint file and digest, network seed, objective, window length, run
  and contract identity), which the scorer reads through the same module
  and checks against the checkpoint file. An interrupted free run is
  archived and restarted; a complete one is never overwritten.
- `profile` times one batch-10 optimiser step and one validation batch per
  window length, and one free-run day only on the free runs' configured
  device, and extrapolates one seed.
- `check-data` pins the record the same way and builds the first batch of
  every split at every window length (digests, record counts, finiteness,
  rebalanced starts), so the finished record is exercised before any run.

## Cost

Measured on the CPU with the driver's own `profile` (real dg50 card, the
7,696-parameter network with the Eq. 6 scales, ten windows per call, one
optimiser step timed after a compiled first step; other users' jobs kept the
load average at 11--26 on 80 cores). Batches per epoch are the training and
validation lattice counts, `floor(days / w) / 10`.

| Window | Steps per epoch | One step (batch 10) | Validation batches | One validation batch | Five epochs |
|---:|---:|---:|---:|---:|---:|
| 1 day | 350 | 188 s | 84 | 59 s | 98.4 h |
| 2 days | 175 | 390 s | 42 | 119 s | 101.9 h |
| 4 days | 87 | 768 s | 21 | 236 s | 99.7 h |
| 8 days | 43 | 1,507 s | 10 | 469 s | 96.5 h |

Each window length costs about the same, because the training days per
epoch are fixed. Training is 396 h per seed; the four 20-year free runs add
34--36 h (4.2--4.5 s per model day); loading, digests and start rebalancing
add about 0.5 s per batch (measured on the real chunks), about half an hour.
One seed is therefore about 430 h on this CPU, so training needs the GPU.

**Superseded (round 8):** the operator's `profile --device cuda` at
`2a227dbc9` measured one batch-10 step at 6.2, 12.4, 24.8 and 49.7 s (1, 2,
4, 8 days), 30--31 times the CPU, so the 4.6x expectation below was far too
pessimistic; the round-8 integration receipt has the numbers.

**GPU expectation (not measured).** The one GPU/CPU pair measured on this
machine for the SWG step, the compiled one-day forward scan of the 6.25 km
model, runs 4.6 times faster on a V100S (4.49 against 20.73 ms per step). If
the batch-10 training program gains the same factor, one seed takes about
94 h (3.9 days) on one GPU: 86 h of training and 7.5 h of free runs. This is
a different program (reverse mode with rematerialisation, a convolutional
network, 70,560 cells per step against 409,600), so the first GPU action is
`profile --device cuda` (about half an hour at that ratio), which prints each
window length as it completes and replaces this estimate. One candidate (three
seeds) on the two GPUs would then take about 8 days; all three candidates
about 20.

## Controls and held-out gate

Both no-closure controls (`nu=10` Maddison replication and `nu=80`
grid-Reynolds control) and every trained seed are 20-year free runs from
rest, scored by one harness on the eight non-overlapping two-year windows
after year 1 that the 6.25 km truth pins (days 366--6205), against the truth
climatology made by the selected objective's own operator (`symmetric`: box
restriction; `template`: filtered restriction). Every number is a median
over those windows. The bars are the truth's own window-to-window spread,
frozen in code before any trained model exists: each truth window is scored
against the climatology of the other seven, and the bar is the largest of
the eight values. The median seed must beat both controls by more than the
truth's half-range in live-face-transport RMSE, KE-spectrum L1 and wet
EKE-map NRMSE; every seed's EKE ratio must lie in the truth's own
eight-window band; no seed may exceed the worse control's top-octave
fractions or grid-noise index by more than that control's own half-range.
Finite fields, positive thickness and sub-unit Courant numbers are
unconditional on every window and on the whole run. Measured: the perfect
model passes and both controls fail, for both objectives; `freeze --verify`
recomputes the record from the pinned data and refuses any difference.

A review of round 6 held three points of this gate open (outside the
driver): the spectral-health allowances come from the worse control's
spread, not the truth's; three seed labels can bind one free-run root, and
the scorer does not require the manifest to name a checkpoint or seed (the
driver's free-run manifests now do); and the snapshot/daily-mean separation
trusts a manifest label and treats missing sampling metadata as snapshots
(the driver labels every file).

Two adversarial plants stay mandatory: replacing every time-varying anomaly
by its time mean must fail the EKE band, and a native two-cell checkerboard
above the worse control must fail spectral health.

## Stochastic branch

Stochastic readiness is **interface only**. The rollout API carries an
explicit RNG key and member-indexed noise pytree. The loss API accepts `S`
trajectories and dispatches between point loss and energy score without
changing solver or closure signatures. The dormant energy score follows
Brolly Eqs. 9--10: Euclidean score at each lead, pair term divided by
`2 S (S-1)`, then uniform lead averaging; it requires `S >= 2` and
caller-supplied normalizers. There is no inactive trainable noise parameter
in deterministic mode; any future trainable noise scale must first pass the
no-inert gate. Collapsed ensemble variance is a planted failure.

## What enforces each requirement

| Requirement | Design answer | Enforcing test or measurement |
|---|---|---|
| Conservative, balanced starts | Stored state, sea level rebalanced; truth-relative admission that selects nothing | Rebalancer and contract tests (`tests/ml/test_swg_online.py`); admission tool tests |
| Hourly targets | Dual-operator v2 chunks, right endpoints, next-day bitwise restart check | Contract plants; real-chunk driver test |
| Adequate reference | 3.125 km early path and year-1 statistics | Open: instrument defect in the rotational-velocity NRMS |
| Curriculum | 1/2/4/8 days, five epochs, batch 10, network carried, optimiser re-initialised | Configuration tests; two-window end-to-end test reads the optimiser step count and the carried weights |
| Optimiser | Strict wrapper, Keras AdamW, constant rate, no clipping | Unset-field, refusal and pinned-update tests |
| Network scales | Eq. 6 shallow-water analogue from the deck | Formula and moved-deck tests; ten-start measurement |
| Start gates | 1% initial acceleration, no inert leaf, finite every step | End-to-end test (gate recorded); real-chunk test |
| Checkpoint and resume | Atomic per-epoch checkpoints; resume refuses any changed choice | Killed-run test: resumed epoch equals the killed process's state word for word |
| Free runs | 7,300 days from rest on explicit CPU/libm or CUDA/native, snapshot and daily mean every day, one manifest schema with the checkpoint file's digest | Round-8 end-to-end test plus round-9 two-day bitwise CPU replays and CUDA device-record refusal tests |
| Held-out gate | Truth-spread bars, three seeds | Evaluation tests; three review findings open |
| Non-divergent forcing | F-potential rotated gradient at every RK3 stage | Divergence, wall, momentum, zero-seam, JVP/FD tests |

## Before the pilot

1. `check-data` passes on the committed pilot configuration (67 sources and
   all six split/window combinations in round 9).
2. The GPU smoke is finite with tiny differences from frozen CPU days 1-2.
3. The two 7,300-day GPU controls run; the device check must PASS before a
   trained CUDA free run is scored. Training need not wait for the controls.
   The device check scores each GPU control on the eight frozen windows and
   requires, for every scored metric, that the GPU-minus-CPU difference of
   window medians is strictly less than that CPU control's own half-range
   over the same eight windows (half of max minus min of its frozen
   per-window values), and that both GPU controls still fail the unchanged
   pass mark. **Amended 2026-10-15 (user decision):** the tolerance was the
   truth's half-range until round 9; that is the pass mark's allowance for
   the truth's internal spread and is far tighter than two runs of the same
   model differ by sampling, so the first GPU check failed with every
   difference at 0.01-0.64 of the control's own spread. The record now
   carries both tolerances (the control one decides; the truth one is
   information) and the ratio difference / control half-range; its schema is
   `legoesm.swg_online_device_check.v2`, and admission refuses v1 records.
   The pass mark (frozen bars, thresholds, noise allowance) is unchanged.

## Fixed pilot decisions

Template objective; first interior wall ring excluded; seed 20261013; 1- and
2-day windows, five epochs each, batch 10; template optimiser; first two
training years; `windows_per_call=10`; one 7,300-day CUDA/native free run after
each window length. The frozen truth, frozen CPU controls, pass mark, gate
tolerances, decks, schemes and all other scientific choices are unchanged.

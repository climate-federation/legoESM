# SWG online learning — round 2 receipt

Date: 2026-10-13

Status: **HELD**. The conservative target and its continuity proof are ready,
but the dg50 30-day NEMO ladder first exceeds the unchanged numerical bar on
day 9 and reaches `9.47853e-15` on day 30. Training may not start. The hourly
target, balanced-start, balanced-twin, and 3.125 km GPU acquisitions also remain
open. No neural network, optimizer, or closure was added.

Incoming tip: `e49a84492831597866e41f806e726aa8c1646477`. Writable clone:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-2559510110`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/`.
The binding design and round-1 review logs have SHA-256
`ba7e203aa98384b2efeaf27735856118a27e857b751b0bb6dd0aba5cdaab375d`
and `6ea51ac0dc4dd82b8ff32022f97e23e1e4d11caebbc71e5d14af7a2a42b50645`.

## Coarse-model certification

The admitted operator acquisitions are the stage root
`round1/oracle_swg_dg50_r1/` and 30-day root
`round1/oracle_swg_dg50_r1_30d/`, both acquired at `e49a84492`. The ladder
pins their admission records and every consumed file. The executed DINO branch
updates and rotates the first-stage levels at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:204-210`, writes the second-stage
record at `SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:220-235`, and advances
the third stage and restart indices at
`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:360,366,376`.

| Deck / ladder | Exact rows | Rows numerically at `1e-15` | Maximum absolute difference |
|---|---:|---:|---:|
| fine 6.25 km, given entry (established) | 276/338 | 338/338 | `1.0842e-19` |
| fine 6.25 km, chained (established) | 113/338 | 338/338 | `2.1684e-19` |
| dg50, given entry | 276/338 | 338/338 | `8.67362e-19` |
| dg50, chained | 113/338 | 338/338 | `1.30104e-18` |
| dg50, 30 daily boundaries | 339/690 | 561/690 | `9.47853e-15` |

The 30-day run is within the numerical bar through day 8; SSH first crosses on
day 9 (`1.01308e-15`), and the largest day-30 debt is V velocity
(`9.47853e-15`). Thus the stage rows reproduce the fine deck's exact-row
pattern, but the month ladder does **not** certify dg50 under the unchanged
bar. The reports are `certification/dg50_stages_given.json` (SHA-256
`d660d19f4e7b4a99827a5799cb31463663e489ce67368e90661faddd61c59bbd`),
`dg50_stages_chained.json`
(`ceb81e57a7d6db918d4ac1ff2b9847b6b3b69195a7d6e224c58bf15cf422d65e`),
and `dg50_month.json`
(`4902e7f29fe50efe246af7a2a1d04145be329a77bd2e86bc1a29a0020422c004`).

Both controls are live. A one-ULP recorded stage-entry U change alters one SSH
cell after the production stage by `7.57912e-22`; a one-ULP admitted daily U
boundary change alters three SSH cells after a production step by as much as
`1.38778e-17`. Their reports have SHA-256
`70399f0764900ddac840c6cf7d1c548d315634037320f59ea38aadef6185330e`
and `f8f018a6699ff0237f00e3c85905c20ae864779e89835453b9ea8714519dba02`.

The 400 s clock preserves the fine deck's per-axis gravity-wave Courant,
`sqrt(g H) dt / dx = 0.560190`. The executed RK3 pressure/free-surface stages
are `SWG_DG/BLD/ppsrc/nemo/stprk3.f90:140-171`,
`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:196-227`, and
`SWG_DG/BLD/ppsrc/nemo/stprk3.f90:252-290`. The two-dimensional imaginary
argument is `sqrt(2) * 2 * 0.560190 = 1.58446`, below RK3's `sqrt(3)` bound.
This is a full-gravity (`9.80665 m s-2`) explicit free surface over 500 m, not
a reduced-gravity model.

## Conservative coarse-grained truth

The replacement operator sums fine T-cell live volumes and fine U/V face
transports (`e2*h*u` and `e1*h*v`) over each 8-by-8 coarse footprint. It then
diagnoses coarse SSH and face velocity with the coarse QCO live thickness.
Closed faces stay exactly closed, and there is no velocity-mean restoration.
The round-1 independent Gaussian state filter remains callable only under its
explicit diagnostic name.

The content-pinned day-1095 real-data audit passes at a `1e-15 m s-1` gate:
maximum coarse continuity residual `4.52016e-18 m s-1` in the interior and
`2.88669e-18 m s-1` in the outer four wet-cell rings. Its JSON SHA-256 is
`09bcac95dfb8ee9c9a6c4bd61e9e1c7a10aab4ac6d61aa10a3af22d6593f0f2c`.
The planted independent filter is refused with an interior residual
`5.31269e-5 m s-1`. Synthetic tests separately exercise analytic C-grid
staggering, walls, the Gaussian path, and a non-conservative plant.

Gaussian smoothing is applied only after restriction and only to relative
vorticity, with reflecting walls and standard deviation one coarse cell.
Maddison defines Gaussian “width” specifically as standard deviation 62.5 km,
filters relative vorticity, and subsamples; see
[Maddison (2026), sections 2.3–2.4](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004883).

All 1,095 manifest-listed fine days were converted for snapshots and daily
means. The conservative archive manifests have SHA-256
`aaec7f829d7530a275c07a263cd96185a4fa6245be411c63626156e35fa65d2c`
and `451deab5a6f400384134f3f45aca203895f20ff117a0e7e64ecd937a5a64fb05`.

## No-closure controls and climate gap

The original 20-year `nu=10 m2 s-1` control remains numerically finite:
minimum total thickness `497.67697 m`, maximum U Courant `0.0209152`, and
maximum V Courant `0.0146682`. It is nevertheless spectrally noisy. Against
the conservative daily-mean truth, the yearly results are:

| Year | KE, dg50 / truth (`m2 s-2`) | EKE, dg50 / truth (`m2 s-2`) | Top-octave KE, dg50 / truth | Top-octave enstrophy, dg50 / truth | Wet SSH RMSE (`m`) | Wet EKE-map RMSE (`m2 s-2`) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.0177706 / 0.0169408` | `0.00796814 / 0.0110588` | `0.19691 / 0.10124` | `0.76489 / 0.56855` | `0.0636915` | `0.0112048` |
| 2 | `0.0219401 / 0.0210643` | `0.0103577 / 0.0135839` | `0.18029 / 0.08826` | `0.75150 / 0.55305` | `0.108627` | `0.0118595` |
| 3 | `0.0217900 / 0.0208713` | `0.00931310 / 0.0132631` | `0.18327 / 0.08654` | `0.76417 / 0.54671` | `0.0807035` | `0.0133358` |

Thus years 2–3 have 4.16% and 4.40% excess KE against the new conservative
target, but about twice the target's top-octave KE and 1.36–1.40 times its
top-octave enstrophy. The review's 49% KE excess and 3.0%/34.3% target spectral
fractions were correct for the now-rejected component-filtered target; they are
not silently reused for the conservative target. The native checkerboard index
stays around roundoff (`~2e-18`), so that index alone does not detect the broad
high-wavenumber excess. Years 2–3 are the least-confounded available comparison,
not an equilibrated claim. Report SHA-256 values are
`f88a34d9add8de58517a2abdebde46ee980b0286e7bc5f51548a6c593cbebe0c`,
`9df7d3570794a79c4c642aa6315110e352369fdb3a10e34b38a849cf2ac51724`,
and `94bc0ad117684343aace9c1a4576139f2b4be4e75f3b184a121d6474225f581d`.
The separate 20-row yearly health report has SHA-256
`47d5b3973e7105f23c1e6ea0088604365ca8bf74264a262732ab554bc9ee3392`;
after spin-up (years 2–20), top-octave KE stays in `0.17680–0.18453` and
top-octave enstrophy in `0.75056–0.76368`, confirming persistent rather than
explosive grid noise.

The preregistered `nu=80 m2 s-1` 30-day control passes its numerical preflight:
all finite, minimum thickness `498.90918 m`, maximum U/V Courants
`0.0115257/0.0111759`. Its manifest SHA-256 is
`eab59bc0614afc2b753f261009571291d2a7de1de324102c957d84e577fcb1fe`.
The 20-year CPU run and yearly spectral report are recorded below after their
final clean completion.

## Hourly targets and balanced starts

The driver now restricts fine T volume and U/V transports inside each compiled
fine-model hour and writes only 24 coarse right-endpoint fields per day. It
admits both the restart and uninterrupted next-day snapshot by manifest hash;
the first restarted day must reproduce the uninterrupted fine snapshot
bit-for-bit before a segment can complete. The physical GPU is a required
operator argument, has no default, and is recorded in the run manifest. The
wrapper no longer assumes GPU 1. Direct tests cover both physical devices,
missing/mismatched device refusal, coarse-only storage, and restart admission.

A zero-closure 4/8-day A/B test from one dg50 state passes bit-for-bit between
the day-scan composition and flat training-segment composition. The ten-start
instrument is committed for days 731, 771, ..., 1091. It reports hourly
divergent KE, SSH-tendency RMS, and relative-vorticity error, with the deciding
shock/closure ratio at 6, 24, and 48 hours. Actual hourly caches and balanced
start ratios were **not acquired in this CPU-only round**; exact GPU commands
are in OPEN.

## Predictability and reference adequacy

The balanced analyzer recomputes relative-vorticity, divergence, and
Dirichlet-streamfunction rotational-velocity norms. Its preregistered fit starts
at the first lead after which divergence/vorticity remains at most 0.07 and
requires at least 20 samples. Applied to the existing white-SSH twin, it selects
days 18–50 and reproduces a provisional vorticity e-folding time
`10.4248019 d`, `R2=0.981892`, one seed. Report SHA-256:
`f4206c17c7ab54f89be19a3b1ce73df1bb71d6a92f7aec0390b05c74df2be507`.
A one-production-step forward-mode balanced tangent is finite and nonzero;
report SHA-256:
`b95a3d8ce711a5769682eba84bc399457d0bd0aae578e955003adc7253bf800d`.
The smooth F-point perturbation has zero SSH and roundoff transport divergence.
Nonlinear amplitudes `1e-11` and `1e-10 s-1` are preregistered for the open
120-day twins.

The growing fine truth had no completed manifest during this round, so no
unlisted day was read. The complete three-year archive is not stationary:
yearly restricted KE is `0.0168545`, `0.0209993`, and `0.0208006 m2 s-2`;
EKE is `0.0109924`, `0.0135185`, and `0.0131942 m2 s-2`; consecutive annual
mean-velocity differences are `0.0476459` and `0.0420307 m s-1`. The report
SHA-256 is
`c37973819bd902c09d5a57369f7d901776b1dc668b1b3c482de14e908d5a2923`.

The explicit `dg3125` sensitivity deck uses `nn_SWG=32`, `dx=3.125 km`, and
`dt=25 s`, preserving gravity-wave Courant and every other fine-deck parameter.
Scaling the measured 45 minutes/model-year at 6.25 km by four times as many
cells and two times as many steps estimates 6 GPU-hours/model-year: about 18
hours for the first three comparable years or 120 hours for 20 years. The
preregistered gate compares each common year after restriction to 50 km:
relative KE, relative EKE, jet-position RMSE divided by basin width, and
KE-spectrum relative L1 error must each be at most 10%. Any failure makes
6.25 km truth adequacy a user decision.

## Design for rounds 3+

The rewritten design note answers all ten design-review requirements in a
requirement/design/enforcement table. It specifies a scalar F-point potential
whose matched rotated gradient enters every RK3 stage, hourly balanced fields,
the 1/2/4/8-day initial curriculum, explicit forcing and RNG pytrees,
ensemble-aware swappable point/energy losses with `S>=2`, no inert parameters,
both viscosity controls, at least three network seeds, quantitative held-out
gates, and collapsed-variance/grid-noise plants. Closure absence performs no
addition or reassociation, so the certified card stays untouched. These are
design contracts only; no network, optimizer, loss, or closure code exists in
this round.

## Review finding to disposition to evidence

| Finding | Disposition | Evidence |
|---|---|---|
| D1 conservative balanced starts | Implemented; acquisition open | Exact restriction and real continuity gate; ten-start shock instrument |
| D2 hourly targets | Implemented; acquisition open | In-scan 24/hour coarse-only writer and restart bitwise gate |
| D3 reference adequacy/stationarity | Implemented; adequacy acquisition open | Three-year stationarity numbers; explicit dg3125 deck and 10% gate |
| D4 balanced predictability | Implemented; nonlinear acquisition open | 10.4248-day balanced reanalysis, smooth perturbation, JVP |
| D5 cap curriculum at 8 days | Implemented in design | Initial 1/2/4/8 only; 16/32/60 require network-inclusive profile |
| D6 F-potential closure | Implemented in design only | RK3-stage, walls, divergence, momentum, zero-seam and JVP/FD gates specified |
| D7 add nu80 | Implemented | 30-day preflight passed; 20-year CPU control and yearly health instrument |
| D8 structural stochastic branch | Implemented in design only | Explicit RNG/noise, ensemble loss API, `S>=2`, no inactive trainable scale |
| D9 quantitative multi-seed gates | Implemented in design only | Three seeds, beat-better-control metrics, variance/noise fail plants |
| D10 physics/composition | Implemented | Full-gravity wording; zero-closure 4/8-day bitwise A/B passes |
| R1 continuity failure | Replaced, not patched | Real residual `4.52e-18` interior; old plant refused at `5.31e-5 m s-1` |
| R2 mean restoration | Removed from training operator | Exact volumes/transports and live thickness; old operator diagnostic-only |
| R3 divergent U/V closure and raw loss | Rewritten in design | F-potential force; balanced vorticity/rotational-velocity loss |
| R4 daily targets and 60-day curriculum | Rewritten | Hourly writer; 1/2/4/8 only |
| R5 incomplete health | Fixed | Top-octave KE/enstrophy and checkerboard index, with live high-k defect |
| R6 spin-up/dry-cell dilution | Fixed | Separate yearly reports and wet-only map scores; years 2–3 named exploratory |
| R7 white-SSH interpretation | Corrected | Balanced analyzer reproduces provisional 10.4248 days; balanced twins open |
| R8 dg50 oracle unregistered | Fixed, with debt found | Admissions/hashes committed; both stage ladders and month ladder executed |
| R9 insufficient phase-5/6 gates | Rewritten in design | nu controls, three seeds, `S>=2`, collapsed/noise plants |
| R10 non-discriminating tests | Fixed | Gaussian constant path live, mean monkeypatch plant, exact zero Courant, `1e-15` scan gate with `2e-15` plant |
| R11 whitespace/Ruff failures | Fixed | Final `git diff --check` and targeted Ruff commands pass |

## Hygiene and commits

Every new Python module has a direct test. The final affected CPU suite, Ruff,
diff, compilation, shell syntax, and oracle citation gate commands and their
actual outputs are recorded here after the final clean run.

Round commits before this receipt are `840d5930b`, `39a61da7a`, `f44690f0b`,
`65d598ad0`, `66d7115c4`, `9503820fa`, `fe9596c60`, `f65ee16a8`, `68646355a`,
`18862d4cf`, and `5e8bce04b`.

## OPEN

All GPU commands must be run from the clean writable clone above. Either
physical device may replace `0` when idle; no running job may be killed.

1. Generate ten four-day hourly segments and prove next-day restart parity:

   ```bash
   cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-2559510110
   for d in 731 771 811 851 891 931 971 1011 1051 1091; do n=$((d+1)); scripts/run/run_swg_double_gyre_gpu.sh 0 --deck maddison --days 4 --restart /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_1095d/day$(printf '%04d' "$d").npz --hourly-targets --verify-next /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_1095d/day$(printf '%04d' "$n").npz --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/hourly_targets/start_day$(printf '%04d' "$d"); done
   ```

   Then run the committed balanced-start analyzer with raw root
   `round1/filtered_truth_dg50/snapshots`, conservative root
   `round2/conservative_truth_dg50_1095d/snapshots`, hourly root
   `round2/hourly_targets`, and output
   `round2/balanced_start/balanced_start_10x48h.json`:

   ```bash
   PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_balanced_start.py --raw-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round1/filtered_truth_dg50/snapshots --conservative-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/conservative_truth_dg50_1095d/snapshots --hourly-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/hourly_targets --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/balanced_start/balanced_start_10x48h.json
   ```

2. Acquire the balanced 120-day twins from admitted day 1095:

   ```bash
   cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-2559510110
   scripts/run/run_swg_double_gyre_gpu.sh 0 --deck maddison --days 120 --restart /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_1095d/day1095.npz --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/fine_balanced_control_gpu_120d
   scripts/run/run_swg_double_gyre_gpu.sh 0 --deck maddison --days 120 --restart /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_1095d/day1095.npz --perturb-kind balanced-vorticity --perturb 1e-11 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/fine_balanced_1e-11_gpu_120d
   scripts/run/run_swg_double_gyre_gpu.sh 0 --deck maddison --days 120 --restart /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/legoesm_runs/dg_gpu_1095d/day1095.npz --perturb-kind balanced-vorticity --perturb 1e-10 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/fine_balanced_1e-10_gpu_120d
   ```

   Score each perturbed root against the control with the committed analyzer:

   ```bash
   PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_predictability.py --control-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/fine_balanced_control_gpu_120d --perturbed-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/fine_balanced_1e-11_gpu_120d --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/balanced_1e-11_analysis.json
   PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_predictability.py --control-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/fine_balanced_control_gpu_120d --perturbed-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/fine_balanced_1e-10_gpu_120d --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/predictability/balanced_1e-10_analysis.json
   ```

3. Acquire the first three years of the 3.125 km sensitivity, then restrict
   and score all three common years:

   ```bash
   cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgonl-2559510110
   scripts/run/run_swg_double_gyre_gpu.sh 0 --deck dg3125 --days 1095 --daily-means --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_gpu_1095d
   PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python packages/ocean/legoesm/ocean/fidelity/swg_coarse_graining.py --source-deck dg3125 --input /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_gpu_1095d --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_conservative_dg50_1095d
   PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_online_reference_adequacy.py --candidate-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/conservative_truth_dg50_1095d/daily_means --sensitivity-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/dg3125_conservative_dg50_1095d/daily_means --years 3 --json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/reference/adequacy_3y.json
   ```

4. Localize and close the dg50 day-9-to-30 numerical ladder debt before any
   training. The bar remains `1e-15`; it is not a user-tunable decision.

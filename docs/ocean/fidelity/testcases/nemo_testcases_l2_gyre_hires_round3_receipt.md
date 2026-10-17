# NEMO testcase L2 GYRE high-resolution round 3 receipt

**Date:** 2026-10-13  
**Incoming tip:** `7b772afb8`  
**Implementation commits:** `2215f79fe`, `f5120e1e9`, `9cdfd04c1`,
`7e3c56d46`, `38602e64c`, `41bff60ce`

## Verdict

Both acquired quarter-degree ladders are **DEBT** under the unchanged exact
bar: zero unequal active float64 values in T, S, U, V, and SSH at every
before-state entry kt=1..10 under CPU/fp64/libm.  Every field's one-ULP plant
fires.  The r4 and r4_smag ladders are identical through kt=2, so that first
residue is inherited from the resolution rung.  The first Smagorinsky-specific
movement is at kt=3, in all five fields.  It is therefore incorrect to label
either quarter-degree ladder a trajectory match.

The five-year climate claim remains deliberately different: the native-XLA GPU
and certified-libm CPU/NEMO trajectories are expected to diverge.  This round
preregisters and implements a matched-window statistical comparison, not a
trajectory comparison.  No physics or card configuration changed.

## Side-by-side ten-step ladders

The table reports each field's maximum active absolute error.  kt=1 is exactly
zero for both decks and all five fields.

| entry | field | r4 | r4_smag | disposition |
|---|---:|---:|---:|---|
| kt=2 | T | 1.0658141036401503e-14 | 1.0658141036401503e-14 | inherited |
| kt=2 | S | 2.1316282072803006e-14 | 2.1316282072803006e-14 | inherited |
| kt=2 | U | 1.1032841307212493e-15 | 1.1032841307212493e-15 | inherited |
| kt=2 | V | 1.1032841307212493e-15 | 1.1032841307212493e-15 | inherited |
| kt=2 | SSH | 2.6020852139652106e-18 | 2.6020852139652106e-18 | inherited |
| kt=3 | T | 3.3392346665550576e-09 | 8.223264913453932e-08 | port-specific increment |
| kt=3 | S | 1.171116537079797e-10 | 6.677225883322535e-09 | port-specific increment |
| kt=3 | U | 1.2599359056064685e-11 | 2.9869591345862978e-05 | port-specific increment |
| kt=3 | V | 1.39709216417927e-11 | 2.987027689110161e-05 | port-specific increment |
| kt=3 | SSH | 1.8483160314775038e-11 | 6.404415412407383e-08 | port-specific increment |
| kt=10 | T | 2.2064320148729166e-08 | 1.2194657207675874e-02 | port-specific increment |
| kt=10 | S | 1.3992078606861469e-09 | 8.031588703971693e-04 | port-specific increment |
| kt=10 | U | 7.613925046001668e-10 | 1.7707810187277805e-03 | port-specific increment |
| kt=10 | V | 1.990153063180511e-09 | 5.351313327050338e-03 | port-specific increment |
| kt=10 | SSH | 1.9252387011003957e-10 | 1.3831116240601085e-05 | port-specific increment |

The executed NEMO step calls stages 1, 2, and 3 in that order, then swaps the
completed stage into `Nbb`, in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:195-216`.  The committed gate
reads the acquired after-stage records and scores them with the same active
masks and ULP implementation as the before-state ladder.

## Resolution-rung diagnosis

The first differing RK stage is **stage 1** and the first field in the gate's
canonical T, S, U, V, SSH order is **T**: seven active T cells differ and the
maximum is exactly **1 ULP** (`3.552713678800501e-15`).  S has six differing
cells and is also 1 ULP.  The earliest differing measured operands are the
**stage-1 advective transports**, before `tra_adv`: `zFu` differs in 194,349
active values (maximum `2.8421709430404007e-13`, 5 ULP at the maximum-absolute
cell), `zFv` in 189,688 (maximum `3.410605131648481e-13`, 6 ULP there), and
`zFw` in 287,138 (maximum `5.331957098064777e-12`, 24,013 ULP there).  `zFu`
is therefore the first differing term in the record's `zFu`, `zFv`, `zFw`
order.  NEMO forms horizontal `zFu`/`zFv` from metric, thickness, qco, velocity,
and barotropic-correction factors and forms `zFw` from the vertical velocity in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:260-302`.

`tra_adv` consumes that already-different bundle: immediately afterward T has
280,360 unequal accumulator values with maximum absolute residue
`1.7724826165786277e-20`; S has 270,982 with maximum
`2.795787752367978e-20`.  The following surface-boundary call changes neither
count nor maximum.  The compiled call order is cited at
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:453-476`; the acquired record
is written around those calls in
`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:604-634`, and the thickness-
weighted stage update that turns the accumulator residue into the observed
one-ULP T/S state residue is
`GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:650-675`.  This localizes the
owner boundary to stage-transport construction/composition; it refutes the
stronger claim that exact transports first diverge inside `tra_adv`.  Which
multiplicand first moves inside `zFu` remains open.

The preregistered first candidate, the stage-entry TKE/EVD vertical coefficient
bundle, is **refuted as the owner**.  Its direct comparison is not exact (avt:
73,436 unequal, maximum `2.6020852139652106e-18`, 9 ULP; avm: 73,601 unequal,
maximum `2.42861286636753e-17`, 7 ULP), but substituting only NEMO's recorded
avt/avm leaves the kt=2 T and S maxima unchanged at 3 ULP.  This also agrees
with program order: `zdf_phy` builds the profiles before the RK ladder in
`cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:159-183`, while implicit vertical
momentum/tracer completion is a stage-3 operation, not the stage-1 advection
boundary that first differs.

The remaining ordered candidates are dispositioned without guessing.  NEMO's
run resolves 50 split-explicit substeps and the card also computes 50, so the
substep **count** is not different.  The card derives its 26.5 km metrics using
the source sequence `106000 / REAL(nn_GYRE,wp)`, rather than a typed 26500, as
shown in `cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90:84-94`.  Round 2 already proved
the seasonal forcing clock and phase bit-equal.  The measured remainder is
composition debt already present in the stage-transport bundle, not ownership
by the tested stage-entry TKE/EVD coefficient bundle, nor a substep-count,
grid-value, forcing, or configuration transcription difference.  No additional
NEMO record is needed for this round's required localization; isolation below
the `zFu` expression is carried as open debt rather than guessed.

## Preregistered climate instrument

`nemo_testcase_l2_gyre_hires_climate.py` refuses any NEMO deck other than the
registered r4_smag manifest and resolved selectors.  It requires eight T, U,
and V tiles, places them from one-based `DOMAIN_position_first`, verifies the
global 122x82 domain, daily averaging operation, 3600 s sample interval, and
86400 s write interval.  NEMO SST is decoded as same-file
`sosstsst/vovvle3t`; the legoESM provider requires a complete content-pinned
manifest and reads the driver's thickness-weighted daily-temperature means.
It content-pins all 24 NEMO tiles, verifies the exact operator-launched
`7b772afb8` legoESM commit and GPU/native/non-certified policy, refuses
non-finite daily fields or incomplete/unpinned day sequences, and refuses any
scored window other than days 1461..1825.  A real NEMO day passed this path and
assembled four 82x122 fields.

The primary comparison window is year 5, days 1461..1825.  The 31-day
high-pass filter is a centered uniform filter with nearest-edge treatment; it
loads days 1446..1825 so the target begins after 15 real context days.  Both
models use the identical window and operator.  The instrument reports:

- daily basin-mean surface KE and its window mean;
- the 31-day high-pass EKE map, basin mean, and EKE/KE eddy fraction;
- the row of the year-mean absolute-U maximum;
- the zonal-mean SSH zero crossing nearest that row, as row and latitude;
- time-mean SSH, thickness-weighted SST, and surface speed fields; and
- the mean isotropic kinetic-energy wavenumber spectrum.

Before the legoESM run completes, the registered scalar bars are fixed as:
year-5 mean basin KE relative difference at most 20%, eddy-fraction relative
difference at most 20%, absolute-U jet-row difference at most two rows, and
SSH-zero-crossing-row difference at most two rows.  The fields and spectrum
are report-only pattern diagnostics.  Trajectory divergence is expected and
is not a failed climate result.

KE, high-pass EKE, eddy fraction, and the reference RMS-speed map use the
operator reference's native masked U/V index convention.  A separately named
time-mean speed field and the spectrum use the shared C-grid interpolation to
T points; the two conventions are not silently interchanged.

Land is NaN-masked in every reported map.  The report gives bias, mean absolute
difference, RMSE, normalized RMSE, and spatial/spectral correlation over common
finite support for every report-only pattern.  A content-pinned PNG places the
two common-scale model maps next to their difference and overlays the daily KE
series and KE spectra.

## Movie instrument

`nemo_testcase_l2_gyre_hires_movie.py` renders daily-mean frames for year 1
(days 1..365) and year 5 (days 1461..1825).  Each frame has one NEMO row and
one legoESM row with surface relative vorticity/f, SSH, surface speed, and SST.
It uses the shared C-grid velocity and vorticity diagnostics, scans both models
first to establish common per-year color limits, and streams 1280x640 RGB
frames to `imageio_ffmpeg.write_frames` using H.264.  The output manifest pins
provider provenance (including NEMO tile content hashes and the legoESM run
manifest), windows, frame count, scales, fps, and movie digests.

## Tests, evidence, and controls

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round3/`.

- `r4_ladder.json` and `r4_ladder.log`: final r4 stage, term, vertical-control,
  ten-step, ULP, and planted-control report.
- `r4_smag_ladder.json` and `r4_smag_ladder.log`: final r4_smag stage and
  ten-step report against the operator-acquired round-2 records.
- Five ladder-reader/control tests, six climate tests, three movie tests, and
  one driver test pass directly.  They cover runtime record dimensions,
  one-ULP firing,
  one-based tile placement and overlap refusal, same-file weighted-SST decode,
  the exact native-U/V 31-day high-pass calculation, land masks, exact
  registered bars and window, GPU-run metadata refusal, report-only comparison
  arithmetic, common movie scales, the real registered-grid movie diagnostics,
  and the encoder frame shape.
- The real NEMO-day provider smoke check passes.  The full climate score and
  movies correctly remain unrun while the already-launched legoESM five-year
  GPU run has only a pending manifest.
- The receipt citation gate is run explicitly after this receipt is committed;
  its report and planted controls are stored with the round-3 evidence.

## Inherited pins / OPEN

Inherited unchanged are the r4_smag decision (quarter-degree grid, 3600 s
clock, 25 km isoneutral Laplacian tracers, mode-32 level Smagorinsky
bilaplacian momentum), 31-level z-coordinate, TEOS-10, FCT2, vector/C2/ENE,
SCO pressure gradient, split-explicit AB3/AM4, nonlinear drag, TKE/EVD,
two-band shortwave, seasonal analytic forcing, qco/vco-1d/RK3, closed
free-slip walls, no fixers, the native-transcendental non-certified GPU mode,
and the certified CPU/libm ladder policy.  The shipped one-degree card and its
digest are untouched by this round.

OPEN:

1. the operator-launched legoESM r4_smag run must finish all 1825 days and
   atomically publish `manifest.json`; this round does not relaunch it;
2. after completion, run the committed year-5 climate gate and the year-1 and
   year-5 movie instrument against the existing five-year NEMO reference;
3. the r4 resolution rung remains measured debt already present in the
   RK-stage-1 advective-transport bundle; the first moving multiplicand inside
   `zFu` is not yet isolated.  r4_smag adds a separately measured port-specific
   increment from kt=3; neither is a trajectory match;
4. the later 1/12-degree deck, acquisitions, integrations, bars, and movies
   remain future work.

# Preregistration — NEMO testcase L2 GYRE round 158

Date: 2026-09-23

Incoming lane tip: `1558fa0b13fa7dd3f1ddc70db97b5d5f7b0b1fd8`.  Evidence
belongs under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round158/`.
This document is frozen before any legoESM stage-2 advection measurement is
run.

## The order this round executes

Round 157's OPEN section: split `dyn_adv` at
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:525`, entered
through the vector-invariant arm selected at `:523`, into its two compiled
halves, and calibrate each against NEMO's OWN recorded operands before naming
a statement.  No new acquisition: the admitted round-156 record already
carries NEMO's `uu(Kmm)`, `vv(Kmm)`, `ww`, `r3u/r3v(Kmm)` and `umask/vmask`
at this stage boundary.

## What was read from the oracle side before freezing this document

Recorded here so that nothing below is a post-hoc prediction.

1. The resolved namelist of the record's run
   (`round157/oracle_developed_stage2/namelist_cfg:162-163`) sets
   `ln_dynadv_vec = .true.` and `nn_dynkeg = 0`, so the compiled selector at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:153` takes
   `np_VEC_c2`, which calls `dyn_keg` with the C2 scheme and then `dyn_zad`.
2. `ocean.output:554` of that run reports `ln_zad_Aimp = F`, so `dyn_zad`
   takes its explicit arm and no adaptive-implicit vertical advection runs.
3. `ln_vortex_force` is not set on this deck, so `dyn_zad` takes its
   `We = ww` branch and reads no Stokes drift.

The two compiled halves, cited from the record's own build:

| half | compiled lines | what it reads |
|---|---|---|
| `dyn_keg`, C2 | `dynkeg.f90:121-125` (the T-point kinetic energy) and `:129-130` (its gradient added to `Krhs`) | `puu/pvv(Kmm)`, `r1_e1u`, `r1_e2v` |
| `dyn_zad`, explicit | `dynzad.f90:112-120` (the vertical transport and its shear product) and `:123-126` plus the bottom statement at `:134-137` | `ww`, `puu/pvv(Kmm)`, `e1e2t`, `r1_e1e2u/v`, `e3u/e3v_3d`, `r3u/r3v(Kmm)`, `umask/vmask` |

## Instrument

One production step from NEMO's admitted day-180 entry through
`LatLonCGridOceanModel.step` under production just-in-time compilation, the
same bridge and the same entry bundle round 157 used.  The two halves are
read from legoESM's existing stage-2 operator components (`keg_u/keg_v` and
`zad_u/zad_v`), which the model already publishes next to the `advection_u/v`
bucket round 157 scored.

Whichever exposure route is used, it is CONTROLLED before it is believed: its
stage-2 `after_adv` row must reproduce round 157's production after-advection
row (active rms `3.844166e-12` u, `6.428546e-12` v) bit for bit.  A route that
fails that control is not used, and the receipt says which route was.

## The rebuild, which is the calibration

legoESM's OWN transcription of each half is driven by NEMO's OWN recorded
operands and the card's static metrics:

* `R_keg` — legoESM's C2 kinetic-energy helper on NEMO's `uu(Kmm)`/`vv(Kmm)`,
  differenced with the card's `r1_e1u`/`r1_e2v`;
* `R_zad` — legoESM's NEMO-advective vertical-advection kernel on NEMO's
  `uu(Kmm)`/`vv(Kmm)`, NEMO's `ww`, the live stage thickness built from
  NEMO's `r3u/r3v(Kmm)` and the card's reference thickness, the card's face
  areas, and NEMO's `umask/vmask`.

`Δ_adv` is NEMO's own advection increment, `after_adv − after_vor` from the
record, which carries one rounding on the oracle side and is labelled as such.

The per-half differences are `D_keg = keg_production − R_keg` and
`D_zad = zad_production − R_zad`.

## Predictions, frozen

1. **The transcription is exact given NEMO's operands.**  The rebuilt sum
   `R_keg + R_zad` reproduces NEMO's `Δ_adv` with an active rms at least one
   hundred times smaller than the production after-advection difference
   (`3.844166e-12` m/s^2, u).  REFUTED if the rebuilt sum's active rms
   difference exceeds `3.9e-14` m/s^2 on u.
2. **The vertical advection owns the magnitude.**  `D_zad` has a larger
   active rms than `D_keg` on both components, by at least a factor of two.
   REFUTED if `D_keg` is larger on either component, or if the two are within
   a factor of two (in which case no half owns it and the receipt says so).
3. **The split accounts for the whole difference.**  `D_keg + D_zad`
   reproduces the production after-advection difference to better than one
   percent on both components.  REFUTED outside that.
4. **The two halves sum to the bucket.**  legoESM's own `keg + zad`
   reproduces its `advection` bucket, so the split adds no term.  REFUTED by
   any active face differing by more than the one instrument-side addition
   (one unit in the last place at the term's scale).
5. **Inside the named half, one operand carries it.**  After the half is
   named, one operand at a time is replaced by NEMO's recorded value in the
   rebuild, with the all-legoESM rebuild as the null arm.  Prediction: a
   SINGLE operand removes at least half of that half's difference.  REFUTED
   if no single operand removes half, in which case the receipt names the
   set rather than a statement.
6. **The plant fires.**  A plant that moves NEMO's recorded `ww` by one unit
   in the last place changes the reported `R_zad` row and the instrument
   exits nonzero with `STATUS PLANT-FIRED`.

## Landing rule

No candidate lands unless it is a SINGLE source-exact statement proven under
production just-in-time compilation and it passes the full Decision 43/45
gate: the month, day 240 and day 360 not worse, kt2 T/S at the bar,
first-over-bar not earlier, every moved row registered with the run-to-run
noise floor (about `2e-10` K, round 129) quoted next to any change below
`1e-6` K, DINO measured if the statement is shared, the generic NEMO-GYRE
card, the LOCK_EXCHANGE and OVERFLOW tanks, the six-file push gate, and a
Claude reviewer's verdict quoted verbatim.  Ranking is by what a statement
carries at day 240, not at day 180.  The expected outcome of a walk round is
HELD.

## Rules

No configuration choice, no carried-state change, no stabiliser NEMO lacks,
no NEMO source modified, no `makenemo` and no `mpirun` from inside the round.
Independent review is not run by codex this round (codex is paused); a Claude
reviewer is spawned instead for any landing.

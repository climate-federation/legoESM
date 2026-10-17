# ORCA2 round 302 — per-region history, D132 scratch arm, and the kt-1 stage-3 momentum EVD

Date: 2026-10-16 (Claude round; codex review: paused (quota)).

Status: **CENSUS PENDING** (this section is replaced when the Decision-96
census returns).

## Headline

- The interior half of the kt-2 departure is ONE transcription defect: the
  ORCA2 card mixes MOMENTUM with the enhanced vertical diffusion (EVD,
  `nu_conv = 100 m2/s`, inherited from GYRE whose deck sets `nn_evdm = 1`),
  while every ORCA2 deck runs `nn_evdm = 0`.  NEMO's `zdf_evd` touches the
  viscosity only inside `IF( nn_evdm == 1 )`
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121` to
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:135`; identical in the record
  build at `ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/zdfevd.f90:121` and
  `ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/zdfevd.f90:135`; the OMT-4 record
  resolves `nn_evdm = 0` in its `ocean.output`, line 565).  With
  `nu_conv = 0` the kt-1 stage-3 interior U of rounds 298-301's run falls
  from 6.64e-2 to 1.60e-16 m/s (V 2.61e-2 -> 1.60e-16).
- The fold-band half is unchanged by it: it is the held private fold unit's
  territory (stage-1 V mask, 1,319 faces, rounds 258/283/286).
- D132 scratch arm: the carried after-SSH slot leaves kt 1 bit-identical and
  moves kt 2-3 by at most 0.04% in either region — **unchanged**.
- Correction to rounds 298-301: their "free run" was not the shipped card.
  It carried round 249's private atomic fold unit and round 245's barotropic
  surface-mask arm.  On the shipped OMT-4 card everything is over the bar at
  kt-1 stage 1 already.

## Provenance

- Base `08d30232a`; preregistration `0727ea851` (frozen before measuring).
- History probe `scripts/validate/ocean_fidelity/orca2_l4/
  nemo_testcase_l4_orca2_round302_region_history.py`, measured at clean tree
  `f66ad3adb` (CPU, fp64, libm, production JIT).  Card = round 223's OMT-4
  card; record = round 222's `orca2_omt4_frames_10step_a_np2` (stage frames
  s0-s3, both ranks; NEMO wet masks from its rank mesh files).  Fold band =
  last three T rows; interior = rows 0-144.  Relative = RMS(card - NEMO) /
  RMS(NEMO) over the region's wet points.
- The card's own initial state equals NEMO's kt-1 entry bit for bit on every
  wet cell (dry T cells differ only by signed zero), so the "given NEMO
  entry" label is the same run; the probe refuses otherwise (entry-bit plant
  fires, exit 2).
- Evidence: `orca2_rounds/round302/` (`history_{card,unit,carried}.json`,
  `kt1_*.npz` scratch dumps and the scratch scripts that wrote them).

## 1. Per-kt, per-region history (step entry, relative RMS %, fold / interior)

Rounds 298-301's run (`unit`: private fold unit + surface-mask arm):

| kt | U | V | sea level | T | S |
|---:|---|---|---|---|---|
| 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | 12.3 / 1.72 | 62.6 / 0.947 | 0 / 0 | 2.47 / 1.9e-14 | 1.69 / 9.3e-15 |
| 3 | 58.7 / 2.74 | 78.3 / 3.18 | 57.9 / 2.09 | 2.78 / 2.6e-4 | 1.89 / 8.7e-5 |
| 4 | 122 / 3.48 | 109 / 5.54 | 51.1 / 2.48 | 3.32 / 1.3e-3 | 2.25 / 3.0e-4 |
| 5 | 204 / 4.80 | 190 / 6.67 | 60.8 / 2.86 | 4.49 / 3.4e-3 | 3.07 / 4.6e-4 |
| 6 | 281 / 4.81 | 276 / 6.25 | 52.1 / 2.44 | 7.76 / 6.7e-3 | 5.32 / 6.8e-4 |
| 7 | 1170 / 4.94 | 1630 / 7.00 | 85.4 / 2.17 | 33.5 / 8.0e-3 | 23.4 / 8.2e-4 |

The review's scratch numbers reproduce (U fold 12/59/281%, interior
1.7/2.7/4.8% at kt 2/3/6).  U unequal at kt-2 entry: fold 3,463 / 3,933,
interior 203,314 / 409,097 (as round 301).

The shipped OMT-4 card (`card`, no private hooks):

| kt | U | V | sea level | T | S |
|---:|---|---|---|---|---|
| 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | 105 / 5.12 | 86.4 / 4.68 | 195 / 5.51 | 0.048 / 2.7e-4 | 0.002 / 9.4e-5 |
| 3 | 126 / 6.08 | 98.9 / 7.15 | 193 / 5.35 | 0.159 / 1.4e-3 | 0.0064 / 5.2e-4 |
| 6 | 105 / 6.78 | 80.7 / 8.04 | 95.6 / 3.98 | 0.317 / 7.5e-3 | 0.0145 / 1.2e-3 |
| 7 | 85.3 / 6.33 | 92.2 / 8.30 | 103 / 3.72 | 0.380 / 8.8e-3 | 0.0158 / 1.3e-3 |

## 2. D132 scratch arm (not landed)

Same card, `nemo_first_wzv_after_ssh='rk3_extrapolated_carried'`, kt 1-3,
every stage.  kt 1: all rows identical.  kt 2-3, all five fields, both
regions: relative RMS moves by -0.036% to +0.009%.  The kt-2 departure is
**unchanged** (neither collapses nor shrinks), in the fold band and in the
interior.

## 3. The earliest departure, by stage (kt 1 is where the kt-2 entry debt is made)

The prompt's premise needs one correction: the kt-2 entry debt is created
during kt 1, so the earliest over-the-bar departure is a kt-1 stage.

| run | region | first over the bar | what |
|---|---|---|---|
| card | fold + interior | kt 1 stage 1, every field | sea level 195% fold / 5.5% interior; every departing cell of stages 1-2 lies in rows 121-147 (fold error spread by the barotropic substeps) |
| unit | fold | kt 1 stage 1, V and T | 1,319 V faces = the stage-1 V mask (rounds 258/283; resolved-mask route held, round 286) |
| unit | interior | kt 1 stage 2 at rounding (U 4.7e-16 m/s), then stage 3 | stage 3: U 6.64e-2 m/s, V 2.61e-2 m/s, sea level exact, T 1.4e-12 K |

The stage-3 interior jump is bottom-weighted (argmax `[100,139,20]`, the
column's bottom cell: NEMO -0.1327, card -0.0663 m/s; the two bottom cells
come out nearly equal on the card, i.e. over-mixed).  Stage 3 is where NEMO
applies vertical viscosity.  Statement: `zdf_evd`'s momentum branch, which
NEMO skips under `nn_evdm = 0` (cited above).  Measured on both runs at kt-1
stage 3 with `nu_conv = 0`, everything else held:

| run | U interior max (m/s) | V interior max (m/s) | U fold max | sea level / T |
|---|---|---|---|---|
| unit, `nu_conv = 100` | 6.64e-2 | 2.61e-2 | 1.83e-2 | unchanged |
| unit, `nu_conv = 0` | 1.60e-16 | 1.60e-16 | 1.83e-2 | unchanged |
| card, `nu_conv = 100` | 6.64e-2 | 5.30e-2 | 6.18e-2 | unchanged |
| card, `nu_conv = 0` | 3.93e-2 (rows >= 121 only) | 5.30e-2 | 6.18e-2 | unchanged |

CONFIRMED (one-variable arm, both runs).  Stages 1-2 are bit-identical
between the arms, as they must be.

## Landing

The card's ORCA2 identity now states `nu_conv = 0.0` next to the citation;
GYRE's identity is untouched.  `test_orca2_evd_mixes_tracers_only_while_gyre_mixes_momentum_too`
passes, and fails with the fix reverted (file restored, checked).

Census: PENDING.

## Configuration choices

- ASKED: none.  The `nu_conv` change is the transcription of the deck's
  `nn_evdm = 0` that the round prompt authorised landing; reported here so it
  can be reverted if read as a choice.
- UNASKED: none.  D126 and D132 unchanged.

## OPEN

1. Fold band: the held private fold unit (stage-1 V mask + barotropic fold
   statements) still owns the earliest fold departure on both runs.

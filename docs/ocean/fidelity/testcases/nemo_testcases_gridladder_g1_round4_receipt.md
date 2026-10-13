# Grid ladder rung G1 — round 4 receipt (gridladder_r4)

Status: **STOPPED_FOR_RECORD**, with one cited stage statement landed
(card-selected). The 100-step record is NOT at the bar and its headline did
not move: sea level first leaves the bar at step 17, before and after. Both
open questions were walked to the same place, the inside of the
split-explicit substep loop, and the acquisition that records it is written
(section 8).
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round4.md
(commit 98b405b9f, before any measurement). Code commits 5204bad9d (option,
instrument, tests), 2c602e127 (acquisition), aa0f7ee8f (review fixes, line
numbers). Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round4/final/
(json stamped 5204bad9d; the replay and the step-6 stage pair stamped
aa0f7ee8f; the commits between them change comments, row labels and the
record checker, no model expression).
Record: the operator's round-3 acquisition,
.../gridladder_rounds/round3/oracle_g1_r3/p3_a_100step (stage frames at
steps 1 to 50). Baseline: the round-3 receipt's tables.

## Take-aways

1. **The step-17 departure is not made inside step 17.** Handed NEMO's whole
   entry state, T and S included, the card's step 17 is at rounding
   everywhere: sea level 1.4e-17 over the domain, 1.7e-18 at the cell.
   **Round 3's reading "made inside one step" is retracted**: its one-step
   arm kept the card's own T and S, and those carry the event.
2. **The carrier is two tracer values.** Entering step 17 the card's T is 2
   and its S is 3 last-place units above NEMO's at that one cell. The first
   statement where this is above rounding is the density feeding the pressure
   gradient of the right-hand side; the solver turns it into the sea-level
   error.
3. **The root is the solver's last place, from step 1.** Tracer advection is
   off on this deck, so T and S are re-divided by the layer thickness at
   every stage. The card's sea level leaves the solver differing from NEMO's
   in the last place at about 10,000 cells per step; that flips the rounding
   of the thickness quotient at about one cell per step. Not registered as a
   pure rounding row: the solver statement is not named yet.
4. **Fold residual: also the solver.** At steps 31 and 41 the right-hand side
   and all three stages are at rounding on the fold rows; the solver's output
   is the first row beyond rounding (2e-10, then 2e-07). F-P6 HOLDS.
5. **Landed:** NEMO's stage-1 sea level, in NEMO's order. The stage-1 row
   given NEMO's solver output goes from 2,870 unequal cells to 0 (step 6). It
   moves no step of the headline.

## 0. Admission

- check_records, 100 steps, 50 stage steps, twin and reference: PASS_G1_RECORD
  (admission_r4.json, byte-identical to the operator's admission file,
  SHA-256 b07e282effabd67e). No self-check refused; nothing re-admitted.
- N-P0 HOLDS: every frame of round 1's record equals the new record's byte
  for byte, steps 1 to 100 (0 differing). The round-3 baseline carries.
- Geometry: 36 of 36 rows bit-identical on the new record.

## 1. The step-17 walk (item 1)

Cell: inner-domain (72, 70), 19 cells from land, 75 rows from the fold (round
3's cell; re-located by the independent chain on the new record). Compiled
order, error at the cell (NEMO's value), two arms that differ in one thing:
whether the entry T and S are NEMO's or the card's own chain's. Every other
entry field is NEMO's in both.

| row | NEMO's T, S (GIVEN-NEMO-ENTRY) | card's T, S (measurement arm) | NEMO's value |
|---|---|---|---|
| entry T / S at the cell | 0 / 0 | +7.1e-15 / +1.07e-14 (2 / 3 last-place units) | 20 / 30 |
| right-hand side, u | 0 | +2.45e-19, and -2.45e-19 on the west face | 3.75e-09 |
| right-hand side, v | -4.1e-25 | +9.78e-19, and -9.68e-19 on the south face | 2.11e-09 |
| handoff sea level (stage-1 level) | 0 | -1.91e-15 | 9.12e-03 |
| handoff barotropic u / v | 0 / -6.5e-19 | -1.3e-17 / -2.8e-17 | 1.44e-03 / -3.0e-05 |
| stage 1, 2, 3 sea level | 0, 0, 0 | 0, 0, 0 (fed NEMO's solver output) | |
| stage 1, 2, 3 T | 0, 0, 0 | +7.1e-15 each (the entry value, carried) | 20 |
| whole step, sea level | +1.7e-18 | **-5.74e-15** | 3.90e-03 |
| whole step, largest sea-level error anywhere | 1.4e-17 | 5.74e-15 (this cell) | |

- First non-bit operand: the entry T and S at the cell.
- First statement above rounding: the density and pressure gradient of the
  right-hand side, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:137-138`. The
  error is a dipole on the cell's own faces and zero elsewhere (largest open
  face error 2.45e-19 on u, 9.78e-19 on v).
- The handoff reads the stage-1 level, one third of the way
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:148`): 3 x -1.91e-15 =
  -5.73e-15, the after-step error.
- Arithmetic only, PLAUSIBLE: 2.45e-19 m/s2 across one cell is a density
  change near 1e-16 of the reference density, the size of one last-place
  unit of a 1020 kg/m3 density. Not measured on the density itself.
- INDEPENDENT label, same step: -5.75e-15 at the cell, 2 cells over the bar
  (round 3's census, reproduced by the chain here).

Steps 20 and 22 (I-P5): the whole step from NEMO's entry is at the bar at
both (1.0e-16, 1.7e-16), so the same carrier explains them. The stage-by-stage
walk was not repeated there.

## 2. Where the tracer difference comes from

Census, both labels (wet cells where T / S differ from NEMO's after the
step; 16,433 wet):

| step | INDEPENDENT chain: T cells / S cells / sea-level error | one step from NEMO's entry: T cells / S cells |
|---|---|---|
| 2..6 | 0 / 0 / 5e-17 | 0 / 0, except step 5: 1 / 1 at (82, 57) |
| 7 | 0 / 1 / 5.0e-17 | 0 / 1 |
| 8 | 2 / 1 / 5.0e-17 | 0 / 0 |
| 11 | 6 / 4 / 7.4e-17 | 2 / 2 |
| 16 | 6 / 2 / 7.0e-17 (the cell (72, 70) appears here) | 0 / 0 |
| 17 | 5 / 0 / **5.8e-15** | 0 / 0 |
| 20 | 13 / 6 / 8.6e-15 | 2 / 2 |
| 22 | 22 / 12 / 6.9e-15 | 2 / 2 |
| 25 | 49 / 29 / 4.7e-15 | 3 / 3 |
| 29 | 90 / 25 / 8.9e-15 | 0 / 0 |
| 30 | 97 / 34 / 8.0e-15 | 2 / 2 (sea level 1.1e-15: the fold arrives) |
| 40 | 778 / 631 / 1.4e-12 | 234 / 256 |
| 49 | 1879 / 1485 / 7.1e-11 | 709 / 743 |

Each difference is 1 to 5 last-place units (largest 1.8e-14) until the fold
error arrives near step 30.

The statements, replayed offline from NEMO's frames alone (numpy, no model
step; 16,433 wet cells times 49 steps each):

| statement | NEMO's order against NEMO's record | the card's order against NEMO's record |
|---|---|---|
| stage-1 sea level, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:148` | 0 unequal | **321,126 unequal**, largest 1.4e-17 |
| stage-2 sea level, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:211` | 0 | 0 |
| after-level thickness ratio, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:257` | 0 | 0 |
| stage-1 ratios t, u, v, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:171-173` | 0, 0, 0 | not replayed in the card's order |
| stage-2 ratios t, u, v, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:215-217` | 0, 0, 0 | t: 1 unequal (step 40) |
| stage-1, stage-2 T and S, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:503-505` | 0 | stage 1: 0; stage 2: 1 (step 40) |
| stage-3 T and S, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:284-285`, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:294`, diagonal `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:233` | 0 | not replayed |

The constants are `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:52-53`;
the update arm is `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:48`.
The tracer rows test only the thickness quotient: the trend is zero on this
deck and the record does not hold it (review finding 2, a stated limit).

What the card does with NEMO's operands (stage arm, NEMO's solver output and
stage entry fed): stage 1 and stage 3 T and S have 0 unequal cells at steps
6, 17, 31 and 41; stage 2 has 1 cell at steps 6 and 41. So the whole-step
tracer flips are not made by the stage statements. They appear only when the
card's own solver output is used, and that output differs from NEMO's in the
last place: 1,010 sea-level cells at step 1, 10,762 at step 17, largest
6.9e-18 (round 2 found the same row first, with a bit-identical right-hand
side at step 1). The thickness quotient rounds that difference away at all
but about one cell per step.

Open, in compiled order (none of these is named to a statement yet):

| row not bit-identical, NEMO's operands | from | cells | size |
|---|---|---|---|
| solver handoff (sea level, barotropic u, v) | step 1 | 1,010 to 12,000 | 7e-18 |
| right-hand side on open faces | step 2 (zero unequal at step 1) | 608, 7,668 at step 17 | 5e-23 (NEMO near 4e-09 at the event cell) |
| stage u, v given NEMO's stage entry | step 1 | 300 to 6,300 | one last-place unit |
| stage-2 T, S | steps 6, 41 | 1 | 2 last-place units |

Prediction I-P4 asked whether every operand is NEMO's bit for bit. It is not:
the after-level sea level is not. So this is NOT registered as a rounding
row, and nothing was fitted to it. It closes only when the solver's first
non-bit statement is named.

## 3. Landed: NEMO's stage-1 sea level

NEMO: two rounded products and one sum,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:148`. The shared RK3 path
formed the level as the before value plus a third of the increment.
New option LatLonCGridOceanConfig.rk3_stage1_ssh_nemo_literal, default False,
the LAST field of its class; the statement sits inline at the one site,
on the three lines the old expression occupied (no line of the model file
moves, so earlier rounds' citations stay anchored), using the shared
rounding guard. Searched first: grep for the stage-1 level and for
citations of this block in packages/ocean (found the ratio helper
nemo_r3t_rk3_stage1_stretch, which already does this for the T ratio; no
sea-level twin). Caller grep afterwards (grep -rn rk3_stage1_ssh_nemo_literal
packages): the field, its one use, the G1 card (selection and validator). No
other card; the default path is the same expression as before.

| measure | before | after |
|---|---|---|
| stage-1 sea level given NEMO's solver output, step 6: unequal cells | 2,870 (largest 3.5e-18) | **0** |
| the other fourteen stage rows at step 6 | | unchanged counts |
| handoff sea level, step 17: unequal open cells | 11,437 (uncommitted-tree run) | 10,762 |
| 100-step record, steps 1 to 77 (483 rows) | | identical in maximum, RMS and count |
| 100-step record, steps 78 to 100 | | 79 of 115 rows move, by at most 2.9e-10 of their value; no status change |
| first step over the bar: ssh / u / uu_b / vv_b / v | 17 / 32 / 32 / 23 / 19 | 17 / 32 / 32 / 23 / 19 |

Why the record first moves at step 78 was not examined (UNVERIFIED).

**The 100-step record (INDEPENDENT)**, normalised error, before -> after:

| step | ssh | u | v | uu_b | vv_b | region of the ssh maximum |
|---|---|---|---|---|---|---|
| 10 | 4.34e-17 -> 4.34e-17 | 2.17e-17 -> 2.17e-17 | 1.82e-17 -> 1.82e-17 | 2.17e-17 -> 2.17e-17 | 1.82e-17 -> 1.82e-17 | open |
| 16 | 6.98e-17 -> 6.98e-17 | 2.43e-17 -> 2.43e-17 | 1.56e-17 -> 1.56e-17 | 2.36e-17 -> 2.36e-17 | 1.73e-17 -> 1.73e-17 | open |
| 17 | 5.75e-15 -> 5.75e-15 | 1.06e-16 -> 1.06e-16 | 2.48e-16 -> 2.48e-16 | 1.06e-16 -> 1.06e-16 | 2.48e-16 -> 2.48e-16 | open |
| 20 | 8.61e-15 -> 8.61e-15 | 2.94e-16 -> 2.94e-16 | 7.95e-13 -> 7.95e-13 | 2.94e-16 -> 2.94e-16 | 3.59e-16 -> 3.59e-16 | open |
| 30 | 8.04e-15 -> 8.04e-15 | 6.44e-16 -> 6.44e-16 | 5.35e-07 -> 5.35e-07 | 6.44e-16 -> 6.44e-16 | 5.96e-16 -> 5.96e-16 | open |
| 32 | 1.16e-14 -> 1.16e-14 | 9.99e-15 -> 9.99e-15 | 1.60e-06 -> 1.60e-06 | 9.99e-15 -> 9.99e-15 | 4.18e-15 -> 4.18e-15 | fold |
| 40 | 1.39e-12 -> 1.39e-12 | 3.95e-12 -> 3.95e-12 | 4.34e-05 -> 4.34e-05 | 3.95e-12 -> 3.95e-12 | 2.39e-12 -> 2.39e-12 | fold |
| 60 | 4.68e-10 -> 4.68e-10 | 1.09e-09 -> 1.09e-09 | 5.02e-04 -> 5.02e-04 | 1.09e-09 -> 1.09e-09 | 8.05e-10 -> 8.05e-10 | fold |
| 80 | 5.11e-09 -> 5.11e-09 | 1.30e-08 -> 1.30e-08 | 4.43e-03 -> 4.43e-03 | 1.30e-08 -> 1.30e-08 | 7.77e-09 -> 7.77e-09 | fold |
| 100 | 5.30e-08 -> 5.30e-08 | 8.29e-08 -> 8.29e-08 | 4.55e-03 -> 4.55e-03 | 8.29e-08 -> 8.29e-08 | 6.60e-08 -> 6.60e-08 | fold |

The v column is the stored top V row (zero against NEMO's image), open since
round 2.

**Ladders, steps 1 to 10, card as landed** (max abs / RMS; all AT-BAR;
identical to round 3's to every printed digit):

| step | INDEPENDENT ssh | u | v | cells not bit-equal (ssh) | GIVEN-NEMO-ENTRY ssh | u | v | cells |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.16e-17 / 9.39e-19 | 6.94e-18 / 2.18e-19 | 8.67e-18 / 2.30e-19 | 1070 | 4.16e-17 / 9.39e-19 | 6.94e-18 / 2.18e-19 | 8.67e-18 / 2.30e-19 | 1070 |
| 2 | 4.86e-17 / 1.29e-18 | 1.21e-17 / 3.74e-19 | 1.21e-17 / 3.27e-19 | 2711 | 2.78e-17 / 9.11e-19 | 1.04e-17 / 3.13e-19 | 8.67e-18 / 2.63e-19 | 2654 |
| 3 | 3.99e-17 / 1.36e-18 | 1.73e-17 / 5.61e-19 | 1.21e-17 / 4.21e-19 | 4833 | 2.43e-17 / 9.14e-19 | 1.04e-17 / 3.55e-19 | 8.67e-18 / 2.74e-19 | 4724 |
| 4 | 5.12e-17 / 1.62e-18 | 2.34e-17 / 6.69e-19 | 1.56e-17 / 4.72e-19 | 6841 | 2.26e-17 / 8.26e-19 | 1.04e-17 / 3.28e-19 | 6.07e-18 / 2.41e-19 | 6663 |
| 5 | 5.38e-17 / 1.93e-18 | 2.34e-17 / 7.79e-19 | 1.30e-17 / 5.08e-19 | 8832 | 1.73e-17 / 7.86e-19 | 1.04e-17 / 3.40e-19 | 5.64e-18 / 2.56e-19 | 8556 |
| 6 | 4.68e-17 / 2.04e-18 | 2.17e-17 / 8.60e-19 | 1.99e-17 / 5.87e-19 | 10424 | 1.39e-17 / 7.52e-19 | 6.94e-18 / 3.04e-19 | 6.07e-18 / 2.38e-19 | 10165 |
| 7 | 5.03e-17 / 2.26e-18 | 1.99e-17 / 9.08e-19 | 1.30e-17 / 6.11e-19 | 11673 | 1.73e-17 / 7.62e-19 | 1.04e-17 / 3.17e-19 | 6.07e-18 / 2.36e-19 | 11220 |
| 8 | 5.03e-17 / 2.27e-18 | 1.95e-17 / 9.73e-19 | 1.21e-17 / 6.73e-19 | 12847 | 1.39e-17 / 7.50e-19 | 9.54e-18 / 2.99e-19 | 6.94e-18 / 2.47e-19 | 12012 |
| 9 | 5.29e-17 / 2.46e-18 | 2.26e-17 / 1.05e-18 | 1.39e-17 / 7.17e-19 | 14432 | 1.39e-17 / 7.17e-19 | 6.94e-18 / 3.04e-19 | 3.98e-18 / 2.27e-19 | 12829 |
| 10 | 4.34e-17 / 2.58e-18 | 2.17e-17 / 1.15e-18 | 1.82e-17 / 7.63e-19 | 15429 | 1.73e-17 / 7.56e-19 | 1.21e-17 / 3.32e-19 | 5.20e-18 / 2.30e-19 | 13000 |

GIVEN-NEMO-ENTRY to step 49 (new this round; histories from the card's chain
except where a restart exists): sea level at the bar through step 29 (at most
2.3e-16), 1.1e-15 at step 30, 2.9e-12 at step 41, 6.5e-11 at step 49.

Three-way split, GIVEN-NEMO-ENTRY, after: region max error over the region's
own NEMO maximum (cells not bit-equal).

| step | field | pivot row | row below | third row | seam | south | coast | open |
|---|---|---|---|---|---|---|---|---|
| 1 | ssh | no signal | no signal | no signal | no signal | no signal | 2.1e-15 (9) | 5.3e-16 (1061) |
| 5 | ssh | 3.5e-15 (2) | 4.2e-15 (4) | 3.5e-15 (6) | 1.8e-15 (136) | 2.8e-15 (27) | 2.3e-16 (808) | 6.5e-16 (7573) |
| 10 | ssh | 5.7e-16 (56) | 3.4e-16 (53) | 5.9e-16 (49) | 3.4e-16 (197) | 6.6e-16 (36) | 2.9e-16 (1569) | 6.4e-16 (11040) |
| 10 | u | 1.1e-15 (46) | 9.6e-16 (27) | 1.6e-15 (23) | 4.9e-16 (194) | 1.9e-15 (28) | 5.2e-16 (903) | 1.4e-15 (9356) |
| 17 | ssh | 6.4e-16 (60) | 4.4e-16 (57) | 6.0e-16 (45) | 4.4e-16 (230) | 7.0e-16 (34) | 3.9e-16 (1870) | 4.6e-16 (11954) |
| 17 | u | 1.0e-15 (36) | 1.2e-15 (37) | 7.7e-16 (42) | 6.5e-16 (241) | 7.3e-16 (37) | 5.4e-16 (1231) | 1.1e-15 (11091) |
| 17 | v | 1.0 (68) | 4.0e-16 (60) | 6.9e-16 (48) | 1.2e-15 (208) | 9.4e-16 (39) | 3.5e-16 (1318) | 7.4e-16 (10747) |
| 31 | ssh | 1.2e-10 (70) | 7.2e-11 (69) | 1.5e-11 (66) | 4.2e-16 (216) | 9.5e-16 (43) | 2.5e-15 (1857) | 2.9e-15 (12170) |
| 31 | u | 1.9e-10 (68) | 2.3e-10 (67) | 3.5e-11 (63) | 6.6e-16 (224) | 2.5e-15 (39) | 8.9e-16 (1325) | 3.7e-15 (11545) |
| 41 | ssh | 6.7e-08 (70) | 1.4e-08 (69) | 1.9e-09 (66) | 5.1e-16 (245) | 9.9e-16 (45) | 4.5e-12 (1782) | 5.2e-12 (11959) |
| 41 | u | 1.8e-07 (68) | 4.3e-08 (67) | 4.8e-09 (63) | 7.0e-16 (229) | 6.4e-16 (39) | 2.6e-12 (1312) | 3.7e-12 (11634) |

Seam and south wall: at rounding at every step shown. One step from NEMO's
restarts (steps 11 to 91): unchanged from round 3's after column to every
printed digit (whole domain 1.39e-17, 1.39e-17, 1.01e-15, 2.94e-12, 7.41e-11,
1.61e-10, 4.95e-10, 3.12e-09, 9.91e-09). Plants: score and entry exit 1,
unplanted exits 0.

## 4. The fold walk (item 2)

One step from NEMO's entry with NEMO's own barotropic histories (restarts at
steps 30 and 40), compiled order; error over the row's own NEMO maximum:

| step | arm | field | pivot row | row below | third row |
|---|---|---|---|---|---|
| 31 | right-hand side | u / v | 1.6e-16 / 1.0 (stored top V row) | 1.5e-16 / 1.9e-16 | 2.0e-16 / 1.5e-16 |
| 31 | **solver handoff** | barotropic u | **1.9e-10** | 2.3e-10 | 3.5e-11 |
| 31 | solver handoff | barotropic v | 2.1e-10 | 2.1e-10 | 7.3e-11 |
| 31 | solver handoff | sea level | 1.3e-10 | 5.3e-11 | 7.3e-12 |
| 31 | stages 1, 2, 3 | u | 2.2e-17, 4.4e-17, 8.7e-17 | 1.2e-16, 3.0e-17, 2.4e-16 | 6.5e-17, 6.5e-17, 2.6e-16 |
| 31 | stages 1, 2, 3 | sea level | 0 unequal | 0 unequal | 0 unequal |
| 41 | right-hand side | u / v | 1.1e-16 / 1.0 | 1.7e-16 / 9.6e-17 | 2.6e-16 / 2.9e-16 |
| 41 | **solver handoff** | barotropic u | **1.8e-07** | 4.3e-08 | 4.8e-09 |
| 41 | solver handoff | barotropic v | 6.2e-08 | 6.2e-08 | 9.6e-09 |
| 41 | solver handoff | sea level | 1.4e-08 | 6.6e-09 | 2.0e-09 |
| 41 | stages 1, 2, 3 | u | 4.6e-16, 4.6e-16, 3.4e-16 | 1.7e-16, 2.3e-16, 3.4e-16 | 2.1e-16, 1.1e-16, 8.5e-16 |

The first fold row beyond rounding is the solver's output; the statement is
inside the loop that starts at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:359`. The error grows 1,000
times while the amplitude grows 16 times (1.6e-05 m to 2.5e-04 m): still
quadratic, a sea-level-times-flow term. No statement is named and nothing
was landed for the fold. The held Coriolis-coefficient operand of round 3 was
not re-tested: without the loop's own frames it would be a guess.

Round 3's open instrument question (stage arm, pivot-row u, 0.09 and 0.20):
those two numbers are step 5 alone, where NEMO's row maximum is 2e-105 (first
arrival). At steps 6 to 10 the same rows read 1e-15 (step 10: 5.5e-16 and
1.1e-15), and at steps 31 and 41 they are at rounding (table). Not a stage
defect at any amplitude that matters; the step-5 reading is unexplained.

## 5. ORCA2 pointer

| statement | G1 result | ORCA2 |
|---|---|---|
| stage-1 sea level, two products and a sum | landed for G1; stage-1 row bit-identical | SHARED path, same expression on the ORCA2 card; UNMEASURED there. Default off, so ORCA2 is unchanged until its card selects it |
| tracer carry through the thickness quotient and the density | measured on G1 (advection off) | ORCA2 advects tracers and has real stratification: the one-cell flips are not expected to stand out; UNMEASURED |
| the solver's last-place row | open on G1 from step 1 | same solver path; whatever the substep record names applies there |
| the fold residual inside the solver | open | same; the round-3 pointer stands |

## 6. Predictions

| id | result |
|---|---|
| N-P0 new record equals round 1's byte for byte | HOLDS |
| I-P3 first row over the bar at the cell is a stage sea level | REFUTED: given NEMO's entry no row leaves the bar; with the card's T, S the first row is the right-hand side |
| I-P4 every operand bit-exact, order of a sum only | REFUTED as stated: NEMO's stage statements replay bit for bit, but the after-level sea level they read is not NEMO's in the last place |
| I-P5 same owner at steps 20 and 22 | HOLDS at the whole-step level (at the bar from NEMO's entry); stage walk not repeated |
| F-P5 the stage-arm pivot-row reading is a feed fault | REFUTED: a step-5-only reading at 2e-105 |
| F-P6 first fold row beyond rounding at steps 31 to 41 is the handoff | HOLDS |
| F-P7 a named fold operand takes step 31 below 1e-13 and step 100 down 100-fold | NOT REACHED: no statement named |

## 7. Corrections to my own work before quoting

1. The replay's first stage-3 tracer row used the stage-1 statement and
   disagreed with NEMO at 3,065 cells; stage 3 is the vertical-mixing solve.
   Replaced; now 0 unequal. Found by my own test.
2. I cited the stage-1 statement at line 147 (its comment line) in code and
   tests; it is 148, and the ratio lines are 171 to 173 and 215 to 217.
   Corrected in commit aa0f7ee8f; the first code commit's message still says
   147.
3. A first table read "the landing changes nothing"; an exact row comparison
   shows 79 rows move from step 78 on (section 3).

## 8. Acquisition written (operator runs it)

scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid/run_r4.sh:
the round-3 acquisition under NEW target names, plus a read-only record of
the solver's substeps at steps 1, 2, 17, 31 and 41. The writer is the
campaign's existing substep writer (VORTEX round 196) under G1 names, opened
only at those steps; the patch is that round's patch on NEMO's shipped solver
file with eight added writes of the Coriolis coefficients at loop entry (76
added lines, 0 removed; every added line is a writer call). The checker
admits the streams and refuses a missing, truncated, non-finite, stray,
schema-short or twin-divergent one. Preflight at aa0f7ee8f: patch applies,
PREFLIGHT_OK. About 1.7 GB. Not compiled by me (UNVERIFIED): the eight added
calls pass allocated module arrays of the inner-domain extent.

## 9. Choices

ASKED (round brief): the two walks in order; landing a cited statement
selected by the G1 card; the re-score; the receipt; one review.
UNASKED, offered for revert: (1) landing the stage-1 sea level, which is a
proven transcription difference but not the owner of either item; (2) the
substep steps 1, 2, 17, 31, 41 and the eight coefficient writes; (3) the new
target names; (4) the stage arms now take NEMO's histories where a restart
exists; (5) the per-cell report, the first-step option and the
chain-tracer measurement arm of the instrument; (6) renaming round 3's
interior-event test, whose name asserted the retracted reading.

## 10. Tests and review

Single review (codex): **findings**, four CONFIRMED, two acted on. Log:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round4/codex_review_r4.log.
- "the literal stage-1 SSH ... `_h_live_one_third` retains legacy
  interpolation ... can be internally inconsistent". True as read. Not
  acted on: that thickness is a different statement, not cited this round.
  Registered open.
- "tracer replay omits the cited ... trend term entirely". True; the trend
  is zero here and unrecorded. Stated as a limit (section 2), no code change.
- "when --twin is supplied but its substep file is missing, comparison is
  silently skipped". Fixed: refused when the twin ran that step; test added.
- "no positive-icycle, header-geometry, expected-field-schema ... checks".
  Fixed: substep count, a minimum field list per frame, and field extent;
  tests added.
"VERDICT: findings". The fixes were not re-reviewed. After the review the
landed statement was moved from a small helper to the same arithmetic inline
(the citation gate showed the helper's four lines had shifted 39 earlier
citations); that re-expression was not re-reviewed either. Its check is
the test against NEMO's record (0 unequal cells) and the re-run in section
11. No second reviewer: a
single review, not a dual one.

Focused tests, synchronous, final tree:
- The three G1 files: **107 passed** (24 ladder, 21 tools, 62 card), including each new option's plant: the stage-1 row has 0 unequal
  cells with the option and over 500 without; step 17 is at the bar with
  NEMO's T, S and over it at the cell with the card's.
- Six ratchets: 6153 passed, 2 skipped, 4 failed. None is in a file this
  round touched (git diff 43f4f070c is empty for each): the slab-land
  dispatch guard, the jra55 literal, the inline coefficients in the land
  restart file (the three recorded in rounds 1 to 3), and the shortwave
  parameter spec.
- NOT RUN: any closed-card ladder (GYRE, VORTEX, TSUNAMI, ORCA2) and the
  wider RK3 unit selection. The option is default-off and the legacy
  expression is untouched, but no closed card was re-scored.

Compliance, stated because no gate enforces it: search-before-build (section
3; the acquisition reuses the existing writer and patch); one variable per
arm (each pair differs in one option or one entry field); instruments
checked before quoting (the replay must reproduce NEMO with 0 unequal;
section 7); non-vacuity (option plants, score and entry plants, checker
refusals shown red). Dual review: NOT met.

UNVERIFIED:
- that the density itself changes by one last-place unit at the cell
  (inferred from the size of the pressure-gradient dipole);
- why the landing first moves the record at step 78;
- that the round-4 acquisition compiles;
- that the stage-2 single-cell tracer difference is the ratio's order (the
  replay's guess of the card's order predicts step 40, the stage arm shows
  steps 6 and 41).

## 11. Evidence hashes (SHA-256 prefixes, final/)

fold_kt31_given_entry.json=561fe14a851f2fd5, fold_kt31_handoff.json=c7e09578c9b70df0, fold_kt31_rhs.json=19ab38903ff81355, fold_kt31_stages.json=3de932ca036ce085, fold_kt41_given_entry.json=c2004f7c41c26640, fold_kt41_handoff.json=eb362b7a2ce84820, fold_kt41_rhs.json=736f5a498bc9a24c, fold_kt41_stages.json=dc84b87774990ebc, geometry.json=0ba8a9d2dc93b609, given_entry_49_after.json=79eb860136c0c4e6, kt17_given_entry_chainTS.json=66fbb202e811687b, kt17_given_entry.json=4190cccf05e16513, kt17_handoff_chainTS.json=75de5977399df375, kt17_handoff.json=789195bd715b6630, kt17_rhs_chainTS.json=795f4cef06ba3106, kt17_rhs.json=7a256139364d6bcf, kt17_stages_chainTS.json=15f63977eaccce96, kt17_stages.json=832b7b1d530697e2, record100_after.json=9d1e9cb1434d9333, record100_before.json=4a94b9172e9745e8, restart_entry_after.json=f7779c570ab53ffe, stage_replay.json=dd3793caab169bc3, stages_kt6_after.json=022031bffa7d9e3d, stages_kt6_before.json=e1e2b4e9c2d4dd01.
One level up: codex_review_r4.log=960e72d3c161cd37, preflight_r4.log=42c7c63018ea2a10, admission_r4.json=b07e282effabd67e.

## OPEN for round 5

1. On the substep record: the solver's first non-bit statement at step 1,
   then re-score step 17.
2. On the same record at steps 31 and 41: the fold statement inside the
   loop; then re-test the held Coriolis-coefficient operand.
3. The right-hand side's last-place row from step 2 (the vorticity trend on
   open faces).
4. The stage thickness that still interpolates the legacy way (review
   finding 1), and the stage u, v last-place rows.
5. The stored top V row of the 3-D velocity.

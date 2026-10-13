# Grid ladder rung G1 — round 3 receipt (gridladder_r3)

Status: **LANDED** (one cited statement, selected only by the G1 card). The
100-step record is NOT at the bar: a third fold statement and the interior
events from step 17 are open, and both need NEMO's stage frames beyond step
10 (acquisition written, section 9).
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round3.md
(commit d587063ff, before any measurement). Code commit 697f82b39; every
number in sections 2 to 5 was regenerated from that clean commit or the two
instrument-only commits after it (stamp in each json).
Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round3/final/.
Record: round 1's (.../gridladder_rounds/round1/oracle_g1_r1/p3_a_100step).
Baseline: the tables of the round-2 receipt; the before arms below reproduce
them (step 31: 1.74e-12, step 41: 2.35e-10, step 100: 1.39e-03).

## Take-aways

1. **The second fold statement is the solver's face depth on the fold-image V
   row.** NEMO adds the sea-level mean to the reference depth there; the card
   left that row at the reference depth, so the flow through the north face
   of the pivot row lost its sea-level part. The error is quadratic in
   amplitude, which is why steps 1 to 10 could not show it.
2. **Landed, card-selected.** Step 100 sea level: 1.39e-03 -> 5.30e-08
   (26,000 times smaller). One step from NEMO's state at step 31: 1.74e-12 ->
   1.01e-15.
3. **Not finished.** One step from NEMO's state still leaves 7e-11 of the fold
   band's maximum at step 31 and 2e-07 at step 81: a third fold statement.
4. **The step-17 interior departure is NOT coastal and NOT the fold**
   (prediction I-P1 REFUTED). It is made inside one step, at single cells (the
   first one 19 cells from any land), each about 8e-15 m. Its statement is not
   named: no stage frame exists at step 17.
5. A second fold operand (the same sea-level image in the solver's Coriolis
   coefficients) is NEMO's bit for bit but makes the result 1.7 to 2.4 times
   worse with landing 2 in place. HELD as a patch, not in the tree.

## 1. The walk, steps 1 to 10 (the only steps with stage frames)

Per fold row, each row against NEMO's OWN row maximum (the signal falls about
100-fold per row towards the pivot; round 2's band-wide maximum hid the pivot
row). GIVEN-NEMO-ENTRY, worst over steps 5 to 10, compiled order:

| arm | field | pivot row | row below | third row |
|---|---|---|---|---|
| right-hand side | u | 2.6e-16 | 2.5e-16 | 2.3e-16 |
| right-hand side | v | 1.0 (stored top V row) | 3.2e-16 | 4.8e-16 |
| external handoff | ssh | 3.5e-15 | 4.3e-15 | 3.5e-15 |
| external handoff | uu_b | 3.4e-15 | 4.3e-15 | 3.6e-15 |
| external handoff | vv_b | 4.0e-15 | 4.0e-15 | 4.1e-15 |
| stage 1 | ssh / u | 2.0e-16 / 0 | 2.0e-16 / 0 | 1.4e-16 / 0 |
| stage 2 | u | 9.1e-02 (see below) | 9.4e-15 | 6.5e-16 |
| stage 3 | u | 2.0e-01 (see below) | 1.9e-14 | 4.8e-15 |
| whole step | ssh / u at step 10 | 5.7e-16 / 1.1e-15 | 3.4e-16 / 9.6e-16 | 5.9e-16 / 1.6e-15 |

Steps 1 to 4: no fold cell differs (F-P0 HOLDS; NEMO's rows are zero there,
reported "no signal", not a pass).

The handoff is at rounding on every fold row: **F-P1 REFUTED**. Nothing in
steps 1 to 10 is beyond rounding on the fold except two instrument rows:

- v on the pivot row, 1.0 everywhere: the card stores zero on the top V row
  where NEMO stores the fold image. Known since round 2, still open.
- u on the pivot row in the stage-2 and stage-3 arms, 0.09 and 0.20. NOT a
  finding yet. The whole-step arm is exact on that row (1.1e-15), and NEMO's
  stage-3 frame equals its after-step frame there bit for bit (0 of 180
  unequal, both antisymmetric under the fold). So the two card arms disagree
  with each other; the stage arm feeds the card NEMO's solver output and
  stage entries on that row, and that feed is the suspect. Unreconciled
  (cause UNVERIFIED); not used for attribution. First item for round 4.

## 2. Naming the statement (measured where the amplitude allows)

One step from NEMO's restart state, fold-band sea-level error over the band's
maximum: 2e-13 at an amplitude of 1.8e-10 m (step 21), 1e-07 at 1.6e-05 m
(step 31), 9e-07 at 2.5e-04 m (step 41). The ratio to the amplitude stays
between 1e-3 and 3e-2 per metre over eight decades: quadratic, with a
coefficient near one over the 100 m depth. A sea-level-over-depth factor was
wrong by order one on a fold row. Three candidates, each tested:

| candidate | operand against NEMO (step 10) | effect on the one-step error |
|---|---|---|
| stage face ratio on the top V row | card 0.92 off NEMO's nonzero row | none: identical to all printed digits (the stage's top V row carries no flow) |
| the sea-level image in the solver's Coriolis coefficients | card zero, NEMO 5.8e-36; with the image 0 of 180 unequal | 1 percent; and worse once the next row landed (section 4) |
| **the solver's face depth on the fold-image V row** | card zero, NEMO 4.7e-46 (68 nonzero cells); with both operands 1.6e-16 | step 31: 1.74e-12 -> 1.01e-15 |

NEMO's statement: `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:419-421` adds, to the reference depth,
half the area-weighted sum of the sea level on rows jj and jj+1, times the
surface V mask; its loop includes the fold-image V row, where jj+1 is the T
fold halo (the image of the row below the pivot). `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:437`
multiplies it into the transport. The same form at `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:480`
feeds `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:558` and `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:583-584`. The mask is
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dommsk.f90:249`, wet on that row in NEMO's mesh file.

Two operands of that one statement differed in the card on that row: the mask
(the model's own V mask is dry there) and the north neighbour (the legacy
code read the pivot row's own mirror, not the row below). Offline operand
table (no model step; NEMO side = its stored weighted mean times its
reference depth), step 10:

| arm | fold-image V row | row below |
|---|---|---|
| NEMO's magnitude | 4.736e-46, 68 nonzero | 4.736e-46 |
| card before | 0.0 (relative error 1.0) | 1.6e-16 |
| card, NEMO's mask, legacy neighbour (probe) | 7.3e-47 (0.85) | 1.6e-16 |
| card after (both operands) | 1.6e-16 | 1.6e-16 |

Each operand alone is not enough; together they are one statement. The
1.6e-16 is the rounding of one multiply-divide pair in the NEMO-side
reconstruction, equal on the untouched row below.

## 3. Landed: before and after

New `BarotropicConfig.barotropic_fold_top_v_face_ssh`, default False, the
LAST field of its class. Only `build_orca2g1_flat_card` selects it; its
validator refuses the card without it. Searched first:
`grep -n "fold" barotropic_latlon_cgrid.py` (reused: the T fold halo source
and the existing private north-neighbour arm), `grep -rn
nemo_t_pivot_north_neighbor packages`. Caller grep afterwards
(`grep -rn barotropic_fold_top_v_face_ssh packages`): the config field, the
two helpers, and the G1 card (selection and validator). No other card.

**One step from NEMO's restart state** (GIVEN-NEMO-ENTRY; every prognostic
field and the six histories are NEMO's). Sea level, metres (relative to the
region's own maximum), before -> after:

| kt | fold band | pivot row | seam | south | coast | open | whole domain |
|---|---|---|---|---|---|---|---|
| 11 | 1.1e-49 (1e-15) -> same | 2.3e-51 (1e-15) -> same | 2.7e-48 (1e-15) | 5.5e-40 (1e-15) | 5.2e-18 (3e-16) | 1.4e-17 (5e-16) | 1.39e-17 -> 1.39e-17 |
| 21 | 2.8e-23 (2e-13) -> 1.0e-25 (6e-16) | 2.8e-23 (7e-13) -> 2.9e-26 (7e-16) | 3.2e-27 (5e-16) | 7.9e-23 (1e-15) | 1.4e-17 (6e-16) | 1.4e-17 (4e-16) | 1.39e-17 -> 1.39e-17 |
| 31 | 1.7e-12 (1e-07) -> 1.0e-15 (7e-11) | 1.7e-12 (2e-07) -> 1.0e-15 (1e-10) | 1.4e-20 (4e-16) | 2.0e-19 (9e-16) | 1.3e-14 -> 7.2e-17 | 3.1e-14 -> 9.4e-17 | 1.74e-12 -> 1.01e-15 |
| 41 | 2.4e-10 (9e-07) -> 2.9e-12 (1e-08) | 2.4e-10 (5e-06) -> 2.9e-12 (7e-08) | 1.4e-19 (5e-16) | 1.9e-19 (1e-15) | 3.2e-11 -> 1.6e-13 | 3.1e-11 -> 1.3e-13 | 2.35e-10 -> 2.94e-12 |
| 51 | 2.0e-08 (2e-05) -> 7.4e-11 (7e-08) | 2.0e-08 (2e-05) -> 7.4e-11 (9e-08) | 4.3e-19 (6e-16) | 4.1e-20 (4e-16) | 3.8e-10 -> 7.1e-12 | 4.3e-10 -> 1.2e-11 | 1.96e-08 -> 7.41e-11 |
| 61 | 7.2e-08 (4e-05) -> 1.6e-10 (9e-08) | 7.2e-08 (5e-05) -> 8.0e-11 (6e-08) | 4.3e-19 (3e-16) | 2.7e-19 (6e-16) | 9.0e-09 -> 2.7e-11 | 8.3e-09 -> 4.7e-11 | 7.18e-08 -> 1.61e-10 |
| 71 | 7.1e-07 (1e-04) -> 4.9e-10 (1e-07) | 7.1e-07 (2e-04) -> 3.9e-10 (9e-08) | 3.3e-19 (4e-16) | 3.3e-19 (4e-16) | 7.7e-08 -> 4.8e-11 | 7.1e-08 -> 9.0e-11 | 7.06e-07 -> 4.95e-10 |
| 81 | 8.7e-06 (5e-04) -> 3.1e-09 (2e-07) | 8.7e-06 (6e-04) -> 1.2e-09 (8e-08) | 3.3e-19 (5e-16) | 8.7e-19 (8e-16) | 6.1e-07 -> 2.5e-10 | 6.9e-07 -> 4.7e-10 | 8.66e-06 -> 3.12e-09 |
| 91 | 1.3e-05 (3e-04) -> 9.9e-09 (2e-07) | 1.3e-05 (7e-04) -> 9.9e-09 (5e-07) | 3.3e-19 (4e-16) | 2.2e-19 (7e-16) | 1.6e-07 -> 6.7e-10 | 4.0e-07 -> 1.0e-09 | 1.34e-05 -> 9.91e-09 |

Seam and south wall: identical in both arms, 1e-15 or better at all nine
restarts. The coast and open columns fall with the fold: the one-step
interior error was fold error spreading inside the step, not a second source.

**The 100-step record (INDEPENDENT)**, normalised error, before -> after:

| kt | ssh | u | v | uu_b | vv_b | region of the ssh maximum after |
|---|---|---|---|---|---|---|
| 10 | 4.34e-17 -> 4.34e-17 | 2.17e-17 -> 2.17e-17 | 1.82e-17 -> 1.82e-17 | 2.17e-17 -> 2.17e-17 | 1.82e-17 -> 1.82e-17 | open |
| 16 | 6.98e-17 -> 6.98e-17 | 2.43e-17 -> 2.43e-17 | 1.56e-17 -> 1.56e-17 | 2.36e-17 -> 2.36e-17 | 1.73e-17 -> 1.73e-17 | open |
| 17 | 5.75e-15 -> 5.75e-15 | 1.06e-16 -> 1.06e-16 | 2.48e-16 -> 2.48e-16 | 1.06e-16 -> 1.06e-16 | 2.48e-16 -> 2.48e-16 | open |
| 20 | 8.61e-15 -> 8.61e-15 | 2.94e-16 -> 2.94e-16 | 7.95e-13 -> 7.95e-13 | 2.94e-16 -> 2.94e-16 | 3.59e-16 -> 3.59e-16 | open |
| 25 | 4.71e-15 -> 4.71e-15 | 5.55e-16 -> 5.55e-16 | 5.34e-08 -> 5.34e-08 | 5.55e-16 -> 5.55e-16 | 6.31e-16 -> 6.31e-16 | open |
| 27 | 4.87e-15 -> 1.96e-15 | 2.46e-15 -> 3.10e-16 | 6.81e-08 -> 6.81e-08 | 2.46e-15 -> 3.10e-16 | 3.58e-15 -> 3.10e-16 | open |
| 30 | 4.27e-13 -> 8.04e-15 | 1.96e-13 -> 6.44e-16 | 5.35e-07 -> 5.35e-07 | 1.96e-13 -> 6.44e-16 | 2.58e-13 -> 5.96e-16 | open |
| 40 | 1.55e-09 -> 1.39e-12 | 1.40e-09 -> 3.95e-12 | 4.34e-05 -> 4.34e-05 | 1.40e-09 -> 3.95e-12 | 2.41e-09 -> 2.39e-12 | fold |
| 50 | 1.21e-07 -> 1.87e-10 | 1.16e-07 -> 4.11e-10 | 1.16e-04 -> 1.16e-04 | 1.16e-07 -> 4.11e-10 | 1.86e-07 -> 2.38e-10 | fold |
| 60 | 1.13e-06 -> 4.68e-10 | 1.12e-06 -> 1.09e-09 | 5.02e-04 -> 5.02e-04 | 1.12e-06 -> 1.09e-09 | 1.82e-06 -> 8.05e-10 | fold |
| 80 | 3.88e-05 -> 5.11e-09 | 3.74e-05 -> 1.30e-08 | 4.43e-03 -> 4.43e-03 | 3.74e-05 -> 1.30e-08 | 6.25e-05 -> 7.77e-09 | fold |
| 100 | 1.39e-03 -> 5.30e-08 | 1.65e-03 -> 8.29e-08 | 4.55e-03 -> 4.55e-03 | 1.65e-03 -> 8.29e-08 | 2.39e-03 -> 6.60e-08 | fold |

First step over the bar, before -> after: ssh 17 -> 17; u 27 -> 32;
uu_b 27 -> 32; vv_b 23 -> 23; v 19 -> 19. No row earlier. The v column is the
stored top V row (zero against NEMO's image), unchanged and open.

**Ladders, steps 1 to 10, card as landed** (max abs and RMS against NEMO; all
rows AT-BAR; identical to round 2's tables to every printed digit, because
the landed term is below rounding at these amplitudes):

INDEPENDENT

| kt | ssh max | ssh RMS | u max | u RMS | v max | v RMS | cells not bit-equal (ssh) |
|---|---|---|---|---|---|---|---|
| 1 | 4.16e-17 | 9.39e-19 | 6.94e-18 | 2.18e-19 | 8.67e-18 | 2.30e-19 | 1070 |
| 2 | 4.86e-17 | 1.29e-18 | 1.21e-17 | 3.74e-19 | 1.21e-17 | 3.27e-19 | 2711 |
| 3 | 3.99e-17 | 1.36e-18 | 1.73e-17 | 5.61e-19 | 1.21e-17 | 4.21e-19 | 4833 |
| 4 | 5.12e-17 | 1.62e-18 | 2.34e-17 | 6.69e-19 | 1.56e-17 | 4.72e-19 | 6841 |
| 5 | 5.38e-17 | 1.93e-18 | 2.34e-17 | 7.79e-19 | 1.30e-17 | 5.08e-19 | 8832 |
| 6 | 4.68e-17 | 2.04e-18 | 2.17e-17 | 8.60e-19 | 1.99e-17 | 5.87e-19 | 10424 |
| 7 | 5.03e-17 | 2.26e-18 | 1.99e-17 | 9.08e-19 | 1.30e-17 | 6.11e-19 | 11673 |
| 8 | 5.03e-17 | 2.27e-18 | 1.95e-17 | 9.73e-19 | 1.21e-17 | 6.73e-19 | 12847 |
| 9 | 5.29e-17 | 2.46e-18 | 2.26e-17 | 1.05e-18 | 1.39e-17 | 7.17e-19 | 14432 |
| 10 | 4.34e-17 | 2.58e-18 | 2.17e-17 | 1.15e-18 | 1.82e-17 | 7.63e-19 | 15429 |

GIVEN-NEMO-ENTRY

| kt | ssh max | ssh RMS | u max | u RMS | v max | v RMS | cells not bit-equal (ssh) |
|---|---|---|---|---|---|---|---|
| 1 | 4.16e-17 | 9.39e-19 | 6.94e-18 | 2.18e-19 | 8.67e-18 | 2.30e-19 | 1070 |
| 2 | 2.78e-17 | 9.11e-19 | 1.04e-17 | 3.13e-19 | 8.67e-18 | 2.63e-19 | 2654 |
| 3 | 2.43e-17 | 9.14e-19 | 1.04e-17 | 3.55e-19 | 8.67e-18 | 2.74e-19 | 4724 |
| 4 | 2.26e-17 | 8.26e-19 | 1.04e-17 | 3.28e-19 | 6.07e-18 | 2.41e-19 | 6663 |
| 5 | 1.73e-17 | 7.86e-19 | 1.04e-17 | 3.40e-19 | 5.64e-18 | 2.56e-19 | 8556 |
| 6 | 1.39e-17 | 7.52e-19 | 6.94e-18 | 3.04e-19 | 6.07e-18 | 2.38e-19 | 10165 |
| 7 | 1.73e-17 | 7.62e-19 | 1.04e-17 | 3.17e-19 | 6.07e-18 | 2.36e-19 | 11220 |
| 8 | 1.39e-17 | 7.50e-19 | 9.54e-18 | 2.99e-19 | 6.94e-18 | 2.47e-19 | 12012 |
| 9 | 1.39e-17 | 7.17e-19 | 6.94e-18 | 3.04e-19 | 3.98e-18 | 2.27e-19 | 12829 |
| 10 | 1.73e-17 | 7.56e-19 | 1.21e-17 | 3.32e-19 | 5.20e-18 | 2.30e-19 | 13000 |

Three-way split, GIVEN-NEMO-ENTRY, after: region max error over the region's
own NEMO maximum (cells not bit-equal).

| kt | field | pivot row | row below | third row | seam | south | coast | open |
|---|---|---|---|---|---|---|---|---|
| 1 | ssh | no signal (0) | no signal (0) | no signal (0) | no signal (0) | no signal (0) | 2.1e-15 (9) | 5.3e-16 (1061) |
| 1 | u | no signal (0) | no signal (0) | no signal (0) | no signal (0) | no signal (0) | 1.7e-15 (14) | 5.9e-16 (1028) |
| 4 | ssh | no signal (0) | no signal (0) | no signal (0) | 4.0e-15 (21) | no signal (0) | 3.5e-16 (696) | 6.2e-16 (5946) |
| 5 | ssh | 3.5e-15 (2) | 4.2e-15 (4) | 3.5e-15 (6) | 1.8e-15 (136) | 2.8e-15 (27) | 2.3e-16 (808) | 6.5e-16 (7573) |
| 5 | u | 0.0 (0) | 1.9e-14 (2) | 0.0 (0) | 1.8e-15 (98) | 4.5e-15 (21) | 1.9e-16 (471) | 1.2e-15 (6505) |
| 6 | ssh | 2.1e-15 (52) | 1.9e-15 (52) | 2.1e-15 (50) | 1.3e-15 (228) | 1.7e-15 (32) | 4.8e-16 (1013) | 5.2e-16 (8738) |
| 6 | u | 5.6e-15 (21) | 5.8e-15 (26) | 3.1e-15 (26) | 1.3e-15 (182) | 2.3e-15 (26) | 5.5e-16 (591) | 8.0e-16 (7557) |
| 8 | ssh | 9.7e-16 (70) | 7.9e-16 (68) | 8.7e-16 (66) | 4.8e-16 (262) | 2.5e-16 (29) | 2.9e-16 (1347) | 5.1e-16 (10170) |
| 8 | u | 1.5e-15 (46) | 1.4e-15 (44) | 1.3e-15 (28) | 4.2e-16 (223) | 1.5e-15 (21) | 5.2e-16 (739) | 1.1e-15 (8653) |
| 10 | ssh | 5.7e-16 (56) | 3.4e-16 (53) | 5.9e-16 (49) | 3.4e-16 (197) | 6.6e-16 (36) | 2.9e-16 (1569) | 6.4e-16 (11040) |
| 10 | u | 1.1e-15 (46) | 9.6e-16 (27) | 1.6e-15 (23) | 4.9e-16 (194) | 1.9e-15 (28) | 5.2e-16 (903) | 1.4e-15 (9356) |
| 10 | v | 1.0 (68) | 3.4e-16 (45) | 4.5e-16 (42) | 8.2e-16 (161) | 5.1e-16 (32) | 5.2e-16 (972) | 9.9e-16 (8618) |

Geometry: 36 of 36 rows bit-identical (unchanged). Plants: score and entry
plants exit 1, unplanted exits 0.

## 4. Held: the sea-level image in the solver's Coriolis coefficients

NEMO builds the V-face and F-point thickness ratios from the T row above
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:268-269`, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:281-285`), called at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:161` and exchanged at `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:163`; the
solver's Coriolis coefficients read them on the pivot row
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960-981`, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:987-1008`). The card's
coefficient builder takes the north neighbour of the pivot row from the SOUTH
row (a cyclic roll).

Operand proof. An offline replay of NEMO's V-face statement from NEMO's entry
sea level with its halos reproduces NEMO's recorded ratio on the fold-image
row bit for bit (0 of 68 unequal, steps 8 and 10; NEMO's row 1.8e-62 and
4.7e-48); with the halo row zeroed it is 0.97 and 0.92 off. The card's
operand on that row is zero before and bit-identical to NEMO's halo-row
product with the image (0 of 180 unequal; the area on the halo row is the
card's own image, the sea level is NEMO's).

Effect, one step from NEMO's state, whole-domain sea level:

| kt | card before | landing only | landing + this operand |
|---|---|---|---|
| 31 | 1.74e-12 | 1.01e-15 | 2.44e-15 |
| 41 | 2.35e-10 | 2.94e-12 | 5.02e-12 |

Faithful but worse: a signal that a compensating error is still in the
band (the third statement), not a reason to discard it. It is NOT in the
tree (a knob no card selects is not shipped); the exact diff is
scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_g1_round3_dyn_cor_fold_ssh_held.patch.
The numbers in this section come from an uncommitted tree holding both
changes (json files restart41_A_dirty, restart41_AB_dirty, stamped dirty):
PLAUSIBLE-grade until re-run from the patch.

## 5. The interior events from step 17 (item 2)

Census arm (INDEPENDENT chain, committed): sea-level cells over the 1e-15 bar.

| kt | cells over bar | largest cell (row, col) | error (m) | cells to land | rows to fold | same cell, one step from NEMO's previous frame |
|---|---|---|---|---|---|---|
| 1..16 | 0 | | max 7.0e-17 | | | |
| 17 | 2 | (72, 70) | -5.75e-15 | 19 | 75 | -5.74e-15 |
| 18 | 1 | (69, 70) | 1.08e-15 | 21 | 78 | 3.2e-17 (remnant of step 17) |
| 20 | 1 | (59, 59) | 8.61e-15 | 13 | 88 | 8.61e-15 |
| 21 | 7 | (59, 59) | -3.50e-15 | 13 | 88 | -2.0e-17 (remnant) |
| 22 | 21 | (62, 77) | 6.85e-15 | 22 | 85 | 6.87e-15 |
| 23 | 38 | (47, 52) | 8.05e-15 | 3 | 100 | 8.06e-15 |
| 24 | 53 | (57, 61) | 9.62e-15 | 13 | 90 | 8.95e-15 |
| 25 | 65 | (57, 61) | -4.71e-15 | 13 | 90 | -6.2e-17 (remnant) |
| 28 | 14 | (60, 76) | 8.16e-15 | 24 | 87 | 8.23e-15 |
| 29 | 31 | (88, 83) | 8.88e-15 | 5 | 59 | 8.81e-15 |
| 30 | 29 | (61, 71) | -8.04e-15 | 22 | 86 | -7.67e-15 |

All over-bar cells are in the open interior through step 24; the first
coast-adjacent one is at step 25 (2 of 65), the first fold-band one at step
30 (2 of 29). At step 17 the cell's entry error is 2.9e-17 and its eight
neighbours end at 6e-16 (north, south) and 9e-17: nothing arrives from
anywhere.

- **I-P1 REFUTED.** Not coastal: the first event is 19 cells from land, and
  the right-hand side, handoff and stage rows on coast-adjacent faces are at
  rounding in steps 1 to 10 (1e-16 to 4e-16 relative, the same as the open
  faces; none is bit-exact, so the falsifier's "every operand bit-exact"
  clause is NOT met either). Not the fold: 75 rows away, unchanged by the
  landing. The round-2 "coast columns non-bit from step 1" are the bump's own
  footprint touching coast-adjacent cells, at the same rounding level as the
  open cells.
- **I-P2 HOLDS.** One card step from NEMO's step-16 frame gives -5.74e-15 at
  the same cell. It is made inside step 17, not accumulated.
- It is not a sensitivity of the card: the card's two results (own entry,
  NEMO's entry) differ by 1.7e-17 at that cell.
- Fresh events are alike: eight of eight between 5.7e-15 and 9.0e-15. That
  is the size of one unit in the last place of a 50 m or 100 m number
  (7.1e-15, 1.4e-14). Hypothesis only (PLAUSIBLE): a one-unit flip of a
  depth-sized intermediate. Tested and ruled out (scratch probes, not
  committed evidence): neither side's sea level sits on the 100 m or 50 m
  rounding grid at the event cells; a last-place change of T or S at the
  cell changes no bit of the card's sea level; the card's T and S differ
  from NEMO's restarts at isolated cells by 1 to 3 units in the last place
  (2 and 1 cells at step 10; 13 and 6 at step 20; 97 and 34 at step 30).
- NOT registered as a rounding-accumulation row: without stage frames at
  step 17 no operand table exists, so "every operand bit-exact" cannot be
  shown. Cause unknown. The statement needs the stage record (section 9).

Instrument limits: T and S have no frame after step 10, so the one-step
re-run takes them from the card's chain; the six barotropic histories too.

## 6. ORCA2 pointer

| statement | G1 result | ORCA2 |
|---|---|---|
| solver face depth on the fold-image V row (wet mask + T fold halo) | NEW, landed for G1: step 100 error 26,000 times smaller | UNMEASURED on ORCA2. Its card runs the same solver path with the same dry top V row, and the north-neighbour half is the ORCA2 lane's own round-148 private arm, never selected: the same omission is PLAUSIBLE there. Round 2's "the held ORCA2 fold unit is inert on G1" was measured in steps 1 to 10 only, where this term is below rounding; that reading does not carry to large amplitude. |
| sea-level image in the solver's Coriolis coefficients | operand proven, HELD (worse with the landing) | same builder, same roll: PLAUSIBLE there; hold it there too until the third statement is named |
| stage face ratio on the top V row | 0.92 off NEMO's row, inert on G1 | UNMEASURED; could matter where the stage's top V row carries flow |
| seam, south wall | exact to rounding at all nine restarts, both arms | deliverable |

## 7. Predictions

| id | result |
|---|---|
| F-P0 no fold cell differs before step 5 | HOLDS |
| F-P1 first fold row beyond rounding is the external handoff | REFUTED: the handoff is at rounding on every fold row in steps 5 to 10 |
| F-P2 the statement is the substep Coriolis reading the image V transport | REFUTED: it is the solver's face depth on that row; the Coriolis-coefficient operand is real but minor and held |
| F-P3 stage rows at rounding except the stored top V row | NOT SCORED: the stage-2 and stage-3 pivot-row U rows read 0.09 and 0.20 but contradict the exact whole-step arm; unreconciled |
| F-P4 step-31 one-step fold error below 1e-13 of the band; step-100 error at least 100-fold smaller | HALF: 7e-11 (not reached: a third statement is open); 26,000-fold (reached) |
| I-P1 the step-17 cell traces to a coastal statement | REFUTED |
| I-P2 the excursion is made inside step 17 | HOLDS |

## 8. Corrections made to my own instruments before quoting

1. A first ratio probe subtracted 1 from "one plus ratio" and reported the
   ratio as zero; replaced by the ratio itself.
2. A test of the new land-distance helper expected 5 where the answer is 6;
   the expectation was wrong, the helper right (hand count in the test).
3. The first edit script for the landing aborted on a count check and wrote
   nothing; the run that followed reproduced round 2's numbers exactly, which
   served as the before control.

## 9. Acquisition written (operator runs it)

scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid/run_r3.sh:
the round-1 acquisition (same deck, domain inputs, patches) under NEW target
names, with the step writer's stage frames extended from step 10 to step 50
(one substitution, five places; the wrapper refuses if the patch is anything
else). Preflight at commit 88f21363c: patch applies, PREFLIGHT_OK. It gives
stage frames at steps 15 to 24 (the interior events) and 31 to 41 (the third
fold statement at an amplitude where it stands above rounding). No substep
record is added; if the third statement sits inside the solver loop a substep
writer is the next request. About 0.7 GB.

## 10. Choices

ASKED (round brief): the two items, in order; landing a cited statement
selected by the G1 card; the receipt; one review.
UNASKED, offered for revert: (1) the per-row fold regions added to the
instrument; (2) "beyond rounding" = 1e-12 of the row's own NEMO maximum;
(3) holding the Coriolis-coefficient operand as a patch instead of landing
it; (4) the acquisition's stage-frame range, steps 1 to 50, and its new
target names; (5) run.sh now reads its target names, stage count and patch
from the environment (defaults unchanged); (6) the pass-through for private
model hooks in the instrument (measurement arm only).

## 11. Tests and review

Single review (codex): **findings**, one defect CONFIRMED and acted on. Log:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round3/codex_review_r3b.log.
The first attempt (whole diff) ran into its own 540 s limit with no output
(codex_review_r3.log) and was abandoned; the second covered the packages diff
only, so the instrument, the tests and the acquisition wrapper were NOT
reviewed. Verbatim:
- "CONFIRMED — hidden default is unchanged."
- "CONFIRMED — no sign, row, permutation, or staggering error found."
- "CONFIRMED — no unintended whole-mask side effect relative to NEMO. The
  replacement also affects `ssh_avg_v`, `r1_v`, and `r1_v0`/`r3_v`, but NEMO
  uses the same `ssvmask` in its SSH average and reciprocal-depth expressions."
- "CONFIRMED — solver inconsistency. All five calls inside `_run_substep_loop`
  receive the new operands, but the final primary-transport-average conversion
  calls `_nemo_ssh_avg_apply` without `fold_top_v_wet`".
  Measured before acting: that call now receives the operand too, and 0 of
  238 scored rows change in maximum, RMS or count (restart entry to step 41,
  independent chain to step 40, against the files in final/). The path is
  not consumed on G1; the edit removes the inconsistency for a later card.
- "CONFIRMED — it is the last `BarotropicConfig` field."
"VERDICT: findings." The fix after the review was not re-reviewed. No second
(GLM) reviewer was available: a single review, not a dual one.

Evidence stamp: the json files carry commit 697f82b39 (interior_events:
99d891ba3). Later commits fold two card lines into their neighbours and pass
the operand at the sixth call site; neither moves a scored number (checked
as above for the second; the first changes no expression).

Focused tests, synchronous:
- The two G1 test files at the final tree: **35 passed** (17 ladder, 18
  tools), including the option's plants: the operand row with the option off
  reads zero against NEMO's nonzero row, and the one-step fold error at step
  21 is above 1e-13 without it and below 1e-14 with it.
- Six unit files nearest the edited solver helper (literal barotropic
  continuity, seed faces, prognostic barotropic state, fold row, tripole
  fold, zero-area face): **55 passed**.
- G1 files plus six ratchets, before the review fix: 6121 passed, 2 skipped,
  4 failed. Three are the reds recorded in rounds 1 and 2 (the jra55 literal,
  the slab-land dispatch guard, the inline coefficients in the land restart
  file; `git diff e579a1a2e -- packages/land tests/unit/test_jra55_do.py` is
  empty). The fourth was my own new test's wrong expectation, corrected.
- NOT RUN: the wider barotropic unit selection (it did not finish inside the
  500 s limit and was abandoned); any closed-card ladder (GYRE, VORTEX,
  TSUNAMI, ORCA2). The option is default-off and returns the same objects
  when off (tested), but no closed card was re-scored.

Citation gate on this receipt: PASS, 12 citations, 0 unmapped, whole-map
audit clean; the shifted-citation plant exits 1 (citation_receipt.json,
citation_plant.json under final/).

Compliance, stated because no gate enforces it: search-before-build (section
3); one variable per arm (each before / after pair differs in the one
option); instruments checked before quoting (sections 1 and 8; the stage-arm
pivot row is flagged, not used); non-vacuity (option plants, score and entry
plants, citation plant shown red). Dual review: NOT met (single, partial
scope).

UNVERIFIED:
- that the stage arm's pivot-row U disagreement is an instrument fault and
  not a stage defect compensated inside the whole step;
- the held operand's effect table (uncommitted tree);
- the three scratch probes of section 5 (not committed instruments);
- that the round-3 acquisition's rebuilt reference and domain file reproduce
  round 1's bit for bit (the operator's admission decides).

## 12. Evidence hashes (SHA-256 prefixes, final/)

fold_v_depth_after.json=7f398f9679eb7ba9, fold_v_depth_before.json=138e367edc4e5c8e, geometry.json=db4d6b37a91658c7, given_entry_after.json=663c848f5c97e1a3, handoff_after.json=f4b0acf2b3359ab0, interior_events.json=56ef83bd5e289b54, record100_after.json=1cb3720df4091248, record100_before.json=b2d901f53e9c38ca, restart_entry_after.json=34413215a016424b, restart_entry_before.json=af740f381b7be77b, rhs_after.json=e4356fe76987e82c, stages_after.json=2efa0efdf548a230.
Scratch (one level up, not committed evidence): probe_kt17_tracers.json=38cebaae40f35d95, probe_kt17_fingerprint.json=52784972646d0653.

## OPEN for round 4

1. Reconcile the stage arm with the whole-step arm on the pivot U row
   (section 1) before anything else: it is either an instrument fault or the
   third fold statement.
2. The third fold statement, on the new record's stage frames at steps 31 to
   41; then re-test the held Coriolis-coefficient operand.
3. The interior events, on the new record's stage frames at steps 17, 20, 22.
4. The stored top V row of the 3-D velocity (zero against NEMO's image).
5. ORCA2 lane: measure the fold-row face depth at large amplitude.

# PREREGISTRATION — grid ladder rung G1, round 2 (gridladder_r2)

Committed BEFORE any comparison of the card against NEMO's G1 record. Done
before this freeze, and nothing else: the operator's record was re-admitted
(check_records.py, json byte-identical to the operator's, five plants fired);
the card was built on the record's deck and its array shapes printed; NEMO's
own log was read (substeps 13, Courant number 0.78121620771566980, bump centre
(51,74), radius 9.9415838254534226 degrees — round-1 prediction P2 HOLDS).
No card array has been compared with any NEMO array and no step has been run.

Branch tip acdb00b92. Record:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round1/oracle_g1_r1
(domain file SHA-256 73801c6f...abdd; 100 step files, stage frames at kt 1..10).

## 0. State of the branch that the predictions depend on

The TSUNAMI branch (PR #1920) is NOT merged here (searched
`git log --all | grep -i tsunami`, `grep -rn build_tsunami_zco_card packages`:
only the two advection-OFF arms are present). So the one-level STEP has never
run on this branch, and round 1's P6 ("exact given the merge") has lost its
premise. The predictions below are for the branch as it is.

## 1. Geometry identity (round-1 P3, restated)

Round 1 froze "19 of 19". Its own committed list has 17 names (nine metric and
Coriolis operands, six face masks and thicknesses, tmask, e3t_0), and its test
asserts 19, so that test cannot pass as written. That is a miscount in the
round-1 text and test, corrected here before the test has ever run.

Round 2 compares EVERY mesh array the card carries, 36 rows, each against the
named variable of NEMO's own mesh file written by the record run:

| group | rows | NEMO variable |
|---|---|---|
| raw operands e1t e2t e1u e2u e1v e2v e1f e2f | 8 | same names |
| raw operand ff_f; grid ff_f; grid f_T | 3 | ff_f, ff_f, ff_t |
| grid dx_T dy_T; dx_u dy_u (west record dropped); dx_v dy_v (south record dropped) | 6 | e1t e2t e1u e2u e1v e2v |
| coordinate nemo_e2u, nemo_e1v | 2 | e2u, e1v |
| tmask (is_active), raw umask vmask fmask | 4 | same names, level 1 |
| coordinate e3t_0; raw e3u_0 e3v_0 e3f_0; drag e3u_0 e3v_0 | 6 | e3t_0 e3u_0 e3v_0 e3f_0, level 1 |
| coordinate e3w_0, gdept_0, gdepw_0 | 3 | e3w_1d gdept_1d gdepw_1d, level 1 |
| bottom_level + 1 | 1 | mbathy |
| h_partial | 1 | e3t_0 x tmask, level 1 |
| bump inputs glamt, gphit (read from the domain file) | 2 | glamt, gphit |

**G-P1: 36 of 36 bit-identical.** Falsifier: any unequal row; the first one is
the round's first owner, ahead of any ladder number. Rows most at risk, named
in advance: fmask (no-slip value 2 on the coast and the fold-owned row), umask
and vmask on the top row, grid dx_u / dx_v on the fold and seam. A planted
element must turn exactly its row red and exit nonzero.

Derived carriers (hu_0, hv_0, hf_0, fe3mask, the e1e2 products) have no
variable in NEMO's mesh file; they are NOT counted and stay unmeasured here.

## 2. Regions (frozen; 0-based on the 148 x 180 array the card carries)

NEMO's record is 152 x 184 with two halo points on each side; the card
carries no halo, so halo rows and columns are not scored.

| region | cells |
|---|---|
| fold band | rows 145..147 (NEMO jj 148..150: the top three rows, the pivot row last) |
| seam | columns 0, 1, 178, 179, outside the fold band |
| south | rows 0..2, outside the seam |
| interior | everything else |

The brief words the fold band as rows jpj-3..jpj (149..152); two of those are
halo rows the card does not carry, so the scored band is the top three carried
rows, as round 1's section 7 froze it. Interior is further reported as
coast-adjacent (a wet T cell with a land T cell among its eight neighbours, or
that land cell) and open, as an attribution aid only.

## 3. Ladders, kt = 1..10, two labels never mixed (D52)

INDEPENDENT: the card's step chained from the card's own initial state.
GIVEN-NEMO-ENTRY: the card handed NEMO's recorded entry state of each step
(sea level, both barotropic velocities, both level-1 velocities, T, S); every
other carried field is the card's initial one. The redundant west and south
face records are seeded as the card's own step leaves them (read off the
independent chain and stated in the receipt). Rows: ssh, u, v, uu_b, vv_b
against NEMO's after-step frame; per region max abs, RMS, count of bit-unequal
cells; normalised max = max abs / max(max abs NEMO, 1), bar 1e-15 (the shared
trajectory gate's scorer).

Domain of dependence, by counting (13 substeps per step, one cell per
substep): the signal covers about 14 cells per step from the bump (rows
61..85, columns 46..54). It reaches the seam columns at about kt 4, the south
rows at about kt 5 and the fold band at about kt 5, at denormal-to-tiny
amplitude. Bit-equality does not depend on amplitude, so the per-region
unequal-cell COUNT is the discriminator inside kt = 1..10.

| id | prediction | falsifier / reading |
|---|---|---|
| L-P0 | The card steps one wet level without raising. | An exception: the owner is the one-level step (TSUNAMI blocker B5), nothing about the grid. STOP and report. |
| L-P1 | kt = 1, both labels: fold, seam and south counts are 0 for ssh, u, v (the signal has not arrived; both sides hold exact zeros or rest values there). | A nonzero count there at kt = 1 = the card or NEMO writes something at a boundary with no signal: a boundary statement, named by region. |
| L-P2 | kt = 1 interior, given-NEMO-entry: at the bar (normalised <= 1e-15), not necessarily bit-equal (TSUNAMI's substeps sat at 1.0e-15..1.7e-15 with a different filter). | Over the bar in the open interior = a PROGRAM statement (filter nn_bt_flt = 3 on one level, EOS-80 pressure gradient, or the one-level stage), not a grid statement. |
| L-P3 | HEADLINE. At the first kt where any boundary region has a nonzero ssh count under GIVEN-NEMO-ENTRY, compare the three: fold-only nonzero => fold statement; seam-only => cyclic exchange; south-only => wall. Predicted: all three stay at the bar through kt = 10 (round-1 P4), with the coast-adjacent cells the most likely place to leave it first (the coast and no-slip are new versus every certified one-level run). | A fold-band row over the bar while seam and south rows of the same kt are at the bar refutes P4 and names a fold statement. |
| L-P4 | INDEPENDENT rows equal the GIVEN-NEMO-ENTRY rows at kt = 1 exactly (same entry state) and grow no faster than the entry error is carried afterwards. | Unequal at kt = 1 = the harness seeds something the card's initial state does not hold; an instrument bug, fixed before any number is quoted. |

## 4. The 100-step record (INDEPENDENT only)

Per-step normalised error of ssh, u, v, uu_b, vv_b against the after-step
frame, bar 1e-15, with the same four regions.

**R-P1:** no prediction of "at the bar through kt = 100" is made: the wave
meets the fold at about kt 20, the Antarctic coast at about kt 23 and the seam
at about kt 33 at FINITE amplitude, and none of the three has been measured.
Reported: the first kt over the bar per field, and the region holding the
maximum at that kt. Reading: if the first departure coincides with a boundary
arrival and sits in that region, the boundary is the PLAUSIBLE owner
(location-only evidence, as TSUNAMI round 4 labelled its j-seam).

## 5. First non-bit statement

Walked in NEMO's order with the existing private hooks (searched
`_NEMOWSRK3TestHooks` on this branch: the right-hand-side observer, the stage
exposure and the stage-entry and external-output overrides exist; no substep
record was acquired for G1, so the substep loop is one unit here): stp_2D
right-hand side -> external handoff at stage 1 -> stages 1, 2, 3. The first
unequal row in that order, at the first kt that has one, is named with its
compiled-source citation in the ORCA2G1_OMIP_L1_P3 build.

If it is a fold statement already landed on this branch, the receipt says
which and why G1 still differs. If it is new, ONLY a cited statement lands, as
an addition the G1 card selects, with its plant shown to fail. A choice of
scheme or default is asked, not made.

## 6. Instrument controls (run before any number is quoted)

- score plant: one element of the candidate moved => that row leaves the bar,
  exit nonzero;
- entry plant: one element of the seeded entry moved => the kt = 1 row moves;
- region masks partition the array exactly once (asserted);
- frame identity: NEMO's after-step frame of kt equals its entry frame of
  kt + 1 for kt = 1..9, bit for bit (else the reference is mislabelled);
- geometry plant as in section 1.

## 7. Choices in this round

ASKED (the round brief): admission, geometry identity, both ladders, the band
split, the 100-step record, the first non-bit statement.
UNASKED, offered for revert: (1) the scored fold band is the top three carried
rows; (2) seam = two columns each side; south = three rows; (3) the
coast-adjacent / open split of the interior; (4) one harness script for G1
rather than extending TSUNAMI's, which is not on this branch.

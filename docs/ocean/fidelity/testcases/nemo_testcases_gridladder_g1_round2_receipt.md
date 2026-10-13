# Grid ladder rung G1 — round 2 receipt (gridladder_r2)

Status: **LANDED** (one cited statement, selected only by the G1 card; the
100-step record is NOT yet at the bar — a second fold statement is open).
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round2.md
(commit 93ea4d2df, before any comparison). Code commit 6613a0982; every number
below was regenerated from that clean commit (stamp in each json).
Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round2/final/.
Record: .../gridladder_rounds/round1/oracle_g1_r1 (domain file SHA-256
73801c6ff5c78183b85b5e654c0a8c152355007d85cea1ef3ae8e6b151deabdd).

## Take-aways

1. **The seam and the south wall are exact to rounding; the fold is not.**
   One step from NEMO's restart state at kt 31: fold-band sea level off by
   8.9e-8 m (6e-3 of the band's maximum); seam 4e-16, south wall 9e-16.
2. **First non-rounding statement: NEMO's EEN Coriolis trend on the pivot row.**
   It reads the V transport on the fold-image row above the pivot T row; the
   card's mask kept that row dry. Proved by an offline replay (section 5).
3. **Landed, card-selected.** With the image row supplied, the same one-step
   error at kt 31 is 1.7e-12 m (51,000 times smaller) and the card's pivot-row
   right-hand side matches NEMO to 1.6e-16.
4. **Not finished.** After the landing the fold error still grows (1.4e-3 at
   kt 100). Sea level also leaves the 1e-15 bar at kt 17 in the open interior,
   unrelated to the fold and unexplained.
5. The ORCA2 lane's held private fold unit changes no bit on G1 (probe, not
   committed evidence): this statement is not that unit.

## 1. Admission

`check_records.py` on the operator's record: PASS_G1_RECORD, 100 steps, stage
frames at kt 1..10, twin identical, restarts at kt 10 and 100 identical to the
reference build; json byte-identical to the operator's
(g1_round2_admission.json, SHA-256 123cc1eb...67dd). Five plants refuse.
NEMO's own log: 13 substeps, Courant number 0.78121620771566980 — round-1
prediction P2 HOLDS. No ocean.output carries an error.

## 2. Geometry identity

**36 of 36 rows bit-identical** to NEMO's mesh file. Plant: moving one element
of the card's fmask turns exactly that row red, exit 1.

G-P1 as frozen was 35/36 on first run: the row "bottom_level + 1 vs mbathy"
differed on all 10,207 land cells. The expectation was wrong, not the card:
NEMO writes its clamped index, 1 on land, by
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domzgr.f90:575` and `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domwri.f90:164-165`. The row now compares
max(bottom_level + 1, 1). Counted honestly: G-P1 REFUTED as worded, HOLDS
after the row was corrected to what NEMO writes.

Round 1's "19 of 19" was a miscount of a 17-name list; its test could never
pass and is replaced by this gate. Not compared (no variable in NEMO's file):
hu_0, hv_0, hf_0, fe3mask, the area products.

## 3. Ladders, kt = 1..10 (card as landed; max abs and RMS against NEMO)

Bar: max abs / max(max abs NEMO, 1) <= 1e-15 (shared scorer). Every row below
is AT-BAR. None is bit-identical. T and S are bit-identical through kt 6, then
differ at 1 or 2 cells by one unit in the last place (7.1e-15 on 30); they are
not uniform (T reaches 20.030: the one-level thickness moves).

**INDEPENDENT** (card's own chain from its own initial state):

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

**GIVEN-NEMO-ENTRY** (NEMO's recorded ssh, both barotropic velocities, u, v,
T, S at each step entry; the six barotropic histories, which the step record
does not hold, from the card's own chain — see section 7):

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

Before the landing the two tables are the same to the printed digits (the
fold holds 1e-41 or less at kt 10); the before arm is in the json files.

### The discriminator: fold against seam against south wall

GIVEN-NEMO-ENTRY, region max error divided by the region's own NEMO maximum,
with the count of cells not bit-equal. Seam, south, coast and open columns are
the after arm (the before arm is the same to rounding there).

| kt | field | fold before | fold after | seam | south | coast | open |
|---|---|---|---|---|---|---|---|
| 1 | ssh | 0.0e+00 (0) | 0.0e+00 (0) | 0.0e+00 (0) | 0.0e+00 (0) | 2.1e-15 (9) | 5.3e-16 (1061) |
| 1 | u | 0.0e+00 (0) | 0.0e+00 (0) | 0.0e+00 (0) | 0.0e+00 (0) | 1.7e-15 (14) | 5.9e-16 (1028) |
| 1 | v | 0.0e+00 (0) | 0.0e+00 (0) | 0.0e+00 (0) | 0.0e+00 (0) | 5.2e-15 (9) | 7.5e-16 (1071) |
| 4 | ssh | 0.0e+00 (0) | 0.0e+00 (0) | 4.0e-15 (21) | 0.0e+00 (0) | 3.5e-16 (696) | 6.2e-16 (5946) |
| 4 | u | 0.0e+00 (0) | 0.0e+00 (0) | 3.6e-15 (25) | 0.0e+00 (0) | 4.2e-16 (382) | 1.1e-15 (5079) |
| 4 | v | 0.0e+00 (0) | 0.0e+00 (0) | 1.7e-16 (11) | 0.0e+00 (0) | 7.4e-16 (430) | 8.1e-16 (4664) |
| 5 | ssh | 3.5e-15 (12) | 3.5e-15 (12) | 1.8e-15 (136) | 2.8e-15 (27) | 2.3e-16 (808) | 6.5e-16 (7573) |
| 5 | u | 3.8e-17 (5) | 3.8e-17 (2) | 1.8e-15 (98) | 4.5e-15 (21) | 1.9e-16 (471) | 1.2e-15 (6505) |
| 5 | v | 2.1e-03 (17) | 2.1e-03 (17) | 3.7e-15 (98) | 2.7e-15 (25) | 3.0e-16 (494) | 8.7e-16 (5786) |
| 6 | ssh | 2.1e-15 (154) | 2.1e-15 (154) | 1.3e-15 (228) | 1.7e-15 (32) | 4.8e-16 (1013) | 5.2e-16 (8738) |
| 6 | u | 3.1e-15 (87) | 3.1e-15 (73) | 1.3e-15 (182) | 2.3e-15 (26) | 5.5e-16 (591) | 8.0e-16 (7557) |
| 6 | v | 1.8e-02 (127) | 1.8e-02 (127) | 1.3e-15 (174) | 1.3e-15 (32) | 9.0e-16 (601) | 8.2e-16 (6719) |
| 7 | ssh | 9.4e-16 (204) | 9.4e-16 (205) | 7.4e-16 (273) | 1.1e-15 (31) | 1.5e-16 (1192) | 6.5e-16 (9519) |
| 7 | u | 6.5e-15 (194) | 2.2e-15 (87) | 8.1e-16 (224) | 2.3e-15 (25) | 4.6e-16 (651) | 1.2e-15 (8180) |
| 7 | v | 3.8e-02 (198) | 3.8e-02 (174) | 1.7e-15 (194) | 4.8e-16 (30) | 4.4e-16 (677) | 8.2e-16 (7217) |
| 8 | ssh | 5.2e-13 (205) | 8.7e-16 (204) | 4.8e-16 (262) | 2.5e-16 (29) | 2.9e-16 (1347) | 5.1e-16 (10170) |
| 8 | u | 1.2e-11 (198) | 1.3e-15 (118) | 4.2e-16 (223) | 1.5e-15 (21) | 5.2e-16 (739) | 1.1e-15 (8653) |
| 8 | v | 6.0e-02 (200) | 6.0e-02 (182) | 5.6e-16 (187) | 1.4e-16 (26) | 3.6e-16 (782) | 9.7e-16 (7731) |
| 9 | ssh | 4.5e-11 (205) | 3.6e-16 (179) | 2.8e-16 (248) | 4.6e-16 (28) | 2.7e-16 (1470) | 5.1e-16 (10904) |
| 9 | u | 1.0e-09 (198) | 1.3e-15 (91) | 4.6e-16 (203) | 1.8e-15 (23) | 6.3e-16 (804) | 8.1e-16 (9169) |
| 9 | v | 8.5e-02 (200) | 8.5e-02 (173) | 1.3e-15 (169) | 5.1e-16 (27) | 8.8e-16 (920) | 6.3e-16 (8227) |
| 10 | ssh | 1.0e-09 (205) | 5.9e-16 (158) | 3.4e-16 (197) | 6.6e-16 (36) | 2.9e-16 (1569) | 6.4e-16 (11040) |
| 10 | u | 2.3e-08 (198) | 1.6e-15 (96) | 4.9e-16 (194) | 1.9e-15 (28) | 5.2e-16 (903) | 1.4e-15 (9356) |
| 10 | v | 1.1e-01 (200) | 1.1e-01 (155) | 8.2e-16 (161) | 5.1e-16 (32) | 5.2e-16 (972) | 9.9e-16 (8618) |

Reading. kt = 1: the signal has not reached any boundary; fold, seam and south
counts are 0 (L-P1 HOLDS). First arrival: seam kt 4, fold and south kt 5 (the
count predicted kt 4, 5, 5). Seam and south never exceed 4.5e-15. Before the
landing the fold's sea level leaves rounding at kt 8 (5.2e-13) and reaches
1.0e-9 at kt 10; after, it stays at 9e-16 or less. The fold's v column is the
top V row of the carried 3-D velocity, which the card leaves at zero where
NEMO stores the fold image: a gap in what the state stores, shown not to feed
the dynamics (section 6), and still open.

By the 1e-15 bar itself nothing leaves the bar inside kt 1..10 in either arm,
so L-P3 HOLDS as worded; the regional measure is what names the fold. Round-1
P4 (fold statements exact on G1) is REFUTED.

### One step from NEMO's restart state

The restart files hold every prognostic field and the six histories, and equal
the step frames bit for bit, so no evolving field comes from the card's chain.
NEMO's after-ssh slot is not seeded; this card does not read it (section 11).
Sea level, metres (and relative to the region's maximum):

| kt | fold before | fold after | seam after | south after | coast after | open after | whole-domain normalised before -> after |
|---|---|---|---|---|---|---|---|
| 11 | 9.8e-43 (1e-08) | 1.1e-49 (1e-15) | 2.7e-48 (1e-15) | 5.5e-40 (1e-15) | 5.2e-18 (3e-16) | 1.4e-17 (5e-16) | 1.39e-17 -> 1.39e-17 |
| 21 | 1.3e-13 (8e-04) | 2.8e-23 (2e-13) | 3.2e-27 (5e-16) | 7.9e-23 (1e-15) | 1.4e-17 (6e-16) | 1.4e-17 (4e-16) | 1.29e-13 -> 1.39e-17 |
| 31 | 8.9e-08 (6e-03) | 1.7e-12 (1e-07) | 1.4e-20 (4e-16) | 2.0e-19 (9e-16) | 1.3e-14 (5e-13) | 3.1e-14 (9e-13) | 8.85e-08 -> 1.74e-12 |
| 41 | 1.1e-05 (4e-02) | 2.4e-10 (9e-07) | 1.4e-19 (5e-16) | 1.9e-19 (1e-15) | 3.2e-11 (9e-10) | 3.1e-11 (1e-09) | 1.07e-05 -> 2.35e-10 |
| 51 | 1.9e-05 (2e-02) | 2.0e-08 (2e-05) | 4.3e-19 (6e-16) | 4.1e-20 (4e-16) | 3.8e-10 (8e-09) | 4.3e-10 (2e-08) | 1.94e-05 -> 1.96e-08 |
| 61 | 1.4e-04 (8e-02) | 7.2e-08 (4e-05) | 4.3e-19 (3e-16) | 2.7e-19 (6e-16) | 9.0e-09 (2e-07) | 8.3e-09 (3e-07) | 1.40e-04 -> 7.18e-08 |
| 71 | 4.1e-04 (8e-02) | 7.1e-07 (1e-04) | 3.3e-19 (4e-16) | 3.3e-19 (4e-16) | 7.7e-08 (1e-06) | 7.1e-08 (2e-06) | 4.13e-04 -> 7.06e-07 |
| 81 | 1.1e-03 (6e-02) | 8.7e-06 (5e-04) | 3.3e-19 (5e-16) | 8.7e-19 (8e-16) | 6.1e-07 (1e-05) | 6.9e-07 (2e-05) | 1.09e-03 -> 8.66e-06 |
| 91 | 4.7e-04 (1e-02) | 1.3e-05 (3e-04) | 3.3e-19 (4e-16) | 2.2e-19 (7e-16) | 1.6e-07 (3e-06) | 4.0e-07 (1e-05) | 4.70e-04 -> 1.34e-05 |

Seam and south wall: 1e-15 or better at all nine restarts. This is the
round's headline number.

## 4. The 100-step record (INDEPENDENT), normalised error against the 1e-15 bar

| kt | ssh before | ssh after | u before | u after | v before | v after | region of the ssh maximum after |
|---|---|---|---|---|---|---|---|
| 10 | 4.34e-17 | 4.34e-17 | 2.17e-17 | 2.17e-17 | 1.82e-17 | 1.82e-17 | interior_open |
| 16 | 6.98e-17 | 6.98e-17 | 2.43e-17 | 2.43e-17 | 1.56e-17 | 1.56e-17 | interior_open |
| 17 | 5.75e-15 | 5.75e-15 | 1.06e-16 | 1.06e-16 | 2.48e-16 | 2.48e-16 | interior_open |
| 18 | 1.08e-15 | 1.08e-15 | 1.35e-16 | 1.35e-16 | 3.42e-16 | 3.42e-16 | interior_open |
| 19 | 4.55e-16 | 4.55e-16 | 1.04e-16 | 1.04e-16 | 1.90e-14 | 1.90e-14 | interior_open |
| 20 | 8.61e-15 | 8.61e-15 | 5.14e-15 | 2.94e-16 | 7.95e-13 | 7.95e-13 | interior_open |
| 25 | 2.94e-09 | 4.71e-15 | 3.51e-09 | 5.55e-16 | 5.34e-08 | 5.34e-08 | interior_open |
| 27 | 1.97e-08 | 4.87e-15 | 3.07e-08 | 2.46e-15 | 6.81e-08 | 6.81e-08 | fold |
| 30 | 1.93e-07 | 4.27e-13 | 2.66e-07 | 1.96e-13 | 5.35e-07 | 5.35e-07 | fold |
| 40 | 8.01e-06 | 1.55e-09 | 5.95e-06 | 1.40e-09 | 4.34e-05 | 4.34e-05 | fold |
| 50 | 1.05e-04 | 1.21e-07 | 7.99e-05 | 1.16e-07 | 1.16e-04 | 1.16e-04 | fold |
| 60 | 2.08e-04 | 1.13e-06 | 4.76e-04 | 1.12e-06 | 5.02e-04 | 5.02e-04 | fold |
| 80 | 2.33e-03 | 3.88e-05 | 5.20e-03 | 3.74e-05 | 4.43e-03 | 4.43e-03 | fold |
| 100 | 7.93e-03 | 1.39e-03 | 1.59e-02 | 1.65e-03 | 4.80e-03 | 4.55e-03 | fold |

First step over the bar, before -> after: ssh 17 -> 17; u 20 -> 27;
uu_b 20 -> 27; vv_b 20 -> 23; v 19 -> 19. All five rows are at the bar for
kt 1..16 in both arms.

- kt 17, sea level 5.75e-15: single cells in the open interior (row 72,
  column 70; then (69,70), (59,59)), far from every boundary, identical in
  both arms. NOT the fold. Cause unknown. 2e-13 of the local sea level.
- v at kt 19: the zero top V row of the carried velocity (section 3).
- From kt 27 the maximum sits in the fold band and grows: the open statement.
- Seam and south stay at 1e-14 relative until the contaminated wave arrives
  (seam 5e-12 at kt 60).
- The card's six carried histories against NEMO's restart files follow the
  same curve (kt 10: 4.7e-17; kt 30: 1.8e-7 before, 4.3e-13 after).

R-P1 made no at-the-bar prediction; nothing to score.

## 5. The first non-bit statements, in NEMO's order

**Rounding level, kt = 1 (at the bar).** The stp_2D right-hand side is
bit-identical on every wet face (0 unequal). The first unequal rows are the
external-mode handoff: 1103 cells of uu_b, 5.9e-16 relative. So the first
non-bit arithmetic is inside the split-explicit substep loop, as TSUNAMI found
for its own loop. No substep record exists for G1, so it is not split further.
Stage arithmetic given NEMO's handoff: 2 to 78 cells at 1e-16.

The 5.1e-5 difference on 1160 U and 1049 V LAND faces of the right-hand side
is inert (those faces are masked afterwards); which side holds the unmasked
value was not determined.

**Beyond rounding: the pivot row.** Given NEMO's entry at kt 10 the recorded
right-hand side is the Coriolis trend alone. It matches to 3e-16 on every row
up to the one below the pivot. On the pivot T row all 68 wet U faces are wrong
by 0.92 relative.

NEMO's statement: the trend `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:779-780` multiplies the two
northern triads by the V transport of the row above, built at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:767` from the velocity, thickness and mask on that row. On the
pivot row that is the fold-image V row. The other operands are
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:720`, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:727`, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:772-775`, with the
thickness fallback `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:919`.

Offline replay (numpy, NEMO's entry frame with its halos, NEMO's mesh file;
no model step on the NEMO side), kt 6, 8, 10, pivot row, 68 wet faces:

| arm | against | kt 6 | kt 8 | kt 10 |
|---|---|---|---|---|
| replay as compiled | NEMO's record | 0 unequal | 0 unequal | 0 unequal |
| replay, image-row transport set to zero | card before | 1.6e-16 | 2.6e-17 | 1.7e-16 |
| card before | NEMO's record | 1.0 | 0.96 | 0.92 |
| card after | NEMO's record | 1.6e-16 | 5.1e-17 | 1.6e-16 |

Row below the pivot: replay 0 unequal against NEMO; card 2.5e-16 or less in
both arms. CONFIRMED: one operand, the transport on the fold-image V row.

## 6. What landed, and what was ruled out

**Landed.** A new model option `een_fold_image_v_row`, default False. When a
card sets it, the 3-D triad operator receives the fold-image V row: velocity
with the fold sign, face thickness and wet mask, each the mirror of the row
below under the grid's existing fold permutation. No new fold machinery.
Searched first: `grep -rn "fold_perm\|apply_north_fold\|fold_row"` in packages
(helpers reused); `grep -rn een_fold_image packages` afterwards: the config
field, the operator helper and its two pass-through sites, and the G1 card.
NEMO's own mask on that row equals the mirror of the row below (checked on the
mesh file).

No silent default: library default False; the legacy path returns the same
objects (tested by identity); only `build_orca2g1_flat_card` selects it, and
its validator refuses the card without it. ORCA2's cards do not select it.
No previously cited line moved: the edits keep their line counts and the
helper is appended at the end of its file.

Plant: the card with the option switched off (`--before-arm`) fails the test
(pivot-row error above 0.5 where the landed card is below 1e-15).

Net effect, registered: every row at kt 1..16 unchanged in status; u, uu_b,
vv_b first-over-bar later (20 -> 27, 20 -> 27, 20 -> 23); no row earlier; ssh
and v unchanged (17, 19). Closed cards: untouched by construction (default
off, Python-level branch); no closed-card ladder was re-run this round.

**Ruled out (measurement arms, in the json files).**
- Writing the fold image into the top V row of the carried velocity between
  steps changes that row only; the sea-level error is identical to four
  digits at kt 20 and kt 100. The stored zero row is not the cause.
- The ORCA2 lane's held private unit (the round-253 hook set and its raw
  V-mask arm) is traced on G1 and changes no bit of any field through kt 10.
  From an uncommitted probe: PLAUSIBLE-grade, to be re-run if it matters.

## 7. Instrument corrections made before any number was quoted

1. Given-entry seed. The first version reset the six barotropic histories to
   the cold-start value at every kt and showed 6e-4 at kt 5 while the
   independent chain sat at 5e-17. NEMO carries those histories
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:799-805`, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:817-822`) and the step
   record does not hold them. Fixed: they come from the card's own chain, and
   the restart-entry arm removes even that. kt = 1 was never affected
   (L-P4 HOLDS).
2. The count of bit-unequal cells cannot separate the regions: rounding
   differences travel with the wave. The regional relative error replaces it.
3. Right-hand-side rows are scored on NEMO's wet faces; land faces are
   reported apart.
4. The geometry gate sets its own precision; under the caller's default it
   compared single against double.

Controls run: score plant and entry plant exit 1; regions partition asserted;
NEMO's after-step frame equals its next entry frame for kt 1..9; NEMO's own
top V row is the fold image at every kt; restart equals frame at kt 10..90.

## 8. ORCA2 pointer

| statement | G1 result | ORCA2 |
|---|---|---|
| cyclic east-west exchange | exact to rounding through kt 100 arrival | deliverable: certified here on ORCA2's own grid |
| south wall and coast, no-slip mask | exact to rounding | deliverable |
| required filter alpha, 13 automatic substeps | at the bar kt 1..16 | deliverable (D115) |
| raw-mesh operands, fold masks (D114) | 36/36 geometry | deliverable |
| EEN Coriolis on the pivot row needs the fold-image V row | NEW, landed for G1 | UNMEASURED on ORCA2: its vector-form trend goes through the same operator with the same mask, so the same omission is PLAUSIBLE there. The option is wired for that path but not selected. |
| ORCA2 round-217/253 private fold unit | inert on G1 | not this statement |

`fe3mask` is the card's; NEMO defines it at `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dommsk.f90:258`. The replay
reproducing NEMO bit for bit is the only check it has had.

## 9. Predictions

| id | result |
|---|---|
| G-P1 36/36 geometry | REFUTED as worded (35/36), HOLDS after one row was corrected to NEMO's written quantity |
| L-P0 one-level step runs | HOLDS |
| L-P1 no boundary activity at kt 1 | HOLDS |
| L-P2 open interior at the bar at kt 1 | HOLDS (4.2e-17) |
| L-P3 all boundaries at the bar through kt 10 | HOLDS by the bar; the fold is nevertheless defective (regional measure) |
| L-P4 both labels equal at kt 1 | HOLDS |
| round-1 P2 substeps and Courant number | HOLDS |
| round-1 P4 fold statements exact | REFUTED |
| round-1 P6 interior exact at kt 1 | HOLDS without the TSUNAMI merge |

## 10. Choices

ASKED (round brief): everything measured; landing a cited statement selected
by the G1 card. UNASKED, offered for revert: (1) fold band = top three carried
rows, seam = two columns a side, south = three rows; (2) coast / open split;
(3) the option is also passed through the vector-form call site (inert unless
a card selects it); (4) round 1's geometry test replaced by the 36-row gate;
(5) regional relative error as the discriminator in place of the cell count.

## 11. Tests and review

Single review (codex): **findings**, three CONFIRMED, all acted on. Log:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round2/codex_review_r2.log.
Verbatim:
- "CONFIRMED — `restart_entry` is not "COMPLETE": it omits NEMO's live `ssha`
  restart field, retaining `eta_rk3_after=None` from the card chain".
  Acted on: relabelled (the arm now says the after-ssh slot is not seeded and
  refuses a card that reads the carried slot; this card's selection does not).
  Section 3's wording "complete state" above means: every prognostic field and
  the six histories, NOT that slot.
- "CONFIRMED — plants are unsound: `--plant geometry` is accepted but never
  applied, while generic plant success merely detects any pre-existing DEBT".
  Acted on: the dead name is removed; a plant now runs only at kt = 1 on the
  two whole-step arms and must move one of the five rows that are at the bar
  unplanted (T and S excluded). Re-run: both exit 1; unplanted exits 0; a
  plant on any other arm exits 2. The geometry plant was always the separate
  `--geometry-plant` and did fire.
- "CONFIRMED — the new config field was inserted mid-`NamedTuple`, shifting
  every later positional argument despite the class contract requiring
  chronological append-only fields". Acted on: the field is now the LAST field
  of the class; the base commit's field list is an exact prefix of the new one
  (checked by parsing both).
"VERDICT: findings". The review raised nothing against the sign, index or
staggering of the image row, the starred call, the replay, or the tests of
the landed statement. The fixes after the review were not re-reviewed. No
second (GLM) reviewer was available: a single review, not a dual one.

Evidence stamp: the json files carry commit 6613a0982. The later commits move
one config field to the end of its class, relabel the restart arm and tighten
the plants; none changes arithmetic. The focused tests below re-measure the
pivot row with and without the option at the final tree.

Focused tests at the final tree (one synchronous run): **6541 passed, 8
skipped, 1 failed**, with two known reds deselected (the slab-land dispatch
guard and the jra55 literal, both recorded in round 1). The one failure is
`tests/test_no_inline_physics_coeffs.py` on packages/land/legoesm/land/restart.py,
a file this branch's round did not touch (`git diff acdb00b92 -- packages/land`
is empty). The new file tests/ocean/fidelity/test_nemo_testcase_g1_orca2grid_ladder.py:
9 passed, including the option's plant. NO GATE ran for: any closed-card
ladder (GYRE, VORTEX, ORCA2); the option is default-off behind a Python-level
branch, which the identity test covers, but no closed card was re-scored.

Citation gate on this receipt: PASS, 11 citations, 0 unmapped, whole-map audit
clean; the shifted-citation plant exits 1 (citation_receipt.json,
citation_plant.json under final/).

Compliance, stated because no gate enforces it: search-before-build (stated
in sections 6 and the preregistration); one variable per arm (before / after
differ in the one option); instruments validated before quoting (section 7);
non-vacuity (option plant, geometry plant, score and entry plants, citation
plant shown red). Dual review: NOT met (single).

## 12. Evidence hashes (SHA-256 prefixes, final/)

geometry.json=5dd45424364238e4, geometry_plant.json=a2e00c7f0337ebff, given_entry_after.json=af38cdea69e71221, given_entry_before.json=de69ea98b9edc655, handoff_after.json=8259fefd55707606, record100_after.json=e3dad376b67fcd5b, record100_before.json=dfea8f4bf76432fa, restart_entry_after.json=d750d0075f24c2cb, restart_entry_before.json=0c746b98e93df2ef, rhs_after.json=b490e043dd9fc8d7, stages_after.json=f2ad548f272e0b79, vor_replay_after.json=c9b434ffcfcd5f97, vor_replay_before.json=b31c3bb321066332

## OPEN for round 3

1. The second fold statement: after the landing, one step from NEMO's state
   leaves 1.7e-12 m at kt 31 and 1.3e-5 m at kt 91 in the fold band. Start
   with the replay pattern on the V trend of the row below the pivot and on
   the substep Coriolis; a record with stage frames and a substep record
   around kt 20..25 would split the external mode (not yet written).
2. The open-interior excursion from kt 17 (5.75e-15), cause unknown.
3. The stored top V row of the 3-D velocity (zero against NEMO's image).
4. ORCA2 lane: measure the pivot-row statement on ORCA2's vector-form trend.
5. Operator: the TSUNAMI branch is still not merged here; G1 did not need it.

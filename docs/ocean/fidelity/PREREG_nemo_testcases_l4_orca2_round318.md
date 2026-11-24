# PREREG — ORCA2 round 318: the north-fold set under Decision 122

Frozen before any arm runs (Claude round; codex review: operator).
Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round318_fold_d122_gate.py`
(round-285 ladder, round-276 classifier, round-278 decision).  CPU, fp64,
libm, production JIT.  Seven D96 pairs: rung 0, OMT-1..4 (independent),
rung 10 (independent and given entry).

Import: gridladder `2b75714433..7a31b67ef` package hunks (rung G9, lane A
rounds 45-52), default-off, commit `7ccd8a1a1`.  Measured before arms: the
step jaxprs of rung 0, OMT-0..4, rung 10, GYRE, VORTEX (flux, vector) and
DINO are identical to base `3e9bb7dea` (11/11).

Resolved before measuring (round-313 resolved configs): tracer advection
(FCT, nn_fct 2) runs on rung 0, OMT-4, rung 10; OMT-1..3 run none.  Every
card runs vector-form momentum (G9 itself is flux form).

| ID | Prediction | Falsifier |
|---|---|---|
| R318-P1 | FCTF and KRHS3 move 0 rows on OMT-1, OMT-2, OMT-3 (no tracer advection). | any row moves. |
| R318-P2 | Each G9 statement alone, and G9 together, completes (no refusal) on all seven pairs. | a refusal or non-finite state. |
| R318-P3 | G9 together passes D122 (c)+(d) on all seven pairs. | an exact row lost, or first over-bar row `away`. |
| R318-P4 | With G9, the three held G6 fold statements (G9FOLD) complete on rung 0, OMT-4 and rung 10 (round 303: the fold unit completes only with the stage transport V mask). | a refusal. |
| R318-P5 | TOPV's kt-10 sea-level growth (round 308: 0.39 -> 7.4 m) is absent with G9: G9TOPV's kt-10 max abs ssh error <= 1.5x control on rung 0 and OMT-4. | > 1.5x control. |
| R318-P6 | The best passing set lowers the kt-10 max abs ssh error on rung 0 and OMT-4 by at least 2x (the user's dynamics-only question; PLAUSIBLE that the fold owns it). | less than 2x. |

Census (toward/away/equal), FOLDR's re-measure (Decision 137) and every
other number are registered, not predicted.

## Addendum (frozen before the arms2 runs)

First batch so far (rung 0, OMT-1..4, part of rung 10):
- KRHS3 (and so G9, G9FOLD, G9TOPV, G9FOLDR, ALL) refuses on all seven
  pairs: rung 0/OMT-1..4 keep the forced stage-3 heat/salt forms off (no
  forcing module, B90), which KRHS3 requires; rung 10 runs the diffusive
  BBL, which KRHS3's stage-3 path refuses (G9 has none).  R318-P2 is
  FALSIFIED for KRHS3.  Running it would need new code, not an import.
- VMASK moves 0 rows on every measured pair (with top-V off the card's
  pivot-row V velocity is zero, so the mask multiplies zero).
- FCTF alone: 0 rows lost, first over-bar equal, census all away (0/179 on
  rung 0, OMT-4), kt-10 max abs ssh error 0.394 -> 6.24 m.
- PIVOT alone: first over-bar row (kt-1 stage-1 T) `away` on rung 0 and
  OMT-1..4 -> fails (d).
New arms: G9N (G9 without KRHS3), VT (VMASK+TOPV), VTX (VT+FOLDX), FP
(FCTF+PIVOT), G9NTOPV, G9NFOLD, G9NFOLDR, ALLN.

| ID | Prediction | Falsifier |
|---|---|---|
| R318-P7 | FP passes (c)+(d) on rung 0, OMT-4, rung 10 (both labels) and its kt-10 max abs ssh error stays <= 1.5x control. | first over-bar away, exact row lost, or > 1.5x. |
| R318-P8 | VT moves rows and passes (c)+(d) on rung 0, OMT-4, rung 10; its kt-10 ssh error <= 1.5x control. | any of those fails. |
| R318-P9 | G9N passes (c)+(d) on all seven pairs. | an exact row lost, first over-bar away, refusal. |
| R318-P10 | G9NFOLD completes on rung 0, OMT-4, rung 10 (replaces P4 for the KRHS3-free set). | a refusal. |
| R318-P11 | G9NTOPV's kt-10 max abs ssh error <= 1.5x control on rung 0 and OMT-4 (replaces P5). | > 1.5x. |

## Addendum 2 (frozen before the arms3 runs; diagnostic, not a landing arm)

G9NFOLD on rung 0: 195/200 rows moved, 195 toward / 0 away, first over-bar
toward, kt-10 max abs ssh error 0.394 -> 0.0070 m.  Each fold member alone
leans away.  Arms3 (rung 0, OMT-4, rung 10 independent): NO<member> =
G9NFOLD without one of VMASK, FOLDX, FCTF, PIVOT, FOLDUNIT, RAWV, TOPV;
FOLDSET = those seven only (no CLOCKP, TWZV, R3UV).

| ID | Prediction | Falsifier |
|---|---|---|
| R318-P12 | Every NO<member> arm raises the kt-10 max abs ssh error on rung 0 above 2x G9NFOLD's (each fold member is needed). | some NO<member> stays within 2x. |
| R318-P13 | FOLDSET's kt-10 max abs ssh error on rung 0 is within 2x G9NFOLD's (CLOCKP/TWZV/R3UV are not fold partners). | > 2x. |

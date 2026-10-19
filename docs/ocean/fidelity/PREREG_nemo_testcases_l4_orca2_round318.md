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

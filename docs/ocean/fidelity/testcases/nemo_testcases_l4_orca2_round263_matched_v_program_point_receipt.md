# ORCA2 round 263 — scalar V ratio and matched stage program point

Date: 2026-10-12. Base: `dbef7e692`. Preregistration commit:
`6d81ccc26`. Gate commit: `a411c02de`. Post-measurement classifier commit:
`a7ba9f306`. Disposition: **HELD**.

This round changes no production model, card, deck, selector, carried-state
contract, stabiliser, sea-ice declaration, or `unmeasured_features` entry. It
measures OMT-4 separately as **independent** and **given NEMO's entry**, on
CPU fp64/libm production JIT. No NEMO acquisition was needed.

## Compiled statements and program points

NEMO computes the after-stage free-surface ratios and associates `r3va` as a
V-point **scalar**, with sign `+1`, before storing the live stage ratio:
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:168-170`. The
compiled V-point fold copy is in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`. This must not
be confused with the completed stage velocity association: NEMO associates
`vv(Kaa)` with V-point vector sign `-1` only after the stage update, at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`.

The admitted NEMO stage frame is recorded after that latter association.
legoESM's passive `stage_outputs[0]` observation point is before it. Comparing
those two without applying the compiled association is therefore a
program-point mismatch.

## Frozen predictions and verdicts

R263-P1 is **CONFIRMED**. The source-rounded scalar replay uses the V rule
with sign `+1`. On the pivot row, the unassociated `r3va` spans
`[-3.13279489435256e-4, 7.986329322912133e-5]`; the compiled scalar copy
spans `[-5.631981017715069e-4, 1.9270219165004197e-4]`. Substituting vector
sign `-1` changes all 180 row slots, with maximum difference
`1.1263962035430138e-3`. The admitted downstream source replay is exact on
valued support under both labels: live `e3v` is 0/1,977 unequal, `r1_hv` is
0/100, and `zvb` is 0/100. `zFv` retains only the registered 181/8,100
signed-zero differences, maximum absolute error 0.0.

R263-P2 is **CONFIRMED** and corrects round 262's boundary framing. Before
the compiled post-stage association, fold-band V differs from NEMO by maximum
`0.027227136128767482` m/s and RMS `0.002202566269216334` m/s. At the matched
post-association program point, the maximum is
`3.469446951953614e-18` m/s and RMS `2.2497900613824645e-19` m/s, below the
`2e-10` campaign floor. These values are identical for **independent** and
**given NEMO's entry**. The passive observer reproduces the unobserved
completed state array-exactly in every carried field. The `0.0272` value is a
real pre-association intermediate, but it is **not** a NEMO-vs-legoESM model
boundary at the recorded stage-frame point; round 262's claim that it was the
first valued model boundary is retracted.

R263-P3 is **REFUTED and retained**. The preregistration predicted that the
post-stage association alone would close the next stage-2 tracer boundary.
On the current tree, fold-band stage-2 salinity is `5.037097791516988` PSU
(RMS `0.14785615504257743`) before the causal association arm and
`1.8200702777361712` PSU (RMS `0.055623934176120826`) after it. Temperature
similarly falls from `0.27211560747403896` K to
`0.09829516579631314` K. The association is a measured contributor, but it
does not close the boundary and is not named as its sole owner.

R263-P4 is **CONFIRMED**. Both labels use the same admitted record hashes; the
one-bit control fires; and all nine plants refuse. The classifier verdict is
`PASS_R263_PROGRAM_POINT_RECONCILIATION`.

No package file changed, so no ORCA2, GYRE, DINO, tank, or other trajectory
can move on the final tree. No Decision-96 landing vote is reached. Decision
116 remains pending with the user and is not re-emitted.

## Evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round263/`.
Principal SHA-256 values are:

- independent measurement:
  `b8a99efa3bf5f7fe7e9da466c5ccee24c13cdb49bdca2dd70515f0704f40f096`;
- given-NEMO-entry measurement:
  `2dbfc7963e1f1098f2503d7d80b942cc369f213914a247985988aa741ab13b5c`;
- two-label classification:
  `4babca4c84839ef7d294190c5b4df47dec1f44fda11182db00353612e1c7df10`.

The focused round-260/262/263 control set passes 36/36; its log SHA-256 is
`eeefc976b08362ad074fa2cc067b96e32d0c663eb75d1fdf2a53b0e693bc9885`.
The once-only `tests/ocean/fidelity -n 12` battery reached 99% with every
reported test passing, then stopped emitting output on
`test_nemo_testcase_phase3_stage_sweep_gate.py::test_planted_stage_control_exits_nonzero_end_to_end`.
After a bounded wait it was terminated. The exact ID was rerun serially with
`-p no:xdist`; it also emitted no result and was terminated after a bounded
wait. This is an unresolved pre-existing long-run/hang, not a round-263
assertion failure. The battery and isolation log SHA-256 values are
`c1fbf5e7ad07c39dc3a29aed517e07a394061ce11c3723d5299eb1bd6a5fb608`
and `99c72c2902c7f25e44f9e9851888ab59ed62d66f422723eb41895d54fa6602cb`.

The requested separate `codex exec --sandbox read-only` review failed before
review because its in-process app-server could not initialize on the
read-only filesystem. **Independent review unavailable in-sandbox.** Its log
SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

Re-score the round-262 complete private unit at matched NEMO program points,
not at raw pre-association `stage_outputs`. Starting from the exact stage-1 V
point, walk the first valued stage-2 T/S boundary in compiled order. Keep the
scalar `r3va` association, post-stage U/V/T/S association, stage transport,
and tracer consumer distinct in the table; the current-tree causal arm proves
the post-stage association reduces but does not close the boundary. Offline
replay only. The complete unit stays private unless it later passes the full
Decision-96 gates.

ASKED choices: none. Unasked configuration choices made: none. Round
disposition: **HELD**.

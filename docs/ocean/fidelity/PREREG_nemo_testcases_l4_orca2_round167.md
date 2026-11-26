# Preregistration — ORCA2 round 167 kt=8 exit-depth walk

Date: 2026-10-07. Frozen base: `7ad068658`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round167/`.

Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the table. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Frozen record and source order

The operator-run round-166 acquisition is admitted before measurement only if
its existing checker retains exactly two rank records, 65 external substeps,
2,106 self-described groups per rank, exact rank coverage and all 20 terminal
restarts byte-identical to the round-96 baseline. A failed admission stops the
round without interpreting a value.

The compiled rung-0 oracle first constructs exit U depth and its masked
reciprocal at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:761-767`, then associates
the exit velocities, face depths, reciprocals and SSH in one `lbc_lnk` call at
`dynspg_ts.f90:770-780`. The recorder publishes the post-call arrays at
`dynspg_ts.f90:791-796`. The walk therefore scores, at kt=8 external substep 2:

1. the legoESM exit face depth against recorded `j002_hu_e`;
2. the legoESM exit reciprocal against recorded `j002_hur_e`;
3. the complete post-association seven-array image against the same record.

The implementation reuses round 129's self-describing two-rank assembler,
native-face mappings and scorer. It runs the same complete private arm as
rounds 164-166: raw reference face depth, unmasked V transport, materialised
`zhV` and the seven-array external-mode association. The ordinary arm must
still complete kt=7 and expose the same kt=8 trace without changing any
completed checkpoint.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R167-P1 | The acquired record remains admissible and additions-only. | The committed checker reports two exactly-once rank slabs, 65 substeps, 2,106 groups per rank and 20 byte-identical restarts. | Any census, content, rank coverage or restart comparison fails. |
| R167-P2 | The 42 non-finite U reciprocals from round 166 are caused by an already-wrong but finite exit depth, not by the reciprocal association itself. | At every registered non-finite cell, legoESM's pre-association exit depth is finite and differs from finite NEMO `j002_hu_e`; the reciprocal is the next source boundary and NEMO `j002_hur_e` is finite. | The pre-association exit depth is bit-exact at any registered cell, is itself non-finite, or NEMO's reciprocal is non-finite. |
| R167-P3 | Substituting NEMO's recorded exit depth only at the reciprocal consumer removes all 42 non-finites, while substituting only the recorded reciprocal cannot repair an earlier depth debt. | The depth-to-reciprocal replay produces zero non-finite U reciprocals; the reciprocal-only replay leaves the depth row unequal and is not treated as an owner. | The depth replay leaves a non-finite reciprocal or the claimed one-variable split is not isolated. |
| R167-P4 | The first cited owner remains upstream of the reciprocal and seven-array association, so the atomic halo/V-transport unit stays HELD this round. | The walk names the first preceding finite-magnitude boundary or the exact next missing operand; no production package/configuration change lands. | A single cited NEMO statement closes the complete unit and passes both ORCA2 ladders, the independent month boundary, GYRE, DINO and tank gates under Decision 96. |

## Controls and terminal rule

The gate must reject a record-bit plant at the registered substep/cell, a
trace-bit plant, a reordered source registry and a depth-replay plant that does
not affect the reciprocal. Full-domain operand scores are retained alongside
active-face scores so a fold or halo value cannot be hidden by a prognostic
mask.

No stabiliser, clip, configuration choice, carried-state change, sea-ice
change, bar relaxation or partial source-unit landing is permitted. Failed
predictions remain in the receipt.

ASKED choices: continue round 166's compiled-source recorded-operand walk.
UNASKED choices: empty.

# Preregistration — ORCA2 round 145 initial-growth walk

Date: 2026-10-04. Base: `9d8a8c9010891a7ec1ab720871272bc965bb6129`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round145`.
Every trajectory value is **independent**: hierarchy rung 0 starts from its own
climatological T/S, zero velocity, and zero sea surface. Decision 52's NEMO
entry operand is not used. Sea ice and
`unmeasured_features = ("linear_implicit_bottom_drag",)` stay unchanged.

The operator reports a complete round-144 target. This round first runs the
committed admission checker unchanged, then compares NEMO and legoESM at
column `(j,i)=(87,159)` and its incident faces for steps 1 through 10 in the
compiled source order: entry external mode, `stp_2D` output, then stage-1
transport. No value from the record has been inspected before this file.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R145-P1 | The existing round-144 record admits without changing its checker or bytes. | Status is `PASS_R144_INITIAL_GROWTH_RECORD`; all nine plants fire; 20 calibration restarts remain byte-identical. | Reject every science value; repair only a demonstrated checker defect or request a fresh target. |
| R145-P2 | Step-1 entry SSH and thickness are bit-exact at the target column. | `ssh_entry` and `r3t_entry` have zero bit differences and zero absolute error at step 1. | Retain REFUTED; the earliest unequal entry row becomes the boundary and no step-1 statement owns it. |
| R145-P3 | The mechanically first row above the frozen `2e-10` floor is step-1 `ssh_after`, immediately after `stp_2D`. | All preceding rows are at bar and step-1 `ssh_after` exceeds the floor. | Retain REFUTED; the mechanically selected row in step/source order owns the walk. |
| R145-P4 | Every registered value stays finite through step 10. | Neither model has a non-finite registered value. | Retain REFUTED and make the first non-finite row the boundary. |
| R145-P5 | If P3 is confirmed, the existing step-1 barotropic operand streams are sufficient to split the boundary without a new NEMO acquisition. | A one-variable substitution reaches a cited compiled statement and its synthetic control fires. | Name the exact missing operand stream and write a new-target, rank-complete acquisition; infer no owner. |
| R145-P6 | This round is measurement-only unless one compiled, one-variable statement closes the bracketed boundary under every standing gate. | No model, card, deck, carried state, stabiliser, sea-ice selector, or unmeasured-feature change lands without that proof. | Revert the unsupported change and finish HELD or STOPPED_FOR_RECORD. |

## Round bar

The analysis parses the self-describing record, proves exact both-rank
coverage, runs CPU fp64/libm with production JIT, and compares an instrumented
trajectory with a separately compiled ordinary trajectory bit-for-bit. The
first row over `2e-10` is selected mechanically by step and compiled source
order. Only the bracketed at-bar/over-bar boundary may be split, one recorded
operand at a time. The already-cited implicit bottom-drag divisor is measured
only if the source-ordered walk reaches it; Decision 94's EVD change is not
implemented independently. No configuration choice or stabiliser is allowed.

## Compiled-source anchors read before measurement

The entry frame precedes `stp_2D`, whose output precedes stage 1, in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3.f90:202-221`.
The source-ordered slow forcing computes the depth average, then drag, before
the split-explicit solve in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stp2d.f90:206-234`.
The split-explicit solve copies forcing and carried external-mode state before
its substep loop in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:288-380`.
NEMO's implicit bottom-drag diagonal divides by the live partial bottom-face
thickness in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynzdf.f90:303-306,470-474`.

# Preregistration — ORCA2 round 144 early-growth bracket

Date: 2026-10-04. Base: `d4fbba7d2`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round144`.
Every trajectory value is **independent**: hierarchy rung 0 starts from its own
climatological T/S, zero velocity, and zero sea surface. Decision 52's recorded
NEMO entry is not used. Sea ice and
`unmeasured_features = ("linear_implicit_bottom_drag",)` stay unchanged.

The operator reports exit 0 from the committed round-143 launcher. This round
first re-runs its admission checker against the existing files, then compares
NEMO and legoESM at column `(j,i)=(87,159)` and its incident faces for steps
11 through 30 in the compiled source order: entry external mode, `stp_2D`
output, then stage-1 transport.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R144-P1 | The existing record admits without modifying its checker or bytes. | The checker prints `PASS_R143_EARLY_GROWTH_RECORD`; all nine admission plants fire; calibration restarts remain byte-identical. | Reject all science values; repair only a demonstrated checker defect or request a fresh target. |
| R144-P2 | The first registered target row above the frozen `2e-10` floor is step-11 `ssh_entry`. | Steps 11..30 reproduce passively, and source-ordered selection returns step 11 `ssh_entry`. | Retain REFUTED; the mechanically selected earlier/later row owns the walk. |
| R144-P3 | Step 11 is already above the floor, so the new interval is left-censored. | Step-11 `ssh_entry` exceeds `2e-10`; no at-bar predecessor exists in this record. | Retain REFUTED and split the first at-bar/over-bar boundary found inside steps 11..30. |
| R144-P4 | The target remains finite through step 30 in every registered row. | Neither model has a non-finite registered value. | Retain REFUTED and make the first non-finite row the boundary. |
| R144-P5 | This is measurement-only unless a single compiled, one-variable statement closes an internally bracketed boundary under every standing gate. | No model, card, deck, carried-state, stabiliser, sea-ice, or `unmeasured_features` change lands without that proof. | Revert the unsupported change and finish HELD or STOPPED_FOR_RECORD. |

## Round bar

The analysis must parse the self-describing record, prove exact both-rank
coverage, run CPU fp64/libm with production JIT, and compare an instrumented
trajectory against a separately compiled ordinary trajectory bit-for-bit.
The first row over `2e-10` is selected mechanically by step and compiled source
order. If step 11 is already over the floor, the round writes a fail-closed,
new-target acquisition for rank-complete steps 1..10 rather than inferring an
owner. No configuration choice or stabiliser is permitted.

## Compiled-source anchors read before measurement

The entry frame precedes `stp_2D`, whose output precedes stage 1, in
`ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/stprk3.f90:202-221`. Stage-1 `wzv`,
the `e1e2t*ww` construction, and the recorded transport frame are ordered in
`ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/traadv.f90:299-320`.

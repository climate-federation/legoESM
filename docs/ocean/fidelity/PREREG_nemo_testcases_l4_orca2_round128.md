# Preregistration — ORCA2 round 128 northern U-grid EEN operands

Date: 2026-10-03. Base: `2baaf09e4e870f482dff0e1158b150f5cc3b3d82`.
Every hierarchy-rung-0 number is **independent** because rung 0 starts from
NEMO's own from-rest state. Shipped rung-7 numbers are **given NEMO's recorded
entry** under Decision 52. No configuration, forcing, initial state, carried
state, stabilizer, sea-ice selector, or `unmeasured_features` entry may change.

## Admitted boundary and compiled statements

Round 127 made both recorded northern NW/NE quotients exact and found the next
source-ordered operands at the northern U neighbor: live thickness differs in
95 NW and 91 NE magnitude cells, followed by frozen `umask` differences in
1,270 NW and 1,283 NE cells. The first unequal cell on both paths is
`(j,i,k)=(147,31,3)`.

NEMO reads `e3u_0` as a U-point field with copy fill and applies the ordinary
U-grid lateral boundary (`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/domzgr.f90:186-201`,
`:288-303`). It builds `umask`, applies the same U-grid lateral boundary, and
only then uses that mask (`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258`).
For a T pivot with `nn_hls=2`, the executed U branch fills the northern rows
from the reflected row with the U-origin permutation and sign `+1`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:639-700`). The EEN V
recurrence consumes the resulting `e3u` and `umask` at `(ji-1,jj+1)` and
`(ji,jj+1)` (`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1339-1348`).

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R128-P1 | The admitted round-120 fraction and round-118 recurrence records remain passive and complete. | Admission passes; both ranks cover the domain exactly once and inherited streams plus terminal restarts retain identity. | Refuse numerical interpretation; request a new record only if the existing record is incomplete or perturbed. |
| R128-P2 | NEMO's ordinary U-grid T-pivot association is the row below the stored pivot, U-origin permutation, sign `+1`. | Applying that association independently makes all 95 NW and 91 NE live-thickness differences bit-exact. | Retain **REFUTED** and stop at the first surviving source-row, permutation, or sign alternative; do not change production. |
| R128-P3 | The same U-grid association owns the frozen-mask boundary. | Applying it independently makes all 1,270 NW and 1,283 NE mask differences bit-exact. | Retain **REFUTED** and stop at the first surviving mask construction or association boundary. |
| R128-P4 | Thickness-only and mask-only controls each leave the other registered boundary unchanged; the combined arm makes both stored products bit-exact. | Each isolated arm changes only its named operand; combined NW/NE product rows become 0/0 bit/magnitude unequal. | Any cross-movement is a confound; any surviving product bit blocks recurrence interpretation. |
| R128-P5 | Once both operands and products are exact, ordinary IEEE zero addition makes both recorded northern recurrences bit-exact. | NW/NE before/after rows become 0 unequal without oracle substitution. | Stop at the first unequal recurrence level and keep the failed prediction. |
| R128-P6 | The later U-grid metric scale uses the same association but remains downstream and is not interpreted until P2--P5 pass. | Scale is measured only after both recurrences are exact. | Any earlier surviving item ends the walk before scale. |
| R128-P7 | Resolved execution scope remains literal-EEN cards only. | ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC execute; GYRE and both tanks do not. | Any scope movement refuses the arm. |

## Measurement and landing bar

The measurement reuses the admitted round-118 recurrence and round-120
fraction records plus the round-127 source-order gate. It runs production JIT
on CPU under fp64/x64/libm, refuses a dirty or wrongly stamped tree, and carries
independent oracle-bit, candidate-bit, wrong-row, wrong-permutation,
thickness-only, mask-only, and scope-route controls. No new NEMO run is
predicted necessary.

This round may land a production association only if the separately measured
thickness and mask operands, their products, and both recurrences become
bit-exact; every control fires; no earlier source item moves; both ORCA2 ladder
registries and every executing-card gate pass; and the citation plant fires.
Otherwise it remains **HELD** at the first non-bit item with the next
source-ordered acquisition or measurement named.

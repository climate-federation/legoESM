# Preregistration — ORCA2 round 158 exact U-cyclic and V-fold operands

Date: 2026-10-05. Frozen base: `a5a94421d`.
Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity, and zero sea surface. No
configuration, initial state, forcing, carried-state form, stabiliser,
sea-ice selector, or `unmeasured_features` entry may change.

## Compiled source program

The rung-0 deck resolves `jpni=2`, `jpnj=1`, `nn_hls=2`,
`ln_nnogather=.TRUE.`, and a T-point north fold. The compiled double-precision
`lbc_lnk` packs and sends west/east slabs at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:1961-1979`, receives and
writes them at `lbclnk.f90:2060-2073`, then dispatches the MPI north fold at
`lbclnk.f90:2104-2114`.

For the no-gather fold, the compiled program derives the V extra-line count
from the T-pivot and V-stagger selectors at `lbcnfd.f90:1561-1577` and
`lbcnfd.f90:1629-1663`. It maps the received neighbour rank/index at
`lbcnfd.f90:1712-1739`, then applies the partial-line overwrite and field sign
at `lbcnfd.f90:1741-1767`. The topology setup marks the T-pivot U half-line,
the V/F full line, and the periodic edge cells at `mppini.f90:1412-1440`.

Round 157 proved that the current private split composes exactly to the
already bit-exact seven-field call: U cyclic changes 24 compact closure cells,
while V fold changes 180 cells and carries the large stage-1 growth. This
round checks each split directly against the admitted two-rank oracle record,
rather than inferring faithfulness from composition alone.

## Frozen protocol

1. Parse the two self-describing rank records through the existing admitted
   reader. Require the resolved local shapes, owned slabs, origins, and 65
   substeps before inspecting values.
2. Gate all eight two-cell west/east halo assignments in rank/source order.
   Compare bits, including signed zero. Map NEMO's inner west halo to the
   compact U closure and require the round-157 U-cyclic arm to use that exact
   source.
3. At substep 1, split the V fold into its source line, V permutation,
   `psgn=-1`, and pivot-row overwrite. Require each operand to match the card
   descriptor and the completed compact row to match NEMO bit for bit.
4. Require U-cyclic-then-U-fold and V-cyclic-then-V-fold to remain bit-exact
   to the full helper on all 65 substeps. The ordinary observed state must
   remain array-identical to the unobserved production state.
5. Re-run the independent rung-0 production ladder and require 0/200 moved
   rows against round 157. No component is promoted to production this round.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R158-P1 | The admitted record expresses the literal two-rank west/east ordering. | All eight rank/side/two-halo source-target comparisons are bit-exact. | Any comparison differs, including only by zero sign. |
| R158-P2 | The compact U closure is NEMO's inner west halo sourced from the peer rank's last owned U column. | The source operand and U-cyclic result are bit-exact on all 148 rows; the operation remains the first non-bit split against unchanged production. | Wrong source column/order, any oracle bit mismatch, or the split becomes vacuous. |
| R158-P3 | The V fold uses the row below the stored pivot, the card's V permutation, and sign `-1`. | Each operand is exact and the 180-cell pivot-row result is bit-exact to NEMO. | Any operand association or result differs. |
| R158-P4 | The split remains complete and passive. | Both 65-substep compositions are exact and all seven ordinary state fields are exact. | Any composition or observer bit moves. |
| R158-P5 | Production remains unchanged. | The independent rung-0 ladder moves 0/200 rows, loses no AT-BAR row, and keeps the same first debt. | Any row moves, an AT-BAR row leaves, or first debt advances. |

## Controls and terminal rule

Plants must independently perturb one U rank-halo source, one V source-line
value, the V permutation, the V sign, and one composition result; each must
make the gate refuse. The gate must also reject a missing rank, wrong local
shape, wrong owned placement, or an unknown split name.

This is measurement-only. Even if both operations are exact, the round is
**HELD**: the next round may combine the two exact associations into the one
compiled seven-field source statement and run the complete landing gates.

ASKED choices: none. UNASKED choices: empty.

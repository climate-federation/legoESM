# Preregistration — ORCA2 round 159 external-mode halo landing

Date: 2026-10-05. Frozen base: `b7b1e747b`.
Hierarchy rung 0 is **independent**; shipped rung 7 is **given NEMO's
recorded entry**. These labels remain separate. No configuration, initial
state, forcing, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry may change.

## Compiled statement and isolated candidate

The resolved rung-0 program associates the seven live external-mode arrays in
one call at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`.
Its two-rank implementation packs and sends the west/east halos at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:1961-1979`, receives and
writes them at `lbclnk.f90:2060-2073`, then dispatches the north-fold exchange
at `lbclnk.f90:2104-2114`. The T-pivot no-gather branch selects the V
neighbour, permutation, overwrite and sign at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1712-1767`.

Round 158 proved every active U-cyclic and V-fold operand and result bit-exact
against the admitted two-rank record, including signed zero, and proved that
the split composes exactly to the complete helper. This round isolates that
one seven-field association. The held raw-reference-depth, unmasked-V-
transport and V-materialisation arms remain OFF. This is therefore not the
four-arm candidate that round 155 found non-finite at kt=8.

The instantiated ORCA2 rung-0 and rung-7 cards both resolve production JIT,
fp64/libm, `rk3_ws`, 65 external substeps, `nemo_ssh_avg` face depth, and a
stored T-pivot fold. The shared-card census and gates below determine all
other executing cards; no card is waived by declaration.

## Frozen protocol

1. Re-run round 158's exact operand gate and all seven plants at the frozen
   preregistration commit.
2. Run the independent rung-0 and given-entry rung-7 kt=1..10 ladders with
   only the complete external-mode association enabled. Compare each against
   the same-tree production control: register all 200 rows per ladder,
   exact-row losses, first-debt movement, and kt=10 stage-3 salinity maximum.
3. If both pass, transcribe the already-proved association onto the shared
   NEMO-literal barotropic path with no new selector. The retained private arm
   becomes a no-op control against production.
4. Run GYRE's certified ladder and year gate, DINO, VORTEX, LOCK_EXCHANGE,
   OVERFLOW, generic-card, citation, focused-test, and push-gate coverage.
   Every moved row is registered. Run the full ocean-fidelity battery once.
5. If either ORCA2 ladder refuses, loses an exact row, advances its first debt,
   or exposes the registered salinity compensation, keep production unchanged
   and name the exact red row. Because this is the second halo round in the
   batch, merge the pending GYRE lane before any further halo walk.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R159-P1 | The complete association remains the exact composition proved in round 158. | Round-158's 520 U-halo comparisons, 65 U/V compositions, seven ordinary fields, and all plants retain their verdicts. | Any exact comparison moves, an observer moves production, or a plant stays green. |
| R159-P2 | The isolated association completes both ORCA2 ladders safely. | Both 200-row ladders finish with no exact-row loss and unchanged-or-later first debt. | Refusal, non-finite value, exact-row loss, or earlier first debt. |
| R159-P3 | The association does not revive the held salinity exposure. | kt=10 stage-3 S maximum is no greater than the same-tree control on either ladder. | Either maximum increases; an approximately 31 PSU row is an immediate veto. |
| R159-P4 | The landed shared statement satisfies every executing-card gate. | All shared-card trajectory/year gates meet their registered predicates, with every move recorded. | Any unregistered or over-floor regression, or any executing card remains unmeasured. |
| R159-P5 | The retained private complete arm becomes a non-vacuous no-op control after landing. | Explicitly enabling it changes zero bits, while reverting the production association makes the control detect the round-158 U/V operands. | The explicit arm moves production, or the revert control stays green. |

## Terminal rule

Landing requires R159-P1 through R159-P5 and all repository gates. A red
ORCA2 ladder ends the halo candidate **HELD** with no production physics
change; the required pending GYRE merge may still land as a separate merge
unit in this round. A red shared gate leaves the candidate unlanded. No
stabiliser, tolerance relaxation, per-card switch, or configuration choice is
permitted.

ASKED choices: none. UNASKED choices: empty.

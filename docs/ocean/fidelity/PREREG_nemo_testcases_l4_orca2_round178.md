# Preregistration — ORCA2 round 178 independent external-stage SSH walk

Date: 2026-10-08. Frozen base: `86c2a9051`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round178/`.

Every trajectory number is **independent hierarchy rung 0**: both models start
from the rung-0 deck's climatological T/S, zero velocity and zero sea surface.
No given-NEMO-entry result is mixed into this round. The shipped rung-10 card,
its six sea-ice selectors and its `unmeasured_features` tuple remain unchanged.

## Frozen record and source order

The oracle is the admitted round-96 rank-complete split-explicit record. Its
two self-describing files contain entry operands, all 65 external substeps and
the final weighted outputs. Payload shapes and names are parsed from the record
headers; no byte count or field list is predicted by the new gate.

The compiled rung-0 program copies completed slow SSH/U/V forcing first
(`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:287-293`), zeroes every
barotropic history at the initial step (`:341-349`), then runs the 65 substeps
in source order (`:471-848`), normalizes the weighted endpoints and applies the
final U/V association (`:888-937`). The walk scores those boundaries in that
order and stops at the first row above the fixed `2e-10` floor. Bit-exactness
is reported separately and requires `np.array_equal`.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R178-P1 | The round-96 record remains admissible without reinterpretation. | Both ranks parse from their own headers, cover 148x180 exactly once, carry 65 substeps, and retain the admitted restart/content provenance. | Any header, name, shape, placement, coverage, substep-count or provenance check fails. |
| R178-P2 | The corrected private rung-0 entry and the record's initial barotropic histories are bit-exact. | T/S/u/v/ssh and the initial SSH/U/V histories have zero unequal active cells. | Any entry or initial-history row differs. |
| R178-P3 | Replacing only the initial histories with NEMO's recorded histories is a null arm because the compiled `ll_init` branch sets them all to zero. | Baseline and history-only endpoint/row scores are identical. | Any score moves. |
| R178-P4 | The first non-bit source-ordered operand is one of the three completed slow-forcing fields copied before the substep loop. | Entry/history rows are exact and the first above-floor row is `ssh_frc`, `u_frc`, or `v_frc`. | All three forcing rows are at the floor, or an earlier entry/history row differs. |
| R178-P5 | Substituting only NEMO's completed slow forcing moves the external-stage SSH endpoint toward NEMO; adding the already-exact histories changes nothing further. | Slow-only endpoint RMS and maximum both decrease, and slow-only equals slow+history. | Either endpoint metric does not decrease, or history changes the slow-only result. |
| R178-P6 | This is measurement-only. | No `packages/`, card selector, threshold, stabiliser, carried state, sea-ice field or configuration changes. | Any such change lands. |

Failed predictions remain in the receipt. A forcing row above the floor names
an upstream completed-RHS boundary; it does not attribute an operator without
an offline source-ordered replay of that RHS. If all forcing rows clear, the
walk continues through the recorded substeps, weighted endpoint and final
association and stops at the first non-bit statement.

## Controls and terminal rule

The new gate reuses the existing record parser and private barotropic trace. It
must prove the trace passive against the ordinary solver, and its plants must
refuse rank placement, one record bit, source order, arm identity and one
endpoint ULP. Candidate, oracle and geometry arrays are float64; execution is
production JIT on CPU under the fp64/libm policy.

No production statement lands unless the walk reaches a cited NEMO statement
and a separate one-variable implementation arm passes the full ORCA2/GYRE/DINO/
tank gates. Otherwise the round is HELD at the named boundary.

ASKED choices: replay the admitted split-explicit record on the corrected
independent rung-0 card and stop at its first non-bit barotropic statement.  
UNASKED choices: empty.

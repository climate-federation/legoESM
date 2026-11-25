# Preregistration — ORCA2 round 180 independent HPG component walk

Date: 2026-10-08. Frozen base: `ef22a09a7`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round180/`.

Every scientific number is **independent hierarchy rung 0**. Both models start
from the rung-0 deck's climatological T/S, zero velocity and zero sea surface.
No given-NEMO-entry result is mixed into this round. The shipped rung-10 card,
its six sea-ice selectors and its `unmeasured_features` tuple remain unchanged.

## Frozen record decision and source order

The round-41 component record is not admissible for this claim: its own header
and receipt label it **given NEMO's entry**, and it was produced by the shipped
ORCA1ICE deck rather than hierarchy rung 0. Round 180 therefore acquires the
same additions-only component boundaries from the admitted round-92 rung-0
build. No number from the new record is read before its header, rank coverage,
content pin and additions-only restart identities pass.

The compiled rung-0 HPG statement first builds the along-surface meridional
accumulator `zhpj`, then the local s-coordinate correction `zvap`, and assigns
`pvv = zhpj + zvap`
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`). The walk compares
those three boundaries in that order on round 179's registered 68 northern-fold
faces. If `zhpj` is first, its surface seed and then its top-down recurrence are
split using the record's local north-halo operands before any implementation
claim is made.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R180-P1 | The old component record is ineligible and a new rung-0 record is necessary. | The round-41 admission says `given NEMO's entry`, while the round-92 deck and record say hierarchy rung 0. | The old record proves the same independent rung-0 entry and deck byte-for-byte. |
| R180-P2 | The additions-only writer is passive. | Both rank records parse from their self-describing headers, cover 148x180 exactly once, and all 20 ocean restarts are byte-identical to the admitted round-92 run. | Any header, field, rank, provenance, restart or inherited-record identity fails. |
| R180-P3 | `zhpj` is the first non-bit HPG component on the 68 registered faces. | `zhpj` differs above the fixed `2e-10` floor and no earlier admission/calibration boundary differs. | `zhpj` is at the floor: continue to `zvap`, then `sum_v` in compiled order. |
| R180-P4 | The first `zhpj` debt is the northern-fold north-neighbour association, not a local density or metric value. | Local owned `rhd`, live `e3w`, live `gdept_z0` and `r1_e2v` are bit-exact; substituting NEMO's recorded north halo closes the first component boundary. | Any local operand differs first, or the north-halo-only replay does not close `zhpj`. |
| R180-P5 | This round is record acquisition and measurement only. | No `packages/`, card selector, threshold, stabiliser, carried state, sea-ice field or configuration changes. | Any such change lands. |

Failed predictions remain in the receipt. A component statement is not eligible
to land by itself: round 179 already proved that raw HPG, `e3v`, `vmask` and
`r1_hv0` form a cancelling unit. The complete unit must later satisfy Decision
96 under both ORCA2 ladders and all shared-card gates.

## Controls and terminal rule

The acquisition checker parses the record's magic, header integers and every
per-array `(name, rank, n1, n2, n3, payload)` description. Plants must fire for
the writer layout, header, field name, field dimensions, truncation, swapped
rank and restart identity. The launcher is pinned by the committed acquisition
artifacts' content, never by a moving producer commit.

If the operator has not run the new acquisition, round 180 stops
`ACQUISITION_NEEDED` without an HPG component claim. Candidate, oracle and
geometry arrays are float64; any later replay uses production JIT on CPU under
the fp64/libm policy and retains the fixed `2e-10` floor.

ASKED choices: source-ordered split of round 179's independent raw HPG V debt.  
UNASKED choices: empty.

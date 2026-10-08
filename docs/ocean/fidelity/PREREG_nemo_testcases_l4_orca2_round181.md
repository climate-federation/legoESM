# Preregistration — ORCA2 round 181 independent HPG component walk

Date: 2026-10-08. Frozen base: `a55bbc186`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round181/`.

Every scientific number is **independent hierarchy rung 0**. Both models start
from the corrected rung-0 climatological T/S, zero velocity and zero sea
surface. No given-NEMO-entry result is mixed into this round. The shipped
rung-10 card, its six sea-ice selectors and its `unmeasured_features` tuple
remain unchanged.

## Frozen source order and measurement

Round 180's record is eligible only if its self-describing admission remains
`PASS_R180_HPG1_ADMISSION`, both ranks cover the 148x180 domain exactly once,
the inherited round-92 RHS streams are byte-identical, and all 20 restarts are
byte-identical. The measurement reads no component payload before those checks.

The executing rung-0 HPG routine builds the meridional along-surface
accumulator `zhpj`, then the local s-coordinate correction `zvap`, then assigns
`pvv = zhpj + zvap`
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`). The walk compares
those boundaries in that exact order on round 179's registered 68 northern-fold
faces and their 1,319 wet contributing levels. It uses the already-certified
pure literal implementation offline on the corrected passive entry; no new
in-executable observer is allowed.

If `zhpj` is first, its inputs are split in compiled order: the current and
north-neighbour `e3w*rhd` products, their subtraction, the metric scaling, and
the top-down accumulation. The local/north inputs are taken from the record's
owned cells and explicit north halo rather than inferred from a periodic roll.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R181-P1 | The round-180 record is passive and eligible. | Admission, rank coverage, inherited RHS identity and all 20 restart identities pass. | Any admission or identity check fails; stop without reading a component result. |
| R181-P2 | The record and pure literal replay calibrate. | Replaying NEMO's recorded inputs reproduces recorded `zhpj`, `zvap`, and `sum_v` within the fixed `2e-10` floor, and replaying candidate inputs reproduces round 179's candidate raw HPG boundary within that floor. | Either replay misses its own completed boundary; stop as an instrument failure. |
| R181-P3 | `zhpj` is the first non-bit component on the 68 registered faces. | `zhpj` differs above the floor; no earlier admission/calibration boundary differs. | `zhpj` is at the floor: continue to `zvap`, then `sum_v` in compiled order. |
| R181-P4 | The first `zhpj` debt is the northern-fold north-neighbour association. | Current-cell density/thickness and metric operands are at the floor, the north operand differs first, and a recorded-north-only replay closes the first component boundary. | A local operand differs first, or recorded north alone does not close the component. |
| R181-P5 | This round is measurement only. | No `packages/`, card selector, threshold, stabiliser, carried state, sea-ice field or configuration changes. | Any such change lands. |

Failed predictions remain in the receipt. No component statement lands alone:
round 179 established a four-operand cancelling unit comprising raw HPG,
`e3v`, `vmask`, and `r1_hv0`. The complete unit remains subject to Decision 96
under both ORCA2 ladders and every shared-card gate.

## Controls and terminal rule

Plants must fire for rank placement, record bits, source-order selection,
target-mask census, self-replay calibration, first-boundary selection, and a
one-ULP endpoint. Candidate, oracle and geometry arrays are float64 under the
fp64/libm policy on CPU with production JIT. A replay that does not reproduce
its own completed boundary is an instrument refusal, not a physics result.

Round 181 stops after naming the first component and, only if calibrated, its
first operand statement. The cancelling unit is analysed jointly in the next
round before any landing.

ASKED choices: source-ordered split of round 179's independent raw HPG V debt.  
UNASKED choices: empty.

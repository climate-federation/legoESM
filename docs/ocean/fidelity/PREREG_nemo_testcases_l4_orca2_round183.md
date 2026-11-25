# Preregistration — ORCA2 round 183 HPG fold-operand walk

Date: 2026-10-08. Frozen base: `0ac196427`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round183/`.

Every scientific number is **independent hierarchy rung 0**. Both models start
from the corrected rung-0 climatological T/S, zero velocity and zero sea
surface. No given-NEMO-entry result is mixed into this round. The shipped
rung-10 card, its six sea-ice selectors and its `unmeasured_features` tuple
remain unchanged.

## Frozen admission and source order

The round-182 record is eligible only if its self-describing checker reports
`PASS_R181_HPG_FOLD_ADMISSION`, its two ranks cover the 148x180 domain
exactly once, and all 20 terminal restarts remain byte-identical to the
admitted round-180 independent record. No operand payload is read before those
checks.

The executing rung-0 HPG routine builds the meridional along-surface
accumulator in this order:

1. surface north product `north_e3w * north_rhd`;
2. surface current product `current_e3w * current_rhd`;
3. north-minus-current subtraction and metric/gravity scaling;
4. at each deeper level, north and current two-level density sums;
5. their thickness products, subtraction, scaling, and top-down accumulation.

This is the compiled program in
`ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-416` and
`:445-453`. The offline replay reads the record's evaluated fold operands
and compares only the 1,319 wet levels on round 179's registered 68 northern
fold faces. It uses the existing certified pure literal operator and may not
add an in-executable observer.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R183-P1 | The round-182 acquisition is passive and eligible. | Admission, exactly-once rank coverage, and all 20 restart identities pass. | Any admission or restart check fails; stop before reading operand payload. |
| R183-P2 | The exact-operand replay calibrates. | Replaying all five recorded fields reproduces recorded `zhpj` within the fixed `2e-10` floor; candidate replay reproduces round 181's completed `zhpj` debt. | Either side misses its own completed boundary; stop as an instrument failure. |
| R183-P3 | The first non-bit primitive input is the evaluated north-fold thickness. | `north_e3w` differs above the floor before `north_rhd`, current thickness/density, or metric. | `north_e3w` is at the floor or an earlier admission/calibration boundary fails. |
| R183-P4 | The first non-bit executable statement is the north product at the surface seed, and recorded north operands alone close `zhpj`. | The surface north product differs first; substituting recorded north thickness+density closes every registered `zhpj` level within the floor. | The first debt is a later product/subtraction/accumulation, or the north-only replay does not close. |
| R183-P5 | The four-operand depth-average unit remains cancelling and is not landable statement-by-statement. | Raw HPG, `e3v`, `vmask`, and `r1_hv0` are reported side by side; no partial unit lands. | A partial operand is changed in production. |
| R183-P6 | This round is measurement only unless the complete unit is proved eligible under Decision 96. | No package/card/configuration/carried-state/stabiliser/sea-ice change lands without both ORCA2 ladders and every shared-card gate. | Any ungated production change lands. |

Failed predictions remain in the receipt. If the first statement is named but
the complete four-operand unit is not yet mechanically closed, the round is
**HELD** with that unit as OPEN; it does not speculate past the first
source-ordered boundary.

## Controls

Plants must fire for admission status, rank placement, record bits, source
order, target census, self-replay, first-input selection, first-statement
selection, north-only arm non-vacuity, and a one-ULP endpoint. Candidate,
oracle, and geometry arrays are float64 under the fp64/libm policy on CPU with
production JIT.

ASKED choices: source-ordered offline split of the admitted independent HPG
fold operands and atomic analysis of the registered four-operand unit.  
UNASKED choices: empty.

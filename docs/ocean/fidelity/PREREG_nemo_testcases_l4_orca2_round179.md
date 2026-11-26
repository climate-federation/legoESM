# Preregistration — ORCA2 round 179 independent V-RHS operator walk

Date: 2026-10-08. Frozen base: `0f7aa62f4`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round179/`.

Every trajectory number is **independent hierarchy rung 0**: both models start
from the rung-0 deck's climatological T/S, zero velocity and zero sea surface.
No given-NEMO-entry result is mixed into this round. The shipped rung-10 card,
its six sea-ice selectors and its `unmeasured_features` tuple remain unchanged.

## Frozen records and source order

The oracle inputs are three already-admitted records. Round 92 supplies the
rank-complete kt=1 three-dimensional cumulative momentum RHS after HPG, LDF,
VOR, KEG and ZAD. Round 169 supplies the rank-complete static `e3v_3d`,
`vmask` and `r1_hv_0` operands. Round 96 supplies the completed kt=1 `Ve_rhs`
that round 178 proved differs on 68 northern-fold faces. Every payload is
parsed from its self-describing header; cross-record closure is required before
any operator number is accepted.

The compiled rung-0 program calls HPG, LDF, VOR, KEG and ZAD in that order
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175`), then constructs
`Ve_rhs` as the written vertical sum of `e3v_3d * Krhs * vmask`, followed by
`r1_hv_0` (`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`). The walk
replays that two-dimensional boundary after each cumulative operator and stops
at the first row above the fixed `2e-10` floor.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R179-P1 | The three admitted records close without reinterpretation. | Both rank pairs cover 148x180 exactly once; NEMO's recorded after-ZAD operands replay the round-93 depth-V boundary and the round-96 completed V forcing on the registered 68 faces. | Any header, placement, shape, provenance or cross-record closure check fails. |
| R179-P2 | The pure offline candidate replay is calibrated to production. | Its after-ZAD two-dimensional V result agrees with round 178's candidate completed V forcing on all 26,640 faces within the fixed floor and has exactly the same 68-face over-floor set. | The replay creates, removes or moves any over-floor face. |
| R179-P3 | HPG is the first cumulative operator above the floor on the registered 68-face set. | HPG differs on at least one registered face and no earlier entry/record calibration row differs. | HPG is at the floor: continue through LDF, VOR, KEG and ZAD in compiled order. |
| R179-P4 | The first non-bit operand in HPG's two-dimensional association is `vmask`. | Candidate and NEMO `e3v_3d` and `r1_hv_0` are bit-exact on every contributing value; `vmask` differs, and replacing only NEMO's mask with the candidate mask reproduces the candidate HPG boundary on the registered faces. | Geometry or reciprocal differs first, the mask is exact, or the mask-only replay does not reproduce the candidate boundary. |
| R179-P5 | This is measurement-only. | No `packages/`, card selector, threshold, stabiliser, carried state, sea-ice field or configuration changes. | Any such change lands. |

Failed predictions remain in the receipt. The candidate operators are evaluated
only by the existing pure tendency implementation on the passive completed
stage entry; no callback or in-executable observer is added. The candidate-side
calibration must reproduce its already-recorded completed RHS before any
intermediate is read.

## Controls and terminal rule

The gate must refuse rank overlap, a record bit, source-order movement,
cross-record closure loss, a vacuous target mask, a mask-arm identity failure,
and one planted endpoint ULP. Candidate, oracle and geometry arrays are float64;
execution is production JIT on CPU under the fp64/libm policy.

No production statement lands unless the first operand is cited, its
one-variable implementation arm passes the complete ORCA2/GYRE/DINO/tank gates,
and Decision 96's net-improvement predicate passes. A faithful operand that
still worsens the endpoint stays named and HELD; the compensating partner is
walked next.

ASKED choices: offline attribution of round 178's independent V forcing debt.  
UNASKED choices: empty.

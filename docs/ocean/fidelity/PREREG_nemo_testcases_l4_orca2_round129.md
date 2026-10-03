# Preregistration — ORCA2 round 129 substep-2 U residual

Date: 2026-10-03. Base: `1635f0a4ae3bcb2f23ee8b70d57cba6b29a70c35`.
Every hierarchy-rung-0 number is **independent**. No configuration, forcing,
initial state, carried state, stabilizer, sea-ice selector, or
`unmeasured_features` entry may change.

## Admitted boundary and compiled statements

Round 128 made all eight literal EEN coefficient arrays bit-exact. The last
measured downstream debt was therefore the separate substep-2 U Coriolis row:
68 active values, maximum absolute error `2.9617669311254642e-8`, with its
maximum at global `(j,i)=(147,134)`.

NEMO seeds the external loop with `hu_0*(1+r3u)` and the separately formed
`r1_hu_0/(1+r3u)`
(`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:351-375`), forms the
mid-step flux depth as `hu_0` plus the metric-weighted sea-surface average
(`:513-546`), and after each substep again stores `hu_0+zsshu_a` and its
reciprocal (`:761-779`). The rung-0 record carries those operands at every
substep.

The VORTEX seamount walk independently found the same partial-cell boundary:
NEMO's reference face thickness is the recorded face field, not a value
reconstructed from the T-cell column, and the correct repair is to consume the
card's carried `e3u_0/e3v_0`; their vertical sums are the carried `hu_0/hv_0`.
ORCA2 already carries the exact raw `e3u_0/e3v_0` and `hu_0/hv_0` in its EEN
operand bundle, while the split-explicit face-depth prep currently rebuilds
the reference face columns from bathymetry.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R129-P1 | The admitted two-rank substep record remains passive and complete, and round 128's production coefficients remain bit-exact. | Exactly-once rank coverage, observer passivity, and 0 unequal values in all eight coefficients. | Refuse interpretation and stop at the first moved record or coefficient. |
| R129-P2 | On the round-128 tip the first magnitude debt remains substep-2 U at 68 active cells. | Substep 1 U/V is bit-exact and substep 2 U has 68 unequal values with the registered maximum. | Retain the new census and reconcile it against round 99 before any arm. |
| R129-P3 | The first unequal input to substep-2 Coriolis is the carried barotropic velocity produced at substep 1, downstream of a reference face-depth operand; the now-exact coefficients and a strict application on NEMO's midpoint velocity remain exact. | Source-order rows through substep-1 Coriolis are exact; one of `hu_e`, `hur_e`, `zhup2_e`, or their V counterparts is the first unequal operand feeding the substep-1 exit/substep-2 midpoint. | Stop at the earlier unequal row actually measured; do not apply the depth arm out of order. |
| R129-P4 | Replacing only the reconstructed reference face columns with the card's carried `hu_0/hv_0` closes the named operand and the 68-cell substep-2 U row without moving a previously exact row. | The targeted depth rows and substep-2 U become bit-exact; a one-ULP raw-depth plant fires; reconstructed-depth and wrong-fold controls retain debt. | Hold with the first surviving or newly moved row; no production landing. |
| R129-P5 | The statement executes only on cards selecting NEMO's ssh-average barotropic face depth and carrying raw reference operands; cards that execute it pass their registered gates. | Card census is explicit; ORCA2 rows do not regress; GYRE, DINO, VORTEX, and tank gates satisfy their standing predicates. | Any scope surprise, exact-row loss, earlier debt, or unregistered movement blocks landing. |

## Measurement and landing bar

The baseline and arm reuse the admitted round-96 substep record and the
round-128 literal-EEN builder. They run production JIT on CPU under
fp64/x64/libm. The arm changes one operand family only: the fixed reference
U/V column depths supplied to the existing NEMO ssh-average formula. No new
NEMO run is predicted necessary.

A model statement may land only if source order names this operand as the first
non-bit boundary, the arm closes it and the 68-cell result bit-for-bit, every
plant fires, both ORCA2 ladders lose no exact row or first-debt position, and
all shared-card, citation, and test gates pass. Otherwise the round is HELD at
the first measured boundary.

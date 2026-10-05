# Preregistration — ORCA2 round 150 midpoint V-depth operand split

Date: 2026-10-05. Frozen base: `203d5bc225`.
Claim class: rung-0 results are **independent**. No configuration, initial
state, forcing, sea-ice, stabiliser, carried-state form, threshold, or
`unmeasured_features` choice is authorised.

## Source statement and scope

NEMO extrapolates the midpoint sea surface at compiled
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519`, then constructs
the midpoint V-face depth at `dynspg_ts.f90:542-545` from, in written order,
the reference V depth `hv_0`, one half, reciprocal V area `r1_e1e2v`, the
local and north `e1e2t*zsshp2_e` products, and `ssvmask`.  That depth feeds the
V transport, south difference, and SSH update at `dynspg_ts.f90:568-591`.

Round 149 leaves the first full-domain mismatch at substep 2 `mid_depth_v`:
30 cells, maximum 899 m.  Its descendants differ on 68 cells in `transport_v`
and `continuity_dv` (maximum `155776.5627856178`) and in `after_ssh`
(maximum `0.003203816535399729` m).  All earlier recorded operands are exact.

The round splits the compiled depth statement without changing production:
it scores each already-carried geometry operand against the card's pinned raw
NEMO mesh and replays the written binary operations from the passive midpoint
SSH trace.  Only after one operand is named may a private/default-off arm
substitute that single raw operand in the existing face-depth builder.

## Frozen protocol

1. Reuse the admitted round-95/97 rank-complete substep record and the
   round-146/149 passive production-JIT trace on CPU, fp64/libm, x64.
2. Extend the existing round-146 gate and private test-hook pattern; do not add
   a second solver, callback, public selector, or card field.
3. Require observer passivity, seven post-association arrays, eight EEN
   coefficients, and round 149's 30/68/68/68 control census to remain exact.
4. At substep 2 score `zsshp2_e`, south and north area-SSH products, `hv_0`,
   `r1_e1e2v`, and `ssvmask` separately.  Preserve NEMO's source rounding at
   every binary operation and score the complete recorded domain.
5. If one operand is first non-bit, substitute only that operand and re-score
   `mid_depth_v`, `transport_v`, `continuity_dv`, and `after_ssh`.  Refuse a
   one-ULP plant in a baseline-exact operand and an operand-registry reorder.
6. Do not run either ORCA2 ladder unless the complete chain becomes bit-exact;
   the registered approximately 31 PSU salinity exposure remains a hard veto.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R150-P1 | The passive prerequisites and round-149 control census reproduce. | Observer, post-association rows, and EEN coefficients are exact; substep-2 chain is 30/68/68/68 at the registered maxima. | Any prerequisite or census moves. |
| R150-P2 | The midpoint SSH and both written area-SSH products are bit-exact. | `zsshp2_e`, local product, and associated north product have zero unequal cells. | Any is non-bit. |
| R150-P3 | The first non-bit operand is reference `hv_0`, not reciprocal area or `ssvmask`. | Raw NEMO `hv_0` versus the builder's reference V depth has 30 unequal cells and 899 m maximum; `r1_e1e2v` and `ssvmask` are exact. | Census/max differ, another operand is earlier, or all are exact. |
| R150-P4 | Substituting only raw NEMO `hv_0` closes the depth and its immediate chain. | `mid_depth_v`, `transport_v`, `continuity_dv`, and `after_ssh` become bit-exact with no earlier loss; plants refuse. | Any target bit remains, any earlier exact row moves, or a plant stays green. |
| R150-P5 | A measurement-only result leaves production unchanged. | Hook is private/default-off; no production card/deck/default/carry change remains. | Any production behaviour or configuration changes. |

## Controls and terminal rule

The gate must refuse an operand-registry reorder, one ULP in an exact midpoint
operand, and the existing observer/post-call/passivity/sign controls.  If
R150-P4 is confirmed, the round still stays HELD until the registered salinity
exposure is re-tested under the complete source unit; no ladder runs from a
partial diagnostic arm.  A refuted prediction is retained and the first
non-bit component becomes the next OPEN item.

Pre-implementation search: existing round-129 and round-146 gates already
score the substep record and run private causal arms; round 150 extends those
paths.  No new solver or independent midpoint-depth implementation is added.

ASKED choices: none. UNASKED choices: empty.

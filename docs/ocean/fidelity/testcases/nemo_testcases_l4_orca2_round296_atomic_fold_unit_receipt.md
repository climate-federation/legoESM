# ORCA2 round 296 — atomic raw-depth/raw-mask fold-unit score

Date: 2026-10-16

Status: **HELD**. The complete private vector-form fold unit now carries both
newly named operands: NEMO's stored V reference depth and raw surface V mask.
Both local statements remain bit-exact, and both independent OMT-4 and given
NEMO's entry complete kt=1..7. Both refuse while entering kt=8 on the existing
live-W-thickness guard. The first post-floor greater-than-10x boundary is kt=7
stage 1 in the fold band: salinity maximum error grows from 325.6037473733219
to 12046.543040268563 PSU (36.9975565006522x). No production package, card,
deck, selector, carried state, sea-ice field, or NEMO record changes.

## Provenance and instrument

- Lane base: `2e20b6f3a`; preregistration commit: `13c1942a0`; committed
  classifier: `c783dbd2f`; accepted measurement tree: `c783dbd2f`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round296/`.
- Oracle frames: the admitted round-222 OMT-4 rank-complete kt=1..10 stage
  record. The measurement uses CPU, production JIT, fp64/x64 and libm.
- The reused round-249 runner materialises only the already passive completed
  stage outputs; traced and untraced completed states are array-identical for
  every carried field at kt=1..7 under both labels.
- The private unit preserves every earlier named constituent: raw reference
  U/V depth, raw surface V mask and T-pivot north neighbour, literal unmasked
  and materialised V transport, the seven-array external-mode association,
  and the literal HPG compact-mask boundary. No new executable observer was
  added.

## Executing statements and local closures

NEMO builds the midpoint V depth from stored `hv_0` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:509-520`, and applies
the vector-form V update with raw `ssvmask` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:666-678`. Inside the
atomic unit, the round-293 raw-depth row remains 0/8,589 unequal with maximum
0 m, and the round-295 raw-mask update remains 0/8,589 unequal with maximum
0 m. Thus the two local statements close exactly but do not stabilise the
complete trajectory; they remain private and cannot land separately.

The unit also retains NEMO's literal V transport and seven-array association
at `ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:533-560` and
`:747-756`. These cited statements identify the complete source unit being
scored; this round makes no new attribution inside them.

## Atomic OMT-4 result

Every number in this table is reported separately for **independent OMT-4**
and **given NEMO's entry**. The two labels produce the same valued result.

| label | last complete step | next boundary | first post-floor >10x row | prior max | row max | ratio |
|---|---:|---|---|---:|---:|---:|
| independent | 7 | kt=8 live-W-thickness refusal | kt=7 stage 1, fold band | 325.6037473733219 | 12046.543040268563 | 36.9975565006522 |
| given NEMO's entry | 7 | kt=8 live-W-thickness refusal | kt=7 stage 1, fold band | 325.6037473733219 | 12046.543040268563 | 36.9975565006522 |

At that first growth boundary, both labels have the following fold-band
errors against NEMO:

| field | maximum | RMS | argmax `[j-within-band,i,k]` | unequal / support |
|---|---:|---:|---|---:|
| T | 650.7944752951817 K | 7.010041686458133 K | `[2,48,1]` | 4,164 / 16,200 |
| S | 12046.543040268563 PSU | 130.25416007620328 PSU | `[2,48,1]` | 4,164 / 16,200 |
| u | 25.711291833760093 m/s | 0.4432034496308001 m/s | `[2,45,3]` | 6,256 / 16,200 |
| v | 24.85880449669763 m/s | 0.49912258358863837 m/s | `[1,47,3]` | 8,220 / 16,200 |
| ssh | 2.9264709940293105 m | 0.246167576662627 m | `[2,48]` | 205 / 540 |

All kt=1..7 recorded completed fields remain finite. The kt=8
`raw-mesh e3w_int must contain only finite values > 0` refusal is downstream
damage; the gate attributes the round only to the preceding kt=7 stage-1
growth boundary.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R296-P1 | **CONFIRMED** | Both labels, CPU/fp64/libm, 28 ordered boundaries each, and whole-state passivity are pinned. |
| R296-P2 | **CONFIRMED** | Raw `hv_0` depth and raw `ssvmask` update each remain 0/8,589 unequal. |
| R296-P3 | **REFUTED** | Both labels refuse entering kt=8 instead of completing kt=10. |
| R296-P4 | **CONFIRMED** | First post-floor >10x row is kt=7 stage 1, fold band, 36.9975565006522x. |
| R296-P5 | **NOT REACHED** | Decision-96 landing gates are off-claim because the atomic unit does not complete OMT-4. |
| R296-P6 | **CONFIRMED** | All eight targeted plants refuse with the named invariant. |

Decision 127 changes no row here: the reported valued differences are normal
binary64 numbers, not NEMO-nonzero-subnormal versus candidate-zero pairs.

## Validation

- Accepted evidence SHA256: independent prefix
  `8391f12813444ea039e6e79f47f7cdde20fafa769d40a748805580f0a389beb4`;
  given-entry prefix
  `3a5d2e6c283251679e0424900501c048ebfac91f10510698205de260b3bbaee9`;
  independent refusal
  `33da212f70b1940c801fdf244f74042f223e9770c91292a6949fceeeb9d52768`;
  given-entry refusal
  `a6a106cb3ae062d249d1166435d706b2c7092941862bf4d4eb9ced7d26770bda`;
  classification
  `62e39352d0a4200f68cffe299a95effd32b6671c90d0a7243f9f9ccae4b27247`.
- Label coverage, raw-depth closure, raw-mask closure, passivity, boundary
  order, refusal identity, first-growth selection, and one-bit plants each
  print `STATUS PLANT-FIRED` and exit 2.
- No model file changed, so GYRE, DINO, tank, rung-0, and rung-10 trajectories
  are off-claim; their production configurations are unchanged.
- Independent review: operator, as required for this lane. A nested Codex
  review is unavailable inside this sandbox.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange, the resolved stage mask,
  raw `hv_0`, raw `ssvmask`, and the complete fold unit remain off in
  production.
- Decision 126 remains pending; shipped runoff is unchanged and is not
  re-raised. Sea ice and the shipped card's `unmeasured_features` tuple are
  unchanged.

## OPEN

1. Replay kt=7 stage 1 offline from the finite kt=7 entry, in the compiled
   RK3 operator order, and name the first writer that creates the fold-band
   growth. Use only passive completed states; do not add an in-executable
   observer.
2. If an admitted operand stream cannot split that first writer, write one
   fail-closed, rank-complete acquisition using the established ORCA2 record
   format. Do not walk the kt=8 damage field.
3. Keep the complete unit private and atomic. No constituent is eligible for
   a Decision-96 landing until OMT-4 completes under both labels.

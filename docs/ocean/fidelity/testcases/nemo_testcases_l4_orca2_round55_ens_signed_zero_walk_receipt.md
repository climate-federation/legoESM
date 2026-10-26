# ORCA2 round 55 receipt — OVERFLOW ENS signed-zero operand walk

Date: 2026-09-27  
Base: `39fedb35cbc4158338750449fb4a5d67f07102c6`  
Preregistration: `cf542793b9aa663c12f4b03a9f2e84becab2d8ce`  
Instrument: `9d8eea1a382ee960046cf1ae6ca67ef2e3df7c28`  
Disposition: **HELD**  
ORCA2 claim label: **given NEMO's entry** (Decision 52; held QCO arm only)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Answer

The acquired ENS operands confirm round 52's signed-zero statement exactly.
At all 16,135 active-u bit differences, the NEMO accumulator is negative zero,
the vorticity product is positive zero, and the compiled addition returns
positive zero.  The acquired before and after arrays are bit-exact against the
round-50 after-HPG and after-VOR endpoints on all 16,900 active-u cells.

That statement is not the held QCO change's compensating owner.  A controlled
pair prototype restored round 48's source-ordered QCO assignment and
materialized this positive-zero vorticity addition before flux-form advection.
The OVERFLOW ten-step gate reproduced all five round-48 U violations unchanged,
including the kt=9 maximum of 57.8125 row-scale ULP.  Prediction R55-P5 is
therefore **REFUTED**; the complete model prototype was removed.  No model,
configuration, selector, carried state, stabiliser, or sea-ice field lands.

## Compiled statement and executed record

The record's compiled ENS arm computes and adds the u vorticity product at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:666`.  The round-53
self-describing record is kt=3, stage 2, `Kmm=3`, `kvor=np_CME=5`, fp64, shape
`202 x 3 x 100`, owned origin `(3,3)`, and SHA-256
`aa8051df195d8f0af3818a52d0cb7dedf9843368faa86a2d0ff7dd406f5240b6`.
Its producer stamp is the admitted base `39fedb35c`.

Round 52's first unequal legoESM-local cell `[1,31,0]` maps to Fortran owned
index `(ji,jj,jk)=(34,4,1)`.  Its source-order record is:

| operand | value | fp64 bits |
|---|---:|---:|
| `zwz_prediv` | +0.0 | `0x0000000000000000` |
| `zwz_postdiv` | +0.0 | `0x0000000000000000` |
| `zuav` | +0.0 | `0x0000000000000000` |
| `zwz_pair_u` | +0.0 | `0x0000000000000000` |
| `product_u` | +0.0 | `0x0000000000000000` |
| `rhs_before_u` | -0.0 | `0x8000000000000000` |
| `rhs_after_u` | +0.0 | `0x0000000000000000` |

The committed gate also recomputes the binary64 addition at every active-u
cell.  It finds zero addition mismatches and zero endpoint mismatches.  This is
a bit-pattern statement, not an arithmetic-magnitude error; every changed
cell has absolute difference zero.

## Controls and failed pair

The existing record payload and producer-stamp plants both refuse.  The new
source-order control flips the actual first affected `product_u` from positive
to negative zero; exactly one recomputed addition then disagrees with the
recorded endpoint.  Its independent endpoint plant advances the same recorded
zero by one representable fp64 value and likewise produces exactly one
refusal.  Both exit 2, so neither control perturbs an unscored zero.

The pair prototype was a controlled two-statement retry of the already-held
round-48 QCO statement plus the newly measured downstream statement.  Against
the unchanged round-47 OVERFLOW reference, it returns:

| row | worsening in row-scale ULP | result |
|---|---:|---|
| kt=6 U, cell 668 | 4.783 | violation unchanged |
| kt=7 U, cell 615 | 2.180 | violation unchanged |
| kt=8 U, cell 668 | 4.783 | violation unchanged |
| kt=9 U, cell 450 | 57.8125 | violation unchanged |
| kt=10 U, cell 450 | 57.625 | violation unchanged |

The first-over-bar checkpoint remains kt=2 T/U, but the tank gate permits no
unregistered worsening above two row-scale ULP.  The candidate was therefore
rejected before the GYRE year and ORCA2 ladder gates; running those gates could
not cure the already-binding OVERFLOW refusal.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R55-P1 | **CONFIRMED** | Record admission is `AT_BAR`; header, parent, producer stamp, payload digest, and both existing plants pass their frozen checks. |
| R55-P2 | **CONFIRMED** | The first cell is exactly `-0.0 + +0.0 -> +0.0`, and its recorded after value is bit-equal to round 50. |
| R55-P3 | **CONFIRMED** | The complete active-u census is 16,135/16,900 such chains, with zero before-endpoint, after-endpoint, or recomputed-addition mismatches. |
| R55-P4 | **CONFIRMED** | Product-sign and endpoint plants each introduce exactly one addition refusal and exit 2. |
| R55-P5 | **REFUTED** | The measured pair leaves all five blocking OVERFLOW U rows and the 57.8125-ULP maximum unchanged; the prototype is removed. |

## Verification

- Focused round-50/52/53/55 controls: **22 passed**.
- Shared-card battery: **170 passed**, 9 warnings, in 390.23 s.
- The held pair's OVERFLOW trajectory completed all 50 rows through kt=10;
  the oracle-relative gate returns **FAIL**, five violations, and the same
  first-over-bar checkpoint as round 48.
- Separate read-only `codex exec` claim review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- Default citation gate: **274 citations, PASS**, zero failures and zero
  unmapped; this receipt: **1 citation, PASS**.  Shifting the accumulator
  citation by two lines fires `SYMBOL-NOT-AT-LINE` and exits nonzero.
- `tests/ocean/fidelity -n 12` collected 1,957 items, emitted the same five
  registered failures, reached 99% with 1,937 passed nodes, and reproduced the
  registered xdist tail stall before interruption.  Serial rerun preserves all
  five signatures: round-129 stale certification, round-51 private trace
  registry, SI3 scalar-math provenance, three worktree-stamp offenders, and
  the `hires_lane_surface` case-board omission.
- The final `packages/` diff is empty.  Therefore the GYRE year/trajectory,
  ORCA2 ladder, DINO, tank, and generic-card landing gates are not claimed as
  candidate passes.  Sea ice and the card's `unmeasured_features` tuple remain
  unchanged at `STOP_SELECTOR_GAP`.

## OPEN

1. Keep the source-ordered QCO statement held.  The ENS signed-zero statement
   is measured and real but is excluded as its compensating owner.
2. Continue the kt=3 stage-2 compiled program from the admitted after-VOR
   endpoint through `dynadv_up3` to the existing after-ADV endpoint.  Walk its
   transport, face-value, flux-divergence, and accumulator statements one
   variable at a time before another pair retry.
3. After the OVERFLOW pair closes, return to ORCA2's whole-card kt=1 stage-1 T
   owner, Decision 52's independent-start trajectory, and the round-20
   slow-forcing walk.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: admit the ENS operand record, walk the signed-zero statement, and land
only if the full shared gate passes.  
UNASKED: none.

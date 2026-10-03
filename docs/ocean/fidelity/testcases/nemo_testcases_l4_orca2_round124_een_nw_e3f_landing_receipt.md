# ORCA2 round 124 — northwest EEN northern thickness landing

Date: 2026-10-03. Measurement tip `ee9c4ec59`. Every hierarchy-rung-0
number is **independent** because rung 0 starts from NEMO's own from-rest
state. The shipped rung-7 ladder is **given NEMO's recorded entry** under
Decision 52. No configuration, forcing, initial state, carried state,
stabilizer, sea-ice selector, or `unmeasured_features` entry changed.

## Verdict

**LANDED.** Northwest fraction 3's separate northern `e3f_0vor` boundary is
the same compiled F-grid association round 123 measured for northeast:
native source row `j=145`, F-origin permutation `179-i`, sign `+1`, followed
by northwest's recorded `ji-1` offset. The controlled arm makes 521/521
thickness and denominator differences and 1,431/1,431 quotient differences
bit-exact. Northeast remains bit-exact. The next northwest boundary is its
frozen mask: 1,154/1,154 cells, first `(j,i,k)=(147,30,0)`.

NEMO constructs `e3f_0vor`, applies the F-grid lateral boundary, and restores
fully dry entries in that order
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937`). Northwest
fraction 3 consumes the associated northern value at `(ji-1,jj+1)`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1328`). The
shared literal EEN builder now applies the same association to both 2-D
`ff_f` and 3-D `e3f_0vor` (`barotropic_latlon_cgrid.py:920-941`) before the
northern division (`barotropic_latlon_cgrid.py:1036-1039`).

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R124-P1: NW uses row 145, F permutation `179-i`, sign `+1`, then `ji-1` | **CONFIRMED**, 521/521 thickness differences become 0/0 |
| R124-P2: close thickness, denominator, and all 1,431 quotient cells | **CONFIRMED**, each becomes 0/0 bit/magnitude unequal |
| R124-P3: frozen mask is next | **CONFIRMED**, 1,154/1,154; `r3f` stays 0/0 |
| R124-P4: NE stays exact and isolated | **CONFIRMED**, thickness/denominator/quotient remain 0/0; mask remains 1,160 |
| R124-P5: literal-EEN scope stays fixed | **CONFIRMED**: ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC true; GYRE and both tanks false |

The oracle-bit, candidate-bit, wrong-row, northeast-isolation, and scope-route
plants each exit 2 with `STATUS PLANT-FIRED`. The wrong-row control leaves 215
northwest thickness magnitudes unequal. The final clean measurement reports
`MEASURED_R124_EEN_NW_E3F_ASSOCIATION` under production JIT on CPU with
fp64/x64/libm.

## Source-ordered measurement

Counts are bit-unequal / magnitude-unequal over 425,872 executed level-cells.

| path and item | before | after |
|---|---:|---:|
| NW fraction 3 `e3f_0vor` | 521 / 521 | 0 / 0 |
| NW fraction 3 `r3f` | 0 / 0 | 0 / 0 |
| NW fraction 3 `fe3mask` | 1,154 / 1,154 | 1,154 / 1,154 |
| NW fraction 3 denominator | 521 / 521 | 0 / 0 |
| NW fraction 3 quotient | 1,431 / 1,431 | 0 / 0 |
| NW complete sum | 1,431 / 1,431 | 0 / 0 |
| NE fraction 1 thickness / denominator / quotient | 0 / 0 each | 0 / 0 each |

The admitted two-rank record remains passive: exactly-once global coverage,
all 20 restart shards and 10 inherited streams byte-identical, and record
SHA-256 values unchanged from rounds 120–123.

## Landing gates

- ORCA2 rung 0 (**independent**): complete kt=1..10 ladder passed; 0/200 rows
  moved versus round 122, no exact row left the bar, and the first debt remains
  kt=1 stage-1 T.
- ORCA2 rung 7 (**given NEMO's entry**): complete kt=1..10 ladder passed;
  0/200 rows moved versus round 122 and the first debt is unchanged.
- GYRE: 0/70 ten-step rows moved; all 210 residual arrays are array-equal and
  retain SHA-256
  `377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
  All 30 daily snapshots are byte-identical; day 30 remains
  `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`.
- DINO: the fp64 CPU month completed 960 steps. Day-30 wet 3-D T RMS remains
  `2.053801168e-03 K`, below the `2.244317642e-03 K` bar, and its model
  snapshot is byte-identical with SHA-256
  `83b9ba659ac3b00c873c6999a225cc6c4215901f484c121b550f40fc5292d78c`.
- LOCK_EXCHANGE and OVERFLOW do not execute this literal-EEN statement. Their
  gates retain first debt at kt=4 U and kt=2 T/U respectively, with zero
  oracle-relative worsening.

## Instruments, citations, tests, and review

The first shipped-card ten-step attempt produced no artifact when its process
ended at the execution session boundary; it is **incomplete, not evidence**.
The later complete gate used the same CPU/fp64/JIT protocol. A proposed
persistent-cache workaround for OVERFLOW emitted a CPU-feature mismatch and
is **discarded**; the uncached complete gate is the result above.

The citation map and the round-122 prose citations were re-anchored in the
model commit. The default citation gate passes with zero failed or unmapped
citations; shifting `barotropic_latlon_cgrid.py:920-941` makes it fail with
`SYMBOL-NOT-AT-LINE`.

Focused tests pass 37/37. The required `tests/ocean/fidelity -n 12` invocation
reached 98% and reproduced the registered xdist-controller tail stall after
the count of real pytest processes reached zero. It recorded 2,348 passed and
six displayed, registered pre-existing failures: SI3 scalar-math provenance,
round-129 spread-floor stamp, round-51 private-arm scope, round-35 escape
scope, worktree stamp, and recipe case-board. No round-124 test failed. The
battery is **incomplete, not PASS**.

Separate `codex exec --sandbox read-only` claim and final-diff reviews were
attempted. Verdict: **independent review unavailable in-sandbox** (`failed to
initialize in-process app-server client: Read-only file system`). A second
external reviewer was not callable in this sandbox.

## OPEN

1. Walk the northern frozen masks separately in source order: 1,160 NE cells,
   then 1,154 NW cells, before the accumulator/final-scale cancelling pair.
2. Resume the distinct 68-cell substep-2 U residual only after coefficient
   construction closes.
3. Continue the independent hierarchy/card month program after the rung-0
   source walk. This receipt makes no independent ORCA2 month claim.

## UNVERIFIED

- The source association and landing eligibility of the two northern frozen
  mask boundaries.
- The owner of the distinct 68-cell substep-2 U residual.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

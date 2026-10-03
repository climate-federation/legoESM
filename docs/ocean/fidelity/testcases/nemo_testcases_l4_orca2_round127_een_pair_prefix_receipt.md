# ORCA2 round 127 — northern EEN pair prefix walk

Date: 2026-10-03. Measurement tip `4ce7522d9`. Every hierarchy-rung-0
number is **independent** because rung 0 starts from NEMO's own from-rest
state. No model physics, card field, configuration value, forcing, initial
state, carried state, stabilizer, sea-ice selector, or `unmeasured_features`
entry changed.

## Verdict

**HELD.** The accumulator/final-scale pair is not yet the first non-bit
boundary. After rounds 121--126 make all three recorded northern quotient
fractions bit-exact, the next source-ordered operand is the neighboring U-face
thickness: 95/95 bit/magnitude differences for northwest and 91/91 for
northeast, all at global `j=147`. The neighboring U mask is also non-bit
(1,270/1,270 and 1,283/1,283), so the later term, recurrence, and scale cannot
yet be interpreted as independent owners.

The executed source consumes the northern U thickness and mask before each
ordinary accumulator addition
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1339-1348`).
Only after the vertical loop does it record the accumulators, form the
northern `e2u` scales, and write the final coefficients
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1360-1367`).
Source order therefore requires walking the U-grid northern thickness and mask
association before returning to the measured accumulator/scale pair.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R127-P1: admitted records remain passive and complete | **CONFIRMED**, round-120 admission passes, both fraction and recurrence records retain exactly-once two-rank coverage, and inherited streams plus terminal restarts retain identity |
| R127-P2: corrected quotient operands make NW/NE recurrence terms bit-exact | **REFUTED**, the first non-bit item is neighboring U thickness at 95 NW and 91 NE cells; the masks then differ at 1,270/1,283 cells |
| R127-P3: IEEE zero addition closes all four V accumulators | **NOT REACHED**, an earlier magnitude operand is non-bit |
| R127-P4: north-fold association closes both V scales | **NOT REACHED** in source order |
| R127-P5: complete pair closes all eight final coefficients | **NOT REACHED** in source order |
| R127-P6: literal-EEN card scope remains fixed | **CONFIRMED**, ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC execute; GYRE and both tanks do not |
| R127-P7: complete pair passes the trajectory gates | **NOT REACHED**, no production candidate exists |

## Source-ordered measurement

Counts are bit-unequal / magnitude-unequal over 425,872 executed level-cells.
The instrument runs production JIT on CPU under fp64/x64/libm.

| item | NW | NE |
|---|---:|---:|
| completed three-fraction quotient | 0 / 0 | 0 / 0 |
| local V thickness | 0 / 0 | 0 / 0 |
| neighboring northern U thickness | 95 / 95 | 91 / 91 |
| neighboring northern U mask | 1,270 / 1,270 | 1,283 / 1,283 |
| stored product | 1,270 / 1,270 | 1,283 / 1,283 |
| accumulator before | 1,231 / 1,231 | 1,249 / 1,249 |
| accumulator after | 1,298 / 1,298 | 1,316 / 1,316 |

The first NW and NE unequal cells are both at `(j,i,k)=(147,31,3)`.
legoESM supplies `10.00542762603709 m`; NEMO records
`5.99678419243628 m`. This is a real magnitude boundary, not a signed-zero
or final-scale effect. The term/recurrence counts are reported only as
downstream consequences, not attributed findings.

The oracle-bit control changes the locked NW thickness census from 95 to 96;
the candidate-bit control moves the first boundary back to the quotient; and
the scope-route control makes GYRE execute the statement. Each exits nonzero
with `STATUS PLANT-FIRED`. Evidence is under
`orca2_rounds/round127/{een_pair_prefix.json,een_pair_prefix.log,plant_*.log}`.

## Trajectory scope

No `packages/` or card file changed. The independent rung-0 ladder, the
given-entry rung-7 ladder, GYRE, DINO, tanks, and generic cards therefore
cannot move and are not represented as reruns. This receipt makes no
month-scale claim.

## Tests, citations, and review

The pre-implementation search found and reused the round-119 recurrence,
round-121 fraction, round-126 association, and cumulative citation machinery;
no duplicate reader or numerical path was created. The round-119--127 plus
cumulative-citation focused battery passes 50/50.

The default citation gate passes 274 citations and this receipt passes its
compiled-source citations with zero failures or unmapped entries. Shifting
`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1339-1348`
makes the gate fail with `SYMBOL-NOT-AT-LINE` and exit 1.

The required `tests/ocean/fidelity -n 12` invocation collects 2,384 tests and
reaches 95%; 2,268 tests pass before reproducing the registered xdist-controller
tail stall after the count of real pytest processes reaches zero. It is
interrupted after the grace period and is **incomplete, not PASS**. Its five
displayed failures are the registered pre-existing SI3 scalar-math provenance,
round-51 private-arm scope, round-35 escape scope, worktree-stamp, and recipe
case-board rows. The five-ID serial replay reproduces exactly those failures;
no round-127 test fails.

Separate `codex exec --sandbox read-only` review was attempted. Verdict:
**independent review unavailable in-sandbox** (`failed to initialize
in-process app-server client: Read-only file system`). A second external
reviewer was not callable in this sandbox.

## OPEN

1. Walk the compiled U-grid northern association separately for the 95/91
   live-thickness cells and the 1,270/1,283 mask cells; preserve the exact
   quotient as an isolation control.
2. Return to the northern accumulator/final-scale pair only after the stored
   products are bit-exact.
3. Keep the distinct 68-cell substep-2 U residual separate until coefficient
   construction closes. The independent rung-0 month remains open.

## UNVERIFIED

- The exact source row/permutation independently required by the neighboring
  U thickness and mask operands.
- The first unequal statement after those operands become exact.
- The owner of the later 68-cell substep-2 U residual.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

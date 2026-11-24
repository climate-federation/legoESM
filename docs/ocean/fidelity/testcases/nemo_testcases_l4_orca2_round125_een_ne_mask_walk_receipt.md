# ORCA2 round 125 — northeast EEN northern frozen-mask walk

Date: 2026-10-03. Measurement tip `dd0c460ad`. Every hierarchy-rung-0
number is **independent** because rung 0 starts from NEMO's own from-rest
state. No model physics, card field, configuration value, forcing, initial
state, carried state, stabilizer, sea-ice selector, or `unmeasured_features`
entry changed.

## Verdict

**HELD.** Northeast fraction 1's 1,160-cell northern frozen-mask boundary is
NEMO's ordinary F-grid north-fold association: native source row `j=145`,
F-origin permutation `179-i`, sign `+1`. The one-variable arm makes all
1,160/1,160 bit/magnitude differences exact. Its already-exact thickness,
`r3f`, denominator, quotient, partial sum, and complete sum remain 0/0.

Northwest remains independently non-bit at its separate frozen-mask boundary:
1,154/1,154 cells, first `(j,i,k)=(147,30,0)`. No partial production change
lands before that boundary is measured.

## Compiled statement and execution scope

The executed source applies the ordinary F-grid lateral boundary to `fmask`
and only then freezes `fe3mask = fmask`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258`). The
T-pivot F-point branch fills the northern halo from the row two places below it
under `nn_hls=2`, with sign `+1`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:722-747`). Northeast
fraction 1 reads the associated mask at `(ji,jj+1)`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1328`).

Resolved cards executing the literal statement are ORCA2, both NEMO-faithful
DINO recipes, VORTEX, and VORTEX_VEC. GYRE, LOCK_EXCHANGE, and OVERFLOW do not
execute it. The scope-route plant makes this check refuse.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R125-P1: NE uses row 145, F permutation `179-i`, sign `+1` | **CONFIRMED**, every recorded mask bit agrees |
| R125-P2: close exactly 1,160 NE mask cells | **CONFIRMED**, 1,160/1,160 becomes 0/0 |
| R125-P3: already-exact descendants remain exact | **CONFIRMED**, thickness, `r3f`, denominator, quotient, partial, and sum remain 0/0 |
| R125-P4: NW remains isolated | **CONFIRMED**, mask remains 1,154/1,154 while thickness, denominator, quotient, and sum remain 0/0 |
| R125-P5: literal-EEN card scope stays fixed | **CONFIRMED**, exact registered execution set above |

## Source-ordered measurement

Counts are bit-unequal / magnitude-unequal over 425,872 executed level-cells.

| path and item | before | controlled arm |
|---|---:|---:|
| NE fraction 1 `e3f_0vor` | 0 / 0 | 0 / 0 |
| NE fraction 1 `r3f` | 0 / 0 | 0 / 0 |
| NE fraction 1 `fe3mask` | 1,160 / 1,160 | 0 / 0 |
| NE fraction 1 denominator | 0 / 0 | 0 / 0 |
| NE fraction 1 quotient | 0 / 0 | 0 / 0 |
| NE partial and complete sums | 0 / 0 each | 0 / 0 each |
| NW fraction 3 `fe3mask` isolation | 1,154 / 1,154 | 1,154 / 1,154 |

The gate runs production JIT on CPU under fp64/x64/libm. The admitted record
retains exactly-once two-rank coverage, its two payload SHA-256 values remain
`94024493b7adc9bf11be2778546df7e3540d1713e6d95e03b25367f7fa87ac9d`
and `0024d4431c4df177062538f03b95082574d6885a642ade60170b65e5b4d41a6e`,
and all inherited streams plus 20 terminal restarts remain byte-identical.

All five controls fire with exit 2 and `STATUS PLANT-FIRED`. Oracle-bit and
candidate-bit each leave one mask cell unequal; the wrong-row control leaves
143 mask cells unequal; northwest-isolation and scope-route refuse directly.

## Trajectory scope

No `packages/` or card file changed. Therefore the ORCA2 rung-0 independent
ladder, the rung-7 given-entry ladder, GYRE, DINO, the tanks, and generic cards
cannot move in this round; their trajectory gates are not represented as
reruns. This receipt makes no month-scale claim.

## Tests, citations, and review

The round-120–125 plus cumulative-citation focused battery passes 40/40,
including a nonzero-`r3f` control that distinguishes the correct mask from the
wrong one. The default citation gate passes 274 citations and this receipt's
gate passes all three citations with zero failures or unmapped entries;
shifting the compiled `dommsk` citation makes the gate fail with
`SYMBOL-NOT-AT-LINE`.

The required `tests/ocean/fidelity -n 12` invocation collected 2,381 tests and
reached 98%; 2,333 tests passed before reproducing the registered
xdist-controller tail stall after the count of real pytest processes reached
zero. It was interrupted after the grace period and is **incomplete, not
PASS**. Its six displayed failures are the registered pre-existing SI3 scalar-
math provenance, round-129 spread-floor stamp, round-51 private-arm scope,
round-35 escape scope, worktree-stamp, and recipe case-board rows. No
round-125 test failed.

Separate `codex exec --sandbox read-only` review was attempted. Verdict:
**independent review unavailable in-sandbox** (`failed to initialize
in-process app-server client: Read-only file system`). A second external
reviewer was not callable in this sandbox.

## OPEN

1. Walk northwest fraction 3's separate 1,154-cell northern frozen-mask
   boundary with northeast preserved as an isolation control.
2. If both paths are exact, land the common compiled association under the
   complete ORCA2/GYRE/DINO/tank/card/citation gates.
3. Then walk the northern accumulator/final-scale cancelling pair before the
   distinct 68-cell substep-2 U residual. The rung-0 independent month remains
   open.

## UNVERIFIED

- Whether northwest's 1,154-cell mask boundary takes the same association and
  is eligible to land with northeast.
- The owner of the later 68-cell substep-2 U residual.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

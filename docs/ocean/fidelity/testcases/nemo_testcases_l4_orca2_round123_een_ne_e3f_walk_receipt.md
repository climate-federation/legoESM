# ORCA2 round 123 — northeast EEN northern thickness walk

Date: 2026-10-03. Measurement tip `2eb957c6d`. Every ocean number below is
**independent** because hierarchy rung 0 starts from NEMO's own from-rest
state. No model physics, card field, configuration value, carried state,
stabilizer, sea-ice selector, or `unmeasured_features` entry changed.

## Verdict

**HELD.** Northeast fraction 1's first remaining operand is NEMO's northern
`e3f_0vor` association. Selecting native source row `j=145` under the F-origin
permutation `179-i`, sign `+1`, makes its thickness, denominator, and quotient
bit-exact: respectively 514/514, 514/514, and 1,431/1,431 bit/magnitude
differences become 0/0. Its next source-ordered boundary is the recorded frozen
mask at 1,160 cells, first `(j,i,k)=(147,29,0)`.

This round deliberately changes only the northeast measurement arm. Northwest
fraction 3 remains at its separate 521-cell northern thickness boundary, first
`(147,29,0)`. No partial production association lands before that northwest
operand is walked independently.

## Compiled statement and source order

The executed `nn_e3f_typ=0` branch forms `e3f_0vor` from the four masked T-cell
thicknesses divided by four, applies the F-grid lateral boundary, and then
restores zero entries from mesh `e3f_3d`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937`). The NE
fraction reads `(ji,jj+1)` before the two current-row fractions; NW instead
reads its northern value in fraction 3
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1328`).

The already-admitted rank-complete record remains passive: all 20 kt=1..10
restart shards and all 10 inherited streams are byte-identical, its two rank
files retain SHA-256 `94024493b7adc9bf11be2778546df7e3540d1713e6d95e03b25367f7fa87ac9d`
and `0024d4431c4df177062538f03b95082574d6885a642ade60170b65e5b4d41a6e`,
and its arithmetic replay stays exact.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R123-P1: hold the already-folded northern row | **REFUTED and retained**: it reduces 514/514 to 228/228, not zero |
| R123-P1 falsifier: score the preceding-row F permutation | **OBSERVED post-falsifier**: row 145, reverse `179-i`, sign `+1` is 0/0 |
| R123-P2: close 514 NE thickness and denominator cells | **CONFIRMED** under the P1 discriminator: both become 0/0 |
| R123-P3: frozen mask is next | **CONFIRMED**, 1,160/1,160; `r3f` remains 0/0 |
| R123-P4: northwest stays isolated | **CONFIRMED**, its 521/521 thickness boundary is unchanged |
| R123-P5: literal-EEN card scope stays fixed | **CONFIRMED**: ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC true; GYRE and both tanks false |

The initial R123-P1 run refused mechanically at 228/228. A later import defect
in the discriminator crashed before producing a number; it is an instrument
failure, not evidence. The corrected content-committed gate then produced the
numbers above.

## Source-ordered measurement

Counts are bit-unequal / magnitude-unequal over 425,872 executed level-cells.

| path and item | before | controlled arm | first remaining |
|---|---:|---:|---|
| NE fraction 1 `e3f_0vor` | 514 / 514 | 0 / 0 | — |
| NE fraction 1 `r3f` | 0 / 0 | 0 / 0 | — |
| NE fraction 1 `fe3mask` | 1,160 / 1,160 | 1,160 / 1,160 | `(147,29,0)` |
| NE fraction 1 denominator | 514 / 514 | 0 / 0 | — |
| NE fraction 1 quotient | 1,431 / 1,431 | 0 / 0 | — |
| NW fraction 3 `e3f_0vor` | 521 / 521 | 521 / 521 | `(147,29,0)` |

The gate runs production JIT on CPU under fp64/x64/libm. Its oracle-bit,
candidate-bit, cyclic-wrap, northwest-isolation, and scope-route controls each
exit 2 with `STATUS PLANT-FIRED`.

## Trajectory scope

No `packages/` or card file changed. Therefore neither ORCA2 ladder, GYRE,
DINO, the tanks, nor generic cards can move in this round; trajectory gates are
not represented as reruns. This receipt makes no month-scale claim.

## Tests, citations, and review

The focused round-120/121/122/123 and cumulative-citation battery passes
33/33. The required `tests/ocean/fidelity -n 12` invocation collected 2,374
tests and reached 97%; 2,291 tests passed before reproducing the registered
xdist-controller tail stall after the count of real pytest processes reached
zero. It was interrupted after the grace period and is **incomplete, not
PASS**. Its six displayed failures are the registered pre-existing SI3 scalar-
math provenance, round-129 stamp, round-51 private-arm scope, round-35 escape
scope, worktree-stamp, and recipe case-board rows. No round-123 test failed.

The default citation gate and this receipt's citation gate both pass with zero
unmapped or failed entries; shifting the compiled `dynvor` citation exits 1.
Separate `codex exec --sandbox read-only` review was attempted. Verdict:
**independent review unavailable in-sandbox** (`failed to initialize
in-process app-server client: Read-only file system`). A second external
reviewer was not available in this sandbox; the measurement remains guarded by
the five independent runtime plants and the focused tests above.

## OPEN

1. Walk northwest fraction 3's separate 521-cell `e3f_0vor` boundary against
   the same compiled association; preserve northeast as an isolation control.
2. Then walk the northern frozen masks separately: 1,160 NE cells followed by
   1,154 NW cells, before the northern accumulator/final-scale cancelling pair.
3. Resume the distinct 68-cell substep-2 U residual only after coefficient
   construction closes. The rung-0 independent month remains open.

## UNVERIFIED

- Whether the northwest 521-cell thickness boundary takes the same source-row
  association and becomes bit-exact.
- Whether the later complete northern accumulator/final-scale pair passes the
  trajectory landing gates.
- The owner of the distinct 68-cell substep-2 U residual.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

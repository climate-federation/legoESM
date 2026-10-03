# ORCA2 round 126 — northern EEN frozen-mask landing

Date: 2026-10-03. Measurement tip `5dade0773`. Every hierarchy-rung-0
number is **independent** because rung 0 starts from NEMO's own from-rest
state. Every shipped rung-7 number is **given NEMO's recorded entry** under
Decision 52. No configuration, forcing, initial state, carried state,
stabilizer, sea-ice selector, or `unmeasured_features` entry changed.

## Verdict

**LANDED.** Northwest fraction 3's 1,154-cell frozen-mask boundary is the same
ordinary F-grid north-fold association measured independently for northeast:
native row `j=145`, F-origin permutation `179-i`, sign `+1`, followed by the
northwest operand's `ji-1` offset. The controlled arm makes all 1,154/1,154
bit/magnitude differences exact; northeast remains exact at 0/0. The common
production statement now preserves NEMO's associated northern `fe3mask`
operand before forming the literal EEN divisor.

NEMO applies the F-grid lateral boundary and then freezes `fe3mask = fmask`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258`). The
T-pivot F branch uses the row two places below the northern halo, its F-origin
permutation, and sign `+1`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:722-747`). Northwest
fraction 3 consumes `(ji-1,jj+1)`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1328`). The
shared literal builder transcribes the frozen-mask association at
`barotropic_latlon_cgrid.py:1004-1010`.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R126-P1: NW uses row 145, F permutation `179-i`, sign `+1`, then `ji-1` | **CONFIRMED**, every recorded mask bit agrees |
| R126-P2: close exactly 1,154 NW mask cells | **CONFIRMED**, 1,154/1,154 becomes 0/0 |
| R126-P3: already-exact descendants remain exact at the recorded entry | **CONFIRMED**, thickness, denominator, quotient, partial, and sum remain 0/0 |
| R126-P4: NE remains exact and isolated | **CONFIRMED**, all 20 source-ordered NE rows remain 0/0 |
| R126-P5: literal-EEN card scope stays fixed | **CONFIRMED**, ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC true; GYRE and both tanks false |
| R126-P6: production trajectories move zero rows | **REFUTED**, rung 0 moves 175/200 and rung 7 moves 195/200; the recorded entry has zero `r3f` on the corrected support, but later evolving sea surface makes that support active |

The oracle-bit, candidate-bit, wrong-row, northeast-isolation, and scope-route
controls each exit 2 with `STATUS PLANT-FIRED`. The wrong-row discriminator
leaves 137 northwest mask magnitudes unequal. The clean gate reports
`MEASURED_R126_EEN_NW_MASK_ASSOCIATION` under production JIT on CPU with
fp64/x64/libm.

## Source-ordered operand measurement

Counts are bit-unequal / magnitude-unequal over 425,872 executed level-cells.

| path and item | before | controlled arm |
|---|---:|---:|
| NW fraction 3 `e3f_0vor` | 0 / 0 | 0 / 0 |
| NW fraction 3 `r3f` | 0 / 0 | 0 / 0 |
| NW fraction 3 `fe3mask` | 1,154 / 1,154 | 0 / 0 |
| NW fraction 3 denominator | 0 / 0 | 0 / 0 |
| NW fraction 3 quotient and complete sum | 0 / 0 each | 0 / 0 each |
| NE complete source-ordered path | 0 / 0 each | 0 / 0 each |

The admitted two-rank record remains passive: exactly-once global coverage,
all 20 terminal restarts and ten inherited streams byte-identical, and the two
payload hashes remain
`94024493b7adc9bf11be2778546df7e3540d1713e6d95e03b25367f7fa87ac9d`
and `0024d4431c4df177062538f03b95082574d6885a642ade60170b65e5b4d41a6e`.

## Trajectory landing gates

- ORCA2 rung 0 (**independent**): all 200 rows complete. 175 move versus round
  124 (RMS: 65 toward, 110 away); no bit-identical row leaves the bar and the
  first debt remains kt=1 stage-1 T.
- ORCA2 rung 7 (**given NEMO's recorded entry**): all 200 rows complete. 195
  move (RMS: 111 toward, 84 away); no bit-identical row leaves the bar and the
  first debt remains kt=1 stage-1 T. Every moved row is registered in
  `round126/orca_ladder_compare.json`.
- GYRE: 0/70 ten-step rows move and the residual archive remains SHA-256
  `377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
  All 30 daily snapshots are byte-identical; day 30 remains
  `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`.
- DINO: the fp64 CPU month completes 960 steps. Day-30 wet 3-D T RMS remains
  `2.053801168e-03 K`, below the `2.244317642e-03 K` bar, and its model
  snapshot remains byte-identical with SHA-256
  `83b9ba659ac3b00c873c6999a225cc6c4215901f484c121b550f40fc5292d78c`.
- LOCK_EXCHANGE and OVERFLOW do not execute the literal statement. Their
  matching stop-at-first-debt gates move 0 rows and retain kt=4 U and kt=2 T/U
  respectively.

The first DINO attempts ended before step 1 because quiet jobs were terminated
with their parent execution sessions; they are incomplete, not evidence. A
one-day cache warm is likewise not evidence. The final heartbeat-protected
960-step run above is the certified result. An initial tank comparison with
extra post-debt rows was discarded as a protocol mismatch; the matching
stop-at-first-debt comparisons are the results above.

## Tests, citations, and review

The round-122–126 plus literal-builder focused battery passes 42/42. Citation,
full fidelity, and final review results are recorded in the final validation
commit.

## OPEN

1. Walk the northern accumulator/final-scale cancelling pair now that every
   recorded northern quotient operand is bit-exact.
2. Then resume the distinct 68-cell substep-2 U residual.
3. Continue the independent hierarchy/card month program after the rung-0
   source walk. This receipt makes no independent ORCA2 month claim.

## UNVERIFIED

- The first unequal statement in the northern accumulator/final-scale pair.
- The owner of the distinct 68-cell substep-2 U residual.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

# ORCA2 round 122 — northern EEN F-grid association landing

**Status: LANDED.** On hierarchy rung 0, whose numbers are **independent**
because it starts from NEMO's own from-rest state, the first northern-V EEN
operand difference was the northern `ff_f` association: 1,431
magnitude-unequal cells in each of northeast fraction 1 and northwest fraction
3, all on global row `j=147`. NEMO reads `ff_f` as an F-point field with sign
`+1` and `jpfillcopy`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/domhgr.f90:233-237`). Its executed
T-pivot/F-point boundary branch fills the northern halo from the row two places
below it under `nn_hls=2`, with the F permutation and sign `+1`
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:722-747`). The literal
EEN builder now transcribes that association (`barotropic_latlon_cgrid.py:920-941`)
before forming the northern quotient (`barotropic_latlon_cgrid.py:1040-1043`).

No configuration, forcing, entry state, carried state, stabiliser, sea-ice
selector, or `unmeasured_features` entry changed.

## Frozen prediction correction and disposition

R122-P1 predicted source row 146, reversed as `179-i`. **REFUTED and
retained.** The compiled field includes two halo rows: the measured native
target row 147 is sourced from native row 145, reversed `179-i`, sign `+1`.
The first draft of the gate also changed the stored northern row instead of the
actual `jj+1` EEN operand; it refused rather than supporting a claim. The
instrument was corrected and recommitted before the production edit.

| prediction | disposition |
|---|---|
| R122-P1: row 146, reverse `179-i`, sign `+1` | **REFUTED** as above; row 145 is bit-exact and bijective |
| R122-P2: one association closes both 1,431-cell boundaries | **CONFIRMED**, both become 0/0 bit/magnitude unequal |
| R122-P3: `e3f_0vor` is next | **CONFIRMED**, 514 NE and 521 NW cells |
| R122-P4: resolved scope is literal EEN cards only | **CONFIRMED**: ORCA2, both DINO recipes, VORTEX, VORTEX_VEC true; GYRE and both tanks false |
| R122-P5: no ORCA2 exact row leaves the bar; shared gates pass | **CONFIRMED** |

The oracle-bit, candidate-bit, permutation-shift, and scope-route plants each
exit 2. The final clean measurement reports
`PASS_R122_EEN_NORTH_FF_WALK` under production JIT, CPU fp64/x64/libm.

## Source-ordered measurement

The admitted two-rank record retains exactly-once owned coverage and its
passive restart/inherited-stream checks. All counts below are **independent**.

| path and operand | before bit/magnitude unequal | after bit/magnitude unequal | first remaining cell |
|---|---:|---:|---|
| NE fraction 1 `ff_f` | 1,431 / 1,431 | 0 / 0 | — |
| NE fraction 1 `e3f_0vor` | 514 / 514 | 514 / 514 | `(j,i,k)=(147,28,0)` |
| NW fraction 3 `ff_f` | 1,431 / 1,431 | 0 / 0 | — |
| NW fraction 3 `e3f_0vor` | 521 / 521 | 521 / 521 | `(147,29,0)` |

The mapping covers all 180 longitudes exactly once: target row 147, source row
145, source indices 179 through 0, sign `+1`. The walk stops before the later
mask differences (1,160 NE; 1,154 NW), sums, final scaling, and substep output.

## Landing gates

The rung-0 ladder is **independent**; the shipped rung-7 ladder is **given
NEMO's recorded entry** under Decision 52. Their complete comparator reports
0/200 moved rows on each ladder, no exact-row loss, and unchanged first debt at
kt=1 stage-1 T.

- GYRE: 0/70 ten-step rows moved, first debt remains kt=3, and the residual
  SHA-256 remains
  `377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
  All 30 daily snapshots are byte-identical; day 30 remains
  `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`.
- DINO: the fp64 CPU month completed 960 steps. Day-30 wet 3-D T RMS is
  `2.053801168e-03 K`, below the `2.244317642e-03 K` bar, with snapshot SHA-256
  `83b9ba659ac3b00c873c6999a225cc6c4215901f484c121b550f40fc5292d78c`.
- LOCK_EXCHANGE and OVERFLOW do not execute the statement. Their re-run first
  debts remain kt=4 U and kt=2 T/U respectively. VORTEX/VORTEX_VEC execute the
  literal EEN builder; the focused builder and null-mode controls cover them.

## Tests, citations, and review

The required `tests/ocean/fidelity -n 12` battery was run once. It reached 99%
and reproduced the registered xdist-controller tail stall. Before interruption
it emitted the known pre-existing SI3 provenance, round-129 stamp, round-51
private-arm, round-35 scope, worktree-stamp, and recipe case-board failures;
the three citation failures were caused by this round's initial occurrence
anchor and were repaired before the final citation/focused gates. No round-122
test failed.

The final focused battery passes **74/74** in 97.38 s. The cumulative and
round-specific citation gates pass after the correction; their plants fire.
Separate `codex exec --sandbox read-only` review was
attempted. Verdict: **independent review unavailable in-sandbox** (`failed to
initialize in-process app-server client: Read-only file system`).

## OPEN

1. Walk the next northern denominators separately: 514 NE `e3f_0vor` cells,
   then 521 NW cells; do not combine them with the mask differences.
2. Then walk the 1,160/1,154 northern mask cells before the accumulator/final-
   scale pair and the distinct 68-cell substep-2 U residual.
3. Continue the independent hierarchy/card month program after the rung-0
   source walk. This receipt makes no independent ORCA2 month claim.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

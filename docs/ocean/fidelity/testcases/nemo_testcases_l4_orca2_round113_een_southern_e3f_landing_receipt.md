# ORCA2 round 113 — southern EEN thickness association landing

**Status: LANDED.** Given NEMO's recorded rung-0 operands, the first
non-bit source-ordered EEN quotient operand was the southern association of
`e3f_0vor`: seven cells on global row `j=0`, first `(j,i,k)=(0,28,0)`.
NEMO builds the masked four-cell thickness, applies the F-grid boundary
operation, and then replaces boundary zeros with the mesh `e3f_3d`
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937`). The
double-precision boundary path defaults an omitted `kfillmode` to constant
zero
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1811-1820` and
`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1864-1872`),
while the replacement mesh field was read with `jpfillcopy`
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domzgr.f90:179-188`). That copy
association takes the nearest interior row
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:2021-2046`).

The production literal EEN builder now applies that association only to the
southern halo of `e3f_0vor`
(`barotropic_latlon_cgrid.py:909-911`,
`barotropic_latlon_cgrid.py:1011-1022`). The open `r3f` and frozen-mask
associations stay cyclic for this measurement, so this is one source
statement. No configuration, initial or carried state, forcing, stabiliser,
sea-ice selector, or `unmeasured_features` entry changed.

## Frozen prediction disposition

The round-113 preregistration was committed before measurement.

| prediction | disposition |
|---|---|
| R113-P1: baseline has seven `south_e3f0` differences, first `(0,28,0)` | **CONFIRMED** |
| R113-P2: the source association makes `south_e3f0` exact | **CONFIRMED**, 0 unequal |
| R113-P3: the denominator closes and 66-cell `south_mask` is next | **CONFIRMED**, first `(0,29,0)` |
| R113-P4: ORCA2, VORTEX, VORTEX_VEC and two DINO recipes execute; GYRE and tanks do not | **CONFIRMED** |
| R113-P5: no exact ORCA2 row leaves the bar and first debt is not earlier | **CONFIRMED** |

The oracle-bit, candidate-bit, and resolved-scope-route plants each exit 2.
The route census is closed over all registered cards.

## Retained failed implementation

The first implementation incorrectly replaced the reconstructed masked
thickness by the mesh field on every row. The gate refused it with 32,844
`south_e3f0` differences, first `(1,48,0)`; samples were factors of two or
four from NEMO. This is retained as an implementation refutation, not a
refutation of R113-P2: the compiled statement restores `e3f_3d` only where
the boundary-filled reconstructed value is zero. The corrected implementation
changes only row `j=0`, and the final gate closes all seven registered cells.

## Source-ordered measurement

All values in this section are **given NEMO's recorded entry**. The admitted
record has two self-describing rank shards, each owning `3 x 3 x 92 x 150`
cells, with exactly-once coverage.

| item | before bit/magnitude unequal | after bit/magnitude unequal | first after |
|---|---:|---:|---|
| southern `e3f_0vor` | 7 / 7 | 0 / 0 | — |
| southern denominator | 7 / 7 | 0 / 0 | — |
| southern `r3f` | 0 / 0 | 0 / 0 | — |
| southern frozen mask | 66 / 66 | 66 / 66 | `(0,29,0)` |
| southern quotient/fraction | 7 / 7 | 0 / 0 | — |

The walk stops at the next raw operand. No mask, product, accumulation,
fold, or later-substep attribution is claimed.

## ORCA2 and shared-card gates

The rung-0 ladder is **independent** from NEMO's rung-0 from-rest entry. The
shipped rung-7 ladder is **given NEMO's recorded entry** under Decision 52.
Both 200-row comparisons move 0 rows, lose no exact row, and retain their
first non-bit rows. Both comparator plants fire.

- GYRE does not execute the statement. All 70 ten-step rows are
  array-identical, with residual SHA-256
  `377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
  All 30 daily snapshot payloads are byte-identical to round 112; only the
  manifest's producer commit, clone path, and wall time change.
- DINO executes the same literal builder. Its certified fp64 CPU month
  completes all 960 steps and is unchanged: day-30 wet 3-D T RMS is
  `2.053801168e-03 K` against the `2.244317642e-03 K` bar, and the snapshot
  SHA-256 remains
  `83b9ba659ac3b00c873c6999a225cc6c4215901f484c121b550f40fc5292d78c`.
- LOCK_EXCHANGE and OVERFLOW resolve generic/average Coriolis and cannot
  execute this statement. Their direct trajectory reruns refused before
  model execution on the registered pre-existing scalar-math-v2 provenance
  check (`oracle root is not certified scalar-math v2 at kt=2`); no oracle
  record was altered or bypassed. The closed route control, focused tank/card
  scope tests, and the full fidelity battery remain green for this change.
- VORTEX and VORTEX_VEC execute the literal builder. The focused VORTEX
  `e3f_0vor` controls and barotropic null-mode tests pass; no VORTEX trajectory
  claim is made here.

## Tests, citations, and review

The focused battery passes **65/65** in 100.37 s. The one required
`tests/ocean/fidelity -n 12` battery collected 2,337 tests, reached 99%, and
then reproduced the registered xdist-controller stall. Before termination it
emitted exactly the six pre-existing failures: SI3 scalar-math provenance,
GYRE round-51 private/default-off operands, round-35 escape scope, worktree
stamping, the recipe case-board row, and the GYRE round-129 record stamp. No
round-113 test failed.

The cumulative citation gate passes with zero failures, unmapped citations,
or map-audit failures. Its round-113 compiled-source plant fires. Separate
`codex exec --sandbox read-only` review was attempted. Verdict:
**independent review unavailable in-sandbox** (`failed to initialize
in-process app-server client: Read-only file system`).

## OPEN

1. Walk the new first item: the 66 southern frozen-mask cells, first
   `(j,i,k)=(0,29,0)`, without combining it with product or accumulation.
2. Then continue the fraction/product/accumulator signed-zero walk only after
   the mask operand is exact.
3. Continue the distinct south-U residual, northern-fold V pair, and later
   substep-2 U residual only in source order.
4. The independent-start hierarchy/month program remains open; this receipt
   makes no independent ORCA2 month claim.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

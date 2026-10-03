# ORCA2 round 114 — southern EEN frozen-mask association landing

**Status: LANDED.** Given NEMO's recorded rung-0 operands, the first
non-bit source-ordered EEN quotient operand was the southern association of
the frozen `fe3mask`: 66 cells on global row `j=0`, level `k=0`, first
`(j,i,k)=(0,29,0)`. NEMO applies the ordinary F-grid lateral boundary
operation to `fmask` and only then freezes `fe3mask = fmask`
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258`). The call
passes neither a fill mode nor a fill value, so the executed double-precision
boundary path gives the closed southern halo its default constant zero
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1811-1820`,
`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1864-1872`, and
`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1999-2004`). The literal
EEN quotient reads this frozen mask at `jj-1`
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1244-1248`).

The production builder now exposes the source association as a small shared
helper (`barotropic_latlon_cgrid.py:914-916`) and uses it only for the
southern frozen-mask operand (`barotropic_latlon_cgrid.py:1016-1030`). The
already-landed `ff_f` and `e3f_0vor` associations and the already-exact `r3f`
association do not move. No configuration, initial or carried state, forcing,
stabiliser, sea-ice selector, or `unmeasured_features` entry changed.

## Frozen prediction disposition

The round-114 preregistration was committed before measurement.

| prediction | disposition |
|---|---|
| R114-P1: baseline has 66 `south_mask` differences, first `(0,29,0)` | **CONFIRMED** |
| R114-P2: default zero fill makes `south_mask` exact | **CONFIRMED**, 0 unequal |
| R114-P3: every recorded fraction operand and ordered sum is exact | **CONFIRMED**, 0 unequal on all 20 rows |
| R114-P4: ORCA2, VORTEX, VORTEX_VEC and two DINO recipes execute; GYRE and tanks do not | **CONFIRMED** |
| R114-P5: no exact ORCA2 row leaves the bar and first debt is not earlier | **CONFIRMED** |

The oracle-bit, candidate-bit, and resolved-scope-route plants each exit 2.
The production-binding unit test fails if the literal builder stops calling
the helper, and the zero-fill/cyclic control proves the association differs
on a nonzero southern source row.

## Source-ordered measurement

All values in this section are **given NEMO's recorded entry**. The admitted
record has two self-describing rank shards, each owning `3 x 3 x 92 x 150`
cells, with exactly-once coverage. Production JIT ran on CPU under
fp64/x64/libm.

| item | before bit/magnitude unequal | after bit/magnitude unequal |
|---|---:|---:|
| southern frozen mask | 66 / 66 | 0 / 0 |
| southern denominator | 0 / 0 | 0 / 0 |
| southern quotient | 0 / 0 | 0 / 0 |
| west-plus-centre ordered sum | 0 / 0 | 0 / 0 |
| complete three-fraction ordered sum | 0 / 0 | 0 / 0 |

The mask differences were operand-visible but arithmetically inert because
the recorded stretch operand makes their denominator contribution zero.
Closing the mask completes every raw operand, quotient, and ordered sum in the
round-110 fraction record bit-exactly. No product, accumulator, fold, or later
substep attribution is claimed.

## ORCA2 and shared-card gates

The rung-0 ladder is **independent** from NEMO's rung-0 from-rest entry. The
shipped rung-7 ladder is **given NEMO's recorded entry** under Decision 52.
Both complete 200-row comparisons move 0 rows, lose no exact row, and retain
their first non-bit rows.

- GYRE does not execute the statement. All 70 ten-step rows are
  array-identical; all 210 residual arrays are `np.array_equal`; residual
  SHA-256 remains
  `377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
  All 30 daily snapshot files are byte-identical; day-30 SHA-256 remains
  `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`.
- DINO executes the same literal builder. Its certified fp64 CPU month
  completes 960 steps and is unchanged: day-30 wet 3-D T RMS is
  `2.053801168e-03 K` against the `2.244317642e-03 K` bar. The snapshot
  SHA-256 remains
  `83b9ba659ac3b00c873c6999a225cc6c4215901f484c121b550f40fc5292d78c`.
- LOCK_EXCHANGE and OVERFLOW resolve generic/average Coriolis and cannot
  execute this statement. Their registered scalar-math-v2 record refusal is
  unchanged; no record or gate was bypassed.
- VORTEX and VORTEX_VEC execute the literal builder. The focused VORTEX
  `e3f_0vor` controls and barotropic null-mode tests pass; no VORTEX trajectory
  landing claim is made here.

## Tests, citations, and review

The focused fraction-chain, routing, VORTEX, and barotropic battery passes
**44/44**. The single required `tests/ocean/fidelity -n 12` battery is reported
in the final round commit after its one invocation.

The cumulative citation gate and this receipt's planted citation check are
required before the final commit. The edited model file's cumulative map was
re-anchored from a `difflib.SequenceMatcher` pre/post map; no rigid-shift
guess was used. Separate `codex exec --sandbox read-only` review was attempted.
Verdict: **independent review unavailable in-sandbox** (`failed to initialize
in-process app-server client: Read-only file system`).

## OPEN

1. Resume the per-level recurrence at the now-exact `zpvo_nw`: walk the two
   recorded stored-product signed zeros, then the accumulator-before and
   accumulator-after signed-zero semantics in compiled source order.
2. Then resume the separate 68-cell south-U debt, northern V cancelling pair,
   and later 68-cell substep-2 U residual.
3. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy deliverables.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

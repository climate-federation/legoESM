# ORCA2 round 111 — frozen EEN thickness-mask landing

**Status: LANDED.**  Given NEMO's recorded rung-0 entry, the admitted
round-110 fraction record names legoESM's live `fmask` operand as the first
non-bit item in the registered EEN quotient decomposition.  NEMO freezes a
separate `fe3mask` from `fmask` during mask construction
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dommsk.f90:252-258`) and consumes
that frozen array in all three EEN denominators
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1248`).  The
shared literal coefficient builder now consumes the card's already-carried
`fe3mask`.  No card option, forcing, initial state, stabiliser, or ice selector
changed.

## Frozen predictions and admission

The committed round-111 preregistration preceded every measurement.  The
operator-produced record re-admitted as `PASS_R110_EEN_FRACTION_ADMISSION`:
two rank shards, exactly-once 148x180 coverage, inherited streams passive, and
rank-0/rank-1 kt=10 restart hashes unchanged.  Both admission plants fire.

| prediction | disposition |
|---|---|
| the operator record admits without changing the model | **CONFIRMED** |
| west and centre fractions are exact; the south fraction owns all 180 remaining cells | **CONFIRMED** |
| the south difference is an operand association rather than arithmetic inside a recorded quotient | **CONFIRMED**; `south_ff` is now first |
| the rank seam is not involved | **CONFIRMED**; all 180 cells are global `j=0`, `k=0` |
| stop at the first source-ordered statement | **CONFIRMED**; the southern `ff_f` halo remains OPEN |

## Source-ordered measurement

All numbers in this section are **given NEMO's recorded entry**.  The original
probe output incorrectly stamped the label `independent`; that metadata is
**RETRACTED**.  The committed probe now stamps the binding label above, and
`een_fraction_walk_candidate_final.json` is its clean-tree rerun.

Before the statement, the registered raw-mask differences were 23,649 west,
16,830 centre, and 18,077 south executed cells.  The west and centre final
fractions nevertheless cancelled exactly.  After substituting only the
compiled-source `fe3mask` operand, west mask, centre mask, west fraction, and
centre fraction are all bit-exact.  The next non-bit item is:

| remaining item | bit-unequal cells | first `(j,i,k)` |
|---|---:|---|
| southern `ff_f` operand | 180 | `(0,0,0)` |
| southern `e3f_0vor` | 7 | `(0,28,0)` |
| southern `fe3mask` | 66 | `(0,29,0)` |
| southern denominator | 7 | `(0,28,0)` |
| southern fraction and three-term sum | 180 each | `(0,0,0)` |

The western and central paths therefore close without a cancelling residual.
The southern path is a separate boundary owner: NEMO reads the southern halo
in the cited EEN statement, while its geometry setup applies an F-grid
copy-fill lateral boundary operation
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:135-143`).  The current
model replay cyclically rolls the northern row into that operand.  This round
does not land that second statement.

## Trajectory gates

The rung labels below follow Decision 52 explicitly.  The rung-0 and shipped
rung-7 ladders are **given NEMO's recorded entry** (the latter uses the pinned
Decision-52 SSH bridge); none of their numbers are an independent-start month
claim.

The committed row comparator checked all 200 rows on each ladder, retained the
first non-bit statement, and fired both its bit-loss and earlier-first plants.

| ladder | moved / 200 | max-abs toward / away | rms toward / away | exact-row losses | first non-bit |
|---|---:|---:|---:|---:|---|
| rung 0 dynamics core | 175 | 75 / 100 | 68 / 107 | 0 | kt=1 stage1 T, unchanged |
| rung 7 shipped ocean card | 195 | 105 / 90 | 16 / 179 | 0 | kt=1 stage1 T, unchanged |

Every moved row, including its before/after max, rms, unequal count, and first
unequal index, is registered in `round111/orca_ladder_compare.json`.  Headline
kt=10 stage-3 values are retained only as a compact audit:

| rung / field | max before -> after | rms before -> after |
|---|---:|---:|
| rung 0 T | 0.8929704006615697 -> 0.8929683387777949 | 0.006317420525208628 -> 0.006317492864627037 |
| rung 0 S | 0.4156326670091559 -> 0.4156325968351702 | 0.0016155151271706628 -> 0.0016155413057429675 |
| rung 0 u | 0.6887186133134653 -> 0.6887252032647637 | 0.004518520574918117 -> 0.004518849843573174 |
| rung 0 v | 1.555861413441975 -> 1.5558805820532111 | 0.005041743314861328 -> 0.0050426701546735925 |
| rung 0 ssh | 0.42915183233564513 -> 0.42899556346554946 | 0.028059540421323897 -> 0.02805942934825134 |
| rung 7 T | 1.2418085267426164 -> 1.2418655862627084 | 0.010203358679641124 -> 0.010203381952993567 |
| rung 7 S | 0.28802595721631974 -> 0.2880329286832648 | 0.002805115370263332 -> 0.0028051321318774934 |
| rung 7 u | 0.3076159482679226 -> 0.3076289547130745 | 0.005748711510742764 -> 0.005749917586656584 |
| rung 7 v | 0.5378473427951197 -> 0.5378434774585444 | 0.005767169532357182 -> 0.005769571791550708 |
| rung 7 ssh | 0.30405069937287454 -> 0.3041716331751489 | 0.027579326439968365 -> 0.02758299032157362 |

Already-debt ORCA2 rows move in both directions; the mechanical landing
predicate is satisfied because no bit-exact row leaves the bar and first debt
is not earlier.

## Shared-card gates

- GYRE: all 70 ten-step rows and the residual digest
  `377dd4c...` are unchanged.  The first-over-bar step remains kt=3.  All 30
  daily snapshots through day 30 are byte-identical to the certified baseline;
  day 30 hashes to `4e36c106...` on both arms.
- LOCK_EXCHANGE: certified first debt remains kt=4 U; the complete residual
  artifact is byte-identical at
  `a28d1f8ac5482b0f3b7cdf026018050c1e9b9554f22743c49f43d5478bbd0b1a`.
- OVERFLOW: certified first debt remains kt=2 T/U; the complete residual
  artifact is byte-identical at
  `43e297c25b34b950fc038d40ad989d36f05cc95006ad3c9f439128a36bb523c5`.
- DINO uses its experiment recipe and does not carry or call the testcase
  literal EEN operand bundle changed here.  The direct shared-builder/card
  scope tests pass; a GPU DINO month was not run because this round is CPU
  only and DINO does not execute the statement.

An early shell comparison used `day01` names against the baseline's `day001`
names and reported 30 missing/different files.  That filename comparison is
**RETRACTED**.  The corrected zero-padded comparison found 0/30 differences.

## Tests, citations, and review

- Focused fraction, comparator, VORTEX scope, and barotropic tests: **32
  passed**.  Reverting the production operand makes the direct control fail.
- The required `tests/ocean/fidelity -n 12` invocation collected 2,330 tests,
  reached 97%, emitted the documented pre-existing failure markers, and then
  stalled on the OVERFLOW undetected-plant test without a terminal summary.
  That exact test passed alone in 132.16 s.  The round-111 tests were green.
- The isolated provenance ratchets retain their pre-existing offender lists;
  no round-111 file appears in either list.
- The citation-map failure created while drafting this receipt was the wrong
  occurrence number for the repeated `dommsk` loop header.  It is fixed to the
  second occurrence; the default receipt and this receipt are required green
  below, with the shifted-line plant firing.
- Separate `codex exec --sandbox read-only` review was attempted.  Verdict:
  **independent review unavailable in-sandbox** (`failed to initialize
  in-process app-server client: Read-only file system`).

## OPEN

1. Walk the new first non-bit item, the 180-cell southern `ff_f` halo
   association, against NEMO's F-grid copy-fill semantics.  Preregister and
   land it separately if its full gates pass.
2. Then walk the remaining seven southern `e3f_0vor`/denominator cells and 66
   mask cells before returning to product and accumulator signed-zero debt.
3. Continue the distinct 68-cell south-U, northern-fold V cancelling pair,
   and later substep-2 U residual only after the fraction sum is exact.
4. The independent-start hierarchy/month program remains open; this receipt
   makes no independent month claim.

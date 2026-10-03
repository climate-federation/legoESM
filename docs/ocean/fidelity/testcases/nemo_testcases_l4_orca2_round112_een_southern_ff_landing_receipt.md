# ORCA2 round 112 — southern EEN Coriolis association landing

**Status: LANDED.** Given NEMO's recorded rung-0 entry, the first non-bit
source-ordered EEN quotient operand was the southern `ff_f` association: 180
cells on global row `j=0`, level `k=0`. NEMO reads `ff_f` from the configured
domain file on the live `ln_read_cfg` branch
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:101-109`) and requests
`jpfillcopy` for that F-grid field
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:233-236`). The boundary
code copies the nearest interior row rather than cyclically wrapping the
northern fold row
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1199-1225`), before the
literal EEN quotient consumes `ff_f(ji,jj-1)`
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1244-1248`).

The production literal EEN coefficient builder now gives only that numerator
the source-exact southern association
(`barotropic_latlon_cgrid.py:904-906`, `barotropic_latlon_cgrid.py:1002-1008`).
The denominator's independently open `e3f_0vor`, `r3f`, and `fe3mask`
associations are unchanged. No configuration, forcing, initial state, carried
state, stabiliser, sea-ice selector, or `unmeasured_features` entry changed.

## Retraction and compiled-path correction

Round 111 described `domhgr.f90:135-143` as the source of this copy-fill.
That call is in the user-defined-grid arm and is **dead on ORCA2**, whose
resolved deck has `ln_read_cfg=.true.`. That source attribution is retracted.
The measured 180-cell first boundary is not retracted; this round binds it to
the executed `hgr_read` / `iom_get(..., kfill=jpfillcopy)` path above.

## Frozen prediction disposition

The round-112 preregistration was committed before the candidate measurement.

| prediction | disposition |
|---|---|
| R112-P1: baseline remains 180 `south_ff` cells, first `(0,0,0)` | **CONFIRMED** |
| R112-P2: copy association makes `south_ff` bit-exact | **CONFIRMED**, 0 unequal |
| R112-P3: next item is seven `south_e3f0` cells, first `(0,28,0)` | **CONFIRMED** |
| R112-P4: no additional card beyond ORCA2/VORTEX executes | **REFUTED and retained**: both NEMO-faithful DINO recipes execute; their month gate was added before landing |
| R112-P5: no exact row leaves the bar and first debt is not earlier | **CONFIRMED** |

The operand gate's oracle-bit, candidate-bit, and resolved-scope-route plants
all exit 2. Its final clean-tree scope census is: ORCA2, VORTEX,
VORTEX_VEC, DINO `nemo_dino_kamm`, and DINO `nemo_dino_kamm_mlf` execute;
GYRE, LOCK_EXCHANGE, and OVERFLOW do not.

## Source-ordered measurement

All values in this section are **given NEMO's recorded entry**. The admitted
record comprises two rank shards with exactly-once owned coverage of
`3 x 3 x 92 x 150` cells each. Production JIT ran on CPU under fp64/x64/libm.

| item | before bit/magnitude unequal | after bit/magnitude unequal | first after |
|---|---:|---:|---|
| southern `ff_f` | 180 / 180 | 0 / 0 | — |
| southern `e3f_0vor` | 7 / 7 | 7 / 7 | `(0,28,0)` |

Thus the single source statement closes exactly and the walk stops at the
next registered operand. No downstream fraction, product, accumulation, fold,
or substep attribution is claimed.

## ORCA2 and shared-card gates

The rung-0 ladder is **independent** (NEMO's rung-0 from-rest entry); the
shipped rung-7 ladder is **given NEMO's recorded entry** through Decision 52.
The complete comparator reports 0/200 moved rows on each ladder, no exact-row
loss, and unchanged first debt at kt=1 stage-1 T. Both comparator plants fire.
The operand repair is therefore trajectory-inert over kt=1..10 while its
recorded source boundary is bit-exact.

- GYRE: all 70 ten-step rows are array-identical and residual SHA-256 remains
  `377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
  All 30 daily snapshots are byte-identical; day 30 remains
  `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`.
- LOCK_EXCHANGE: first debt remains kt=4 U and the residual digest remains
  `a28d1f8ac5482b0f3b7cdf026018050c1e9b9554f22743c49f43d5478bbd0b1a`.
- OVERFLOW: first debt remains kt=2 T/U and the residual digest remains
  `43e297c25b34b950fc038d40ad989d36f05cc95006ad3c9f439128a36bb523c5`.
- DINO: the certified fp64 CPU month completed 960 steps. Day-30 wet 3-D T
  RMS is `2.053801168e-03 K` versus NEMO kt=960, below the
  `2.244317642e-03 K` bar. This is +`1.3512403e-05 K` (+0.66%) versus the
  certified `2.040288765e-03 K`, the registered shared-builder movement from
  Decision 84, and passes the binding bar. The snapshot SHA-256 is
  `83b9ba659ac3b00c873c6999a225cc6c4215901f484c121b550f40fc5292d78c`.
- VORTEX/VORTEX_VEC execute the same literal EEN builder; the focused
  VORTEX `e3f_0vor` gate and barotropic null-mode controls pass. No VORTEX
  landing claim is made here.

## Tests, citations, and review

The final focused battery passes **62/62** in 99.41 s. The single required
`tests/ocean/fidelity -n 12` battery reached 99% and then exhibited the known
xdist-controller stall without a terminal summary. It emitted exactly the six
registered pre-existing failures: SI3 scalar-math provenance, GYRE round-51
private/default-off operands, round-35 escape scope, worktree stamping, the
recipe case-board row, and the GYRE round-129 record stamp. No round-112 test
failed.

The cumulative citation gate passes with zero failure, unmapped citation, or
audit rows; its shifted `stprk3.F90:186` plant fires. Separate
`codex exec --sandbox read-only` review was attempted. Verdict:
**independent review unavailable in-sandbox** (`failed to initialize
in-process app-server client: Read-only file system`).

## OPEN

1. Walk the new first item: seven southern `e3f_0vor`/denominator cells,
   first `(j,i,k)=(0,28,0)`, without combining it with the 66 open mask cells.
2. Then walk the 66 southern mask cells before returning to fraction/product/
   accumulator signed-zero debt.
3. Continue the distinct 68-cell south-U residual, northern-fold V pair, and
   later substep-2 U residual only after the fraction sum is exact.
4. The independent-start hierarchy/month program remains open; this receipt
   makes no independent ORCA2 month claim.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

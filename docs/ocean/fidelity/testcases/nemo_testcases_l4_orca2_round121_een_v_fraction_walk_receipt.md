# ORCA2 round 121 — northern-V EEN fraction walk

Date: 2026-10-03. Base `74fd18c2bd1d377e5ca6192291cbaa4f0192d1dd`;
locked measurement gate `d44c1dfae`. Every ocean number below is
**independent**: hierarchy rung 0 starts from NEMO's own from-rest state. No
model physics, card field, configuration value, carried state, stabilizer,
sea-ice selector, or `unmeasured_features` entry changed.

## Verdict

**HELD.** The admitted round-120 record resolves the round-119 completed-sum
boundary. Northeast V first differs in fraction 1 and northwest V first
differs in fraction 3, exactly the two fractions that read the northern halo.
For both paths the first unequal operand is `ff_f`: 1,431 bit- and magnitude-
unequal executed cells, all at global `j=147`, spanning all 180 longitudes and
both ranks.

The later northern `e3f_0vor` and frozen-mask operands also differ, but they do
not own the first statement: the ordered expression reads `ff_f` first. No
partial recurrence or production change is eligible. The next walk is the
source-cited F-grid copy-fill association that supplied this halo.

## Round-120 record admission

The existing target
`orca2_rounds/round120/acquisition/orca2_rung0_een_v_fraction_ranked_10step_np2`
admits with `STATUS PASS_R120_EEN_V_FRACTION_ADMISSION`:

- the two ranks cover the `148 x 180` domain exactly once and execute 231,368
  and 194,504 level-cells;
- every denominator, quotient, ordered sum, and inherited completed `zpvo`
  replays bitwise;
- all twenty kt=1..10 restart shards and all ten inherited
  round-105/107/110/116/118 streams are byte-identical;
- all thirteen runtime plants fire; and
- rank-0/rank-1 record SHA-256 values are
  `94024493b7adc9bf11be2778546df7e3540d1713e6d95e03b25367f7fa87ac9d`
  and `0024d4431c4df177062538f03b95082574d6885a642ade60170b65e5b4d41a6e`.

Thus the write-only recorder is observationally passive, self-described,
arithmetically sufficient, and rank-complete before any fraction is
interpreted.

## Compiled source boundary

NEMO forms northeast V from `(ji,jj+1)`, `(ji,jj)`, `(ji-1,jj)` and northwest
V from `(ji,jj)`, `(ji-1,jj)`, `(ji-1,jj+1)` in that literal order
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1328`).
Therefore the observed NE fraction 1 and NW fraction 3 are the only two
northern-halo fractions and their first recorded source operand is `ff_f`.

This deck reads `ff_f` from the domain file as an F-grid field with
`jpfillcopy`, rather than recomputing it from latitude
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/domhgr.f90:233-237`). That executed
copy-fill association, including the northern fold permutation, is the next
statement to walk; its result is not inferred here.

## Source-ordered measurements

Counts are bit-unequal / magnitude-unequal over executed cells. Unlisted
fractions and operands are 0 / 0.

| path | first item | `ff_f` | `e3f_0vor` | `r3f` | `fe3mask` | denominator | fraction | partial | final sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE fraction 1 | `1_ff` | 1,431 / 1,431 | 514 / 514 | 0 / 0 | 1,160 / 1,160 | 514 / 514 | 1,431 / 1,431 | 1,431 / 1,431 | 1,431 / 1,431 |
| NW fraction 3 | `3_ff` | 1,431 / 1,431 | 521 / 521 | 0 / 0 | 1,154 / 1,154 | 521 / 521 | 1,431 / 1,431 | 0 / 0 | 1,431 / 1,431 |

Both first-operand supports contain every global longitude `i=0..179`.
Rank 0 carries 723 unequal level-cells and rank 1 carries 708 for each path;
the support is neither a rank seam nor a one-rank artifact. The first unequal
cell is `(j,i,k)=(147,0,0)` for both paths.

The resolved-card census is unchanged: ORCA2, both DINO recipes, VORTEX, and
VORTEX_VEC execute the literal EEN association; GYRE and both tank cards do
not. This is scope evidence, not authorization for a partial landing.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R121-P1: recorder passive | **CONFIRMED** by exact restart and inherited-stream identity |
| R121-P2: rank-complete and arithmetically sufficient | **CONFIRMED** by exactly-once coverage, bitwise arithmetic replay, and all runtime plants |
| R121-P3: NE fraction 1 and NW fraction 3 own the first difference | **CONFIRMED**, each first at its `ff_f` operand with 1,431 / 1,431 |
| R121-P4: first operand support is northern-fold-only | **CONFIRMED**, only global `j=147` |
| R121-P5: no rank seam owns the difference | **CONFIRMED**, all 180 longitudes and 723 / 708 cells across ranks |
| R121-P6: measurement-only round | **CONFIRMED**, no `packages/` or card file changed |

## Mechanical gates, tests, and review

The committed round-121 gate reports
`STATUS MEASURED_R121_EEN_V_FRACTIONS`. It locks every operand and sum census,
the literal shift association, fp64/libm CPU production-JIT execution, clean
worktree stamp, resolved-card scope, full-longitude support, and both-rank
support. Its oracle-bit, candidate-bit, rank-seam, and scope-route plants each
refuse.

The default citation gate passes 274 citations and this receipt passes its two
citations, each with zero unmapped or failed entries. Shifting the compiled
fraction citation by two lines makes the gate refuse. The focused
round-119/120/121 plus citation battery passes 28/28.

The single required `tests/ocean/fidelity -n 12` invocation collected 2,366
tests and reached 98%, then reproduced the registered xdist-controller stall
after every real pytest process had exited. It was interrupted after the
one-minute grace period without a summary and is **incomplete, not PASS**.
Re-running the six displayed failing IDs together reproduces the same six
registered pre-existing reds: SI3 scalar-math source provenance, the round-129
certified-year stamp, round-51 private-arm scope, round-35 dirty-escape
scoping, worktree stamping, and the `hires_lane_surface` case-board row. No
round-121 failure was exposed.

The separate read-only Codex review returned **independent review unavailable
in-sandbox**: `failed to initialize in-process app-server client: Read-only
file system`.

No model or card file changed, so ORCA2 rung-0/rung-7, GYRE, DINO, tank, and
generic-card trajectories cannot move and are not represented as rerun gates.

## OPEN

1. Walk the northern `ff_f` F-grid `jpfillcopy` association, recording the
   source longitude and fold sign used for all 180 halo values. Gate the
   permutation against both NE fraction 1 and NW fraction 3 before changing
   production code.
2. Then walk the already-measured 514/521 `e3f_0vor` and 1,160/1,154 frozen-
   mask cells in source order. Treat the northern accumulator/final-scale rows
   as the round-106 cancelling pair; do not land one half.
3. Resume the later 68-cell substep-2 U residual only after coefficient
   construction closes. The package-exposed rung-0 card and independent
   240-step month remain open hierarchy work.

## UNVERIFIED

- The exact northern F-grid source permutation and whether it alone closes all
  1,431 `ff_f` values.
- Whether the complete northern accumulator/final-scale pair passes every
  trajectory landing gate.
- The owner of the later 68-cell substep-2 U residual.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

# ORCA2 round 47 — source-ordered stage-1 tracer ratio landing

**Date:** 2026-09-27  
**Base:** `655c72e707e941ddf21bdf7f1d2b3c4d735527f5`  
**Preregistration:** `0573d48c0203250823b458cee12cd1044a4c53b7`  
**Measured candidate:** `e14e02a0da842f1dc7e5d9c98e7327bd87a6b837`  
**Disposition:** **LANDED**

## Claim boundary

Unless a row below says **INDEPENDENT**, every ORCA2 number is **GIVEN NEMO'S
ENTRY** under Decision 52 and the pinned `VARIANT_ORACLE_ORCA1ICE` record.
The independent initial state is still absent; no number from that claim is
mixed into the twin tables.

Sea ice remains out of scope.  The ORCA2 card's six selectors and
`unmeasured_features` tuple are unchanged.

## Compiled owner and implementation

The executing preprocessed source first forms the before- and after-level
`r3t` ratios, then assigns Kaa as
`r2_3 * r3t(Kbb) + r1_3 * r3ta` in
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`.
Thus interpolating SSH before dividing by depth is real-equivalent but is not
the compiled floating-point statement.

The landing adds `nemo_r3t_rk3_stage1_stretch`, with explicit
`nemo_source_round` barriers, and routes only the nonlinear-free-surface
WS-RK3 stage-1 tracer Kaa ratio through it.  Euler cards do not execute it;
the generic linear-free-surface GYRE recipe returns the structural unit ratio.
No selector, stabiliser, or carried-state field was added.

## Statement replay — GIVEN NEMO'S ENTRY

| row | base | candidate | verdict |
|---|---:|---:|---|
| Kaa ratio | 2 / 8,613 unequal; max 1.1102230246251565e-16 | 0 / 8,613 | **CONFIRMED** |
| interpolated-SSH control | not present | 2 / 8,613 unequal; max 1.1102230246251565e-16 | control fires |
| production stage-1 T | 57,160 / 228,641 unequal; max 7.105427357601002e-15 K | 57,141; max 3.552713678800501e-15 K | toward |
| production stage-1 S | 57,180 / 228,641 unequal; max 1.4210854715202004e-14 | 57,169; same maximum | toward by count |

The source-ordered literal replay remains 0 / 228,641 unequal for both T and
S.  Reverting the production call makes the direct eager/JIT/source-binding
test fail, so the control can fail.

## ORCA2 ten-step ladder — GIVEN NEMO'S ENTRY

All 40 checkpoints through kt=10 completed.

- 141 / 200 field rows moved: 25 toward NEMO by maximum, 35 away, and 81 with
  the same maximum.
- No AT-BAR row left the bar.
- The first moved row is kt=1 stage-1 T.
- The first whole-card non-bit statement is unchanged: kt=1 stage-1 T,
  233,341 / 399,600 unequal, maximum 0.0014770192519700243 K.

The statement repair therefore closes its two-cell operand debt without
claiming the first whole-card divergence.

## Shared-card Decision 43/45/55/59 gate

The full shared landing gate passed.

| check | result |
|---|---|
| GYRE ten-step ladder | 70 / 70 rows accepted; first-over-bar kt=3 unchanged; zero ULP worsening; 0 moved rows |
| residual artifact | base and candidate are byte-identical, SHA-256 `43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c` |
| GYRE day 30 T rms | 6.572574374770603e-05 -> 6.572572612618985e-05 K, decreases |
| GYRE day 240 T rms | 1.6448360792898062e-02 -> 1.6448361130295849e-02 K; +3.373977867926481e-10 K = 1.68699 floor units |
| GYRE day 360 T rms | 1.1225659968396697e-02 -> 1.122565978973451e-02 K, decreases |
| LOCK_EXCHANGE-zco | 70 rows accepted; zero ULP worsening; first-over-bar remains kt=4 U |
| OVERFLOW-zps | 70 rows accepted; zero ULP worsening; first-over-bar remains kt=2 T/U |
| generic NEMO-GYRE | all three-step certifications unchanged; 0 moved rows |
| DINO recipes | source-derived census says both use Euler and do not execute the statement |

The +1.68699-floor-unit day-240 move is within Decision 59's strict
10-floor-unit / 2e-9 K allowance.  A planted +20-floor-unit day-240
regression makes the gate fail.

## Tests, citations, and review

- Focused gate and statement tests: **17 passed**.
- Shared-card battery: **160 passed**, then **10 passed**.
- The required `tests/ocean/fidelity -n 12` attempt reached 99% and repeated
  the documented xdist-controller stall; it was interrupted and is not called
  green.  The five known pre-existing red node IDs were rerun serially and
  remained the same five reds: round-129 stale certification, round-51
  private trace registry, SI3 scalar-math provenance, worktree-stamp ratchet,
  and the `hires_lane_surface` case-board ratchet.
- Default receipt citation gate: zero failures and zero unmapped citations
  after the required SequenceMatcher re-anchor.  This receipt's citation gate
  also passes, and its two-line-shift plant fires.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round47/`.
The decisive files are `qco_base.json`, `qco_tip.json`,
`orca2_compare.json`, `gyre_ladder_compare.json`,
`gyre_base_year_gap.json`, `gyre_tip_year_gap.json`,
`decision43_45_55_59.json`, `lock_compare.json`,
`overflow_compare.json`, and `generic_compare.json`.

## OPEN

1. The first whole-card non-bit statement remains kt=1 stage-1 temperature.
   Continue source order from the next unresolved tracer RK assignment; do not
   re-open the now exact Kaa ratio.
2. The round-20 slow-forcing/barotropic walk remains open and separate.
3. Build and bit-gate the **INDEPENDENT** ORCA2 initial T/S/SSH state required
   by Decision 52 before any year claim switches away from NEMO's entry.
4. Sea ice remains exactly the card's current `unmeasured_features` tuple at
   `STOP_SELECTOR_GAP`.

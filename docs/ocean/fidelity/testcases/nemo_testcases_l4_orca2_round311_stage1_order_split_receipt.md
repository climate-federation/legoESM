# ORCA2 round 311 — G8's stage-1 order split from G1b's KE-gradient guard; its D122 row

Date: 2026-10-17 (Claude round; codex review: paused (quota)).

Status: **LANDED-pending** (operator review + push gate).

## Headline

- Defect in our code, fixed: G8's stage-1 order and G1b's KE-gradient guard
  shared one operator-order setting, so selecting the order silently dropped
  the guard.  NEMO runs both.  Each now has its own setting; together the
  stage-1 block keeps the guard; the stage-1 order with vector-form
  advection but without G1b's order raises.
- No behaviour change: every card's step jaxpr is identical to base (31
  card/tree pairs, including G8, the only card selecting the order); the
  G1b and G8 certification tests pass on the patched gridladder tree.
- D122 row for the stage-1 order with the guard kept: passes on all seven
  pairs (0 exact rows lost, first over-bar unchanged).  Landed on rung 0,
  OMT-3, OMT-4 and rung 10.  It moves 0 rows on OMT-1/OMT-2 (no diffusion),
  where round 310's guard-dropping version moved 117 and 165.

## Provenance

- Base `b725669d1` (round 310 landed).  Preregistration `aef55b97c`
  (frozen before code and runs); fix `fb3e4f5e6`; D122 gate `08e7f5fec`
  (arms ran on it); jaxpr probe `fb3e4f5e6`, `beeb2d401`, `f27bc3939`; split
  test `bb12d31b7`; selection `42b2f3516`.
- Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round311_stage1_order_d122_gate.py`
  (round-285 ladder, round-276 classifier, round-278 decision).  Probe:
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round311_jaxpr_probe.py`.
  CPU, fp64, libm, production JIT.  Evidence `orca2_rounds/round311/`
  (`arms/`, `final/`, `jaxpr/`, `cert_gl*_{base,tip}.log`).

## Step 1 — the two statements in NEMO and in our code

NEMO, one stage-1 accumulator in `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:63`
(stp_2D): dyn_hpg `:141`, dyn_ldf `:144`, dyn_vor `:147`, then dyn_keg
`:164`, whose subtraction is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:129-130` in
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:61` (dyn_keg).  The same lines in
the build G1b certified on (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:63`;
calls `:138`, `:141`, `:144`, `:161`;
`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynkeg.f90:61`,
`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynkeg.f90:129-130`) and in G8's
(`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:63`; calls
`:141`, `:144`, `:147`, `:164`;
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynkeg.f90:61`,
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynkeg.f90:129-130`).

Our code before: the model passed the tendency routine one operator-order
argument — G1b's flag, or, when the stage-1 order was on, the stage-1 tuple
in its place.  The stage-1 block then rebuilt the RHS with the KE gradient
as a plain sum, overwriting the guarded subtraction G1b's flag had applied.

After: the stage-1 order is its own argument; G1b's flag keeps its own.

| stage-1 order | G1b order | momentum advection | NEMO runs it | ours now |
|---|---|---|---|---|
| on | on | vector | yes (ORCA2) | stage-1 order, guarded dyn_keg |
| on | off | flux, dynadv off | yes (G8) | stage-1 order, plain add (as certified) |
| on | off | vector | no | raises |
| on | on | flux | no | raises (already did) |
| on | private stage-RHS order arm | any | — | raises (was: order silently dropped) |
| off | either | either | — | unchanged |

## Step 2 — no behaviour change

Step jaxprs on base vs fixed tree (`jaxpr/`, `cmp` whole file):

| tree | cards | identical |
|---|---|---|
| this branch | rung 0, OMT-0..4, rung 10 (with kt-1 forcing), ORCA2G1 (the TSUNAMI program on ORCA2), GYRE, DINO, VORTEX zco flux/vector, VORTEX seamount flux/vector | 15/15 |
| gridladder `242eaacc47` + the split | G1, G1b, G2, G3, G4-G8 (G4-G8 with their kt-1 recorded forcing) | 9/9 |

G8 carries the stage-1 order (flux form, G1b order off): identical.
V1-V3: the vmix lane (`0b9443cef2`) has no stage-1 order code (0
occurrences), so the split cannot reach those cards; not traced.  GYRE
ten-step/day-30 not rerun: jaxpr identical.

Outputs: the seven shipped-card controls on the fixed tree reproduce round
310's shipped rows row for row (200/200; OMT-1 160/160).

Certification tests on the patched gridladder copy (copies made git repos
for the provenance stamp):

- G1b and G8 selection (`-k "g1b or g8"` over the G8, G1b-card and G1
  ladder test files): base 37/37 pass, tip 36/37.  Includes G8's kt-3 step
  from NEMO's entry bit-identical on every level, G1b's KEG split + plant
  and G1b's stage-1 replay.
- The one tip failure is a G1b source-count check, by design: it counts the
  guard's calls in the tendency routine (2); the fix adds the call inside
  the stage-1 block (4).  Its numerical FMA check passes.  That test lives
  on the gridladder branch only; it needs 2 -> 4 when the split is imported
  there.

New unit test (4): fails 3/4 on the base tree (the fourth pins flux form's
plain add) and 1/4 with only the guard selection reverted.

## Step 3 — D122 row: stage-1 order, G1b's guard kept, seven pairs

(a) G8 receipt `nemo_testcases_gridladder_g8_round1`, addendum 106.  (b)
Step 1's lines run in ORCA2_OMIP_L4; on OMT-0..2 NEMO never reaches
dyn_ldf (ln_dynldf_OFF), so (b) fails there.  Control = card as shipped
by round 310.

| pair | rows moved | toward/away/equal | early kt 1-3 | late kt 4+ | lost | first over-bar | max kt-10 ssh err (m) |
|---|---|---|---|---|---:|---|---|
| rung 0 | 175/200 | 88/69/18 | 4/16/15 | 84/53/3 | 0 | kt-1 stage-1 T, equal | 0.394173 unchanged |
| OMT-1 | 0/160 | 0/0/0 | — | — | 0 | unchanged | 0.74699 unchanged |
| OMT-2 | 0/200 | 0/0/0 | — | — | 0 | unchanged | 0.348812 unchanged |
| OMT-3 | 175/200 | 64/87/24 | 3/10/22 | 61/77/2 | 0 | kt-1 stage-1 T, equal | 0.363341 unchanged |
| OMT-4 | 175/200 | 114/38/23 | 11/7/17 | 103/31/6 | 0 | kt-1 stage-1 T, equal | 0.394004 unchanged |
| rung 10 indep | 169/200 | 66/91/12 | 12/10/7 | 54/81/5 | 0 | kt-1 entry ssh, equal | 0.277238 unchanged |
| rung 10 given | 173/200 | 71/90/12 | 7/16/10 | 64/74/2 | 0 | kt-1 stage-1 T, equal | 0.261719 unchanged |

Seven-pair decision (round-278 classifier): PASS_R278_D122_DECISION.
Leans away on OMT-3 and rung 10, toward on rung 0 and OMT-4; sea level
unchanged everywhere.

Against round 310's guard-dropping version (same arm name): 117/160 rows
differ on OMT-1 and 165/200 on OMT-2 — exactly the rows that version moved;
with the guard kept those cards are bit-identical to shipped.

## Shipped cards (selection `42b2f3516`)

- Resolved configs vs base (572 fields; rung 10 635): rung 0, OMT-3, OMT-4,
  rung 10 differ in the stage-1 order only; OMT-0..2 identical.
- SHIPPED_ARM_LINE

## Predictions

| ID | Verdict | Result |
|---|---|---|
| R311-P1 | **CONFIRMED** | 23 order-off card/tree pairs identical; V1-V3 not traced (no stage-1 code on their lane). |
| R311-P2 | **CONFIRMED** | G8 jaxpr identical; G1b/G8 cert tests pass (one G1b source count moves 2 -> 4 by design, above). |
| R311-P3 | **CONFIRMED** | seven controls = round 310's shipped rows, row for row. |
| R311-P4 | **CONFIRMED** | moves 169-175 rows on the four diffusion pairs; differs from round 310's arm on OMT-1/2 (and moves 0 rows there). |
| R311-P5 | **CONFIRMED** | 0 exact rows lost, first over-bar unchanged on all seven pairs. |

## Gates and tests

- New: `tests/ocean/unit/test_orca2_round311_stage1_keg_split.py` (4),
  `tests/ocean/fidelity/test_orca2_round311_selection.py` (14): reverting
  the selection fails 9/14; tree restored, `git status` clean.  Round 310's
  selection test now expects the order on its four cards; its shipped-arm
  test moved to round 311's.
- BATTERY_LINE
- CITATION_LINE
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Configuration choices

- ASKED (round prompt + D122): the split; the stage-1 order with the guard
  on rung 0, OMT-3, OMT-4, rung 10 if D122 passes.
- ASKED (round prompt): a combination NEMO never runs raises — stage-1
  order + vector form without G1b's order; stage-1 order beside the private
  stage-RHS order arm (a measurement hook; it silently dropped the order).
- UNASKED: none.  Fold rows (D137), runoff depth (D126), TKE switches,
  double diffusion and G6 fold statements untouched.

## OPEN

1. DECISION D137 (fold rows) still pending; pick: keep off.
2. Gridladder: when it imports the split, its G1b source-count check moves
   2 -> 4.
3. Unchanged: D126, D135, double diffusion (V4), G6 fold statements (G9).

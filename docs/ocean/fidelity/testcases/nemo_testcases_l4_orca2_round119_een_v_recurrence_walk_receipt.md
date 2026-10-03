# ORCA2 round 119 — V EEN recurrence walk

Date: 2026-10-03. Base `ec024f3ca8ab205402f148d2ae52f1f33cfdc16f`;
measurement gate `d541f5fb00df5a448cafc17f2e95b0104e90eb3b`.
Every ocean number below is **independent**: hierarchy rung 0 starts from
NEMO's own from-rest state. No model physics, card field, configuration value,
carried state, stabilizer, threshold, sea-ice selector, or
`unmeasured_features` entry changed.

## Verdict

**HELD.** The admitted round-118 record completes the source-ordered eight-path
recurrence coverage. Southwest and southeast V confirm ordinary IEEE-zero
addition as the complete owner: their operands and products are bit-exact,
and the arm closes respectively 2,557/6,206 and 2,998/6,647 before/after
signed-zero bits without moving a nonzero value.

Northwest and northeast V stop earlier. Each first differs in 1,431 `zpvo`
magnitudes, all on the northern-fold row. Later neighboring U thickness and
mask differences are also fold-only. The registered zero-addition arm removes
the non-fold sign differences but cannot close those magnitudes. This confirms
the round-106 northern accumulator/final-scale cancelling-pair boundary; no
partial recurrence production edit is eligible.

## Round-118 record admission

The existing target
`orca2_rounds/round118/acquisition/orca2_rung0_een_v_recurrence_ranked_10step_np2`
admits again with `STATUS PASS_R118_EEN_V_ADMISSION`:

- both ranks cover exactly 425,872 executed cells for each V path;
- recorded product, executed recurrence, and inherited terminal accumulator
  replay bitwise on both ranks;
- all twenty kt=1..10 restart shards and every inherited round-105/107/110/116
  stream are byte-identical;
- all twelve runtime plants fired in the operator admission; and
- rank-0/rank-1 record SHA-256 values are
  `4b4b6ad96b99e5b1d9491f4f3f55d693b5049739cfbb8ba4e06b6dfb8697e512`
  and `43d660d97f7291137c7bd4da69a76d6a98e611c1e453f6ce31a476844cf7f79d`.

Thus the write-only instrument is observationally passive and complete before
the measurement interprets any operand.

## Compiled source boundary

NEMO forms southwest and southeast `zpvo`, then northeast and northwest
`zpvo`, before adding the four coefficient recurrences in NW/NE/SW/SE order
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1317-1327`).
The first unequal executed statement for NW and NE is therefore the cited
three-fraction `zpvo` construction, not either later recurrence.

## Source-ordered measurements

Counts below separate unequal bit patterns from unequal numeric magnitudes.
Every first NW/NE difference is at global `j=147`; the first cell is
`(j,i,k)=(147,0,0)` for both paths.

| path | first non-bit item | `zpvo` bits / magnitude | U thickness bits / magnitude | U mask bits / magnitude | product bits / magnitude | before bits / magnitude | after bits / magnitude |
|---|---|---:|---:|---:|---:|---:|---:|
| NW | `zpvo` | 1,431 / 1,431 | 95 / 95 | 1,270 / 1,270 | 1,277 / 1,270 | 3,886 / 1,231 | 7,464 / 1,298 |
| NE | `zpvo` | 1,431 / 1,431 | 91 / 91 | 1,283 / 1,283 | 1,284 / 1,283 | 3,926 / 1,249 | 7,503 / 1,316 |
| SW | before | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2,557 / 0 | 6,206 / 0 |
| SE | before | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2,998 / 0 | 6,647 / 0 |

The live V thickness and `mbkv` are bit-exact for all four paths. Under only
ordinary IEEE-zero addition, SW and SE become completely bit-exact. NW and NE
retain exactly their magnitude differences: 1,231/1,298 and 1,249/1,316 in
before/after. That arm therefore distinguishes recurrence signs from the
earlier fold association without conflating the two.

The resolved-card census remains unchanged: ORCA2, both DINO recipes, VORTEX,
and VORTEX_VEC execute the literal EEN association; GYRE and both tank cards
do not. This is scope evidence, not authorization for a partial landing.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R119-P1: round-118 instrument passive and complete | **CONFIRMED** by exact coverage, replay, inherited streams, restarts, and operator-fired runtime controls |
| R119-P2: NW first differs at northern-fold `zpvo` | **CONFIRMED**, 1,431/1,431 at `j=147` |
| R119-P3: NE shares that first boundary | **CONFIRMED**, 1,431/1,431 at `j=147` |
| R119-P4: SW/SE operands and products are exact, recurrence signs first | **CONFIRMED**, before counts 2,557 and 2,998, zero magnitude |
| R119-P5: IEEE zero addition closes paths whose earlier inputs are exact | **CONFIRMED** for SW/SE, candidate 0 unequal throughout |
| R119-P6: northern debt remains a cancelling pair and does not partially land | **CONFIRMED**, NW/NE magnitudes remain and no `packages/` file changed |

## Mechanical gates, tests, and review

The committed round-119 gate reports
`STATUS MEASURED_R119_EEN_V_RECURRENCES`. It locks every baseline and candidate
field census. Its oracle-bit, candidate-bit, and scope-route plants each
refuse. Validation results are filled below before the receipt commit.

No model or card file changed, so ORCA2 rung-0/rung-7, GYRE, DINO, tank, and
generic-card trajectories cannot move and are not represented as rerun gates.

## OPEN

1. Walk the NW/NE `zpvo` northern-fold association in the three fractions in
   compiled order, then the separately recorded neighboring U thickness and
   mask. Treat the accumulator and final scale as the round-106 cancelling
   pair; do not land one half.
2. After the pair is bit-exact, land the complete eight-path recurrence
   semantics only under the ORCA2 rung-0/rung-7, GYRE, DINO, tank,
   generic-card, citation, and push gates.
3. Resume the later 68-cell substep-2 U residual only after coefficient
   construction closes. The package-exposed rung-0 card and independent
   240-step month remain open hierarchy work.

## UNVERIFIED

- Which of NW/NE's three `zpvo` fractions first owns the northern-fold
  difference, and the exact NEMO fold permutation for that operand.
- Whether the complete northern accumulator/final-scale pair passes every
  trajectory landing gate.
- The owner of the later 68-cell substep-2 U residual.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

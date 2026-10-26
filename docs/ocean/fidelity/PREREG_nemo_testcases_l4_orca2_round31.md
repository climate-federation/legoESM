# NEMO testcase Lane 4 — ORCA2 card round 31 preregistration

Date: 2026-09-26

Parent: `5d3334700705`

Status: **PREREGISTERED BEFORE ROUND-31 SCIENTIFIC SCORING.**

Round 31 converges the seven-round split (rounds 24–30).  Part A lands the
lateral-diffusion half of Decision 54 by giving `dyn_ldf` its OWN
consumer-local F thickness, exactly as the compiled statement computes it.
Part B gates the CONSTRUCTION of `dyn_vor`'s frozen `e3f_0vor` statement by
statement against a literal transcription of the compiled lines, on the card's
own carried mesh.  No further trajectory arms are run for the EEN half unless
the construction gate reaches 0 unequal cells.

Trajectory results are **independent with Decision-52 SSH**; the direct
operator replay is separately labelled **given NEMO's entry**.  The six
sea-ice selectors and the card's `unmeasured_features` tuple stay frozen.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round31/`.

## Compiled statements

The executing lateral-diffusion F curl reads the MESH reference F thickness
`e3f_3d` stretched by the live `r3f` through `fe3mask` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`;
its T divergence and its Kmm face divisors are at `:127-129` and `:132-140`.
`e3f_3d` IS `e3f_0`, by the compiled substitution macro at
`domzgr_substitute.h90:100`, and the field is read from the domain file at
`domzgr.F90:173`.

The executing EEN vorticity reciprocal instead reads dynvor's OWN frozen
array `e3f_0vor` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738`.
That array is built as the masked four-cell average of `e3t_3d` times 0.25 at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:914-919`,
exchanged on the F fold at `:935`, and only THEN has its remaining zeros
replaced by `e3f_3d` at `:937`.  The two consumers therefore take two
DIFFERENT reference arrays, and `r3f`/`fe3mask` are shared between them.
The admitted `ocean.output` resolves `ln_dynvor_een = T`, `nn_e3f_typ = 0`,
`ln_dynvor_msk = F`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R31-P1 | The "carried raw F thickness" rounds 28–30 routed into BOTH consumers is `e3f_0`, i.e. dyn_ldf's operand, never dynvor's `e3f_0vor`. | The round-30 seam's carried source is `nemo_een_barotropic.e3f_0` and no acquisition carries an `e3f_0vor` field. | Some carried field is dynvor's own frozen array. |
| R31-P2 | Giving the LDF consumer the carried `e3f_0` reference, with the shared `r3f`/`fe3mask` untouched, leaves the exposed stage-2 EEN component bit-identical to the parent. | The EEN digest is unchanged and the direct LDF tendency moves in >100,000 cells. | The EEN digest moves, or the LDF tendency is inert. |
| R31-P3 | The independent ten-step ladder completes kt=1..10 with no `e3w` refusal, and the kt=10 stage-3 velocity maxima fall by more than one order of magnitude from 15.365503106245665 / 42.669598831454074 m/s. | Forty checkpoints and both maxima below 1.6 / 4.3 m/s. | A refusal, or either maximum above that bar. |
| R31-P4 | The given-NEMO's-entry LDF operator stays bit-exact against the literal compiled-statement replay on the same operands. | 0 unequal scored U and V cells; a one-ULP plant moves exactly one. | Any non-zero unequal count without a plant. |
| R31-P5 | GYRE stays byte-identical, because on its `zco` box the two references differ only where the four-T-cell vertex mask already zeroes `ahmf`. | Day 30 `6.572574374770603e-05` K, day 240 `1.644836070117868e-02` K, day 360 `1.1225660018551306e-02` K, ladder digest `cf06a8fc7d0e90f2`. | Any of those four moves. |
| R31-P6 | legoESM's `e3f_0vor` builder differs from the compiled statements in exactly TWO named statements: the zero-substitution operand (it uses an UNMASKED four-cell `e3t_0` average instead of `e3f_3d`, `dynvor.f90:937`) and the ORDER (it substitutes before the F-fold exchange instead of after, `:935` then `:937`). | Repairing those two statements, and nothing else, reaches 0 unequal cells against the transcription. | A third statement is needed, or the two leave unequal cells. |
| R31-P7 | The 651.2256783597969 m maximum of rounds 29–30 is NOT a construction defect: it is the genuine gap between two different NEMO arrays. | At the argmax cell the carried `e3f_0` and the transcribed `e3f_0vor` differ by that value while the repaired builder equals the transcription exactly. | The repaired builder still differs from the transcription at that cell. |
| R31-P8 | Every gate binds. | Each plant exits non-zero and the citation plant reports `SYMBOL-NOT-AT-LINE`. | A plant passes. |

Failed predictions remain **REFUTED** and are not rewritten.

## Landing and stop rules

- Commit this preregistration before implementing or measuring.
- Reuse the admitted ORCA2 deck and record, fp64/libm policy, CPU backend,
  production JIT, forcing, Decision-52 entry, and the round-1 ten-step ladder.
- Part A lands ONLY if: the given-entry operator identity stays bit-exact, the
  ladder completes kt=1..10, no ORCA2 row labelled "given NEMO's entry"
  worsens, GYRE is byte-identical, and the DINO / LOCK_EXCHANGE / OVERFLOW
  and generic card gates plus the ORCA2 push gate all pass.
- Part B runs NO trajectory arm unless its construction gate reaches 0
  unequal cells.
- No stabilizer, clipping, NEMO-source edit, sea-ice edit, score change,
  configuration default change, or acquisition is authorized.

## Choices

ASKED: Decision 54 authorizes the whole `dyn_ldf` attribution; round 31's
order authorizes landing its lateral-diffusion half first and gating the
`e3f_0vor` construction without further trajectory arms.

UNASKED: none.

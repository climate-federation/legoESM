# ORCA2 round 107 — EEN U-accumulator operand walk

Date: 2026-10-02. Base `e72a4839eb6fdc1c5b531afce684240aabb9cdec`.
Scope is ocean-only measurement on hierarchy rung 0. Every numerical result
below is **independent**: the rung starts from NEMO's own from-rest state.

## Verdict

**STOPPED_FOR_RECORD.** The first source-shaped arm is refuted. The next arm
proves that NEMO's literal `mbku` loop owns all 438 magnitude differences in
the first `ffu_nw` accumulator, but 3,577 signed-zero differences remain and
the southern U corners retain 68 non-fold magnitude differences. No model
physics, card field, configuration value, threshold, stabilizer, carried
state, sea-ice selector, or ORCA2 `unmeasured_features` entry changes.

A fresh two-rank acquisition is preflight-ready. It records the per-level
`zpvo_nw`, two live thicknesses, neighbor mask, product, accumulator before and
after, and `mbku` in a self-describing stream. The operator must run it before
the first remaining signed-zero statement can be named.

## Reproduced baseline and source order

The committed probe first reproduces round 106's entire eight-accumulator
census. For `acc_u_nw` this is 3,953 unequal bits, of which 438 are magnitude
differences and 3,515 are signed-zero-only. Both rank records cover the 148 by
180 native domain exactly once; their admitted digests remain
`afb887211a66...` and `50d8c51f16bf...`.

The compiled source constructs the three-term `zpvo_nw` and applies the live U
thickness, live V thickness, neighboring V mask, and triad in one recurrence
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1231-1244`). Its U
vertical loop ends at `mbku` (`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1229-1230`);
the V sibling ends at `mbkv`
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1262-1263`). Fully
dry faces keep a bottom index of one even though their masks are all zero
(`ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dommsk.f90:223-230`). Therefore a
full-depth loop with thicknesses multiplied by the local face mask is not the
same source program.

## Frozen prediction disposition

- R107-P1 is **UNMEASURED against NEMO**. The unchanged-expression control is
  bit-identical to itself, but the admitted record has no per-level `zpvo_nw`.
- R107-P2 and P3 are **REFUTED**. Removing only the trailing own-face masks
  worsens `acc_u_nw` from 438 to 3,150 magnitude differences and from 3,953 to
  6,665 unequal bits.
- R107-P4 and P5 are **NOT RUN** because no production statement landed.
- R107-P6 is **CONFIRMED**. Unmasked source thicknesses under the literal
  `mbku` loop reduce `acc_u_nw` magnitude differences from 438 to zero.
- R107-P7 is **REFUTED**. The same arm leaves 3,577 signed-zero differences.
- R107-P8 is **REFUTED**. `acc_u_nw` and `acc_u_ne` become magnitude-exact,
  but `acc_u_sw` and `acc_u_se` retain 68 non-fold magnitude differences each.

The literal-loop census is:

| accumulator | unequal bits | magnitude unequal | non-fold magnitude | fold magnitude |
|---|---:|---:|---:|---:|
| `acc_u_nw` | 3,577 | 0 | 0 | 0 |
| `acc_u_ne` | 3,577 | 0 | 0 | 0 |
| `acc_u_sw` | 3,644 | 68 | 68 | 0 |
| `acc_u_se` | 3,644 | 68 | 68 | 0 |
| `acc_v_nw` | 3,578 | 67 | 0 | 67 |
| `acc_v_ne` | 3,577 | 67 | 0 | 67 |
| `acc_v_sw` | 3,647 | 0 | 0 | 0 |
| `acc_v_se` | 3,647 | 0 | 0 | 0 |

The V fold pair is only reported here; round 106's cancellation finding is not
reframed or landed. The later 68-cell substep-2 U residual also remains
separate.

## Acquisition and controls

The fresh target is `ORCA2_OMIP_L4_R107EENSTEP`; its requested run directory
is `round107/acquisition/orca2_rung0_een_step_ranked_10step_np2`. The committed
patch is additions-only. Its writer opens one absolute, rank-tagged stream at
initialization, writes once, and leaves the model arrays untouched. The checker
parses each field name, rank, and dimensions from the record header rather
than predicting byte counts. Admission requires exactly-once rank coverage,
the final recorded `ffu_nw` accumulator to be bit-identical to round 105, and
all twenty terminal restart shards to be byte-identical.

Fortran preprocessing, module compilation, and patched `dynspg_ts` syntax all
pass. The clean committed preflight ends:

```text
SYNTAX_PROOF_PASS l4_r107_een_step.f90 dynspg_ts.f90
ORCA2_ROUND107_EEN_STEP_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round107/acquisition/orca2_rung0_een_step_ranked_10step_np2
```

The layout, absolute-path, duplicate-write, producer-content, and rank-log
plants all fire. The offline operand walk's current-census and candidate-bit
plants also fire. Focused probe and acquisition tests pass 6/6.

## Validation and review

- Measurement execution: production JIT, CPU, fp64/x64, libm.
- The round changes no `packages/` file, so GYRE, DINO, tank, rung-0 trajectory,
  and rung-7 trajectory gates are unchanged by construction.
- Citation gate and prescribed battery results are recorded in the final
  validation addendum below.
- Independent read-only review verdict is recorded in the final validation
  addendum below.

## OPEN

1. Operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round107_een_step_acquisition/run.sh`.
2. Admit both self-describing records and all twenty restart comparisons.
3. Compare the first signed-zero cell in source order: `zpvo_nw`, live U
   thickness, live V thickness, neighbor mask, product, carried accumulator.
4. Walk the separate 68-cell south-U accumulator debt only after `ffu_nw` is
   bit-exact; then return to the northern V cancelling pair and later substep-2
   residual.
5. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy deliverables.

## UNVERIFIED

- Which per-level operand first owns the 3,577 signed-zero differences.
- Whether the 68 south-U magnitude cells share that owner.
- Runtime success, restart identity, and observational passivity of the new
  acquisition until the operator runs it.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

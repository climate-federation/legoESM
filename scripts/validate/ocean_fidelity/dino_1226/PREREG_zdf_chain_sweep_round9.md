# Preregistration: row-12 literal matrix rerun

Date: 2026-08-28. Status at commit: **production fix not implemented and
post-fix measurement not run**.

The accepted fix adds `tke_matrix_evaluation="nemo_literal"`. It is the
faithful default only on `nemo_dino_kamm` and `nemo_dino_kamm_mlf`;
`factored` is their explicit historical opt-in and remains the default for
every other card. The literal path carries raw `e3t_0*(1+r3t)` and restart/
previous-step `dissl`, then evaluates `zcof`, `zd_up`, `zd_lw`, and `zdiag`
in `cfgs/DINO/MY_SRC/zdftke.F90:499-510` source order before calling the
existing differentiable Thomas solver.

## Frozen row-12 rerun

The post-fix probe scores, in coefficient order:

1. `zcof=(-0.5*rn_Dt)*tmask`;
2. upper and lower `MAX(p_avm+p_avm,2e-5)` sums;
3. raw live `e3t(jk,Kmm)` and `e3t(jk-1,Kmm)`;
4. raw live `e3w(jk,Kmm)` in its control-volume slot;
5. carried `dissl(jk)` at line 510;
6. final `zd_up`, `zd_lw`, and `zdiag` delivered to the first Thomas solve.

Each checkpoint uses the existing `1e-15` normalized maximum-per-column bar,
all 9,920 wet columns, and the four registered southern focus columns.
VERIFIED requires 0/9,920 failures, 4/4 focus passes, finite values, and all
planted perturbation/roll/nonfinite controls firing. Expected result if the
registered owner and construction are complete: all six checkpoints and all
three coefficients pass with zero failures. Any nonzero failure stops the
sweep inside row 12 and is localized in the expression order above.

If row 12 verifies, row 13 resumes under the original committed call table and
bars in `PREREG_zdf_chain_sweep.md`. Rows 13--32 are not inferred from a row-12
pass. Climate arms remain unauthorized until row 32 is VERIFIED.

## Required production tests

- a hand-computed nonuniform column with `e3t != e3w`, including surface and
  bottom rows, matching the literal coefficient values;
- a planted replacement of `e3t` by `e3w` that changes the matrix;
- a carried-`dissl` test showing a poisoned recomputed mixing length cannot
  change line 510;
- eager/JIT/gradient finiteness;
- exact `factored` default/output pins for every non-DINO card and an explicit
  `factored` DINO historical control;
- restart bridge and one-step carry tests for `dissl` and raw `e3t_0`.

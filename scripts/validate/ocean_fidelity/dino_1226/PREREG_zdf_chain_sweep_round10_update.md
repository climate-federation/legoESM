# Preregistration: row-10 line-463 update association

Date: 2026-08-28. Frozen after the first literal-source rerun and before the
line-463 update fix or its rerun.

The first rerun localized the remaining row-10 miss to exactly two of 9,920
columns (maximum normalized column error `1.997411720059583e-15`); all four
southern focus columns passed. Every source operand and the complete literal
source rate scored 0/9,920. The remaining operation is therefore the final
NEMO statement at `cfgs/DINO/MY_SRC/zdftke.F90:463`:

```fortran
en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji) * &
               ( zwlc * zwlc * zwlc ) / zhlc(ji)
```

legoESM still passed a rate already divided by `zhlc` to the generic solver,
which evaluated `en + rn_Dt * source`. The registered fix carries a separately
assembled, source-ordered post-Langmuir `en` into the literal matrix RHS while
retaining the source-rate channel as a diagnostic. It must evaluate
`((rn_Dt * zus3) * (zwlc*zwlc*zwlc)) / zhlc` and then add it to `en` only on
the guarded levels. The path is permitted only for the one-iteration literal
DINO chain; all historical source-rate paths remain unchanged.

The frozen bar remains `1e-15` per column. CONFIRM is 0/9,920 failures and
4/4 southern focus passes in the direct post-Langmuir dump. Any nonzero failure
REFUTES closure and stops at row 10. A red hand case must show that moving the
division ahead of `rn_Dt` changes at least one bit.

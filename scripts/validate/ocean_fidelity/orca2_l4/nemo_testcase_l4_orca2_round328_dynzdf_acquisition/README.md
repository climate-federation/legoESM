# ORCA2 round-328 `dyn_zdf` acquisition

This operator-only acquisition re-runs the admitted round-326 ten-step,
two-rank ORCA2 trajectory with additions-only, write-only hooks in `dyn_zdf`.
It emits one kt=1, stage-3 record per rank. The record contains the input
viscosity and drag operands, four RHS boundaries, the U/V tridiagonal matrix
after each source-order construction block, both implicit-solve outputs, and
the ordinary post-`dyn_zdf` barotropic-correction endpoint.
It never reads recorded data back into production state.

The default invocation is a non-running preflight:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh
```

Only the operator may acquire it, using `--run`. The script pins the parent
binary and `dynzdf.F90`, compiles the patched units, enforces the 150-GB memory
rule, proves the patch is additions-only, checks the D150/vector/split-explicit
selectors, compares every inherited frame and trajectory product byte for
byte, and requires all eight admission plants to fire. Evidence is written to
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition/orca1ice_dynzdf_kt1_np2`.

After interruption, `--admit-existing` performs only the immutable passivity
and record-admission checks against an already completed target.

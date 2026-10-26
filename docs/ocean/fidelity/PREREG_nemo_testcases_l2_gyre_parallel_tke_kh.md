# GYRE L2 parallel TKE K_H statement walk — preregistration

Date: 2026-09-16.  Parent tip: `3e7a15c1e64e`.  This is a held-candidate
investigation under user decision 41; it cannot land before the stage walk
reaches its owning stage.

## Frozen evidence and protocol

The admitted oracle is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round59/oracle_tke_operands/oracle_tke_operands_kt00000002.bin`,
SHA-256 `b8f3bead4f30b78257153a0c40c5dd77656aee227b1a94a46625a5604faaca5d`,
producer `1a695951be1396abdd4d7a85f57b31de9c0917df`.  The existing committed
reader/walk is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round54_tke_operands.py`.
All model evaluations use CPU, JAX fp64, and the repository's scalar-libm
precision policy.

The compiled program calls `tke_tke` and then `tke_avn` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:190-201`.  On the
recorded GYRE branch, `nn_mxl=3`, `ln_mxl0=.TRUE.`, and `nn_pdl=1`.  The
executing `tke_avn` statements initialise and derive the mixing length at
`:589-629`, apply the `nn_mxl=3` bounds at `:634-683`, form the raw
diffusivity at `:690-695`, and multiply it by the stored inverse-Prandtl
factor at `:699-702`.

The statement walk will score each of those boundaries bit-for-bit against
the record, in compiled execution order, using NEMO's own recorded entry and
intermediate operands.  It will not substitute legoESM's live entry state.

## Frozen predictions and falsifiers

1. **Baseline reproduction.**  The unchanged production walk will reproduce
   the receipt's `K_H` row: `5,310 / 17,400` unequal consumed interfaces and
   max absolute difference `5.636255351326724e-13`.  Any different count or
   maximum is a provenance/configuration failure and stops the lane.
2. **First non-bit statement.**  All executed boundaries before NEMO's raw
   buoyancy-length assignment at `zdftke.f90:627-630` will be exact.  The
   first non-bit row will be that assignment because the GYRE card currently
   selects the algebraically equivalent factored expression.  An earlier
   non-bit row refutes this prediction.
3. **Locally exact candidate.**  The candidate will select the already-present
   NEMO-literal raw expression, preserve the compiled `nn_mxl=3` terminal
   update at `zdftke.f90:669-683`, and consume NEMO's inverse-Prandtl factor
   directly as `p_pdlr * p_avt` per `zdftke.f90:699-702`.  Given NEMO's
   recorded operands, every statement row through final `K_H` will have zero
   unequal cells.  Any nonzero final row refutes local exactness.
4. **Non-vacuity.**  A one-ULP change to one consumed, non-floor-bound recorded
   entry operand will make its previously exact statement row non-exact and
   the proof command exit nonzero.  A plant that does not flip a row refutes
   the instrument.
5. **Information-only trajectory.**  In a scratch copy only, the isolated
   candidate will be compared with this tip using the certified kt=1..10
   ladder gate.  Every moved row will be reported; no trajectory outcome can
   authorize landing under decision 41.

No model source, configuration, record, gate threshold, or carried state is
changed by this preregistration.

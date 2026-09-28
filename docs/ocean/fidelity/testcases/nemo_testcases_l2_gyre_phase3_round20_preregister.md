# GYRE phase-3 Round-20 preregistration

Frozen on merged base `03c6e8d96ff7f69207abdee43ec28e223d083f4e` before
any Round-20 production or gate edit.  The merge parents are Round 19
`c83f73c23ff82cd7148a68bee020ec3568e5d1d8` and the independently reviewed
integration line `b46e617d02a5ddef5d6b1a04d8124db75bcfbaf2`.

All scored runs are CPU, production JIT, binary64, Oracle V2, with the explicit
`PrecisionPolicy.fp64(transcendentals="libm")` NEMO-identity policy.  Each
candidate below is a one-variable private arm.  No GYRE-only public switch and
no shipped NEMO source edit is eligible.

## P20-0 — 19-frame gate route repair

The gate failure is a harness ownership prediction, not a physics prediction.
`LatLonCGridOceanModel.step` forces the compiled production kernel even inside
an outer `jax.disable_jit` context, while the gate captures
`_nemo_substep_trace_test_hook` arrays in a Python closure and calls
`np.asarray` before the compiled result returns.  The already-shipped
`_NEMOWSBarotropicTrace` returned pytree (`expose_barotropic_substeps`) is the
registered WRITE-only seam.

- **CONFIRM:** delete both gate-side `disable_jit` uses, receive the trace as
  part of the compiled step result, materialize it only after return, recover
  `13 passed`, and reproduce the integration census 19-frame verdict.
- **REFUTE:** any gate requires eager execution, any production field changes,
  or the census verdict changes.
- **Control:** the existing planted entry/exit runs remain nonzero.

## P20-1 — precision provenance

Five census-probe paths currently select dtype-only `PrecisionPolicy.fp64()`;
four later paths select explicit scalar-libm.  The prediction is a provenance
repair with zero numerical movement for OVERFLOW, whose executed per-step path
contains no transcendental.

- **CONFIRM:** every NEMO-identity command resolves to scalar-libm and the five
  rerun census rows are bit-identical to their dtype-only baseline.
- **REFUTE:** any row moves; the first operator and bits become a finding.

## P20-2 — coastal surface-stress handoff

NEMO forms U/V-point stress literally in `src/OCE/SBC/sbcmod.F90:539-546`:
the adjacent T-point sum is multiplied by `0.5`, by `(2-umask)` or
`(2-vmask)`, and by the adjacent-T-mask maximum.  The predicted owner is the
missing two mask factors at wet coastal faces; each written operation will use
the shared source-rounding helper.

- **CONFIRM:** only coastal U/V faces move, the direct NEMO-input operator is
  bit-exact, and the old handoff fails its synthetic coastal operand test.
- **REFUTE:** an interior face moves or the literal operator does not reproduce
  dumped NEMO stress.

## P20-3 — stage-2 tracer input

The ordered NEMO tracer program is `tra_adv`, `tra_sbc_RK3`, then the QCO
stage update at `src/OCE/stprk3_stg.F90:508-565`; stage-3-only QSR/LDF/BBC/ZDF
calls at `:568-603` cannot own a stage-2-entry residual.  Candidate ranking:

1. association/order inside `tra_adv` at the already-identified one-bit cells;
2. `tra_sbc_RK3` accumulation placement;
3. the QCO numerator and division at `:552-554`.

- **CONFIRM:** the first source-local substitution removes at least one ulp at
  the child, its magnitude scales to that child, and stage-2 Kaa becomes
  bit-exact after the ordered register exhausts.
- **REFUTE:** the substituted boundary does not move the child.  It is then
  labelled refuted, not exonerated.

## P20-4 — downstream register

Only after stage-2 Kaa clears, compare stage-3 `zFu/zFv/zFw`, then the actual
`tra_zdf` and `dyn_zdf` matrix coefficients/RHS/solutions, and finally TKE.
Each boundary is dumped WRITE-only from the executing config-local `MY_SRC`,
scored before its child, and receives an owner label only after a scaling and
causal arm.  If an upstream boundary remains open, downstream rows are
`UNMEASURED` or `OWNER_UNMEASURED_UPSTREAM`; they are not tuned.

## Cross-card stopping rule

After every shared numerical change, run cellwise oracle-relative OVERFLOW and
LOCK stage/trajectory gates with row-scale ulp.  Quote the merged-tree
OVERFLOW trajectory against both required baselines: Round 19 (four worsened
rows, maximum 5.5 ulp, no status change) and the integration census table
(49/50 identical, kt2 SSH DEBT to AT-BAR).  A changed operator must first be
bit-exact on NEMO inputs; any faithful-but-worse row remains named debt under
Rule 12.

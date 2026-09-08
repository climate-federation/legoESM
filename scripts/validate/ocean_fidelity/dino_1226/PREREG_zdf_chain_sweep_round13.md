# Preregistration: row-18 libm-lowering ownership

Date: 2026-08-28. Status: **bars and decision frozen; not measured**.

Rows 1--17 are verified (row 16 is waived). Row 18 misses its registered
`1e-15` per-column bar in 173/9,920 columns, while all four southern focus
columns pass. The source-ordered operand peel has verified the dumped
`-gdepw(Kmm)/htau` argument in 0/9,920 columns and, after substituting that
oracle argument, finds 59/9,920 columns over bar at the immediately following
`EXP` in `cfgs/DINO/MY_SRC/zdftke.F90:590`.

Campaign Rule 1b permits this numerical difference to clear without a new
elementary-function implementation if it is proved to live wholly in the
oracle's own host arithmetic. No existing ctypes/libm ownership probe was
found. The new offline probe will reuse `zdf_chain_sweep.py`'s registered dump
loader, focus registry, per-column metric, SHA helper, and time-level registry.

## Frozen measurement

Inputs are the write-only NEMO dumps `tke_dump_etau_argument.bin` and
`tke_dump_etau_exp.bin` from the bracketed one-step CPU run. Select exactly the
59 wet columns that fail the isolated JAX/XLA-versus-NEMO EXP comparison at
the unchanged `1e-15` row-18 bar. Score every wet vertical element in those
columns; do not select individual levels post hoc.

For each selected dumped argument `x`, evaluate:

1. `libm = ctypes.CDLL("libm.so.6").exp(x)` with scalar fp64 calls;
2. `xla = jax.jit(jnp.exp)(x)` on the CPU with x64 enabled;
3. `oracle_minus_lego = nemo_dump_exp - xla`;
4. `libm_minus_xla = libm - xla`;
5. `ownership_residual = oracle_minus_lego - libm_minus_xla`.

The residual column score is
`max_k(abs(ownership_residual))/RMS(nemo_dump_exp)` on the selected wet
population. Its frozen bar is `1e-16` per column. The probe also reports direct
bitwise equality and ULP distances for `libm` versus the NEMO EXP dump.

The host reports glibc 2.34 before measurement. The probe must independently
record `gnu_get_libc_version`, the resolved libm path/`ldd` line, and prove the
instrumented NEMO binary imports `exp@GLIBC_2.29`. It stamps the probe, inputs,
oracle source, oracle binary, mesh, MLD map, Python, NumPy, JAX, jaxlib,
platform, CPU backend, and x64 policy.

## Frozen decision

- **CONFIRM / `WAIVED-LIBM`:** all 59 selected columns have residual
  `<=1e-16`; the selected-column set reproduces exactly; the planted control
  fires; no nonfinite value occurs; and the already-registered four southern
  focus columns pass row 18. Record the reason as: ULP-level transcendental
  lowering difference, glibc-version-dependent, bounded by the measured row-18
  maximum `2.2779670337896653e-15`. Continue immediately to row 19 without a
  production EXP reimplementation.
- **REFUTE:** any selected column exceeds `1e-16`, selection is not exactly 59,
  provenance fails, or the planted control does not fire. Row 18 remains
  `DIVERGED`; the registered `tke_etau_exponential_evaluation=nemo_literal`
  design proceeds and rows 19--32 remain unmeasured.

## Planted control

Scan selected wet `(j,i,k)` elements in row-major order and choose the first
whose argument's one-ULP step toward `+inf` changes scalar libm `exp`. Replace
only that libm input with the stepped value while keeping the oracle and XLA
arrays fixed. The control must make that column exceed the same `1e-16`
ownership bar. If no eligible element exists or the planted column still
passes, abort without a waiver.

This probe changes no production configuration or physics and runs CPU-only.

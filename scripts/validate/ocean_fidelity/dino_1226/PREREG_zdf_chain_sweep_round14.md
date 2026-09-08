# ZDF chain sweep round 14 preregistration — literal vector EXP

Date frozen: 2026-08-28.  This file is committed before measuring a JAX
implementation of the candidate arithmetic.

## Trigger and oracle operand

The round-13 scalar `ctypes` discriminator REFUTED the proposed LIBM waiver:
all 59 selected row-18 columns miss the registered `1e-16` ownership-residual
bar.  Row 18 therefore remains `DIVERGED` and the already-registered
`tke_etau_exponential_evaluation=nemo_literal` production option proceeds.

The linked DINO oracle imports both scalar `exp@GLIBC_2.29` and vector
`_ZGVbN2v_exp@GLIBC_2.22`.  Disassembly of the active vector call resolves to
the glibc 2.34 two-lane implementation in `/usr/lib64/libmvec.so.1`.  An
offline, non-production reconstruction of that implementation matched all
332,214 wet values in `tke_dump_etau_exp.bin` bit-for-bit.  That observation
only targets the implementation below; it is not the registered production
measurement.

## Candidate, scope, and frozen bars

Add `TKEConfig.tke_etau_exponential_evaluation` with values:

- `nemo_literal`: reproduce the ordinary-range glibc 2.34 vector algorithm in
  source order: 1024-bin range reduction, split `ln(2)/1024`, cubic residual
  polynomial, bit-assembled scale, final multiply.  The 1024-entry table is
  captured as exact binary64 bit patterns from the linked oracle library and
  stamped by SHA-256.  No host `libm` call is permitted in production.
- `jax_expression`: retain the existing single `jnp.exp` expression exactly.

`nemo_dino_kamm` and `nemo_dino_kamm_mlf` select `nemo_literal` by default.
Every other DINO card and every direct `TKEConfig()` construction retain the
byte-identical `jax_expression` default.  The two oracle cards may opt back
into `jax_expression` for the legacy control.

The literal path is accepted only if all of these hold:

1. Direct comparison with `tke_dump_etau_exp.bin`: **0 / 9,920 columns fail**
   the row bar `1e-15`; all four registered southern-basin focus columns pass.
2. The complete row-18 penetrating-TKE field: **0 / 9,920 columns fail** the
   same bar; all four focus columns pass.
3. On every wet dumped argument in the ordinary range used by DINO, the
   literal helper is bit-identical to the dumped NEMO EXP operand.
4. A red control changes one selected dumped argument by one ULP toward
   `+inf`; the direct exactness check must fail.
5. `jax.jit`, forward-mode AD, and reverse-mode AD execute with finite values.
6. Explicit byte-identity pins show `jax_expression` equals the pre-change
   expression and every unchanged DINO card still selects it.

Exceptional EXP inputs outside the DINO row-18 operand range retain
`jnp.exp`; the literal ordinary-range path is selected elementwise.  Tests
must cover the range guard.  This keeps the new path a DINO-oracle arithmetic
option rather than a process-global transcendental replacement.

If either row-18 bar fails, stop at row 18 and localize the first remaining
operation.  If both pass, continue at row 19 in the committed execution-order
table and stop at the first `DIVERGED` row.

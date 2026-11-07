# NEMO testcase lane 2 GYRE phase 3 round-16 preregistration

Base: `b9311e65de2b784b4f12c1ceffc4ad4e394b4e74` (round 15).
Regime: production JIT, CPU, fp64, scalar `libm.so.6`; oracle V2 is
`round15_oracle_v2_scalarmath_sbc`.  No shared numerical code changes before
the discriminator result.

## D1 — scalar-math eligibility discriminator

Oracle statements are the executed preprocessed `traqsr.f90:615-645`
(`zz0`, `zz1`, `zatt`, `zzatt`, and the RK3 increment; especially
`:621,629-630,642`) and shipped `usrdef_sbc.F90:84-145,153-184`.  The
independent transcription will use the oracle-dumped `qsr`, stage-3 `r3t`,
`gdepw_1d`, `e3t_0`, `gphit`, `tmask`, and stage-entry surface T, preserving
each Fortran statement boundary and its precomputed reciprocals.

The discriminator produces, for every qsr intermediate/increment and every
SBC field, the count of binary64 cells differing from oracle V2, maximum ULP,
and the first differing operand.  It also compares the exact transcendental
arguments and results against scalar glibc.

- **CONFIRMS H1:** scalar-libm results equal scalar glibc at every executed
  call, the card demonstrably routes through the callback, and the independent
  source-literal transcription reduces or eliminates the 15,891 qsr-increment
  mismatches and/or the qns/emp mismatches.  The first mismatch in the current
  card must precede the transcendental result or occur at a later statement
  boundary.
- **CONFIRMS H2:** the independent source-literal transcription's
  transcendental input equals NEMO's, but the result differs from scalar
  glibc, or the poisoned-callback control does not propagate through the
  production card.
- **REFUTES BOTH / STOP:** the first difference is an unavailable or already
  different oracle input; instrument that operand before proposing a fix.

The routing control poisons one scalar-libm result by one representable step.
The unpoisoned production path must match direct ctypes scalar-libm bits, and
the poison must change both the direct shared wrapper and the card's executed
qsr output; otherwise the gate exits nonzero.

## D2 — conditional landing and stop rule

If D1 confirms H1, change only the shared qsr/SBC statement association, using
the public NEMO source-round primitive at statement boundaries.  Success is
zero wet-cell bit mismatches for the qsr increment and all five dumped SBC
fields.  Any remaining mismatch stops the round at its first differing
operand.  The stage-1 slow-forcing walk is entered only after that exact gate
passes.

## D3 — V1/V2 ancestry and gate follow-ups

Independently of D1's landing, classify all 41 V1/V2-changed records by the
first executed transcendental ancestor.  A record without such an ancestor is
a finding.  Replace the cell-local ULP admission scale with
`spacing(max(max(abs(oracle_row)), 1.0))`, rerun all four OVERFLOW/LOCK shared
stage/trajectory comparisons and their three planted controls, and relabel the
DINO correlation-only third-consumer measurement as a consistency probe.

## D4 — conditional stage-1 slow-forcing walk

Entered only if D2 is bit-exact.  The executed RK3 path is `stp2d.F90`:
`eos/dyn_hpg/dyn_ldf/dyn_vor/wzv/dyn_keg/dyn_zad` accumulate the 3-D `Krhs`
at `:126-171`; the vector-invariant arm depth-means it with
`SUM(e3u_0*Krhs*umask)*r1_hu_0` and the V analogue at `:177-181`, applies
`dyn_drg_init` at `:196`, and adds wind stress at `:198-202`.  Only then does
`dynspg_ts.F90:280-300` copy `Ue_rhs/Ve_rhs` to `zu_frc/zv_frc` and remove the
barotropic Coriolis trend; the non-RK3 depth-mean block at
`dynspg_ts.F90:310-346` is dead in this configuration.

A config-local, WRITE-only `stp2d.F90` record will dump the level operands,
the completed depth mean, post-drag value, stress increment, and post-wind
value.  Ranked candidate owner, preregistered before that run:

1. **CONFIRM candidate:** legoESM's fused `jnp.sum(stack(...))` reduction and
   divide do not reproduce the source-ordered per-face Fortran `SUM` product,
   while `e3*`, `Krhs`, masks, and reciprocal depth are bit-exact.  A shared
   source-ordered reduction arm removes the existing `6.00e-21` slow-forcing
   gap.
2. **Second candidate:** the separately reconstructed face thickness/column
   depth differs from NEMO's `e3u_0/r1_hu_0` or V counterpart.
3. **Later candidates:** `dyn_drg_init`, then the association of
   `r1_rho0*utauU*r1_hu(Kbb)` into the depth mean.  These are tested only if
   the preceding boundary is exact.

Scaling is evaluated before any owner label: each first operand residual is
propagated through the remaining literal statements and compared with the
post-wind `zu_frc/zv_frc` residual.  A one-variable arm substitutes exactly
one oracle operand.  The plant flips one live dumped operand and must exit
nonzero.

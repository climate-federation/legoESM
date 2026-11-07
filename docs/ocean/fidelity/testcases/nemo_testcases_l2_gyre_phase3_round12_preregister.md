# NEMO testcase lane 2 GYRE — phase-3 round-12 preregistration

Date: 2026-09-04
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`
Reviewed baseline: `bef8033900877e1e1eefb704a69bc05080c21064`

## Frozen boundary and ranked prediction

Round 11 localized the stage-2 momentum debt upstream of literal HPG: replacing
the complete HPG operands with NEMO's values reduces its u/v residuals to
`2.40e-20`, while the production `rhd` operand first differs by
`9.378348791999613e-16`.  Stage-2 `rDt=7200 s` makes the effective tendency bar
`1e-15/7200 = 1.388888888888889e-19`; therefore `rhd` must be bit-exact or within
the few ulps that keep every downstream HPG/Kaa row under the fixed bar.

The preregistered owner ranking is:

1. **Live depth association under QCO.**  NEMO first stores
   `r3t = ssh*r1_ht_0` (`domqco.F90:159-161`) and the macro expands
   `gdept(Kmm)` as `gdept_0*(1+r3t)`
   (`domzgr_substitute.h90:50,56,75,139`).  The shared pressure-gradient path
   presently asks `nemo_bn2_live_ladders` for its default quotient evaluation,
   while the already certified TKE and HPG geometry paths explicitly request
   `nemo_reciprocal`.  Prediction: `pts(T)` and `pts(S)` are bit-identical, but
   `pdep`/`zh` are the first non-bit-exact rows.  CONFIRM if replacing only the
   live-depth construction makes `pdep` and `zh` bit-identical and moves `prd`
   by at least 0.9 of its production residual.  REFUTE if NEMO and candidate
   `pdep` are bit-identical or the one-variable depth arm moves `prd` by less
   than 0.1 residual.
2. **Normalized salinity association/constants.**  NEMO evaluates
   `zs=SQRT(ABS(S+rdeltaS)*r1_S0)` (`eosbn2.F90:262`) after assigning
   `rdeltaS=32` and `r1_S0=0.875/35.16504` (`:1926-1929`).  CONFIRM only if
   `pdep`/`zh` match and `zs` is the first differing row, and a one-variable
   precomputed-reciprocal arm clears it.  Multiplication by stored reciprocals,
   never a replacement division, is the faithful arm.
3. **Coefficient or reference-density bits.**  The 52 `EOS###` values are set
   at `eosbn2.F90:1931-1982`; `rho0=1026` and
   `r1_rho0=1._wp/rho0` at `:1898,2331-2334`.  CONFIRM only if every input and
   normalized coordinate matches but the first `znN`, `zn`, or `prd` departure
   follows a coefficient/reciprocal bit difference and that isolated literal
   replacement clears it.
4. **Wrong EOS selection.**  The resolved run deck pins
   `ln_teos10=.true., ln_eos80=.false., ln_seos=.false.`
   (`round11_oracle_stage2_v5/namelist_cfg:124-129`), and
   `eos_init` maps that single flag to `np_teos10`
   (`eosbn2.F90:1911-1924`).  The card resolves `nemo_teos10`; any disagreement
   is a harness/config defect, not a tuning arm.
5. **XLA association.**  If all operands/constants match and NumPy's literal
   statement transcription is bit-identical to NEMO while production JIT is
   not, ownership moves to compiled association.  JIT and the same callable
   under the existing parity hook must first agree within two ulps; a barrier
   restructuring is permitted only if this discriminator confirms it.

## Source-ordered oracle record

At stage 2 of kt=2, `stprk3_stg.F90:321-324` calls `eos(ts,Kmm,...)` immediately
before `dyn_hpg`.  A config-local WRITE-only `MY_SRC/eosbn2.F90` will record,
in source order, `pts(T)`, `pts(S)`, `gdept(Kmm)`, `zh`, `zt`, `zs`, `ztm`,
`zn3`, `zn2`, `zn1`, `zn0`, `zn`, and `prd`, plus the scalar normalization,
coefficient, and reference-density values used.  The record is accepted only
if its header, dimensions, stage/time level, fp64 dtypes, finite values, and
complete field registry pass, and if ordinary stage/restart files remain
bit-identical to Round 11.  A planted wet-cell mutation must exit nonzero.

Search before implementation found one shared evaluator in `ocean/eos.py`, one
existing NumPy transcription and coefficient parser in
`nemo_testcase_phase3_eos_gate.py`, and the Round-11 stage-state/HPG diagnostics.
This round extends those surfaces; it does not create another EOS module or
constructible selector.

## Ordered consequences

The shared EOS may change only to reproduce the executed NEMO statements; no
GYRE arm or card guard is permitted.  After a confirmed fix, production-JIT
stage-2 HPG/Kaa, GYRE kt=1…10, and the OVERFLOW/LOCK_EXCHANGE stage and
trajectory gates will be rerun.  Cross-card reporting includes both the pending
compare-to verdict and oracle-relative before/after residual rows; the gate and
its criterion are immutable.  Any row moving away from NEMO by more than two
ulps, or crossing the bar downward, is reported without tuning.

Only if EOS and corrected Kaa clear may the stage-3 `zFu/zFv/zFw` operand walk
begin.  The ZDF matrix walk is outside round 12.  No external review of this
preregistration has occurred.

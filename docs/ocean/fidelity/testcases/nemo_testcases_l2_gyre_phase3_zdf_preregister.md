# NEMO testcase lane 2 GYRE — RK3 ZDF time-level preregistration

Date: 2026-09-03  
Session: `01a05cb9-7625-7f40-9e83-b3fa767b1945`  
Reconciled baseline: `57429ecf5f377ce2bf220f36bc05313cd29e0dfd`

This preregisters the next ordered boundary after bottom drag cleared all 800
barotropic frames and the stage-2 source walk localized only sub-bar HPG
roundoff.  It is derived from NEMO source, not from the abandoned round-7 WIP
receipt.

## Source and live mismatch

NEMO RK3 evaluates `eos_rab` and `bn2` on `ts(...,Nbb)`, copies
`rab_n=rab_b` and `rn2=rn2b`, then calls `zdf_phy(kstp,Nbb,Nbb,Nrhs)` before
`stp_2D` or any RK stage (`src/OCE/stprk3.F90:154-181`).  The resolved EVD
branch replaces `avt` and `avm` where `MIN(rn2,rn2b) <= -1e-12`
(`src/OCE/ZDF/zdfevd.F90:88-120`).  Thus both EVD trigger arms consume the
whole-step entry Nbb tracer state in this RK3 program.

The GYRE card currently leaves `evd_n2_time_level="solver_state"`.  Its direct
gate reports candidate `avt=avm=100 m2/s` against oracle maxima
`0.008188154756539789` and `0.08188154756539788 m2/s`.  A source check of the
pinned initial profile gives positive N2 at all 29 wet interfaces
(`4.710361706917593e-9` to `6.78022058531194e-5 s-2`), so NEMO's EVD branch is
off at kt=1.  The candidate's 100 value therefore comes from sampling the
post-stage solver tracer, not from the resolved Nbb program.

## Frozen discriminator

Select the already-canonical `nemo_now_before` EVD time-level machinery and
make its RK3 interpretation source-exact: both arms receive the step-entry
tracers because NEMO sets `rn2=rn2b`.  No new public switch is permitted.

- CONFIRM the time-level owner if candidate avm/avt no longer contain the
  `100 m2/s` EVD replacement and the oracle avm/avt causal injection moves by
  less than 10% of the former whole-step u/v residual.
- REFUTE it if any active coefficient remains `100 m2/s`, or if the kt=2
  u/v movement is below 10% of the `0.0252569 m/s` residual.
- The production coefficient rows and kt=2 fields retain the exact `1e-15`
  gate.  Any remainder is DEBT; this arm cannot clear unrelated TKE coefficient
  or stage-3 source errors.

A recipe test must pin the EVD selector, a time-level unit test must prove the
RK3 before operand is the entry tracer, and the existing oracle-coefficient
injection remains the one-variable causal control.  No external adversarial
review has occurred for this preregistration.

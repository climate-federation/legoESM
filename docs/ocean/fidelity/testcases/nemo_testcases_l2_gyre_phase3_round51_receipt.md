# NEMO testcase L2 GYRE phase-3 round-51 receipt

## Verdict

The preregistered claim is **REFUTED**.  The first unequal live kt=2 operand is
stage-1 `u_Kmm`, not SSH: 17,399/17,400 wet cells differ, maximum
`2.7478404751243857e-12 m s-1`; `v_Kmm` follows at 17,100/17,100 and
`3.305560306813421e-12 m s-1`.  Thus the round-46 given-input operator results
were exact because they injected NEMO's operands; the independent live program
already enters kt=2 with legoESM's unequal kt=1 result.  The observer control is
bit-exact for returned T/S/u/v/SSH (zero unequal fields).

This is the executed selection: legoESM binds `u0/v0` directly from the live
state (`ocean_model_latlon_cgrid.py:5401-5402`) and hands those values to the
stage-1 tendency (`:6317-6322`).  NEMO calls `stp_2D` before its three RK stages
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-215`), and compiled
stage 1 calls EOS/HPG/LDF/VOR/WZV/KEG/ZAD on Kbb
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175`).  The stage identity is therefore right;
the values have upstream kt=1-exit debt.

## Live handed operands

| stage | first/live-state rows | geometry / other selection |
|---|---|---|
| 1 | U `2.748e-12`, V `3.306e-12`; T `1.421e-14`, S `2.132e-14`; SSH `4.337e-19` | e3 Kmm/Kbb exact; r3 `~1.1e-16`; W `7.908e-7`; ZAD/WZV `r1_Dt` exact |
| 2 | U `9.173e-6`, V `9.389e-6`; T `2.188e-11`, S `1.499e-12`; SSH `2.358e-7` | e3 `1.65e-8`; r3 `5.48e-11`; W `4.833e-7`; live `r1_Dt` selects full dt instead of NEMO dt/2 |
| 3 | U `7.546e-6`, V `9.986e-6`; T `8.369e-7`, S `6.795e-8`; SSH `3.536e-7` | e3 Kmm `2.47e-8`; LDF Kbb thickness selection differs `1.103e-4`; W `4.833e-7`; `r1_Dt` exact |

The complete per-field/count table is the stamped JSON.  NEMO sets
`rDt=rn_Dt/3,/2,1` and `r1_Dt=1/rDt` at compiled
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:138-146,196-200,236-244`; stage 1's precomputed RHS instead
comes from `stp2d`.  Later-stage rows are downstream observations, not causal
owners once stage 1 is non-bit.

## Decision 33: raw barotropic memory

Substituting only round-48 raw NEMO Ub/Ubb/Vb/Vbb/SSHb/SSHbb through the model
path is **REFUTED**.  Against the kt=3 oracle, U/V maxima remain exactly
`7.193412341835043e-4 / 8.606873052712932e-4`; SSH worsens from
`7.072558982027660e-7` to `7.072558982030913e-7`.  The substitution itself
changes U 13,076 cells (`1.862e-15`), V 11,991 (`1.894e-15`), and SSH 563
(`1.046e-17`).  NEMO's AB3-AM4 use and rotation are compiled
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:456-489,749-761`;
legoesm reconstructs deviations at `barotropic_latlon_cgrid.py:2063-2085` and
stores them at `:2890-2901`.
Decision 33 remains pending; the raw-carry patch is preserved under manifests
and explicitly not applied.

## Gates, Rule 12, and disposition

The GYRE kt=1..10 first-over-bar remains kt=2 U/V, before=after
`2.7478405293344943e-12 / 3.305560415233638e-12`; no production fix landed.
ZAD was not rerun because item 1 landed no fix.

| card | disposition |
|---|---|
| GYRE-zco | TESTED: preregistration and raw-history candidate REFUTED |
| LOCK_EXCHANGE-zco | VALUE-INERT: private hooks default off; no production edit |
| OVERFLOW-zps | VALUE-INERT: private hooks default off; no production edit |
| ORCA2 | UNMEASURED-WITH-SPEC: acquire the same live stage/history records |
| DINO | SHARED-STATEMENT RISK: separate resolved program; do not infer |

Unit/regression tests: 14 passed.  Operand, history, and false-stamp plants exit
nonzero.  ASKED: live trace, raw substitution/decision packet, trajectory,
Rule 12.  UNASKED and untouched: NEMO build/run/source, TKE, year harness,
configuration/state choice, bar relaxation.  Open: localize the kt=1-exit
U/V/T/S owner before interpreting downstream stage-2/3 operand debt.

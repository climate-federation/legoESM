# NEMO testcase L2 GYRE phase-3 round-52 receipt

Date: 2026-09-11. Base: `471bf6b399de116997631a2697911b7e87c56b09`.
All measurements used CPU, fp64, production JIT and independent committed
NEMO records. No NEMO source, executable, or record was modified or run.

## Compiled-source decision

`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-215` calls `stp_2D` and then the three stages with fixed
Kbb and swapped Kmm/Kaa. The pre-stage WZV/ZAD executes on the full-step clock
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175`;
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:308-310`). The shared stage routine selects
dt/3, dt/2 and dt at stages 1–3
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:138-146,196-200,236-244`),
calls WZV on `(Kbb,Kmm,Kaa)` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:329-335`, and WZV multiplies
the Kaa−Kbb stretching by the live reciprocal dt
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90:277-298`). Stage 2 also constructs Kaa=N+1/2
(`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:215-235`): clock and SSH delta are a cancelling pair.

Stage 3 calls `dyn_ldf(Kbb,Kmm)` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:690-712`.
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:121-140` reads Kbb velocity and Kbb e3t/e3u/e3v, but divides
the output curl by Kmm e3u/e3v.

## Predictions and measurements

| card | live boundary result | kt=1 exit U / V | verdict |
|---|---|---:|---|
| A, dt/2 alone | stage-2 ADV `9.93e-24/9.93e-24` AT-BAR, but stage-3 ADV `2.46e-10/4.81e-10` DEBT | `2.020e-6 / 3.955e-6` | REFUTED |
| A2, dt/2 + Kaa=N+1/2 | stage-3 ADV `2.06e-16/2.48e-16` AT-BAR | unchanged `2.748e-12 / 3.306e-12` | REFUTED: no strict improvement |
| B, six LDF operands | all five recorded Kbb/Kmm thickness rows bit-exact; post-LDF `2.06e-16/2.48e-16` AT-BAR | unchanged `2.748e-12 / 3.306e-12` | REFUTED by trajectory |

The A prediction failed first on GYRE: its Rule-12 result
worsened 62 rows, moved first-over-bar from kt2 U/V to kt2 T/S/U/V, and had a
worst worsening of `77,310,976,623.0625` ulps. The paired A2 result passed
Rule 12 (11 sub-bar moved rows, worst `0.00390625` ulp) but did not reduce the
exit debt, so neither A arm landed.

| A Rule-12 card, kt1–10 | before → candidate first-over-bar | moved / worsened rows | verdict |
|---|---|---:|---|
| GYRE-zco | kt2 U/V → kt2 T/S/U/V | 62 / 62 | FAIL |
| LOCK_EXCHANGE-zco | kt4 U → kt4 U | 0 / 0 | PASS |
| OVERFLOW-zps | kt2 T/U → kt2 T/U | 0 / 0 | PASS |

The two tanks resolve `rk3_ws` with the generic WZV arm; their clock candidate
artifacts are bit-identical to baseline at all 50 registered rows.

| B Rule-12 card, kt1–10 | before → candidate first-over-bar | moved / worsened rows | verdict |
|---|---|---:|---|
| GYRE-zco | kt2 U/V → kt2 U/V | 53 / 53; worst `94,289,849.5` ulps | FAIL |
| LOCK_EXCHANGE-zco | kt4 U → kt4 U | 0 / 0 | PASS |
| OVERFLOW-zps | kt2 T/U → kt2 T/U | 0 / 0 | PASS |

LOCK/OVERFLOW resolve `A_h=0`, `lateral_viscosity_operator=vector_laplacian`,
and `lateral_viscosity_e3_weighting=off`; the rejected literal-LDF handoff is
therefore inert on both. Every moved GYRE row is retained in
`card_b_final_gyre_rule12.json`. ORCA2 remains UNMEASURED_WITH_SPEC: run the
same ten-step residual-sidecar comparison when an independent, pinned ORCA2
entry record with this schema exists. DINO resolves the separate `nemo_mlf`
leap-frog program; the WS-stage call site does not execute there, while the
shared PE/LDF kernel remains a focused-test regression risk.

## ZAD and disposition

The Round-47 ZAD patch was NOT_ENTERED: its explicit precondition—A and B both
landed—was false. Consequently the former 57 worsened rows were not rescored
under an invalid parent state, and no kt3-owner claim is made. Final production
physics equals the round-51 base; GYRE's first-over-bar remains kt2 U/V at
`2.7478404751243857e-12 / 3.305560306813421e-12`.

The writable measurement clone commits were: preregistration `3e3bf969d391`,
gate `df8f97e89262`, A/A2 experimental commits and explicit reverts, Card-B
experimental commits and explicit reverts, ending at `44e6aa6adc9f` before
this receipt. The cross-card A replay/revert is `e885d343bbd6` / `b7f7556812fa`.
The operator checkout's read-only `.git` prevented staging.

The exact-source citation gate passed all nine citations at `9f192341cb6a`.
Its shifted-line plant, the stage gate's bad-stamp plant, and its one-cell
plant each exited nonzero. The clean measurement clone passed all 21 focused
Round-50/52 and citation-gate tests.

## ASKED / UNASKED and self-review

| item | disposition |
|---|---|
| A, A2, B | ASKED measurements; all failed a frozen landing condition and were reverted |
| ZAD candidate | ASKED conditional; NOT_ENTERED because A+B did not land |
| ORCA2 | ASKED; UNMEASURED_WITH_SPEC above |
| DINO | ASKED risk report; separate leap-frog branch, no dynamic transfer claim |
| new configuration, tolerance, carried state, NEMO run/edit | UNASKED/forbidden; none |

Post-code self-review found no retained physics delta, no forcing/TKE/year-
harness edit, no earlier first-over-bar, and no unstamped cited measurement.
Independent Claude review remains the user's stated next review step.

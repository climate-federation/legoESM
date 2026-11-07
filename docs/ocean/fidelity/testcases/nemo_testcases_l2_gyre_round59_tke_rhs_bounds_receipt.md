# NEMO testcase L2 GYRE round 59 — TKE RHS acquisition retraction

Date: 2026-09-11. Frozen record producer and parent
`ee903981f2df22ba75ce7227d6cdef78051d85b5`; CPU/fp64.
**CALIBRATION FAILED; R58 ACQUISITION RETRACTED; MODEL SUBSTITUTION,
PHYSICS LANDING, AND TRAJECTORIES UNMEASURED.** Round 57's preregistration
remains the governing contract.

## Calibration-first verdict

The previously gated terminal closure remains internally exact: Prandtl is
0 unequal over 17,400 wet interior interfaces, and `avm`, `avt`, and `dissl`
are each 0 unequal over 18,000 wet closure interfaces. Those statements are
the R58 record's own compiled
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:394-412` and
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:691-701`; the pre-EVD
copy is
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354`.

The required wider calibration rejects the record before substitution.
Replaying the matrix/RHS recurrences, terminal solve, and wet floor in their
compiled order (`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:466-483`)
gives **1,431 unequal `en` cells over 17,400 wet solved interfaces**. In the
same source-exact gate, both `nn_mxl=3` outputs rebuild with **0 unequal cells
over 18,000 wet interfaces**, including initialization, the stress/tmask
surface anchor, raw buoyancy length, and the two bounds
(`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:589`,
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:601-603`,
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:612-619`,
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:628-629`,
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:634`, and
`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:669-682`). The
executable gate therefore exits 1 rather than accepting the terminal-only
0/0/0/0 counts.

## Root cause and repair

The record's compiled NEMO allocates `en` on the explicit no-halo
`Nis0:Nie0,Njs0:Nje0` domain
(`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdf_oce.f90:85-87`). Its compiled
writer instead declares `p_rhs(jpi,jpj,jpk)` and copies a global-index slice
(`GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:173-189`). That
full-domain dummy is incompatible with the reduced actual array and records
the wrong RHS cells. The post-sweep `en`, recorded directly from the correctly
bounded module array, exposes the inconsistency.

The repaired writer declares `p_rhs(A2D(0),jpk)` at
`nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90:154-170`.
The reader now source-rebuilds `en/zmxlm/zmxld` at
`nemo_testcase_l2_gyre_round54_tke_operands.py:116-199`; the new `sweep` plant
corrupts one consumed RHS value and must exit nonzero. The operator script now
uses the fresh target `GYRE_OMIP_L2_P3_SM_R59TKE` and
`phase3/round59/oracle_tke_operands`. Preprocessing plus
`gfortran -fsyntax-only` passes; the focused gate suite passes 10/10.

## Frozen predictions and Rule 12

No substitution ladder row is admissible, so there is no first non-bit
legoESM statement and no physics change. P1 (derived floor as sole coefficient
owner) is neither confirmed nor refuted; P2's kt3 T/S values
`1.6275031290e-4 K` / `6.3278533133e-6 g/kg` are unscored. The decision-35
after-arm remains the required before arm; the combined decision-35-to-round-56
floor effect and its sub-resolution one-ULP component remain unmeasured.

| card | Rule-12 disposition |
|---|---|
| GYRE | BLOCKED on clean R59 reacquisition; kt=1..10 and days 1-30 unmeasured. |
| legacy `nemo_v1` | UNMEASURED-with-spec; its non-ULP floor move remains registered. |
| LOCK_EXCHANGE | Resolved constant mixing; shared TKE path not executed. |
| OVERFLOW | Resolved constant mixing; shared TKE path not executed. |
| DINO | Shares `tke.py` on a separate branch; no statement was changed, so shared-statement risk is unchanged. |
| ORCA2 | UNMEASURED-with-spec against its own compiled operands. |

ASKED: calibration-first stop, writer/gate repair, new target, plants, tests,
receipt. UNASKED: none. No NEMO source/run, model physics, configuration,
carried state, year harness, reconciliation gate, freshwater pair, or #1484
guard changed.

Open: operator commits this packet and reacquires R59; the new record must pass
all seven exact calibration counts before the source-ordered model walk resumes.

# Candidate preregistration — Round 149 live LDF geometry routing

Date: 2026-09-22

Parent measurement commit: `a4dca09f7fc1b5d886d25d081142d3a95a6f0e07`.
Evidence remains under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round149/`.  This document is
frozen after the developed walk named the first non-bit operand and before the
candidate is implemented or measured.

## Measured statement and candidate

At NEMO's exact day-180 entry, the production step's `ahmf`, metrics, and U/V
inputs are BIT.  The first non-bit factor in compiled source order is the live
F-point thickness in
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:157-176`:
16,530/16,530 wet cells differ, maximum `3.872632379170682e-3 m`.
The later U/V live face thicknesses differ in all 17,400/17,100 wet cells.
NEMO forms those ratios from SSH and the reference face columns at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90:256-286`.

The candidate routes legoESM's existing shared NEMO-QCO live U/V/F thickness
builders into the WS-RK3 LDF call.  At the external stage Kbb=Kmm; at stage 3
the cell and U/V Kbb operands remain step-entry values while F and the outer
U/V divisors use the stage Kmm SSH, matching
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:158-164` and
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stprk3_stg.f90:704-739`.
No new selector or configuration value is introduced.

## Frozen local and trajectory predictions

Given the day-180 NEMO entry, the candidate is predicted to make the recorded
`e3f`, `e3u`, and `e3v` rows BIT and to make the literal `zcur`, `zdiv`, U/V
isolated LDF terms, and post-LDF accumulators BIT.  Any non-bit row before the
accumulator refutes local exactness and vetoes trajectory measurement.

The candidate is predicted to preserve all kt=1 AT-BAR rows and kt2 T/S at the
bar, keep the first-over-bar at kt2 U/V, and reduce the month and year T3D RMS.
The registered headline predictions are: kt2 U/V no larger than
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S no larger than
`8.659373840202989e-7` / `7.027291104577671e-8`, day-30 below
`6.890431487825909e-5 K`, day-240 no larger than
`1.644674023317539e-2 K`, and day-360 no larger than
`1.122357124784366e-2 K`.

The landing falsifier is any failed local row, any kt=1 AT-BAR row leaving the
bar, an earlier first-over-bar, a day-30 value that does not decrease, or a
worse day-240/day-360 value.  Every moved ladder, month, and year row is
registered.  DINO's Euler route must be measured before/after; the prediction
is bit identity because this candidate changes only the WS-RK3 time-level
routing, not the shared operator's default arm.  LOCK_EXCHANGE and OVERFLOW
resolve the operator off.  ORCA2 remains `UNMEASURED-WITH-SPEC` for developed
live LDF geometry.


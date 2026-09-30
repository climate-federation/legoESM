# Preregistration — round 185, carried-N2 day-240 owner re-ranking

Committed before any Round-185 process trace or owner score.  Round 183
reduced the day-240 GYRE temperature RMS from
`1.6448360701178680e-02 K` to `6.5861718814795174e-05 K`; therefore the
Round-124 process ranking is stale.  This round first re-runs that same
day-180-to-240 process budget on the landed carried-step-entry-N2 trajectory.
No second harness, physics change, configuration choice, carried-state change,
or stabilizer is authorized by this preregistration.

Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round185/`.

## Compiled process boundaries

The admitted oracle writer records stage-3 temperature before the process
chain and after each compiled boundary at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830`,
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:849-869`, and
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-970`.
The order is advection, surface boundary, shortwave, lateral diffusion, then
the implicit vertical solve.  The solve forms its thickness-weighted RHS and
performs the forward/backward recurrences at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-578`.

The already committed `nemo_testcase_l2_gyre_year_owners.py` process trace
drives the production JIT step from rest and records those same boundaries for
steps 1081--1440 while carrying only an independently evaluated ordinary
production state.  Round 185 may generalize that gate's frozen endpoint from
the obsolete Round-124 scalar to an explicit command-line expected value; the
expected value remains mandatory and exact.

## Frozen predictions and falsifiers

1. The owner self-check passes before measurement.  The Round-123 NEMO record
   re-admits as exactly 360 ordered frames with all calibration, chain, and
   activity controls unchanged.  Any failed admission stops the ranking.
2. A fresh carried-N2 process trace from the clean Round-185 instrument commit
   reproduces the immutable Round-183 `after_year` member bit-for-bit at days
   180 and 240 in T/S/u/v/SSH, carries zero unequal production-state bytes,
   and reproduces day-240 T3D RMS exactly
   `6.5861718814795174e-05 K`.  Any unequal bit or different endpoint stops
   direction claims.
3. The endpoint budget closes: incoming plus every compiled-order process row
   plus explicit rounding reconstructs the independent day-240 temperature
   error with maximum wet-cell residual no larger than `4e-15 K`; the signed
   carries sum to the endpoint within fp64 summation.  A larger residual
   refuses the ranking.
4. Frozen magnitude prediction: the incoming day-180 state gap is the largest
   absolute signed carry after the carried-N2 landing, because the day-180 and
   day-240 RMS values are now comparable.  Vertical diffusion is predicted
   not to retain its old `+2.4168271578053416e-02 K` carry.  Either prediction
   may be REFUTED; the measured absolute-carry ranking wins and is kept in the
   receipt.
5. The table reports every owner, day-240 signed carry, component RMS,
   strongest ten-day birth block, and largest depth/longitude/latitude
   partition.  Day-30 process carry remains unavailable because the oracle
   process record starts at day 180; it is labelled unavailable, not inferred.
6. If a physical process row is largest, this round names its first still-
   unresolved boundary from the existing developed-state records only when
   those records match the landed trajectory's required operands.  A stale
   pre-landing operand record cannot prove a post-landing statement.  If the
   incoming row is largest, its birth before day 180 remains unmeasured and no
   downstream statement is fabricated.
7. The trace stamp, one-ULP trace, and production-effect plants each print
   `STATUS PLANT-FIRED` and exit nonzero.  The scorer gains an endpoint-value
   plant or focused regression proving the explicit expected value is
   fail-closed.
8. Any candidate statement must be compiled-source cited and one-variable,
   then pass the full Decision-43/45/55/59 ladder, month, year, card census,
   DINO, tanks, generic-card, plant, citation, and review gates.  Otherwise
   nothing lands and the OPEN section names the next magnitude measurement.

The required separate `codex exec --sandbox read-only` review runs after the
complete diff.  A `DO NOT SHIP` verdict blocks any landing.  No NEMO
acquisition or user configuration decision is expected for the ranking.

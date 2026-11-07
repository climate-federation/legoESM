# Preregistration — NEMO testcase L2 GYRE round 150

Date: 2026-09-22

Incoming lane tip: `4f95ca0db122`.  Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round150/`.  This document is
frozen before repairing or measuring the Round-149 generic-card regression.

## Compiled statement and defect

The landed GYRE-zco path must keep NEMO's six live lateral-momentum diffusion
thickness operands.  The compiled operator multiplies the curl by live `e3f`,
weights the divergence by live Kbb T/U/V thickness, and divides its curl
contribution by live Kmm U/V thickness at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:157-176`.
The compiled free-surface program builds the F-point ratio from the four
surface-area-weighted SSH values at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90:273-286`, while its
reference F thickness is the masked four-T-cell average at
`GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynvor.f90:911-936`.

Round 149 computes that F thickness for every WS-RK3 card before dispatching
the momentum-viscosity operator.  The generic `build_nemo_gyre_recipe()` card
resolves `lateral_viscosity_operator="vector_laplacian"` and
`lateral_viscosity_e3_weighting="off"`; it therefore does **not** execute the
compiled `nemo_div_curl`/`nemo_e3` statement, but the unconditional operand
construction nevertheless raises because its native z-star coordinate carries
no bridge-only `nemo_een_barotropic` bundle.  The four named generic-card tests
currently fail at that guard.  The repair must not change this card's operator
selection or silently substitute reference thickness into a NEMO-e3 call.

## Candidate

Extend the existing shared F-thickness builder, rather than add a second
formula, so it forms every consumed value from the card's own reference
T thickness, active mask, grid areas, and current SSH.  GYRE-zco's recorded
F operands become an independently checked oracle, not a required production
input.  Route the six-tuple only when the resolved card actually selects both
`nemo_div_curl` and `nemo_e3`; other viscosity operators receive `None` and do
not execute dead NEMO geometry.

No configuration, default, carried state, scheme, stabilizer, canonical NEMO
source, or time level changes.

## Frozen predictions and falsifiers

1. The four pre-existing generic-card failures become green.  The generic card
   remains `vector_laplacian`/`off`; its exact base (`903625dd1`) versus
   candidate three-step endpoint has zero moved cells in all 15 T/S/u/v/eta
   rows.  Any moved certified cell is registered; any worsened certification
   requires `DECISION_NEEDED` and refuses this landing.
2. On GYRE-zco, own-state and bridge-recorded F operands are BIT on every
   consumed wet F cell.  The Round-149 developed-state LDF U/V proof remains
   0/16,530 and 0/17,100 unequal under production JIT.  Any owned operand or
   LDF-term difference refutes the candidate.
3. The certified kt=1..10 ladder and 30-day trajectory are bit-identical to the
   Round-149 after arm.  Day-240/day-360 are therefore unchanged by construction
   and need not be rerun unless a certified month/ladder cell moves.  Any moved
   GYRE-zco cell triggers the full Decision-43/45 year gate before landing.
4. The recipe-derived execution census continues to name only GYRE-zco for the
   `nemo_div_curl`+`nemo_e3` route.  DINO's Euler cards select
   `nemo_div_curl` with e3 weighting off, so they do not execute the changed
   F-thickness builder; their real-card tests remain bit-identical.  A census
   showing DINO executes the changed statement requires before/after DINO
   measurement.
5. A synthetic perturbation of a consumed own-state F operand must move the
   exact operator row under production JIT and make the plant exit nonzero.
   Removing the execution guard must reproduce the generic-card failure.

Expected status is **LANDED** only if every prediction above and the four-file
push gate pass.  Otherwise production is restored and the round is **HELD**.

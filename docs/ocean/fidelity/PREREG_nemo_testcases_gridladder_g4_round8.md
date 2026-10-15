# Preregistration — grid ladder G4 round 8 (gridladder_r28), stage-3 heat RHS

Frozen before any round-28 G4 measurement. Housekeeping commit `f78411ff8`
changes receipts and one test assertion only; no arithmetic.

## Fixed program and labels

- Rung G4, Decisions 120/121/123 unchanged. Record
  `gridladder_rounds/round21/oracle_g4_r21` (build `ORCA2G1_OMIP_L1_P5`).
  fp64, CPU. Bar: the ladder's existing normalized 1e-15; no threshold change.

## Where the INDEPENDENT chain first leaves NEMO (read off round-25 evidence)

Both labels start from the same state at kt 1, so the first leave of the
INDEPENDENT chain is the first unequal value at kt 1: stage 1 and 2 are equal
in every value (56 signed-zero u bits only); stage 3 T is unequal in 489 cells
(max 3.55e-15); stage-3 S, ssh, uu_b, vv_b, u, v are value-identical. Since
stage-3 S is exact, the shared implicit vertical solve is not the owner; the
owner is a temperature-only stage-3 Krhs statement.

## Candidate (read off the code, not yet measured)

NEMO's stage-3 temperature Krhs is cleared, then (advection OFF) receives, in
order: the non-solar heat `r1_rcp * qns * z1_rho0_e3t` with
`z1_rho0_e3t = r1_rho0 / (e3t_3d(1)*(1+r3t(Kmm)*tmask))`
(`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:306-307`), the runoff content
(`:322-329`), then `tra_qsr` on Kmm. The card instead forms
`(dT_dt(Kbb) - qsr(Kbb)) * h(Kbb)/h(Kmm) + qsr(Kmm)`, deriving non-solar heat
from `q_net = round(qns+qsr)` on the Kbb thickness and removing the Kbb solar
by subtraction. Candidate: a default-off option, selected only by G4, builds
the stage-3 T rate in NEMO's written order from NEMO's recorded `qns`,
supplied explicitly as a new optional forcing field. Same line on the full
card (Decision 122): `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:306-307`.

## Predictions

1. Before: the route reproduces stage-3 T 489 unequal, max 3.552713678800501e-15,
   everything else value-identical.
2. With the option: stage-3 T unequal count falls (target 0); stage-3 S, ssh,
   uu_b, vv_b, u, v unchanged.
3. Both kt 1..10 labels: no AT-BAR row leaves the bar; INDEPENDENT first-over-bar
   (kt 6 u/v) unchanged or later. Certification only if every field is at the
   bar under both labels.

## Falsifiers and controls

- Stage-3 T count unchanged (489) → candidate REFUTED, reverted, walk continues
  to `tra_qsr` (Kmm solar) and the stage-3 tridiagonal RHS.
- Any other stage row moves → hold the candidate.
- Deselecting the option must restore 489 (non-vacuity).
- Missing `qns` with the option on must raise, never fall back.

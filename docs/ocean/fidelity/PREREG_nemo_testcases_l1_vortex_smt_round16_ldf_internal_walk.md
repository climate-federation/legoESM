# Preregistration — VORTEX_SMT round 16 (lane round 228): internal LDF walk

Frozen before parsing either Round-227 payload or running legoESM against it.
Base: lane tip `cffa2ab79` (round 227).  Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round228/`.  The operator has
reported the requested acquisition ready at
`phase3/round227/oracle_vortex_smt3_ldf_internal/`; this preregistration uses
only that readiness report and the record names, not payload values.

## Compiled branch read first

The record's compiled build is
`tests/VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo`.  It constructs density
and the standard slopes once from the step-entry state before the RK3 stages
(`stprk3.f90:159-177`), then dispatches the selected standard-isoneutral
Laplacian operator through `traldf_iso_lap` (`traldf.f90:105-110`).  The
compiled order inside that operator is A33/MSC (`traldf_iso.f90:788-837`),
tracer gradients (`:178-228`), horizontal tensor factors and fluxes
(`:230-269`), vertical factors and flux (`:273-306`), and the RHS divergence
(`:307-312`).  The slope producer forms raw gradients, bounds them with live
U/V thicknesses, applies the mixed-layer recurrence, and Shapiro-filters its
U/V/W outputs (`ldfslp.f90:202-345`).

## Predictions and falsifiers

* **R16-P1 — admission and calibration.**  The acquired writer is passive:
  the plain and instrumented step-10 restarts are byte-identical; both files
  parse to EOF with exactly the required unique group names; the stored
  `rhs_increment` is bitwise `rhs_after-rhs_before`; all three parser plants
  exit nonzero; and both commit stamps match the payload and producer.  Any
  failed clause REFUTES admission and stops the walk.
* **R16-P2 — aggregate reproduction.**  The round-228 production-JIT walk
  reproduces round 226's stage-3 post-LDF temperature maximum
  `2.0915088416728622e-07 K` and its pre-LDF maximum
  `7.418332614861356e-11 K`, to their recorded fp64 values.  A mismatch is a
  Rule-1e disagreement and stops attribution until reconciled.
* **R16-P3 — source-order owner.**  The first owned non-bit seam is in the
  once-per-step slope program, before A33: at least one final `uslp`, `vslp`,
  `wslpi`, or `wslpj` row is non-bit under the production step, while every
  preceding recorded input needed for that row is either bitwise or its
  inherited difference is explicitly substituted.  This is REFUTED if all
  four final slope fields are bitwise; the walk then advances, in compiled
  order, to A33, gradients, horizontal fluxes, vertical fluxes, and divergence.
* **R16-P4 — magnitude closure.**  Substituting the first non-bit family with
  NEMO's recorded value removes at least 90% of the additional
  `2.090767008411376e-07 K` introduced across `tra_ldf`.  Less than 90%
  REFUTES sole magnitude ownership and leaves the statement diagnostic-only.
* **R16-P5 — landing bar.**  A production statement lands only if its local
  production-JIT row becomes bitwise given NEMO's preceding seam and it passes
  the standing SMT/flat-VORTEX/tank registries, GYRE ladder and year, generic
  recipe, DINO month, citation, plant, and independent-review gates.  Otherwise
  this round is HELD with the first non-bit statement and the next
  discriminating measurement.  No downstream or cross-stage bundle is allowed.

## Controls and execution labels

The committed walk extends the existing Round-218/Round-226 production-step
hook and the shared self-describing parser; it does not create a second stage
harness.  Rows are labelled `production_step_jit`; isolated eager/JIT values
may be supporting diagnostics only.  A one-ULP perturbation at each claimed
seam must make the corresponding plant exit nonzero.  Every array shape and
mask is printed before indexing, and the final report names wet support.

## No hidden choices

No configuration, physical coefficient, scheme, carried state, stabiliser,
threshold, timestep, resolution, acceptance bar, or record source changes.
Decision 93 already selects SMT-3.  The only candidate permitted is the first
compiled statement that this admitted record measures as non-bit.

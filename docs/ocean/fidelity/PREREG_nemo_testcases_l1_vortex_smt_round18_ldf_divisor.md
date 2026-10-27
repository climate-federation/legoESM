# Preregistration — VORTEX_SMT round 18 (lane round 230): live tracer divisor

Frozen before changing production or running a new scientific arm.  Base:
`536f3da0dee7` (round 229).  Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round230/`; the immutable
oracle record remains
`phase3/round228/oracle_vortex_smt3_ldf_internal/` from build
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3`.

## Compiled branch read first

The resolved `traldf_iso` loop forms horizontal and vertical fluxes, then adds
their divergence to tracer `Krhs`.  On both the ordinary and deepest levels,
NEMO divides by the live Kmm T-point thickness
`e3t_3d*(1+r3t(Kmm)*tmask)` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`
and `:327-331`.  The stored area reciprocal is a separate multiplier.  The
current legoESM operator instead divides by `dz_ref*jacobian`, even when the
same dispatch has already selected NEMO's live-QCO Kmm face thicknesses.

Pre-implementation search found the shared literal T-thickness builder
`vertical.py:nemo_qco_live_t_thickness`, the existing `redi_flux_eta` Kmm
time-level route, and the existing Round-228 production-step diagnostic hook.
This round extends those paths; it creates no second thickness formula, stage
harness, or public configuration option.

## Predictions and falsifiers

* **R18-P1 — frozen reproduction.**  The clean production-JIT walk reproduces
  the Round-229 recorded-flux residual
  `4.8452277850122817e-11 K s-1`, the current production LDF RHS maximum
  `7.2621621143171395e-11 K s-1`, and the NEMO-divisor reconstruction floor
  `2.9778502051908996e-22 K s-1`.  Any disagreement stops attribution under
  Rule 1e.
* **R18-P2 — one-variable production closure.**  Threading only the card's
  already-verified live Kmm T-point thickness into the final Redi divergence
  divisor leaves every upstream slope, tensor and flux diagnostic bitwise
  unchanged and reduces the production-step RHS maximum to at most
  `1e-20 K s-1`.  A larger residual, any changed upstream row, or a need for a
  second statement REFUTES the candidate.
* **R18-P3 — local source exactness.**  On the R16 NEMO operands the literal
  thickness builder reproduces recorded
  `e3t_3d*(1+r3t(Kmm)*tmask)` bit-for-bit in production JIT and eager.  A
  one-ULP perturbation to that thickness must make the claimed divisor row
  non-bit, print `STATUS PLANT-FIRED`, and exit nonzero.  Eager-only closure is
  insufficient.
* **R18-P4 — landing.**  If P2 and P3 hold, the shared NEMO statement lands
  only after every moved row is registered across both SMT cards, all six flat
  VORTEX cards, LOCK_EXCHANGE, OVERFLOW, the certified 954-row GYRE ladder and
  certified year, the generic NEMO-GYRE recipe, and the private-directory DINO
  month gate.  No AT-BAR row may leave the bar and first-over-bar may not move
  earlier.  A red ratchet, a DINO regression, or a review verdict `DO NOT SHIP`
  holds production and preserves the candidate as a manifest patch.
* **R18-P5 — magnitude.**  On SMT-3 the landed arm removes at least 90% of the
  Round-229 `7.2621621143171395e-11 K s-1` RHS maximum and moves the certified
  tracer trajectory toward NEMO.  Less than 90% or an unmoved trajectory
  REFUTES the sole-magnitude prediction even if the source-local row is exact.

## Controls and choices

The production discriminator uses the existing full production step under
JIT; isolated eager/JIT rows are supporting labels only.  The non-vacuity test
must fail when the live divisor is replaced with the prior Jacobian thickness.
No coefficient, scheme, timestep, threshold, carried state, record source,
default, stabiliser, or configuration field changes.  Decision 93 already
selects SMT-3, and NEMO's compiled statement supplies the only permitted
arithmetic.

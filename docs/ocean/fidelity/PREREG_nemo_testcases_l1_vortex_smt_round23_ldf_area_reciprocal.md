# Preregistration — VORTEX_SMT round 23 (lane round 235): stored LDF area reciprocal

Frozen before extending the existing production-step discriminator or reading
a new scientific result. Base: `12b192dbc` (round 234). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round235/`; the immutable
oracle record remains `phase3/round228/oracle_vortex_smt3_ldf_internal/` from
build `VORTEX_SMT3_VEC_R16_OMIP_L1_P3`.

## Compiled branch read first

NEMO materialises `e1e2t=e1t*e2t` and then stores
`r1_e1e2t=1/e1e2t` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90:155`.
The regular tracer-LDF update consumes that stored reciprocal at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`;
the deepest-level form consumes the same operand at `:327-331`. The committed
writer records both fields and the before/after RHS at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/vortex_r23_ldf_terms.f90:149-172`.

Pre-implementation search found the exact-flux production-JIT arm, the complete
source-rounded divergence arm, the self-describing record parser, and the
recorded `e1e2t`/`r1_e1e2t` groups in the Round-228 walk. This round extends
those paths; it creates no second parser, operator, stage harness, record, or
configuration field.

## Predictions and falsifiers

* **R23-P1 — frozen reproduction.** The exact-flux plus live-thickness
  production arm reproduces `8,613` unequal cells and
  `2.5978639524806749e-14 K s-1`; all tensor, gradient, flux, and divergence
  controls remain bit-exact. Any disagreement stops attribution under Rule 1e.
* **R23-P2 — recorded reciprocal closure.** Substituting only the recorded
  `r1_e1e2t` into the complete source-rounded production-JIT arm takes the RHS
  to at most `1e-20 K s-1`, with a predicted maximum of
  `2.9778502051908996e-22 K s-1`. A larger residual REFUTES the prediction.
* **R23-P3 — reproducible source operand.** Computing
  `source_round(1/source_round(e1t*e2t))` from the production metrics is bitwise
  equal to the recorded reciprocal and gives the same RHS result as R23-P2. If
  the recorded arm closes but the local arm does not, the record proves the
  operand boundary but does not provide a shared landing candidate.
* **R23-P4 — final seam split.** If both reciprocal arms are inert, retain the
  exact reciprocal and split the final wet-mask multiplication from the
  higher-level Krhs addition/subtraction seam. The first arm that reaches the
  recorded-input floor names the next boundary; if none does, no statement or
  landing is claimed.
* **R23-P5 — controlled plants.** A one-ULP change to a nonzero reciprocal at
  the largest active RHS location changes the full-step tendency digest,
  prints `STATUS PLANT-FIRED`, and exits nonzero. If R23-P4 is entered, its
  newly tested operand gets an independent production plant. A missed plant or
  moved upstream control invalidates the result.
* **R23-P6 — landing discipline.** Only a locally reproducible arm that closes
  the RHS and is NEMO's cited statement enters the complete shared-card gates:
  both SMT cards, all six flat VORTEX cards, LOCK_EXCHANGE, OVERFLOW, the
  certified 954-row GYRE ladder and year, the generic NEMO-GYRE recipe, and the
  private-directory DINO month gate. Every moved row is registered, no AT-BAR
  row leaves the bar, and first-over-bar does not move earlier. A red ratchet,
  DINO regression, or review verdict `DO NOT SHIP` holds production.

Failed predictions remain in the receipt. Production JIT is authoritative;
eager or isolated reconstructions are supporting labels only.

## No hidden choices

Decision 93 already authorises the SMT-3 rung. This round changes no scheme,
coefficient, timestep, threshold, stabiliser, carried state, card default,
record source, or public configuration. If closure requires a scientific
choice rather than NEMO's cited arithmetic, the round stops with
`DECISION_NEEDED` instead of selecting one.

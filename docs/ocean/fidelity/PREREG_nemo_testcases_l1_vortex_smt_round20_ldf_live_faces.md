# Preregistration — VORTEX_SMT round 20 (lane round 232): live-face LDF set

Frozen before changing the existing production-step discriminator or reading a
new scientific result.  Base: `a9cf6fe559` (round 231).  Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round232/`; the immutable
oracle record remains
`phase3/round228/oracle_vortex_smt3_ldf_internal/` from build
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3`.

## Compiled branch read first

NEMO forms A11/A22 from the live Kmm U/V thicknesses at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-244`,
then the closed-bottom four-W-mask factors and A13/A23 at `:246-252`, and then
the parenthesised horizontal U/V fluxes at `:254-259`.  The same operator adds
their divergence with the stored area reciprocal and live Kmm T thickness at
`:306-310` and `:327-331`.

Pre-implementation search found the recorded live U/V arrays, the mask and
divisor operands, and both input-plant paths in the existing Round-228 walk.
This round extends that one discriminator; it creates no second parser,
operator, thickness formula, stage harness, or configuration field.

## Predictions and falsifiers

* **R20-P1 — frozen reproduction.**  The ordinary production arm reproduces
  `7.2621621143171395e-11 K s-1`; the closed-bottom-mask arm reproduces
  `4.8307864472774789e-11 K s-1`; and the measured mask-plus-divisor pair
  reproduces `4.9526264855300654e-12 K s-1`.  Any disagreement stops
  attribution under Rule 1e.
* **R20-P2 — live-face source closure.**  Adding only NEMO's recorded live Kmm
  U/V face thicknesses to that pair makes A11 and A22 bit-exact while A13 and
  A23 remain bit-exact.  Any non-bit A11/A22 cell, or any regression in
  A13/A23, REFUTES the source-closure prediction.
* **R20-P3 — production RHS closure.**  The full production-JIT arm reaches the
  recorded compiled-rounding floor, `2.9778502051908996e-22 K s-1`, and in all
  cases is at most `1e-20 K s-1`.  If A11/A22 close but the RHS exceeds that
  bound, the first remaining statement is the parenthesised horizontal-flux
  association at compiled lines 254-259.  If A11/A22 do not close, their
  live-face statement remains first.
* **R20-P4 — controlled arm and plants.**  The triple arm changes only the
  registered mask, divisor and U/V face operands.  The tracer-gradient and
  vertical-flux rows remain bitwise identical to the ordinary production
  step.  Separate one-ULP changes to the U and V face inputs each change the
  full-step tendency digest, print `STATUS PLANT-FIRED`, and exit nonzero.
  A moved upstream row or a missed plant invalidates the result.
* **R20-P5 — landing discipline.**  Only if P2 and P3 hold does this same-stage
  set enter the complete shared-card trajectory gates: both SMT cards, all six
  flat VORTEX cards, LOCK_EXCHANGE, OVERFLOW, the certified 954-row GYRE ladder
  and year, the generic NEMO-GYRE recipe, and the private-directory DINO month
  gate.  Every moved row is registered, no AT-BAR row leaves the bar, and
  first-over-bar does not move earlier.  A red ratchet, DINO regression, or
  review verdict `DO NOT SHIP` holds production and preserves a manifest patch.

Failed predictions remain in the receipt.  Production JIT is authoritative;
eager or post-hoc reconstructions are supporting labels only.

## No hidden choices

Decision 93 already authorises the SMT-3 rung.  This round changes no scheme,
coefficient, timestep, threshold, stabiliser, carried state, card default,
record source, or public configuration.  If closure requires a scientific
choice rather than NEMO's cited arithmetic, the round stops with
`DECISION_NEEDED` instead of selecting one.

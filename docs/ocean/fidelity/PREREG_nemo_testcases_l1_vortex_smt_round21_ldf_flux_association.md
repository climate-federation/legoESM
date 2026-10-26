# Preregistration — VORTEX_SMT round 21 (lane round 233): horizontal-flux association

Frozen before changing the existing production-step discriminator or reading a
new scientific result.  Base: `59639d2819` (round 232).  Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round233/`; the immutable
oracle record remains
`phase3/round228/oracle_vortex_smt3_ldf_internal/` from build
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3`.

## Compiled branch read first

NEMO forms each vertical-gradient quartet as two explicit pairs inside its
horizontal U/V flux statements at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:254-259`:
`(a+b)+(c+d)`.  The same statements add the diagonal and cross terms inside
parentheses and multiply that result by the masked diffusivity.  legoESM's
active transcription at the base tip evaluates the quartet as `a+b+c+d` and
then evaluates the same outer multiply/add form.

Pre-implementation search found the complete mask, live U/V face and live T
divisor set plus their input plants in the existing Round-228 production-JIT
discriminator.  This round extends that discriminator with one private
association selector.  It creates no second parser, operator, stage harness,
configuration field or record.

## Predictions and falsifiers

* **R21-P1 — frozen reproduction.**  The ordinary production arm reproduces
  `7.2621621143171395e-11 K s-1`; the mask-plus-divisor arm reproduces
  `4.9526264855300654e-12 K s-1`; and the mask-plus-divisor-plus-live-faces arm
  reproduces `2.5978639524806749e-14 K s-1`, with A11/A22/A13/A23 bit-exact.
  Any disagreement stops attribution under Rule 1e.
* **R21-P2 — pairwise quartet closure.**  Changing only the four-gradient
  quartet to NEMO's `(a+b)+(c+d)` form makes zfu and zfv bit-exact and takes the
  production-JIT RHS to the recorded floor
  `2.9778502051908996e-22 K s-1`, in all cases no larger than `1e-20 K s-1`.
  Any non-bit zfu/zfv cell or larger RHS REFUTES the prediction.
* **R21-P3 — controlled production arm.**  hmsku/hmskv, A11/A22/A13/A23,
  dit/djt/dkt and both vertical-flux rows remain bitwise identical to the
  Round-232 triple arm.  Independent one-ULP U-face and V-face input plants
  each change the full-step tendency digest, print `STATUS PLANT-FIRED`, and
  exit nonzero.  A moved upstream row or a missed plant invalidates the result.
* **R21-P4 — outer association discriminator.**  If P2 is refuted while every
  upstream row remains exact, the next arm separates NEMO's outer
  `aht*(A11*dit + A13*pair4)` multiply/add association from the quartet alone.
  No source-owner or landing claim is made from the failed quartet arm.
* **R21-P5 — landing discipline.**  Only a production-JIT arm that closes zfu,
  zfv and the RHS enters the complete shared-card gates: both SMT cards, all
  six flat VORTEX cards, LOCK_EXCHANGE, OVERFLOW, the certified 954-row GYRE
  ladder and year, the generic NEMO-GYRE recipe, and the private-directory DINO
  month gate.  Every moved row is registered, no AT-BAR row leaves the bar,
  and first-over-bar does not move earlier.  A red ratchet, DINO regression or
  review verdict `DO NOT SHIP` holds production and preserves a manifest patch.

Failed predictions remain in the receipt.  Production JIT is authoritative;
eager or isolated reconstructions are supporting labels only.

## No hidden choices

Decision 93 already authorises the SMT-3 rung.  This round changes no scheme,
coefficient, timestep, threshold, stabiliser, carried state, card default,
record source or public configuration.  If closure requires a scientific
choice rather than NEMO's cited arithmetic, the round stops with
`DECISION_NEEDED` instead of selecting one.

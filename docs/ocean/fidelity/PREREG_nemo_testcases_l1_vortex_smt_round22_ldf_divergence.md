# Preregistration — VORTEX_SMT round 22 (lane round 234): LDF divergence association

Frozen before changing the existing production-step discriminator or reading a
new scientific result. Base: `888a3201e` (round 233). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round234/`; the immutable
oracle record remains `phase3/round228/oracle_vortex_smt3_ldf_internal/` from
build `VORTEX_SMT3_VEC_R16_OMIP_L1_P3`.

## Compiled branch read first

NEMO's regular-level update forms the U-face difference, the V-face difference,
the vertical-flux difference, their source-ordered sum, the area multiply and
the live-thickness division in one update at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`.
The deepest level uses the same order with the lower vertical flux alone at
`:327-331`. The writer records the increment as `rhs_after-rhs_before`, so the
known `2.9778502051908996e-22 K s-1` recorded-input residual is the comparison
floor rather than an exact-zero target.

Pre-implementation search found the exact-flux production-JIT arm, its U/V
plants, the shared source-rounding identity, the self-describing record parser,
and the existing regular/deepest vector divergence in the Round-228 walk and
shared Redi operator. This round extends those paths; it creates no second
parser, operator, stage harness, record, or configuration field.

## Predictions and falsifiers

* **R22-P1 — frozen reproduction.** The ordinary production arm reproduces
  `7.2621621143171395e-11 K s-1`; the complete live-operand plus literal-flux
  arm reproduces exact zfu/zfv and `2.5978639524806749e-14 K s-1`; all tensor,
  gradient, and vertical-flux controls remain bit-exact. Any disagreement stops
  attribution under Rule 1e.
* **R22-P2 — source-order closure.** Starting from exact literal zfu/zfv,
  retaining each compiled subtraction, the two horizontal additions, the
  vertical addition, the area multiply and the live-thickness division takes
  the production-JIT RHS to at most `1e-20 K s-1`, with a predicted maximum of
  `2.9778502051908996e-22 K s-1`. A larger residual REFUTES the prediction.
* **R22-P3 — one-boundary walk.** The discriminator evaluates cumulative arms
  in compiled order: source-rounded face differences; their horizontal sum;
  the regular/deepest vertical addition; and the multiply-then-divide scale.
  The first arm that reaches the floor names the first non-bit source boundary.
  If none reaches the floor, no statement or landing is claimed.
* **R22-P4 — controlled production plants.** Every arm changes only the private
  evaluation selector. zfu/zfv and all upstream controls stay bit-exact. A
  one-ULP change to the exact U flux and an independent change to the exact V
  flux each alter the full-step tendency digest, print `STATUS PLANT-FIRED`,
  and exit nonzero. A moved upstream row or missed plant invalidates the result.
* **R22-P5 — landing discipline.** Only an arm that closes the RHS and is
  NEMO's cited statement enters the complete shared-card gates: both SMT cards,
  all six flat VORTEX cards, LOCK_EXCHANGE, OVERFLOW, the certified 954-row GYRE
  ladder and year, the generic NEMO-GYRE recipe, and the private-directory DINO
  month gate. Every moved row is registered, no AT-BAR row leaves the bar, and
  first-over-bar does not move earlier. A red ratchet, DINO regression, or
  review verdict `DO NOT SHIP` holds production and preserves a manifest patch.

Failed predictions remain in the receipt. Production JIT is authoritative;
eager or isolated reconstructions are supporting labels only.

## No hidden choices

Decision 93 already authorises the SMT-3 rung. This round changes no scheme,
coefficient, timestep, threshold, stabiliser, carried state, card default,
record source, or public configuration. If closure requires a scientific
choice rather than NEMO's cited arithmetic, the round stops with
`DECISION_NEEDED` instead of selecting one.

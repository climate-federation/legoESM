# Preregistration — VORTEX_SMT round 24 (lane round 236): land the LDF mask/divisor pair

Frozen before changing production or reading a new trajectory result. Base:
`9af53c118` (round 235). Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round236/`; the immutable
oracle operand record remains `phase3/round228/oracle_vortex_smt3_ldf_internal/`
from build `VORTEX_SMT3_VEC_R16_OMIP_L1_P3`.

## Compiled branch read first

NEMO's horizontal Redi tensor reads the closed `jpk` W mask while forming
`zmsku/zmskv` at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:246-249`.
Its regular-level RHS divides by the live Kmm T thickness at `:306-310`; the
deepest-level form uses the same divisor at `:327-331`. The exact source
statements are therefore the already measured round-231 pair: close the
deepest W level instead of vertically wrapping it to the surface, and use
`e3t_3d*(1+r3t(Kmm)*tmask)` rather than the reference/Jacobian thickness.

Pre-implementation search found the shared `nemo_iso_lap` operator, the
existing literal QCO T-thickness helper, the round-229 held mask patch, the
round-228 production-JIT walk and plants, the shared trajectory gate, and the
existing card builders. This round extends those paths. It creates no second
operator, helper, parser, harness, card, configuration field, or stabiliser.

The instantiated fp64 card census is frozen: only `GYRE-zco` and
`VORTEX_SMT3_VEC-zps` among the fourteen testcase cards run `nemo_iso_lap`;
only SMT-3 selects its QCO-live thickness arm. DINO also shares the operator
and must pass its private month gate. All six flat VORTEX cards, both other SMT
cards, SMT-1/2 and both tanks do not execute the operator and must remain
bit-identical; the generic GYRE recipe and ORCA2 are explicitly measured or
registered as required below.

## Predictions and falsifiers

* **R24-P1 — local reproduction.** Before production is switched, the
  production-JIT walk reproduces the ordinary `7.2621621143171395e-11 K s-1`
  RHS maximum and the mask-plus-live-divisor arm reproduces
  `4.9526264855300654e-12 K s-1`, removing `0.9318023144131400`. After the
  switch, the ordinary production arm reproduces the latter value and the
  explicit candidate arm is identical. Any disagreement stops under Rule 1e.
* **R24-P2 — exact operands.** The production statement makes hmsku/hmskv and
  A13/A23 bit-exact against NEMO. The literal live T thickness is bit-exact
  against the recorded Kmm divisor in eager and through the production-JIT
  closure. A one-ULP W-mask/divisor plant must move the full-step tendency,
  print `STATUS PLANT-FIRED`, and exit nonzero. A non-bit operand or missed
  plant holds the candidate.
* **R24-P3 — SMT-3 trajectory.** The SMT-3 50-row registry changes in the
  direction of NEMO on the first tracer-debt rows, with no AT-BAR-to-DEBT
  crossing and no earlier first-over-bar. Every changed field/step row and the
  cellwise ratchet counts are registered. A red ratchet, any AT-BAR loss, or
  an earlier first-over-bar holds production and preserves a held patch.
* **R24-P4 — shared-card blast radius.** GYRE's 954-row ladder and certified
  year remain at the round-223 values; the generic GYRE recipe passes; the
  other SMT card, six flat VORTEX cards, LOCK_EXCHANGE and OVERFLOW move zero
  registry rows; DINO's day-30 T rms does not exceed its pinned bar. Any
  unregistered movement, red card gate, or DINO regression holds production.
  ORCA2 is not silently waived: its exact merge pointer is the shared
  `nemo_iso_lap` W-mask plus QCO-live T-divisor statements, and its own
  certified ladder is run if its available card selects this path; otherwise
  the receipt states the source/config proof that it does not.
* **R24-P5 — landing.** If P1-P4 pass, the pair lands in the single shared
  implementation, the round-229 manifest is marked landed, citations are
  mapped and the shifted-citation plant exits nonzero. Separate read-only
  Codex review must not say `DO NOT SHIP`; if review is unavailable in the
  sandbox, its complete failure verdict is quoted as required. Failed
  predictions remain in the receipt.

Production JIT is authoritative. Eager and isolated rows are supporting
labels only. The `2.597863951157186e-14 K s-1` downstream arithmetic residue
is not a blocker for this magnitude landing: round 235 closed that bit walk,
and the candidate consists only of the two cited operands already measured as
NEMO's own statements.

## No hidden choices

Decision 93 authorises SMT-3 and the round-235 OPEN orders this landing. This
round changes no scheme, coefficient, timestep, threshold, stabiliser, carried
state, record source, or card selection. The QCO-live arm follows the card's
already explicit resolved geometry branch; no new default or option is added.
If the gate requires a choice rather than NEMO's cited statements, the round
stops with `DECISION_NEEDED`.

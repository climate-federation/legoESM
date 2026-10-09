# ORCA2 round 195 preregistration — substep-3 V-transport operands

Date: 2026-10-09. Frozen base:
`a7928d44c06c9f9ef8b52e4354f9604778afa282`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round195/`.

Every number is **independent hierarchy rung 0**. The card starts from its
corrected climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry or rung-10 number is mixed into this round. The shipped
ORCA2 card, sea ice, its six selectors and its `unmeasured_features` tuple
remain unchanged. This is an offline, measurement-only split; no executable
observer is permitted.

## Frozen source order and protocol

NEMO first extrapolates the substep midpoint V velocity
`va_e = za1*vn_e + za2*vb_e + za3*vbb_e` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:502-511`, then builds the
midpoint V-face depth from `hv_0` and the area-weighted neighbouring sea
surface at `dynspg_ts.f90:535-546`, then evaluates the left-associated product
`zhV = e1v * va_e * zhvp2_e` at `dynspg_ts.f90:564-570`. The admitted
round-96 rank-complete record writes `va_e`, `zhvp2_e` and `zhV` before their
downstream accumulator use at `dynspg_ts.f90:600-619`.

Reuse and parameterise the existing round-146 pure operand splitter; do not
introduce a second product implementation. Run round 194's already-passive
private arm on CPU with JIT, fp64 and libm, but read only substep 3. Compare in
compiled order: static `e1v`, midpoint `va_e`, midpoint depth `zhvp2_e`, the
first product, then completed `zhV`. Each candidate operand is replaced by the
recorded NEMO operand one variable at a time; a cumulative replay is allowed
only to identify cancellation after all individual rows are printed.

The gate must parse the record's self-describing header, require both ranks
exactly once, and require all three substep-3 operand names and lengths. It
must also reproduce round 194's prerequisite: substeps 1-2 completed `zhV`
bit-exact and substep 3 non-bit. Plants independently reorder the registry,
perturb one exact operand bit, and remove the record field; every plant must
refuse.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R195-P1 | The admitted round-96 record is sufficient. | Both rank shards expose `e1v` through the card plus recorded substep-3 `va_e`, `zhvp2_e`, and `zhV`, with the expected shapes. | Any absent/malformed stream: **REFUTED**; write a fail-closed acquisition and stop. |
| R195-P2 | Static `e1v` is bit-exact and cannot own the new substep-3 debt. | `0 / 26,640` unequal against the raw NEMO metric. | Any unequal value: **REFUTED**; stop at `e1v` and name its producer. |
| R195-P3 | Midpoint `va_e` is the first non-bit operand at substep 3. | `e1v` exact and `va_e` non-bit; replacing only `va_e` moves completed `zhV` toward or to NEMO. | Exact `va_e`: **REFUTED**; continue in source order to `zhvp2_e`. A substitution that does not move `zhV` is recorded as a cancelling/null arm. |
| R195-P4 | One recorded operand substitution closes the substep-3 `zhV` row. | The source-ordered single-variable replay reaches `0 / 26,640` unequal. | Residual after all single substitutions: **REFUTED**; report the cancelling unit and do not revisit the reciprocal or trajectory. |
| R195-P5 | Controls bind. | Registry-order, operand-bit and missing-stream plants each refuse. | Any plant stays green: instrument invalid; report no operand claim. |

If P1, P2, P4 or P5 fails, no trajectory gate can reverse the source-order
result. The final package tree must equal the frozen base; a newly named
operand is the OPEN item for round 196.

ASKED choices: round 194's OPEN source-ordered three-operand split.  
UNASKED choices: empty.

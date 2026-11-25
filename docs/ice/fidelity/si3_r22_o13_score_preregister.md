# SI3 round 22 — O13 model preparation and master-record scoring preregistration

Date: 2026-10-28. Lane `si3`, round `si3_r22`, parent `87a1c8843803`.
Scope: rung O13 only: the production HFN `ice_itd_rem` stage at `jpl=5`,
including boundary motion, category transfers, 10 ice-energy layers, 5
snow-energy layers, layer salt, age, surface temperature, snow, pond area,
pond volume, and pond-lid volume. No configuration, namelist, acquisition, or
bar change is permitted.

This preregistration is committed before reading any Kind 11 or Kind 12 binary
record from the admitted O8--O16 master record.

## Fixed evidence and execution policy

- Master root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master`.
- O13 projection: Kinds 11 and 12, 200 records, projection-manifest SHA-256
  `a94d126ddaee3194a053ef437d762707dff93045287392d4e0586aaee995a4aa`.
- Run-manifest SHA-256:
  `bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0`.
- Complete-record inventory SHA-256:
  `fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85`.
- CPU, binary64, `PrecisionPolicy.fp64(transcendentals="libm")`; real eager
  calls run under `jax.disable_jit()`, compiled calls take every changing array
  as a JIT argument.
- Equality bar: zero ULP, including signed zero, for every Kind 12 EXIT field
  at every `kt=1..100`. No tolerance relaxation or post-hoc field selection.

## Frozen predictions and falsifiers

P1 — admission. The existing master admission and O13 projection digests pass.
Any mismatch refutes P1 and stops the round.

P2 — scalar-model identity. The one production SI3 remapper matches an
independent scalar transcription that imports no model code or model constants,
under both real eager and JIT execution. Any unequal bit refutes P2.

P3 — master score. Feeding every recorded Kind 11 ENTRY through the production
entry point yields every Kind 12 EXIT field at zero ULP for all 100 steps. Any
normal-range difference refutes P3; the first differing statement, field,
step, category, cell, NEMO source line, and model line are localized and the
round stops without relaxing the bar.

P4 — conservation. On synthetic boundary crossings and all admitted frames,
the remapper preserves category sums of ice area except the NEMO first-category
thin-ice melt term and the configured aggregate-area cap; it preserves ice
volume, snow volume, age content, ice/snow energy, layer salt, pond area,
pond volume, and lid volume through category transfers. A planted dropped
transfer must fail each conservation check.

P5 — production wiring and controls. Removing the production call or replacing
it with the pre-round generic remapper fails. One-ULP EXIT and model-arithmetic
plants fail. The D5 C1D full-year hash remains
`b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`;
all non-O13 callers are bit-identical to the parent on every output.

P6 — differentiation. Gradients are finite and agree with central differences
away from routing boundaries, including zero-category and vanished-ice states.
The literal forward switches retain their NEMO branch behavior; derivative-only
guards may not alter forward bits.

## Stop rule

If P1 fails, or if the first full master score is not zero ULP, record the
first differing statement and a proposal, then stop. Do not change configuration,
the admitted record, the field set, or the equality bar.

# Preregistration — VORTEX_SMT round 19 (lane round 231): same-stage LDF pair

Frozen before changing the existing production-step discriminator or reading a
new scientific result.  Base: `501810fe4d` (round 230).  Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round231/`; the immutable
oracle record remains
`phase3/round228/oracle_vortex_smt3_ldf_internal/` from build
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3`.

## Compiled branch read first

NEMO forms the two four-W-mask reciprocals, their A13/A23 cross terms, and the
horizontal U/V fluxes in that order at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:246-259`.
The deepest horizontal loop reads the closed `jpk` W level; it does not wrap
the surface level onto the floor.  The same compiled operator then adds the
horizontal and vertical flux divergences with the stored area reciprocal and
the live Kmm T-point thickness in the ordinary and deepest-level arms at
`VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310`
and `:327-331`.

Pre-implementation search found all three required private operands in the
existing Round-228 walk: Round 229's held closed-bottom W-mask patch, the
recorded live U/V face thickness tuple, and Round 230's recorded live divisor.
This round extends that single production-JIT discriminator.  It creates no
second parser, operator, thickness formula, stage harness, or configuration
field.

## Predictions and falsifiers

* **R19-P1 — frozen reproduction.**  The ordinary production step reproduces
  the Round-230 LDF RHS maximum `7.2621621143171395e-11 K s-1`, the W-mask-only
  maximum `4.8307864472774789e-11 K s-1`, and the divisor-only maximum
  `8.4505340078032935e-11 K s-1`.  Any disagreement stops attribution under
  Rule 1e.
* **R19-P2 — cancelling pair.**  Applying NEMO's closed-bottom W mask together
  with the recorded live Kmm divisor in one full production-JIT step removes
  less than 90% of the baseline RHS maximum.  Removal of at least 90% REFUTES
  this prediction and promotes the pair to local closure testing.
* **R19-P3 — live-face third arm.**  If P2 is confirmed, adding NEMO's recorded
  live U/V face thicknesses makes A11/A22 and A13/A23 bit-exact but still leaves
  more than 10% of the baseline RHS maximum.  The first remaining non-bit
  source boundary is then the horizontal `zfu`/`zfv` association at compiled
  lines 254-259.  A residual at or below `1e-20 K s-1`, at least 90% removal,
  or an earlier non-bit tensor operand REFUTES this prediction.
* **R19-P4 — controlled production arms.**  The pair and triple arms change
  only their registered operands.  The vertical flux rows and tracer-gradient
  rows remain bitwise identical to the ordinary production step.  A one-ULP
  plant in each newly combined input changes the full-step tendency digest,
  prints `STATUS PLANT-FIRED`, and exits nonzero.  Any upstream movement or
  missed plant invalidates the result.
* **R19-P5 — landing discipline.**  A same-stage set reaches the shared-card
  trajectory gates only if it removes at least 90% of the baseline maximum and
  reaches the compiled-rounding floor under the production JIT.  Otherwise all
  members remain held, production stays unchanged, and the first remaining
  compiled statement is the round's deliverable.

Failed predictions remain in the receipt.  Eager or post-hoc reconstructions
may support the result but cannot replace the full production-step rows.

## No hidden choices

Decision 93 already authorises the SMT-3 rung.  This measurement changes no
scheme, coefficient, timestep, threshold, stabiliser, carried state, card
default, record source, or public configuration.  If closure requires a new
scientific choice rather than NEMO's cited arithmetic, the round stops with
`DECISION_NEEDED` instead of selecting one.

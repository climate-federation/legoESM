# PREREGISTRATION — round 194 / VORTEX round 10: stage-2/3 ZAD operand walk

Frozen before comparing any new ZAD operand or substitution arm.  Lane tip at
the start: `034c89d6b`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round194`.

## Compiled program and existing instrument

The admitted round-192 record and the round-193 production-JIT walk remain the
one stage harness.  This round extends that walk; it does not create a second
stage driver.  The record carries the stage-2 and stage-3 `Kmm` velocities,
`ssh(Kmm)`, and `ww` consumed at the ZAD boundary.

The record's compiled branch is
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo`.  For each of stages 2 and 3 it
builds `ww` from the stage `Kmm` velocity at
`stprk3_stg.f90:289-295`, then calls vector advection at
`stprk3_stg.f90:323-340`.  The vector dispatch calls KEG and then ZAD at
`dynadv.f90:135-141`.  ZAD consumes, in compiled order, the unmasked T-point
area times `ww`, the `Kmm` velocity difference, and the live
`e3t_1d*(1+r3u/r3v(Kmm)*mask)` face thickness at
`dynzad.f90:105-137`.  The `Kmm` face ratios themselves are the
surface-weighted two-cell values written at `domqco.f90:263-270`.

## Frozen predictions and falsifiers

1. **Calibration.**  The unchanged cumulative walk reproduces the round-193
   ZAD maxima: stage-2 U/V
   `1.8741000428408085e-09 / 1.8522221618642312e-09 m s-2` and stage-3 U/V
   `1.8905432975169711e-09 / 1.8685471232664545e-09 m s-2`.  A disagreement
   above one percent stops attribution.
2. **Execution regime.**  Every row and substitution runs through
   `model.step` and `_step_jitted`, CPU fp64/libm.  Isolated-closure or eager
   rows may be supplemental only and cannot name the producer.
3. **Operand prediction.**  The recorded `Kmm` stage velocities and the live
   face thicknesses are predicted BIT.  The first non-bit ZAD operand is
   predicted to be `ww`.  A non-bit velocity or face-thickness row refutes
   that ordering and becomes the first producer.
4. **Magnitude prediction.**  Replacing only `ww` with NEMO's recorded value
   is predicted to remove at least 50 percent of each stage's ZAD U/V maximum;
   replacing velocity alone or face thickness alone is predicted to remove
   less than 10 percent.  Any contrary result is retained as REFUTED and the
   largest measured arm owns the next walk.
5. **Vertical-index convention.**  The gate scores only levels `1:jpkm1` and
   separately proves the NEMO surface accumulator is zero and the bottom
   statement consumes only the carried top-interface product.  A plant on an
   excluded surface/bottom word must not masquerade as an interior control.
6. **Plants.**  One live operand in every substitution family (`ww`, velocity,
   face thickness) is moved by one ULP before the production call.  Each plant
   must change its own ZAD row and exit nonzero.  A plant that changes only a
   different row or exits zero leaves that family UNMEASURED.

## Landing criterion

The first non-bit producer is named before any production edit.  A change is
eligible only if it is NEMO's compiled statement, is one-variable, makes the
owned ZAD boundary bit-exact given NEMO's recorded operands, and passes the
full Decision 43/45/55/59 scope: both VORTEX ladders; GYRE ladder and
day-30/day-240/day-360 rows; generic NEMO-GYRE card; private-workdir DINO
month gate; LOCK_EXCHANGE and OVERFLOW; a complete moved-row registry;
citations and plants.  GYRE must remain byte-identical unless its movement is
registered under the applicable user decision.  If the record cannot
distinguish the first operand, this round writes a fail-closed acquisition and
stops for that record rather than inferring.

## Choices

No configuration, card, timestep, resolution, carried-state, stabilizer, or
default choice is made.  ORCA2 remains on its separate lane.  The flux VORTEX
card's stage-3 owner is attempted only after the vector card's ZAD walk closes
or is proven to need a new record.

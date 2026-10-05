# PREREGISTRATION — round 192 / VORTEX round 8: acquire the stage-2/3 momentum term walk

Frozen before editing an instrument, building NEMO, or measuring a new row.
Lane tip at the start: `ca1f29be7`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round192`.

Round 191 leaves the vector-EEN card's first-over-bar row unchanged at `kt=2`:
`u = 3.3693297863401916e-06`, `v = 3.337024e-06`,
`T = 3.625489e-09`, and `ssh = 3.709010e-08`. The round-4 stage-local walk
already bounds the remaining momentum error to stages 2 and 3: given NEMO's
recorded stage entry, stage 2 is non-bit at `u = 1.6701813777553198e-06` and
stage 3 at `u = 3.3737007034684297e-06`. The current records carry only each
stage's entry and output. They do not carry the ordered momentum accumulator
or the `ww` operand inside either stage, so they cannot distinguish the terms
the operator ordered for this round.

## Compiled program and record required

The card's compiled program is
`tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90`.

- Stage 2 and stage 3 compute HPG first, then vorticity, then momentum
  advection at `stprk3_stg.f90:318-335`.
- The vector advection call is itself kinetic-energy gradient followed by
  vertical advection at
  `VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-139`.
- Stage 3 alone then adds lateral mixing and runs the implicit vertical solve
  at `stprk3_stg.f90:387-405`.
- Stages 2 and 3 build the vertical velocity consumed by vertical advection at
  `stprk3_stg.f90:287-294`.
- The stage-2 explicit update is the velocity-form statement at
  `stprk3_stg.f90:361-380`; stage 3 is integrated inside `dyn_zdf` before the
  common barotropic correction at `stprk3_stg.f90:403-419`.

No existing committed record contains those stage-local boundaries. Extend
the existing VORTEX acquisition and self-describing checker; do not create a
second trajectory harness. The new target is
`VORTEX_VEC_R8_OMIP_L1_P3`. Its writer is additive and writes, for `kt=1`
and stages 2 and 3 only:

1. the accumulator at zero, after HPG, after VOR/EEN, after KEG, after ZAD,
   and, for stage 3, after LDF;
2. the `ww` field handed to ZAD;
3. the stage `Kmm` velocity and free surface needed to identify the operand
   time level;
4. the stage output after the explicit update (stage 2) or implicit vertical
   solve (stage 3).

Each file uses the campaign's self-describing group layout: 16-byte magic,
header integers, then `(name, rank, n1, n2, n3, payload)` groups to EOF. The
checker predicts neither file size nor a complete header tuple. It hard-codes
only the magic and required group names, parses every group extent, and
refuses missing or duplicate groups.

## Frozen predictions and falsifiers

1. **Passivity.** The instrumented step-10 restart is byte-identical to the
   paired uninstrumented restart. Any changed restart refuses the record.
2. **Coverage.** Exactly two stage-term records exist, stage 2 and stage 3,
   and every required named group parses from its own header. A missing stage,
   missing boundary, duplicate group, trailing byte, or malformed extent
   refuses the record.
3. **Ordered controls.** The stage-2 LDF boundary is absent because the
   compiled stage does not call LDF there. The stage-3 record has it. Both
   records carry a nonempty `ww`. A contrary layout means the writer does not
   describe the compiled branch and refuses admission.
4. **Calibration.** The new build's ordinary step-entry and stage-output
   records reproduce the existing VORTEX-vector record at every shared field
   up to the registered compiled-rounding comparison; the restart identity is
   the binding passivity control. A missing shared record refuses admission.
5. **Header plant.** Corrupting one group extent exits nonzero and prints a
   named refusal. A zero exit means the parser is not fail-closed.
6. **Scientific prediction for the next round, not a result of this one.**
   At least one of HPG, VOR/EEN, KEG, ZAD, LDF, ZDF, or the stage velocity
   feeding WZV is non-bit given NEMO's recorded stage entry, because the whole
   stage-2 and stage-3 outputs are already non-bit. If every term is bit-exact,
   the first non-bit statement is instead the stage update/correction boundary;
   the walk must report that result rather than force an operator attribution.

## Landing criterion

This round lands no physics and changes no card. It is `STOPPED_FOR_RECORD`
after the additive patch, checker, syntax-only proof, preflight, tests,
citation gate and review are committed. Round 193 admits the operator-produced
record, runs the production-JIT term walk with a plant on every substituted
arm, names the first non-bit statement, and only then evaluates a candidate
under the full Decision 43/45/55/59 gate.

## Choices

No configuration, physics, timestep, resolution, carried-state, or default
choice is made. The target name and record contents are instrumentation only;
they follow the user's ordered term list and the compiled branch above.

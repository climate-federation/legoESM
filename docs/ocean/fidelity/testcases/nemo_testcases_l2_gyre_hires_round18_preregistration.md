# GYRE high-resolution round 18 transport/stencil preregistration

Date: 2026-10-19.  This document is committed before the transport injection
seam is implemented or any transport/stencil substitution is scored.  The
round-17 q-only result is already measured and is not preregistered here.

## Prior result and next question

The admitted common-state run starts scalar-math NEMO, vector-math NEMO, and
legoESM from the same vector-math day-180 restart (SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`).
On the registered stage-3, western-boundary, native-surface-V signed mean, the
round-17 q-only substitution returned `REFUTE_Q_FORMATION`: the baseline
legoESM-minus-vector value was `-8.6472830507021795e-16 m s-2`; scalar q left
`-4.3038944112607703e-16 m s-2` against the scalar NEMO term; and the
scalar-minus-vector NEMO envelope was `4.1439741816112516e-23 m s-2`.
Stage-3 HPG and advection stayed bitwise unchanged.

The executed V stencil consumes only two recorded inputs: the already formed
potential vorticity `zwz` and the zonal metric transport `zwx`.  The recorded
meridional transport `zwy` feeds the U stencil and is therefore the negative
control for the registered V row.  Round 18 asks whether the remaining
residual is in `zwx` formation or in the completed V stencil.

## Registered substitutions and verdicts

The precision, common restart, step (`kt=4321`), stage (`Kmm=2`), deck, and
western-boundary wet-span mask remain exactly those of round 17.  The round-17
signed-mean comparison of a pure legoESM term against NEMO's post-hoc
`after-before` subtraction remains reported as context, but it is not the
round-18 causal bar: NEMO executes an in-place accumulator statement.  The
source-associated comparison is therefore `NEMO_before + legoESM_term` against
`NEMO_after`, with that addition performed once in float64.  The primary exact
bar is zero unequal words on the registered native surface-V row.  This
composition correction is committed before any transport/stencil arm is run.

Each arm runs the same production-jitted legoESM step and changes only the
named already formed operand:

1. `q_only`: scalar-NEMO `zwz`, live legoESM `zwx` (the measured round-17 arm).
2. `Fu_only`: live legoESM q, scalar-NEMO `zwx`.
3. `q_plus_Fu`: scalar-NEMO `zwz` and scalar-NEMO `zwx`.
4. `Fv_control`: scalar-NEMO `zwy`, with live q and `zwx`.

All other velocities, thicknesses, masks, arithmetic association, and the
final stencil remain live.  HPG and advection must be bitwise unchanged in
every arm.  `Fv_control` must leave the complete surface-V array bitwise
unchanged because NEMO's V statement does not read `zwy`.

- **TRANSPORT FORMATION:** `q_plus_Fu` makes
  `NEMO_before + legoESM_term` bit-identical to `NEMO_after` on the registered
  row.  The smallest exact subset among `q_only`, `Fu_only`, and `q_plus_Fu`
  names the required input set.  A joint-only result is recorded as a coupled
  input boundary, not assigned to either operand.
- **COMPLETED STENCIL:** `q_plus_Fu` leaves any unequal registered word while
  the recorded scalar NEMO before/after arrays are finite.  The next walk is
  the final two-point arithmetic; no input-formation fix follows.
- **INVALID:** any manifest/passivity/provider check fails, an untouched row
  moves, `Fv_control` moves V, or the recorded completed term cannot be formed
  from its before/after arrays.

The existing active-q plant is strengthened because the normal q-only row now
already refuses: the plant must change the q-substituted scored V row relative
to the unplanted q arm.  Merely returning the same refusal is vacuous.  A
separate active-`zwx` plant must likewise move the `Fu_only` scored row.

## Executed source order

The selected stage-3 call order is HPG, vorticity/Coriolis, advection, and
lateral diffusion in
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:399-412`.
The configured total-vorticity ENE dispatch calls `vor_ene(...,ntot,...)` in
`GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:234-239`.
That routine forms q at `:536-557`, forms `zwx`/`zwy` at `:559-562`, and the V
statement reads `zwx` but not `zwy` at `:570-573`.

## Scope guard

This preregistration authorizes only private diagnostic/injection seams and
the existing scorer's extension.  A shared-path arithmetic change is eligible
only if the controlled substitutions localise it and the executed NEMO source
shows a transcription difference.  No public option, deck change, or long run
is authorized.

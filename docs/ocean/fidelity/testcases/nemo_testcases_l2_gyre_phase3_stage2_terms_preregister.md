# NEMO testcase lane 2 GYRE — stage-2 RHS term and tracer-coupling preregistration

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Parent measurement: stage-2 operand gate rooted at preregistration commit
`0a83eb45d47fb5e488bca917b58f18018ba1087b`.

The accepted boundary is the gauge-invariant stage-2 baroclinic RHS:
`1.4419237554233738e-11` U and `3.430804646662383e-11` V.  Its `7200 s`
increment accounts for the complete corrected stage-2 residual.  No component
owner is assigned before this term walk.

## Source order and coupling hypothesis

For stage 2, `stprk3_stg.F90:338-351` calls `eos(ts,Kmm)`, `dyn_hpg`,
`dyn_vor`, then vector-invariant `dyn_adv`.  `Kmm=3` is the corrected stage-1
state after the first Nnn/Naa swap (`stprk3.F90:215-225`).  The tracer program
is executed inside every `stp_RK3_stg` call (`stprk3_stg.F90:525-571`), so the
stage-2 EOS/HPG consumes stage-1 Kaa T/S, not frozen step-entry T/S.

The legoESM search found the canonical WS tracer stage helper, Kmm transport
geometry, two-step FCT identity, corrected momentum-stage hook, and momentum
diagnostic component machinery.  It also found a structural gap:
`ocean_model_latlon_cgrid.py:3834-3849` precomputes density from step-entry
T/S and `:4161-4195` reuses that frozen bundle for all momentum substages;
the WS tracer helper runs later at `:5599-5631`.  The registered hypothesis is
that missing stage-1 tracer -> stage-2 EOS/HPG coupling owns the near-zero
candidate RHS.  This must be measured, not assumed.

## Frozen measurements

A separate WRITE-only NEMO stream will capture stage-2 U/V accumulators:

1. immediately before HPG;
2. immediately after `dyn_hpg`;
3. immediately after `dyn_vor`;
4. immediately after `dyn_adv`.

Successive differences define the literal HPG, vorticity, and advection terms.
Each is converted to its fixed-depth baroclinic component before comparison,
so the later `zub/zvb` gauge cannot contaminate the result.  A private
diagnostic return will expose the already computed legoESM HPG, vorticity, and
advection components without changing production state.  The old combined
tendency must equal the source-ordered component sum at roundoff.

The existing oracle stage dump already contains stage-1 T/S.  The gate will
add private post-step exposure of legoESM tracer stages 1 and 2 and score both
T and S on the native active mask.  A one-variable causal arm may inject the
oracle stage-1 T/S bundle only into the stage-2 EOS/HPG evaluation.  It does
not replace velocity, transport, vorticity, advection, the external target, or
the returned tracer state.

## Confirm/refute table

- If the first component debt is HPG and oracle stage-1 T/S injection moves
  the stage-2 RHS/corrected Kaa by the faithful residual and clears them, label
  `CONFIRMED_CAUSAL_OWNER_OF_STAGE2_HPG_COUPLING`.
- If a later term is first over bar, continue at that term's operands before
  any owner label.
- If the tracer injection moves less than one tenth of the residual, label it
  `NEAR_NULL_NO_DISCRIMINATING_POWER`, not exonerated.
- If it improves but does not clear all downstream rows, label it
  `CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER`.
- A production landing is allowed only as the collapsed full WS-RK3 stage
  identity.  NEMO exposes no switch that freezes tracers for momentum stages,
  so no public micro-selector or mixed-stage Frankenstein configuration is
  permitted.

Scaling is printed before labels.  The component reader must reject corrupt
magic/header/payload.  A planted `+1` in the post-HPG U accumulator must fire
at magnitude at least `1.0`; a source-recurrence test must reject a permuted
HPG/vorticity/advection order.  All earlier artifacts are bit-identity
controls.  After each supported landing the CPU/fp64 gate reruns kt=2-10 and
continues to stage-2 T/S, stage 3, and the whole-step boundary until kt=2 is
AT-BAR or the register is honestly exhausted.

# NEMO SWG level-1 round 3 — SWE RK3 stage-1 production replay

Date: 2026-10-11. Base: `fbece7bc7c7f546aefef462dcc9634a99465d66b`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round3`.

## Preregistration

This section was committed before reading a numerical value from the admitted
stage-entry payload or running a legoESM stage measurement.  The complete
three-stage program is too large to implement and validate as one controlled
round, so this round lands the stage-1 program first, as the operator's round-3
instruction explicitly permits.  Stages 2 and 3, including all stage-3-only
forcing, remain fail-closed and unmeasured.

The admitted compiled program zeroes the momentum RHS, calls vector advection
and total ENS vorticity, adds the reduced-gravity surface-pressure gradient,
advances SSH with `rn_Dt/3`, rebuilds all QCO ratios, advances velocity from the
before level, applies U/V masks and lateral boundary conditions, then rotates
the stage indices at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:164-210`.  Vector advection dispatches
to C2 kinetic energy followed by vertical advection at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-139`; the one executed layer makes
the vertical-advection flux divergence structurally zero.  Continuity is the
live-thickness transport divergence and masked SSH update at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:135-149`, followed by the T/U/V/F QCO
ratios at `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:207-234`.

Frozen predictions and falsifiers:

1. The SWG card will select a guarded `swe_rk3` outer program while ordinary
   `step()` remains fail-closed until all three stages exist.  The production
   stage entry point will accept stage 1 only and execute under JIT in fp64;
   an absent/unknown selector, stage 2 or 3 silently running, a non-fp64
   output, or eager-only replay **REFUTES** the production boundary.
2. A nonzero manufactured stage will match independent source-literal C2 KE,
   ENS, pressure, continuity, QCO-ratio, and masked velocity statements.  The
   direct ENS division must remain direct and the QCO T ratio must not execute
   legoESM's generic positive-thickness floor.  Replacing C2 by the centered
   square-of-mean form, changing the source order, enabling a thickness floor,
   or dropping any wet/staggered mask **REFUTES** the transcription.
3. Driving the jitted stage from the admitted `kt=1, stage=1` entry will score
   its 13-field output bundle against NEMO's `kt=1, stage=2` entry.  Every
   computed or pass-through row is predicted bit-identical and at the
   `1e-15` relative bar.  Any unequal wet value is **DEBT**, never rounded up
   to AT-BAR, and **REFUTES** this prediction.
4. The zero and one controls each perturb one wet U stage-entry value by
   exactly one `nextafter` step *before* the jitted production stage.  The
   resulting production output must differ from its corresponding unperturbed
   production output and the ladder must exit nonzero.  A post-computation
   perturbation, a control that never enters JIT, or either control remaining
   bit-identical **REFUTES** the instrument.
5. `g=0.02` will be read from the card's constants by the only stage-1 gravity
   consumer, the inline pressure-gradient statements at
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:170-176`.  A literal gravity in
   the stage program, a different resolved value, or another stage-1 gravity
   consumer that does not receive the card value **REFUTES** the constants
   path.
6. No non-SWG card selects `swe_rk3`.  TSUNAMI, VORTEX, VORTEX_VEC, GYRE,
   LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 must retain their selectors and
   certified hashes/test disposition.  Any existing AT-BAR row leaving the
   bar or a first-over-bar boundary moving earlier **REFUTES** Rule 12
   eligibility.

No configuration or carried-state choice is introduced.  `key_RK3`, vector
advection, C2 kinetic energy, total ENS with `ln_dynvor_msk=.true.`, QCO/VCO,
one layer, `rn_Dt=3600 s`, and reduced gravity `0.02 m s-2` are already stated
by the shipped deck and the round-1 card.  The stage-entry value object is a
replay boundary, not a new prognostic carry.

## Measurement

Preregistration commit: `edcc0c9c15b2b4d33886c6be164fb3da9b3a1f77`.
Clean measured implementation: `bd41322dcfc3ea8f66e88bcbb8649706feb3dac8`.

1. **CONFIRMED.**  The card selects `outer_integrator="swe_rk3"`.  The
   production entry point is JIT compiled in fp64 and accepts stage 1 only;
   stage 2, stage 3, ordinary full-step execution, and a non-SWE selector all
   raise.  The remaining two stage arms stay in `SWG_UNMEASURED`.
2. **CONFIRMED.**  A nonzero SSH impulse is checked against the independently
   evaluated source-order pressure and QCO scalars: the selected wet U output
   is exactly `9.600000000000001e-07 m s-1` and the T ratio is exactly
   `2e-06`.  The stage is differentiable through that pressure path.  Existing
   stored C2 and ENS direct-division tests remain green; the SWE T-ratio path
   exposes the unfloored source value, while the generic guarded stretch
   operator is unchanged.
3. **CONFIRMED.**  Driving the production JIT from the admitted `kt=1,
   stage=1` entry gives 13/13 bit-identical rows against the recorded
   `kt=1, stage=2` entry.  Every row has `max_abs=0`, zero unequal wet cells,
   and status AT-BAR.  The clean artifact is
   `swg_stage1_clean.json`, SHA-256
   `600e270b3a5da0bd1d40ca98c45e6fd6c21e7806994e0047bb1bda120b36d60e`.
4. **CONFIRMED.**  Both controls change a wet U-staggered `utau_before` input
   at `[1,1]` before the JIT replay.  The zero control moves by `5e-324`; the
   one control moves by `2.220446049250313e-16`.  Each produces exactly one
   unequal production output, is classified DEBT by the bit endpoint, and
   exits 1.  The corresponding artifact SHA-256 values are
   `78233e8624e4ee4f0ecb8ec58866485ea8af3af6ddb3b9e6a92ae79f0b3fad0a`
   and
   `54c5ac90b7f3709dafe77bc0b92bf12066a4b126332a9a79cac6f9775d2cc11a`.
5. **CONFIRMED.**  The card, physics configuration, and stage consumer all
   resolve `g=0.02`.  The source-literal impulse test becomes wrong if that
   constants value does not reach the pressure statements at
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:170-176`; the stage contains no
   literal `0.02`.
6. **CONFIRMED.**  No other card selects `swe_rk3`.  The six existing fp64
   digests are unchanged: TSUNAMI `8bb5ed6c1bc031bd`, VORTEX
   `f7489467bc156303`, VORTEX_VEC `5041b91c67bd1443`, GYRE
   `f02d2f42a570f413`, LOCK_EXCHANGE `d5dfdf83b3a1e0ab`, and OVERFLOW
   `83de0010e1170032`.  DINO and ORCA2 retain their EEN programs and cannot
   enter the new outer dispatch.

## First non-bit statement

There is no non-bit statement on the measured stage-1 boundary: all 13 output
rows are bit-identical.  The first source-order statement not yet represented
is the stage-2 program and its `rn_Dt/2` stage length at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:213-215`; it is UNMEASURED, not
DEBT.  This round makes no claim beyond the preregistered kt=1 stage-1
boundary.

## Rule 12 disposition

The only existing shared statements touched are exposed names or optional
arguments whose old call signatures and arithmetic remain unchanged.  C2's
historical private name aliases the same public function; the default QCO
F-thickness builder still executes its old statements when no override is
supplied; and the generic positive-thickness stretch retains its original
body.  The new stage program is reachable only when the SWG card explicitly
selects `swe_rk3`.

The TSUNAMI card and ladder, both VORTEX families, the GYRE card reconciliation
suite, and the ORCA2 round-92 card gate were re-run on CPU.  One large combined
process hit the repository's documented JAX compiler-resource limit at the
GYRE full-step compile after reporting only the three already-known stale
GYRE/LOCK_EXCHANGE/OVERFLOW expected-digest pins.  The affected GYRE full-step
test passed alone in a fresh process, and ORCA2 passed 4/4 separately.  No
AT-BAR row moved and no first-over-bar boundary became earlier.

## Gates

The focused SWG battery is 162/162 green.  Direct shared-operator tests are
19/19 green.  The split cross-card runs keep all substantive tests green; the
only assertion failures are the three pre-existing stale digest pins, whose
actual values remain the unchanged values above.  The geometry gate remains
34/34 bit-identical, and its signed-y Coriolis, wet-cell, and layer-thickness
plants each exit nonzero.  The zero and one production-input plants each exit
1.  The citation gate accepts every cited compiled statement, and its shifted
stage-program citation plant exits nonzero.

## Independent review

Independent review run by the operator.

## OPEN

1. Build and preregister stage 2 from the admitted stage-2 entry, beginning at
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:213-215`, then score its output
   against the admitted stage-3 entry.
2. Build stage 3 with its stage-3-only viscosity, centred wind stress, and
   before-velocity layer drag; only then enable ordinary full-step execution.
3. Run the chained ten-step ladder, then acquire/score 30 days, then score the
   shipped 172800-step statistics.

No acquisition or configuration/carried-state decision is requested.

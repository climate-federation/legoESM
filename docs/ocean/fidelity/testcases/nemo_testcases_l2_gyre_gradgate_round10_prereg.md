# Certified GYRE reverse-mode gradient gate — round 10 preregistration

Date: 2026-10-24

This round repairs only the GPU-only production-primal failure measured by the
Round 9 operator acquisition.  It changes no ocean physics, GYRE configuration,
loss, perturbation, direction, finite-difference step, precision policy, or
agreement bar.  GPU agreement and resource numbers remain unmeasured until the
operator runs the corrected acquisition entrypoint on physical GPU 1.

## Existing evidence and owner

The Round 9 GPU artifacts were produced from clean commit `2c62c8e57`.  The
ordinary production rollout, the base and both finite-difference production
rollouts, and every JVP boundary were byte-identical.  The first one-step VJP
boundary was not: 15 state leaves changed, with maximum absolute differences
`7.105427357601002e-15` in both temperature and salinity.  Its raw transform
replay changed 18 leaves.  The first changed boundary leaf in pytree order was
velocity `u` (`6.938893903907228e-18`).  All compared leaves retained float64,
and both arms used the same fp64/native policy.

The owner is therefore the primal value returned by `jax.vjp` of the
production-step transform, not the separately materialized production step,
the native transcendental selection, or a dtype conversion.  The current
`compiled_step_vjp` returns that transformed primal as its advertised boundary
output.  An optimization barrier does not turn the transformed evaluation into
the already materialized production buffer; GPU reverse lowering still
reassociates/fuses the large step differently.  Forward-mode lowering does not
show the defect.

## Registered change

The explicit per-step AD composition will pass the separately executed,
materialized production after-state into each compiled VJP/JVP transform.  The
transform may replay the unchanged production program solely to obtain its
pullback or tangent, but the local boundary will return and propagate the
materialized production state.  The raw replay stays in the JSON as a
diagnostic and is never accepted as the forward state.  A planted replay-primal
selection must flip the byte gate.

The Round 9 GPU entrypoint will create its evidence directory itself before it
refuses pre-existing row files.

## Predictions and falsifiers

1. **P1 — CPU boundary identity:** the certified one-step CPU fp64/libm row
   remains byte-identical for VJP and JVP and retains the unchanged `1e-9`
   reverse/forward and `0.05` reverse/FD bars.  Any changed production leaf or
   off-bar derivative refutes P1.
2. **P2 — materialized-boundary control:** a hostile synthetic transform whose
   replay differs by one ULP returns the supplied production bits through eager,
   JIT, JVP, and VJP composition while retaining the replay derivative.  A
   planted replay selection must change the bits and fail the boundary gate.
   Failure to distinguish control and plant refutes P2.
3. **P3 — certified forward identity:** the ten-step GYRE trajectory remains
   byte-identical to the Round 9 production control.  Any scientific JSON or
   residual-NPZ SHA change refutes P3.
4. **P4 — GPU boundary identity:** under fp64/native on physical GPU 1, the
   one-step VJP and JVP boundary reports contain zero changed leaves.  Any
   changed boundary leaf refutes P4.  Raw replay differences do not refute P4
   because they are recorded and excluded from the production primal.
5. **P5 — GPU gradient and resources:** only after P4 passes, the unchanged
   1/6/60/360-step agreement rows and all nine resource rows are classified by
   their existing bars and plants.  No numerical outcome is predicted beyond
   the already measured one-step derivative agreement; absent or refused rows
   remain UNMEASURED.  No bar will be changed in response to a result.

## Acquisition

GPU rows are acquired only by the committed Round 10 operator entrypoint from
a clean commit.  Every JSON records fp64/native as **GPU, XLA transcendentals —
not the certified bit-exact policy**, commit provenance, physical GPU 1,
boundary identity, agreement, peak device memory, and blocked warm reverse
time.  The runner continues after classified failures and exits nonzero unless
every required row and plant has the registered outcome.

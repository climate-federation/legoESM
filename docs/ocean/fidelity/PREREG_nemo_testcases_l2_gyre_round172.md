# Preregistration — round 172, fixed-shape tracer-solve sensitivity

Committed before measuring any Round-172 arm.  Round 171 refused the
`e3t(Kaa)` versus tracer-content ranking because adding a fifth tuple slot
changed `7,538` state bytes in the production-jitted step.  This round first
replaces that variable-arity diagnostic seam with one fixed-shape input
contract.  Disabled, identity, baseline, directed, and planted forms all pass
the same six arrays and the same six-element dynamic selection vector through
the same production closure.

## Compiled statements and admitted records

The record build's selected adaptive-implicit branch forms the two
`e3w(Kmm)`-weighted off-diagonals and then consumes `e3t(Kaa)` in the diagonal
at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-474`.
It forms the temperature content RHS from the before tracer/content and the
now-level accumulated tendency/content at `trazdf.f90:549-567`.  The same
matrix and content enter the ordered backward recurrence at
`trazdf.f90:577-582`.

The admitted Round-125 record supplies `zwt_mix`, `e3w_Kmm`, `e3t_Kaa`,
`rhs_T`, every rebuilt matrix boundary, and every output for steps 1081--1440.
The Round-132 daily restart supplies the exact NEMO step-1080 entry.  Existing
admission and literal-rebuild controls must remain BIT before an arm is
interpreted.

## Fixed-shape production contract

The contract contains post-closure heat K, viscosity K, formed tracer K,
tracer `e3w(Kmm)`, tracer `e3t(Kaa)`, temperature content, and one dynamic
six-element boolean selection vector.  No Python arity or value branch may
distinguish the following forms after tracing:

1. `ordinary`: the existing production step with no diagnostic input.
2. `disabled`: the fixed contract is present with every selector false.
3. `identity`: every selector is true and every supplied array is the model's
   own value captured at the consumed boundary.
4. `complete_K_e3w`: every selector is true; only formed K and `e3w(Kmm)` are
   replaced by NEMO values, reproducing Round 169.
5. `complete_K_e3w_e3t`: as above, with only `e3t(Kaa)` additionally replaced.
6. `complete_K_e3w_content`: as the common base, with only temperature content
   additionally replaced.

The fixed contract is private diagnostic scaffolding, not a public
configuration, production-physics arm, or carried state.

## Frozen predictions and falsifiers

1. At step 1081, `disabled` and `identity` are each byte-identical to
   `ordinary`, and `identity` remains byte-identical over steps 1081--1440.
   Any moved state byte repeats Round 171's refutation, withholds all directed
   sensitivities, and ends the JAX-side ranking.
2. If prediction 1 passes, `ordinary` reproduces Round 169's developed
   day-240 T RMS `1.584259320940647e-02` K and `complete_K_e3w` reproduces
   `1.241262968697578e-03` K.  Either mismatch stops the ranking.
3. At step 1081 the e3t arm consumes recorded `e3t_Kaa` BIT on all 18,000 wet
   cells, and the content arm consumes recorded `rhs_T` BIT on all 18,000 wet
   cells.  Every registered upstream tracer boundary remains unchanged.
4. Neither operand alone carries half the Round-169 remainder: CONFIRM for an
   arm if it removes less than `6.20631484348789e-04` K from the common
   `complete_K_e3w` arm; REFUTE for an arm meeting or exceeding that threshold.
5. The live `e3t(Kaa)` arm removes more day-240 T RMS than the content arm.
   CONFIRM only if its reduction is strictly larger; equality or the reverse
   ordering REFUTES this prediction.  The larger measured reduction names the
   producer to walk next.  If both reductions are below the `2e-10` K
   run-to-run floor, both are exonerated and the next compiled solve boundary
   is promoted.
6. For each directed input, multiplying every finite positive targeted wet
   value at step 1081 only by exactly `1 + 2**-20` changes at least one
   registered consumed boundary and one wet day-240 temperature cell, prints
   `STATUS PLANT-FIRED`, and exits nonzero.  A zero count, success marker, or
   exit zero invalidates that operand's measurement.

If the fixed-shape disabled or identity form is not passive, no isolated JAX
closure result may replace it.  The receipt will preregister the required
paired NEMO-side record and request that acquisition under a new target; it
will not name either operand as the magnitude owner.

No production physics, configuration, carried state, immutable before arm, or
pending user decision changes.  No NEMO acquisition is authorized unless the
fixed-shape passivity prerequisite is refuted.

## Post-refutation acquisition preregistration

Prediction 1 is REFUTED and remains above: the fixed disabled form is
byte-identical through steps 1081--1440, but the all-selected identity form
moves `7,558` step-1081 state bytes (T `3,849` wet cells, S `3,680`, U/V/SSH
zero).  Before creating any replacement record, freeze the NEMO-side paired
sensitivity requested by Round 171's OPEN section.

1. Extend the existing developed-state observer to write only the model's
   ordinarily consumed `e3t(Kaa)` and temperature-content arrays for all 360
   steps.  The observer advances from the admitted NEMO step-1080 restart via
   an independently compiled ordinary production result.  Its traced and
   ordinary carried states must be byte-identical at every step.  A scale
   plant on each recorded family must move every selected wet cell and the
   manifest must fail closed on a truncated payload.
2. A new NEMO target, cloned file-for-file from the Round-125 producing card,
   runs three independent from-rest trajectories through step 1440: baseline,
   replace only the matrix's live `e3t(Kaa)` factor over steps 1081--1440, and
   replace only temperature content over the same steps.  The replacement
   arrays are the admitted ordinary legoESM record, mapped into NEMO's
   compiled interior box.  Salinity content is never replaced; the e3t arm
   retains NEMO's shared T/S matrix semantics.
3. This is an every-step one-family forced-input sensitivity, not a local JAX
   closure and not a landing candidate.  The baseline day-180 and day-240
   restarts must be byte-identical to the uninstrumented Round-125 source run.
   The two directed arms must differ from baseline at day 240, or the record is
   non-discriminating and refused.
4. Round 173 will score the three day-240 restarts against the immutable NEMO
   state.  The arm removing more of the production `1.241262968697578e-03` K
   complete-K/e3w remainder is the promoted producer.  Because the directed
   NEMO arms consume the free legoESM input sequence rather than their own
   evolving operand, this ranks family leverage only; it cannot by itself
   establish a source-exact landing.

The acquisition script must use a new target name, prove the additive source
patch with `gfortran -fsyntax-only`, print a named REFUSE line before every
nonzero exit, use shell timing rather than `/usr/bin/time`, and admit the
baseline by restart byte identity.  The operator runs it because PMIx sockets
are unavailable in the sandbox.

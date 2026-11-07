# ORCA2 round 87 preregistration — rung-0 OFF-owned surface fields

Base: `448fd3323e1f4788dbf610a927bbcb099709eb32`.  Claim label:
**independent**.  This round repairs write-only NEMO instrumentation; it does
not change a legoESM package, a hierarchy assignment, a NEMO physics statement,
the shipped ORCA2 card, carried state, a threshold, or any sea-ice selector.

## Frozen question

Can the rung-0 entry/stage recorder represent every surface operand whose
owning module is disabled as an explicit `ABSENT` self-describing entry, then
produce all 80 rank/step/stage frames without changing any of the 20 admitted
terminal restarts?

The resolved rung-0 deck has runoff and icebergs disabled.  The compiled source
allocates `rnf`, `rnf_b`, `rnf_tsc`, and `rnf_tsc_b` only for runoff, and returns
from iceberg initialization before allocating `utau_icb`, `vtau_icb`, or the
four `berg_grid` payloads.  In contrast, NEMO explicitly allocates the four
dumped `sbc_ice` arrays in its `nn_ice == 0` branch; those arrays are present,
not synthetic zero substitutes.

## Frozen predictions and falsifiers

1. The repaired root-rank surface record has exactly 35 named fields: 25 with
   finite payloads and 10 explicit `ABSENT` headers.  The absent names are
   `utau_icb`, `vtau_icb`, `rnf`, `rnf_b`, `berg_calving`,
   `berg_calv_hflx`, `berg_float_melt`, `berg_stored_heat`, `rnf_tsc`, and
   `rnf_tsc_b`.  Any different census refutes this prediction.
2. The checker accepts an absent runoff field only when both `namelist_cfg` and
   `ocean.output` resolve runoff off, and accepts an absent iceberg field only
   when both records resolve icebergs off.  A plant that changes either owner
   to on while leaving the field absent must refuse.  A field represented by a
   plausible all-zero payload instead of `ABSENT` must also refuse.
3. The optimized scalar-math acquisition writes 80 entry/stage frames and 20
   step/rank terminal restarts.  Any missing frame or restart refutes this
   prediction.
4. Every one of those 20 terminal restarts is byte-identical to the admitted
   round-83 rung-0 record.  One differing byte refutes the additions-only
   instrumentation claim and blocks admission.
5. The existing round-84 frame parser still reaches physical EOF for every
   frame, and its header, field-name, truncation, non-finite, and stamp plants
   all fire.  Any green plant blocks admission.

## Frozen landing rule

This round may commit the recorder, checker, acquisition launcher, tests, and
receipt only if their offline preflights and planted violations pass.  The
scientific record remains `ACQUISITION_NEEDED` until the operator runs the new
optimized target.  No legoESM physics or rung-0 card lands before the 80 frames
and restart-identity comparison admit.

## Frozen labels

All rung-0 results are **independent**.  Debug-build output is diagnostic only
and is never a scientific record.  Failed predictions remain in the receipt as
`REFUTED`; unavailable acquisition results remain `UNMEASURED`.

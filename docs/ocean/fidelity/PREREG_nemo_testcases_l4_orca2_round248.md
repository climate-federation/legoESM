# Preregistration — ORCA2 round 248 kt=5 land-support writer

Date: 2026-10-11. Frozen base: `02d3962d6`. Scope: diagnose the first
non-finite writer inside OMT-4 kt=5 stage 1 under the private atomic candidate
retained by round 247. Results remain separately labelled **independent** and
**given NEMO's entry**. Measurement is offline from passive completed states;
no in-executable observer is allowed. No deck, selector, carried state,
stabiliser, sea-ice field, or `unmeasured_features` entry may change.

## Cited source order

The executing oracle build constructs reference face depths from the associated
three-dimensional masks at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domain.f90:193-201`, then stores the
reference reciprocals with the surface masks, including dry columns, at
`domain.f90:212-215`:

```
r1_hu_0 = ssumask / (hu_0 + 1 - ssumask)
r1_hv_0 = ssvmask / (hv_0 + 1 - ssvmask)
```

The live ratios use those stored reciprocals at `domqco.f90:207-218`. The
stage correction and transport consume `r1_hu_0/(1+r3u)` and
`r1_hv_0/(1+r3v)` at `stprk3_stg.f90:269-285`. Therefore the source-ordered
land-support walk is: raw `hu_0`/`hv_0`; resolved `ssumask`/`ssvmask`;
reference reciprocals; live reciprocals; `zub`/`zvb`; corrected stage
velocities/transports; then continuity/W and the completed stage state.

The current shared QCO helper derives its reciprocal predicates from
`hu_0 > 0` / `hv_0 > 0` at `packages/ocean/legoesm/ocean/vertical.py:241-244`.
Round 248 tests whether the private atomic route supplies depths and masks from
different mesh routes, rather than assuming that this algebra is safe.

## Frozen predictions and falsifiers

1. **R248-P1 — reciprocal-mask owner.** At `[j=1,i=49]`, or at another
   zero-depth U/V column no later in source order, the candidate's reciprocal
   wet predicate differs from NEMO's resolved surface mask. The predicted
   signature is raw face depth `0`, resolved surface mask `0`, candidate wet
   predicate `1`, and a non-finite reference or live reciprocal. Report those
   six operands at `[1,49]` and census every `hu_0==0` / `hv_0==0` column.
   Any exact predicate/reciprocal table, or any first non-finite earlier than
   the reciprocal, **REFUTES** P1.
2. **R248-P2 — first writer.** If P1 confirms, routing the reciprocal predicate
   to the same resolved `ssumask`/`ssvmask` used by the depth and transport
   makes all reference/live reciprocals, `zub`/`zvb`, and corrected transports
   finite on full support, and the atomic OMT-4 run crosses kt=5. If P1
   refutes, retain the refutation and continue the offline source-order table
   until the first finite-to-non-finite writer is named; no reciprocal change
   is eligible to land.
3. **R248-P3 — atomic completion.** A cited repair may proceed only if the
   completed-V transport remains bit-exact and the complete private unit
   crosses kt=5, then the former kt=8 boundary, and completes kt=10 under both
   labels. Any earlier refusal **REFUTES** the repair as the complete partner;
   report the first boundary and keep the unit private.
4. **R248-P4 — Decision 96.** Only a candidate satisfying P2-P3 proceeds to
   OMT-1..4, independent rung 0, given-entry rung 10, GYRE, DINO, tanks,
   moved-row, and month gates. It lands only under the standing Decision-96
   predicates. Otherwise the model change is retracted and the round is HELD.
5. **R248-P5 — controls.** Both claim labels, full land support, the named
   writer, candidate passivity, exact completed transport, completion, and
   citation plants must each refuse when violated.

## Stop conditions

The first non-finite value is attributed to the statement that writes it, not
to the downstream live-W guard. A plausible zero on land is not accepted:
the oracle's mask-safe reciprocal must be checked on every land column. No
configuration choice or NEMO acquisition is authorised. Decision 116 remains
pending separately and is not re-emitted.

# Preregistration: NEMO-testcases L2 GYRE round 82 kt=2 external-step walk

Date: 2026-09-13. Frozen after reading the Round 79--81 preregistrations and
receipts, the operator's acquisition completion line, the Round-81 record
reader, the live shared trace registry, and the acquired target's compiled
source, but before reading a scientific value from the acquired record or
running a live comparison.

## Inherited boundary and retraction

The Decision-37 six-absolute-history candidate remains **HOLD**. Round 80
refuted its required whole-step kt=2 U/V movement: all 954 registered rows were
unchanged and the kt=2 U/V normalized maxima remain
`2.7478404751243857e-12` / `3.305560306813421e-12`. Round 81 independently
reconfirmed that the candidate makes the kt=2 substep-1 U current, histories,
and midpoint bit-exact and that the first observed U mismatch is substep-2
current, 580/580 wet faces, maximum `3.032539284029834e-09`. Neither fact is
reinterpreted in this round.

The operator reports that the Round-81 acquisition exited zero and printed
`CONSUMED_FIELD_ADMISSION PASS: exact=46/70 changed=24 admitted=264 plant=False`
and `ROUND81_BTSTEP_READY`. This is admission evidence, not a model comparison.
The acquired record must still pass its stamp, byte-layout, replay, inherited
record, twin, and plant checks before a scientific array is read.

## Compiled source order

The running target is
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90`. The compiled
branch initializes current U/V/SSH independently of the six carried absolute
histories at `:341-381`. For every external substep it forms the forward
coefficients and U/V/SSH midpoints at `:481-541`, face depths and metric
transports at `:553-588`, and the continuity divergence and SSH result at
`:598-615`. It forms the backward coefficients/SSH interpolation and surface
pressure gradient at `:660-680`, then Coriolis, explicit bottom drag, imported
slow forcing, and the executing vector-form U/V update at `:683-714,726-750`.
Boundary exchange precedes the absolute-history swap at `:792-850`.

The imported `zu_frc`/`zv_frc` fields are copied from the complete two-
dimensional forcing and have the selected Kmm barotropic Coriolis term removed
at `dynspg_ts.f90:291-327`. Their compiled producer depth-averages the full
three-dimensional K(r)hs momentum RHS, then adds baroclinic drag and wind at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:141-236`; the active
GYRE switches skip the later atmospheric-pressure, ice-load, and wave-load
arms before the split-explicit call at `:243-308`.

## Frozen admission and direct comparison

First admit the acquired record without inspecting model values:

1. producer commit and stream stamp must equal `295a42edc9d9f45707a5349097e7a0183f57463c`;
2. exact stream size must be 10,439,164 bytes, fp64, kt=2, 50 consecutive
   substeps, 37 arrays per substep, owned bounds 3:34 by 3:24, and EOF;
3. all compiled arithmetic replays and the identity with the admitted Round-77
   U record must remain bit-exact;
4. stamp, header, truncation, replay-ULP, swap-ULP, and U-midpoint-ULP plants
   must exit nonzero; and
5. twin admission must retain exactly 46/70 byte-identical records, 24
   classified changed records, and 264 admitted differences; both the consumed-
   field and unexpected-inventory plants must exit nonzero.

Any failed check refuses the record and stops this round without a scientific
claim.

The live comparison uses the production-JIT, CPU/fp64/libm kt=2 seeded context
already shared by the Round-72/78/79 walks. It maps the one shared named
barotropic trace to the acquired arrays and scores only the native wet T/U/V
masks. For each substep the frozen source order is: three forward coefficients;
U/V/SSH current, b, and bb; U/V/SSH midpoint; midpoint U/V depth; metric U/V
transport; SSH forcing, divergence, and continuity result; four backward
coefficients and backward SSH; U/V pressure gradient; U/V Coriolis; U/V drag
coefficient and inverse depth; U/V combined trend; slow U/V forcing; exchanged
U/V result; and post-swap U/V/SSH current. Scalars and arrays require fp64 bit
equality; differing-cell count and absolute maximum are recorded for every row.

Round 81's frozen prediction is preserved literally: every row through the
combined U/V trends at substep 1 is bit-exact and the first non-bit row is
`slow_u` or `slow_v`, the imported whole-step forcing. It is **CONFIRMED** only
if no earlier source-order row differs and at least one of those two rows is
non-bit. It is **REFUTED** by any earlier mismatch, by both slow rows being
bit-exact, or by a different first row. No post-hoc reorder is permitted.

The comparison has three non-vacuity controls. One ULP in an exact early
history row must make that row first; one ULP in the oracle slow-U row must move
the reported slow-U result; and replacing the oracle slow-U row with the live
row must either move the first mismatch or loudly refuse because slow-U was not
the mismatch. Each plant exits nonzero.

## Conditional next owner

If `slow_u` or `slow_v` is first, it is an input, not an external-step formula;
the next walk remains the already compiled producer in `stp2d.f90:141-236`.
This round may reuse the admitted kt=2 K(r)hs and stage records to compare, in
that compiled order, the three-dimensional RHS, reference-thickness depth
average, baroclinic-drag correction, wind contribution, and final forcing.
The first non-bit input owns the following round. A production numerical change
is eligible in this round only if all operands of the first shared statement
are bit-exact and that shared result alone is non-bit. Otherwise no downstream
formula may be changed to compensate for a non-bit input.

## Rule 12 and exclusions

| lane | frozen Round-82 disposition |
|---|---|
| GYRE source order | Admit the 37-array record and compare the production kt=2 substep trace in the compiled order; preserve the `slow_u`/`slow_v` prediction as CONFIRMED or REFUTED |
| GYRE kt=1..10 | Recorded before arm remains `decision36_nemo_face_shear/after_kt1_10.json`; held-candidate artifact remains Round 79's after ladder; do not rerun either unless a production statement changes |
| GYRE days 1..30 | Recorded before arm remains `decision36_nemo_face_shear/after_day_gap.json`; Round 79 candidate remains prior evidence only; do not rerun unless a production statement changes |
| LOCK_EXCHANGE-zco | Preserve Round 79's 50-row result unless the shared split-explicit program changes; then rerun its recorded kt rows |
| OVERFLOW-zps | Its boxcar card does not execute the cross-window continuation history statement; any newly changed shared substep statement must be shown not to execute or measured against its record |
| DINO | SHARED-STATEMENT RISK: DINO uses its leapfrog card and separate histories, but any shared forcing/substep statement and restart surface remains at risk; 96--98% row cancellation forbids neutrality inference |
| ORCA2 | UNMEASURED-WITH-SPEC: independently align T/S/U/V/SSH and six absolute histories on native staggered masks; require elementwise fp64 equality and normalized L-infinity through kt=1..10; reject AT-BAR loss, earlier first-over-bar, or wet-face history mismatch |

No configuration/default, selector, coefficient, timestep, carried-state
representation, stabilizer, year harness, reconciliation gate, freshwater pair,
#1484 guard, held manifest, canonical NEMO source, or existing evidence changes.
No scientific choice is made. If the walk reaches a required configuration or
carried-state choice not already authorized by Decision 37, the round stops
with `DECISION_NEEDED`.

# ORCA2 round 254 preregistration — south-boundary association audit

Date: 2026-10-12. Frozen base: `b65a32ff0`. Scope: OMT-4, both
**independent** and **given NEMO's entry** labels, with round 253's complete
private fold/transport unit. The admitted round-253 external-substep record is
available, but the first measurement is the cheaper candidate-side south-row
audit required by the round-253 OPEN. No deck, selector, carried-state
contract, stabiliser, sea-ice field, or `unmeasured_features` entry may change.

## Compiled statement and candidate image

The compiled OMT-4 oracle associates `ua_e`, `va_e`, `hu_e`, `hv_e`,
`hur_e`, `hvr_e`, and `ssha_e` in one `lbc_lnk` call at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:730-741`.  The
T-pivot north-fold implementation writes only the northern `ipj`/`ipj-1`
rows at `lbcnfd.f90:584-700`; it does not rewrite the southern closed-boundary
row.

legoESM's private compact image currently constructs the V velocity, depth,
and reciprocal by stripping both boundary rows and calling
`pad_ns_vector_v`/`pad_ns_scalar` at
`barotropic_latlon_cgrid.py:737-744`.  Those helpers deliberately create a
south boundary (`operators_latlon_cgrid.py:178-208,304-323`).  This round asks
whether rebuilding that row, rather than preserving the incoming closed-row
values, is the first source-exact defect exposed by the complete unit.

## Frozen predictions and falsifiers

1. **R254-P1 — provenance, passivity, and record admission.** The round-253
   admission remains `PASS_R253_OMT4_SPG8_ADMISSION`. Both claim labels run
   fp64/libm on CPU and reproduce the ordinary completed state bit-for-bit
   through the inspected boundary. A moved production bit, stale D114 operand
   route, missing label, or unadmitted record invalidates every later number.
2. **R254-P2 — south-row writer.** At the first external-substep association,
   rows `j=0..2` of pre/post `v`, `depth_v`, `inverse_v`, and `eta` are compared
   under unit OFF and ON. NEMO and unit OFF retain the southern closed-boundary
   values. Unit ON first differs in `depth_v_post` or `inverse_v_post` on the
   south row because the scalar pad rebuilds it; `v_post` remains the NEMO zero
   boundary. CONFIRM requires an ON-only exact inequality at `j=0` (the compact
   image of NEMO's southern `j=1`) before the round-253 `[j=1,i=49]` damage.
   If every ON row equals OFF/NEMO, this prediction is REFUTED and the admitted
   kt=8 midpoint operands are split in compiled order instead.
3. **R254-P3 — one-statement repair.** If P2 confirms, the private seven-array
   association is changed to write only NEMO's northern fold rows for every
   field while preserving each incoming south row. The direct association
   table must then be exact on `j=0..2`; a south-row sentinel plant must fire
   against the old reconstructing behavior, and a north-fold plant must prove
   the repair did not disable the T-pivot exchange.
4. **R254-P4 — atomic rescore.** The north-only repair is tested only as part
   of the whole private unit. Pre-registered success is completion of OMT-4
   kt=8..10 under both labels, with the round-253 live-W refusal gone. If it
   still refuses, the round is HELD and names the first boundary from the
   admitted kt=8 histories; no partial association lands.
5. **R254-P5 — landing.** A package change lands only if P2-P4 confirm and the
   complete unit passes Decision 96 on OMT-1..4, independent rung 0, given-entry
   rung 10, GYRE, DINO, tanks, every moved-row registry, month, citation,
   independent review, and tests. Otherwise packages are restored and the
   receipt retains every failed prediction and measured boundary.
6. **R254-P6 — controls.** Clean-tree, label coverage, passivity, admitted
   record, south-row support, first-writer order, old-behavior sentinel,
   north-fold non-vacuity, terminal completion, and one-bit controls must each
   refuse when planted.

## Stop conditions

The operator has already admitted the round-253 record; no acquisition is
planned. Decision 116 remains pending and is not re-emitted. No floor, clip,
damping, alternate configuration, or partial held association may land.

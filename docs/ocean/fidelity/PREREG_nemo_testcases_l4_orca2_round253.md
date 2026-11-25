# ORCA2 round 253 preregistration — kt=7-to-8 live-thickness boundary

Date: 2026-10-12. Frozen base: `99232a720`. Scope: OMT-4, both
**independent** and **given NEMO's entry** labels, with the complete private
fold/transport unit used by round 252. This round is measurement-only until a
single cited NEMO statement both owns the first finite-to-invalid boundary and
qualifies under Decision 96. No deck, selector, carried-state contract,
stabiliser, sea-ice field, or `unmeasured_features` entry may change.

## Existing instrument and compiled order

The round reuses the already-certified completed-stage return and external
substep trace. It must reproduce the ordinary private-unit state bit-for-bit
through completed kt=7 before any trace value is read. No new in-executable
observer is allowed.

The executing oracle build forms midpoint transports, writes the external
continuity update, and then diagnoses the face SSH/depth in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:476-599`. In source
order the candidate first writer is the continuity/SSH statement at
`dynspg_ts.f90:546-558`; the later velocity, depth and reciprocal writes are
at `dynspg_ts.f90:655-718`. NEMO then installs the external SSH, constructs
the after-stage `r3t`, and exchanges its U/V/F ratios at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:138-178` before the
three-dimensional stage program consumes the live thickness.

## Frozen predictions and falsifiers

1. **R253-P1 — passivity and coverage.** Both labels reproduce the round-252
   private unit: completed kt=1..7, finite kt=8 entry, and the same
   `raw-mesh e3w_int must contain only finite values > 0` refusal before kt=8
   completes. Every completed state obtained with the existing trace is
   array-identical to the independently compiled untraced state. A moved bit,
   missing label, non-fp64/non-CPU execution, stale geometry route, or an
   invalid kt=8 entry invalidates every downstream number.
2. **R253-P2 — first invalid writer.** The first invalid live W thickness is
   produced during kt=8 stage 1 by one external substep's continuity/SSH write
   (`dynspg_ts.f90:546-558`): the prior substep exit is finite and the first bad
   exit has `ssh <= -ht0` at the guard cell. If every stage-1 external-substep
   exit remains finite-positive, this prediction is **REFUTED** and the first
   later completed-stage writer becomes the retained boundary.
3. **R253-P3 — growth table.** Before any owner claim, print kt=7 entry and
   stages 1..3, kt=8 entry, and every exposed kt=8 stage-1 external substep.
   For SSH/T/S/u/v and live e3w report maximum and RMS error versus the admitted
   NEMO stage where one exists, split into the northern fold band and interior;
   report the first boundary whose maximum grows by more than 10x. Also print
   the first invalid `[j,i,k]`, its `ssh`, `ht0`, `1+ssh/ht0`, and fold-identity
   counts. Missing boundary coverage or a non-finite NEMO comparison is a hard
   refusal.
4. **R253-P4 — ownership.** The guard is damage, never an owner. A statement
   owns the walk only if the immediately preceding boundary is finite, its
   output is the first invalid quantity in compiled order, the production
   D114 operand route is used, and a one-bit/boundary-order plant fires. If the
   continuity write is first but all of its live operands are already
   explosive, ownership remains upstream and the round is HELD at the first
   non-bit operand; it does not land a downstream guard or stabiliser.
5. **R253-P5 — landing.** A source change is eligible only if a cited missing
   NEMO statement closes its own boundary and the complete private unit then
   completes kt=8..10 under both labels. It must additionally pass OMT-1..4,
   independent rung 0, given-entry rung 10, GYRE, DINO, tanks, moved-row,
   month, citation, review and test gates. Otherwise packages remain unchanged
   and the receipt is **HELD** with the first writer named.
6. **R253-P6 — controls.** Claim-label coverage, clean-tree provenance,
   production-route identity, passivity, boundary order, finite-entry,
   first-invalid, fold/interior support, growth threshold, NEMO guard, and
   one-bit plants must each refuse when violated.

## Stop conditions

No NEMO acquisition is planned: the OMT-4 stage record and existing external
substep trace are sufficient for this boundary classification. If a required
rank-complete operand is absent after the first writer is located, stop with a
fail-closed acquisition script rather than infer it. Decision 116 remains
pending and is not re-emitted. No floor, clip, damping, alternate
configuration, or partial unit may land.

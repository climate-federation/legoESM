# NEMO testcase Lane 4 — ORCA2 Phase-2h systematic-writer preregistration

Date: 2026-09-06

Parent: `c19944cea7ed4d2ca5da97a70237509e3892a9b6`

Decision owner: user Phase-2h resume (ASKED).  A one-writer-at-a-time response
to nondeterministic record storage is withdrawn.  Every `WRITE` statement in
every config-local ORCA2 oracle writer is audited before another acquisition
run is accepted.

## Frozen writer-audit claim

The completed Phase-2g O1-canonical twins are
`variant_icebergs_off_phase2g_o1canon_a_10step_np2` and
`variant_icebergs_off_phase2g_o1canon_b_10step_np2`.  Both must first pass the
existing schema, ordinary-output, and planted-control gates.  Their complete
92-record inventories are compared by raw bytes.  The observed sole differing
stream, `oracle_rkstage1_transport_operands_kt00000001.bin`, is a trigger for
the systematic audit, not a special-case correction.

For each config-local `WRITE`, the committed audit will enumerate every
operand, its declared/storage shape (scalar, `A1D`, `A2D(n)`, full
`jpi,jpj[,jpk]`, category/layer extension, or literal), its active resolved
owner, and whether halo, land, or inactive/unallocated components can be
undefined.  Every potentially undefined array payload will be passed through
an existing or new zero-first WRITE-only view.  A view may read its source and
the resolved masks/ownership bounds, but may assign no model array and may not
change any model expression, header, payload ordering, derived count, time
level, or call cadence.

Acceptance after the replacement twin is mechanical: both runs independently
pass schemas, identity, and plants, then **92 / 92 records must be raw-byte
identical**.  Defined-cell equality is an interim diagnostic only and cannot
pin the canonical VARIANT V2 record set.  Earlier runs and records remain
preserved and provenance-labelled; none is deleted.

## Frozen ocean ladder

The Phase-2g O1-M result stands: each of the nine CORE fields is exact on
13,320 rank-0 cells.  The O1 record used for the shared-bulk handoff is the
Phase-2g arm-A provisional O1, whose source-defined frame is valid; its path,
digest, schema field list, and frame semantics will be recorded in the
receipt.  O1-B remains a shared `LANE3B_OWNER` debt and is not repaired here.

The next comparison uses the oracle O1 bulk outputs (`qsr`, `qns`, `emp`,
`utau`, `vtau`, `taum`, and `wndm`) as `ORACLE_SUPPLIED` operands.  This is an
explicit operand substitution, not certification of the bulk operator.  The
ordered boundaries are:

1. RGB `tra_qsr` using the deck chlorophyll (`ORCA2_OWNER`);
2. EOS/HPG (`GYRE_OWNER`, shared);
3. external mode with tripolar north fold (shared external mode; fold portion
   `ORCA2_OWNER`);
4. stage transports (`GYRE_OWNER`, shared);
5. FCT tracer advection (`GYRE_OWNER`, shared);
6. BBL (`ORCA2_OWNER`, census-round shared identity);
7. TKE/EVD/IWM entry (shared TKE/ZDF, ORCA2-owned IWM input path).

Each entered boundary runs production JIT on CPU, fp64 plus scalar-libm, and
is scored cellwise on the rank-0 source-defined domain.  AT_BAR means exactly
`0 / n`; every scorer receives a one-variable planted mutation through the
same validator and that invocation must exit nonzero.  Stop at the first
over-bar boundary in NEMO execution order and name its owner.  Per the standing
ownership rule, Lane 4 may change only ORCA2-specific input/fold/RGB/runoff/
geothermal/IWM/BBL setup.  Shared numerical debt is registered and handed to
its owner without a Lane-4 repair.  SI3 remains
`UNMEASURED_PENDING_ICE_MERGE` with oracle-supplied exchange fields.

## Run/build boundary

After the systematic writer correction, rebuild `ORCA2_OMIP_L4` with
`conda-scalarmath`, require zero dynamic `_ZGV*` symbols, and prepare two fresh
hash-guarded 10-step directories using two ranks (`jpni=2`, `jpnj=1`).  The
agent does not execute MPI.  If the post-bulk ladder needs an additional NEMO
record, prepare that run at the same stop instead of improvising a model-side
quantity.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| systematic census of all config-local writes | ASKED | complete operand/shape/undefined-risk table, source-cited |
| canonicalize every risky write operand | ASKED | zero-first WRITE-only views; no model assignment |
| prepare a fresh reproducibility twin | ASKED | same binary/deck/layout, two independent directories |
| pin canonical VARIANT V2 | ASKED after rerun | only after 92 / 92 raw-byte identity |
| certify ORCA2 `fld_read` | ASKED | retain Phase-2g 0 / 13,320 result for all nine fields |
| hand NCAR bulk to the ice-thermo lane | ASKED | exact O1 record contract in receipt; no shared repair here |
| continue after bulk by operand substitution | ASKED | oracle O1 outputs, visibly labelled `ORACLE_SUPPLIED` |
| execute MPI | UNASKED and prohibited | user shell executes prepared launchers |
| change a shared numerical operator | UNASKED and forbidden | register first debt and owner only |
| delete earlier runs or records | UNASKED and forbidden | preserve and label all prior evidence |

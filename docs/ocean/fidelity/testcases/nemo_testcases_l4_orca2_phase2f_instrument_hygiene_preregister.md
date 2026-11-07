# NEMO testcase Lane 4 — ORCA2 Phase-2f instrument-hygiene preregistration

Date: 2026-09-06

Parent: `eac79aa7d006836cf2b58b0b6f09b480caa735ea`

Decision owner: user Decision 8 (ASKED).  Undefined record slots are instrument
hygiene, not physics.  The writers must be reproducible, and the interim gate
must compare only source-defined slots while naming every excluded class.

## Frozen claim and stop boundary

This round changes only config-local `MY_SRC` WRITE paths.  It does not assign a
model array, reorder a model expression, change the resolved deck, change a
record header, or change a payload count.  The numerical ladder remains stopped
before the mapped-input O1 score until two runs of one rebuilt executable prove
record reproducibility.  The accepted icebergs-off uninstrumented run remains
the ordinary-output identity control.

The seven inherited streams found nondeterministic in Phase 2e are the complete
scope:

| stream | copied writer | source-defined slots retained | canonical zero slots |
|---|---|---|---|
| `oracle_slow_forcing` | `stp2d.F90` | owned wet T/U/V cells at each written level | rank-0 halo bands and masked land |
| `oracle_ocean_surface_input` | `stprk3.F90` | owned wet T/U/V cells; allocated active components | rank-0 halo bands, masked land, and unallocated iceberg components |
| `oracle_rkstage3_wzv` | `traadv.F90` | owned wet T cells for both `ww` frames and `pFw`; inactive unallocated `wi` has zero elements and no schema slot | rank-0 halo bands and masked land |
| `oracle_bt_substeps` | `dynspg_ts.F90` | owned wet T/U/V cells | rank-0 halo bands and masked land |
| `oracle_bt_drag_operands` | `dynspg_ts.F90` | owned wet U/V cells | rank-0 halo bands and masked land |
| `oracle_bt_advmean_operands` | `dynspg_ts.F90` | scalar/weight values plus owned wet U/V cells | rank-0 halo bands and masked land |
| `oracle_bt_ordered_operands` | `dynspg_ts.F90` | scalar coefficients plus owned wet T/U/V/F cells | rank-0 halo bands and masked land |

`owned` means the rank-0 local indices corresponding to the halo-free
`A2D(0)` slab: full-array indices
`1+nn_hls:jpi-nn_hls, 1+nn_hls:jpj-nn_hls`.  For already reduced arrays the
whole stored slab is owned.  Wetness is selected with `ssmask`, `ssumask`,
`ssvmask`, or `ssfmask` according to the NEMO grid staggering; 3-D fields use
the matching level mask.  Writer-local result arrays are initialized to
`0._wp` and only the intersection of those two sets is copied from model
operands.  No multiplication by a zero mask is accepted because it can retain
NaNs or read undefined storage.  Scalars and the initialized time-integration
weight vector are not masked.

The interim defined-cell identity gate mirrors this inventory and may exclude
only the named canonical-zero classes.  It must still compare all headers,
scalars, weights, and source-defined f64 slots byte-for-byte, report the actual
included/excluded counts per stream, and carry a planted one-ULP mutation in an
included slot that exits nonzero.  This is a temporary comparison of the
schema-fixed run to the prior accepted instrumented run; it is not authority to
waive the twin-run raw-byte gate.

## Mechanical acceptance gates

Two separately prepared 10-step directories will use the same rebuilt
scalar-math executable and identical hash-guarded inputs.

1. Each run reaches `time.step=10`, `LAUNCHER_RC=0`, `RUN DONE`, and produces
   the frozen 92-file inventory.
2. Every record passes its frozen magic/header/frame/count/finite schema walk;
   the O1 record has both frames and its binding header/count and one-ULP plants
   exit nonzero.
3. Every one of the 92 `oracle_*.bin` files is exact bytes between the two runs
   (`92 / 92`, with the denominator derived from the inventory loop).
4. Each run independently passes ordinary-output identity against the accepted
   uninstrumented VARIANT control: all restart shards exact, history data raw
   bytes exact under the registered global-timestamp exception, and the other
   ordinary files under their Phase-1 rules.  Counts are derived from loops.
5. A plant mutating one source-defined payload value in each of the seven
   stream families must make the defined-cell interim comparator exit nonzero.
6. `nm -D` on the rebuilt executable reports zero `_ZGV*` symbols and the build
   retains `-fno-tree-vectorize`.

If any record differs between the twins, this round stops at instrument hygiene
and names the exact stream/field/slot.  If they pass, the next turn may validate
and pin the canonical VARIANT records, then resume O1 and the ocean ladder in
NEMO execution order.  No legoESM numerical boundary is entered in the present
rerun-preparation turn.

## ASKED / UNASKED preregistration

| item | status | disposition |
|---|---|---|
| canonicalize every undefined/unowned slot before writing | ASKED | implement with zero-filled writer-local values only |
| compare defined cells in the meantime | ASKED | exact comparison with exclusions and counts enumerated above |
| prove reproducibility with two cheap reruns | ASKED | prepare both directories from one binary; user executes |
| validate O1 and ordinary inertness | ASKED | retain schema, plants, and dynamic identity gates |
| continue numerical ladder | ASKED after hygiene is proved | deferred because NEMO reruns are required |
| alter a model operand or shared numerical operator | UNASKED and forbidden | not done |
| delete or replace prior runs | UNASKED and forbidden | all retained and labelled by provenance |

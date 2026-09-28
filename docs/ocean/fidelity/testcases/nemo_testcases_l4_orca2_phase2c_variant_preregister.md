# NEMO testcase Lane 4 — ORCA2 Phase-2c variant preregistration

Date: 2026-09-06  
Parent: `1b83a3bcbeee08c299303067f5ca2f5c8bda69a1`  
Prior handoff: `nemo_testcases_l4_orca2_phase2b_exchange_handoff_receipt.md`  
Scope: icebergs-off oracle arm, tripolar Coriolis repair, and three run recipes

## User decision 7: comparison oracle

The comparison oracle is a one-variable variant of the shipped ORCA2 deck:

| resolved field | shipped-deck record | comparison variant |
|---|---:|---:|
| `ln_icebergs` | `.true.` | `.false.` |

Every other deck byte is held fixed apart from the already registered run-length
controls.  In particular, resolved `ln_rnf_icb=.false.` stays false and
`nn_fsbc=2` stays two.  The existing Phase-1 and Phase-2b records are retained
and labelled **SHIPPED_DECK_RECORD**; they are superseded only as comparison
operands and are not deleted or relabelled as variant results.

This arm is named **VARIANT oracle** everywhere.  Its three registered runs are:

1. an instrumented ten-step run (`nn_itend=10`, `nn_stock=10`,
   `nn_istate=1`) with the complete accepted 90-stream instrument set and the
   post-`sbc` surface-input stream;
2. an uninstrumented ten-step byte-identity control with the same controls; and
3. an uninstrumented 30-day reference (`240` steps at `rn_Dt=10800 s`,
   `nn_stock=240`, `nn_istate=0`).

All use the scalar-math build policy (`-fno-tree-vectorize`, zero dynamic
`_ZGV*` symbols), CPU, two MPI ranks, and the established `jpni=2,jpnj=1`
layout.  The user shell executes all MPI runs; this turn stops after preparing
the three self-contained, hash-guarded directories.

## Frozen expected change

`sbcmod.F90:457-470` computes `utau_icb,vtau_icb` only when
`ln_icebergs=.true.`, and `sbcmod.F90:484` calls `icb_stp` only under the same
flag.  Within that call, `icbthm.F90:286-295` supplies the only iceberg
thermodynamic ocean write-back: floating melt is subtracted from `emp` and
calving heat is added to `qns`.  `icbstp.F90:71-142` otherwise contains the
iceberg-cadence work that becomes inactive.  The resolved
`ln_rnf_icb=.false.` means the independent prescribed runoff-iceberg path at
`sbcrnf.F90:122-144` remains inactive in both decks.

Therefore the frozen comparison is:

- every state/operand before the first possible `icb_stp` effect must be
  bit-identical to the shipped-deck record;
- the variant has no iceberg contribution to `emp` or `qns`;
- the inactive-only diagnostic slots `utau_icb,vtau_icb` and the iceberg-grid
  slots are recorded as zero, with no model-array assignment; and
- no other physics field is expected to move.

The accepted post-`sbc` writer currently reads iceberg arrays unconditionally.
`icbini.F90:75` returns before allocating/initialising the iceberg component
when the flag is false.  The instrumented variant therefore requires a
WRITE-only safety correction: when inactive, the writer emits a local zero
scratch array for only those inactive diagnostic slots.  This changes no model
array or arithmetic and requires a replacement instrumented build.  The
uninstrumented control and 30-day run use the existing accepted scalar-math
uninstrumented binary.

## Acceptance after user execution

The variant remains unmeasured until all three runs finish.  The later gate
must require:

- launcher return zero, `RUN DONE`, expected terminal `time.step`, and no NEMO
  error for every run;
- a full 90-stream schema walk plus the post-`sbc` stream, including derived
  allocation-class header counts, monotone `kt`, exact sizes, finite payloads,
  and inactive iceberg slots exactly zero;
- byte identity of every ordinary output and all available restart shards in
  the instrumented and uninstrumented variant ten-step pair, excluding only
  the already registered timing/timestamp metadata;
- pre-`icb_stp` fields bit-identical between the shipped-deck and variant
  records, while post-`sbc` changes are restricted to the frozen field set;
- the existing nine Phase-1 plants and the Phase-2b post-`sbc` plants exit
  nonzero, plus a planted nonzero inactive-iceberg diagnostic exits nonzero;
  and
- every build, source, launcher, deck, input link, record, restart, and ordinary
  output is SHA-256 pinned.

No ocean stage is entered in this turn.

## Review blocker: Coriolis ownership

The current tripolar loader silently substitutes native NEMO F-point `ff_f`
for legoESM's v-face `f_v`.  This is forbidden because generic face-Coriolis
consumers use `f_v`; on the ORCA2 domain the two staggerings differ materially.

The preregistered repair restores the pre-change `f_v` construction exactly:
analytic T-point Coriolis from the configured rotation rate and stored T-point
latitude, then the historical adjacent-row average plus boundary copies.
Native NEMO `ff_f` is appended as a separate optional geometry field.  Only
the NEMO EEN/ENE vorticity arm may read it.  Generic `vertex_coriolis`, forward-
backward Coriolis, and every other existing consumer continue to read `f_v`.

Source ownership is explicit: `dynvor.F90:350-378` uses `ff_t` in the T-point
energy arm, while its ENE/ENS/EEN F-point arms use `ff_f`
(`dynvor.F90:449-490,584-613,752-782`).  Tests will pin the restored `f_v`
bytes on the ORCA2 domain and on the GYRE control, pin the separate `ff_f`
mapping, prove generic and NEMO consumers select different fields, and include
a planted swapped-field failure.  The Phase-2 ORCA2 entry gate must reproduce
its pre-repair JSON byte-for-byte; a change is a hard stop.

This repair changes shared geometry plumbing but not a shared numerical
operator.  Any later over-bar result in EOS/HPG, transports, ZDF, TKE, or
external mode will be registered at its boundary with a GYRE-owner label and
handed off rather than fixed in Lane 4.  SI3 remains
`UNMEASURED_PENDING_ICE_MERGE`; the icebergs-off producer is deliberately
inactive by Decision 7.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| comparison oracle sets only `ln_icebergs=.false.` | ASKED | one-variable variant; no shipped file edited |
| preserve shipped-deck records | ASKED | label `SHIPPED_DECK_RECORD`, superseded only for comparison |
| prepare instrumented 10-step, control 10-step, and 30-day runs together | ASKED | user-shell MPI execution; stop before measurement |
| post-`sbc` instrument included | ASKED | complete accepted instrument inventory retained |
| restore legacy `f_v`; add separate NEMO `ff_f` | ASKED review blocker | shared geometry repair with consumer-isolation tests |
| ORCA2-only ownership and GYRE handoff for shared debt | ASKED standing rule | no shared numerical debt fixed here |
| inactive iceberg slots emitted from writer-local zeros | UNASKED enabling correction | required because the component is not initialized; WRITE-only and planted |
| rebuild only the instrumented executable | UNASKED operational consequence | source changed only in config-local writer; uninstrumented binary retained |
| regular deck copies plus absolute immutable input/binary symlinks | UNASKED inherited packaging | disclosed and hash-guarded |
| any legoESM ocean-stage measurement | UNASKED and forbidden at this stop | none performed |

No unasked scientific configuration choice is made.

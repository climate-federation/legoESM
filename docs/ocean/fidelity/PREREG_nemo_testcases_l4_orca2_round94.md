# Preregistration — ORCA2 round 94 rung-0 slow-boundary walk

Date: 2026-10-02. Base: `cffea640d`. Every model number produced by this
round is labelled **independent**: the rung-0 state comes from NEMO's admitted
from-rest stage-0 frame, not from the shipped ORCA2+SI3 entry.

## Frozen measurement

Before this file was committed, only the round-93 checker, writer, compiled
call sites, and the first 160 bytes of the existing rank-0 record were read.
No field payload was decoded and no NEMO/legoESM boundary was compared. The
first field names itself `depth_u` and declares dimensions 90x148, while the
file header declares the rank-local domain 94x152. The current checker rejects
that self-description because it replaces the field dimensions with the domain
dimensions.

The round will produce:

1. A header-driven parser that reads all `nfields` field headers, derives each
   payload length from that field's recorded rank/dimensions, and only then
   verifies the exact required name set, uniqueness, finiteness, owned-domain
   coverage, and end of file. No field order, payload size, or header tuple is
   hand-predicted.
2. Admission of the existing two-rank record if and only if its self-described
   fields are complete and the 20 terminal restart shards remain byte-exact to
   the admitted round-92 parent. No NEMO rerun is requested for a sound record.
3. An independent production-JIT CPU/fp64 walk in compiled order: depth
   average, drag, wind/final momentum forcing, sea-surface RHS, and the
   split-explicit output. The first active non-bit boundary owns the next walk.
4. One-bit, duplicate/missing-name, dimension/payload, truncation, rank-layout,
   restart-byte, entry-bit, and trajectory-passivity controls. Every applicable
   plant must refuse before any model number is admitted.

NEMO writes the vector-invariant depth average at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219`, applies drag at
`:231-236`, wind at `:238-250`, forms the sea-surface RHS at `:290-312`, and
calls the split-explicit solver at `:317-326`. The recorder writes each field's
own name, rank, dimensions, and payload at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/l4_r93_slow_frames.f90:46-55`.

## Predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R94-P1 | The existing record is sound; the refusal is checker-only. | Both rank files parse to the exact required unique name set, consume their payloads exactly to EOF, cover the global owned domain once, and preserve all terminal restarts bit-for-bit. | Any malformed/missing/duplicate field, impossible dimensions, trailing/truncated byte, coverage defect, or restart movement: **REFUTED**; repair the writer and request a fresh named acquisition. |
| R94-P2 | The depth-averaged U/V forcing is bit-exact because its admitted 3-D RHS is exact and rung-0 geometry is shared. | Both active depth rows have zero unequal bits. | Either depth row is non-bit: **REFUTED**; stop there and walk thickness, mask, reciprocal depth, reduction order, and multiply in compiled order. |
| R94-P3 | Drag is the first non-bit boundary because the rung-0 measurement card declares NEMO's exact implicit linear-drag composition unbuilt. | Depth is exact and at least one post-drag active row is non-bit. | Drag is exact: **REFUTED**; continue through wind/final, sea-surface RHS, and split-explicit output. |
| R94-P4 | The measurement observer is passive. | Ordinary and observed final T/S/u/v/ssh arrays are bit-identical. | Any state bit moves: **REFUTED**; withhold all scientific rows. |

## Landing and refusal bar

This is a measurement/instrument-repair round. No model statement or card
selector lands unless the walk isolates one and its own bit-exact operand gate,
ORCA2 ladder gate, and all shared-card gates pass. The shipped ORCA2 card, its
sea-ice debt tuple, configuration choices, stabilizers, thresholds, and carried
state conventions remain unchanged. Failed predictions stay recorded as
**REFUTED**.

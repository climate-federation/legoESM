# Preregistration — round 187, developed shortwave statement walk

Committed before parsing or scoring the acquired Round-186 shortwave record.
The operator's run reached `STOP 0` and wrote a 1,174,060-byte record, but the
operator script refused it because it expected 1,175,916 bytes.  Round 187
first reconciles that difference from the compiled writer, admits the existing
record only if its passive restarts and replay controls pass, and then walks the
executed `qsr_2BD` statements at developed step 1080.  Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round187/`.

No model physics, configuration value, carried-state schema, default,
stabilizer, or NEMO source is authorized to change before the walk names a
source-exact one-variable candidate.  Any candidate remains subject to the
complete Decision-43/45/55/59 trajectory and executing-card gates.

## Compiled layout and order

The compiled stage writer emits a 16-byte magic, nine default integers, then
`gdepw_1d`, `qsr`, `r3t(:,:,Kmm)`, `e3t_3d`, `tmask`, `wmask`, the actual
increment, and the post-call replay.  Unlike the other arrays, compiled `qsr`
is allocated only over the no-halo domain: `Nis0:Nie0,Njs0:Nje0`, or 32 by 22
on this 36 by 26 grid.  The frozen layout is therefore

`16 + 9*4 + 8*(31 + 32*22 + 36*26 + 5*36*26*31) = 1,174,060 bytes`.

This is a parser/check defect, not permission to truncate a full-domain field.
The repaired reader must expose `qsr` on its recorded 32-by-22 bounds and map
it into the same interior indices used by the compiled loops.  Source citations
are registered only after the record is admitted and the exact compiled spans
are re-read.

## Frozen predictions and falsifiers

1. The size refusal is caused only by counting `qsr` as 36 by 26 instead of its
   compiled 32 by 22 allocation.  The corrected formula above must equal the
   existing file exactly.  Any residual byte, header, finite-value, or unread
   payload failure **REFUTES** this diagnosis and stops the walk.
2. The existing step-180 and step-1080 restarts are byte-identical to the
   uninstrumented Round-132/year record.  The post-call replay is bit-identical
   to the actual increment.  The actual-increment one-ULP plant prints
   `STATUS PLANT-FIRED` and exits nonzero.  Any failed passivity, replay, stamp,
   or plant row refuses the record; NEMO is not rebuilt or rerun.
3. The corrected synthetic-record test uses a 32-by-22 `qsr` payload and fails
   against the pre-fix parser.  A separate layout plant that substitutes a
   full 36-by-26 `qsr` payload must be refused rather than silently accepted.
4. Walk rows, in compiled order, are: the two surface coefficients, initial
   surface attenuation, live-depth exponential arguments and attenuation,
   live `ze3t`, absorbed flux/division, associated RHS update, and the
   near-surface/deep branch boundary.  Each row reports cells unequal and
   maximum absolute difference against the recorded NEMO replay, for isolated
   eager, isolated JIT, production eager, and production JIT wherever the
   existing production trace exposes the row.  Isolated rows are never called
   production rows.
5. Frozen attribution prediction: the shared legoESM kernel is not bit-exact at
   the first live attenuation expression under production JIT because its
   vectorized whole-column construction changes compiled association relative
   to NEMO's level-ordered loop.  This is **REFUTED** if all earlier rows and
   that attenuation row are bit-exact, or if an earlier source operand is
   already non-bit.  The first measured non-bit row wins and this prediction
   remains in the receipt.
6. A row exact only from NEMO operands but non-bit in the independent model
   trajectory is inherited.  A statement is owned only when it remains
   non-bit given the recorded NEMO operands through the production closure.
   If the existing trace cannot supply a production-closure row, the receipt
   labels it unmeasured rather than promoting an isolated result.
7. No landing occurs unless a one-variable transcription makes the owned row
   bit-exact in production JIT and passes the full ladder, day-30, day-240,
   day-360, DINO, generic-card, tank, ORCA2-with-spec, census, plant, citation,
   and review gates.  Otherwise the round is HELD with the named first row or
   STOPPED_FOR_RECORD with a fail-closed acquisition script.

The final diff receives a separate read-only Codex review.  A `DO NOT SHIP`
verdict blocks any landing.  No configuration decision is expected.

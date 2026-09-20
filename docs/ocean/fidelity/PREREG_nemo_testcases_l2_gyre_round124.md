# Preregistration — NEMO testcase L2 GYRE round 124 day-240 process ranking

Date: 2026-09-19

Incoming lane tip: `af3f7215060fc17c71adc6794817c710df8ee471`

This document is frozen before any Round-124 process-boundary measurement or
legoESM process trace.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round124/`.  The operator has
reported that the Round-123 NEMO acquisition completed, but that report is not
the science result: this round must independently admit the record, produce the
matching legoESM trace on its own trajectory, and close the day-180-to-day-240
temperature budget before ranking a process.  No physics, configuration,
restart schema, card default, or stabilizer changes in this round.

## P0 — admit the NEMO trace before scoring it

The only candidate oracle record is
`phase3/round123/oracle_process_budget`.  Its claimed producer is the compiled
configuration `GYRE_OMIP_L2_P3_SM_R123PROC`.  In that target's compiled stage
program, the writer is enabled only for stage 3 and steps 1081--1440, writes the
header, timestep, before tracer, and three free-surface ratios at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830`, then writes
the accumulated temperature RHS after advection and surface forcing at
`:861-869`, after shortwave and lateral diffusion at `:930-952`, and the solved
temperature immediately after vertical diffusion at `:964-970`.  These exact
compiled lines, not the acquisition patch or a source-tree analogue, define
the record.

Frozen admission prediction: the normal Round-123 reader reports exactly 360
records of 1,415,300 bytes, steps 1081--1440 in order, `rDt=14400`, all expected
processes active, zero chained-cell mismatch, and zero per-step decoded
rounding closure.  The recorded step-1080 and step-1440 restarts are predicted
byte-identical to the admitted year run with SHA-256 values
`6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976`
and `96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a`.
The `process-stamp`, `process-truncation`, `process-sbc-ulp`,
`process-sbc-effect`, and `process-trajectory-ulp` plants must each print
`STATUS PLANT-FIRED` and exit nonzero.  One failed hash, byte comparison,
layout check, activity check, closure check, or plant REFUTES admission and
stops the round without using the record.

## P1 — one production-JIT legoESM process trace

The existing `nemo_testcase_l2_gyre_year_owners.py` instrument will be extended;
there will not be a second owner or year harness.  From rest, seed/member zero
will advance the resolved `GYRE-zco` NEMO-identity recipe for 1,440 four-hour
steps on CPU/fp64 using `LatLonCGridOceanModel.step`, hence the production
`self._step_jitted` closure.  The existing year harness is read-only.  Surface
forcing and initialization are reused from its committed helpers without
changing their values.

Steps 1--1080 are ordinary production steps.  At steps 1081--1440 a private,
write-only diagnostic hook exposes the same stage-3 temperature boundaries as
the oracle record: before tracer and free-surface ratios; after-advection
content; after-surface-boundary content; after-shortwave content;
after-lateral-diffusion content; and after the production implicit vertical
solve.  The trace follows the already established live-operand pattern: the
diagnostic closure and the ordinary production closure both run from the same
entry state, while the state carried to the next step is only the ordinary
production closure's state.  Returning diagnostics therefore cannot become a
new trajectory writer.  The hook is private, has no recipe/configuration
selector, and is inert unless explicitly requested by the owner instrument.

Frozen trajectory predictions:

1. At every traced step, the carried state is bit-identical to a separately
   evaluated ordinary production step from the same entry.
2. The independently generated day-180 and day-240 states are bit-identical to
   the immutable GYRE year member under
   `phase3/year_equivalence/gyre/lego_seed0_year` (or the exact member root
   named by that run's manifest), and the day-240 wet-T3D RMS against NEMO is
   exactly `1.6446741930292448e-2 K`.
3. Every exposed active boundary moves at least one wet interior temperature
   cell during the interval.  The combined explicit boundary reconstructs the
   production pre-implicit content up to a separately reported rounding row;
   it is never silently forced to close.

Any changed carried bit, immutable-snapshot mismatch, headline mismatch,
inactive required boundary, missing record, dirty/incorrect producer commit,
or hidden closure residual REFUTES the process ranking.  An eager or isolated
JIT calculation cannot substitute for this production-step trace.

## P2 — non-vacuous production controls

The trace reader and producer must be fail-closed.  Its manifest records the
clean instrument commit, resolved recipe, fp64 policy, CPU platform, member,
step range, shapes, and file hashes.  A wrong producer commit, shortened trace,
wrong step sequence, changed shape, or altered trace byte must make scoring
exit nonzero.

The primary numerical plant is an effect-scale change to one wet surface-
boundary cell inside the full production diagnostic closure at one traced
step.  It must move the registered post-surface row and every downstream
explicit boundary by the analytically injected content, leave the upstream
geometry/advection rows unchanged, and make the unplanted interval comparison
fail.  The separately evaluated ordinary carried state must remain bit-exact,
which proves both that the plant is detected and that the write-only trace does
not own the trajectory.  A one-ULP trace-file plant must also make the reader
or closure gate print `STATUS PLANT-FIRED` and exit nonzero.  A plant that
merely changes an unused value or prints PASS while firing is a failed control.

## P3 — frozen process budget and ranking

The NEMO rows are decoded exactly as preregistered in Round 123.  The compiled
implicit solve forms its RHS from before thickness/tracer and middle
thickness/accumulated RHS at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-560`, then completes
the implicit column at `:563-578`.  For each model and each traced step, define
the temperature increments in this compiled order:

```
incoming             = T(step 1080)
geometry              = B0 - Tbb
advection             = Badv - B0
surface_boundary      = Bsbc - Badv
shortwave             = Bqsr - Bsbc
lateral_diffusion     = Bldf - Bqsr
vertical_diffusion    = Taa - Bldf
rounding_closure      = (Taa - Tbb) - sum(the six process increments)
```

Here the `B` fields are the thickness-normalized accumulated boundaries.  Each
model is decoded from its own freely evolving state; no NEMO operand is fed to
legoESM and no post-hoc state substitution is allowed.  Sum each increment for
steps 1081--1440, subtract NEMO from legoESM, and add the incoming day-180
error.  The resulting component arrays must reconstruct the independently
measured day-240 error array.  Closure is retained as a ranked row, never
distributed among processes.

For endpoint error `E240`, each row reports:

```
signed_carry_p = dot(C_p, E240) / dot(E240, E240) * RMS(E240)
component_rms_p = RMS(C_p)
```

over the existing owners instrument's wet T3D cells.  Rank by descending
`abs(signed_carry_p)`, while reporting the sign, component RMS, fraction of the
day-240 squared error, and cancellation.  Signed carries including incoming
and rounding closure must sum to the day-240 RMS within measured fp64
summation.  For each process, also report the largest birth sub-interval,
horizontal third, latitude band, and depth band using the frozen Round-122
partitions.  A day-30 process carry is `UNAVAILABLE-BY-RECORD` because the
admitted process trace begins at day 180; the measured day-30 endpoint RMS is
reported as context and is not invented as a process attribution.

No advance owner is guessed.  The largest owner is mechanically the admitted
row with greatest absolute signed carry, not the row with greatest local RMS
or best-looking spatial pattern.  The ranking is REFUTED if either trajectory
does not reproduce its endpoint, the component arrays do not reconstruct
`E240`, the signed projections do not reconstruct its RMS, or any control is
vacuous.

## P4 — verdict boundary and cross-card scope

This is the magnitude-ranking round required by the Round-123 OPEN section.
It lands only a diagnostic instrument, tests, citation mappings and receipt;
it does not land the measured process as physics in the same round.  The
largest admitted row becomes the next round's sole candidate.  If expressing
that candidate requires a configuration or carried-state choice, the receipt
must stop with `DECISION_NEEDED` and state one recommended choice.  Otherwise
the receipt specifies its compiled source statement and the future
Decision-43/45 ladder, month, full-year, DINO and recipe blast-radius gates.

LOCK_EXCHANGE, OVERFLOW, DINO and ORCA2 execute no new production statement in
this diagnostic round.  Their trajectories must therefore remain bit-exact;
the private trace hook is fail-closed off outside this instrument.  ORCA2 is
`UNMEASURED-WITH-SPEC`: a corresponding magnitude budget would require the
same native-card production trace, masks, component ordering and endpoint
closure.  Receipt citations must all map to the exact Round-123 compiled build;
the citation gate must pass and its shifted-citation plant must fail nonzero.
A separate read-only Codex review must try to refute the trace passivity,
process algebra, controls, ranking, and HELD/decision verdict.

# Preregistration: ORCA2 round 227 — OMT-4 implicit tracer solve

Date: 2026-10-10. Frozen base: `66af9c7e6`. Scope: continue round 226 in
compiled source order through OMT-4's stage-3 implicit tracer solve. This file
is committed before any round-227 replay or trajectory measurement.

No configuration, carried-state, stabiliser, threshold, sea-ice selector, or
`unmeasured_features` change is authorised. OMT-5 remains blocked. Replays are
CPU, production JIT, fp64/libm, and use only admitted passive stage states and
oracle records; no in-executable observer is permitted.

## Existing implementation and upstream boundary

Repository search found the production literal matrix and three ordered Thomas
recurrences in
`packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py`, the
passive production trace used by ORCA2 round 136, and the GYRE round-35 matrix
gate. Round 227 extends those readers/replays rather than writing a second
solver.

Upstream commit `5e368e87ba83eb508758b228bf92840aa9df90c1` adds a different
levels-first band-in-sweep implementation to the generic `shared_thomas` path.
The OMT-4 card resolves `zdf_implicit_solver_evaluation="nemo_literal"`, so
the upstream implementation is not presumed relevant. If measurement names a
statement that the upstream rewrite already transcribes, this round stops with
the requested merge-versus-in-lane decision instead of landing a duplicate.

## Compiled source order

The executed stage calls `tra_zdf` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:700-760`.
The resolved OMT-4 branches use `avt` for both tracers and no isoneutral,
double-diffusive, mass-flux, or adaptive-implicit addition. NEMO therefore:

1. copies `avt` and sets the surface interface to zero at
   `ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:171-216`;
2. builds lower, upper, and diagonal coefficients at `trazdf.f90:218-235`;
3. eliminates the diagonal at `trazdf.f90:249-273`;
4. forms thickness content and advances the forward RHS at
   `trazdf.f90:283-291`;
5. back-substitutes at `trazdf.f90:293-299`.

The already exonerated off-diagonal `e3w_1d` versus diagonal `e3t_3d`
association is frozen and will not be re-walked.

## Frozen predictions and falsifiers

1. **R227-P1 — admitted operands are sufficient.** CONFIRM: the OMT-4
   rank-0 `avt` record aligns unambiguously to one half of the admitted stage-3
   record, and recorded `Kbb`, `Kmm`, `Kaa`, `r3t`, and post-LDF `Krhs` provide
   every active operand of the compiled OMT-4 branch. REFUTE: rank ownership is
   ambiguous or any executed operand is absent; then request only that missing
   record.
2. **R227-P2 — source-order first boundary.** CONFIRM: substituting recorded
   `avt`, then the recorded thickness-content RHS, one variable at a time names
   the first matrix/content statement whose candidate output differs from the
   NEMO `Kaa` field. REFUTE: a later substitution closes a difference while an
   earlier recorded operand remains non-bit, or the replay cannot reproduce
   its own candidate solve.
3. **R227-P3 — ordered solve discrimination.** CONFIRM: given NEMO's recorded
   matrix inputs and content, the literal diagonal elimination, forward RHS,
   and reverse substitution either reproduce NEMO's `Kaa` bit-for-bit or name
   the first recurrence that does not. A one-bit planted recurrence operand
   must change the solved field. REFUTE: the replay differs before its stated
   boundary or the plant is inert.
4. **R227-P4 — upstream-overlap classification.** CONFIRM: the named first
   non-bit statement is classified mechanically as either outside upstream
   `5e368e87ba` or inside its changed generic band/sweep statements. If inside,
   status is `STOPPED_FOR_DECISION` with the prescribed choice; no duplicate
   transcription lands. REFUTE: the classifier cannot tie its verdict to both
   the selected OMT-4 dispatch and the upstream diff.
5. **R227-P5 — statement sufficiency.** If a source-literal statement outside
   the upstream rewrite is named, a private atomic candidate containing the
   complete rounds-215--226 unit plus this statement must remove or delay the
   independent OMT-4 kt=8 live-W-thickness refusal before any Decision-96
   census. An identical refusal refutes sufficiency and the candidate is
   retracted.
6. **R227-P6 — controls.** Plants for rank ownership, first-boundary order,
   recurrence liveness, upstream classification, and trajectory sufficiency
   must each refuse.

## Stop conditions

Missing executed operands produce `ACQUISITION_NEEDED`; no inferred values are
substituted. A first statement already covered by upstream `5e368e87ba`
produces `DECISION_NEEDED` with a pick between merging main and duplicating the
transcription. A statement outside that diff lands only after sufficiency and
the complete Decision-96 gate pass; otherwise the round is HELD with the exact
failed prediction retained.

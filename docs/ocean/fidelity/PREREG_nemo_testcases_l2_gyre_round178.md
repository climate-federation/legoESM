# Preregistration — round 178, developed tracer-LDF statement walk

Committed before admitting or interpreting the Round-177 record.  The operator
run completed NEMO and proved both step-1080 and step-1440 restarts
byte-identical, but the acquisition script refused its own header.  The
compiled writer emits `Kbb`, `Kmm`, and `Krhs` in that order and the acquired
header is `(1,1081,1,2,3,...)`; the parser incorrectly expected
`(1,1081,1,1,3,...)`.  This round first reconciles that contract from the
compiled source, admits the existing record without rebuilding NEMO, fires its
record-corruption plant, and only then performs the developed-state walk.

No production physics, configuration, carried state, restart schema, card
default, or immutable Round-163 trajectory changes unless the completed walk
names a one-variable source-exact candidate that passes the full trajectory
gate.  The diagnostic extends the existing year-owner harness; it does not
create a second developed-state bridge or walk.

## Compiled order and header correction

The acquired build calls `tra_ldf(kstp,Kbb,Kmm,ts,Krhs)` at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3_stg.f90:928-934`,
dispatches those same levels into `traldf_iso_lap` at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf.f90:105-118`, and writes
the literal header `1,kt,Kbb,Kmm,Krhs,...,38,11` at
`GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:403-419`.
Therefore the parser's fourth integer changes from `1` to `2`; no byte of the
record, writer, NEMO binary, or run directory changes.

The walk follows the compiled program: the A33 call at
`traldf_iso.f90:167`, masked gradients at `:215-250`, horizontal coefficients
and fluxes at `:272-299`, vertical coefficients and fluxes at `:311-344`, and
the final RHS additions at `:346-367`.  Every recorded row is classified in
that order under both the complete production JIT step and complete eager
execution from the admitted step-1080 NEMO state.

## Frozen predictions and falsifiers

1. With only the parser's `Kmm` expectation corrected to `2`,
   `--admit-existing` passes the 8,903,548-byte record, both restart comparisons,
   the 18,000-cell active census, and the Round-123 increment calibration.
   Any other correction, moved restart, failed calibration, or changed record
   byte REFUTES the contract diagnosis and stops interpretation.
2. `--plant-admission` corrupts one magic byte, prints
   `STATUS PLANT-FIRED: record-magic`, and exits nonzero.  Zero exit or a
   failure before the corrupted byte is checked invalidates the admission.
3. The ordinary production result and each diagnostic execution differ in
   zero returned-state bytes.  Any moved byte invalidates that execution mode.
4. `T(Kbb)`, masks, static diffusivities, and the masked tracer-gradient rows
   are bit-exact given NEMO's recorded entry.  Any unequal cell before the
   live-QCO horizontal coefficient is a REFUTED prediction and becomes the
   first owned/inherited boundary in the receipt.
5. The first non-bit direct operand is the horizontal live-QCO face thickness
   used by `zA11`/`zA22`: NEMO reads `e3u_3d*(1+r3u(Kmm)*umask)` and its v twin
   at `traldf_iso.f90:275-276`, while the instantiated GYRE card resolves the
   production transcription's `redi_flux_face_thickness_evaluation` to
   `tpoint_jacobian`.  The predicted statement rows are therefore `zA11` and
   `zA22`; a prior non-bit row, or bit-exact live-face operands, REFUTES this
   attribution.
6. Production JIT is authoritative.  Complete eager execution is predicted to
   name the same first non-bit operand/statement; disagreement retains both
   tables and adopts the JIT result.
7. A one-ULP perturbation to one active recorded live-face thickness must move
   its coefficient, horizontal flux, and final increment rows under production
   JIT, while leaving all preceding registered rows unchanged.  A blind plant
   or upstream movement invalidates the instrument.

If the first non-bit row is an inherited operand, Round 178 names it and stops
at that boundary.  If a single transcription statement is source-exact under
NEMO inputs and can be changed one-variable, it is eligible only after the
full Decision 43/45/55/59 month/year/card gate.  Failed predictions remain in
the receipt.

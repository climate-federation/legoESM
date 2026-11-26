# ORCA2 round 230 preregistration — tracer-consumer acquisition repair

Date: 2026-10-10. Frozen base: `ebdafe2ef`. Scope: repair only the failed
round-229 rank-complete post-`tra_adv_trp` operand acquisition. No model,
card, deck, carried-state, sea-ice, or scoring change is authorised.

## Existing evidence and source boundary

The operator-run round-229 launcher failed while compiling its additions-only
writer. The preprocessed call at
`ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:558` expands the
whole-array `e3t(:,:,:,Kmm)` macro into a product of 3-D `e3t_3d`/`tmask`
and 2-D `r3t(:,:,Kmm)`, and gfortran refuses the inconsistent ranks. The
scalar macro is defined at
`ORCA2_OMIP_L4_R229FOLDTRP/BLD/inc/domzgr_substitute.h90:126`; the backing
arrays are declared 3-D at
`ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/dom_oce.f90:170,180`.

The repair passes reference thickness, the selected 2-D free-surface ratio,
and the 3-D mask separately to the write-only module. The writer constructs
each vertical level with the same scalar order NEMO executes in the live FCT
consumer, exemplified at
`ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/traadv_fct.f90:538`:
`e3t_3d(i,j,k) * (1 + r3t(i,j,Kmm) * tmask(i,j,k))`.

## Frozen predictions and falsifiers

1. **R230-P1 — compile repair.** A fresh target build compiles the repaired
   writer and reaches the NEMO run. Any rank error, any other build error, or
   any change outside additions-only instrument sources REFUTES.
2. **R230-P2 — passivity.** Both rank-0 and rank-1 kt=10 terminal restarts are
   byte-identical to the admitted OMT-4 source run. One moved byte REFUTES and
   the record is not read.
3. **R230-P3 — record contract.** Exactly two self-describing stage-1 files
   contain, in order, `zFv_after_trp`, `T_Kmm`, `S_Kmm`, `e3t_Kmm`, and
   `tmask`, with header-derived shapes/payload lengths, finite values, and
   strictly positive wet live thickness. Missing ranks/fields, malformed
   payloads, a non-finite, or a non-positive wet thickness REFUTES.
4. **R230-P4 — controls.** Rank, field-name, and truncation plants must each
   refuse. A plant staying green REFUTES the acquisition.
5. **R230-P5 — attribution remains frozen.** No transport/tracer conclusion
   is allowed until R230-P1..P4 pass. If the operator cannot run the fresh
   target, the round stops `STOPPED_FOR_RECORD`; it does not reuse the failed
   build or infer a value.

## Disposition

The failed target is preserved. The recovery uses a new configuration and run
directory under round 230. A passing record resumes round 229's frozen P2/P3
test in the next round; this repair round lands no physics.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.

# ORCA2 round 328 — kt 1 stage-3 `dyn_zdf` walk stops for a missing record

Date: 2026-10-23 (Codex round)

Base: `735cab5ba22e`

Disposition: **ACQUISITION_NEEDED; no NEMO run, model, card, or configuration
change**

independent review: operator (Claude), **HOLD**; all seven confirmed findings
are addressed below for operator rereview

codex review: operator

## Answer

**CONFIRMED.** The first departing stage-3 statement cannot be named from the
admitted frames. The given-NEMO-entry state is exact through completed stage 2,
but the next admitted frame is the completed stage-3 state. There is no frame
at `dyn_zdf` entry or after any of its RHS, matrix, or solve statements. Naming
one of those statements, or a cell subset, would therefore be post-hoc
inference. Per the round contract, the walk stops at **ACQUISITION_NEEDED**.

The first *unmeasured candidate boundary* is NEMO's vector-form explicit
update, not a finding: NEMO enters stage 3 and calls `dyn_zdf` before tracer
transport at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:488-555`.
Inside that call it forms `Kaa = Kbb + rDt*Krhs`, subtracts the barotropic
velocity, then adds bottom and top drag RHS terms at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:116-181`.
It next constructs and solves the U matrix at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:187-349`
and the V matrix at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:356-517`.

Certified ladder coverage: **none** for this statement-level ORCA2 boundary.
The current rung-10 card carries the completed stage states, TKE/`avm`, the
Decision-150 drag, L3/L9, and Decision-154 `hm_i`; none certifies an interior
`dyn_zdf` row. The existing legoESM path does contain the literal bottom/top
drag bundle at `ocean_model_latlon_cgrid.py:14658-14700` and its U/V implicit
solve calls at `ocean_model_latlon_cgrid.py:14876-14885`, but comparing only
their final outputs cannot identify the first mismatching statement.

Cell subset: **UNMEASURED**. It may be under ice, coastal, fold/seam, or
depth-local, but this round makes no such claim.

## Acquisition written

The committed acquisition is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/`.
It re-runs the admitted round-326 two-rank trajectory with additions-only,
write-only hooks and records, in source order:

1. `before`, `Krhs`, `avm`, barotropic U/V, bottom/top `rCdU`, and stress;
2. explicit update, barotropic subtraction, bottom RHS, and top RHS;
3. U and V lower/diagonal/upper matrices after base ZDF, ZAD, LDF, bottom drag,
   and top drag; and
4. both implicit-solve outputs and the ordinary post-`dyn_zdf` barotropic-
   correction endpoint.

The writer never reads its payload into NEMO state. Admission requires both
ranks, exact headers/order/field-specific extents, finite fp64 payloads, active
`avm`, active bottom and top matrix changes, an active solve, a producer stamp,
and byte identity of every inherited oracle frame and terminal restart. Nine
planted violations (header, order, truncation, stamp, one-ULP `avm`, one-ULP
bottom, one-ULP top, one-ULP solve, endpoint) must refuse before the clean
record can pass.

The preflight pins the parent `dynzdf.F90`, the round-326 record build's own
`stprk3_stg.F90`, both input manifests, the real `ocean.output`, and the binary.
It verifies the D150 implicit-ice-drag, vector-form, split-explicit, and
`ln_zad_Aimp=F` selectors; proves both patches are additions-only; compiles the
writer and both patched units with the record build's Fortran toolchain;
enforces the 150-GB memory rule and a 900-second timeout; and defaults to
non-running mode. The operator alone may invoke `--run`.

## Repair of the operator HOLD

1. The target now copies and patches the round-326 build's 889-line
   `MY_SRC/stprk3_stg.F90`; it never substitutes pristine NEMO's 664-line unit.
   The same-program gate includes both `dynzdf` and `stprk3_stg` and proves that
   each differs only by its committed insertions.
2. Every non-observer object is compared through the imported round-326
   `objects_equal` implementation, so only its config-root-normalised ELF view
   is accepted; the implementation is not copied.
3. The reader accepts the writer's real field-specific layouts: state and
   matrix fields `(nx,ny,jpkm1)`, `avm` `(nx+1,ny+1,jpk)`, and drag coefficients
   `(nx+1,ny+1)`. A committed gfortran harness compiles and executes the actual
   writer, then the Python reader admits those exact stream bytes.
4. `wi` was removed from the record because the pinned real selector is
   `ln_zad_Aimp=F`; no unallocated `wi` actual argument is read.
5. Passivity compares deterministic `oracle_*.bin` frames and terminal ocean/
   ice restarts only. The read-only calibration finds all 140 inherited round-5
   files bit-identical in round 326. The eight grid-T/U/V/W diagnostics are not
   compared raw because their wall-clock `TimeStamp` attributes differ.
6. A4 now compares rank 0 with the admitted kt-1 stage-3 frame and compares
   both ranks with their admitted kt-2 entry endpoints. The admitted rank-0
   stage-3 and kt-2 entry U/V payloads have 0 unequal bits. A5 mutates exactly
   one payload word by one ULP in each of `avm`, bottom drag, top drag, and solve;
   a writer-emitted bit checksum makes every such plant refuse.
7. The launcher restores the round-326 `work_cfgs.txt` guard and failure cleanup
   for target/run directories created by the invocation. It pins both copied
   manifests; checks the real committed round-326 `ocean.output` excerpt with a
   refusing selector plant; and includes refusing continuation, namelist,
   reference-config, disk, memory, target-exists, and same-program plants.

The receipt's earlier UNASKED matrix-boundary instrument choice is now approved
by the operator ruling: the acquisition records every boundary the walk needs
and no `wi` or diagnostic-only channel beyond that walk.

## Fix disposition

No import or model change is proposed yet. **PLAUSIBLE:** the owner is one of
the recorded RHS/matrix/solve boundaries, but the acquisition is deliberately
designed to distinguish a wrong input coefficient from a missing or
mis-transcribed statement. If the acquired walk identifies one cited
statement already carried by a certified rung, the next round may propose its
minimal import; otherwise it proposes the smallest new cited transcription.
Decision 122 still forbids landing unless all seven pairs, both labels, with
Decision-150 drag strictly improve or become bit-identical.

## Validation and evidence

- Preregistration: `464c57a01`.
- Writer plus patched NEMO units: syntax proof passed with the round-326 build
  modules and compiler; the patch contains 18 write-only calls and no removed
  source line.
- HOLD-repair direct battery: 18 passed (compiled exact-layout writer, clean
  record, all nine plants, both-observer same-program plants, and the imported
  config-root-normalised ELF test).
- Clean-commit non-running preflight: PASS at `7c52fa74b`; all 8 preflight
  plants fired, 140 deterministic round-5/round-326 files matched, all 3
  patched/writer units passed syntax, and the same-program census found 412
  identical preprocessed units and 424 identical normalized objects. Log
  `gates/preflight_hold_repair.log` SHA-256
  `49e2cdfb722ddd3f411c1bbec3705151e659081538be23be438e92517359f737`.
- The earlier preflight hash and 157-test count are superseded by this HOLD
  repair. The repaired landing-gate count is recorded after it runs.
- Evidence root reserved at
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/`.
- No production result exists and no NEMO executable was run in this round.

Choices: stopping for the absent record is **ASKED**. Recording every
source-order matrix boundary, rather than only entry/exit, was an **UNASKED**
instrument choice and is now operator-approved; it is the minimum record that
separates input, drag-diagonal, and Thomas-solve ownership without an
executable legoESM observer.

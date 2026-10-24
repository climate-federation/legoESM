# ORCA2 round 328 — kt 1 stage-3 `dyn_zdf` walk stops for a missing record

Date: 2026-10-23 (Codex round)

Base: `735cab5ba22e`

Disposition: **STOP_PROPOSAL; acquired record admitted and replayed; no model,
card, or configuration change**

independent review: operator (Claude), Resume-4 **HOLD**; the wrong-field
comparison is **CONFIRMED** and corrected below

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

1. `before`, `Krhs`, `avm`, mid- and after-level `r3u`/`r3v`, barotropic U/V,
   bottom/top `rCdU`, and stress;
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

## Resume-2 repair of the confirmed signed-zero defect

The earlier A4 endpoint check compared all raw `corrected_u/v` bits against
canonical stage/entry frames. That was wrong on dry cells: the frame writer
initialises its canonical view to positive zero and only copies wet cells,
whereas `dyn_zdf` can retain negative zeros in its raw arrays. The operator
measured 178--642 such dry negative zeros per field and rank in the round-326
kt-10 restarts, so the prior gate would have refused a correct acquisition.

A4 now compares each endpoint only on the corresponding admitted frame's wet
support. The compiled writer fixture carries raw negative zero at dry U and V
cells while its canonical frames carry positive zero; the clean record passes.
The endpoint plant changes one nonzero frame cell by exactly one ULP and still
refuses, so the repair does not make the endpoint check vacuous.

An admission refusal after a completed `STOP 0` run now preserves the entire
run by moving it to a timestamped, clearly named directory below
`round328/acquisition/refused/`. A failed build or a run that crashes before
`STOP 0` still has its partial target and run directories deleted. Executable
tests exercise both cleanup branches and retain the completed run's record.

Argon operand ruling: `r3u(:,:,Kaa)` and `r3v(:,:,Kaa)` are now recorded at
`dyn_zdf` entry. They are required: in the compiled QCO program the U matrix's
after-level `e3u` divisor expands through `r3u(Kaa)` within
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:187-349`, and
the corresponding V divisor expands through `r3v(Kaa)` within
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:356-517`.
The exact-layout reader and fixture require both new 2-D fields.

Resume-3 closes the remaining base-matrix operand gap. The U lower/upper
coefficients also divide by `r3u(:,:,Kmm)` at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:193,195`, and
the V coefficients likewise divide by `r3v(:,:,Kmm)` at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:362,364`.
Both Kmm fields are now recorded next to the Kaa pair on both ranks; the
reader, exact extents, compiled layout fixture, and admission all require them.

The whitespace-tolerant real-log checker now also requires `nn_mxlice = 2`
and `jpl = 1`. Both verbatim lines are committed in the pinned round-326
`ocean.output` excerpt, and separate wrong-value plants prove each check fires.

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
- Per-receipt citation gate: PASS on the clean committed receipt, 6 citations
  found and 0 failures.
- Repaired default landing gate plus private-import ratchet: 147 passed in
  1,035.88 s (receipt citation defaults, private imports, TKE, recipe,
  freshwater, stage-face mask, prognostic barotropic state, and ORCA2 parallel
  receipt). The earlier preflight hash and 157-test count are superseded.
- Evidence root reserved at
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/`.
- No production result exists and no NEMO executable was run in this round.
- Resume-2 focused battery: 22 passed (the 21 round-328 fixture/admission/
  cleanup tests plus round 326's config-root-normalised ELF calibration).
- Resume-2 clean-commit non-running preflight: `PREFLIGHT_READY` at
  `5941c6635fc8`; all 10 preflight plants fired, all 140 inherited outputs
  matched, all three patched/writer Fortran units compiled, and the
  same-program census remained 412 preprocessed units / 424 normalized
  objects.
- Resume-2 per-receipt citation gate: PASS, 6 citations found and 0 failures.
- Resume-2 default landing battery and private-import ratchet: 147 passed in
  1,401.47 s; the ratchet's 6 tests are included in that count.
- Resume-3 focused battery: 22 passed (the 21 round-328 fixture/admission/
  cleanup tests plus round 326's config-root-normalised ELF calibration).
- Resume-3 clean-commit non-running preflight: `PREFLIGHT_READY` at
  `f77ae176f7d3`; all 10 preflight plants fired, all 140 inherited outputs
  matched, all three patched/writer Fortran units compiled, and the
  same-program census remained 412 preprocessed units / 424 normalized
  objects. The first attempt refused before compilation on a stale unified-
  diff hunk count; `f77ae176f7d3` contains the corrected patch header.
- Resume-3 per-receipt citation gate: PASS, 8 citations found and 0 failures.
- Resume-3 default landing battery and private-import ratchet: 147 passed in
  990.27 s; the ratchet's 6 tests are included in that count.

Choices: stopping for the absent record is **ASKED**. Recording every
source-order matrix boundary, rather than only entry/exit, was an **UNASKED**
instrument choice and is now operator-approved; it is the minimum record that
separates input, drag-diagonal, and Thomas-solve ownership without an
executable legoESM observer.

## Resume-4: corrected consumed-profile score and term localization

### Retraction and preserved replay result

**CONFIRMED.** The previous Resume-4 `avm` departure is retracted. Commit
`228faed00` compared `state.tke_avm`, the closure's private `avm_k`, with
NEMO's final `avm`. NEMO first copies `avm_k`/`avt_k`, then applies double
diffusion and internal-wave mixing at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:349-350`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:363`, and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:372`. legoESM likewise keeps the
carried TKE pair separate and composes DDM then IWM in `k_profiles.py:484-578`.
The old all-414,119 score, its `[1,49,0]` first cell, and the claim that TKE
alone owned the difference were comparisons of unlike fields.

**CONFIRMED.** The record-fed `dyn_zdf` replay remains sound: both ranks are
bit-identical at the vector explicit update, barotropic subtraction, bottom
and top RHS, base matrix, drag diagonals, wind, and ordered solve. All ten
state/solve rows have zero unequal bits on each rank. Replaying NEMO's `avm`
through legoESM's face interpolation and literal matrix arithmetic is also
exact for U and V on both ranks. Timing is not an owner: NEMO calls `zdf_phy`
once on Nbb before the RK stages, and the card uses step-entry shear/N2.

### Correct consumed coefficients

**CONFIRMED.** The corrected replay starts from the ordinary step's carried
TKE `avm_k`/`avt_k`, applies legoESM's certified literal DDM statement
(`double_diffusion.py:335-362`), then its certified literal IWM statement
(`internal_wave_mixing.py:435-486`). The resulting consumed `avm` is
bit-identical at 408,595 of 414,119 wet interfaces and differs at 5,524
(1.334%). The first unequal cell is `[j,i,k]=[1,119,19]`; 666 are under ice
and 4,858 are open water. The maximum absolute difference is
0.012502789085852083 m2/s at `[81,25,28]`.

The final rank-0 tracer records are also scored: `avt` differs at 3,389 of
224,547 wet interfaces and `avs` at 3,388 of 224,547. The recorded TKE energy
`en` differs at 3,248 of 224,547. These are final-field scores, not ownership
boundaries.

Using the consumed profile leaves 10,093 unequal first base-lower matrix
coefficients, not 796,409: U has 2,900 rank-0 plus 2,106 rank-1 inequalities;
V has 2,941 plus 2,146. The first U inequality is global `[5,54,20]`; the
first V inequality is the same. NEMO's first U statement remains
`ORCA2_ORCA1ICE_OMIP_L4_R328ZDF/BLD/ppsrc/nemo/dynzdf.f90:206-210`, but the
record-fed exact replay proves that statement consumes the departure rather
than creates it.

### Owning-term split and bounded stop

**CONFIRMED.** Of the 5,524 final-`avm` residuals, 4,596 are at zero-based
interface 28 and 282 are at interface 0. At interface 28 the card is larger
at 4,585/4,596 cells (99.76%); at interface 0 it is larger at 279/282. The
literal model split changes all 5,524 residual cells in IWM and 5,185 in DDM;
at interface 28 those counts are 4,596 and 4,564, and at interface 0 they are
282 and 273. Neither the post-IWM-minus-final nor post-DDM-minus-final test is
zero, so final fields cannot select one term.

**PLAUSIBLE, not an ownership finding.** For the deepest-interface family the
first candidates are TKE's bottom boundary/mixing-length route—NEMO's bottom
energy assignment at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:285-293` and
final coefficient construction at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:706-711`, corresponding to legoESM's carried closure and final
K construction at `tke.py:3574-3664`—or IWM's depth distribution. For the
interface-0 family the first IWM candidate is the column-depth normalization
`ht_0*(1+r3t)` followed by the `gdept_1d` ladder at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:234-275`; legoESM transcribes those
operations at `internal_wave_mixing.py:435-471`. DDM cannot be excluded in
either family because its NEMO-side boundary was not recorded; its momentum
addition is `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:172` and legoESM's is
`double_diffusion.py:362`.

Certified coverage is **V1** for TKE, **V3** for literal double diffusion, and
**V4** for literal internal-wave mixing. Round 313 imported V3 and V4 onto
rung 10. Those certifications justify the offline split, but they do not turn
legoESM-only intermediate values into NEMO observations.

The admitted NEMO record contains final `avm`/`avt`/`avs` but no post-TKE or
post-DDM coefficient boundary. Therefore the first owning statement for the
interface-28 and interface-0 families is **UNMEASURED**. The bounded acquisition
request is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_zdf_terms_acquisition/README.md`:
record post-TKE, post-DDM, and post-IWM `avm`/`avt`/`avs` on both ranks, plus
the listed TKE/DDM/IWM operands and one-ULP refusal plants. No NEMO executable
was run and no model or configuration changed.

### Evidence and disposition

Resume-4 evidence is
`orca2_rounds/round328/replay/dynzdf_replay_consumed_resume4.json`, SHA-256
`f449275ba9077cb42885590a4f709e34c189afbe8eefa92772808465321226e4`.
The clean stamped replay commit is `3e2e5c48974b07a4f65e5ac211c0a19472f92edf`.
The operator's independent Claude review is **HOLD**; its wrong-field finding,
5,524 count, level-28 count, level-0 count, and timing ruling are reproduced.
No Decision-122 arm or model proposal is opened because ownership remains
unmeasured.

Resume-4 gates: per-receipt citation gate PASS (14 citations, 0 unmapped, 0
failures, 0 map-audit failures); default landing battery 169 passed in
1,000.32 s; explicit private-import ratchet 6 passed in 4.80 s. The default
battery includes the expanded 22-test round-328 gate.

## Resume-5: operator-ready ZDF term-boundary acquisition

The bounded request is now an executable operator acquisition, not a README
description. Its thin `run.sh` selects a `zdfphy` target in the existing
round-328 `dyn_zdf` acquisition, importing that tool's configuration-root-
normalised same-program proof and retaining its record-build `MY_SRC`,
including `stprk3_stg`. The default mode is read-only; only an operator may
invoke `--run` or `--admit-existing`.

The additions-only observers match the compiled record build. Boundary 1 is
after the TKE call at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:336` and coefficient
copies at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:348-351`.
Boundary 2 is after the complete DDM conditional at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:362-368`.
Boundary 3 is after IWM and before `zdf_mxl_turb` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:370-374`; it is also
before the later lateral boundary exchange at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:380`.

The actual writer supplies the committed gfortran layout fixture. Admission
requires both ranks; exact header, field order, extents, checksums, and finite
fp64 payloads; the producer commit; unchanged inherited deterministic files;
byte-identical re-admission of the existing `dyn_zdf` record; and wet-cell bit
identity of final `avm`, `avt`, and `avs` against the already admitted record.
Header, field-order, truncation, producer, and one-ULP plants at all three
boundaries must refuse. Dry signed-zero differences are accepted, while the
fixture proves a wet one-ULP change refuses.

The direct fixture/same-program battery passes 38 tests. The final validation
below is performed from a clean committed tree. No NEMO executable was run,
and no model or configuration changed.

## Resume-6: rank-local admission and semantic refusal plants

The independent Claude and Argon reviews both returned **HOLD** on the same
rank-reference blocker. Both reviews otherwise confirmed observer placement,
allocated-array use, same-program proof, checklist coverage, and containment.

**CONFIRMED and corrected.** `oracle_zdf_entry_kt00000001.bin` is rank-0-only:
the record routine returns when `.NOT.lwp` and then writes the canonical final
ZDF fields at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/MY_SRC/stprk3.F90:411-422`.
The repaired gate therefore scores rank-0 final `avm`/`avt`/`avs` against that
frame, but scores rank-1 final `avm` against rank 1's already admitted
round-328 `dyn_zdf` `avm_input`. Rank-1 `avt` and `avs` have **no admitted
reference frame** and are explicitly excluded from final-field scoring. The
compiled fixture now writes different values on the two ranks and proves the
rank-1 ZDF `avm` matches only its rank-1 `dyn_zdf` input.

The checksum check now runs before any plant mutation. Boundary 1's one-ULP
plant reaches the recorded `b1_avm + ddm_max_inc = b2_avm` chain; boundary 2's
plant reaches `b2_avt + iwm_zav_wave = b3_avt`; boundary 3's plant reaches the
rank-0 admitted-final bit comparison. The three refusal reasons are therefore
semantic comparisons, not writer-checksum failures.

Boundary-1 `avs` is the preceding step's value: NEMO copies only `avt_k` and
`avm_k` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:348-351`,
then sets `avs` in the later DDM conditional at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:362-368`.
The field is now named `b1_avs_previous` and excluded from every score; a test
changes it arbitrarily while the source-order checks remain green.

The ZDF Python gate imports the existing `dyn_zdf` checksum verifier. Its
Fortran writer imports the existing checksum functions, and its compiled
fixture reuses the existing round-328 fixture stubs; the three duplicate
implementations were removed.

Validation: the clean code commit `9ef69ebe5` reports
`ORCA2_ROUND328_ZDFPHY_PREFLIGHT_READY`; 40 focused tests pass, all 10 shell
preflight plants fire, all 140 inherited deterministic files match, seven
Fortran units pass syntax, and same-program reports 409 identical preprocessed
units plus 421 identical normalized objects. The final receipt citation gate
passes 28 citations with zero failures/unmapped/map-audit failures; the default
landing battery passes 147 tests; the explicit private-import ratchet passes 6
tests. No NEMO executable was run.

Choices: none. All changes are the two reviewers' requested admission and
non-vacuity repairs; no model, card, namelist, or configuration value changed.

## Resume-8: retained-run admission and TKE term ownership

### Refused transport frame diagnosis

**CONFIRMED.** The sole inherited-output difference is not a production
change. `oracle_transport_kt00000001_s2.bin` is a 48-byte header followed by
three native-fp64 arrays of shape `(94,152,31)`: `zFu`, then `zFv`, then
`zFw`. Each array occupies 3,543,424 bytes, so `zFw` begins at one-based file
byte 7,086,897. The first changed byte, 7,089,561, is 2,664 bytes into that
record: zero-based linear element 333, or Fortran `(i,j,k)=(52,4,1)`. Its new
value is `-624.7724791240518`, versus zero in rounds 5 and 326 and in the
round-328 `dyn_zdf` record.

**CONFIRMED.** The complete new `zFw` payload has 442,928 finite values,
233,341 nonzero values, no NaN or infinity, minimum
`-5253981.606405961`, maximum `7254087.19821906`, and smallest nonzero
magnitude `0.01993049235501692`. These are finite transport-scale numbers,
not NaNs. **PLAUSIBLE only:** their magnitudes look numerically reasonable;
they have no semantic meaning in this frame because the array is undefined.

**CONFIRMED.** The writer allocates all three arrays at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:121`, but in
the active vector-invariant branch it computes `ww`, explicitly says `zFw`
is instead computed in `tra_adv_trp`, and then writes the still-unassigned
`zFw` as the third record at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:323-351`.
The consumer selects vertical-transport construction for this same vector
branch at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traadv.f90:196` and overwrites
its `pFw` from a newly computed `ww` before use at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traadv.f90:296-312`.
Therefore the observed third record is uninitialized allocation content, not
an operand consumed by production.

**CONFIRMED.** `zFu` and `zFv` are bit-identical. Across the 160 shared
deterministic oracle/restart outputs, this undefined third record is the only
difference; the stage-2 and stage-3 production frames, stage-2 tracer and RK
operands, both-rank ocean and ice restarts, and both-rank `dyn_zdf` records are
exact. That independently excludes an observer-induced production-state
change.

### Fail-closed re-admission

**CONFIRMED.** Transport passivity now checks the exact schema and size, then
compares the defined 48-byte header plus `zFu` plus `zFv` (7,086,896 bytes)
bit for bit. It deliberately ignores only the writer-proven undefined `zFw`
payload. A third-record mutation is accepted, while a `zFu` mutation refuses;
every other inherited deterministic output remains a whole-file exact
comparison. The retained-run path is pinned to producer commit
`30cb3f2a81866381cbca2a0d8b36d01ac1794d4a` and its recorded binary digest.

**CONFIRMED.** `--admit-existing` admitted the retained output without a build
or NEMO execution: 41 preflight tests passed, all 10 refusal plants fired,
same-program found 421 identical normalized objects and 409 identical
preprocessed sources, deterministic passivity passed for 160 files, and both
the inherited `dyn_zdf` and new `zdfphy` admissions passed. The accepted
record is
`orca2_rounds/round328/acquisition/orca1ice_zdfphy_kt1_np2`.

### First unequal term boundary

**CONFIRMED.** The clean offline replay is
`orca2_rounds/round328/replay/zdf_term_localization_resume8.json`, SHA-256
`61ca3e60837439291357607258a228819d283d0d10c9e10e98796e2d1f730ed1`.
For both requested zero-based families, `k=28` and `k=0`, the first recorded
unequal coefficient boundary is post-TKE, before DDM and IWM. Post-TKE has
5,530 unequal interfaces: 4,596 at `k=28` and 284 at `k=0`. Post-DDM has the
same counts; post-IWM has 5,524 total, 4,596 at `k=28`, and 282 at `k=0`.

**CONFIRMED.** The first owning source-order statement is therefore the TKE
call at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:336` for both
families; the later coefficient copy, DDM, and IWM statements cannot be their
first owner. TKE entry `rn2` is bit-identical at all 414,119 scored interfaces,
while returned `en` is already unequal at 5,661 interfaces (4,757 at `k=28`,
279 at `k=0`). NEMO then derives `p_avm` from that `en` at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:709-711`.

**PLAUSIBLE, not claimed as measured.** The energy solve or its boundary
conditions are the next inner-TKE candidates. This acquisition has no
inside-`zdf_tke` statement boundary, so it does not identify a narrower inner
statement than the confirmed call owner. No model or configuration changed,
and no NEMO executable was run during this resume.

Validation: the focused acquisition battery passes 41 tests; the per-receipt
citation gate passes 33 citations with zero failures, unmapped citations, or
map-audit failures; and the default landing battery passes 147 tests in
969.52 seconds, including the six-test private-import ratchet.

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/orca2_r328/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED_PENDING_OPERATOR_REVIEW — retained record admitted; k=28 and k=0 first diverge in TKE

## Operator amendments after independent review (Claude LAND, Argon LAND)

- **Owner wording narrowed.** "The first owning source-order statement is the TKE call" overclaims: the earlier unrecorded
  inputs zdf_sh2 (record build zdfphy.f90:318), zdf_drg (:327-328) and zdf_mxl (:330) are not excluded by this record.
  CONFIRMED: post-TKE is the first unequal recorded boundary at all 29 levels; TKE entry rn2 is bit-identical at all
  414,119 interfaces; returned en differs at 5,661 (all 4,757 wet k = 28 interfaces, 279 at k = 0). PLAUSIBLE: the owner
  is inside zdf_tke or one of those three unrecorded inputs.
- **Next target (Claude review, PLAUSIBLE).** The bottom TKE value written at the end of the array, en(mbkt+1 = jpk)
  (zdftke.f90:295), feeding the last solve row: it hits all 4,757 full-depth columns but only 6 of 2,760 columns whose
  bottom is one level higher, so the bottom-friction formula itself is not the cause. Argon names the same region
  (top/bottom en boundary conditions, zdftke.f90:272-310).
- Both reviewers confirmed independently: zFw is never assigned in the active vector-invariant branch before the
  writer (record stprk3_stg.f90:121,323-351; tra_adv_trp overwrites pFw, traadv.f90:196,311); 159 of 160 inherited
  files exact and transport_s2 exact over header + zFu + zFv; a zFu flip refuses, a zFw flip passes; re-admission
  pinned to producer 30cb3f2a8 and its binary digest; no packages/ change.

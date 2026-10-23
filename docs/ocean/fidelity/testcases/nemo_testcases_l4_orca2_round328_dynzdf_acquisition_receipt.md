# ORCA2 round 328 — kt 1 stage-3 `dyn_zdf` walk stops for a missing record

Date: 2026-10-23 (Codex round)

Base: `735cab5ba22e`

Disposition: **ACQUISITION_NEEDED; no NEMO run, model, card, or configuration
change**

Independent review: operator (Claude).

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
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:187-346`
and the V matrix at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:356-520`.

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

1. `before`, `Krhs`, `avm`, `wi`, barotropic U/V, bottom/top `rCdU`, and stress;
2. explicit update, barotropic subtraction, bottom RHS, and top RHS;
3. U and V lower/diagonal/upper matrices after base ZDF, ZAD, LDF, bottom drag,
   and top drag; and
4. both implicit-solve outputs.

The writer never reads its payload into NEMO state. Admission requires both
ranks, exact headers/order/extents, finite fp64 payloads, active `avm`, active
bottom and top matrix changes, an active solve, a producer stamp, and byte
identity of every inherited oracle frame, restart, and diagnostic NetCDF.
Eight planted violations (header, order, truncation, stamp, `avm`, bottom,
top, solve) must refuse before the clean record can pass.

The preflight pins the parent `dynzdf.F90` and binary, verifies the D150
implicit-ice-drag, vector-form and split-explicit selectors, proves the patch
is additions-only, compiles the writer and patched unit with the record build's
Fortran toolchain, enforces the 150-GB memory rule and a 900-second timeout,
and defaults to non-running mode. The operator alone may invoke `--run`.

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
- Acquisition gate test: 9 passed (clean record plus all eight plants).
- Evidence root reserved at
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/`.
- No production result exists and no NEMO executable was run in this round.

Choices: stopping for the absent record is **ASKED**. Recording every
source-order matrix boundary, rather than only entry/exit, is an **UNASKED**
instrument choice offered for operator revision; it is the minimum record that
separates input, drag-diagonal, and Thomas-solve ownership without an
executable legoESM observer.

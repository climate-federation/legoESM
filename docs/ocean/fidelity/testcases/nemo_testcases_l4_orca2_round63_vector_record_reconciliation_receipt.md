# ORCA2 round 63 receipt — vector-record reconciliation and KEG walk

Date: 2026-09-28

Base: `990a34547bc1ca1f074a580f5ff9801324400a0d`

Disposition: **STOPPED_FOR_RECORD; no model statement lands**

Claim label: **given NEMO's recorded operands**

Sea ice, its six selectors, and the ORCA2 card's `unmeasured_features` tuple
remain frozen.  This round changes no file under `packages/`, no card selector,
no initial state, and no NEMO arithmetic statement.

## Answer

KEG is not ORCA2's first non-bit vector-advection statement.  A literal
source-order replay is bit-identical on every active owned face across the two
MPI ranks: U `0 / 413030` unequal and V `0 / 415175` unequal, with maximum
absolute difference `0.0` for both.  A one-ULP plant changes one active U cell
per rank and the gate reports `2 / 413030`, proving the comparison fires.

The first unresolved statement is now ZAD.  It cannot be scored from the
round-62 record because the record contains headers but no payload bytes for
both live QCO thickness operands.  The checker recovers all nine fields after
the holes and reaches physical EOF on both ranks, so this is a writer defect,
not a filename, rank-census, or generic payload-length error.  Missing carried
state is not reconstructed.

The acquisition-only repair writes NEMO's actual QCO live U/V thickness
expressions through an allocated three-dimensional payload.  Its clean-tree
preflight applies with zero fuzz, proves Fortran syntax, and names the fresh
target `ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP`.  The operator must run that launcher
before ZAD can be walked.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R63-P1 physical stream reconciliation | **CONFIRMED** | Both ranks have intact fields through `wsd_effective`, header-only `e3u_Kmm` and `e3v_Kmm`, all nine later fields, and physical EOF. |
| R63-P2 cause | **CONFIRMED** | The compiled card is QCO, not ALE; it allocates `e3u_3d`, `e3v_3d`, `r3u`, and `r3v`, while the failed writer reads the unallocated ALE arrays `e3u` and `e3v`. |
| R63-P3 existing-record admission | **CONFIRMED** | The diagnostic is fail-closed and does not invent either missing carried operand. |
| R63-P4 acquisition repair | **CONFIRMED preflight; run pending** | The fresh-target launcher passes its committed-dependency, additions-only patch, and Fortran-syntax checks. |
| R63-P5 KEG then ZAD | **KEG CONFIRMED bit-exact; ZAD UNMEASURED_WITH_SPEC** | KEG is `0 / 413030` U and `0 / 415175` V unequal.  ZAD requires the two absent operands. |
| R63-P6 landing | **CONFIRMED** | No package byte changed; disposition is STOPPED_FOR_RECORD. |

## Record and controls

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round63/`.
The header-driven diagnosis records the two immutable rank digests, the exact
missing-payload signature, nine recovered suffix fields, and physical EOF.
Its offset and signature plants both refuse.  The KEG gate records per-rank
and total active-cell counts; its one-ULP plant fires.

The repaired launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round62_vector_advection_acquisition/run.sh`.
It retains header-derived admission, restart identity, calibration, physical
EOF, and all acquisition plants.  No NEMO run was attempted in the sandbox.

## Shared-card gates

Round 61's resolved-config census remains the routing gate: ORCA2 and GYRE
execute vector-invariant momentum advection; OVERFLOW and LOCK_EXCHANGE
execute flux-form UP3.  This round changes only committed measurement and
acquisition files.  Since `packages/` is byte-identical to the base, no model
card trajectory can move and the GYRE/DINO/tank trajectory gates are not
applicable to this stopped record round.

## Review and validation

The required separate review was attempted with `codex exec --sandbox
read-only`.  It could not initialize its in-process app-server client because
the read-only sandbox rejected a filesystem write.  Verdict: **independent
review unavailable in-sandbox**.

Validation results are recorded after the source citations below.

## Compiled-source reconciliation

The failed compiled writer emitted the two live-thickness headers and then
read the ALE arrays at
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dynadv_round62_writer.f90:91-92`.
This card instead compiles `lk_qco=.TRUE.`, `lk_ALE=.FALSE.`, and the
one-dimensional-plus-three-dimensional vertical mesh at
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dom_oce.f90:136-140`.
The allocation branch gives QCO `e3u_3d/e3v_3d` and `r3u/r3v`, while only the
dead ALE branch allocates `e3u/e3v`, at
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dom_oce.f90:318-367`.

NEMO's executing ZAD statement consumes live U/V thickness as
`e3u_3d*(1+r3u*umask)` and `e3v_3d*(1+r3v*vmask)` at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzad.f90:102-137`.
The acquisition repair transcribes those two expressions into an allocated
payload at
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round62_vector_advection_acquisition/dynadv_round62_writer.F90:79-89`.
It does not alter the compiled model arithmetic.

The KEG replay follows the executing C2 loop, including statement order, at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynkeg.f90:117-130`.
Its bit-exact result discharges KEG and advances the first-non-bit walk to ZAD,
without attributing ZAD before its required carried operands exist.

## Validation readout

- Citation gate: pending final run.
- Focused tests: pending final run.
- Full card battery: pending final run.
- Single ocean-fidelity battery: pending final run.

## OPEN

1. Operator: run the committed acquisition launcher with `--run`; do not
   reuse or overwrite the failed round-62 target.
2. Next round: admit the fresh per-rank streams and restart calibration, then
   replay ZAD in compiled source order and name its first non-bit statement.
3. After the step-level owner is closed, continue to the QCO mixed boundary,
   Decision 52's independent-start ladder, and the ORCA2 month-scale magnitude
   ranking in the standing order.

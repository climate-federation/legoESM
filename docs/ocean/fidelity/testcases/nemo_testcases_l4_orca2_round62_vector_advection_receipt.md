# ORCA2 round 62 receipt — vector-invariant KEG/ZAD boundary

Date: 2026-09-28

Base: `1efb941c314cba10f7b5b85034faebec1de27a8b`

Disposition: **STOPPED_FOR_RECORD; no model statement lands**

Claim label: **given NEMO's recorded operands**

Sea ice and the ORCA2 card's six `unmeasured_features` remain frozen.  This
round changes no file under `packages/`, no card selector, no initial state,
and no NEMO arithmetic statement.

## Answer

The round-56 `651 / 16,900` U-cell movement is not an ORCA2 boundary.  The
four-card execution census has zero disagreements: ORCA2-zps and GYRE-zco
execute vector-invariant momentum advection; OVERFLOW-zps and LOCK-zco execute
flux-form UP3.  ORCA2's compiled stage calls HPG, VOR, and then `dyn_adv` at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:402-420`.
The vector dispatcher then calls KEG before ZAD at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynadv.f90:134-138`.

The admitted rank-0 cumulative frame proves that this combined boundary is
non-bit.  Across active wet faces, 226,236 / 226,236 U cells and 226,637 /
226,637 V cells move from after-VOR to after-ADV.  Maximum absolute changes are
`2.459677580345303e-06` U and `5.397315199560199e-06` V; RMS changes are
`1.1078041055119483e-08` U and `1.800292619852937e-08` V, in the recorded
momentum-tendency units.

That record contains neither the intermediate after-KEG accumulator nor
`ww`, so it cannot say whether KEG, ZAD, or both are the first non-bit
statement.  The first unresolved compiled statement is therefore KEG, whose
C2 arithmetic is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynkeg.f90:117-130`;
the second is ZAD, whose executed transport and bottom update are
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzad.f90:102-137`.
Calling either one the owner from the combined endpoint would violate the
first-non-bit rule.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R62-P1 card execution census | **CONFIRMED** | 0 disagreements; ORCA2/GYRE vector-invariant, OVERFLOW/LOCK flux-form UP3. |
| R62-P2 existing record admission | **CONFIRMED** | Magic `NEMO_L2_RKTRM_1`, header `(1,1,2,3,2,94,152,31,64)`, eight arrays, physical EOF, finite payload, frozen SHA-256 `18a7b4701f167c5fd17f93875c318eb21ea30ce79c523b269ac6d8abd2ebd35d`. |
| R62-P3 ORCA2 combined movement | **CONFIRMED** | Both active face families move.  The imported OVERFLOW shape (`651` U, `0` V) is **REFUTED** by `226236` U and `226637` V. |
| R62-P4 literal KEG versus exposed production KEG | **UNMEASURED_WITH_SPEC** | The old root-only stream does not carry the rank-1 Kmm field or the live exchanged MPI halos needed to drive the production ORCA2 operator without reconstructing carried state.  The frozen zero-unequal prediction is retained, not scored post hoc. |
| R62-P5 ZAD / closure | **CONFIRMED missing; UNMEASURED_WITH_SPEC scientifically** | `after_keg` absent and `ww` absent.  The committed acquisition carries both per rank. |
| R62-P6 landing | **CONFIRMED** | No package byte changed; disposition is STOPPED_FOR_RECORD. |

## Record and controls

The committed gate
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round62_vector_advection_gate.py`
parses the old stream, verifies its registered digest and compiled call order,
joins the record to the rank-0 U/V masks, and emits the two movement rows.  Its
digest, field-order, payload-ULP, and producer-stamp plants all exit nonzero
with `STATUS PLANT-FIRED`.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round62/`:

- `vector_advection_boundary.json` and `.log`: admitted combined boundary.
- `vector_advection_card_scope.log`: four-card execution census.
- `vector_advection_{digest,field-order,payload-ulp,stamp}_plant.log`: four
  firing controls.
- `vector_split_preflight.log`: additions-only patch application and Fortran
  syntax proof.
- `vector_advection_evidence.sha256`: evidence digests.

## Acquisition

The operator entry point is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round62_vector_advection_acquisition/run.sh`.
Its `--preflight-only` mode passes and names the fresh target
`ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP`; no `mpirun` was attempted in the sandbox.

The acquisition patch is additions-only.  It leaves the compiled `dyn_keg`
and `dyn_zad` calls in their original order and adds WRITE-only callbacks
before KEG, after KEG, and after ZAD.  Each MPI rank writes a uniquely tagged,
self-describing stream containing magic, twenty header integers, then named
`(rank,n1,n2,n3,payload)` fields through physical EOF.  The fields carry the
three cumulative boundaries, Kmm U/V, `ww`, effective `wsd`, live U/V
thicknesses, areas, reciprocal metrics, masks, global origins, owned bounds,
and executed flags.  The admission checker derives every payload length from
the stream header and verifies the four ten-step restarts byte-for-byte
against the admitted parent.  Header, field-order, truncation, restart, and
stamp plants must all fire.

## Shared-card gates

No model or card file changed.  Therefore the ORCA2, GYRE, DINO, OVERFLOW,
LOCK_EXCHANGE, and generic-card trajectories cannot move and were not rerun.
This is a stopped measurement round, not a physics landing.  The current GYRE
certified digest remains `cf06a8fc7d0e90f2`; this receipt makes no new GYRE
trajectory claim.

## Review and validation

The required separate review was attempted with `codex exec --sandbox
read-only`.  It could not initialize its in-process app-server client because
the read-only sandbox rejected a filesystem write.  Verdict: **independent
review unavailable in-sandbox**.

The citation gate is run both on this receipt and on its default cumulative
receipt.  A shifted-line plant on this receipt's KEG citation must fail.  The
focused gate controls, Python compilation, acquisition preflight, and the
single `tests/ocean/fidelity` battery are recorded in the final commit below.

## OPEN

1. Operator: run the committed acquisition `run.sh --run`.  Do not reuse or
   overwrite any existing target.
2. Next round: admit both self-describing rank streams and their restart
   calibration; replay KEG source order first, score legoESM's exposed KEG,
   then replay and score ZAD.  Name the first non-bit child statement only
   after those two boundaries exist.
3. After this step-level owner is closed, continue to the QCO mixed boundary,
   then Decision 52's independent-start ladder and the ORCA2 month-scale
   magnitude ranking in the standing order.

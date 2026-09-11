# GYRE phase-3 round-47 receipt: returned-twin admission and kt=2 adjudication

Date: 2026-09-11. Base commit: `715c9865008e7e419004ad1e5e643e8b71257c00`.
Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round47/`.
All executions were CPU/fp64. No NEMO executable was run.

## Returned twin

The consumed-field admission is **PASS**: 42 of 55 inherited records are byte
identical. The other 13 contain 88,094 changed elements; the report retains
490 capped diagnostic examples but now records exact reason counts. There are
zero owned-and-defined differences.
`GYRE_OMIP_L2_P3_00000010_restart.nc` and `mesh_mask.nc` are byte-identical.
The planted owned-cell change exits 1. Evidence: `round47_admission.json`
(`dfcfd357...b7dd7c`) and `round47_admission_plant.json`
(`4fde8b63...21b06`).

| record (all kt=1) | changed | halo | undefined slot | undefined region | violation |
|---|---:|---:|---:|---:|---:|
| dynadv_split_s3 | 7 | 7 | 0 | 0 | 0 |
| rkstage1_transport_operands | 4,218 | 2,953 | 0 | 1,265 | 0 |
| rkstage2_ene_operands | 870 | 870 | 0 | 0 | 0 |
| rkstage3_terms | 7 | 7 | 0 | 0 | 0 |
| rkstage3_wzv | 3,151 | 3,151 | 0 | 0 | 0 |
| rktracer_operands_s1 | 5,917 | 5,917 | 0 | 0 | 0 |
| rktracer_operands_s2 | 5,441 | 5,441 | 0 | 0 | 0 |
| slow_forcing | 8 | 8 | 0 | 0 | 0 |
| tracer_transport_s3 | 5,441 | 5,441 | 0 | 0 | 0 |
| transport_s1 | 29,800 | 9,299 | 19,236 | 1,265 | 0 |
| transport_s2 | 27,781 | 6,937 | 19,831 | 1,013 | 0 |
| transport_s3 | 5,441 | 5,441 | 0 | 0 | 0 |
| zdf_matrix | 12 | 12 | 0 | 0 | 0 |

The bottom-plane `zFu/zFv` undefined-region registrations were added
**POST-HOC** after the preregistered narrower waiver inventory was refuted.
They are limited to two snapshots: NEMO fills only `jk=1:jpkm1` before the
TRPOP write at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:292-316` and the
whole-array transport write at `:350-357`; it zeros the bottom only later at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv.f90:248-250`.

The reported “32 bytes = four doubles” inference was false. In both
`oracle_rkstage3_terms_kt00000001.bin` and
`oracle_dynadv_split_kt00000001_s3.bin`, 32 changed byte positions span seven
`ww` words at zero-based `(i,j,k)` indices `(0,0,0)`, `(1,0,0)`, `(2,0,0)`,
`(3,0,0)`, `(4,0,0)`, `(8,0,0)`, `(9,0,0)`. Baseline/candidate values are,
respectively: `6.89846211742727e-310/6.942155855076e-310` (first two),
`4.65920680442315e-310/4.6514531228709e-310` (next two),
`1.1375016612e-313/1.13560867255e-313`,
`4.65920683034776e-310/4.65145314879553e-310`, and
`4.6592068318023e-310/4.65145315025006e-310`. Every cell is on `j=0`, outside
NEMO's owned bounds `j=3:24`; no fidelity gate consumes it.

The acquisition script now treats raw comparison as an inventory and uses the
schema admission for its verdict. No kt=2 record has an independent baseline
that could support the requested cross-run raw-identity claim, and RKTS3/ADVSP
`ww` outer halos are not fully defined. Instead, the kt=2 records
`oracle_rkstage3_terms_kt00000002.bin` and
`oracle_dynadv_split_kt00000002_s3.bin` retain within-run duplicate-payload bit
gates against 28 and 29 shared named fields in
`oracle_momstage_kt00000002_s3.bin`; this proves duplicate identity, not field
definedness.

## kt=2 adjudication

Compiled-source replays of WZV, KEG, and ZAD are bit-exact at kt=1/2 stages
1/2/3 (all 30 rows have zero unequal cells). NEMO constructs stage WZV from
Kmm velocity at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:329-335`;
`dyn_zad` consumes `ww` and Kmm U/V at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:105-137`. Stage 1 is
distinct: its HPG/LDF/VOR/KEG/ZAD RHS is formed in
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-176`, and vector
momentum is deliberately not recomputed in
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:329-335` and
`:365-371`.

The preregistered “first difference after stage-1 ZAD” prediction is
**REFUTED**. The literal source replays remain exact, while the candidate
model-path scores are:

| kt=2 stage | WZV/KEG/ZAD replay unequal | first isolated non-bit component U/V | KEG accumulator U/V | ZAD accumulator U/V |
|---:|---:|---:|---:|---:|
| 1 | 0 / 0 / 0 | LDF 17,400 / 17,100 | 1 / 0 | 10 / 2 |
| 2 | 0 / 0 / 0 | VOR 7,583 / 7,342 | 1 / 0 | 14 / 14 |
| 3 | 0 / 0 / 0 | VOR 7,585 / 7,135 | 0 / 0 | 10 / 22 |

The first measured non-bit accumulator is stage-1 KEG (maximum
`4.1359030627651384e-25`), before ZAD (maximum
`2.6469779601696886e-23`). The first causal NEMO statement remains
**UNRESOLVED**: LDF and VOR are non-bit as isolated subtractions, and although
the record contains both post-statement arrays, the current model-path gate
does not reconstruct/expose those two accumulator comparisons. Evidence:
`round47_candidate_given_dirty.json` (`399d5776...bfa22`).

The reconstructed round-44 candidate was a private `nemo_stage_zad_operands`
argument through `LatLonCGridOceanModel.tendencies` and the PE kernel, supplied
from `_g1[2,4,5]` and `_g2[2,4,5]` at the shared stage-2/3 `_mom_pert_ws`
calls. The stage-1 extension replaced only ZAD by the difference between the
same helper with `_g0[2,4,5]` and without it. Executable changes were removed;
the exact rejected stage-1/2/3 diff is preserved at
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round47_rejected_stage_zad.patch`.

The candidate is ineligible: it remains non-exact on GYRE given-input rows,
and Rule 12 records 57 worsened GYRE rows. GYRE first-over-bar remains kt=2
U/V; the kt=2 absolute maxima move
`2.7478404751243857e-12 -> 2.7377110452773967e-12` (U) and
`3.305560306813421e-12 -> 3.284922138989399e-12` (V), with maximum worsening
`18394930599.77832` row-scale ULPs. No physics is landed; shipped/candidate
kt=1..10 first-over-bar is kt=2 U/V in both arms.

| card | relevant arm | round-47 candidate verdict |
|---|---|---|
| GYRE | shared WS-RK3 explicit ZAD | REJECTED: non-exact; 57 worsened rows |
| LOCK_EXCHANGE | shared RK3 program; flux-UP3, no `dyn_zad` | UNMEASURED this round; round-33/25 records only inherited |
| OVERFLOW | shared RK3 program; flux-UP3, no `dyn_zad` | UNMEASURED this round; round-33/25 records only inherited |
| DINO | separate leap-frog branch | not executed; no direct statement risk from this rejected patch |
| ORCA2 | equivalent stage records absent | UNMEASURED-with-spec |

The kt=1-end bridge proves recorded T/S/U/V/SSH exact after injection, but all
three kt=2 trajectory stages are **UNMEASURED**: the record omits NEMO's
persistent cross-window barotropic arrays `ubb_e`, `ub_e`, `vbb_e`, `vb_e`,
`sshbb_e`, and `sshb_e`. The gate exits 1 instead of emitting a PASS report
with contaminated post-baro rows.

## Disposition

ASKED: returned-twin adjudication, kt=2 score, and Rule-12 landing decision.
UNASKED: no configuration, threshold, public API, or physics choice. Open:
capture the six kt=1-end history arrays; add model-path post-LDF and post-VOR
accumulator scores using the arrays already recorded; specify native ORCA2
per-stage records. There is no safe
reacquisition command yet: current `run.sh` reproduces the incomplete record,
so its writer/schema must first be extended and preregistered.

Final dirty-tree validation gate: `round47_validation_final_dirty.json`
(`2d4442b0...9fba9`), PASS. The all-mode report
`round47_all_final_dirty.json` (`2bf61273...36d9`) exits 1 with status
UNMEASURED. The citation gate passes and its shifted-line plant exits 1; the
focused suite passes 106 tests.

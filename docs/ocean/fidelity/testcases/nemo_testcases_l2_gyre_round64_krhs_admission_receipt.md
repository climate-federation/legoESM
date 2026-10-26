# NEMO testcase L2 GYRE — round 64 admission repair receipt

Date: 2026-09-12. No NEMO build or integration was run in this round.

## Admission failure: no consumed-field perturbation

`round63_admission.json` contains 63 baseline records, 49 candidate records,
27 byte-identical records, 20 classified raw changes, 194 admitted values,
and 16 missing records. Every classified record has
`consumed_equal=true`; every `owned_defined_violation` count is zero. The
requested list of consumed fields that lost exactness is therefore **empty**.

For completeness, these are all raw changed field-record pairs from the JSON;
index is zero-based and magnitude is the absolute difference at that first
index, recomputed from the two binary payloads:

| record | field | first index | magnitude |
|---|---|---:|---:|
| oracle_dynadv_split_kt00000001_s3.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_dynadv_split_kt00000002_s3.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_momstage_kt00000001_s1.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_momstage_kt00000001_s2.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_momstage_kt00000001_s3.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_momstage_kt00000002_s1.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_momstage_kt00000002_s2.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_momstage_kt00000002_s3.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_rkstage1_transport_operands_kt00000001.bin | zFv | [31,14,30] | 6.64024228010635355e-321 |
| oracle_rkstage3_terms_kt00000001.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_rkstage3_terms_kt00000002.bin | ww | [0,0,0] | 3.85117368922021551e-312 |
| oracle_rkstage3_wzv_kt00000001.bin | ww_pre_aimp | [0,0,0] | 3.85117368922021551e-312 |
| oracle_rkstage3_wzv_kt00000001.bin | ww_post_aimp | [0,0,0] | 3.85117368922021551e-312 |
| oracle_rkstage3_wzv_kt00000001.bin | pFw | [0,0,0] | 3.85117368922021551e-312 |
| oracle_rktracer_operands_kt00000001_s1.bin | zFw | [35,1,17] | 6.64024228010635355e-321 |
| oracle_rktracer_operands_kt00000001_s2.bin | zFw | [0,0,0] | 3.85117368922021551e-312 |
| oracle_slow_forcing_kt00000001.bin | utau | [0,0] | 3.85117368937831652e-312 |
| oracle_slow_forcing_kt00000001.bin | vtau | [0,0] | 3.85117368929926601e-312 |
| oracle_tracer_transport_kt00000001_s3.bin | zFw | [0,0,0] | 3.85117368922021551e-312 |
| oracle_transport_kt00000001_s1.bin | zFv | [31,14,30] | 6.64024228010635355e-321 |
| oracle_transport_kt00000001_s1.bin | zFw | [35,1,17] | 6.64024228010635355e-321 |
| oracle_transport_kt00000001_s2.bin | zFw | [0,0,0] | 3.85117368922021551e-312 |
| oracle_transport_kt00000001_s3.bin | zFw | [0,0,0] | 3.85117368922021551e-312 |
| oracle_zdf_matrix_kt00000001.bin | rCdU_bot | [0,0] | 3.85117369001072054e-312 |
| oracle_zdf_matrix_kt00000001.bin | utauU | [0,0] | 3.85117368937831652e-312 |
| oracle_zdf_matrix_kt00000001.bin | vtauV | [0,0] | 3.85117368929926601e-312 |

The first raw-changed record in compiled execution order is
`oracle_zdf_matrix_kt00000001.bin`: `zdf_phy` and its dump run before
`stp_2D` and the three stages at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:168-201`.
Its first changed field is halo `rCdU_bot[0,0]`; it is not consumed.

The values are subnormal uninitialised halo or registered undefined-slot bytes.
Adding a module with saved allocatables and calls changes process memory
layout, so inherited writers expose different garbage in payload regions NEMO
does not own or consume. The admission projection removes only those already
registered regions and proves all consumed values exact.

## Compiled-source delta and named cause

Line-by-line preprocessed diffs contain additions only. The stage driver adds
snapshot calls between unchanged zero/advection/SBC statements:
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:825-863` versus
`GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868`.
The FCT first-guess call remains unchanged and is followed by one recorder call:
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:169-172` versus
`GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/traadv_fct.f90:170-174`.
The ZDF solve is followed by one recorder call:
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:145-149` versus
`GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/trazdf.f90:146-151`.
The TKE solve and RHS loop are bracketed, without changing their statements:
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:188-192` versus
`GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/zdftke.f90:190-196`, and
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:409-427` versus
`GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/zdftke.f90:413-433`.

No added compiled statement changes the trajectory: common consumed fields are
bit-identical. The acquisition-changing statement is instead the round-63
runner's namelist rewrite, observed as
`GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg:21` (`nn_itend=10`) versus
`GYRE_OMIP_L2_P3_SM_R63KRHS/EXP00/namelist_cfg:21` (`nn_itend=2`). It removes
exactly eight later step-entry and eight later barotropic-frame records.

R64 reuses the unchanged WRITE-only instrument and copies the R46 namelist
byte-for-byte, guarded by `cmp`; its calibration expects `nn_itend=10`.
The missing-record class was not new and already has a planted test, so no new
admission waiver or plant was added.

## PROVISIONAL — NOT ADMISSIBLE

The committed CPU-only preview at `146748f07f2a` verifies the rejected record's
SHA/producer stamp, exact T/S content calibration, failed admission reason,
float64 state/geometry, and bit-exact model input versus recorded Kbb.

Zero is exact for T and S (0/18,000 unequal). The first unequal boundary is
complete FCT advection for both tracers. Temperature: content 18,000/18,000
unequal, max 5.954036487310077e-05 K m, RMS 4.361392671238398e-06; derived
Krhs max 6.196947409102175e-11 K s-1. Salt: content 18,000/18,000 unequal,
max 7.651455234736204e-06, RMS 5.281088126648484e-07; derived Krhs max
5.135360597528654e-12 s-1.

These values are **PROVISIONAL ONLY**. They do not confirm an owner and may not
be cited as a finding. The identical walk is preregistered for the admitted R64
record. Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round64/round64_provisional_krhs_walk_v2.json`.

## Compile check, controls, and handoff

The R63 writer and four dry-patched files were preprocessed with the exact R46
key set and compiled with `gfortran -fsyntax-only`; all five cpp commands and
all five compiler commands exited 0 with empty output. Products are retained at
`/tmp/gyre-r64-check-syntax.tzjjig`.

Focused result: 27 passed. The provisional metric's synthetic
one-ULP consumed-value test turns its exact row unequal. The shared admission
gate's missing-record and owned-bit plants both pass.

Exact operator command after applying the stamped commits:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round64_krhs_split/run.sh
```

## ASKED / UNASKED

| kind | item | disposition |
|---|---|---|
| ASKED | Diagnose R63 admission, provisional read, R64 runner | Completed |
| UNASKED | Configuration, production physics, carried state | None changed |
| UNASKED | NEMO build/integration | Not run; operator handoff |

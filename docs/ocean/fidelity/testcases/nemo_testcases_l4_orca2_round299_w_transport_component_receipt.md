# ORCA2 round 299 — kt=7 W-transport component

Date: 2026-10-16

Status: **STOPPED_FOR_RECORD**. Under both **independent OMT-4** and
**given NEMO's entry**, replacing only the recorded vertical active-tracer
transport (`trp_w`) takes the kt=7 stage-1 fold-band salinity result below the
frozen 10x growth threshold. Replacing only `trp_u` or only `trp_v` does not.
The preregistered `trp_v` prediction is therefore **REFUTED** and retained.

This round changes no production package, card, deck, selector, carried state,
stabiliser, sea-ice field, or `unmeasured_features` entry. The private fold
unit remains private and atomic.

## Provenance and controls

- Lane base: `4a46c6da3`; frozen preregistration: `53fbabbcd`.
- Measurement tree: `266242d384141040d772cde681ab78e987eeba89`, clean.
- Admitted input record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round297/acquisition/orca2_omt4_stage7_source_10step_np2`.
- Admission remains `PASS_R297_OMT4_KT7_STAGE1_ADMISSION`: two ranks, 23
  self-described fields, exactly-once 148x180 ownership, unchanged inherited
  frames and kt=10 restarts.
- Raw measurement SHA-256:
  `20ca10dbbaad47acf8fab306ed4e44d58bcd4fdcce432ea774bccc9d261f0d19`.
- Classification SHA-256:
  `0092797a9c2becc3adc90a397fcbe9f56d27d6c7c67fa67b09a5aab052d6133a`.

The passive live trace is array-identical to the ordinary completed state for
every carried field under both labels. The trusted tracer replay reproduces
the production completed T and S arrays exactly, and the bounded eager
binary64 transport replay reproduces that trusted replay exactly. The eager
calculation emitted divide warnings only in inactive land/below-bottom slots;
the active-support result is finite and exact against the trusted replay.
Decision 127 is bound by a separately counted zero-versus-subnormal control;
the normal one-bit control remains unequal.

All nine classifier plants refuse with exit 2 and `STATUS PLANT-FIRED`:
admission, label coverage, component order, passivity, trusted replay,
candidate control, full-NEMO control, first-component selection, and normal
one-bit handling. The acquisition's layout plant refuses with exit 69.

## Compiled source order

The admitted record's compiled stage program calls the active-tracer
transport builder at
`ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/stprk3_stg.f90:565-570` and records
the returned U, V, and W fields in that order at
`ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/stprk3_stg.f90:599-601`.

On the resolved vector/C2, non-Shuman branch, the transport routine first
sets the bottom transports to zero at
`ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:248-252`, calls the W
diagnosis at
`ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:267-302`, and finally
forms `pFw=e1e2t*ww` at
`ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:314-316`.
Inside that diagnosis, NEMO's transport-form horizontal divergence is written
at `ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/divhor.f90:132-153`; the QCO
bottom-up recurrence adds the reference-thickness and before/after ratio term
at `ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/sshwzv.f90:277-299`.

Thus `trp_w` is the first source-ordered causal component supported by the
current record. This is a boundary statement, not yet a leaf-statement claim.

## Component measurement

The frozen threshold is ten times the entry fold-band salinity maximum,
`3256.0374737332186` PSU. Both labels independently produce the same table:

| Label | Entry S max | Candidate S max | U-only arm | V-only arm | W-only arm | Full-NEMO arm |
|---|---:|---:|---:|---:|---:|---:|
| independent | 325.6037473733219 | 12046.543040268563 | 12040.813765686724 | 12046.096355886591 | 354.2948854268084 | 325.50387971291514 |
| given NEMO's entry | 325.6037473733219 | 12046.543040268563 | 12040.813765686724 | 12046.096355886591 | 354.2948854268084 | 325.50387971291514 |

The corresponding fold-band temperature maxima are 650.7944752951817 K for
the candidate, 650.4852414679949 K for U-only, 650.7702255808301 K for
V-only, 19.171090013721386 K for W-only, and 17.612033147439554 K for the
full-NEMO arm. Only W is a single-component closer under the frozen causal
predicate.

## Prediction ledger

| ID | Verdict | Result |
|---|---|---|
| R299-P1 | **CONFIRMED** | Admission, clean CPU/fp64/libm provenance, passivity, and both replay controls pass. |
| R299-P2 | **CONFIRMED** | Both labels independently name `trp_w`. |
| R299-P3 | **REFUTED** | `trp_w`, not the preregistered `trp_v`, is the first causal component. |
| R299-P4 | **CONFIRMED** | Candidate and full-NEMO controls reproduce round 298 exactly. |
| R299-P5 | **CONFIRMED_RECORD_GAP** | The admitted kt=7 stream lacks the W diagnosis's thickness/ratio leaves. |
| R299-P6 | **CONFIRMED_MEASUREMENT_ONLY** | No model or private constituent lands. |

## Acquisition request

The fail-closed operator launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round299_w_acquisition/run.sh`.
Its `--preflight-only` mode passes and names a new target,
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round299/acquisition/orca2_omt4_stage7_w_10step_np2`.

The acquisition extends the admitted round-297 writer from 23 to 31 fields,
preserving the original order and adding only `r1_e1e2t`, `e1e2t`, `e3t_0`,
`r3t(Kbb)`, `r3t(Kmm)`, `r3t(Kaa)`, `tmask`, and the pre-area `ww`. It is
rank-complete and self-describing; the checker derives field names and payload
sizes from each header. Admission requires the 80 inherited frames and both
kt=10 terminal restarts to remain byte-identical, and eight record/admission
plants plus the source-layout plant must fire. The operator runs this script;
this sandbox does not retry `mpirun`.

## Validation and review

- Focused round-299 classifier and acquisition controls: **6/6 pass**.
- Citation gate, focused campaign tests, required fidelity battery, and
  independent read-only review are recorded in the final validation commit.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 126 remains pending; runoff is unchanged and is not re-raised.
- Sea ice and the ORCA2 card's `unmeasured_features` remain unchanged.

## OPEN

1. Operator runs the committed round-299 acquisition and admits the new
   rank-complete record.
2. Replay the W builder in compiled order: bottom zero, transport divergence,
   QCO thickness-ratio recurrence, and `e1e2t*ww`. Name the first non-bit leaf
   under both labels; retain any failed prediction.
3. Keep the complete fold unit private and atomic. Nothing lands until OMT-4
   completes under both labels and the standing cross-card gates pass.

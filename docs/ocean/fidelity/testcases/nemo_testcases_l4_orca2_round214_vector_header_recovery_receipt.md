# ORCA2 round 214 — OMT-1 vector-record header recovery

Date: 2026-10-09. Base: `8460116cc`. Preregistration commit:
`be7a93b34`. Checker commit: `933103565`. Recovery acquisition commit:
`b88a73a1c`. Status: **STOPPED_FOR_RECORD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round214/`.
No trajectory number is reported. Future numbers remain separately labelled
**independent OMT-1** and **given NEMO's entry OMT-1**.

## Verdict

The operator's round-213 NEMO run completed with `STOP 0` and wrote both rank
files, but the record is not admissible. The first refusal, `(Kmm,Krhs)=(1,3)`
versus checker expectation `(1,2)`, was a checker defect: the executing program
passes `Nbb,Nbb,Naa,Nrhs` into the first external solve at
`ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stprk3.f90:204-215`, and `stp_2D`
passes `Kbb,Kbb,Krhs` into `dyn_spg_ts` at
`ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stp2d.f90:303-303`. At kt=1 the live
pair is therefore `(1,3)`. The checker and its synthetic fixture now require
that pair, and a dedicated RK-level plant fires.

That correction exposed a writer defect. `zv_frc` is an owned-only 90x148
array, while `ssvmask` and `va_e` are full-local 94x152 arrays. The compiled
writer labels all three payloads 94x152 at
`ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/dynspg_ts.f90:714-726`. The fail-closed
parser reaches the second field early and refuses `zv_frc dimensions moved`;
it does not infer an unstated payload shape. R214-P1 and R214-P2 are therefore
**REFUTED**, not silently repaired by the reader.

The replacement instrument changes only the first field header to
`SIZE(zv_frc,1), SIZE(zv_frc,2)`. It uses a new configuration and run target,
parses all dimensions from the file, and retains the round-213 requirements:
two complementary ranks, finite payloads, 64 byte-identical inherited frames,
two byte-identical step-8 restarts, and nine firing corruption/passivity
plants. Its committed preflight reports `SYNTAX_PROOF_PASS` and
`ORCA2_ROUND214_VECTOR_HEADER_PREFLIGHT_READY`.

## Frozen-prediction disposition

| ID | disposition |
|---|---|
| R214-P1 | **REFUTED:** the `(1,3)` refusal was checker-only, but the same existing file also carries a false first-field shape. |
| R214-P2 | **REFUTED:** the existing file is malformed at `zv_frc`; frame and restart passivity remain unread because schema admission fails first. |
| R214-P3 | **PARTIAL:** all synthetic admission plants fire, including the new RK-level plant; real-record executions await the corrected acquisition. |
| R214-P4 | **UNMEASURED_WITH_SPEC:** no admissible pre-LBC target exists, so the cancelling pair is not replayed. |
| R214-P5 | **UNMEASURED_WITH_SPEC:** source ordering remains frozen; no trajectory claim is advanced. |

## Validation and scope

- Existing-record refusal log SHA256:
  `539fcee8f9c8a945c47ab5106073c6b7b299e6b4089e32c09cef7f741f27d001`.
- Recovery preflight log SHA256:
  `0398b7d205c0c425317e7b5830daf57ac4298c570da2b83bdeedfa0eeab02748`.
- The focused parser battery passes 10/10; the bad-level, bad-dimension,
  truncation, malformed-header/name/rank/non-finite, frame-byte, and
  restart-byte controls are non-vacuous.
- No `packages/` file, card, configuration selection, sea-ice selector, or
  carried state changed. GYRE, DINO, tanks, rung 0, rung 10, and the ORCA2
  `unmeasured_features` tuple cannot execute changed model code.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round214_vector_header_acquisition/run.sh --run`.
2. Admit the corrected rank pair and require all inherited frames/restarts to
   remain byte-identical before reading `va_pre_lbc`.
3. Replay and score the complete slow-V + raw-mask pair atomically under
   Decision 96. OMT-2 waits for that disposition.

No configuration or sea-ice decision is pending. ASKED choices: Decisions 103
and 109. UNASKED choices: empty.

# Preregistration — ORCA2 round 211 OMT-1 eight-step recovery

Date: 2026-10-09. Frozen base: `a6e08ab99`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round211/`.
Every later trajectory number remains separately labelled **independent OMT-1**
or **given NEMO's entry OMT-1**. This round changes no model configuration,
sea-ice selector, carried state, or stabiliser.

## Frozen observation and source order

The round-210 uninstrumented OMT-1 run completed kt=8 and stopped through
NEMO's own stability control at kt=9: `|V| max = 10.13 m/s` at local
`(22,85,27)`. The compiled check prints the five stop diagnostics at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:293-298`. Therefore a
ten-step OMT-1 record cannot exist. The admissible ladder is kt=1..8; the
step-9 stop is the rung's registered stability boundary.

The record binary already exists at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/bin/nemo.exe`. The executing vector-form
external-mode update is the direct U/V statement at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`.
Round 211 reuses this binary and the unchanged round-209 OMT-1 deck. It does
not run `makenemo`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R211-P1 | The existing record binary and OMT-1 deck are unchanged. | Their SHA256 values equal the round-210 pinned values, and the deck delta from OMT-0 remains only `ln_dynadv_OFF=true→false` plus `ln_dynadv_vec=false→true`. | Any hash or extra deck change: **REFUTED**; stop before NEMO. |
| R211-P2 | OMT-1 is finite through kt=8 under the identical deck. | A two-step smoke and an uninstrumented eight-step calibration finish with `STOP 0`; rank-complete step-8 restarts exist. | Any earlier stop or non-finite field: **REFUTED**; no ladder is admitted. |
| R211-P3 | The passive frame writer records the complete admissible ladder. | Each of two instrumented twins has 64 self-describing frames: two ranks × eight steps × four stages; headers and payloads are finite and twin-array-identical. | Missing/malformed/non-finite frame or any twin bit: **REFUTED**. |
| R211-P4 | The writer is additions-only through kt=8. | The uninstrumented calibration and both instrumented twins have byte-identical rank-complete step-8 restarts. | Any terminal byte differs: **REFUTED**; no frame value is citable. |
| R211-P5 | The registered NEMO boundary is exactly kt=9. | The preserved round-210 run resolves the same OMT-1 deck, aborts through compiled `stp_ctl`, and reports kt=9 with `|V|=10.13 m/s`. | A different resolved deck, stop route, or boundary: **REFUTED**; retain the observed result and stop. |
| R211-P6 | The recovery gates bind. | Header, field-name, truncation, non-finite, missing-frame, twin-ULP, terminal-byte, changed-binary, and wrong-boundary plants each refuse. | Any plant stays green: no record claim is citable. |

## Landing predicate

This round lands only the corrected acquisition/admission contract. The OMT-1
card remains unlanded until the operator record admits and both labelled
kt=1..8 ladders complete. No `packages/` file changes, so GYRE, DINO, tanks,
rung 0, and the shipped rung-10 trajectory cannot move.

ASKED choices: Decisions 103 and 109, plus the operator's binding kt=1..8
recovery protocol. UNASKED choices: empty.

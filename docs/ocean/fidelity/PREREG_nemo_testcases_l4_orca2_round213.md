# Preregistration — ORCA2 round 213 OMT-1 vector pre-boundary record

Date: 2026-10-09. Frozen base: `c971cc955`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round213/`.
Every future trajectory number remains separately labelled **independent
OMT-1** or **given NEMO's entry OMT-1**. This acquisition changes no sea-ice
selector, card, carried state, stabiliser, or `unmeasured_features` tuple.

## Frozen source boundary and record contract

The compiled OMT-1 source selects the direct vector update at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`, then applies
the boundary exchange at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-739`. Round 212
proved that the 35 northern-fold differences in the update's `zv_frc` and
`ssvmask` operands cancel in the raw expression, but the admitted stream writes
`va_e` only after the exchange. The missing target is therefore substep-1
`va_e` immediately after line 679 and before any drag, depth update, or
`lbc_lnk` call.

The writer is additions-only and emits one self-describing kt=1/substep-1 file
per rank. Its header carries magic, version, step, substep, RK levels, rank,
local shape, global origin, owned bounds, precision, and field count. The field
headers carry name, rank, and dimensions; payloads are `zv_frc`, `ssvmask`, and
`va_pre_lbc`. Admission is driven from those headers, not predicted byte counts.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R213-P1 | The executing source boundary is unchanged. | The pinned OMT-1 source/build/deck hashes reproduce, the vector branch is live, and the writer is inserted after the direct update and before every later mutation or exchange. | Any hash, selector, or anchor change: **REFUTED**; stop. |
| R213-P2 | The rank-complete record is well formed. | Exactly two files admit with complementary exactly-once 148×180 owned coverage, kt=1, substep 1, fp64, and the three named arrays with dimensions parsed from their headers. | Missing/extra rank, malformed/truncated field, wrong name/dimensions, overlap/gap, or non-finite owned payload: **REFUTED**; no number is read. |
| R213-P3 | The writer is passive. | All 64 existing per-stage frames and both step-8 rank restarts are byte-identical to the admitted round-211 OMT-1 run. | Any changed frame or restart byte: **REFUTED**; the record is not evidence. |
| R213-P4 | Every admission control fires. | Header, field-name, field-dimensions, truncation, swapped-rank, non-finite, frame-byte, and restart-byte plants all refuse. | Any plant stays green: no round-213 claim is citable. |
| R213-P5 | The record resolves round 212's only missing target. | After operator acquisition, an offline replay can compare the complete recorded `zv_frc` + `ssvmask` expression against `va_pre_lbc` before scoring the atomic pair. | Until acquisition: **UNMEASURED_WITH_SPEC** and `STOPPED_FOR_RECORD`; replay non-closure after admission: **HELD_INSTRUMENT**. |

## Landing predicate

This round lands only the committed acquisition and admission machinery. No
physics statement may land before the operator record satisfies P1-P4 and the
atomic replay satisfies P5. OMT-2 remains blocked behind that disposition.

ASKED choices: Decisions 103 and 109. UNASKED choices: empty.

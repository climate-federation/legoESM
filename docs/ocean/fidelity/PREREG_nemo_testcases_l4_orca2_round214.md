# Preregistration — ORCA2 round 214 OMT-1 vector-record recovery

Date: 2026-10-09. Frozen base: `8460116cc`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round214/`.
All trajectory numbers remain separately labelled **independent OMT-1** or
**given NEMO's entry OMT-1**. This round changes no configuration, model
state, sea-ice selector, stabiliser, or `unmeasured_features` tuple.

## Frozen correction and source path

Round 213's operator run completed with `STOP 0` and wrote both requested
self-describing records, but the checker refused their `(Kmm,Krhs)=(1,3)`
header because its synthetic fixture and assertion incorrectly expected
`(1,2)`. The compiled execution path passes `Nbb,Nbb,Naa,Nrhs` into `stp_2D`
at `ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stprk3.f90:204-215`, then calls
`dyn_spg_ts(kt,Kbb,Kbb,Krhs,...,Kaa)` at
`ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stp2d.f90:303`. The writer records
those live arguments at
`ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/dynspg_ts.f90:709-726`.
Therefore `(1,3)` is the record's source-exact stage-1 level pair; the writer
is unchanged and the checker expectation is corrected.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R214-P1 | The refusal is checker-only. | Both rank headers report `(Kmm,Krhs)=(1,3)`, the compiled call chain above selects those levels, and no writer/deck/binary content changes. | Any source/header disagreement: **REFUTED**; request a new record under a new target. |
| R214-P2 | The existing record is admissible. | Two rank files give exactly-once 148x180 coverage; their three named fp64 fields are finite; all 64 inherited frames and both step-8 restarts are byte-identical to round 211. | Any malformed payload, gap/overlap, or changed byte: **REFUTED** and `HELD_INSTRUMENT`. |
| R214-P3 | Admission controls remain non-vacuous. | Header, RK-level, field-name, dimensions, truncation, rank, non-finite, frame-byte, and restart-byte plants each refuse. | Any green plant: no number is citable. |
| R214-P4 | The complete recorded pair closes NEMO's pre-LBC V update. | Offline source-order replay using recorded `zv_frc` and `ssvmask` is bit-exact against recorded `va_pre_lbc` on every owned cell; either one-variable half retains the registered 35-cell debt. | Pair non-closure: **REFUTED** and `HELD_INSTRUMENT`; one half closing alone: retract the cancelling-pair framing. |
| R214-P5 | The existing pair remains the first numerical OMT-1 statement. | Replaying the recovered record leaves round 212's preceding signed-zero-only continuity statement unchanged and identifies no earlier numerical difference. | Earlier numerical debt: **REFUTED**; walk that statement before scoring this unit. |

## Landing predicate

The checker correction lands only if P1-P3 pass on the existing output. A
physics promotion is eligible only if P4-P5 pass and the atomic unit satisfies
Decision 96 on OMT-1, rung 0, rung 10, GYRE, DINO, and tanks. Otherwise the
round is held with the exact first red predicate. OMT-2 remains next after this
unit's disposition.

ASKED choices: Decisions 103 and 109. UNASKED choices: empty.

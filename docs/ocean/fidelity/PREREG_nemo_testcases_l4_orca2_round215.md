# Preregistration — ORCA2 round 215 OMT-1 vector fold/mask pair

Date: 2026-10-09. Frozen base: `058dc7680`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round215/`.
Every trajectory number remains separately labelled **independent OMT-1** or
**given NEMO's entry OMT-1**. This round changes no configuration, carried
state, stabiliser, sea-ice selector, or `unmeasured_features` tuple.

## Frozen source boundary

The compiled OMT-1 vector branch forms `va_e` from `vn_e`, `rDt_e`,
`zv_spg`, `zv_trd`, and `zv_frc`, then multiplies the complete expression by
raw `ssvmask` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682`.
Only afterward does the executable associate `va_e` across the north fold
with V sign -1 in the seven-array `lbc_lnk` call at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756`;
the T-pivot V mapping is compiled at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:684-718`.
Round 212 found 35 northern-fold cells where candidate `zv_frc` and raw
`ssvmask` each differ and cancel in the written vector expression. Round 214
requested their missing pre-LBC target under a corrected self-describing
header.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R215-P1 | The corrected record is admissible and passive. | Two rank files parse from their own headers, cover 148x180 exactly once, and all 64 inherited stage frames plus both step-8 restarts are byte-identical to round 211. | Any schema, coverage, finiteness, frame-byte, or restart-byte failure: **HELD_INSTRUMENT**; no payload number is read. |
| R215-P2 | The recorded pair closes the pre-LBC vector statement atomically. | Offline source-order replay with both recorded `zv_frc` and `ssvmask` is bit-exact against `va_pre_lbc` on all owned cells; each one-variable half retains the same 35-cell debt. | Pair non-closure: **HELD_INSTRUMENT**; one half closing alone: retract the cancelling-pair claim. |
| R215-P3 | The pair remains the first numerical OMT-1 statement. | The preceding continuity-forcing difference remains signed-zero-only and no earlier source-ordered operand exceeds the fixed floor. | Any earlier numerical debt: walk it first and do not score this pair. |
| R215-P4 | The private atomic pair is a Decision-96 net improvement. | On both labelled OMT-1 ladders, a majority of RMS-moved rows moves toward NEMO, the first-over-bar row moves toward or is unchanged, no exact row is lost, and kt=1 stage-1 SSH maximum does not worsen. Rung 0 and rung 10 are separately scored. | Any predicate fails: **HELD**; retain the pair privately and name the next cancelling partner. |
| R215-P5 | Shared-card gates stay within their standing bars. | GYRE year stays within Decision 59, DINO stays below its fixed month bar, and DINO/lock-exchange/overflow card gates do not worsen beyond their registered allowances. | Any red outside Decision 96's near-zero ratchet allowance: restore production and **HELD** with the exact row. |
| R215-P6 | Every new gate is non-vacuous. | Admission corruption plants, pair-half/pair-closure plants, Decision-96 census plant, and citation rigid-shift plant each refuse. | Any plant stays green: no round-215 landing or claim. |

## Landing predicate

The fold-associated slow-V forcing and raw surface-V mask are one indivisible
NEMO statement pair: neither half may land alone. The pair lands only if
P1-P6 all pass under the full OMT-1, rung-0, rung-10, GYRE, DINO, and tank
gates. Otherwise production is restored and the round is held at the first
red predicate. OMT-2 waits for this disposition.

ASKED choices: Decisions 103 and 109. UNASKED choices: empty.

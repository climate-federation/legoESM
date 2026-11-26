# Preregistration — ORCA2 round 163 pinned GYRE merge and halo remeasurement

Date: 2026-10-07. Frozen base: `86b84f6f382df9e2a574d6219d919a12b3fcc53c`.
Pinned merge parent: `ca822f33d930b9df4987561cc781cd3c1028e0a6`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round163/`.

Every rung-0 number is **independent**: the hierarchy card starts from its own
climatological T/S, zero velocity, and zero sea surface. Every rung-7 number is
**given NEMO's entry** under Decision 52. The labels may not be mixed. Sea ice,
all six sea-ice selectors, and the shipped card's `unmeasured_features` tuple
remain unchanged.

## Frozen merge and measurement protocol

1. Merge exactly the pinned GYRE tip above. The dry-run inventory predicts two
   textual conflict classes: the cumulative GYRE receipt's stale anchors and
   the citation gate's generated map. The resolution is a union: retain both
   lanes' scientific prose and every citation-map entry, then mechanically
   re-anchor changed package citations. Any package-code conflict is an
   unpredicted conflict and stops the merge for a decision.
2. The merge brings the already authorised SMT-3 tracer-LDF pair: NEMO's
   closed deepest W mask in `traldf_iso.f90:243-259` and the live stage-3 Kmm
   tracer thickness divisor at `traldf_iso.f90:306-310,327-331`. It also brings
   SMT-4 records and held measurement tools, but no SMT-4 physics landing.
3. On the committed merge, run the cumulative citation gate and plant, the
   push-gate battery, GYRE's 70-row ladder and certified year, DINO, the
   VORTEX/tank registry, the independent rung-0 ladder, and the given-entry
   rung-7 ladder. Compare every ORCA2 row to the frozen round-160/161 artifacts.
4. Re-run the independent rung-0 240-step month. Report the first non-finite
   step/field/cell, or terminal per-field RMS and maximum if all 240 steps
   complete.
5. Re-run the private U-cyclic/V-fold external-mode association on rung 0 and
   compare it with the same merged-tree control. Count moved rows toward/away
   by RMS and maximum, retain the first-over-bar and exact-row predicates, and
   compare kt=10 stage-3 SSH maximum explicitly.
6. The halo pair lands only if Decision 96 is satisfied: a strict majority of
   moved RMS rows toward NEMO, first-over-bar toward or unchanged, no exact-row
   loss, and SSH maximum no worse. Otherwise production remains unchanged and
   the source walk resumes at the first consumer of the associated vector-
   invariant external mode. The compiled first consumer is the weighted
   velocity accumulation at
   `ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:850-868`.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R163-P1 | The pinned merge has only the two predicted documentation/map conflict classes. | No package-code conflict; cumulative prose and citation map are unions; citation gate and plant pass. | Any package-code conflict, dropped lane content, unmapped citation, or plant staying green. |
| R163-P2 | GYRE remains pinned to round 237 after the merge. | The 70 ladder rows and residual archive are identical; year RMS is day30 `2.3432419318363155e-06`, day240 `6.5816987106668941e-05`, day360 `5.4077212586815052e-05` K with the pinned daily digests. | Any ladder/residual difference, or any registered year checkpoint differs. |
| R163-P3 | The SMT-3 tracer-LDF pair executes on ORCA2 but preserves both certified ladder boundaries. | At least one rung-0 or rung-7 row moves; neither ladder loses an exact row or moves first-over-bar earlier. | Neither ladder moves, an exact row is lost, a first debt moves earlier, or either ladder refuses. |
| R163-P4 | The shared blast radius stays admitted. | DINO remains below its fixed bar; every VORTEX/tank registry row retains its admitted status; generic-card and focused gates pass. | DINO exceeds its bar, any registered card row regresses outside its standing rule, or a generic-card gate refuses. |
| R163-P5 | The rung-0 month still first becomes non-finite at step 36 T `[86,159,0]`. | Exact same first non-finite boundary. | The boundary moves to another step/field/cell or all 240 steps complete. |
| R163-P6 | The halo pair remains ineligible under Decision 96 on the merged tree. | A majority of moved RMS rows go away, or first-over-bar/exact-row/SSH-maximum predicate fails. | A strict majority moves toward by RMS, first-over-bar is toward/unchanged, no exact row is lost, and kt10 stage-3 SSH maximum is no worse. |
| R163-P7 | If P6 confirms, the first consumer arm is non-vacuous: source-materialising NEMO's `za1 * ua_e` / `za1 * va_e` products before the velocity-sum addition changes at least one halo-candidate accumulator bit while the false default is passive. | Candidate moves at least one U/V accumulated-mean bit; unselected production is array-identical to the merged control. | Candidate moves no accumulator bit, the false default moves production, or the arm reaches a different source statement. |

## Terminal rule

The merge lands only after its prescribed gates pass. The halo pair remains a
private arm unless every R163-P6 Decision-96 predicate passes. If it remains
held, only a passive, one-variable consumer discriminator may be retained; no
configuration, carried state, stabiliser, sea-ice selector, or production halo
statement changes. Failed predictions remain in the receipt.

ASKED choices: pinned merge and Decision-96 predicate, authorised by Decision
97. UNASKED choices: empty.

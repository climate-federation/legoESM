# ORCA2 round 79a receipt — independent month remeasurement (corrected tree)

Date: 2026-09-30

Base: `f4e8cef04a760616d8f2ad637dd73cadcb9cddf7` (worktree clean; this branch's
current tip)

Disposition: **HELD; the independent month is remeasured on the corrected
tree and the standing-brief note B16 supersession is now numeric**

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state. No NEMO entry field is substituted. The comparison target is NEMO's
own from-rest 240-step restart (round 67's record; unchanged).

This round changes no model code. It reruns round 71's frozen independent-
month gate, unmodified, against the same NEMO records, on the tree that
carries round 78's TKE mixing-length-floor landing (note B16: NEMO's own
`1.0e-3 m` floor, `ln_zdfiwm`, replacing a legoESM defect that had left the
floor at an effective `1.0 m`). Round 71's numbers are the pre-fold-in
baseline and are **SUPERSEDED** by this round, per note B16.

## Command

Reproduced exactly: same script, same CLI, same six record inputs round 71
used (deck, surface, month, restart-ledger, ten-step calibration root, and
ten-step reference), only the code tree moved.

```
PYTHONPATH=$REPO:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
.venv/bin/python scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round71_independent_month_gate.py \
  --deck-root /data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0 \
  --surface-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round69/acquisition/orca1ice_surface_only_240step_np2 \
  --month-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round67/acquisition/orca1ice_uninstrumented_fromrest_30day_np2 \
  --restart-ledger /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round67/acquisition/round67_outputs.sha256 \
  --ten-step-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2 \
  --ten-step-reference /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round66/independent_ladder.json \
  --output .../round79/independent_month_79a.json
```

No NEMO run was launched. NEMO's from-rest month record (round 67) and
480-frame surface record (round 69) are reused unchanged, as instructed.

## Answer — BEFORE (round 71) -> AFTER (round 79a)

Both rows are the independent 240-step terminal error against NEMO's own
from-rest restart, same scorer (`max_abs`, `rms`, unequal/count per field),
same five fields, same units.

| field | units | round-71 max\_abs (BEFORE) | round-79a max\_abs (AFTER) | round-71 rms (BEFORE) | round-79a rms (AFTER) | unequal/total BEFORE | unequal/total AFTER |
|---|---|---:|---:|---:|---:|---:|---:|
| T | degC | 21.637832697714707 | 21.16204086038964 | 0.17564842520962015 | 0.18871605115627163 | 438484/799200 | 438484/799200 |
| ssh | m | 5.563319462718695 | 5.578935655726782 | 0.06822817352994356 | 0.06850832563417603 | 16433/26640 | 16433/26640 |
| S | g/kg | 2.3984272944912632 | 4.2942151451623225 | 0.03606444372794216 | 0.04399530164405456 | 430552/799200 | 430552/799200 |
| u | m/s | 2.054457499249187 | 1.7552134244778128 | 0.022406667645546077 | 0.019175453374117628 | 444097/799200 | 444184/799200 |
| v | m/s | 1.1310611325760693 | 1.2125945863179692 | 0.013915785410822645 | 0.012956413517199093 | 449483/799200 | 449479/799200 |

Percent change (AFTER vs BEFORE): T max -2.2%, T rms +7.4%; ssh max +0.3%,
ssh rms +0.4%; S max **+79.0%**, S rms +22.0%; u max -14.6%, u rms -14.4%;
v max +7.2%, v rms -6.9%.

**Ranking is unchanged at the top.** Both by `max_abs` and by `rms`, the
order is T > ssh > S > u > v in both the BEFORE and AFTER tables.
**Temperature still owns the month-scale walk.** The magnitude ranking
round 71 established is not overturned by the fold-in; the salinity
row is the largest mover (near-doubling by `max_abs`), consistent with note
B16's reading that the corrected TKE floor uncovered a compensating error
rather than cleanly reducing the month error.

No per-process/per-region decomposition is reproduced in this round; that
walk (interval/depth/region) is round 72's separate analysis and is out of
scope for a same-protocol remeasurement.

## Predicate reproduction

The gate's own frozen predicates (unchanged from round 71's code) all fired
the same way:

- ten-step calibration: `exact: true` — the independent state after step 10
  still reproduces round 66's corrected kt=10 stage-3 row exactly (100 raw-bit
  field comparisons, 0 unequal). The instrument is still valid on this tree.
- initial-state census vs NEMO (round 66, independent): unchanged —
  `ssh` is the sole non-bit kt=1 field, `max_abs 0.015479333813968585`,
  identical to the pre-fold-in number. The TKE floor does not touch the
  entry state.
- terminal restart orientation: `BIT_EXACT_ORIENTATION`, same two restart
  shard digests as round 71 (round 67's record, unchanged).
- chlorophyll clock: `RESOLVED_CENTRES_MATCH`, same corrected piecewise
  clock as round 71 (no stale-instrument regression).
- `prediction_ledger`: `all_five_non_bit`, `original_round66_metric_reproduction`,
  `temperature_largest_max_abs` — all still true.
- `STATUS PASS_INDEPENDENT_MONTH_RANKING`; wall time 9699.045196533203 s
  (round 71: 9296.985275030136 s).

## Compiled source boundary

Unchanged from round 71 — the gate re-reads and re-confirms the same three
compiled call sites on this tree:

- forcing interpolation: `ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/fldread.f90:235-246`
- restart field source: `ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/restart.f90:170-184`
- restart call site: `ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:223-272`

## Validation

- `python -m py_compile` on the unmodified round-71 gate script: clean.
- The gate itself is the scorer and its own self-check: `STATUS
  PASS_INDEPENDENT_MONTH_RANKING`, worktree clean at commit `f4e8cef04`.
- No `packages/` or `scripts/` file changed in this round (this receipt and
  the campaign-state line are the only diff), so the mandatory dual
  adversarial review and the shared GYRE/DINO/tank trajectory gates are not
  applicable — same reasoning round 70 and round 71 gave for their
  measurement-only rounds. This is a documentation-only commit, exempt from
  the codex-review trigger.
- The full per-round citation-gate/codex-review apparatus that rounds 70/71
  ran (default + receipt-specific citation gates, a firing citation plant,
  the 170-test card battery, wide `tests/ocean/fidelity`) was **not** rerun
  here: the task is a same-protocol remeasurement producing one receipt and
  one campaign-state line, not a new statement or a new preregistration, so
  there is no new source citation or code change for those gates to certify.
  Stated explicitly as **NOT RUN**, not silently skipped.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round79/`
(`independent_month_79a.json`, `independent_month_79a.log`).

## Scope ledger

ASKED: reproduce round 70/71's independent-month protocol exactly on the
corrected tip and report BEFORE -> AFTER per field plus the top of the
ranking.

UNASKED and unchanged: configuration, selectors, thresholds, forcing
reconstruction, stabiliser, carried state, model arithmetic, sea ice, the
held QCO/RK statement, and the round-78 TKE floor landing itself (measured
here, not touched).

## OPEN

1. Per standing-brief note B16(b): acquire a per-step vertical-mixing
   (avt/avm/TKE) record on the ORCA2 deck to find what the old 1 m floor was
   covering — the compensating error implied by salinity's +79% `max_abs`
   move is unmeasured mechanism, not yet localized.
2. Every pre-fold-in ORCA2 magnitude number, including round 71's
   independent month, is now formally superseded by this round's table;
   bit-level statements are unaffected.
3. The round-72 interval/depth/region temperature-growth walk has not been
   rerun on the corrected tree; it is a candidate for the next round, not
   this one.

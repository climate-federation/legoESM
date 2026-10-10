# ORCA2 round 229 — vector-unit bisect and rank-complete record gap

Date: 2026-10-10. Frozen base: `1e5d05313`. Preregistration commit:
`dd417e76f6`. Measurement commit: `270a521c7`. Production-restoration commit:
`fc0a5d75d`. Acquisition commits: `d5f03d731`, `00023158e`. Status:
**STOPPED_FOR_RECORD / HELD**. No model statement, card selector, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` entry changed. The final
`packages/` tree is byte-identical to the frozen base. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round229/`.

Every trajectory number below is labelled **independent** or **given NEMO's
entry**. The two labels were run separately. Equality between their results is
measured, not assumed.

## Source boundary and method

The compiled T-pivot V program writes the pivot from the mirrored southern row
and the halo from the next southern row, both with sign -1, including the
special first longitude, at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:973-980`. The vector
external update, seven-array association, and V transport are respectively at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682`,
`:747-756`, and `:533-560`. NEMO associates T and S after every RK3 stage at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:784-796`.

The committed private diagnostic measured four parts of the held round-217
unit, one removal at a time: the slow Ve_rhs/raw-mask pair, the vector-update
raw mask, the seven-array association plus raw reference depths, and the
unmasked materialised V transport. Each of the 12 production-JIT CPU/fp64/libm
runs first proved its live trace bit-identical to its ordinary execution. The
only new executable hook was the raw vector-update-mask operand. That hook was
committed before measurement and removed afterward; the final model diff is
empty. Existing stage traces, OMT-4 card construction, reference-depth builder,
raw mesh mask, slow-forcing override, and round-228 score helpers were reused.

## Frozen predictions

R229-P1 is **CONFIRMED**. The compiled two-row V support, T-pivot right half,
and T-halo source each replay with zero unequal values. The V source-row,
V-sign, special-longitude, and T-halo-source plants each move the synthetic
known answer.

R229-P2 is **UNMEASURED — RECORD GAP**. The leave-one-part-out causal result is
clear, but its required NEMO zFv falsifier is not admitted. Removing only the
materialised V-transport part returns T/S to the unit-OFF scale under both
labels; none of the other removals does. However, the round-222 admission JSON
certifies only the rank-complete T/S/u/v/ssh frames. Its legacy
`oracle_transport_kt00000001_s1.bin` is written only under `lwp` before the
vector consumer at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:342-352`. The active
OMT-4 `ln_shuman=F` program updates/constructs the tracer transport later at
`:551-555`. The legacy stream is therefore rank-zero-only and at the wrong
program point; it is diagnostic-only and cannot confirm or refute transport
exactness.

R229-P3 is **UNMEASURED — SAME RECORD GAP**. The completed stage shows the
large endpoint on the pivot row, but without rank-complete T/S/e3t/tmask and
post-`tra_adv_trp` zFv at the consumer, the round cannot distinguish “correct
transport exposes stale halo operands” from “transport itself is wrong.” No
missing tracer-fold statement is named or proposed for landing.

R229-P4 is **CONFIRMED**. Independent and given-NEMO-entry results are equal
for every value in the table and reproduce round 228's first-boundary endpoint.

R229-P5 is **CONFIRMED**. The source-row/sign/special-longitude/T-halo tests,
part registry, label coverage, leave-one-out, and exact-bit controls fire. The
rank-complete acquisition checker separately fires rank, field-name, and
truncation plants.

## Causal bisect at kt=1 stage 1

Values are maximum absolute fold-band differences from NEMO. These are two
separate claim labels despite their measured equality.

| label | arm | T (K) | S (PSU) |
|---|---|---:|---:|
| independent | unit OFF | 0.0013606315900794863 | 0.0009639248797768118 |
| independent | full unit | 0.17733430832081432 | 3.283356343139289 |
| independent | remove slow V pair | 0.17733430832081432 | 3.283356343139289 |
| independent | remove vector V mask | 0.17733430832081432 | 3.283356343139289 |
| independent | remove association | 0.05508063392965634 | 1.0203855903505996 |
| independent | remove V transport | 0.0012139731153978373 | 0.0009232032446604421 |
| given NEMO's entry | unit OFF | 0.0013606315900794863 | 0.0009639248797768118 |
| given NEMO's entry | full unit | 0.17733430832081432 | 3.283356343139289 |
| given NEMO's entry | remove slow V pair | 0.17733430832081432 | 3.283356343139289 |
| given NEMO's entry | remove vector V mask | 0.17733430832081432 | 3.283356343139289 |
| given NEMO's entry | remove association | 0.05508063392965634 | 1.0203855903505996 |
| given NEMO's entry | remove V transport | 0.0012139731153978373 | 0.0009232032446604421 |

Thus V transport is the **exposure part**: removing it alone restores the
tracer endpoint. That is not evidence that the V transport is wrong. The
frozen decision requires its same-program-point NEMO operand before choosing
between the two hypotheses.

The unadmitted legacy stream would report 668/2,700 unequal pivot values and a
maximum 339003.55099822127 at local `[i=49,k=25]`, identically in every arm
including unit OFF. Those numbers are retained in `classification.json` as
diagnostic-only. Their arm-independence and missing admission are why the tool
now prints `UNMEASURED_RANK_COMPLETE_RECORD_GAP`, not a transport verdict.

## Acquisition

The committed launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round229_fold_transport_acquisition/run.sh`.
Its `--preflight-only` mode passes and names the new target
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round229/acquisition/orca2_omt4_fold_transport_10step_np2`.
It records, per rank and immediately after the active stage-1
`tra_adv_trp`, `zFv_after_trp`, `T_Kmm`, `S_Kmm`, `e3t_Kmm`, and `tmask` in a
self-describing stream. It refuses a dirty producer tree, changed source/deck
hashes, a non-additions-only patch, wrong active branch, missing rank, malformed
field header or payload, non-finite values, non-positive live thickness, and
any terminal restart byte movement. It never invokes `/usr/bin/time`.

## Artifacts

The two assembled scenarios and classifier are `omt4_independent.json`,
`omt4_given_nemo_entry.json`, and `classification.json`. Each of the 12 raw
variant JSONs stamps clean measurement commit
`270a521c79c5ee16e679fa95fe7a758e4bab969e`. `classification.json` ends
`HELD_R229_RANK_COMPLETE_TRANSPORT_ACQUISITION_NEEDED`.

## Validation and review

Focused diagnostic and acquisition tests pass 13/13. The acquisition
preflight ends `ORCA2_ROUND229_FOLD_TRANSPORT_PREFLIGHT_READY`. The receipt
citation gate passes 7/7 citations with no unmapped citation; the cumulative
default-receipt gate also passes, and the rigid-shift plant refuses.

The single required `tests/ocean/fidelity -n 12` battery collected 3,130
tests. Before the pytest processes disappeared without emitting their final
summary, it reported 3,099 passed, 7 skipped, and 4 failed, with 20 tests not
reported. The four failures were re-run individually and reproduce the
branch's known pre-existing reds: the GYRE certified-year harness stamp,
round-35 allow-dirty scoping ratchet, report-emitter worktree-stamp ratchet,
and SI3 scalar-math record provenance. The first unreported VORTEX round-205
ID was also re-run alone; its compiler process again disappeared without a
result, so the full battery was not repeated. These facts and the individual
failure traces are retained in `battery.log`, `battery_isolated.log`, and the
`isolated_*.log` files. The evidence manifest is `SHA256SUMS` (manifest digest
`762a99a9ae5ae14e74b0f42e440e55e8d02a968f1cb155b7911cf2934e7c545b`).

The required `codex exec --sandbox read-only` review could not initialize its
app-server client because the sandbox denied its PATH-alias write. The verdict
is **independent review unavailable in-sandbox**, not PASS.

No `packages/` file differs from the round base. GYRE, DINO, tanks, rung 0,
rung 7/rung 10, and OMT-4 therefore cannot execute a changed model statement;
their certified trajectories are unchanged by construction.

## OPEN

1. The operator runs the committed acquisition above; then `--admit-existing`
   must pass its record and additions-only gates.
2. Re-score the full unit against the admitted, rank-complete
   post-`tra_adv_trp` zFv and its T/S/e3t/tmask operands. If zFv is exact and
   the tracer operands are stale, preregister the missing post-stage tracer
   fold exchange. If zFv differs, walk its source-ordered face product and fold
   association before any tracer exchange is proposed.
3. The round-217 unit remains private, OMT-5 remains blocked, and the kt=8
   live-W refusal remains registered debt.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty. No decision is requested this round.

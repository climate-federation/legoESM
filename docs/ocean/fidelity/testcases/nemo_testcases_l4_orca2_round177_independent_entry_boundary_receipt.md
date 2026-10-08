# ORCA2 round 177 — rung-0 independent entry and first downstream boundary

Date: 2026-10-08. Base: `8214fa83a`. Measurement tip:
`b93c92c7d5b22cb2411efb29d4725f58a6c75ada`. Status: **HELD**.

Every trajectory number below is **independent**: both models start from the
rung-0 deck's own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry number appears in these tables.

## Verdict

The hierarchy rung-0 private card now obeys the resolved deck. Its independent
T/S/u/v/ssh entry is bit-exact to NEMO. The corrected 200-row ladder has five
bit-exact entry rows and 195 debt rows; the first debt remains kt=1 stage 1 T.

The rank-complete source-order replay names the earlier boundary behind that
row. NEMO's assignment of the completed external solution to stage sea surface
is the first non-bit statement reached: all 16,433 active columns differ,
maximum 0.1314585958201272 m and RMS 0.007949297911415664 m. The five entry
fields before it are bit-exact. The owner is therefore inside the preceding
split-explicit barotropic solve, not the tracer update.

The old independent-month result is retracted. It no longer becomes non-finite
at step 36. Ninety-five full production steps finish; the synchronized attempt
at step 96 refuses on the model's existing live-thickness invariant,
`raw-mesh e3w_int must contain only finite values > 0`. No step-96 state is
returned, so a first non-finite cell and a 240-step month score remain
**UNMEASURED-with-spec**.

No production model file, threshold, stabiliser, selector, carried-state rule,
or sea-ice field changed.

## Compiled citations

The compiled rung-0 oracle applies the ORCA_R2 regional alteration only inside
the `nn_cfg == 2 .AND. ln_tsd_dmp` branch
(`ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/dtatsd.f90:218-254`). The admitted
rung-0 namelist has that switch off
(`round175/orca2_rung0_stage1_ranked_10step_np2/namelist_cfg:51`). The private
card therefore calls the existing unaltered-file path; the shipped rung-10 card
continues to use the default altered path.

After the external solve, NEMO copies its completed sea surface into the stage
endpoint and constructs the three free-surface ratios before the momentum and
tracer updates
(`ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:139-181`). That exact
compiled assignment is the first replayed non-bit statement. This round does
not claim which statement inside the preceding external solve produces it.

## Controlled change

Only the private hierarchy rung-0 builder was changed. It reconstructs T/S
from the same deck files with `apply_hand_alterations=False` after the shipped
card has been loaded. Tests bind the change to exactly 1,283 active T cells and
720 active S cells; the builder does not replace u, v, ssh, geometry, forcing,
or model configuration. The shipped card remains on its original code path.
Its six sea-ice selectors and `unmeasured_features` tuple are untouched.

Two defects in the existing offline probe were exposed only after the entry
became exact and were corrected before accepting a number: it now retains the
31st W-interface carried by NEMO's 30-level transport record, and it evaluates
the external sea-surface endpoint before constructing the stage ratios. Failed
probe attempts are not evidence.

## Independent entry and ten-step ladder

The fresh ladder completed all 200 rows and printed
`PASS_RUNG0_TEN_STEP_LADDER`.

| checkpoint | field | unequal / stored | RMS | max absolute | verdict |
|---|---:|---:|---:|---:|---|
| kt1 entry | T | 0 / 799,200 | 0 | 0 | bit-exact |
| kt1 entry | S | 0 / 799,200 | 0 | 0 | bit-exact |
| kt1 entry | u | 0 / 799,200 | 0 | 0 | bit-exact |
| kt1 entry | v | 0 / 799,200 | 0 | 0 | bit-exact |
| kt1 entry | ssh | 0 / 26,640 | 0 | 0 | bit-exact |
| kt1 stage 1 | T | 582,469 / 799,200 | 1.1801028691523249e-05 K | 0.0013726778718971544 K | debt |
| kt1 stage 1 | S | 430,551 / 799,200 | 1.0414642920662643e-05 PSU | 0.0011734531121732061 PSU | debt |
| kt1 stage 1 | u | 443,423 / 799,200 | 0.0008585437906091959 m/s | 0.06171520235435013 m/s | debt |
| kt1 stage 1 | v | 440,386 / 799,200 | 0.0008932210107169522 m/s | 0.03402112470518942 m/s | debt |
| kt1 stage 1 | ssh | 16,433 / 26,640 | 0.00624338462385427 m | 0.13145859582012723 m | debt |
| kt10 stage 3 | T | 430,552 / 799,200 | 0.005963034072069885 K | 0.8633902030298275 K | debt |
| kt10 stage 3 | S | 430,552 / 799,200 | 0.0016088940282822529 PSU | 0.4156673855238111 PSU | debt |
| kt10 stage 3 | u | 444,173 / 799,200 | 0.004343087142442177 m/s | 0.3640084122829801 m/s | debt |
| kt10 stage 3 | v | 450,672 / 799,200 | 0.004356804380110146 m/s | 0.5144145634893218 m/s | debt |
| kt10 stage 3 | ssh | 16,433 / 26,640 | 0.02802652392771078 m | 0.42832517646246693 m | debt |

Artifact: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round177/rung0_independent_ladder.json`.

## Source-order stage-1 replay

| boundary | active unequal | RMS | max absolute | verdict |
|---|---:|---:|---:|---|
| entry T | 0 / 430,552 | 0 | 0 | bit-exact |
| entry S | 0 / 430,552 | 0 | 0 | bit-exact |
| entry u | 0 / 413,030 | 0 | 0 | bit-exact |
| entry v | 0 / 413,856 | 0 | 0 | bit-exact |
| entry ssh | 0 / 16,433 | 0 | 0 | bit-exact |
| external stage ssh | 16,433 / 16,433 | 0.007949297911415664 m | 0.1314585958201272 m at `[143,124]` | **first debt** |
| stage r3t | 16,433 / 16,433 | 1.777244766013621e-05 | 0.0007555091713800876 | downstream |

The replay's NEMO-recorded momentum update, barotropic correction, centred
tracer-advection and zero surface-source rows each remain at the floor when
scored over their active masks. Transport and QCO rows after the non-bit stage
surface are downstream and do not own this walk.

Artifact: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round177/stage1_offline_replay.json`.

## Independent month boundary

The month gate uses the production JIT and synchronizes each complete returned
state before inspecting it. Its entry census is bit-exact in all five fields.
Steps 1 through 95 return finite states. Step 96 returns the live-thickness
runtime refusal instead of a state. This **REFUTES R177-P4** and supersedes the
pre-correction step-36 result; it does not convert the refusal into a numerical
non-finite claim.

Artifact: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round177/independent_month_boundary.json`.

## Frozen prediction ledger

| prediction | outcome | evidence |
|---|---|---|
| R177-P1 | CONFIRMED | All five corrected independent entry fields are array-identical. |
| R177-P2 | CONFIRMED | Active old-to-new T/S counts are exactly 1,283/720; only the private T/S replacement path changed. |
| R177-P3 | CONFIRMED | Fresh gate completed 200 rows; kt1 stage1 T remains first debt. |
| R177-P4 | **REFUTED** | The run passes step 36 and reaches the step-96 live-thickness refusal after 95 completed finite states. |
| R177-P5 | CONFIRMED | The replay clears all five entry rows and names external stage ssh. |
| R177-P6 | CONFIRMED | No `packages/` file or shipped/ice selector changed. |

## Mechanical controls and review

Every extended gate has a planted violation. The card test plants the altered
T/S path and metadata mismatch; the stage replay retains its ULP and source-row
plants; the month gate plants the claim label, entry identity, boundary step
and boundary cell. Each plant was observed to refuse.

The required separate `codex exec --sandbox read-only` review was attempted.
It failed before reading the diff with `failed to initialize in-process
app-server client: Read-only file system`; independent review is unavailable
in-sandbox. Log:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round177/codex_review.log`.

Citation controls and pytest results are recorded in the final verification
commit and their logs under the round-177 evidence root.

## OPEN

1. Rerun the existing source-order split-explicit walk on the corrected private
   rung-0 entry, using the admitted rank-complete round-96 barotropic record.
   Start at the external solve's entry/forcing rows and stop at its first
   non-bit statement; the 0.1314585958201272 m endpoint owns the walk.
2. After that earlier boundary is closed or explicitly held, rerun the
   independent month. Attribute the step-96 live-thickness refusal only if it
   remains; the present round supplies no returned step-96 state.
3. A terminal 240-step per-field RMS/max month score remains
   **UNMEASURED-with-spec**: the same production protocol must reach step 240
   and compare with NEMO's admitted rung-0 from-rest month.

The round-96 record already contains the required rank-complete substep stream;
no acquisition is requested.

ASKED choices: honor rung 0's resolved no-damping initial-state branch and
remeasure its independent trajectory.  
UNASKED choices: empty.  
UNVERIFIED assumptions: none presented as facts; the producer inside the
external solve and the step-96 state are explicitly unmeasured.

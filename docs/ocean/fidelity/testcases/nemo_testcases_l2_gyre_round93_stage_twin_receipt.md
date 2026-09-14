# NEMO-testcases L2 GYRE round 93 receipt: stage-twin consolidation held incomplete

Date: 2026-09-14. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Verdict

**HELD; NO PRODUCTION PHYSICS LANDED.** Decision 41's existing Round-46/51
stage harness now drives kt=1 and kt=2 stages through private recorded-entry
seams, publishes the three actual stage outputs, scores explicit entry
identity, includes the external-step handoff, and compares that isolated table
with the ordinary chained program. The final clean artifact is
`round93/round93_stage_twin_final_unmeasured.json`, SHA-256
`91866fa9c2d8af9ccf50a72d9b12ff7f9a6c0449ba1e67c0fbf4b54159c8a4e8`,
at clean commit `ddfb28c3b759573a46a255e219b851ec6df55b7a`.

The gate correctly prints **`STATUS UNMEASURED` and exits nonzero** because 48
required rows across the two tables lack a direct record/trace closure: the
six absolute external endpoint histories at kt=1 and kt=2, stage `zFw`, and
kt=2 direct `zFu/zFv`. More importantly, only T/S/U/V/SSH are currently
injected at a stage boundary; the large isolated kt=2 stage-3 T result proves
that the stage-local TKE/coefficient/content carries are not all closed. No
stage ownership is promoted while those gaps remain.

Among the closed rows, the earliest measured owned stage is **kt=1 stage 1**.
Given NEMO's recorded entry and completed external handoff, T/S/SSH, three
thickness rows, and zFu/zFv are bit-exact; U/V are AT-BAR but non-bit at
`5.421010862427522e-20`, and W is DEBT in 18,000 of 18,000 active cells at
`3.5937485546815465e-8`. This is a measured boundary, not yet an operator-owner
verdict.

## Compiled program and time-level correction

The compiled driver runs one external solve and then stages 1, 2, and 3 with
pointer swaps at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-215`.
Stage 1 builds its RHS in HPG/LDF/VOR/WZV/KEG/ZAD order at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175`.
The source stage update applies the vector Kbb-plus-stage-RHS assignment in
the executing arm at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:365-371`.
The W recurrence first computes horizontal divergence and then integrates the
QCO stretching term from the bottom at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90:277-298`.

The first external-table attempt was **RETRACTED**. The Round-46 stage-one Kaa
bundle is written inside `stp_2D` before the external solve completes; it is
not the post-solve handoff. The corrected instrument reads U/V from the
admitted post-solve barotropic frame and completed SSH from the next-step
entry. NEMO normalizes `un_adv/vn_adv`, `uu_b/vv_b(Kaa)`, and `ssh(Kaa)` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:797-804`.
Its raw b/bb histories rotate in the executing loop at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:759-761`.

No constructible configuration can select either new seam. Both default to
`None`; the ordinary model path does not inspect record files. The stage trace
returns diagnostics only after the production-jitted step and retains the
existing independent ordinary-result observer.

## Stage-entry identity

The final clean run has 30 entry rows: T/S/U/V/SSH for each of three stages at
kt=1 and kt=2. All 30 are BIT (`n_unequal=0`, `max_abs=0`). Thus every state
field this first version claims to inject is proven exact. This does **not**
claim that the five-field bundle is the whole stage entry; the missing
TKE/coefficient/content carries are the reason for the held verdict.

The one-ULP plant artifact
`round93/round93_stage_twin_final_commit_ulp_plant.json`, SHA-256
`707dcdf6c554c52b7ed792542202ef6ab8d95df91fc7773fa79b1bb46f97e95f`,
is stamped to the same commit. It changes one finite kt=1 stage-2 T entry by
one ULP: the T entry row reports exactly one unequal cell while S/U/V/SSH stay
bit-exact. The command exits 1 because every plant exits nonzero.

The stronger preregistered prediction that this one ULP would also change a
stage output is **REFUTED**: it was rounded away before the next published
output. The retained failed-control log says “one-ULP stage-entry plant did not
flip a stage output.” Decision 41 requires the entry row to flip; that actual
control is confirmed.

## Given-NEMO-entry table

Cell format is `class / unequal cells / max abs`; `U` is
`UNMEASURED_WITH_SPEC`. The JSON contains one literal row per field; this
compact receipt table preserves all values while grouping them by stage.

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|
| 1 | external | — | BIT/0/0 ; BIT/0/0 | BIT/0/0 | Hu_avg BIT/0/0 ; Hv_avg BIT/0/0 | — | six histories U |
| 1 | 1 | BIT/0/0 ; BIT/0/0 | AT-BAR/7620/5.4210109e-20 ; AT-BAR/8460/5.4210109e-20 | BIT/0/0 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/3.5937486e-8 | BIT/0/0 ; BIT/0/0 ; U |
| 1 | 2 | BIT/0/0 ; BIT/0/0 | AT-BAR/4787/1.0842022e-19 ; AT-BAR/4600/6.7762636e-20 | BIT/0/0 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/7.9083153e-7 | BIT/0/0 ; BIT/0/0 ; U |
| 1 | 3 | AT-BAR/6858/1.4210855e-14 ; AT-BAR/6455/2.1316282e-14 | DEBT/17399/2.737711e-12 ; DEBT/17100/3.2849221e-12 | AT-BAR/600/4.3368087e-19 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/7.9083159e-7 | BIT/0/0 ; BIT/0/0 ; U |
| 2 | external | — | uu_b DEBT/580/5.039835e-8 ; vv_b DEBT/570/4.800635e-8 | DEBT/600/7.0725601e-7 | Hu_avg DEBT/580/1.2029896e-4 ; Hv_avg DEBT/570/1.0099464e-4 | — | six histories U |
| 2 | 1 | AT-BAR/4493/3.5527137e-15 ; AT-BAR/4436/7.1054274e-15 | DEBT/17400/9.1413673e-6 ; DEBT/17100/9.3268149e-6 | AT-BAR/273/4.3368087e-19 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/7.9083201e-7 | U ; U ; U |
| 2 | 2 | AT-BAR/4551/3.5527137e-15 ; AT-BAR/4497/7.1054274e-15 | DEBT/17400/7.5775705e-6 ; DEBT/17100/9.6194351e-6 | BIT/0/0 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/4.8326762e-7 | U ; U ; U |
| 2 | 3 | DEBT/17988/7.9703416e-3 ; DEBT/15488/1.0668959e-3 | DEBT/17400/2.476002e-3 ; DEBT/17100/5.3033396e-3 | AT-BAR/600/4.3368087e-19 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/4.8326760e-7 | U ; U ; U |

The kt=2 stage-3 tracer values are **not an ownership result**. They are the
discriminator proving that the current five-field entry is incomplete.

## Chained table

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|
| 1 | external | — | BIT/0/0 ; BIT/0/0 | AT-BAR/600/4.3368087e-19 | Hu_avg BIT/0/0 ; Hv_avg BIT/0/0 | — | six histories U |
| 1 | 1 | BIT/0/0 ; BIT/0/0 | AT-BAR/7620/5.4210109e-20 ; AT-BAR/8460/5.4210109e-20 | BIT/0/0 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/3.5937486e-8 | BIT/0/0 ; BIT/0/0 ; U |
| 1 | 2 | AT-BAR/2/3.5527137e-15 ; BIT/0/0 | AT-BAR/6337/1.0842022e-19 ; AT-BAR/5944/8.1315163e-20 | BIT/0/0 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/7.9083153e-7 | AT-BAR/7540/1.8189894e-12 ; AT-BAR/8315/1.8189894e-12 ; U |
| 1 | 3 | AT-BAR/6858/1.4210855e-14 ; AT-BAR/6457/2.1316282e-14 | DEBT/17398/2.737711e-12 ; DEBT/17100/3.2849221e-12 | AT-BAR/600/4.3368087e-19 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/7.9083159e-7 | AT-BAR/6296/3.1832315e-12 ; AT-BAR/5860/2.1600499e-12 ; U |
| 2 | external | — | uu_b DEBT/580/5.0398351e-8 ; vv_b DEBT/570/4.8006350e-8 | DEBT/600/7.0725590e-7 | Hu_avg DEBT/580/1.2029896e-4 ; Hv_avg DEBT/570/1.0099463e-4 | — | six histories U |
| 2 | 1 | DEBT/15470/2.1881164e-11 ; DEBT/11354/1.4992452e-12 | DEBT/17400/9.1389188e-6 ; DEBT/17100/9.3307505e-6 | DEBT/600/2.3575197e-7 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 | DEBT/18000/7.9083201e-7 | U ; U ; U |
| 2 | 2 | DEBT/17993/8.3278006e-7 ; DEBT/17079/6.7607530e-8 | DEBT/17400/7.4535508e-6 ; DEBT/17100/9.6175689e-6 | DEBT/600/3.5362795e-7 | DEBT/18000/1.6484023e-8 ; DEBT/17400/1.5141097e-8 ; DEBT/17100/1.5097498e-8 | DEBT/18000/4.8326762e-7 | U ; U ; U |
| 2 | 3 | DEBT/18000/1.6274972e-4 ; DEBT/17378/6.3277352e-6 | DEBT/17400/1.2295528e-5 ; DEBT/17100/2.3465017e-5 | DEBT/600/7.0725590e-7 | DEBT/18000/2.4725978e-8 ; DEBT/17400/2.2711674e-8 ; DEBT/17100/2.2646248e-8 | DEBT/18000/4.8326760e-7 | U ; U ; U |

## Frozen predictions and refutations

1. The kt=1 chained U/V predictions are confirmed at all three stages,
   including stage 3's `2.7377110452773967e-12` and
   `3.284922138989399e-12`. Stage-1 T/S/SSH are bit-exact as predicted.
2. The first measured owned stage prediction, kt=1 stage 1, is confirmed for
   the closed rows. The additional W debt is larger than its U/V last-bit
   misses; operator ownership remains withheld pending full entry/output
   closure.
3. The one-ULP **entry-row** sensitivity is confirmed. The stronger predicted
   output sensitivity is refuted and retained.
4. Non-interference is confirmed: the certified trajectory values and day-30
   statistic equal the Round-85 comparator exactly.

Three development attempts are retained but not cited scientifically:
the first failed on the W-interface extent, the second on a double crop of an
already-owned transport payload, and the third exposed the 31-level NEMO
bottom sentinel. A later complete table using the pre-solve Kaa scratch is
explicitly retracted by the corrected time-level run.

## Rule 12 and magnitude rows

The certified ladder artifact `round93/round93_ladder_kt1_10.json`, SHA-256
`940e6867be20a6156778f4bef9f5aa05e56a0750a566bb7028f38f0f13988c7a`,
is stamped clean and reproduces the before arm:

| row | before | Round 93 | disposition |
|---|---:|---:|---|
| kt2 T | 1.4210854715202004e-14 | 1.4210854715202004e-14 | AT-BAR retained |
| kt2 S | 2.1316282072803006e-14 | 2.1316282072803006e-14 | AT-BAR retained |
| kt2 U | 2.7377110452773967e-12 | 2.7377110452773967e-12 | first-over-bar unchanged |
| kt2 V | 3.284922138989399e-12 | 3.284922138989399e-12 | first-over-bar unchanged |
| kt3 T | 1.627497246303733e-4 | 1.627497246303733e-4 | unchanged |
| kt3 S | 6.327735185607253e-6 | 6.327735185607253e-6 | unchanged |

The 30-day member completed all 30 days. The day-gap artifact
`round93/round93_day_gap.json`, SHA-256
`67f748fedac2e4f2ddce3d24b33187590515c80f8db608b6adf9ded78ae1c26c`,
reports day-30 T RMS `1.2397011295506804e-2 K`, exactly the immutable
Round-85 value.

| testcase | disposition |
|---|---|
| GYRE kt=1--10 | No production candidate; every headline row unchanged and first-over-bar remains kt2 U/V |
| GYRE days 1--30 | No production candidate; day-30 T RMS unchanged |
| LOCK_EXCHANGE-zco | Private hooks are off and unconstructible by the card; no numerical statement changed |
| OVERFLOW-zps | Private hooks are off and unconstructible by the card; no numerical statement changed |
| DINO | Shared WS-RK3 risk remains explicit; no numerical neutrality claim, and the 96--98% regional cancellation warning remains in force |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record every native selected-integrator stage entry/output, external history, transport, thickness and content/TKE carry; require the same paired tables and ULP plant; reject missing rows, AT-BAR loss, or earlier first-over-bar |

No configuration, coefficient, timestep, stabilizer, carried-state policy,
year harness, reconciliation gate, freshwater pair, #1484 guard, NEMO source,
or NEMO executable changed.

## Review, citations, and tests

The required separate command was attempted with `codex exec --sandbox
read-only`. Independent review is unavailable in-sandbox. Its terminal result,
quoted verbatim, is: **“Error: failed to initialize in-process app-server
client: Read-only file system (os error 30)”**. The log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
Absence of a verdict is not approval; no physics is landing.

The focused stage, phase-3, year-member, and year-owner suite passed 73 tests
before the final fail-closed disposition; the final stage-specific suite then
passed 11 tests at `ddfb28c3b759`. After the mechanical citation re-anchors,
the citation-gate and citation-metadata suites passed 21 tests. The normal
receipt citation gate passed all six compiled-source citations with no global
map failures; its shifted stp2d range plant exited 1 with
`SYMBOL-NOT-AT-LINE`. The initial default-heading invocation was inapplicable
to this standalone receipt and retained no scientific result. The known unrelated
`test_rk3_ws_differs_from_rk3_and_is_finite` failure was not encountered.

The GitHub #1455 read/post path is unavailable because this shared clone has no
remote recognized by `gh`; the retained log says “none of the git remotes
configured for this repository point to a known GitHub host.” No ledger-post
claim is made.

## OPEN for round 94

1. Extend this same Round-46 gate, not a second harness, to consume the existing
   Round-81 external-step and Round-75 transport records. Publish and score all
   six absolute endpoint histories plus direct kt=1/2 zFu/zFv/zFw. Remove an
   `UNMEASURED_WITH_SPEC` row only when its direct record and a red plant close.
2. Extend the stage-entry seam and identity table with every stage-local field
   the next stage consumes, beginning with the Round-59 TKE operands and the
   tracer content/coefficient carries. The kt=2 stage-3 T discriminator must
   collapse from `7.9703416e-3 K` before ownership can be assigned.
3. Rerun both complete tables. Only after every entry and output row is closed
   may the earliest owned stage be promoted. Then walk that stage in compiled
   order. The current magnitude lead is kt=1 stage-1 W
   (`3.5937485546815465e-8`); its compiled producer is the `wzv` call and QCO
   recurrence cited above, but it is not yet a landing candidate.
4. Preserve the immutable Round-85 ladder and day-30 before arm. No downstream
   stage or held manifest patch is eligible while the kt=1 stage-1 closure is
   incomplete.

No NEMO acquisition or user configuration decision is needed: the named
Round-81, Round-75, Round-59, and year-owner records already exist.

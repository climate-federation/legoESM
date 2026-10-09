# ORCA2 round 190: independent month growth table

Date: 2026-10-08

Status: **HELD**

Every scientific number below is **independent**: hierarchy rung 0 starts from
its corrected, bit-exact initial state. No given-NEMO-entry rung-7 number is
mixed into the table. Sea ice, the shipped rung-10 card and its
`unmeasured_features` declaration are untouched.

## Outcome

The bounded round-189 record admits. The production run completes 95 finite
steps, then reaches its registered live-thickness refusal at step 96. The
completed table confirms two distinct growth boundaries:

- first coarse greater-than-ten interval: exact entry to step 10, with maximum
  T error `0.8633902030298275 K`, or `4316951015.1491375` times the fixed
  `2e-10` floor;
- explosive interval: step 90 to 95, where the largest error moves from
  `13.662396746636144` to `9.742145311480133e22 m/s` in V, a factor
  `7.13062685277294e21`.

The certified ten-step replay still refines the first interval to kt=1 stage 1
(SSH maximum `0.1314585958201272 m`). Its first source-ordered non-bit
statement remains NEMO's vector-invariant vertical average of the completed
three-dimensional momentum RHS at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219`. The compiled branch
selects vector-invariant form at lines 207-212 and evaluates the complete
vertical products and reference-depth scales before drag, wind and the
split-explicit solve.

## Complete independent growth table

Each field cell is `RMS / maximum absolute error`; units are K, PSU, m/s, m/s
and m for T, S, u, v and SSH. The last column is the largest field error and
its ratio to the prior row's largest maximum (the fixed floor at entry).

| step | T | S | u | v | SSH | largest / growth |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.005963034072069885 / 0.8633902030298275 | 0.001608894028282253 / 0.4156673855238111 | 0.004343087142442177 / 0.3640084122829801 | 0.004356804380110146 / 0.5144145634893218 | 0.02802652392771078 / 0.4283251764624669 | T 0.8633902030298275 / 4.316951015e9x |
| 20 | 0.00987585481794429 / 1.261521774806628 | 0.002928313490597025 / 0.9194557187693562 | 0.005481394599912885 / 0.6293770362328093 | 0.00580693369293246 / 1.05782949517034 | 0.04445133691490295 / 1.545557305663133 | SSH 1.545557305663133 / 1.790102899x |
| 30 | 0.01351376546328976 / 1.360888982933254 | 0.004186806067397323 / 1.351918116520221 | 0.006027504898305049 / 0.6355854975360292 | 0.00637059118308224 / 1.136913620343139 | 0.05740879136897435 / 1.378214182442641 | SSH 1.378214182442641 / 0.891726355x |
| 40 | 0.01565010326577977 / 1.267347820066885 | 0.005423826512969751 / 1.71377678230408 | 0.006681873273182129 / 0.9212215054119469 | 0.00797981568868069 / 1.691869463965564 | 0.06731136578573246 / 1.99351376712747 | SSH 1.99351376712747 / 1.446446998x |
| 50 | 0.01807435502408221 / 1.27996752244063 | 0.006545663398795846 / 2.000974454209619 | 0.008297504804495875 / 1.184201109488126 | 0.009288706645736704 / 2.135858885371078 | 0.0787014085676689 / 1.53371318751082 | v 2.135858885371078 / 1.071404131x |
| 60 | 0.02025315726387814 / 1.443214263163405 | 0.007550161882639453 / 2.217040501779255 | 0.01159476170746118 / 2.571345045830754 | 0.01043516060979974 / 2.229907953578703 | 0.09765707518140823 / 2.827831025153025 | SSH 2.827831025153025 / 1.323978398x |
| 70 | 0.02243915306540838 / 1.526465386415985 | 0.008474220517215752 / 2.597241922388932 | 0.01628381983946546 / 3.667095702522686 | 0.01578035824875468 / 2.870538289548756 | 0.132246524733757 / 6.130576756648487 | SSH 6.130576756648487 / 2.167943099x |
| 80 | 0.02457378148214106 / 1.570923137277458 | 0.009341285768616989 / 2.962819511008309 | 0.0245814823088157 / 4.644377470935702 | 0.02784702512231768 / 5.138514242893315 | 0.1713783081140869 / 5.643745271486383 | SSH 5.643745271486383 / 0.920589611x |
| 90 | 0.02759610325434037 / 1.593282171809502 | 0.01014164729812383 / 3.288828592393525 | 0.0405445834652363 / 7.58260447053665 | 0.04393552856835616 / 7.203813998535762 | 0.2460647717119017 / 13.66239674663614 | SSH 13.66239674663614 / 2.420803224x |
| 95 | 2.225902112215362e13 / 1.318394162777601e16 | 2.7092872600454e13 / 1.521497290342251e16 | 1.311569048126625e20 / 6.844821468771605e22 | 1.916085031289751e20 / 9.742145311480133e22 | 18.10369355230873 / 1211.09080617589 | v 9.742145311480133e22 / 7.130626853e21x |

Step 95 is finite. Step 96 is not scored: it refuses during the ordinary stage
program and the guard-bypassed replay finds the after-SSH field globally
non-finite, first deterministic live-thickness interface `[j=1,i=49,k=0]`.

Machine artifacts:

- `round190/month_growth_complete.json`, SHA-256
  `d7bc3e49878a094196fed93c8f08f79a8bbd11844ee08ca8dacc6a90db0cb41c`;
- `round190/growth_table.json`, SHA-256
  `960bfef80dfb0bdd39bf309b4f9ddb4a42be379645af9520e00e7b029ab0bd70`.

## Source-statement landing arm: exact, then HELD

The arm materialised NEMO's multiply/add/final-multiply association. Against
the admitted slow-forcing record, both active depth means close bit-for-bit: U
`0/15,789` unequal and V `0/15,875`. Drag, wind, final forcing and SSH RHS also
remain exact. The first downstream non-bit boundary is the split-explicit
`ssh_after` output.

The independent ten-step ladder makes the arm ineligible under Decision 96:

| measure | toward | away | score-equal |
|---|---:|---:|---:|
| RMS | 87 | 95 | 13 |
| maximum absolute | 49 | 69 | 77 |

All 195 non-entry rows move at the bit level; none of the five exact entry rows
leaves the bar. First-debt T RMS moves toward by `6.776e-21 K`; kt=10 stage-3
SSH maximum moves toward by `2.287e-14 m`. But Decision 96 requires a majority
of RMS-moved rows toward NEMO, and `87 < 95`. This rung-0 refusal is
dispositive, so the larger GYRE/year, DINO and tank landing gates were not run.
Production is restored exactly; the final tree has no `packages/` diff from
the round base. The tested arm remains visible in commits `071670e58` and
`c139fdb7d`, while its machine evidence is retained under `round190/`.

## Frozen prediction ledger

| prediction | outcome | evidence |
|---|---|---|
| R190-P1 | **CONFIRMED** | Ten scientific checkpoints, two ranks, two exact twins and a separate exact step-96 sentinel admit. |
| R190-P2 | **CONFIRMED** | Entry to step 10 is the first strict greater-than-ten coarse interval. |
| R190-P3 | **CONFIRMED** | Step 95 is finite and its largest error grows by `7.13062685277294e21` from step 90. |
| R190-P4 | **CONFIRMED** | Passive replay and the exact operand arm retain the completed-RHS vertical average as the first source statement. |
| R190-P5 | **CONFIRMED** | Step 95 is finite; the step-96 refusal is downstream of the kt=1 debt. |
| R190-P6 | **CONFIRMED** | All six table plants and all record plants fire. |

## Validation and review

The complete-table gate passes with `HELD_FIRST_STATEMENT_SHARED_GATE`; its six
real-input plants each exit 2 at `STATUS PLANT-FIRED`, and its focused tests
pass 7/7. The source-statement unit test passed before the held arm was
restored.

The final-tree citation gate passes both its default receipt (zero failures,
zero unmapped citations, zero audit failures) and this receipt from `##
Outcome`.  Its real two-line shift of the rendered
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219` citation exits 1 and
reports `FAIL`, so the control fires.

The one required wide battery completed in 2,357.90 s: **2,885 passed, 7
skipped, 4 failed**.  All four failures are registered pre-existing reds, not
round-190 regressions: the SI3 scalar-math provenance gate (`A MY_SRC is not
verbatim`), the allow-dirty scope ratchet (13 inherited VORTEX drivers), the
worktree-stamp ratchet (13 inherited report emitters), and the stale GYRE
spread-floor record (`certified year harness moved`).  The durable transcript
is `orca2_rounds/round190/fidelity_battery.log`.

A separate `codex exec --sandbox read-only` review was attempted and returned
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.  The transcript is
`orca2_rounds/round190/independent_review.log`.

No configuration choice was made. The floor, fields, cadence and statistics
are exactly preregistered. No stabiliser was added. The final tree changes only
measurement code and receipts.

## OPEN

1. The source-exact completed-RHS vertical average remains **HELD** because it
   exposes a downstream compensating error on 95 RMS rows. Do not land it
   alone.
2. Walk the first downstream non-bit boundary inside the split-explicit solver
   from exact slow forcing: `ssh_after`, then U/V external mode, in compiled
   `dynspg_ts` substep order and by offline replay only.
3. Re-test the held halo/V-transport unit only after that compensating
   split-explicit statement is named. No partial operand may land.

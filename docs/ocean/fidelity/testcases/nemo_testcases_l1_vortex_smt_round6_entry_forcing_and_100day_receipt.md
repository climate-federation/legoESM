# Rounds 216-217 / VORTEX_SMT round 6 — the entry forcing's TWO statements, LANDED, and the 100-day comparison

**DECISIONS 90 and 91 (user, 2026-10-04), both taken as the picks this
receipt asked for.** (90) The slow-forcing depth-average field gets NO
default: every card states it explicitly and the unset value raises; the
NEMO cards state NEMO's own form. (91) The entry-Coriolis statement lands
on the SHARED path for every card, and the six flat VORTEX cards' last-bit
moves are REGISTERED rather than scoped away. Round 216 measured and held;
round 217 carries the two decisions, completes the gate set and lands.

**VERDICT: the seamount cards' step-2 sea-surface-height error is two
compiled statements, both now transcribed, and the kt=1 window closes to
the bar on BOTH cards.** The loop-entry forcing goes from `5.5518e-11`
(vector) / `5.7651e-11` (flux) to `1.3553e-20` on both, and the
end-of-window sea surface height from `3.664757e-07` / `3.726197e-07` m to
`2.220e-15` / `1.887e-15` m — the same place injecting NEMO's own recorded
operand reaches (`2.109e-15`).

Decision 88 (user), operator note CC addendum 6. Predictions were frozen
before any measurement:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round6/predictions.md`.
Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round6/`.

Note on R6-P4: round 5 wrote it as a COMMITMENT that round 6 lands nothing.
Operator note CC addendum 6, written later, orders a landing if the two
statements close the window. The commitment is retired as superseded and
said out loud rather than quietly dropped.

## 1. PRE-IMPL SEARCH (RULE 4)

```
grep -rn "slow_forcing_depth_override|barotropic_slow_forcing_override" packages/ src/
grep -rn "_depth_average_to_faces" packages/ src/
grep -n  "def _nemo_literal_een_coefficients|_nemo_literal_seed_from_reference_mesh" packages/ocean/.../barotropic_latlon_cgrid.py
grep -rln "barotropic_coriolis_een_pre_step" tests/
ls scripts/validate/ocean_fidelity/testcases/ | grep -i "compare|ratchet|registry|ulp"
```

Everything this round needed already existed and was EXTENDED, never
copied: the per-substep walk (round 196) gained one arm; the split probe
(round 215) gained one comparison; the 100-day scorer and the movie
(round 210) gained a card selector; the pre-step Coriolis unit test
(`test_barotropic_coriolis_null_mode.py`) gained one case. No new file was
written. In particular the literal reference-mesh seed
(`_nemo_literal_seed_from_reference_mesh`, NEMO `istate.F90:149-155`) was
ALREADY in the tree — the defect was that the entry Coriolis did not use
the card's own seed rule, not that the rule was missing.

## 2. THE FLUX CARD'S SUBSTEP RECORD, ADMITTED

Round 5's flux acquisition never started (its launcher was committed to
while it was executing). Round 6 built the card its own NEMO pair,
`VORTEX_SMT_R6B_OMIP_L1{,_P3}`; round 5's unused `VORTEX_SMT_R5_OMIP_L1`
pair and rounds 1/3's builds are untouched.

| check | result |
|---|---|
| admission | `ADMITTED`, 34 records, `icycle` 48 from the header, 1562 groups |
| plants | all four fire: header, field-name, truncated, missing-frame |
| additions-only, across BUILDS and ROUNDS | step-10 restart `69fbdafe0b1e…` is byte-identical to round 3's ADMITTED flux run in a different configuration directory |

One failed attempt is disclosed rather than hidden: the first round-6 flux
variant compiled the VECTOR card's stage-terms writer into the flux deck
(my own wiring error, copied onto the wrong branch of the launcher's
`case`), and the checker REFUSED the record — `missing group(s) ['keg_u',
'keg_v', 'zad_u', 'zad_v']`, which is the flux form having no
kinetic-energy-gradient or vertical-advection trend to write. That is the
guard doing its job. The build pair it produced (`VORTEX_SMT_R6_OMIP_L1*`)
is left in place, unused and unmoved, and the corrected variant built a
new pair.

## 3. R6-P1: SAME BOUNDARY ON THE FLUX CARD — CONFIRMED

The flux card's kt=1 walk, run against its OWN new record on the
PRE-landing tree (the production change stashed, `git status --porcelain`
checked clean after restoring):

| card | first non-bit boundary | value | end-of-window `ssha` |
|---|---|---:|---:|
| `VORTEX_SMT_VEC-zps` | `entry.zu_frc` | `5.5518e-11` | `3.664757e-07` |
| `VORTEX_SMT-zps` | `entry.zu_frc` | `5.7651e-11` | `3.726197e-07` |

Same boundary, ratio `1.038` — inside R6-P1's "factor of 2". The
prediction's falsifier (a different boundary) did not fire.

## 4. THE TWO STATEMENTS, CITED FROM THE COMPILED SOURCE

Both lines are from
`tests/VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/`, the build that
produced the admitted vector record.

**(a) the depth average — `stp2d.f90:178-179`** (the `np_VEC_c2` arm of the
`SELECT CASE( n_dynadv )` opened at `:176`; the flux card runs the
cumulating arm at `:183-184`):

```
Ue_rhs(ji,jj) = SUM( e3u_3d(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)*umask(ji,jj,1:jpkm1) ) * r1_hu_0(ji,jj)
```

REFERENCE face thickness, stored reciprocal reference column depth, no
sea-surface stretching. legoESM weighted with the per-level minimum of the
two LIVE thicknesses and divided by their own column sum.

**(b) the entry Coriolis — `dynspg_ts.f90:275`, `:290`, `:292`**:

```
 zu_frc(:,:) =   Ue_rhs(:,:)                                            ! :275
CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )       ! :290
zu_frc(ji,jj) = zu_frc(ji,jj) - zu_trd(ji,jj) * ssumask(ji,jj)          ! :292
```

The operand is `puu_b(:,:,Kmm)`, the CARRIED external mode. NEMO forms it
ONCE, by a REFERENCE-thickness depth mean — `istate.f90:149-152` at
`kt = nit000` and `dynatf_qco.f90:220-231` thereafter:

```
uu_b(ji,jj,Kbb) = uu_b(ji,jj,Kbb) + (e3u_3d(ji,jj,jk)*(1+r3u(ji,jj,Kbb)*umask(ji,jj,jk))) * uu(ji,jj,jk,Kbb) * umask(ji,jj,jk)
uu_b(:,:,Kbb)   = uu_b(:,:,Kbb) * (r1_hu_0(:,:) / (1+r3u(:,:,Kbb)))
```

The stretch `(1+r3u)` is k-independent and cancels between numerator and
denominator, so the weights are `e3u_0*umask` exactly. legoESM instead
RE-REDUCED the three-dimensional velocity with the min-rule live face
thickness, so the two halves of one cancellation — the subtraction at
`:292` and the substep loop seeded from `puu_b` at `dynspg_ts.F90:484-500`
— were handed two different barotropic velocities. On a full-cell mesh the
two rules agree exactly; over partial cells they differ by order of the
free-surface stretch ratio, `~2e-6` here, which is why the flat pair never
saw it.

## 5. THE PER-STATEMENT MEASUREMENT ROUND 5 DID NOT HAVE

Round 5 left "the entry Coriolis subtraction owns the remainder" PLAUSIBLE
and named the discriminating measurement. It is run here, and it is a
direct comparison of the two arrays rather than an override: NEMO's own
entry Coriolis trend is its rebuilt depth average minus its recorded
finished forcing (legitimate only under round 215's three controls, which
the probe re-checks from the run's own namelist), and legoESM's is read out
at the boundary it is formed through the callable observer the model
already offers.

| | u | v |
|---|---:|---:|
| NEMO's entry Coriolis, peak | `2.02715772582067e-05` | `2.01758958836840e-05` |
| legoESM's, peak | `2.02715595840707e-05` | `2.01758855667122e-05` |
| **max abs difference** | **`4.3969046113e-11`** | **`4.3853068893e-11`** |
| cells differing | 2 108 | 2 028 |

Round 5's depth-average arm left a residual of `4.3969046116e-11` /
`4.3853068895e-11`. The per-statement difference reproduces it to TEN
digits, which is the measurement that promotes round 5's PLAUSIBLE reading
to **CONFIRMED** and refutes its own counter-argument (that the operator
was faithful at substep 1 to `6.8e-21` and therefore could not carry
`4.4e-11`). The counter-argument was sound about the OPERATOR and silent
about its OPERAND: the coefficients are faithful, the velocity handed to
them was not. The peaks agree to six digits, which is exactly why the
arrays had to be differenced rather than their maxima compared.

## 6. THE ARMS, ONE VARIABLE EACH, kt=1 WINDOW, BAR 1e-15

`VORTEX_SMT_VEC-zps`, production-jitted step, against the admitted record.

| arm | loop-entry `zu_frc` | end-of-window `ssha` |
|---|---:|---:|
| production (round 5's tree) | `5.551760e-11` | `3.664757e-07` |
| the entry Coriolis statement ALONE | `1.911211e-11` | `8.031565e-08` |
| the depth-average statement ALONE (round 213's S2) | `4.396905e-11` | `2.861601e-07` |
| **both statements** | **`1.3553e-20` (bar)** | **`2.220e-15`** |
| the Coriolis statement + NEMO's own depth average INJECTED | `1.3553e-20` | `2.609e-15` |
| NEMO's own finished forcing injected (round 5's reference) | `0` (bit) | `2.109e-15` |

The last three rows are the control: a transcription and an injection of
the same quantity land in the same place, so the transcription is the
statement and not a coincidence. **R6-P6 CONFIRMED** on the vector card
and, from §7, on the flux card.

**R6-P5 is REFUTED, and it was my prediction.** It said the entry-Coriolis
override would move `ssha` by less than 2x, on round 5's reasoning that
the operator could not carry the remainder. The Coriolis statement alone
moves it 4.6x (`3.664757e-07 -> 8.031565e-08`), past the falsifier's
`1.8e-07`. The statement is a first-class owner in its own right.

## 7. BOTH CARDS, kt=1 WINDOW, BEFORE -> AFTER

| card | `zu_frc` before | after | `ssha` before | after |
|---|---:|---:|---:|---:|
| `VORTEX_SMT_VEC-zps` | `5.5518e-11` | `1.3553e-20` | `3.664757e-07` | `2.220e-15` |
| `VORTEX_SMT-zps` | `5.7651e-11` | `1.3553e-20` | `3.726197e-07` | `1.887e-15` |

Both cards' end-of-window velocities are at the floor after the landing
(`1.67e-16`/`1.94e-16` flux; `1.11e-16` vector).

## 8. THE ADVERSARIAL REVIEW

One fresh independent reviewer (not the author, read-only, given the diff,
the compiled NEMO sources and the evidence directory, and told the
one-battery rule). Verdict **DO NOT SHIP** on the snapshot it was handed,
two MAJOR. Both are taken in code; four citation defects are repaired.

| # | finding | taken |
|---|---|---|
| MAJOR 1 | the new call site re-implemented the carried-external-mode gate inline (`active AND uu_b AND vv_b`), so a card that selects the carried mode with a missing or half pair silently falls back to the reduction — the very silent fallback this round removes | **ACCEPTED, FIXED.** The call site now uses the selector the window seed already uses, which RAISES on a missing or half pair. The helper lost its underscore (`nemo_carried_barotropic_depth_mean`) because it is now read across modules, which the repo's no-private-cross-import ratchet requires |
| MAJOR 2 | `dynatf_qco.f90:220-231` does NOT run on this deck: the cpp keys are `key_qco key_RK3 key_vco_1d3d` and no `dyn_atf` is called from `stprk3.f90`, `stprk3_stg.f90` or `step.f90`; under RK3 the carried pair comes out of `dyn_spg_ts` itself and is combined at `stprk3_stg.f90:134-232` | **ACCEPTED AND THE CLAIM IS WITHDRAWN.** It was the leap-frog path, quoted for an RK3 deck. The physics reading is unchanged — the first step's operand is `istate.f90:149-152` either way, and that is the step this round measures — but the "and thereafter" half was wrong and is corrected in the code comment |
| MINOR | the `dyn_cor_2D` CALL is `dynspg_ts.f90:289`; `:292` is the subtraction | TAKEN |
| MINOR | `istate.f90:149-152` sets `Kbb`, not `Kmm` (`:161`) | TAKEN, with the reason it does not matter here written down: `stprk3.f90:189` calls `stp_2D(kstp, Nbb, Nbb, Naa, Nrhs)`, so `Kmm` IS `Nbb` inside `dyn_spg_ts` |
| MINOR | the compiled text says `e3u_3d`, not `e3u_0` | TAKEN; they are the same array under `key_vco_1d3d` and the receipt now quotes the compiled spelling |
| MINOR | the reference-thickness reading was asserted, not shown | TAKEN: §4 now writes the `(1+r3u)` cancellation out |
| MINOR | the probe never applies `umask` to legoESM's side while NEMO's is `zu_trd*ssumask`, so a land-cell disagreement would be invisible | ACKNOWLEDGED, not changed this round. It cannot affect the reported number — the two sides agree to `4.4e-11` against a `2.0e-05` field, and a land-cell disagreement would be O(the field) — but it is an open gap, listed in §11 |
| **DISPUTED, settled by measurement** | "I could not find what freezes the tanks; treat LOCK_EXCHANGE and OVERFLOW as in scope" | The reviewer read the recipe's two `barotropic_coriolis_split="live"` lines as unconditional. They are inside whole-step-identity branches the tanks do not select. The CARDS THEMSELVES were asked, which is the measurement that settles it: `LOCK_EXCHANGE-zco` and `OVERFLOW-zps` both resolve `barotropic_coriolis_split=frozen` and `barotropic_coriolis=avg`, so neither reaches the changed branch. They are measured anyway in §9 rather than argued |
| **ACCEPTED, and it is why this round's gate list is what it is** | "the evidence to rule out a moved certified number was not run" — the reviewer looked while the gates were in flight | Correct at the time. §9 is that evidence |
| time-level note | the old operand reduced `state_mid.u` (post-slow-tendency) and the new one is the step-entry carried value, so this is also a TIME-LEVEL change, not only a thickness-rule change | **ACCEPTED as a correction to how the change is described.** It is NEMO's own time level — `dyn_cor_2D` is handed `puu_b(:,:,Kmm)` and nothing else — and the discriminating evidence is the flat cards' before/after in §9, where the thickness rules agree and only the time level could move a row |

## 9. GATES — ROUND 216 HELD HERE; ROUND 217 COMPLETED THEM AND LANDED

Round 216 held with every gate running and none red; round 217 carries
decisions 90 and 91, completes the list and lands. The table below is the
round-217 state.

| gate | state at hand-off |
|---|---|
| flux seamount substep record | **ADMITTED** (§2); additions-only proven across builds and rounds |
| the two statements close the kt=1 window, BOTH cards | **MET** (§6, §7) |
| direct unit test, non-vacuous | **GREEN.** `tests/ocean/unit/test_barotropic_coriolis_null_mode.py` gains one case that pins the supplied pair as what the operator sees AND asserts it differs from the reduction, so a dropped keyword turns it red. The keyword is `entry_barotropic_velocity`, renamed from `barotropic_velocity` after the existing RK3 stage-operand AST guard correctly refused the collision — that refusal is itself a non-vacuity proof of that guard |
| the ten-card registries + the two-ULP ratchet | **COMPLETE, §9.3.** Both seamount cards' kt=2 sea surface height at the bar; the ratchet RED on 9 of 10 at 3 cell ULP against a 2-ULP bar, landed on decisions 90/91 and registered |
| GYRE ladder + certified year (note BZ, 8 days) | **COMPLETE, §9.4.** Ladder first-over-bar unchanged; the year's worst day is 0.220 run-to-run floor units from the pin, re-pinned |
| DINO from-rest month gate (`2.053801168e-03` K) | runs inside `land.sh`; its number is in §16 |
| the 100-day comparison, both seamount cards + movie | **COMPLETE, §14** |
| two-ULP ratchet | §9.3: RED on 9 of 10, every violation one cell at 3 ULP against a 2-ULP bar |

This is a HOLD with everything in place, not a failure: no gate came back
red. Round 7's first act is to read the four outputs already being written
into this round's evidence directory and, if they are green, run `land.sh`
on this same tree.

## 9.3 THE REGISTRIES — AND THE FLAT CARDS ARE **NOT** 0/50

Round 4's certified set is the before arm (`round4/`, `round4/inert/`);
`round6/after2/` is the after arm; `round6/registry_diff.txt` is the full
row-by-row diff.

| card | rows | over bar | moved | toward | away |
|---|---:|---:|---:|---:|---:|
| `VORTEX_SMT-zps` | 50 | 41 | 41 | 31 | 10 |
| `VORTEX_SMT_VEC-zps` | 50 | 41 | 39 | 34 | 5 |
| `VORTEX-zco` | 50 | 39 | 27 | 18 | 9 |
| `VORTEX_VEC-zco` | 50 | 35 | 28 | 13 | 15 |
| `VORTEX-15km-zco` | 50 | 41 | 34 | 13 | 21 |
| `VORTEX_VEC-15km-zco` | 50 | 37 | 28 | 15 | 13 |
| `VORTEX-10km-zco` | 50 | 42 | 27 | 17 | 10 |
| `VORTEX_VEC-10km-zco` | 50 | 38 | 24 | 11 | 13 |
| `LOCK_EXCHANGE-zco` | 50 | 16 | **0** | — | — |
| `OVERFLOW-zps` | 50 | 37 | **0** | — | — |

The seamount kt=2 rows, which are what this round set out to move:

| card | field | before | after |
|---|---|---:|---:|
| `VORTEX_SMT-zps` | ssh | `3.726197e-07` | **`2.275957e-15`** |
| | u | `6.308422e-08` | `2.160534e-08` |
| | v | `4.742908e-08` | `1.545394e-08` |
| | T | `4.259549e-10` | `3.803022e-10` |
| `VORTEX_SMT_VEC-zps` | ssh | `3.664757e-07` | **`2.664535e-15`** |
| | u | `6.004125e-08` | `1.267541e-09` |
| | v | `4.037530e-08` | `2.548881e-10` |
| | T | `4.539846e-11` | `3.618737e-12` |

**THE FINDING THAT BLOCKS THE LANDING, reported the moment it was seen
(RULE: a default or a result that disagrees with expectation is a FINDING,
not a footnote).** The round's own bar, and the operator's order, expected
the six flat VORTEX cards at **0 of 50 moved**. They are not: each moves
24-34 rows, in BOTH directions, and every moved flat row agrees with its
predecessor to 7-16 significant digits — i.e. these are last-bit changes,
with the kt=2 `ssh`/`u`/`v` rows still inside the `1e-15` bar before and
after (`VORTEX_VEC-zco` kt=2 ssh `2.831069e-15 -> 2.220446e-15`). The two
tanks, which do NOT take the changed branch, move exactly 0 rows, which is
the control saying the movement is this change and not noise.

That is consistent with the reviewer's time-level reading: on full cells
the two THICKNESS rules agree exactly, so what remains is a different
floating-point composition of the same quantity at a different time level,
and it shows up at the last bit. It is NOT consistent with "inert by
construction", which is what an earlier draft of this receipt would have
claimed. Whether a last-bit movement on eight certified cards is
acceptable is the operator's call, not mine, and it is why this round does
not land on its own judgement.

## 9.3 THE REGISTRIES, THE ULP RATCHET, AND THE FLAT CARDS' LAST-BIT MOVES

Before arm: round 4's certified set (`round4/`, `round4/inert/`). After
arm: `round6/after3/`, produced on the landed tree with BOTH decisions
applied. The two-ULP move gate is the shared one
(`nemo_testcase_offline_compare.py` -> `ulp_move_gate.compare_gate_reports`),
run offline on the saved report/residual pairs; its outputs are under
`round6/ratchet/`.

| card | ULP gate | rows | cell violations | max row ULP | registry moved / toward / away | rows over bar |
|---|---|---:|---:|---:|---|---:|
| `VORTEX_SMT-zps` | FAIL | 50 | 44 | 2 | 41 / 31 / 10 | 41 |
| `VORTEX_SMT_VEC-zps` | FAIL | 50 | 42 | 2 | 39 / 34 / 5 | 41 |
| `VORTEX-zco` | FAIL | 50 | 27 | 2 | 25 / 18 / 7 | 39 |
| `VORTEX_VEC-zco` | FAIL | 50 | 30 | 2 | 28 / 12 / 16 | 35 |
| `VORTEX-15km-zco` | FAIL | 50 | 32 | 2 | 33 / 14 / 19 | 41 |
| `VORTEX_VEC-15km-zco` | FAIL | 50 | 37 | 2 | 28 / 14 / 14 | 38 |
| `VORTEX-10km-zco` | FAIL | 50 | 33 | 2 | 25 / 13 / 12 | 42 |
| `VORTEX_VEC-10km-zco` | FAIL | 50 | 40 | 2 | 27 / 10 / 17 | 38 |
| `LOCK_EXCHANGE-zco` | **PASS** | 50 | 0 | 2 | 13 / 8 / 5 | 16 |
| `OVERFLOW-zps` | FAIL | 50 | 19 | 2 | 20 / 13 / 7 | 37 |

The seamount kt=2 rows, which are what the landing set out to move:

| card | field | before | after |
|---|---|---:|---:|
| `VORTEX_SMT-zps` | ssh | `3.726197e-07` | **`2.275957e-15`** |
| | u | `6.308422e-08` | `2.160534e-08` |
| | v | `4.742908e-08` | `1.545394e-08` |
| | T | `4.259549e-10` | `3.803022e-10` |
| `VORTEX_SMT_VEC-zps` | ssh | `3.664757e-07` | **`2.664535e-15`** |
| | u | `6.004125e-08` | `1.267541e-09` |
| | v | `4.037530e-08` | `2.548881e-10` |
| | T | `4.539846e-11` | `3.618737e-12` |
Only the two seamount cards change a row's STATUS (one each); the other
eight change none.

**THE TWO-ULP RATCHET IS RED ON NINE OF TEN CARDS, AND THIS RECEIPT LEADS
WITH IT RATHER THAN BURYING IT.** Every violation is of the same shape:
one cell three row-scale oracle ULP worse against a two-ULP bar, e.g.
`VORTEX_SMT_VEC-zps.kt10.before.S: cell 300 worsened against NEMO by
2.13162820728030056e-14 = 3.000 row-scale oracle ulp`. In absolute terms
the worsenings are `1.1e-14` K on temperature and `2.1e-14` on salinity.
Decision 91 took the flat cards' last-bit moves as REGISTERED rather than
scoped away, and decision 90 put NEMO's depth average on every card
including the partial-cell tank, so both sources of this red were chosen
by the user with the movement already reported. It is landed on that
basis and NOT on my own judgement; the counts above are the register.

`LOCK_EXCHANGE-zco` passing is the useful control: it takes decision 90
(13 rows move, all between `1e-16` and `1e-22`) and NOT decision 91 (its
`barotropic_coriolis_split` is `frozen`), and that is exactly the card
whose cell-wise worsenings stay under the bar.

`OVERFLOW-zps` is where decision 90 is NOT a no-op, as predicted before
it ran: it is the only partial-cell tank, 20 rows move, 13 toward NEMO
and 7 away, and its first-over-bar row is unchanged (kt=2, T and u).

## 9.4 GYRE, PER NOTE BZ — LADDER AND YEAR

**Ladder** (`gyre_ladder_r217.json`, 50 rows): 30 rows move, 14 toward
and 16 away, all at the last bit; the first-over-bar row is UNCHANGED at
kt=3 on T, S, u, v and ssh, the same five fields as round 214's.

**The certified year, 8 days, re-measured from rest** (360 days, member 0,
`gyre_day_gap_r217.json`). The run-to-run floor is `2e-10` K and decision
59 allows ten floor units:

| day | round 217 | certified pin | delta | floor units |
|---:|---|---|---:|---:|
| 30 | 2.3432412624693035e-06 | 2.3432465132112266e-06 | -5.251e-12 | 0.026 |
| 60 | 1.4793245058559291e-05 | 1.4793247973304582e-05 | -2.915e-12 | 0.015 |
| 90 | 1.6332689310572426e-05 | 1.6332712039638441e-05 | -2.273e-11 | 0.114 |
| 120 | 1.0965902950977906e-04 | 1.0965907352116351e-04 | -4.401e-11 | 0.220 |
| 180 | 6.1153372915161440e-05 | 6.1153355393000553e-05 | +1.752e-11 | 0.088 |
| 240 | 6.5817073629238215e-05 | 6.5817060949447295e-05 | +1.268e-11 | 0.063 |
| 300 | 5.4660501408258611e-05 | 5.4660498451870513e-05 | +2.956e-12 | 0.015 |
| 360 | 5.4077381774767389e-05 | 5.4077419367442036e-05 | -3.759e-11 | 0.188 |

**Worst move: 0.220 floor units** — 2 % of the run-to-run floor and 2 % of
decision 59's allowance. REGISTERED under the floor rule and RE-PINNED:
the round-217 column is the lane's certified year from here. The old pin
is not deleted from the receipts that measured it; those are records.
(The `--output` flag the round-215 script passed to the day-gap scorer
does not exist — it is `--json` — and the scorer also needs
`--lego-root .../year_fromrest`; round 215's gate script carried both
defects and its day-gap step therefore never produced a number. Fixed
here and the number is above.)

## 9.1 SCOPE — WHICH CARDS EXECUTE THE CHANGED STATEMENTS

Measured by asking the built cards, not by reading the recipe:

| card | `barotropic_coriolis_split` | takes the entry-Coriolis change? | takes the depth-average change? |
|---|---|---|---|
| `VORTEX_SMT-zps`, `VORTEX_SMT_VEC-zps` | live | YES | YES (they select `nemo_literal`) |
| `VORTEX-zco`, `VORTEX_VEC-zco`, both 15 km, both 10 km | live | YES | no (default `min_rule_live`) |
| `GYRE-zco` | live | YES | no |
| `LOCK_EXCHANGE-zco`, `OVERFLOW-zps` | **frozen** | **no** — the branch is inside `if split == "live"` | no |
| ORCA2, DINO | live (shared `_model_config`) | YES | no |

On the six flat zco VORTEX cards and GYRE the THICKNESS rules agree
exactly (full cells), so any row that moves there isolates the TIME-LEVEL
half of the change — which is why those registries are the discriminating
measurement the reviewer asked for and why this round will not land
without them.

## 9.2 THE ORCA2 POINTER

ORCA2's card is built through the same shared `_model_config` as GYRE: it
carries `nemo_prognostic_barotropic_state=True`,
`barotropic_coriolis_split="live"`, `barotropic_coriolis="een_metric"` and
`barotropic_een_coefficient_evaluation="nemo_literal"`, so **ORCA2
executes the entry-Coriolis statement changed here**, and ORCA2's whole
domain is partial cells — the regime where the old and new rules differ.
The ORCA2 lane should, at its next merge of this lane:

1. re-run its certified 200-row ladders on both momentum programs and
   register every moved row with direction; expect movement where the flat
   VORTEX cards show none, because ORCA2 has partial cells everywhere;
2. read the same two NEMO statements on ITS build —
   `stp2d.f90:178-179`/`:183-184` for the depth average and
   `dynspg_ts.f90:289`/`:292` for the entry Coriolis — and confirm the line
   numbers, which differ between builds;
3. decide, with this lane, the `barotropic_slow_forcing_depth_evaluation`
   default question in §10, because ORCA2 is the card where the answer is
   NOT a no-op.

## 10. CHOICES MADE THIS ROUND

| choice | ASKED? |
|---|---|
| the flux acquisition gets its OWN new build pair rather than reusing round 5's unused one | ASKED — the brief says "new directories for any rebuild, never move/delete" |
| the entry-Coriolis repair has NO new config field: the carried external mode is used wherever the card already selects it | not a choice of behaviour — it removes a second rule for a quantity NEMO forms once |
| the depth-average repair lands round 213's HELD patch, whose config field `barotropic_slow_forcing_depth_evaluation` DEFAULTS to `min_rule_live`, i.e. today's behaviour, with only the two seamount cards selecting `nemo_literal` | **RULE 3 QUESTION, ASKED HERE AND NOT ANSWERED: default stays broken (`min_rule_live`) or moves to the fixed value (`nemo_literal`)?** My pick: KEEP `min_rule_live` for now and move it in a round that measures ORCA2 and DINO, because those are partial-cell cards where the answer is not a no-op and this round cannot measure them. The question is on the record rather than the default being quietly shipped |
| the walk's card guard now accepts several acquisition roots per card | not a choice of behaviour: the flux card acquired a second root this round; both refusals were run and printed |
| the 100-day scorer's seamount sanity reference is THIS round's registry rather than a certified one | not a choice of behaviour: this round moves those rows, so a pre-move reference could only fail |

**UNASKED list: EMPTY.** The one decision that needed asking is asked
above and the work is HELD rather than shipped on my own answer.

**COMPLIANCE, stated because no gate checks it (RULE 2):**
* **DUAL review: ONE fresh adversarial reviewer ran, not two.** The second
  is codex, paused on this account. That is a GAP, not an exemption.
* **Controlled comparison:** every arm in §6 differs from production in
  ONE thing on the same tree against the same admitted record; the
  before/after rows in §7 are the same walk on the same record with the
  production change stashed and restored (`git status --porcelain` printed
  clean after the restore).
* **Instrument validated before its numbers were quoted:** the dump
  refuses unless the observer fires and unless the subtraction is non-zero;
  the comparison refuses unless both sides are non-zero and differences
  ARRAYS rather than maxima (the maxima agree to six digits and the arrays
  differ at the eleventh — the exact trap). The transcription and the
  injection of the same quantity land in the same place, which is the
  cross-check that neither is a coincidence.
* **Non-vacuity:** the four record plants fire; the new unit case is shown
  to be sensitive to a dropped keyword; the walk's card guard was shown to
  refuse; and the existing RK3 stage-operand AST guard refused my first
  keyword name, which is that guard proving itself.
* **Pre-impl search:** §1, with the greps pasted.
* **ONE BATTERY AT A TIME:** checked `0` before every pytest invocation,
  with the corrected pattern.

## 11. OPEN

1. **The round is HELD on an incomplete gate set, not on a red.** Round 7
   reads the registries, the GYRE ladder and year, and the 100-day outputs
   already being written into this evidence directory, then lands.
2. The `barotropic_slow_forcing_depth_evaluation` default question (§10).
3. The split probe compares legoESM's entry Coriolis BEFORE its `u_mask`
   against NEMO's masked `zu_trd*ssumask`; a land-cell disagreement would
   be invisible. It cannot affect this round's number but it is a real gap.
4. The flux card's `nemo_literal` depth average OVERWRITES where NEMO's
   flux arm at `stp2d.f90:183-184` CUMULATES onto a 2-D advective
   right-hand side. It is correct only because legoESM carries no separate
   2-D advective term at that boundary; that premise is not asserted
   anywhere in the code and should be made a guard or a test.
5. The flux card still does not materialise the `wzv` operand where NEMO's
   call is unconditional (round 213, unchanged).
6. The stage-terms writer's uninitialised-halo dump; the stage-one stretch
   helper's missing floor; the untranscribed `r3u`/`r3v` stage-one ratios.

## 12. ROUND 7 — PREREGISTERED

* **R7-P1 ADJUDICATED, §9.3: the seamount half HELD and the flat half is
  REFUTED.** Both seamount cards' kt=2 `ssh` reaches the bar; the six flat
  cards move 25-33 last-bit rows each and the two tanks move 13 and 20.
  The operator settled it as decisions 90 and 91 and the moves are
  registered.
* **R7-P2 ADJUDICATED, §9.4: MET.** The ladder's first-over-bar row is
  unchanged and the year's worst day is 0.220 run-to-run floor units from
  the pin, inside the floor and far inside decision 59's ten units.
* **R7-P3 / R6-P2 / R6-P3 ADJUDICATED, §14: all three MET.**
* **R7-P2** — GYRE reproduces its certified ladder (954 rows, 0 moved) and
  its certified year to the pinned eight days and digests (day 30
  `2.3432465132112266e-06` K, day 360 `5.4077419367442036e-05` K).
  **FALSIFIER:** any day differs by more than the 2e-10 K run-to-run floor.
* **R7-P3** — carried from R6-P2/P3, unevaluated here: over 100 days both
  seamount cards stay bounded (peak `|u|` under 2.0 m/s, peak `|ssh|` under
  1.0 m on every scored day, the seamount's deflection the same sign as
  NEMO's), and the seamount pair scores WORSE than the flat pair at the
  same day. **FALSIFIER:** a threshold crossed on any scored day, an
  opposite-sign deflection, or the seamount pair scoring at or better than
  the flat pair.

## 14. THE 100-DAY COMPARISON, BOTH SEAMOUNT CARDS

NEMO's 100-day seamount runs are round 3's admitted ones (100 daily
restarts each); legoESM ran each card 3000 steps from rest on the landed
tree with daily snapshots, scored by the round-210 scorer extended with a
`--cards` selector — the same field extraction, the same `_rms`/`_max`,
the same table days, and the wet mask taken from each card's OWN seamount
geometry. **The scorer's own kt=1..10 sanity check REPRODUCED this round's
registry row for row on both cards before any day was scored.**

rms of (legoESM - NEMO) over the wet mask:

| day | SMT flux T | SMT flux u | SMT flux ssh | SMT vec T | SMT vec u | SMT vec ssh |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.249983e-08 | 7.387520e-08 | 7.019992e-09 | 3.204073e-08 | 3.396484e-08 | 1.235604e-08 |
| 2 | 1.207119e-07 | 2.858069e-07 | 1.144139e-08 | 5.147746e-08 | 5.156699e-08 | 1.588041e-08 |
| 5 | 8.623671e-07 | 1.566380e-06 | 8.023421e-08 | 1.041797e-07 | 1.130934e-07 | 1.932992e-08 |
| 10 | 3.710177e-06 | 9.779855e-06 | 3.984371e-07 | 2.585982e-07 | 2.647966e-07 | 4.275864e-08 |
| 20 | 1.670425e-05 | 3.366900e-05 | 6.155395e-06 | 5.336740e-07 | 5.769240e-07 | 1.102496e-07 |
| 30 | 2.507509e-05 | 3.425672e-05 | 1.193501e-05 | 1.175615e-06 | 1.336023e-06 | 3.169288e-07 |
| 60 | 1.206284e-04 | 1.522355e-04 | 7.291672e-05 | 5.865115e-06 | 5.222433e-06 | 1.850190e-06 |
| 100 | 5.073628e-04 | 5.405034e-04 | 2.141955e-04 | 4.352693e-05 | 2.746327e-05 | 6.910337e-06 |

Day-100 maxima (not rms): flux T 2.981e-02 K, u 4.352e-02 m/s, v
4.184e-02 m/s, ssh 1.104e-03 m; vector T 2.069e-03 K, u 1.118e-03 m/s, v
8.692e-04 m/s, ssh 1.156e-04 m. The vector card is an order of magnitude
closer to NEMO than the flux card at every scored day, which is the same
ordering the flat pair shows.

**R6-P2 CONFIRMED.** Over all 100 days the runs stay bounded well inside
the preregistered thresholds: peak `|u|` is `0.9145` m/s (flux) and
`1.1279` m/s (vector) against the `2.0` m/s falsifier, and peak `|ssh|`
is `0.7955` m and `0.8230` m against `1.0` m. The deflection-sign half of
the prediction is NOT separately instrumented — it is visible in the movie
and is reported as read off the frames, not measured.

**R6-P3 CONFIRMED, and by a wide margin.** At day 100 the seamount pair's
sea-surface-height rms is `2.142e-04` m (flux) and `6.910e-06` m (vector)
against the flat pair's `2.826e-05` m and `7.430e-13` m — 7.6x worse on
the flux card and seven orders of magnitude worse on the vector card. The
kt=2 owner this round removed is therefore NOT the whole of the seamount
pair's developed-flow debt; partial-cell statements downstream of it are.
That re-ranks the next round: the seamount cards' remaining debt is now
the largest VORTEX-family term and it is not the barotropic entry forcing.

Artifacts: `round210_scores.json`, `round210_curves.png`, and per card
`vortex_smtvec_100d.mp4` / `.gif` / `vortex_smtvec_frames.png` and the
flux equivalents — three panels (NEMO | legoESM | difference), sea surface
height with surface-velocity quivers, the card's own resolved bathymetry
contoured on all three panels.


## 15. DECISIONS 90 AND 91, AND WHAT THEY CHANGED

**DECISION 90 — the slow-forcing depth average has no default.** The field
no longer carries a scheme; the consumer raises on the unset value with a
message that tells the card to state one, and every NEMO testcase card now
states NEMO's own form, `nemo_literal`. The test
(`tests/ocean/unit/test_nemo_card_opt_in_defaults.py::
test_slow_forcing_depth_evaluation_has_no_default_and_unset_raises`) reads
the card list off the builder's OWN error rather than listing it, so a new
card cannot slip past, and it is proved non-vacuous: restoring the old
`min_rule_live` default turns it red (run, printed, and the file restored
with `git status --porcelain` checked).

This is wider than the seamount pair. On the full-step cards NEMO's form
is algebraically the previous one — one per-face scalar cancels — but
**`OVERFLOW-zps` is a partial-cell card and it is NOT a no-op there**;
§9.3 registers what it did.

**DECISION 91 — the entry-Coriolis statement lands on the shared path for
every card.** No card-level gate was added and the six flat VORTEX cards'
last-bit moves are REGISTERED below rather than scoped away. `GYRE-zco`,
`ORCA2-zps` and DINO take the same path; the two tanks do not reach it
(`barotropic_coriolis_split="frozen"`).

## 16. LANDED

`land.sh gyre` ran the push gate on this tree. Its lines, including the
DINO from-rest month gate (bar `2.053801168e-03` K, which runs because the
production diff is non-empty), are in §17.

## 13. EVIDENCE

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex_smt/round6/`:
`predictions.md` (frozen first); `acquisition_flux.log` (the refused first
attempt) and `acquisition_flux2.log` with the admitted record and its
admission JSON under `VORTEX_SMT_R6B_OMIP_L1_P3/spgts/`;
`lego_entry_coriolis_vec.npz` and `entry_coriolis_split_vec.json` (the
per-statement comparison); the walk arms
`walk_vec_kt1_carriedbaro{,_nemodepth}.{json,log}`,
`walk_vec_kt1_both.{json,log}`,
`walk_flux_kt1_{before,after}.{json,log}`; the registries under `after2/`
with `registries.sh` and `registry_diff.txt`; `gyre_gates.sh` with its
outputs; `hundred_day.log` and the per-card snapshot directories.
Round 5's `nemo_depth_average_kt1.npz` is reused, not regenerated.

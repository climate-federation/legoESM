# Round 216 / VORTEX_SMT round 6 — the entry forcing's TWO statements, and the 100-day comparison

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

## 9. GATES — THE ROUND IS **HELD**, AND THE REASON IS THE CLOCK, NOT A RED

**Nothing is pushed.** The statement is transcribed, measured on both
cards, reviewed and committed locally on
`fidelity/nemo-testcases-l2-gyre-codex2` at the round-6 clone, but the
landing gate list in note CC addendum 6 is not complete inside this
round's tool budget and a partial gate set is not a gate set.

| gate | state at hand-off |
|---|---|
| flux seamount substep record | **ADMITTED** (§2); additions-only proven across builds and rounds |
| the two statements close the kt=1 window, BOTH cards | **MET** (§6, §7) |
| direct unit test, non-vacuous | **GREEN.** `tests/ocean/unit/test_barotropic_coriolis_null_mode.py` gains one case that pins the supplied pair as what the operator sees AND asserts it differs from the reduction, so a dropped keyword turns it red. The keyword is `entry_barotropic_velocity`, renamed from `barotropic_velocity` after the existing RK3 stage-operand AST guard correctly refused the collision — that refusal is itself a non-vacuity proof of that guard |
| the ten-card registries (two seamount, six flat, two tanks) | **COMPLETE — see §9.3. Both seamount cards' kt=2 sea surface height reaches the bar; the two tanks move 0 rows; THE SIX FLAT CARDS MOVE 24-34 ROWS EACH, at the last bit, which REFUTES the round's own "0/50 expected"** |
| GYRE ladder + certified year (note BZ, 8 days + digests) | **QUEUED**, chained behind the registries; the script is `round6/gyre_gates.sh` |
| DINO from-rest month gate (`2.053801168e-03` K) | NOT RUN — it runs inside `land.sh`, which was not invoked |
| the 100-day comparison, both seamount cards + movie | **QUEUED**, chained behind the registries; scorer and movie already extended and committed |
| two-ULP ratchet | not reached |

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

* **R7-P1 is ALREADY ADJUDICATED, in §9.3, and it is REFUTED.** The
  seamount half held — both cards' kt=2 `ssh` reaches the bar — and the
  flat half did not: the six flat cards move 24-34 last-bit rows each
  while the two tanks move 0. The decision this hands the operator is in
  §9.3 and it is the first thing round 7 must settle.
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

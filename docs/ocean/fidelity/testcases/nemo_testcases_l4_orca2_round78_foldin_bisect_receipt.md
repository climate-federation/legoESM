# ORCA2 round 78 — bisecting the GYRE fold-in, and what the ladder move really is

Status: **the owner of the rows round 77 flagged is NAMED and PROVED by a
one-variable arm, run both at the bisect point and at the lane tip; NOTHING is
landed, because the statement that moved ORCA2 is NEMO's own value for this
deck, read out of the record's own run log.** Round 77's refutation of the
mixing length is **RETRACTED**: its arm perturbed the floor to `1e-8 m`, which
is not the value the pre-merge tree ran. One claim of this round's own first
draft is retracted before it ships as well — see §7.

## 0. Answer, in one paragraph

The rows round 77 flagged — kt 10, end of step, `T` and `S` maxima, `+13.3%`
and `+15.4%` away from NEMO — moved because ORCA2's turbulence **mixing-length
floor** went from an effective **1.0 m** to NEMO's **1.0e-3 m**. The 1.0 m was
not on the card: it was derived by code that ignored the card's own field, which
printed `1e-8`. Putting the floor back on the old derivation, ONE variable,
reproduces the pre-merge ladder **on all 200 comparison rows with every recorded
field equal** at the bisect point, and at the LANE TIP brings those two headline
rows back to within `0.11%` and `0.13%` of the pre-merge values. `1.0e-3 m` is
what NEMO prints for this very run, so nothing is reverted: the fold-in removed
a legoESM defect, and the ladder now shows what that defect was covering.

## 1. Preregistration, and one correction to the order

`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round78_foldin_bisect.md`,
committed before any ladder of this round was run.

The order named "the step-10 end-of-step T rms and S rms — the rows round 77
saw worsen 13%/15%". Those are two different rows, and the preregistration says
so before measuring: the 13%/15% moves belong to the **maxima**; the `T` rms
moves away by 1.73% and the `S` rms moves **toward** NEMO by 3.66%. The bisect
therefore scores kt 10, stage 3, `T` max_abs, with the other three recorded
beside it at every point.

## 2. The endpoints reproduce, exactly

| endpoint | tree | kt10 stage-3 `T` max | vs round 77's committed ladder |
|---|---|---|---|
| good | pre-merge ORCA2 `1142d4182` | `0.9838385161101275` | **200 of 200 comparison rows identical** |
| bad | this lane's tip `629d5f4c5` | `1.114220520669864` | **200 of 200 comparison rows identical** |

Prediction **Q3 CONFIRMED**: the ladder is deterministic across clones, so the
round-77 magnitudes stand as a BEFORE and this round's numbers are comparable to
them. **What "identical" means here, precisely**: the ladder records are
comparison summaries — per row, the unequal-cell count, the count, the maximum
absolute difference, the rms and the at-bar flag — and every one of those fields
is equal on every one of the 200 rows. It is NOT a hash of the candidate state
arrays, and this receipt never claims one. It also confirms round 77's other claim in passing — the tip carries the
stated after-SSH field and reproduces the merge tree's ladder to the last digit,
so that field is inert as round 77 said.

Each ladder is 15-17 minutes of CPU; they were run one at a time.

## 3. The bisect

Space: the GYRE lane's **43** first-parent commits in `d3631f884..ea12107cb`
that touch `packages/` or `src/` (the merge base is the one round 77 recorded).
Each point is built as `merge(1142d4182, X)` — the pre-merge ORCA2 tree with the
GYRE chain truncated at `X`. The four conflicts are the same four round 77 hit
(two receipts, the Decision-43 gate, the citation gate); none is under
`packages/` or `src/`, and all four are taken from the ORCA2 side, which the
ladder never reads. **The construction is verified rather than asserted**: at
`X = ea12107cb` the built tree's `packages/` and `src/` are byte-identical to
round 77's merge commit `ff08f47a9`, and differ from the lane tip only by the
13 lines of the stated after-SSH field.

| point | GYRE commit | kt10 stage-3 `T` max | `S` max | `T` rms | `S` rms | rows on `max_abs`: unchanged / toward / away | verdict |
|---:|---|---|---|---|---|---|---|
| 0 | (pre-merge `1142d4182`) | `0.9838385161101275` | `0.22099092586135072` | `0.010433315642886491` | `0.003029909179584292` | 200 / 0 / 0 | good |
| 20 | `381c4e396` shortwave source association | `0.9838385161101275` | `0.22099092586135072` | `0.010433315642886491` | `0.003029909179584292` | **200 / 0 / 0** | **good** |
| 21 | `0ebdb6d83` rmxl_min arms + TKE card opt-in | — | — | — | — | — | **DOES NOT BUILD** |
| 22 | `702fb95db` carried external mode + mesh Coriolis | `1.1141520089561965` | `0.2550649479550131` | `0.01074163857968142` | `0.0029203697463684375` | 24 / 79 / 97 | **bad** |
| 43 | (tip `629d5f4c5`) | `1.114220520669864` | `0.25508044621046366` | `0.010613818039912676` | `0.002919141602961423` | 24 / 77 / 99 | bad |

**The census columns count `max_abs`, which is not the same as an unchanged
row.** Against the pre-merge tree, both point 22 and the tip have 24 rows whose
`max_abs` is equal but only **17** whose every recorded field is equal — seven
rows keep their maximum while their rms moves. Where this receipt claims a whole
ladder is unchanged, the whole-row count is quoted and it is the one that is
200: **point 20 and the point-22 arm are equal to the pre-merge ladder on 200 of
200 rows on BOTH counts**, and so are the two endpoint reproductions of round
77's records.

**Point 21 is not a bisect point, and that is a fact about the commits, not a
limitation of the method.** `0ebdb6d83` writes `use_mesh_coriolis=True` and
`nemo_prognostic_barotropic_state=True` onto the ORCA2 card while the parameter
and the config field both arrive in `702fb95db`, eighteen seconds later. Built
alone it raises
`TypeError: create_tripole_grid() got an unexpected keyword argument 'use_mesh_coriolis'`
(2 s, log `ladder_p21_0ebdb6d8.log`). The two commits are ONE landing — the
PR-1802 review-fix batch — and the bisect brackets them together.

**Q1 CONFIRMED for the flagged rows, with two limits stated.** Everything before
the pair leaves the ladder unchanged on all 200 comparison rows (point 20), and
everything after it moves the kt 10 `T` maximum by `+0.006%` and the `S` maximum
by `+0.006%` (point 22 to the tip).

**Limit 1: commits 1-20 are shown COLLECTIVELY inert, not individually.** Only
point 20 was run, so a move at some commit below 20 that a later one exactly
cancelled is not excluded by measurement. It would have to restore all 200
summary rows to every recorded digit, which is implausible, but implausible is
not measured, and the receipt says so rather than writing "no earlier commit
moves it".

**Limit 2: the commits AFTER the pair are NOT inert on the ladder as a whole —
only on the flagged rows.** Point 22 to the tip moves 175 of 200 rows (25
unchanged on `max_abs`, and on every field, 60 toward NEMO, 115 away), the
largest being kt 7 stage-3 `v`,
`0.604874 -> 0.363905` (−39.8%). What they leave almost exactly alone are the
four kt 10 end-of-step maxima this round scores. So "the pair owns the move" is
a statement about the rows round 77 flagged, and §4's tip arm is what makes it
one about the tip rather than about the bisect point.

## 4. Which statement inside the pair, by one variable

The pair changes exactly four leaves of ORCA2's resolved configuration
(measured with the committed fingerprint probe,
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_card_fingerprint.py --config-dump`,
run on both trees):

| leaf | point 20 | point 22 |
|---|---|---|
| `physics.vertical_mixing.tke.mxl_min` | `1e-08` | `0.001` |
| `physics.vertical_mixing.tke.nemo_derived_mxl_min` | (field does not exist) | `False` |
| `physics.vertical_mixing.tke.nemo_mxl0_surface_tmask` | (field does not exist) | `True` |
| `barotropic.nemo_prognostic_barotropic_state` | (field does not exist) | `True` |

**The barotropic one is inert here, measured.** Before the pair the window seed
took NEMO's carried external mode whenever the state held the `uu_b`/`vv_b`
pair; the ORCA2 card's state holds it on BOTH trees (`uu_b is None: False` at
point 20 and at point 22), and the new flag is `True`, so the same arm runs
either way. The change removed a state-presence fallback; on this card it
changed nothing.

**That leaves the mixing length — and the pre-merge value was not the one on the
card.** Before the pair, `_mixing_length_floor` ignored `cfg.mxl_min` for every
`tke_mxl_choice` 3/4 card and returned the derivation
`1e-6 / (c_k * sqrt(tke_background))`. On ORCA2's own resolved values,
`c_k = 0.1` and `tke_background = 1e-10`, that is **0.9999999999999998 m**. The
card's `mxl_min = 1e-08` was dead text. After the pair the card selects the
other arm and the floor is its own `1.0e-3 m`.

**The arm.** Point 22 with ONE line changed — `nemo_derived_mxl_min=True`, so
the floor goes back to the pre-merge derivation of 1.0 m, everything else the
point-22 tree — reproduces the pre-merge ladder **on all 200 comparison rows,
with every recorded field of every row equal**: kt 10 stage-3 `T` max `0.9838385161101275`, `S` max
`0.22099092586135072`, `T` rms `0.010433315642886491`, `S` rms
`0.003029909179584292`. Branch `r78-mxlfloor-arm`, commit `ba42e7b05`, record
`ladder_arm_mxlfloor.json`.

**The same arm at the LANE TIP.** Because the commits after the pair move 175
rows of their own, the arm was repeated on the tip — the tree the lane actually
carries — with the same single field flipped (`r78-mxlfloor-arm-tip`, commit
`66ccb8f2f`, record `ladder_arm_tip.json`):

| kt 10, stage 3 | pre-merge | tip | tip with the floor put back |
|---|---|---|---|
| `T` max | `0.9838385161101275` | `1.114220520669864` (+13.25%) | `0.9827840325928534` (**−0.11%**) |
| `S` max | `0.22099092586135072` | `0.25508044621046366` (+15.43%) | `0.22071335585861362` (**−0.13%**) |
| `T` rms | `0.010433315642886491` | `0.010613818039912676` | `0.010319467633720783` |
| `S` rms | `0.003029909179584292` | `0.002919141602961423` | `0.0030291931706825964` |

Against the pre-merge tree the tip arm's whole-ladder census is 25 unchanged
(on `max_abs` and on every field alike), 88 toward NEMO, 87 away — balanced, which is the other 41 commits' own
contribution and not a direction. Against the TIP it is 24 unchanged, 90 toward,
86 away.

So: **the mixing-length floor owns the entire flagged move, at the tip and not
only at the bisect point** — `+13.25%` becomes `−0.11%` and `+15.43%` becomes
`−0.13%` from one field. It does not own the rest of the ladder's movement,
and this receipt does not claim it does.

One honest note on the tip arm's reach: at the tip the card also carries
`nemo_mxl0_rmxl_min_overwrite`, which sets the surface anchor from the same
floor, so flipping the one field moves the anchor too. That is one config field
and one physical quantity, but its downstream reach at the tip is wider than at
point 22 — which is why both arms are reported rather than only the tip one.

## 5. RETRACTION — round 77's mixing-length refutation was wrong

Round 77 wrote: *"A group that moves the ladder by two parts in ten thousand
cannot own moves of ten to thirty per cent."* The measurement was real and the
inference from it was wrong, because **the arm did not restore the pre-merge
value**. It set `mxl_min` back to `1e-8 m` with the derivation switched off,
i.e. a floor of `1e-8 m`. The pre-merge tree ran `1.0 m`. Both `1e-8 m` and
`1e-3 m` are far below the mixing lengths this card actually reaches, so that
arm compared no-floor with no-floor and correctly found almost nothing; `1.0 m`
is a floor that BINDS.

Both numbers are now explained by one curve rather than contradicting each
other: `1e-8 -> 1e-3` moves the ladder by `2.0e-04` (round 77), and
`1e-3 -> 1.0` moves it by the entire 13%/15% (this round).

The general lesson, for the lane: **an arm that reverts a CONFIG FIELD has not
reverted the BEHAVIOUR until the field is shown to be read.** Round 77 read the
before-value off the resolved configuration, where the pre-merge tree printed
`1e-08`, and the pre-merge code never looked at it.

Prediction **Q2 is REFUTED, and it is refuted in the way that matters**: the
culprit IS inside one of the two candidates round 77 named. What was wrong was
not the candidate, it was the arm.

## 6. Is the statement NEMO's for ORCA2's own program? YES

Not inferred from the GYRE receipt — read from ORCA2's own build and from the
record the ladder scores against.

* The deck sets it: the scored run
  (`nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2`)
  carries `ln_zdfiwm = .true.` in its own `namelist_cfg`, line 397.
* ORCA2's compiled source branches on it:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-841` is the two-arm choice —
  `IF( ln_zdfiwm ) THEN` forces `rn_emin = 1.e-10_wp` and
  `rmxl_min = 1.e-03_wp` on lines 836-837 and never evaluates the `ELSE`
  derivation `rmxl_min = 1.e-6_wp / ( rn_ediff * SQRT( rn_emin ) )` on line 840.
* **The run says which arm it took**: that run's own `ocean.output`, line 1024,
  prints
  `==>>>   Internal wave-driven mixing case:   force   rn_emin = 1.e-10 and rmxl_min = 1.e-3`.

So `1.0e-3 m` is NEMO's floor for this run, and the pre-merge `1.0 m` was a
legoESM defect. **Nothing is reverted and nothing is scoped by card.** The
GYRE lane's landing was right for ORCA2 as well as for GYRE.

### What the GYRE receipt claimed about ORCA2

The pair's receipt is
`docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_pr1802_review_fixes_receipt.md`.
It did NOT claim ORCA2 does not execute this. It named ORCA2 explicitly and
flagged the move, quoted:

> **D1 — ORCA2-zps mixing-length floor. ALREADY MOVED, flagged.** This is the
> one place I did not leave the default alone, because all three candidate
> values disagree: main 1.0e-8 m, branch 1.0e-3 m via `cfg.mxl_min` only after
> this fix, branch-before-fix 1.0 m, NEMO 1.0e-3 m. A card whose entire
> contract is NEMO literalness cannot ship a floor a thousand times NEMO's …
> **Recommend: keep the move.** Affects: the ORCA2-zps card and every ORCA2
> lane gate; they need re-running. Revert is one line (`mxl_min=1.0e-3` → drop,
> `nemo_derived_mxl_min=True`).

and, in its behaviour-change list,

> **3. ORCA2-zps mixing-length floor.** 1.0e-3 m (NEMO's `ln_zdfiwm` forced
> value) instead of the branch's derived 1.0 m and main's 1.0e-8 m. The ORCA2
> card must be re-certified; see decision D1.

**The receipt got this exactly right, including the 1.0 m, said the ORCA2 gates
must be re-run, and even wrote down the one-line revert that is this round's
arm.** What went wrong is that the flag was carried across the
merge as a configuration row (`1e-08 -> 0.001`) rather than as the behaviour
change it is, and round 77 then refuted it against the configuration row. The
GYRE receipt is the document that would have prevented this round.

## 7. So the worsening is a compensating error being uncovered

**RETRACTED, in this round, before it shipped: "a thousand times NEMO's
background vertical mixing".** The first draft of this section read the floor as
a floor on the vertical diffusivity, using the compiled source's own comments
(line 837, "associated avt minimum = molecular salt diffusivity (10^-9 m2/s)";
line 839, "standard case : associated avt minimum = molecular viscosity
(10^-6 m2/s)", both inside the span cited above). **That reading does not hold
for this deck, and the record says so.** ORCA2 sets its own background
coefficients well ABOVE both of those values — `rn_avm0 = 1.2e-4 m2/s` and
`rn_avt0 = 1.2e-5 m2/s`, in the scored run's `namelist_cfg` lines 399-400 and
echoed back in its `ocean.output` lines 990-991 — so in a quiescent column the
background dominates whatever the floored mixing length produces, at 1.0 m as
much as at 1.0e-3 m. The adversarial review caught this; the claim is withdrawn
rather than softened.

**What IS established about the change.** It is a factor of 1000 on the MIXING
LENGTH itself, which the scheme reads twice: the length multiplies the
turbulent-energy term that makes the diffusivity, and it divides the dissipation
term. Where the physical mixing length is small — strong stratification — a 1.0 m
floor therefore both raises the coefficient and lowers the dissipation, and where
it is large the floor is invisible. **By how much, on this card, is UNMEASURED**:
no `avt`, `avm` or `zmxlm` record exists for this run (see below), so this round
states the two ends of the range and does not put a number between them.

**What that damped, measured on the ladder.** The move is not one row. Against
the pre-merge ladder, 176 of 200 rows move, 77 toward NEMO and 99 away, in every
field:

| field | rows unchanged | toward | away | largest single move |
|---|---:|---:|---:|---|
| `T` | 5 | 3 | 32 | kt 9 entry `1.47543 -> 1.04571` (−29.1%) |
| `S` | 5 | 17 | 18 | kt 4 entry `0.267004 -> 0.102389` (−61.7%) |
| `u` | 4 | 18 | 18 | kt 8 stage 2 `0.275719 -> 0.362293` (+31.4%) |
| `v` | 5 | 14 | 21 | kt 7 stage 2 `0.446756 -> 0.75685` (+69.4%) |
| `ssh` | 5 | 27 | 8 | kt 9 entry `0.510739 -> 0.508166` (−0.5%) |

and the headline row grows with the step rather than jumping:

| kt | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| stage-3 `T` max, % away | 0.00 | +19.13 | 0.00 | +0.01 | +0.02 | +0.03 | +9.76 | +11.43 | +12.63 | +13.25 |

kt 1 does not move at all, which is the expected signature of a background
mixing floor: it needs steps to act.

**CONFIRMED**: the floor owns the whole of the flagged kt 10 move, at the tip as
well as at the bisect point, and what the old floor was covering lives in every
prognostic field rather than in one row.
**PLAUSIBLE, NOT MEASURED HERE**: that the largest part of what it damped is the
stage advective transport error rounds 73-74 localised — NEMO's `un_adv` is the
first non-bit primitive on all 8,568 active U faces with maximum 13.24 m3/s
(`nemo_testcases_l4_orca2_round74_metric_u_transport_receipt.md`). This round
does not measure that link and does not claim it.

**The discriminator, and why it is round 79's first item.** The substitution the
order asks for — NEMO's own recorded operand at this statement's CONSUMER — is
the recorded vertical diffusivity/viscosity (or the recorded mixing length
`zmxlm`) per step. It is also what would replace the withdrawn claim above with
a number. **The records on disk do not carry it.** The admitted ORCA2
streams are step-entry `T`, `S`, `u`, `v`, `ssh` and the ocean surface inputs,
kt 1..10, full-rank; there is no `avt`/`avm`/`zmxlm` stream. Round 79 therefore
needs a new acquisition under a new target name, writing `avt`, `avm` and
`zmxlm` per step per rank, before the compensation question can be answered by
measurement instead of by argument.

## 8. What this round does NOT license

**Every ORCA2 magnitude on this lane measured before the fold-in was measured
with the turbulence mixing length floored a thousand times higher than NEMO's
value for this deck** (and, at the tip, with the surface anchor that follows it).
How much vertical mixing that actually bought is unmeasured, per §7; that it
moved the flagged rows by 13% and 15% is measured. That includes
round 71's independent month ranking (terminal `T` max `21.637832697714707` K,
rms `0.17564842520962015` K at 240 steps) and the magnitude parts of the
rounds 72-76 walks. Bit-level statements — which row is at the bar, which
statement is the first non-bit one — are unaffected: the first non-bit
statement is kt 1, stage 1, `T` on every tree measured in this round, and the
5 rows at the bar stay 5 on every one. **Magnitudes are superseded; bit rows are
not.**

## 9. Gates

No model file is edited in this round, so no trajectory gate applies: the
receipt, the preregistration and the arm are the whole diff. The arm lives on
branch `r78-mxlfloor-arm` (`ba42e7b05`) and is a one-line change to a worktree,
never to the lane.

| gate | result |
|---|---|
| GYRE, tanks, DINO | **not run, and not required** — nothing shared changed. Round 77 measured them on this tip and they were byte-identical/PASS |
| citation gate, this receipt | **PASS**, 1 citation, 0 unmapped, 0 failures, 0 map entries failing audit |
| the same with a planted two-line shift on that citation | **FAIL**, `SYMBOL-NOT-AT-LINE` — the gate is not vacuous on this receipt |
| citation gate, default cumulative receipt | **PASS**, 274 citations, 0 unmapped, 0 failures; all nine of the gate's own self-tests behaved — the seven planted defects each produced the failure status they are meant to (`AMBIGUOUS-ANCHOR`, `BAD-CITATION` twice, `SYMBOL-NOT-AT-LINE` four times) and the two unplanted baselines stayed `OK` |
| `test_nemo_testcase_receipt_citation_gate.py`, `test_nemo_vortex_card.py`, `test_nemo_card_opt_in_defaults.py` | **`87 passed`** |
| the ORCA2 lane's other four push-gate files | **`111 passed`** |
| card battery | not run — no card changed |

The two premises that are NOT citations — the deck's `ln_zdfiwm` namelist line
and the run log line that says which arm the run took — are read from the run
directory and named in §6 as prose, because the citation map keys compiled
sources only. They are the load-bearing half of §6, and no gate audits them.

## 9b. Review

`codex exec --sandbox read-only` on the whole round (preregistration, receipt,
the one map entry, and the ladder JSONs, which it read and recomputed from).
Verdict, quoted: **HOLD**, with four findings. All four were real; all four are
closed above, one of them by a new measurement rather than by a rewording.

| # | finding | what it changed |
|---|---|---|
| 1 | "the arm was applied at point 22, not the tip: point 22 and the tip differ on 175 of 200 rows, largest 39.84%" — so the arm proved ownership of the point-20-to-22 step, not of the fold-in | **A NEW ARM WAS RUN**, the same one variable at the lane tip (§4). It brings the two flagged rows from `+13.25%`/`+15.43%` to `−0.11%`/`−0.13%`. The later commits' own 175-row movement is now reported (§3, limit 2) instead of being summarised as "+0.006% on this row" |
| 2 | "a thousand times NEMO's background vertical mixing" is false for this deck: ORCA2 sets `rn_avm0 = 1.2e-4`, `rn_avt0 = 1.2e-5 m2/s`, above the values the floor implies | **RETRACTED in §7**, with the namelist and log lines that refute it quoted. The change is a factor of 1000 on the mixing LENGTH; what it buys in mixing is unmeasured and now says so |
| 3 | "byte-identical" exceeds the evidence: the records are comparison summaries, not hashes of the candidate state | wording fixed throughout to "all 200 comparison rows, every recorded field", and §2 says explicitly what the records do and do not contain. **Reopened by the follow-up review and closed again**: the census columns count `max_abs`, where 24 rows match but only 17 match on every field; §3 now says so, and the claims that matter are quoted on the whole-row count, where they are 200 of 200 |
| 4b | "nine self-test plants fired" miscounts: seven are planted defects, two are unplanted baselines | §9 now names the seven statuses and the two baselines |
| 4 | the gates table pointed at a section that does not exist, and the citation gate audits only the compiled-source citation, not the namelist and run-log premises | §9 rewritten with the actual gate results, and the un-audited premises named |

A second pass was run after these fixes. It confirmed findings 1 and 2 closed
and returned two further precision defects, both about counting rather than
about the result; they are rows 3 and 4b above and are closed in the same way.
Across both passes the reviewer confirmed: the arm is a valid one-variable
control on the floor, the retraction of round 77 is correct, and the three
ORCA2/NEMO facts in §6 are at the lines claimed.

## 10. Choices made this round

| choice | status |
|---|---|
| bisect on kt 10 stage-3 `T` max_abs rather than on the rms the order named | ASKED in effect — the order named the rows that moved 13%/15% and those are the maxima; the preregistration says so and records all four |
| treat commits 21 and 22 as one atomic point | NOT A CHOICE — commit 21 alone raises a `TypeError`; the evidence is in the receipt |
| resolve the four merge conflicts from the ORCA2 side at every bisect point | NOT A CHOICE about physics — none is under `packages/` or `src/`, and the construction is verified against round 77's merge tree |
| land nothing | ASKED — the order says a statement that is NEMO's for ORCA2 is not scoped |
| run a second arm at the lane tip after the review | NOT A CHOICE about physics — it is the measurement the reviewer's first finding asked for, same one variable |

UNASKED: none.

UNCHANGED and not touched: every card, every selector, the carried state, the
thresholds, the forcing, the NEMO sources, and the first-non-bit statement walk.

## OPEN

1. **Round 79: acquire the mixing record.** `avt`, `avm` and `zmxlm` per step per
   rank for kt 1..10 on the ORCA2 twin, under a NEW target name, so the
   compensation question is settled by substitution rather than by the argument
   in §7.
2. **Re-rank ORCA2's errors on the corrected tree.** The independent month
   (round 71) has to be re-measured before any magnitude on this lane is quoted
   again; round 77 already had it open at 2.6 h of CPU and this round makes it
   necessary rather than tidy.
3. **The lane's arm discipline.** Round 77's arm reverted a field the old code
   never read. A cheap gate exists: an arm that claims to restore a previous
   behaviour should print the VALUE ITS CONSUMER SEES on both trees, not the
   config leaf. Worth one round.
4. Round 77's remaining OPENs 4 and 5 (the GYRE ten-step residual digest
   like-for-like, and the order-dependent GYRE-zco card digest) are untouched.

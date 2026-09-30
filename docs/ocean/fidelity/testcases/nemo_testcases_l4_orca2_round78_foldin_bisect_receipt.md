# ORCA2 round 78 — bisecting the GYRE fold-in, and what the ladder move really is

Status: **the owner is NAMED and PROVED by a byte-identical one-variable arm;
NOTHING is landed, because the statement that moved ORCA2 is NEMO's own value
for this deck, read out of the record's own run log.** Round 77's refutation of
the mixing length is **RETRACTED**: its arm perturbed the floor to `1e-8 m`,
which is not the value the pre-merge tree ran.

## 0. Answer, in one paragraph

ORCA2's ten-step ladder moved at the fold-in because ORCA2's turbulence
**mixing-length floor** went from an effective **1.0 m** to NEMO's **1.0e-3 m**.
The 1.0 m was not on the card — it was derived by code that ignored the card's
own field — and 1.0e-3 m is what NEMO prints for this very run. Putting the
floor back on the old derivation, one variable, reproduces the pre-merge ladder
on **all 200 rows, every digit**. So the fold-in did not break ORCA2; it removed
a thousand-fold excess of background vertical mixing that was damping the
ladder's errors, and the ladder now shows what was underneath.

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
| good | pre-merge ORCA2 `1142d4182` | `0.9838385161101275` | **200 of 200 rows identical** |
| bad | this lane's tip `629d5f4c5` | `1.114220520669864` | **200 of 200 rows identical** |

Prediction **Q3 CONFIRMED**: the ladder is deterministic across clones, so the
round-77 magnitudes stand as a BEFORE and this round's numbers are comparable to
them. It also confirms round 77's other claim in passing — the tip carries the
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

| point | GYRE commit | kt10 stage-3 `T` max | `S` max | `T` rms | `S` rms | rows unchanged / toward / away | verdict |
|---:|---|---|---|---|---|---|---|
| 0 | (pre-merge `1142d4182`) | `0.9838385161101275` | `0.22099092586135072` | `0.010433315642886491` | `0.003029909179584292` | 200 / 0 / 0 | good |
| 20 | `381c4e396` shortwave source association | `0.9838385161101275` | `0.22099092586135072` | `0.010433315642886491` | `0.003029909179584292` | **200 / 0 / 0** | **good** |
| 21 | `0ebdb6d83` rmxl_min arms + TKE card opt-in | — | — | — | — | — | **DOES NOT BUILD** |
| 22 | `702fb95db` carried external mode + mesh Coriolis | `1.1141520089561965` | `0.2550649479550131` | `0.01074163857968142` | `0.0029203697463684375` | 24 / 79 / 97 | **bad** |
| 43 | (tip `629d5f4c5`) | `1.114220520669864` | `0.25508044621046366` | `0.010613818039912676` | `0.002919141602961423` | 24 / 77 / 99 | bad |

**Point 21 is not a bisect point, and that is a fact about the commits, not a
limitation of the method.** `0ebdb6d83` writes `use_mesh_coriolis=True` and
`nemo_prognostic_barotropic_state=True` onto the ORCA2 card while the parameter
and the config field both arrive in `702fb95db`, eighteen seconds later. Built
alone it raises
`TypeError: create_tripole_grid() got an unexpected keyword argument 'use_mesh_coriolis'`
(2 s, log `ladder_p21_0ebdb6d8.log`). The two commits are ONE landing — the
PR-1802 review-fix batch — and the bisect brackets them together.

**Q1 CONFIRMED, and it is a step, not a gradient.** Everything before the pair
leaves the ladder byte-identical (point 20: 200 of 200 rows unchanged), and
everything after it moves the headline row by `+0.006%` (point 22 to the tip).
The other 41 commits — 455 GYRE-lane commits' worth of merged work — are
together worth six parts in a hundred thousand on this row.

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
point-22 tree — reproduces the pre-merge ladder **on all 200 rows, with every
field of every row equal**: kt 10 stage-3 `T` max `0.9838385161101275`, `S` max
`0.22099092586135072`, `T` rms `0.010433315642886491`, `S` rms
`0.003029909179584292`. Branch `r78-mxlfloor-arm`, commit `ba42e7b05`, record
`ladder_arm_mxlfloor.json`.

That is the whole move, owned by one number, with nothing left over.

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

* The deck sets it: `namelist_cfg:397` of the scored run
  (`nemo-testcases-l4/runs/variant_icebergs_off_phase2v_tke_a_10step_np2`)
  carries `ln_zdfiwm = .true.`.
* ORCA2's compiled source branches on it:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-842` is the two-arm choice —
  `IF( ln_zdfiwm ) THEN` forces `rn_emin = 1.e-10_wp` and
  `rmxl_min = 1.e-03_wp` (`:836-837`) and never evaluates the `ELSE` derivation
  `rmxl_min = 1.e-6_wp / ( rn_ediff * SQRT( rn_emin ) )` (`:840`).
* **The run says which arm it took**: `ocean.output:1024` of that run prints
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

**The mechanism, from NEMO's own comments at the two arms.** The floor is on the
turbulence mixing length, which sets the MINIMUM vertical diffusivity the scheme
can produce: with `c_k = 0.1` and the floor on the square root of the minimum
turbulent energy, a 1.0 m floor gives about `1e-6 m2/s` and a `1e-3 m` floor
about `1e-9 m2/s` — which is precisely what the compiled source calls them
(`:837` "associated avt minimum = molecular salt diffusivity (10^-9 m2/s)";
`:839` "standard case : associated avt minimum = molecular viscosity
(10^-6 m2/s)"). **Every ORCA2 run on this lane before the fold-in carried a
background vertical mixing a thousand times NEMO's**, on both tracers and
momentum.

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

**CONFIRMED**: the floor owns the whole move, and the extra mixing was damping a
developed-state error that lives in every prognostic field, not in one row.
**PLAUSIBLE, NOT MEASURED HERE**: that the largest part of what it damped is the
stage advective transport error rounds 73-74 localised — NEMO's `un_adv` is the
first non-bit primitive on all 8,568 active U faces with maximum 13.24 m3/s
(`nemo_testcases_l4_orca2_round74_metric_u_transport_receipt.md`). This round
does not measure that link and does not claim it.

**The discriminator, and why it is round 79's first item.** The substitution the
order asks for — NEMO's own recorded operand at this statement's CONSUMER — is
the recorded vertical diffusivity/viscosity (or the recorded mixing length
`zmxlm`) per step. **The records on disk do not carry it.** The admitted ORCA2
streams are step-entry `T`, `S`, `u`, `v`, `ssh` and the ocean surface inputs,
kt 1..10, full-rank; there is no `avt`/`avm`/`zmxlm` stream. Round 79 therefore
needs a new acquisition under a new target name, writing `avt`, `avm` and
`zmxlm` per step per rank, before the compensation question can be answered by
measurement instead of by argument.

## 8. What this round does NOT license

**Every ORCA2 magnitude on this lane measured before the fold-in was measured
under a background vertical mixing a thousand times NEMO's.** That includes
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
| citation gate | run on this receipt, see §11 |
| card battery | not run — no card changed |

## 10. Choices made this round

| choice | status |
|---|---|
| bisect on kt 10 stage-3 `T` max_abs rather than on the rms the order named | ASKED in effect — the order named the rows that moved 13%/15% and those are the maxima; the preregistration says so and records all four |
| treat commits 21 and 22 as one atomic point | NOT A CHOICE — commit 21 alone raises a `TypeError`; the evidence is in the receipt |
| resolve the four merge conflicts from the ORCA2 side at every bisect point | NOT A CHOICE about physics — none is under `packages/` or `src/`, and the construction is verified against round 77's merge tree |
| land nothing | ASKED — the order says a statement that is NEMO's for ORCA2 is not scoped |

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

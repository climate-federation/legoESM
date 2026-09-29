# DINO from-rest month regression — bisect, culprit, mechanism, repair, gate

**Status: REPAIRED AND GATED.**  DINO's certified from-rest month had drifted
3.42x away from NEMO.  One line of DINO's recipe owns all of it, the statement
it selects is a statement DINO's compiled NEMO does not execute, and the
landing battery is now able to see the number at all.

| the certified DINO month, day-30 wet 3-D temperature rms vs NEMO | value |
|---|---|
| lane tip before this round (`16d050f62`) | `6.981690958e-03` K |
| lane tip after this round (`a7e853e49`) | `2.040288765e-03` K |
| GitHub main, for reference (`946351212`) | `2.061092384e-03` K |
| the pinned PR #1728 artifact | `2.039e-03` K |

Protocol, identical at every row above and in every bisect row below:
`run_dino.py --config scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml
--grid latlon --recipe nemo_dino_kamm_mlf --n-lon 50 --nemo-faithful-grid
--allow-multiyear --dt 2700 --days 30 --snapshot-every-days 30`, scored by
`twin_nemo_ts_maps.py --run-dino-dir ... --day 30 --nemo-kt 960 --nemo-run
<NEMO>/cfgs/DINO/RUN_TRAJ`.  `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cuda,cpu`,
GPU 0 pinned, one run at a time.  Evidence root
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/dino_regression/`.

---

## 1.  RETRACTION FIRST: the "ten-fold regression" was two different rulers

The round that opened this investigation read DINO's day-30 number as having
gone from `6.889e-04` K to `6.982e-03` K.  **Those two numbers do not describe
the same tree, and the first one was never committed.**

`6.889e-04` K comes from
`/data/abyssal/dbalwada/dino_fromrest_y1/day_gap_after.json`, written
2026-09-11 08:33 by the DINO campaign's round-9 working clone.  That clone's
`packages/ocean/legoesm/ocean/experiments/dino.py` was saved alongside it as
`.dino_full_r9.py`; its blob hash matches **no commit reachable from the lane
tip**.  Diffing it against main shows what it carried that nothing has since:
`shortwave_penetration_ladder = "nemo_2bd"` in place of `"nemo_live"`, and
NEMO's explicit surface restoring in place of the implicit form.  Neither is on
main today and neither is on the lane today.

The archived state itself is real and was re-scored here with the CERTIFIED
comparator against `RUN_TRAJ` kt 960: `6.823904322e-04` K, confirming the
campaign's own number.  So `6.889e-04` K is an honest measurement of a tree
that was never landed, not a baseline any landing regressed from.

**The controlled pair is `2.04`–`2.06e-03` K against `6.98e-03` K — a factor
3.42, not ten.**  Everything below uses that pair.

---

## 2.  The bisect

The lane could not run this protocol at all before 2026-09-16: the card
`scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml` is a GitHub-main file
(added by `aa010f1439`, PR #1728, 09-11) and reached the lane only in the main
merge `524a7487a9` on 09-16.  Lane points earlier than that were therefore
measured by replaying that merge — `git merge 946351212` in a detached
worktree at the lane commit, which is clean at most points — and by applying
round 184's closed-face repair so the run completes.

| sha | what it is | date | day-30 T3D rms [K] | crashes without round 184's fix |
|---|---|---|---:|---|
| `946351212` | GitHub main, the 09-16 merge's main side | 09-16 | `2.061092384e-03` | no |
| `aa010f1439` | main, PR #1728, the card's own landing | 09-11 | `2.061092384e-03` | no |
| `a43e29d6a` (lane idx 175) + main | lane, round ~20s | 09-05 15:03 | `6.982443716e-03` | not measured |
| `8beb3dac5` (lane idx 350) + main | lane, round ~30s | 09-06 20:53 | not measured |
| — same row, measured | | | `6.982425950e-03` | not measured |
| `524a7487a9` | the lane's own 09-16 main merge, first lane commit that can run the card | 09-16 | `6.982426887e-03` | **YES** — `raw-mesh e3w_int must contain only finite values > 0` |
| `16d050f62` | lane tip before this round | 09-29 | `6.981690958e-03` | YES (repaired in round 184) |
| `a7e853e49` | lane tip after this round | 09-29 | **`2.040288765e-03`** | no |

Two things fall straight out of that table.

**The number is flat across the whole lane.**  From 09-05 to 09-29 — three
weeks and several hundred commits of NEMO-literal transcription — the four
lane measurements are `6.982443716e-03`, `6.982425950e-03`, `6.982426887e-03`
and `6.981690958e-03` K: a total spread of `1.1e-04` of the value, against a
step from main of a factor 3.42.  Whatever happened, happened once, early, and
nothing since has moved it by anything comparable.

**The crash and the number are not the same defect.**  The lane crashed on
this card from before 09-16 until round 184 repaired it on 09-27, so the
number was not merely unread in that window, it was unmeasurable.  Section 5
shows the crash's repair is bit-inert on the month.

Two lane points, `61180a677` and `ed62a96ce` (09-04), could not be measured:
replaying the merge there conflicts in
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`, which both
sides were rewriting.  They did not need to be: the owner below was found by
diffing DINO's own card across the merge, and then confirmed by a one-variable
arm at the tip, which is a stronger instrument than any bisect row.

---

## 3.  The culprit, and the statement

`git diff 946351212 524a7487a9 -- packages/ocean/legoesm/ocean/experiments/dino.py`
is seventeen lines.  One of them is

```python
        nemo_prognostic_barotropic_velocity=True,
```

added to `dino_lat_lon_state` by lane commit **`b6a0d6e9f5`, 2026-09-05 08:43,
"feat(ocean): carry NEMO prognostic barotropic velocity"**, and later moved to
the recipe key `"nemo_prognostic_barotropic_state": True` in `DINO_RECIPES`'s
shared base dict (so both DINO cards inherit it) by `0ebdb6d831`.

**The one-variable arm.**  At the lane tip, with that single dictionary value
flipped to `False` and nothing else touched, the 30-day run returns
`|u| = 9.7659e-01`, `|v| = 1.2451e+00`, `|eta| = 6.8831e-01`,
`KE = 7.0466e+12` — main's trajectory to every printed digit
(`9.7660e-01 / 1.2451e+00 / 6.8830e-01 / 7.0466e+12`) — and scores
`2.040288765e-03` K.  That is the whole 3.42x, on one line.

**What the round that landed it said about DINO.**  Its own receipt,
`nemo_testcases_l2_gyre_phase3_round8_receipt.md`, is explicit.  Its
Rule-12 per-card discharge table records DINO as *"not a campaign card; its
deck resolves the vector arm, so it would execute the changed statement —
**OPEN**, not claimed; no DINO gate ran"*.  Its own ledger records *"persistent
`uu_b/vv_b(Kbb)` across legoESM steps | UNASKED boundary exposed by Rule 12 |
next owner; UNMEASURED, no persistence-field implementation attempted"*, and
registers `PERSISTENT_KBB_BAROTROPIC_MEAN_UNMEASURED` in as many words.  The
only DINO row it does carry is a single-solve operand-swap inertness probe
(`INERT, measured: an operand change of 1.249e-01 m/s moves the solve's output
by 3.331e-16 m/s`).  So the lane knew the before-level operand was the open
question, knew DINO would execute the changed statement, measured DINO only
inside one solve, and switched DINO's card on anyway.

This is a lane landing, not a main-side change arriving with a merge.  GitHub
main has never carried it.

---

## 4.  The mechanism

`_carried_nemo_depth_mean`
(`packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py`) gives the
barotropic substep loop **one** carried slot for the depth-mean velocity pair.
The loop seeds `U_bar, V_bar` from that slot instead of reducing the 3-D
velocity, and at the end of the step commits the window-averaged external
solution `U_bar_avg, V_bar_avg` back into the same slot.  The commit's own
comment names the contract that makes this the right time level: *"the outer
`stprk3.F90:213` slot swap makes this returned pair next-step Kbb"*.

**DINO does not run that stepper.**  `cfgs/DINO/cpp_DINO.fcm` declares
`bld::tool::fppkeys key_qco key_vco_3d` — no `key_RK3` — and the compiled
`cfgs/DINO/BLD/ppsrc/nemo/nemogcm.f90:185` therefore calls `stp_MLF`.  Under
the modified leap-frog rotation a value written at the end of step *n* is the
NOW level on step *n+1*, not the BEFORE level.  And DINO sets
`ln_bt_fw = .false.`, so its window seed takes the CENTRED branch and reads
`puu_b(:,:,Kbb)` — the BEFORE level, one rotation older
(`cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90:489-491`; the campaign's own inline
note at `:500-503` states the same thing).

A single-slot carry cannot hold that level.  Selecting the RK3 arrangement on
DINO fed the external window a seed one time level too new, every step, for
three weeks of lane work.

**Why the repair is DINO-shaped and not a shared one.**  GYRE
(`cfgs/GYRE_OMIP_L2_P3_*/cpp_*.fcm`) and both tanks
(`tests/LOCK_EXCHANGE_OMIP_L1/`, `tests/OVERFLOW_OMIP_L1/`) all declare
`key_RK3`.  For them the one-slot carry IS the window seed's level, and the
identity stays required — `nemo_testcase_recipe.py`'s card validator still
demands `nemo_prognostic_barotropic_state=True` for them, and this round does
not touch it.

**Labelled.**  CONFIRMED: the recipe line owns the entire 3.42x (one-variable
arm at the tip); DINO compiles no `key_RK3` and calls `stp_MLF`; DINO's seed
reads the before level; GYRE and both tanks compile `key_RK3`.  PLAUSIBLE, not
separately measured: that the residual is *exactly* the one-rotation offset
rather than that offset plus something else the single slot also gets wrong.
The repair does not depend on which it is — DINO must not select a storage
contract whose enabling statement is absent from its compiled program — but
the distinction is named so nobody quotes the mechanism as measured.

---

## 5.  The closed meridional face is not the month's owner

Round 184 repaired a `NaN` in the meridional QCO closed-face ratio by applying
NEMO's closed-boundary result (`93cae0fae`).  The question was put whether that
zero fill is also what moved the month, or whether it hides a changed operand.

**The fill matches NEMO.**  `dom_qco_r3c` computes `pr3v` on
`DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )` — the last j row is never
computed (`src/OCE/DOM/domqco.F90:165-170`) — and its caller fills that row
through `lbc_lnk( 'dom_qco_zgr', ..., r3v, 'V', 1._wp, ... )` (`:134-135`).
For a closed j boundary the link writes a constant land value, and that value
is zero: `zland = 0._wp` (`lbc_lnk_pt2pt_generic.h90:49`), selected as
`jpfillcst` when there is no neighbour and no self-periodicity (`:104`) and
written into the halo at `:305`.  Round 184's own receipt cites the compiled
equivalents (`DINO/BLD/ppsrc/nemo/domqco.f90:211-215,177-181`,
`lbclnk.f90:1816-1820,1866-1871,2130-2135`); both readings agree.

**And it is bit-inert on the month.**  At the 09-16 merge commit, with round
184's fix REVERTED and instead the whole substep-0 entry-inverse override
removed (`r1_H_u = jnp.where(substep_index == 0, r1_H_u_entry, r1_H_u)` and its
V twin deleted), the 30-day run completes — so the entry-inverse path is
indeed where the `NaN` was born — and its day-30 trajectory is
`|u| = 9.6684e-01`, `|v| = 1.2573e+00`, `|eta| = 6.5844e-01`,
`KE = 7.0993e+12`: identical to every printed digit to the same tree with round
184's fix applied instead.  The two treatments of the closed face are the same
answer, so neither of them can be carrying the 3.42x.

That also settles the crash's own ownership: the operand that goes non-finite
is the entry inverse introduced by `41a5d66223` (09-13, "Transcribe NEMO's
external-window entry inverse depth"), which multiplies the V-face ratio by
`_r1_e1e2v`, infinite on the two closed meridional faces.  A different lane
landing from the one that moved the month.

---

## 6.  The repair

`"nemo_prognostic_barotropic_state": False` in `DINO_RECIPES`'s shared NEMO
base dict, written down in the card rather than left to the library default so
that re-selecting it is a visible edit, with the stepper argument and the
measured cost in the comment beside it.

| the certified DINO month | before | after |
|---|---|---|
| day-30 wet 3-D T rms vs NEMO kt=960 | `6.981690958e-03` K | **`2.040288765e-03` K** |
| day-30 SSH rms | `2.003444252e-02` m | `4.876880015e-05` m |
| day-30 S3D rms | `6.836464249e-04` | `1.343996215e-04` |
| day-30 `KE` | `7.0993e+12` | `7.0466e+12` |

Measured on the committed, clean tip: the comparator stamps
`PROVENANCE git_sha=a7e853e49be2226248ed2ce342a6b1dba04de44a
dirty_tracked_files=0`.  The repaired value is below main's `2.061e-03` K.
Against the pinned PR #1728 artifact it is **close but not equal**: the
artifact prints `2.039e-03` K and this measurement is `2.040288765e-03` K,
0.06 % higher.  The commit message for `a7e853e49` says those two agree "to
the printed digits"; they do not, and this sentence is the correction --
a commit message cannot be edited after the fact, so the receipt carries it.
The bar in the new gate is set from the number measured here, not from the
artifact.

**Non-vacuity.**
`tests/ocean/unit/test_nemo_card_opt_in_defaults.py::
test_no_dino_recipe_allocates_the_rk3_carried_barotropic_pair` asserts the key
is present and `False` in both DINO NEMO recipes, that both resolve `False`,
and that the resolved DINO state holds no `uu_b`/`vv_b` — every one of those
rows fails if the fix is reverted.  Its companion
`::test_the_nemo_testcase_cards_still_require_the_carried_pair` builds the real
GYRE card, asserts it still resolves `True`, and then plants `False` into it
and requires `validate_nemo_testcase_card` to raise — so the DINO row cannot be
made to pass by deleting the requirement everywhere.

---

## 7.  The gate this regression proves was missing

`scripts/validate/ocean_fidelity/dino_1226/dino_fromrest_month_gate.py`.  It
runs the certified protocol and scores it with `twin_nemo_ts_maps.py`; nothing
is re-implemented.  It reads the comparator's own JSON sidecar
(`statistics.T3D.rms_difference`) and compares it to `BAR_T3D_K`, the certified
value plus ten per cent — three times tighter than the regression it exists to
catch, and loose enough to absorb a driver or XLA rebuild.

It costs one 960-step GPU run, so it is **opt-in**: with
`LEGOESM_DINO_MONTH_GATE` unset it prints `SKIPPED` and exits 0.  That is
deliberate, so that `land.sh` can call it unconditionally and the environment
decides which landings pay for it.

**How the operator's `land.sh` should call it.**  After the push-gate battery
passes and before the push, add:

```bash
LEGOESM_DINO_MONTH_GATE=${DINO_MONTH_GATE:-0} JAX_PLATFORMS=cuda,cpu \
  python scripts/validate/ocean_fidelity/dino_1226/dino_fromrest_month_gate.py \
  --work-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/dino_month_gate \
  || { echo "DINO MONTH GATE RED, not pushed"; exit 7; }
```

and set `DINO_MONTH_GATE=1` in the environment for any landing that touches
`packages/ocean/` or `packages/core/`.  A documentation-only landing leaves it
unset and pays nothing.  The autopilots should set it on the round that lands
physics, not on every commit of that round — once per round, on the tip that
ships, is what would have caught this.

**WHAT THIS GATE DOES NOT DO, stated plainly.**  Nothing calls it yet.
`land.sh` lives in the operator's `phase3/claude_rounds/` directory, not in
this repository, so this round can commit the gate and write the call down but
cannot wire it.  Until the operator adds the two lines above, the process gap
is DOCUMENTED, not CLOSED, and a DINO regression can still land silently.  The
commit message for `a7e853e49` says this "adds the landing gate whose absence
let the regression stand for three weeks"; it adds the gate, it does not yet
add the call.  An independent review caught that overclaim and it is recorded
here rather than quietly left standing.

The gate's own verdict is a pure function with a self-test
(`--self-test`, and `tests/ocean/fidelity/test_dino_fromrest_month_gate.py`)
that requires `6.981690958e-03` K to FAIL the bar and the certified value to
pass.

**Why no gate saw this for three weeks.**  The six-file push gate contains no
DINO run.  The card battery contains DINO's mesh, from-rest kt-1 and step-1
gates — all initial-state gates, none of which integrates.  The 90-day DINO
twin starts from NEMO's developed day-180 restart, so it never exercises the
from-rest window.  And for most of the window the from-rest card crashed, so
even a gate would have reported a crash rather than a number.  Nothing was
read wrongly; the quantity was simply never computed.

---

## 8.  What else this repair moves, and what it does not

**DINO's DEVELOPED-STATE numbers will move, and this round did not re-measure
them.**  The config, not the state, decides whether the carried pair is used
(`_carried_nemo_depth_mean` returns `None` on a card that does not select it,
even when the state holds the pair), and
`nemo_state_bridge.py:275,917` still builds bridged states with the pair
allocated.  So nothing raises and nothing is left half-initialised — but the
90-day developed twin (`kamm_twin_90d.py` + `acceptance_gate_90d.py`), whose
certified rows were last recorded as `ACC 65.390274 Sv`, upper contrast
`-0.288146`, deep contrast `-0.011261`, southern sigma max `0.909348`, was
measured WITH the pair active.  Those rows are expected to move and are
**NOT re-certified here**.  They should be re-run and re-pinned on the next
DINO round; the direction is toward the pre-2026-09-05 behaviour, which is the
behaviour every earlier DINO certification used, but that is an expectation,
not a measurement.

**The forward-Euler DINO card** inherits the same recipe value from the shared
base dict and was not measured either: it has its own step-5 implicit-solve
instability (round 184) and cannot complete the month, so there is no number
to take.  Written down at the line rather than left implicit.

---

## 9.  A finding this round is NOT acting on

`nemo_testcase_recipe.py`'s card validator requires
`nemo_prognostic_barotropic_state=True` for every NEMO test-case card.  That is
correct for GYRE and both tanks, which compile `key_RK3`.  It would be wrong
for any future card that does not — ORCA2's build was not checked here.  One
line, reported, not started.

---

## 10.  Review

Both reviewers ran adversarially on the diff, and both came back negative on
the first pass.  Everything they found is fixed or written down; nothing was
argued with.

**Claude `code-reviewer`, fresh context: DO-NOT-SHIP on the first pass.**
One blocker: the new companion test reached for `card.config`, which
`NEMOTestcaseCard` does not have (its configuration is at
`recipe.model_config`), so it raised `AttributeError` before ever reaching the
validator — the non-vacuity it claimed for the GYRE side was not there at all.
It ran both tests by hand to establish that.  Fixed, and both tests were then
run directly: the DINO row passes and FAILS when the recipe value is planted
back to `True`; the companion passes and its planted card does reach the
validator.  Two more, both accepted: the gate is not wired to anything, and
the receipt it cites did not exist at the time.  It independently re-derived
every NEMO citation — including the load-bearing `stpmlf.f90` rotation — and
called the mechanism CONFIRMED rather than correlated.  On the repair
direction it argued for disabling the pair now and deferring a second MLF slot
as unmeasured physics.

**codex `exec --sandbox read-only`: DO NOT SHIP**, six findings.  It agreed on
the broken companion test, the unwired gate, and the missing receipt, and it
independently confirmed the mechanism from the compiled sources.  Three
additions, all acted on:

* the gate documented `JAX_ENABLE_X64=1` instead of setting it, and
  `run_dino.py` only WARNS when x64 is off — an opted-in call that forgot the
  variable would have scored a different model against the bar.  The gate now
  sets it for both child processes.
* *"single slot is correct only under RK3"* is overclaimed: a forward scheme
  could also carry after-to-next-now correctly.  The comment now says what was
  actually checked — right under RK3's `Nbb <==> Naa` swap, wrong under the
  MLF rotation — and quotes both statements.
* the shared base dict also turns the pair off for the forward-Euler DINO
  card, which was not measured.  Written down at the line, with the reason
  (that card has its own step-5 implicit-solve instability and cannot complete
  the month).

**Where the two reviewers DISAGREE, named rather than averaged.**  codex holds
that disabling the pair is *"a sound emergency rollback ... but not the
faithful final repair"*, because NEMO really does carry the pair
prognostically and fidelity would need a second slot rotated with the MLF time
levels.  The Claude reviewer holds that growing that slot now is *"scope creep
against an unfalsified design"*.  They agree on what to do today and differ on
what to call it.  The discriminating fact is that the reduction-based fallback
is what produced every certified DINO number to date, including the pinned
#1728 artifact this repair lands on — so it is the known-good path, and the
second slot is real follow-up debt with no measurement behind it yet.  The
receipt calls the repair a rollback to the certified path and names the second
slot as open work; nobody should read this round as having transcribed NEMO's
prognostic pair for an MLF card.

---

## 11.  Choices made this round

| choice | status |
|---|---|
| turn `nemo_prognostic_barotropic_state` off on the DINO cards — which removes a prognostic pair from DINO's carried state | **ASKED**: the round's instruction was to fix the regression at the root and prove DINO's day-30 returns to at or below the pre-regression value |
| add a from-rest DINO month gate, opt-in behind an environment variable | **ASKED**: instructed |
| bar = certified value + 10 % | **UNASKED**, offered for revert.  Any bar is a choice; this one is three times tighter than the regression and loose enough for a driver rebuild.  Name a different one and it is a one-line edit |
| bisect by replaying the lane's own 09-16 main merge at each point, rather than by cherry-picking the card backwards | UNASKED, method not physics; it changes which commits are testable, not what any of them computes |
| scoring the archived round-9 campaign state with the certified comparator to settle where `6.889e-04` came from | UNASKED, pure measurement of an existing artifact |

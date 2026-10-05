# NEMO testcase fidelity: round 197 / VORTEX round 13 — the barotropic Coriolis statement, named and held

**Status: HELD.** Nothing lands in production beyond a private, proven-inert
instrument.  The round NAMES the compiled statement that owns the barotropic
window — the frozen 2-D Coriolis coefficients are divided by the wrong
vertical scale factor at every vertex with a dry neighbour — and the fix is
kept as a patch because it trips the unchanged two-ULP ratchet.  That is a
DECISION for the user, stated in OPEN.

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round197/`.
Frozen preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_round197.md`,
committed as `63fca3673` before any measurement.
Decision 82 (user): round 194's two-solve candidate stays HELD and the
two-ULP ratchet is unchanged.

## Compiled program

Inside one sub-time-step of the split-explicit solve, NEMO forms the surface
pressure gradient
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:498`), then the 2-D
Coriolis trend
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:503`,
`CALL dyn_cor_2D( ua_e, va_e, zu_trd, zv_trd )`), then the explicit bottom
stress, then the vector-form velocity update
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:535`).  The operands
handed to the Coriolis routine are the MID-STEP extrapolated velocities set
at `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:389`, and the
routine itself
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1112`) is four
multiply-adds against eight coefficients frozen once in
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:933`.

Those coefficients carry the planetary vorticity as `ff_f` divided by a
vertical scale factor at the F point
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960`), and the array
in that denominator is `e3f_0vor`, which is NOT the F-point reference
thickness.  It is built in
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:892`, and with this
deck's resolved `nn_e3f_typ = 0` (`namelist_ref` line 1072; `namelist_cfg` does not override it) it is the four surrounding T cells' MASKED thickness divided by
FOUR — by four, not by the number of wet cells, which is the
`nn_e3f_typ = 1` branch —
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:897`, with a sweep that
restores the full thickness at a fully dry vertex
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:920`).  At a vertex
with one dry neighbour the denominator is therefore three quarters of the
thickness and the coefficient is four thirds of what the plain thickness
gives.

## The new instrument, and its zero-change proof

Round 196 could not substitute a per-substep operand: no override existed
inside the compiled scan.  This round adds one, in the style of the existing
stage-2/3 and loop-entry hooks.  It is a static closure capture keyed by the
scan's own substep index; `None` is the production value, no constructible
model configuration selects it, and the gate is a Python `if` on a
closure-captured static, so an unset hook leaves the traced program
unchanged.  Two operands are reachable: the Coriolis trend and the surface
pressure gradient.

**Unset, it changes nothing.**  The production barotropic walk re-run at
this round's commit reproduces round 196's walk EXACTLY: all **1059 rows**
agree on `cells_unequal`, `max_abs`, `normalized_max_abs` and `status`, all
**336 scalars** agree, and the per-cell Coriolis table and the
first-non-bit-boundary record are identical objects
(`phase3/round197/spgts_walk_kt1_production.json` against
`phase3/round196/spgts_walk_kt1.json`).  The unit controls add a direct
step-level check: a model built with both override fields explicitly `None`
returns a step byte-identical to the model built with no hooks at all.

**Set, it binds, and the controls can fail.**  Handed the model's OWN
per-substep trend the step is byte-identical to the traced production step;
handed zeros, or handed the FIRST substep's trend held for the whole window,
it is not.  That second control is what distinguishes a correct per-substep
index from one frame reused 48 times.

**An instrument caveat this round measured and the earlier ones did not.**
Exposing the substeps compiles the barotropic loop as a `scan` where the
plain step uses a `fori_loop`; on `VORTEX_VEC-zco` the two steps differ by
`3.33e-16` at the worst leaf.  That is a third of the bar, so it cannot
touch any conclusion drawn at `1e-08`, but it IS carried by the floor rows
of every walk in rounds 192-197, and the correct reference for a
trace-reading arm is the traced step, not the plain one.  The unit controls
are written that way.

## The substitution (round 196's OPEN item 1)

One variable, in the production-jitted step, seeded from NEMO's own recorded
`kt=1` step entry: NEMO's recorded `cor_u`/`cor_v` replace legoESM's at
every one of the 48 substeps and nothing else changes.  The arm is
self-checking and the check passes — the walk's `cor.u` and `cor.v` rows
become bit-exact at every substep, so the substitution bound.

| end of the barotropic window, `kt=1` | production | NEMO's own per-substep trend |
|---|---:|---:|
| updated velocity `u` | `1.24589062771389985e-08` | `1.94289029309402395e-16` |
| updated velocity `v` | `1.05595161158665909e-08` | `1.52655665885959024e-16` |
| after-SSH | `3.70900976994493190e-08` | `1.88737914186276612e-15` |

**The window closes.**  Against the preregistered falsifiers (`<= 1e-15`
confirms, `>= 1.2e-09` refutes), P3 is CONFIRMED at the velocity rows.

**But a closed window does not by itself name the operator, and the
preregistration's wording was too generous.**  The trend sits inside a
recurrence: clamping it at every substep also removes the feedback that a
merely different OPERAND would have produced.  The discriminating test is
the one the next section runs.

## Operator or operand: the cross test

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_round197_cor_operator.py`
applies legoESM's own frozen coefficients to NEMO's own recorded mid-step
velocity, at every substep, and compares against NEMO's own recorded trend.
Before it is used on the unknown, it is run against a KNOWN answer: the same
call on legoESM's own mid-step velocity must reproduce legoESM's own trend
bit for bit.  It does, at all 48 substeps, `selfcheck_max_abs = 0.0`, and
the probe refuses rather than reporting if it does not.

With the card as it ships, the cross term is NOT equal to NEMO's.  The
disagreement is on exactly **238 faces** — `2x61 + 2x60 - 4`, the perimeter
of the 61x60 wet u-face window inside the 63x63 closed box — at every
substep, with a relative error that
agrees to fifteen digits at the worst face of every substep sampled
(`0.2506283612741221`, `...219`, `...186`), although the worst face moves (`[61,28]` at substep 1,
`[61,36]` at substep 48).  A fixed relative error at a fixed set of faces is
a frozen coefficient, not an operand.

| substep | cross-term max abs | production trend max abs | operand (mid-step velocity) max abs |
|---:|---:|---:|---:|
| 1 | `2.299e-53` | `3.388e-21` | `5.551e-17` |
| 16 | `1.818e-26` | `6.776e-21` | `8.327e-17` |
| 28 | `2.856e-17` | `2.856e-17` | `9.817e-16` |
| 48 | `7.241e-11` | `7.237e-11` | `9.996e-09` |

The ring's own values start at `1e-53` — the exponentially small far field
of a Gaussian vortex in the middle of the box — and grow by forty orders of
magnitude as the vortex's radiated barotropic gravity wave reaches the wall.
From substep 28 the ring's coefficient error IS the whole production trend
difference to three digits.  **That is what round 196's "2.17x per substep"
growth is: not an unstable recurrence — round 196's one-ULP conditioning
control already showed the recurrence amplifies by 2.0 over the whole window
— but a wave arriving at a boundary where the operator is wrong.**  The two
readings round 196 recorded as disagreeing (a flat 2-3 times `f` per cell,
80 times the operator's norm bound at the field maxima) are both artifacts
of a ratio taken across a four-point stencil, and neither of them decided
anything; this test did.

## The statement, and what the fix does

The VORTEX card hands the frozen-coefficient builder the plain reference
thickness where NEMO's `dyn_cor_2D_init` divides by `e3f_0vor`.  Building
`e3f_0vor` the way `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:897` builds it — the masked
four-cell sum over four, with the restore at a fully dry vertex
(`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:920`) — and changing
NOTHING else:

| measurement, `VORTEX_VEC-zco`, `kt=1` | card as it ships | with `e3f_0vor` |
|---|---:|---:|
| cross term against NEMO's trend, worst substep | `7.240859558674367e-11` | **`0.0` (bit, 48/48 substeps, 0 faces)** |
| production trend difference, worst substep | `7.236813114148063e-11` | `1.0164395367051604e-20` |
| mid-step velocity difference at substep 48 | `9.996e-09` | `1.665e-16` |
| end-of-window velocity `u` | `1.24589062771389985e-08` | `1.66533453693773481e-16` |
| end-of-window velocity `v` | `1.05595161158665909e-08` | `1.38777878078144568e-16` |
| end-of-window after-SSH | `3.70900976994493190e-08` | `2.55351295663786004e-15` |

The barotropic Coriolis operator is now bit-identical to NEMO's, and the
barotropic window is AT-BAR in both velocity components.

## The certified trajectory, before and after

Both VORTEX cards, `kt=1..10`, normalized max abs, same oracle, same gate,
one variable (`phase3/round197/traj_*_{before,after}.json`).  **34 of the
vector card's 50 rows move and 39 of the flux card's do; the table below
shows seven of them, chosen to be the largest, so it is NOT the ledger.**
The ledger is this: of the rows that move, **11 worsen on the vector card**
(`kt3 u`, `kt3 v`, `kt4 T`, `kt4 u`, `kt4 v`, `kt5 T`, `kt5 u`, `kt6 T`,
`kt7 T`, `kt9 v`, `kt10 T`; the largest is `kt4 u`,
`2.3727e-06 -> 2.3866e-06`, 0.6%) and **6 worsen on the flux card**
(`kt3 T`, `kt10 T`, and four salinity rows at `kt4`, `kt6`, `kt8`, `kt9`,
each by exactly one quantum of their own resolution).

| row | VORTEX_VEC-zco before | after | VORTEX-zco before | after |
|---|---:|---:|---:|---:|
| kt2 ssh | `3.709e-08` | `2.831e-15` | `3.709e-08` | `2.665e-15` |
| kt3 ssh | `6.506e-06` | `1.178e-07` | `6.503e-06` | `1.268e-08` |
| kt5 ssh | `8.027e-06` | `6.903e-07` | `7.978e-06` | `9.349e-08` |
| kt10 ssh | `5.357e-06` | `4.880e-07` | `5.336e-06` | `1.899e-07` |
| kt5 u | `3.712e-06` | `3.717e-06` | `1.061e-06` | `3.590e-07` |
| kt10 u | `4.865e-06` | `4.851e-06` | `2.464e-06` | `4.888e-07` |
| kt10 v | `4.725e-06` | `4.717e-06` | `2.192e-06` | `4.860e-07` |

The flux card's velocity rows improve by 3-5x and both cards' sea surface
rows by one to four orders of magnitude.  The vector card's `kt=2` velocity rows do not move AT ALL -- `3.3693e-06`
and `3.3370e-06` before and after -- because that row's owner is the stage
vertical velocity round 194 named and this round does not touch it; its
kt>=3 velocity rows move by under 1% in both directions.

## Why it is HELD: the gate that is red

The cellwise oracle-relative move gate, run with the SAME oracle on both
sides:

```
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=2132967438.5
  first_over_bar={'fields': ['T','u','v','ssh'], 'kt': 2}->{'kt': 2, 'fields': ['T','u','v','ssh']} plant=None   (VORTEX_VEC-zco)
ORACLE_RELATIVE_COMPARE FAIL: rows=50 max_worsening_ulps=540700912.625
  first_over_bar={'fields': ['T','u','v','ssh'], 'kt': 2}->{'kt': 2, 'fields': ['T','u','v','ssh']} plant=None   (VORTEX-zco)
```

The limit is 2 ulps, so this is red by its own letter and the fix is NOT
landed.  What is behind the number, stated so the decision can be taken on
evidence rather than on the headline: across the 50 certified rows,
**1,015,156 cells improve and 143,924 worsen** on the vector card
(1,045,802 against 112,378 on the flux card), the worst row is the vector
card's `kt10` sea surface with **106 cells worse and 3615 better of 3721**,
and `first_over_bar` does not move earlier.  **One certified row crosses the
bar**: the flux card's `kt6` salinity goes AT-BAR to DEBT by one quantum
(`8.1205e-16 -> 1.0151e-15`), and its `kt7` salinity crosses back the other
way (`1.0151e-15 -> 8.1205e-16`) -- the same one-quantum salinity pair that
held round 194's candidate.  The vector card has no status change.  The largest move the fix makes
in any field (`3.7e+10` row-scale ulps) is more than an order of magnitude
larger than the largest worsening.

Decision 82 says the ratchet is unchanged, and nothing here weakens it.  The
fix is kept as
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l1_vortex_round197_e3f_0vor_held.patch`.

## Predictions, kept with their verdicts

* **P1 — the hook is zero-change when unset.** **CONFIRMED** for (a): 1059
  rows, 336 scalars, the exit frame and the Coriolis table bit-identical
  across a commit change, plus a step-level byte-identical control.
  **(b) WAS NOT RUN and is therefore NOT confirmed as preregistered.**  The
  GYRE certified ladder and the round-191 VORTEX-vector registry were not
  re-run for the hook alone.  What stands in their place is weaker and is
  stated as such: the three new `if` statements ARE inside the shared
  barotropic loop every card traces, so they are no-ops rather than absent;
  they test a closure-captured `None` at trace time, no constructible model
  configuration can set it, and the VORTEX walk's 1059 rows are bit-identical
  with the hook present.  The card edit that would actually move certified
  numbers is not landed.  A round that lands the fix must run the full
  ladder.
* **P2 — the substitution binds and says so.** **CONFIRMED.** `cor.u` and
  `cor.v` bit-exact at all 48 substeps in the arm, and not in production.
* **P3 — the end-of-window velocity.** **CONFIRMED** against the
  preregistered threshold (`1.943e-16 <= 1e-15`), and the receipt records
  that the threshold was the wrong discriminator on its own: the cross test
  is what separates the operator from its operand, and it is what named the
  statement.
* **P4 — the pressure gradient would not close the window.** **NOT RUN, and
  it is no longer the next operand.** The cross test named the Coriolis
  coefficients outright, so substituting the pressure gradient would have
  answered a question that was already closed.  Its override is built and
  tested and is one flag away if a later round needs it.
* **P5 — nothing lands unless a cited statement closes every gate.**
  **CONFIRMED.** The cited statement does not close the move gate, so
  nothing landed.

## Gates and tests

* **Production diff.** The round's only `packages/` change is the two
  private hook fields and the three `if` statements that read them; the
  card fix is restored and held as a patch
  (`git status --porcelain` clean after the restore).
* **Citation gate** on this receipt: `"status": "PASS"`,
  `"citations_found": 10`, `"unmapped_citations": []`, no failures, no map
  entry failing its own audit, and all nine of its planted-shift self-tests
  fired (`phase3/round197/citations.json`).  Its planted control, shifting
  the cited `e3f_0vor` construction by two lines, fires
  `SYMBOL-NOT-AT-LINE ... that symbol identifies line 897` and exits 1
  (`phase3/round197/citations_plant.json`).  `dynvor.f90` for this build is
  newly registered with the gate, the same way round 196 registered
  `dynspg_ts.f90`.
* **Citation re-anchor.** The hook adds eleven lines before the model file's
  first cited line and fourteen more before its barotropic call site, so
  **41 citations were re-anchored by rigid shift** -- both endpoints by the
  same delta, pinned extents unchanged -- and the gate re-verified every
  endpoint symbol afterwards.  No citation was weakened or deleted.
* **Round-197 unit controls**: `3 passed in 77.40s`
  (`tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round197_substep_override.py`),
  with three non-vacuity arms: zeros, the first substep's trend held for the
  whole window, and the trend rolled by one substep.
* **Measurement provenance.** An independent reviewer found that the two
  artefacts carrying this round's headline recorded only `<sha>-dirty` and
  no diff hash, so their numbers were not pinned to the patch that produced
  them.  Both emitters now stamp the worktree and both arms were re-run:
  `cor_operator_kt1_e3fvor.json` and `spgts_walk_kt1_e3fvor.json` carry
  `diff_sha256 = fa62e5dfe8760eaa8ef9...`, which is the sha256 of
  `manifests/nemo_testcase_l1_vortex_round197_e3f_0vor_held.patch` itself
  and the same hash the trajectory and move-gate artefacts record.  The
  production walk is stamped clean at
  `dd9548080f95` and reproduces `1.2458906277138998e-08`; the before
  trajectory arms were taken with the patch stashed out on the same
  commit.

## Independent adversarial review (fresh reviewer, this round)

Codex is reserved for the ORCA2 lanes, so the mandatory second opinion was a
fresh reviewer agent given the diff, the evidence root and the claims with no
knowledge of how they were produced.  Its verdict was **DO-NOT-SHIP the
receipt as written, the science holds**, and every finding is accepted and
fixed above.

* **DEFECT (accepted, fixed).** The trajectory table said rows that do not
  move are omitted; 27 of the vector card's moved rows and 33 of the flux
  card's were omitted, including every row that worsens.  The full ledger of
  worsening rows is now in the text and the table is labelled as a selection.
* **DEFECT (accepted, fixed, and it is material to the decision).** The flux
  card's `kt6` salinity crosses AT-BAR to DEBT; the receipt reported only
  that `first_over_bar` does not move.
* **DEFECT (accepted, fixed).** The two artefacts carrying the headline were
  not pinned to the held patch.  Both emitters now stamp the worktree and
  both arms were re-run; the recorded `diff_sha256` is the patch's own
  sha256.
* **DEFECT (accepted, corrected).** "The vector card's velocity rows do not
  move" contradicted the table; only its `kt=2` rows do not move.
* **DEFECT (accepted, corrected).** The per-face relative error agrees to
  fifteen digits across substeps, it is not one constant.
* **DEFECT (accepted, corrected).** P1(b)'s preregistered falsifier -- a GYRE
  ladder run -- was not run, and "confirmed by construction" was substituted
  for it.  It is now recorded as NOT CONFIRMED, with the weaker argument
  named as weaker.
* **GAP (accepted, closed).** The index controls could not see a trend stack
  rolled by ONE substep; that arm is now a third non-vacuity control and the
  module is `3 passed in 77.40s`.
* **DEFECT (accepted, corrected).** 238 is the perimeter of the 61x60 wet
  u-face window, not of the 63x63 box.
* The reviewer independently CONFIRMED: every one of the nine NEMO citations
  says what the receipt says it says; the deck resolves `nn_e3f_typ = 0` and
  the EEN branch is the one that runs; the held patch transcribes the
  compiled stencil faithfully, including which four T cells and NEMO's
  association, and `e3f_0` has exactly one consumer so the arm is one
  variable; the hook is inert unset and reachable from no configuration; the
  override sits at the compiled boundary it claims and no later statement
  rebinds the trend (`ln_drg_OFF = .true.` on this card); the probe's
  conventions survive, because a bit-exact cross term admits no padding or
  stagger error; and every number in the receipt reproduces from the
  committed JSON.

* **Push gate** (`land.sh gyre`, the lane's authoritative battery).  Its
  FIRST run was RED -- `1 failed, 134 passed in 958.52s`, the citation
  gate's own test, because re-anchoring the map had left the default
  receipt's citations on the pre-hook line numbers.  The same old-to-new
  map was applied to the two receipts that cite those lines (no citation
  weakened, none removed) and the second run is green:
  `135 passed in 1087.35s (0:18:07)`, then
  `DINO month gate: day-30 wet 3-D T rms vs NEMO kt=960: 2.040288765e-03 K
  against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS`,
  then `PUSHED f7f57e17ed4d`.  The DINO gate ran because the round changes
  `packages/`.

## Landing verdict: HELD (instrument + named statement + held fix)

## Option choices made this round

| choice | ASKED / UNASKED | note |
|---|---|---|
| every NEMO deck option | ASKED | unchanged from rounds 3-12; Decisions 69, 70, 73, 75 |
| `nn_e3f_typ = 0` | n/a | resolved by the deck's own namelist_ref line 1072, not chosen here |
| which two operands the override reaches | n/a | instrument choice, named in the frozen preregistration |

Nothing on the UNASKED list.

## OPEN — round 198

1. **DECISION NEEDED (Decision 83).** The `e3f_0vor` fix is correct — the
   barotropic Coriolis operator becomes bit-identical to NEMO's and the
   barotropic window goes AT-BAR — and it trips the unchanged two-ULP
   ratchet while improving a million cells and worsening a hundred
   thousand.  Land it and re-certify both VORTEX cards' 50-row registry at
   the new values, or keep it held?  Nothing else in the VORTEX ladder can
   move until this is answered, because every later residual sits on top of
   it.
2. **The same operand is carried by the other cards' frozen barotropic
   coefficients, and that is UNMEASURED.**  The GYRE, ORCA2 and DINO cards
   build their literal barotropic coefficients from the same reference
   thickness; whether their `e3f_0vor` differs from it depends on their
   coastlines, and for GYRE it would move certified numbers.  This is a
   finding, not a change: it is reported here and measured in its own
   round, never carried silently into one.
3. Round 194's held two-solve candidate is still held (Decision 82), and
   the vector card's `kt=2` velocity owner is still the stage vertical
   velocity.  Re-test it once item 1 is answered.
4. The flux card's own `kt=2` owner and Decision 74's 30/15/10-km ladder
   stay blocked behind the vector card's `kt=2` rows.

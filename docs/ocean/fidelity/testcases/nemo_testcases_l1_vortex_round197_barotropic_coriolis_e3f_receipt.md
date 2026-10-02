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
deck's resolved `nn_e3f_typ = 0` (namelist_ref:1072; `namelist_cfg` does not
override it) it is the four surrounding T cells' MASKED thickness divided by
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
disagreement is on exactly **238 faces** — `2x61 + 2x60 - 4`, the boundary
ring of the 63x63 closed box — at every substep, with a relative error that
is the SAME constant `0.2506283612741219` at the worst face of every
substep sampled, although the worst face moves (`[61,28]` at substep 1,
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
`e3f_0vor` the way `dynvor.f90:897` builds it — masked four-cell sum over
four, with `dynvor.f90:920`'s restore at a fully dry vertex — and changing
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
one variable (`phase3/round197/traj_*_{before,after}.json`).  Rows that do
not move are omitted.

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
rows by one to four orders of magnitude.  The vector card's velocity rows do
NOT move, because its `kt=2` owner is the stage vertical velocity round 194
named and this round does not touch it: `3.3693e-06` before and after.

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
and `first_over_bar` does not move earlier.  The largest move the fix makes
in any field (`3.7e+10` row-scale ulps) is more than an order of magnitude
larger than the largest worsening.

Decision 82 says the ratchet is unchanged, and nothing here weakens it.  The
fix is kept as
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l1_vortex_round197_e3f_0vor_held.patch`.

## Predictions, kept with their verdicts

* **P1 — the hook is zero-change when unset.** **CONFIRMED** for (a): 1059
  rows and 336 scalars bit-identical across a commit change, plus a
  step-level byte-identical control.  (b) is CONFIRMED BY CONSTRUCTION
  rather than by a GYRE run: the production diff this round contains no
  line any GYRE, DINO, LOCK_EXCHANGE or OVERFLOW path reads — the two new
  hook fields default to `None` and the three `if` statements that read
  them are the only consumers — and the land gate's own battery, which
  includes the NEMO recipe and prognostic-barotropic-state modules, is
  green.  The VORTEX card edit that WOULD move GYRE-adjacent numbers is
  not landed.
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
* **Citation gate** on this receipt: quoted below with the focused battery.
  `dynvor.f90` for this build is newly registered with the gate, the same
  way round 196 registered `dynspg_ts.f90`.
* **Round-197 unit controls**: `3 passed in 58.63s`
  (`tests/ocean/fidelity/test_nemo_testcase_l1_vortex_round197_substep_override.py`).
* **Measurement provenance.** The production walk and both substitution
  arms were run on the clean commit `721070e2aa5ff2f82e3c149af6bc4b6e13de5ac8`;
  the trajectory and cross-test arms that needed the held patch applied are
  stamped `allow_dirty_escape_used: true` with the patch's own
  `diff_sha256` recorded in each artefact, and the before arms were taken
  with the patch stashed out on the same commit.

## Independent adversarial review

Recorded below with the push gate.

## Landing verdict: HELD (instrument + named statement + held fix)

## Option choices made this round

| choice | ASKED / UNASKED | note |
|---|---|---|
| every NEMO deck option | ASKED | unchanged from rounds 3-12; Decisions 69, 70, 73, 75 |
| `nn_e3f_typ = 0` | n/a | resolved by the deck's own `namelist_ref:1072`, not chosen here |
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

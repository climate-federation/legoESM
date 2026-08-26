# PRE-REGISTRATION — the 90-day START-MODE A/B (#1455)

**Written and committed BEFORE either 90-day arm was launched.** Commit under
test: `47e3b9751` (branch `fix/twin-euler-start-default`), which flips the twin
harness's leap-frog before-level bridge ON by default and puts the legacy
forward-Euler start behind `--legacy-euler-start`.

## What is being measured, and what is NOT

**One variable: the twin's time-integration START.** Both arms are the shipped
`nemo_dino_kamm_mlf` card, the same commit, the same fp64 policy, the same
vertical ladders (`both`, checked by content hash, not by label), the same
seasonal clock, 90 days from the day-180 NEMO restart.

- **Arm B (bridged)** — the new default. `state.{T,S,u,v,eta}_before` seeded
  from NEMO's restart `tb/sb/ub/vb/sshb`.
- **Arm E (euler)** — `--legacy-euler-start`. The model seeds a local
  `before := now` (NEMO's cold-start convention, `istate.F90:97-99`) and
  returns before the step-1 after-level reconciliation.
- **Arms B' / E'** — one member each, identical to B / E plus
  `--perturb-seed 1` (a 1e-14 relative multiplicative kick on the now-level T).
  These exist ONLY to measure the within-arm noise floor **at this commit**,
  so the comparison bar is measured rather than transferred from the older
  4-member ensemble.

**NOT being measured:** whether any recorded campaign number is contaminated.
It is not — see the audit below. This A/B bounds what the Euler start COSTS,
which is the number needed to re-score any Euler-start artifact and to justify
the default flip.

## Pre-run facts, measured before launch

1. **The premise that motivated this work is REFUTED.** "Every recorded twin
   ran the Euler start" is false. Audited every run log under `results/`
   containing a twin build: **18 of 18 carry the before-level bridge
   verification line**, which prints if and only if the bridge ran — all four
   90-day acceptance-gate arms included. `acceptance_gate_90d.py` has passed
   the flag by default since 2026-08-09 (`9caa61f3e`) and `PREREG_verdict360.md`
   specifies it. The recorded gate and verdict-year numbers were never
   Euler-start; the ad-hoc wave-lane runs were.
2. **The Euler start does not raise on this card.** A 1-day
   `--legacy-euler-start` run at fp64 completes normally
   (`results/dino_1455_startmode/probe_euler_1d.log`, HEAD `560ca33df`). The
   twin docstring's "raises ValueError at step 0" predates the #1317
   Euler-branch seeding and is retracted.
3. **The size of the perturbation the start mode applies**, read from the
   restart itself (`DINO_00005760_restart.nc`, wet cells): the two arms'
   step-0 before levels differ by
   `T: max 6.31e-2 K, rms 3.57e-4 K (3.38e-5 relative)`,
   `u: max 1.01e-2 m/s, rms 8.72e-5 m/s (2.37e-3 relative)`,
   `ssh: max 1.69e-4 m, rms 1.65e-5 m`.
   That is **~3.4e9 times the 1e-14 relative kick** the noise-floor ensemble
   uses. The two arms are NOT near-identical trajectories.

## PREDICTION (registered)

**The endpoint metrics move little.** Specifically: no gate metric's
across-arm difference exceeds its 5x gate threshold, and both arms return the
same PASS/FAIL pattern at level 5. Basis: the 5-day evidence says the effect is
LAUNCH-concentrated (impulse response 65x better bridged at launch, only 7.6x
better late, and the day-5 worst cell marginally WORSE bridged).

## THE BAR, and why it is this one

The obvious bar — the recorded 1e-14 ensemble floor (ACC max-pairwise spread
**1.15e-5 Sv** at day 90) — is **the wrong bar on its own**, and registering it
as the decision bar would manufacture a "systematic effect" verdict out of
nothing. Fact 3 above shows the start mode perturbs the state ~1e9 times harder
than the ensemble kick, so the two arms are separated at step 1 by a finite,
structured difference, not an infinitesimal one. A difference far above the
1e-14 floor is therefore the EXPECTED outcome of chaos alone.

The opposite extrapolation is also refused. Scaling the floor linearly by the
amplitude ratio predicts ~1e4 Sv of spread, which is absurd, and the campaign's
own record contradicts it directly: a real physics change (the bottom-drag
time-level fix, worth ~1.26e-3 Sv/step in-loop) moved **no** gate metric past
its floor at day 90. So the day-90 ACC is NOT a saturated chaotic variable —
it is a strongly constrained integral on which structured changes produce small
systematic shifts. Neither "the floor" nor "the saturated spread" is the bar;
the honest bar is the one the campaign already uses to decide whether a change
MATTERS, read alongside a floor measured at the same commit.

Registered bars, per metric:

| bar | value | role |
|---|---|---|
| **F** — within-arm floor, MEASURED at this commit | `|B - B'|` and `|E - E'|` | what a 1e-14 kick alone buys here |
| **1x floor** | ACC 0.091 Sv, upper 1.1e-4, deep 4.5e-5, sigma max/mean 9.5e-5 kg/m3 | the recorded #1492 2.1 ensemble floor |
| **5x gate** | 5 x the above | the level at which the campaign's gate calls a change consequential |

The **channel-band transport** has no floor of its own; it borrows ACC's, and
the tool declares that borrowing (`ACCBAND_FLOOR_IS_A_TRANSFER`) rather than
reusing it silently.

## DECISION RULE (registered, in order)

Let `D_k = |B_k - E_k|` per metric `k`, and `F_k = max(|B-B'|_k, |E-E'|_k)`.

1. **CONFIRMS THE PREDICTION** — "the start mode has no consequence for the
   campaign's conclusions": every `D_k < 5 x floor_k` AND arms B and E return
   the identical PASS/FAIL pattern from the acceptance gate at level 5.
   This is a CONSEQUENCE claim, never a "no difference" claim.
2. **NO SYSTEMATIC EFFECT DETECTABLE**: additionally `D_k <= 3 F_k` for every
   metric — the across-arm difference is not distinguishable from the
   within-arm floor measured at the same commit.
3. **SYSTEMATIC AND MATERIAL** — the Euler start was biasing the recorded gaps:
   any `D_k > 5 x floor_k`, or the two arms disagree on the gate's PASS/FAIL
   pattern. Report DIRECTION (does the Euler arm sit closer to or further from
   NEMO?) and SIZE, loudly, and escalate the verdict-year re-run decision to the
   user with its cost.
4. **UNDERPOWERED** — the honest middle: `3 F_k < D_k < 5 x floor_k`. Reported
   as a BOUND. With n=1 pair this cannot separate a systematic shift from
   trajectory divergence; separating them costs an N-member ensemble per arm
   (~4.5 min of one V100 per member) and is NOT run here.

**Secondary, explicitly weak:** the per-metric direction (which arm is closer
to NEMO). Under pure divergence this is a coin flip per metric, so all six
agreeing is p ~ 1/32 — notable, never decisive. Any direction result is
labelled PLAUSIBLE.

## What would make me retract

- Arms standing on different vertical ladders or dtypes (the scorer refuses on
  the ladder content hash, not the label).
- Either arm blowing up or not reaching day 90.
- A day-0 gate failure on either arm (the twin refuses a non-bit-identical IC).
- The within-arm floor `F` coming back at or above the across-arm difference
  `D`, which would mean the instrument cannot see the variable under test.

## Command

```
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python run_fp64.py kamm_twin_90d.py \
    nemo_dino_kamm_mlf <out>.npz --days 90 --save-3d [--legacy-euler-start] \
    [--perturb-seed 1]
python startmode_ab_score.py B=<b>.npz E=<e>.npz Bp=<bp>.npz Ep=<ep>.npz
python acceptance_gate_90d.py <b>.npz --level 5   # and the same for <e>.npz
```

---

# AMENDMENT 1 — 2026-08-24, after adversarial review, BEFORE any arm was scored

The dual adversarial review (one code, one mechanism) ran while the four arms
were integrating. **No arm had been scored when this amendment was written**;
the scorer had not been run once. The mechanism review refuted the physical
statement this pre-registration rests on and showed the registered decision
rule contains a dead branch and two non-disjoint verdicts. Both are corrected
here rather than after the fact.

## R1 — RETRACTED: "an Euler-started leap-frog trajectory IS the two-point running mean of the true one"

That identity is exact **only when the reference's before level equals its now
level**. A developed NEMO restart's does not. Reproduced independently by the
reviewer:

| regime | max‖Euler − running-mean(leapfrog)‖ |
|---|---|
| reference seeded `before = now`, γ=0 | 2e-16 … 6e-15 (exact) |
| same, **γ = 0.1** (the card's Asselin) | 0.8 % … 6.4 % of signal |
| reference with a **real** before level | residual is the same order as the whole error |

The correct statement is a decomposition, verified to 2e-15 at γ=0:

```
Euler_arm − NEMO = [ RM(NEMO) − NEMO ]  +  RM( Λ[δ, 0] ),    δ = tn − tb
                     half-step delay        injected perturbation
```

- **Term 1, the delay**, does not grow. With γ=0.1 the lag is not 0.500 but
  rises with period toward ≈ 0.5/(1−γ) = 0.556.
- **Term 2, the injection**, is a permanent state perturbation of half the
  before-level gap. The running-mean identity says nothing about it, **and it
  is the term that grows.**
- **Term 3**, the skipped step-1 after-level reconciliation.

Consequence for the campaign: **shifting an Euler-start result half a step does
NOT undo it.** The old wording overstated the exactness and understated the
defect. Corrected at all five sites that carried it.

## R2 — RETRACTED: Fact 3's "3.4e9x the ensemble kick" as the argument for the bar

That ratio compares a smooth, dynamically consistent field against spatially
white noise **in a state norm**. White noise projects onto dissipated
grid-scale modes; `tn − tb` projects onto growing ones. Restated in the gate's
own units, which is what the bar is in:

```
ACC(tn-level u) = 61.946220 Sv     ACC(tb-level u) = 61.950334 Sv
one leap-frog step of ACC = 4.114e-3 Sv
injected physical-mode part ~ delta/2 = 2.06e-3 Sv
```

The effective ratio to the 1e-14 ensemble's day-90 ACC spread is **~180x**, not
3.4e9x. The state-norm numbers in Fact 3 stand as measurements; their use as
the bar's justification is withdrawn.

## R3 — RETRACTED: "one variable: the start"

True of the **flag**, false of the **mechanism**: the Euler branch bundles the
three terms above. Any attribution of the measured `D` to "the half-step lag"
alone is unjustified and will not be made.

## R4 — THE BAR, REPLACED

The registered rule was defective in three ways, all confirmed by direct
measurement of the recorded arms:

- **Rule 2 (`D <= 3F`) is a dead branch.** The 1e-14 ensemble is *still growing*
  at day 90 — ACC max-pairwise 8.75e-8 (d30) -> 3.26e-7 (d60) -> 1.15e-5 (d90) —
  so `F` is the floor of an unfinished perturbation, three orders below the
  comparison. `D ~ 3000 F` by construction.
- **Rules 1 and 4 were not disjoint**, so a saturated, uninformative result
  would have been read out as CONFIRMS.
- **The retraction trigger "F at or above D" was unreachable**, i.e. vacuous.

Replaced by two nulls, in order:

**PRIMARY — the measured saturated config-to-config envelope** (six pairwise
differences among the four recorded 90-day arms in `results/dino_1455_ab90/`,
day 90). These are real one-variable physics changes on this card and horizon:

| metric | NULL band (min … max) | median |
|---|---|---|
| ACC [Sv] | 1.01e-2 … 7.04e-2 | 3.52e-2 |
| upper contrast | 6.55e-7 … 1.04e-5 | 5.20e-6 |
| deep contrast | 1.15e-7 … 1.91e-6 | 9.55e-7 |
| S-band sigma max | 1.17e-6 … 1.67e-5 | 8.35e-6 |
| S-band sigma mean | 1.45e-8 … 6.69e-7 | 3.40e-7 |

Caveat carried, and it is the conservative direction: those four arms sit at
four different source SHAs, so the band is an UPPER envelope.

**SECONDARY — a matched-amplitude within-arm null, measured at THIS commit.**
Three extra bridged members with `--perturb-eps 1e-5` (comparable to the
3.38e-5 relative T gap the start mode applies) give three pairwise differences
from a perturbation that has *saturated*, at one SHA, with no config confound.
This required a one-line `--perturb-eps` (default 1e-14 unchanged, so every
recorded ensemble stays byte-comparable).

## R5 — DECISION RULE, REPLACED (disjoint, and no branch that cannot fire)

Let `D_k = |B_k − E_k|`, `N_k` = the matched-amplitude null band (secondary,
falling back to the primary envelope for any metric it does not cover).

1. **INDISTINGUISHABLE FROM ANY CONFIG CHANGE** — `D_k` inside `N_k` for every
   metric. Read as: *"the start mode's day-90 endpoint effect is bounded above
   by the saturated envelope and is not separable from trajectory divergence at
   n=1."* This is a BOUND, never "no effect".
2. **MATERIAL** — any `D_k` above `N_k`'s upper end, or above `5 x floor_k`.
   Report direction and size loudly; escalate the verdict-year re-run with cost.
3. **INSTRUMENT FAULT** — `D_ACC` below ~1e-3 Sv, which would contradict the
   measured 2.06e-3 Sv day-0 injection. Not a result; investigate the tool.

The gate verdict is still reported for both arms, with one change forced by the
same review: a stamped `twin_start_mode="euler"` is now an UNCERTIFIED reason,
so **arm E scores in full and returns UNCERTIFIED (exit 3) instead of a
PASS/FAIL tally**. The registered comparison therefore reads arm E's neutral
`(within 5x)` / `(over 5x)` markers against arm B's `PASS`/`FAIL` — the same
information without the verdict token.

**The prediction is UNCHANGED** and is now sharper: endpoint metrics move
little in the sense of rule 1. The mechanism review's own forecast, registered
here as an independent prediction to score against: `D_ACC` = 1e-2 … 7e-2 Sv.

## R6 — Provenance of the arms

B / E / Bp / Ep first ran at `13edbe446` (pre-review-fix). The review fixes are
stamping, guards, CLI and prose only — no numerics — so all four are **re-run at
the post-fix HEAD**, and the old and new arm B are checked for bit-identity as a
free control on that claim. Any failure of that control is itself reportable.

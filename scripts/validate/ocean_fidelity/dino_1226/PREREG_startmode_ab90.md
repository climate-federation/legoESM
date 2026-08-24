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

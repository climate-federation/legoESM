# Pre-registration: is legoESM's 100x larger perturbation spread the leapfrog computational mode?

Committed **before any member of the two-time-level arm was launched**. Nothing
below is adjustable after the first number exists. Issue #1455.

Probe: `kick_asymmetry.py`. Tool extensions under review before any number is
cited: `perturb_nemo_tn_90d.py --both-levels`,
`kamm_twin_90d.py --perturb-both-levels`.

---

## 1. The observation this exists to explain

`dino_verdict360_result.md`, "Open item recorded, not investigated":

> legoESM amplifies an identical 1e-14 nudge **78-435x** more than NEMO by day
> 90, while the two growth *rates* match (legoESM e-folding 5.4-12.2 d, NEMO
> 3.4-9.6 d) - so the offset is born in the first 30 days, not in the growth
> rate.

Two further recorded facts point the same way:

* NEMO's ACC spread **falls** from day 30 to day 60 (`verdict360.py`
  `SATURATION_QUARTERS`, measured ratio 0.52) and only then grows. A spread
  that decays and then grows is two superposed modes, not one.
* Both models integrate with a leapfrog + time-filter chain, and the recorded
  kick perturbs the **now** level only, leaving the **before** level
  untouched. A perturbation applied to one leapfrog level and not the other is
  the textbook maximal excitation of the **computational mode** - the
  odd-even-in-time oscillation leapfrog carries in addition to the physical
  solution.

## 2. Hypothesis, and the one thing that discriminates it

**H1 (computational mode).** The 78-435x asymmetry is not physical error
growth. It is how much of the one-level kick each model's time filter converts
into a damped computational-mode transient in the first ~30 days. NEMO's
Asselin/RA filter chain removes more of it than legoESM's does, so NEMO's
spread starts smaller (and briefly decays) while the two physical growth rates
- which do match - carry the rest.

**H0 (physical / other).** The asymmetry is legoESM's genuine sensitivity to
any small perturbation, or a filter property unrelated to the time-level
mismatch. Then removing the time-level mismatch changes nothing.

**The discriminator.** Apply the identical relative kick to **both** leapfrog
time levels (`tn` and `tb` / `T` and `T_before`), same seed, same magnitude, on
**both** models, and re-measure the day-90 spread. A both-level kick carries no
time-level mismatch of its own, so it projects onto the computational mode only
through the mismatch the source restart already had - which is identical for
both models, since both start from the same NEMO restart.

This is a **one-variable** change against the recorded arm: the perturbation
convention. Same restart, same card, same namelist, same binary, same seeds,
same horizon, same reductions, same scorer.

## 3. What is run

| arm | legoESM | NEMO | status |
|---|---|---|---|
| **one-level** (recorded) | `/tmp/dino_verdict360/m{0..3}`, days 0-90 of the existing 360-day members | `RUN_VERDICT360_M{0..3}`, kt 5760-8640 | already on disk, **not re-run** |
| **two-level** (new) | `/tmp/dino_kick2/m{0..3}`, 90 days | `RUN_KICK2_M{0..3}`, 90 days | 4 + 4 runs |

Four members per side per arm: member 0 unperturbed, members 1-3 at seeds
1/2/3. Member 0 of the two-level arm is byte-for-byte the same run
specification as member 0 of the one-level arm (an unperturbed run does not
know which kick convention it was launched under) and is re-run **as the
control on the tool changes** - see control C4.

The one-level arm's day-10..90 numbers are extracted from the existing 360-day
artifacts. The first 90 days of a 360-day integration are the same trajectory
as a 90-day integration; C4 tests exactly that.

Metrics: the ten `verdict360.KEYS` reductions, imported, none re-derived.
Spread statistic: `floor90_ensemble.spread` - sample std as primary (n=4,
~41% relative standard error), max-pairwise range printed beside it and never
compared across.

## 4. The bars, derived before the numbers

For each metric `k`, at day 90:

```
R1(k) = spread_lego(k) / spread_nemo(k)      one-level arm
R2(k) = spread_lego(k) / spread_nemo(k)      two-level arm
F(k)  = R1(k) / R2(k)                        the collapse factor
```

**How much collapse is resolvable.** At n=4 the log of a spread estimate has
standard deviation `1/sqrt(2(n-1)) = 0.408`. `F` is built from four
independent spread estimates, so `log F` carries `sqrt(4) x 0.408 = 0.816`.
A two-sided 95% band is `1.96 x 0.8165 = 1.6006` in log, i.e. a factor of
**e^1.6006 = 4.955**. An `F` below 4.955 is not distinguishable from no change
at all.

*Amendment, before any member of the two-level arm ran*: this paragraph first
rounded the bar to "5.0". The bar is now stated, and computed in the probe, as
the exact `exp(1.96 * sqrt(4) / sqrt(2*(n-1)))` = **4.955** at n=4. Nothing
else changed; the rounding was in the prose, never in a number.

Decided on the **median over the ten metrics** (the median, not the mean, so
one saturated or quantization-limited metric cannot carry the verdict):

* **CONFIRMS H1** if `median F >= 4.955` **and** `median R2 <= 10`.
  The asymmetry collapses, and what is left is within the ~one-decade band
  where an RSS two-sided floor is a genuinely two-sided floor
  (`verdict360.ONE_SIDED_DECADES`).
* **REFUTES H1** if `median F < 4.955`.
  The asymmetry survives removal of the time-level mismatch, so it is not the
  computational mode.
* **PARTIAL** if `median F >= 4.955` but `median R2 > 10`.
  The computational mode contributes a resolvable share and does not own the
  asymmetry. Report the share; do not call it the cause.

Per-metric `F` is tabulated next to the median in every case, and any metric
whose two-level spread is not at least 10x its own float32 storage quantum
(`verdict360.QUANTUM_MARGIN`) is flagged `q` and excluded from the median -
a spread at the storage quantum measures the npz dtype.

## 5. The second, independent signature

If H1 holds, the one-level arm's NEMO decay is the filter removing the
computational mode, and it must **not** appear when there is no computational
mode to remove.

* Pre-registered: for each metric, `s(60)/s(30)` on the NEMO side.
* **Supports H1** if metrics with `s(60)/s(30) < 1` in the one-level arm have
  `s(60)/s(30) >= 1` in the two-level arm.
* **Against H1** if the decay persists under the two-level kick.

This is scored and reported whatever the section-4 verdict says, and it is
independent of it: section 4 is a magnitude, this is a shape.

## 6. Why the answer matters either way - the feedback-tier connection

The basin deficit is currently classified as *sub-floor differences RECTIFIED
by feedback*: individual term-by-term differences that are each below the
measurement floor, made visible by a feedback that amplifies them. A model
that amplifies an infinitesimal perturbation 100x more than its oracle is
exactly that kind of feedback difference, and it is a candidate owner of the
deficit.

* **CONFIRM** removes that candidate. The asymmetry was an artifact of how the
  ensemble was kicked, not a property of the model, and the basin deficit's
  owner is still unidentified. The recorded verdict-360 floors stay valid (the
  arithmetic never assumed symmetry) but limitation 1 of
  `dino_verdict360_result.md` - "the floor is effectively one model's
  dispersion" - becomes an artifact of the kick convention and should be
  re-measured with the two-level convention before being quoted again.
* **REFUTE** promotes it. legoESM genuinely amplifies small differences ~100x
  more than NEMO from the same ocean state, which is a feedback-tier datum in
  its own right and the strongest single candidate for a deficit built out of
  individually sub-floor term differences. It also means every two-sided floor
  this campaign has published is, and will remain, one model's dispersion.

Neither outcome edits any `FLOORS` constant, any recipe, or any card.

## 7. Controls, all of which must pass before any number is read

* **C1 completion.** Every member's own success line: legoESM
  `DONE nsteps=2880 STABLE=True`, NEMO `STOP 0` in the run log with
  `time.step == 8640` and no `E R R O R` in `ocean.output`. An exit code is
  not evidence. A member that did not finish is a finding, never a member to
  drop.
* **C2 one SHA per ensemble.** Every legoESM member of an arm records the same
  `PROVENANCE: HEAD=`.
* **C3 the kick landed.** Growth table, max|dT| of each perturbed member
  against its own side's control by day, both sides, both arms. Plus: no
  metric may be identical across all four members of any ensemble.
* **C4 the tool change is inert on the unperturbed path.** The two-level arm's
  member 0 is compared against the one-level arm's member 0 at day 90, both
  sides. They are the same run specification, so they should agree to storage
  precision. FATAL if the control-to-control difference reaches the size of
  the two-level arm's own ensemble spread, because then the two arms differ in
  something other than the kick and no ratio between them means anything.
* **C5 perturbation receipt** (the focus of the adversarial review). NEMO:
  every variable and every attribute of the restart compared against the
  source; only the intended targets moved; the per-cell **relative** change on
  `tn` and `tb` is the same draw. legoESM: every field of the state pytree
  compared before and after; exactly `T` (and `T_before` under the flag) moved
  and neither is bit-identical to its pre-kick value.
* **C6 same wet domain.** The candidate land mask is checked against NEMO's
  reference mask on every load (`verdict360.lego_state`).
* **C7 NaN is fatal** everywhere; no `nanmax`/`nanmean` anywhere in the
  scoring path.

## 8. What this probe does not do

It prints tables and the mechanical section-4 comparison. It does not edit a
floor, does not re-open the verdict-360 verdict, and does not extend to 360
days - 90 days is where the asymmetry was recorded and 90 days is where it is
tested.

---

# AMENDMENT, before any member was scored — three reviewer findings

Both adversarial reviews landed while the two-level arm was still integrating
and before `kick_asymmetry.py` had scored a single member. Everything below is
recorded here, in the pre-registration, rather than in the result — because two
of the three change what this experiment can conclude.

## A1. RETRACTION: H1's premise is false. The two time filters are identical.

Section 2 says NEMO's filter chain "removes more" of the computational mode
than legoESM's. **That is refuted, offline, from the two models' own
configurations:**

| | Asselin coefficient | source |
|---|---|---|
| NEMO | `rn_atfp = 0.10000000000000001` | the run's own `ocean.output`, inherited from `cfgs/SHARED/namelist_ref` |
| legoESM | `asselin_gamma = 0.1` | the `nemo_dino_kamm_mlf` card itself |

Same coefficient, same plain Robert–Asselin form, and neither model takes a
forward-Euler first step (`ln_1st_euler = .false.`, and legoESM's leapfrog
branch is taken from step 1). There is no asymmetry in the filters for H1 to
live in.

**The closed form, which this repo already records** (the eigen-decomposition
of the two-state `[T_before, T_now]` Asselin/leapfrog recursion, eigenvalues
`1` and `2γ−1`, at `dino.py:255-268`):

```
one-level kick (0, ε)   keeps (1−2γ)/(2(1−γ)) = 4/9  of itself on the physical mode
two-level kick (ε, ε)   keeps 1
arm effect              = 2(1−γ)/(1−2γ) = 2.25,  PER MODEL
collapse factor F       = arm(NEMO)/arm(legoESM) = 2.25/2.25 = 1.00
```

So **F is predicted to be exactly 1.00 against a bar of 4.955**, and the reason
is not that H1 is subtle — it is that the mechanism is symmetric and cancels.

A second, independent offline fact points the same way: the computational mode
decays by `|2γ−1| = 0.8` per step, i.e. `7.9e-4` per day and `1e-31` by day 10.
Nothing that decays that fast can be shaping a day-30-to-day-60 signature, so
**section 5's decay signature cannot bear on H1 in either direction.** It is
still measured and reported, now as a description of NEMO's behaviour rather
than as a test of anything.

**What the experiment is now for.** It is a calibration of the ensemble
instrument against a closed form: the per-model arm ratio should measure 2.25
and F should measure 1.00. A null on F is only informative if the arm ratio
lands on its prediction — otherwise a null means the instrument saw nothing at
all. The probe therefore prints the per-model arm ratio table next to the
collapse-factor table, and both are read together.

## A2. The two arms were about to compare two different models.

The one-level arm was to reuse the recorded 360-day members. Those ran **45
commits back**, across two commits that change the physics of the very card
both arms run (NEMO's EEN transport-metric weighting, and the implicit vertical
solve's face control volume). The recorded day-90 ACC ensemble spread is
2.0e-05 Sv; either change moves day-90 ACC by orders of magnitude more.

**Corrected:** the one-level arm's legoESM side is re-run at the current
revision as four 90-day members, and a new control (C2c) requires both arms'
legoESM members to record the same source revision. The NEMO side is still
reused: NEMO's binary is the same certified executable in both arms, this
repo's commits cannot touch it, and the two namelists differ in `nn_itend`
alone — when the run stops, not what it computes.

## A3. "Below the resolvability bar" is NO INFORMATION, not a refutation.

Section 4 mapped `median F < 4.955` to **REFUTES H1**. 4.955 is the *width of
the null band* at n=4: below it, `F = 1` and `F = 4.9` are the same
measurement. Turning that into a positive claim is wrong in the one direction
that matters, because REFUTE is the branch section 6 attaches a consequence to.

**Corrected rule** (this replaces section 4's three-way rule; the bars
themselves are unchanged):

* **CONFIRMS H1** — `median F >= 4.955` and `median R2 <= 10`.
* **PARTIAL** — `median F >= 4.955` and `median R2 > 10`.
* **REFUTES H1** — `median F < 4.955` **and** the 95% upper limit on F
  (`F x 4.955`, the band being multiplicative) falls short of the collapse H1
  needs, which is `median R1 / 10`. Only then has the run excluded H1 rather
  than merely failed to see it.
* **NOT RESOLVED** — anything else. The run said nothing.

## A4. Section 6's REFUTE consequence is withdrawn pending one free measurement.

Section 6 claims a REFUTE promotes legoESM's amplification to a feedback-tier
candidate for the southern-basin deficit. That step is not sound as written:
ensemble dispersion is the transient gain of the tangent-linear operator about
a trajectory, while the deficit is a difference between two nonlinear
attractors. The two connect only if the amplifying mode has a non-trivial
projection onto the deficit pattern.

**Withdrawn until measured.** The discriminating measurement is free — both
fields are already on disk: project the day-90 ensemble-spread pattern onto the
southern-basin transport-deficit pattern. Until that projection is non-trivial,
the result document will state the refutation and stop, and will NOT claim a
feedback-tier consequence. The CONFIRM branch of section 6 is unaffected.

## A5. Recorded limitations, not fixed here

* **The two models are not kicked on the same cells.** A relative kick cannot
  move a cell that is exactly zero. NEMO's `tn` has 30,394 exact zeros, the
  bridged legoESM T has 12,420, and 17,974 cells (4.8% of the grid) are kicked
  in legoESM and not in NEMO — below-bottom cells where legoESM carries a
  bridged value and NEMO stores zero. The asymmetry is identical in both arms,
  so the collapse factor is protected; the absolute ratios are not. "An
  identical 1e-14 nudge on both models" is true of the magnitude and the
  distribution, not of the cell set or the per-cell values (the two draws are
  filled in different axis orders).
* **legoESM's snapshots are float32.** Between day 10 and day 60 the recorded
  one-level members' temperature differs from their control in 1 to 5 cells of
  372,528, i.e. at 1–4 units in the last place. The legoESM side of "the offset
  is born in the first 30 days" is therefore *censored, not measured*. Day 90
  is fully resolved on both sides. Fixing this needs one fp64 snapshot per
  member per early day — named, not built here.
* **The kick is on temperature; the scored metrics are transports.** Velocity
  and sea surface height acquire a one-time-level perturbation in *both* arms,
  so the two-level kick removes the time-level mismatch from the tracer
  equation only. A refutation therefore refutes tracer-level computational-mode
  content, which is what section 2 proposed, and not every route by which a
  time-level mismatch could matter.
* **The residual mismatch of the relative-factor choice is ~1e-4 of the kick.**
  The zero-computational-mode direction is equal *absolute* increments at both
  levels; the same *relative* factor leaves `f x (T_now − T_before)`, one
  leapfrog step of tendency, which is ~1e-4 of the kick on this configuration
  against the one-level kick's 100%. The choice is correct and changes no
  interpretation.
* **The number motivating section 5 is 1.1σ.** NEMO's recorded day-30→60 spread
  ratio of 0.52 carries a log standard error of 0.577 at n=4, i.e. 1.13σ from
  1.0. It is a hint, not a fact, and the prose it was quoted from also asserted
  the mechanism it was being used as evidence for.

## A6. The measurements this experiment does NOT make, named so they are not forgotten

Ranked by what they would settle, cheapest first. None is built here.

1. **One step, three kick sizes** (1e-14 / 1e-12 / 1e-10), legoESM, fp64: is
   the response linear in the kick? A discrete switch firing — convective
   adjustment, an enhanced-diffusion trigger, a mixed-layer level index — would
   rectify the kick in one step and produce exactly the recorded signature
   (offset immediate, growth rates unchanged). Seconds of compute.
2. **One legoESM member, 60 days, fp64 daily temperature dumps**, overlaid on
   NEMO's: NEMO's raw perturbation decays 224x between day 10 and day 60 and
   then grows; whether legoESM shares that transient is the single curve that
   settles the open item.
3. **Sweep γ on one model** (0.05 / 0.1 / 0.2, one-level kick) and measure how
   the ratio responds: insensitivity kills the filter mechanism with one
   variable and one model.
4. **The spread-pattern projection** of A4.

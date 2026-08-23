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
A two-sided 95% band is `1.96 x 0.816 = 1.60` in log, i.e. a factor of
**e^1.60 = 5.0**. An `F` below 5.0 is not distinguishable from no change at
all.

Decided on the **median over the ten metrics** (the median, not the mean, so
one saturated or quantization-limited metric cannot carry the verdict):

* **CONFIRMS H1** if `median F >= 5.0` **and** `median R2 <= 10`.
  The asymmetry collapses, and what is left is within the ~one-decade band
  where an RSS two-sided floor is a genuinely two-sided floor
  (`verdict360.ONE_SIDED_DECADES`).
* **REFUTES H1** if `median F < 5.0`.
  The asymmetry survives removal of the time-level mismatch, so it is not the
  computational mode.
* **PARTIAL** if `median F >= 5.0` but `median R2 > 10`.
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

# The verdict run: does the shipped DINO card match NEMO after one year?

**Result, in one sentence.** From a shared NEMO day-180 restart, one year of
legoESM drift away from NEMO is **no larger than the two models' own
run-to-run spread in the circumpolar channel**, and **larger than it in the
southern basin**. This is a statement about one year from a common ocean
state; it is not a statement that the two models share a climate, and the deep
ocean has not adjusted on this timescale.

Written for a reader who has not followed the campaign. Pre-registration:
`scripts/validate/ocean_fidelity/dino_1226/PREREG_verdict360.md`, committed
before any run started. Probe: `verdict360.py`. Result commit: `a1387f1f7`.

---

## What was run, and why it takes eight runs

Both models were integrated **360 days** from the **same** NEMO restart
(`DINO_00005760_restart.nc`, day 180 of the DINO spin-up), at 2700 s.

* **legoESM**: the shipped `nemo_dino_kamm_mlf` card, no option flags.
* **NEMO**: the certified binary, the recorded 90-day twin's namelist with
  only the run length changed.

Comparing one legoESM run to one NEMO run tells you a number but not whether
it means anything, because *any* two runs of *any* chaotic model separate over
a year. So each side also ran **three more members**, identical except for a
1e-14 relative temperature nudge at the start (seeds 1/2/3). The spread among
those members is how much a run of that model moves for no physical reason at
all — the **floor** a real difference has to clear.

| | what it answers |
|---|---|
| legoESM member 0 vs NEMO member 0 | **the gap** |
| the other three, each side | that model's own irreducible spread |

**The rule, fixed before any number existed:** a metric is
**INDISTINGUISHABLE** when `|gap| ≤ 2 × floor`, where
`floor = sqrt(legoESM_spread² + NEMO_spread²)` at the same day.

---

## The verdict table — day 360

Gap is legoESM minus NEMO, same day, same reduction on both sides.

| metric | gap [Sv] | floor [Sv] | gap/floor | verdict |
|---|---:|---:|---:|:--|
| channel band (e3t_0, mean) | **+0.0015** | 0.0101 | **0.15** | **INDISTINGUISHABLE** |
| channel band (group reduction) | **+0.0024** | 0.0096 | **0.25** | **INDISTINGUISHABLE** |
| channel band (e3t_1d, median) | +0.0396 | 0.0126 | 3.15 | no — upper bound only¹ |
| north of band | −0.0385 | 0.0041 | 9.32 | no — upper bound only¹ |
| **south of band** | **−0.9519** | 0.0617 | **15.42** | **no** |
| full section (mean) | −0.9880 | 0.0558 | 17.71 | no² |
| ACC (the recorded gate metric) | −0.9704 | 0.0454 | 21.37 | no² |

¹ the ensemble spread is still growing fast enough at this metric that another
year could plausibly close the gap, so the `no` is an upper bound, not a
demonstrated failure. For the southern basin this is **not** the case: closing
it needs the floor to grow 7.7×, and its last quarter grew 1.02×.

² the full-section number is a *median over longitudes* and is therefore not
additive over the three latitude bands. It is reported because it is the
campaign's recorded headline, but it is not the verdict — see below.

### Over a full seasonal cycle

The four horizons above (days 90/180/270/360) are days 270/0/90/180 **of the
year**, i.e. four different seasons, so each endpoint carries a seasonal
phase. Averaging all 19 scored days covers one complete cycle. This statistic
is **post-hoc**, not pre-registered, and it is the cleanest single number:

| metric | full-year gap [Sv] | gap/floor |
|---|---:|---:|
| channel band (e3t_1d, median) | **+0.00058** | **0.12** |
| channel band (e3t_0, mean) | −0.0183 | 5.19 |
| north of band | −0.0258 | 21.4 |
| **south of band** | **−0.3452** | **22.9** |
| ACC (gate metric) | −0.3858 | 28.8 |

**The channel-band gap over a full year is 0.0006 Sv on a 34 Sv transport —
17 parts per million.**

---

## Why three transports instead of one number

The campaign's headline metric integrates every latitude into one number. Doing
that here would report a model that is "wrong everywhere". Split by latitude,
at day 360:

```
south of band   −0.952 Sv
channel band    +0.002 Sv
north of band   −0.038 Sv
                ------
full section    −0.988 Sv   (mean-reduced; the three sum to it exactly)
```

**The full-section gap *is* the southern-basin gap.** The model is right in the
channel and wrong in one basin. One number cannot say that, and which band owns
the error is the actionable fact.

---

## What happens to the southern basin over the year

It does not drift smoothly:

| run day | day of year | south-of-band gap [Sv] |
|---:|---:|---:|
| 10 | 190 | +0.03 |
| 90 | 270 | −0.43 |
| 180 | 0 | −0.23 |
| 270 | 90 | **−0.06** (indistinguishable, 0.92 floors) |
| 300 | 120 | −0.23 |
| 330 | 150 | −0.60 |
| **360** | **180** | **−0.95** |

The deficit grows over the first quarter, **recovers almost completely by day
270**, then grows steeply and monotonically through the final quarter. Reading
only the two endpoints would say "it doubles"; the full trajectory says it
compounds *with a mid-year recovery*, which is a different and more specific
fact.

---

## Decision, per the consequence registered before the numbers existed

The registered bins for the southern basin were: `≤2` closed at one year;
`2–10` on the board, no new work; `>10` the next single work item.

Measured: **15.4** (endpoint), **20.9** (registered 90-day window), **22.9**
(full year). All three agree.

> **The southern basin is the campaign's next single work item. First action: a
> term-by-term southern-basin transport budget.**

---

## Predictions, scored

| | prediction | outcome |
|---|---|---|
| **P1** | the noise floors grow into the 0.05–0.09 Sv class by year-end | **CONFIRMED** — legoESM day-360 ACC spread 0.0447 (std) / 0.102 (range) against the transferred 0.050 / 0.091. This is the **first validation** of a constant the campaign has been transferring from a 10-year ensemble to 90-day runs. Growth from day 90: ×2200. |
| **P2** | the channel is indistinguishable at 360 days | **CONFIRMED** on two of three reductions at the endpoint and on the pre-registered reduction over the full year (0.12 floors). |
| **P3** | *deliberately no prediction* | **ANSWERED**: the deficit compounds, non-monotonically, and lands in the harshest registered bin. |
| **P4** | *no prediction* | density metrics all `no`; absolute gaps 1e-4 to 6e-4 kg/m³ against floors ≤1e-6. |
| **P5** | day 90 reproduces the recorded −0.3823 Sv gap within 1e-3 | **CONFIRMED to 2.6e-07 Sv.** |

---

## Controls, all green before any number above was read

* **NEMO member 0's day-90 state is bit-identical to the recorded 90-day twin**
  (max|ΔT| = max|ΔS| = max|Δu| = 0.000e+00). This year-long run *is* that twin,
  continued — the NEMO side is certified directly rather than inferred.
* All four legoESM members at one source revision, one vertical ladder, one
  seasonal clock, one precision. All four NEMO members on the certified binary.
* Harness self-checks: NEMO year-10 ACC 121.07 Sv against the recorded 121.07;
  band volume within 5.0e-08 relative.
* Every metric separates all four members at day 360 on both sides.
* **Calibration of the rule itself.** On comparisons whose answer is known:
  same-model pairs give a median ratio of **1.211** against an analytic
  prediction of **1.414**, with 69.7% inside the band against a predicted
  66.7%; and every transport is correctly rejected when legoESM is compared
  against its own state at an earlier date.
* The 1e-14 nudge reached 0.33 K (legoESM) and 0.15 K (NEMO) by day 360, so the
  floors measure propagated perturbations, not rounding.

---

## Limitations a reader should carry away

1. **The floor is effectively one model's dispersion.** NEMO is 9× to 16600×
   tighter than legoESM under the identical nudge, so 77–99% of the combined
   floor is legoESM's own spread. The pre-registration justified the
   combination as "√2 × one side when the two wobble equally"; **that
   justification is empirically false here**, and the correction is disclosure,
   not a different formula.
2. **One year is not a climate.** Both models still carry the same deep ocean
   from the shared restart, and the overturning adjusts on decades.
3. **Seasonal aliasing.** The registered horizons and the registered 90-day
   window each carry a seasonal phase; the full-cycle mean above is the
   remedy and is post-hoc.
4. **n = 4.** Every floor carries ~41% relative standard error and is a
   factor-of-two estimate.
5. **"ACC [Sv]" is a reference-geometry proxy** (e3t_1d weighting, which
   differs from the model's own thickness by up to 12.9% on wet cells). It is
   the identical functional on both sides, so the *gap* is valid; the *number*
   must not be quoted against a published or observed ACC.

## Open item recorded, not investigated

> **RETRACTED and answered, 2026-08-23** — see
> `dino_kick_asymmetry_result.md`. Two corrections to what follows. (1) The
> **78–435×** range is too small: the same artifacts, scored by the same
> reductions, give a median **3051×** across the eleven metrics. (2) The
> mechanism proposed below is **refuted** — the two models run the *identical*
> time filter (coefficient 0.1 on both sides), so nudging both leapfrog levels
> moves the ratio by 0.96× against a 4.95× bar. The amplification itself is
> real and unexplained. Note also that the `q` storage-precision flag in this
> document's own tables never fired: it compared an already-single-precision
> state against itself and always returned zero (fixed at the source).

legoESM amplifies an identical 1e-14 nudge 78–435× more than NEMO by day 90,
while the two growth *rates* match — so the offset is born in the first 30
days, not in the growth rate. Plausibly an asymmetry in how the two time
filters damp the computational mode the temperature-only nudge excites.
Cheap discriminator: nudge both leapfrog time levels on both sides and
re-measure day 90.

## Cost

legoESM 4 × ~11 min on two GPUs; NEMO 4 × ~8.6 min serial on 16 cores.
Running the NEMO members four-at-a-time was measured **24× slower in
aggregate** and was abandoned.

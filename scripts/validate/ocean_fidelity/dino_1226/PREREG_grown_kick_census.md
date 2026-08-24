# Pre-registration — the grown-perturbation threshold census and firing-rate comparison

Written and committed **before** `grown_kick_census.py` produced a single
number. Provenance of every input is stamped in the probe's own output.

## The question

`dino_switch_rectifier_result.md` established that convective adjustment's hard
edge rectifies a perturbation by ~300x per step, and that at the ensembles' own
1e-14 kick amplitude **zero** interfaces sit within tipping reach of the
trigger, so the edge cannot fire at that size. The ensembles' perturbations
grow. The unmeasured link in the chain is therefore:

> on the **grown** perturbation, how many interfaces sit within tipping reach of
> the convective threshold, in **each** model, and how often does the trigger
> actually resolve differently between a perturbed member and its control?

If legoESM's near-threshold occupancy or its pair-flip rate materially exceeds
NEMO's, the rectification asymmetry is measured and the chain link closes. If
they are comparable, the chain needs a different explanation for the
amplification asymmetry, and the 2dt-flicker link weakens.

## What exists on disk (enumerated before registering, stamps in the probe)

* legoESM, `/tmp/dino_kick2` (kick on both time levels — the lane that matches
  NEMO's kick) and `/tmp/dino_kick1` (now-level only, legoESM-only sensitivity).
  Members `m0_control`, `m1_seed1`, `m2_seed2`, `m3_seed3`; both lanes launched
  at HEAD `9dc530c86`. Full 3-D T/S snapshots at days **0,10,20,...,90**, stored
  **float32**.
* NEMO, `/tmp/dino_kick2_nemo/RUN_KICK2_M{0,1,2,3}`, per-rank restart tiles at
  `kt = 5760 + 32*day`, i.e. the **same** day 0,10,...,90 cadence, carrying
  `tn/sn` and `tb/sb` in **float64**.

**Day 5 does not exist on either side.** Neither model wrote a 3-D state at day
5; the finest common cadence is 10 days. The mission's "days 5 and 10" therefore
becomes "day 10 and every later available horizon", and day 5 is reported
ABSENT, not estimated.

## The statistics

All three are computed per horizon, per model, per member pair (m1,m2,m3 each
against m0), on the **same** N-squared operator on both sides
(`compute_buoyancy_frequency_nemo_bn2`, NEMO's own `bn2`), with the **same**
static depth ladders, and on the **same** dry-cell mask.

(a) **Perturbation magnitude distribution** — max, 99.9th percentile, median of
non-zero, and the count and fraction of wet cells at which the stored
perturbed-minus-control temperature difference is non-zero.

(b) **Threshold-neighbourhood census** — `reach` = the maximum |dN2| the
perturbation causes over all wet interfaces; `near` = the number of wet
interfaces with `|N2_control - threshold| < reach`. This is the **upper-bound**
convention of the shipped census in `switch_rectifier.py::census` (one global
maximum reach applied to every interface) and it is kept unchanged so the two
numbers are comparable. It over-counts by construction and is labelled so.

(c) **Pair-flip count / rate** — the ACTUAL firing difference: the number of wet
interfaces at which the convective trigger fires in exactly one of
{perturbed member, control}, split into on-flips (fires only in the perturbed
member) and off-flips. The **rate** is that count divided by the number of wet
interfaces. This is the discriminating statistic.

## The trigger, in each model's own code

* legoESM (`enhanced_diffusion.py:195`): `N2 < n2_threshold` on the **single**
  now-level N2; the shipped card sets `n2_threshold = -1e-12` and the switch
  steps vertical diffusivity 1e-5 -> 100 m2/s.
* NEMO (`zdfevd.F90:93`): `MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12`, the
  **minimum over two time levels**. `namelist_cfg` confirms `ln_zdfevd=.true.`,
  `nn_evdm=1`, `rn_evd=100.`

The primary comparison uses the **single-level** rule on both sides, so that the
measured difference is a difference in the two models' STATES and not in the two
rules. NEMO's own two-level rule is reported alongside as a second column, to
size how much the rule itself changes NEMO's firing. NEMO's exact `rn2b` is not
in the restart, so `bn2(tb,sb)` is used as a **proxy** for it and is labelled a
proxy everywhere it appears. legoESM's before-level 3-D state was not saved, so
the two-level rule cannot be evaluated on legoESM at all — stated, not estimated.

## The precision problem, and the control that handles it

legoESM's 3-D snapshots are float32; NEMO's restarts are float64. A float32
temperature near 20 degC has a quantum of about 2.4e-6 K, which maps to a dN2 of
order 5e-10 — five hundred times the trigger's own 1e-12 offset. **Storage alone
can therefore manufacture flips.** The comparison is protected by a
storage-matched control:

* `nemo_fp64` — NEMO's native state. The resolution-unlimited reference.
* `nemo_fp32` — the SAME NEMO state cast to float32 and back, then run through
  the identical pipeline. This is the storage-matched arm, and it is the one
  legoESM is compared against.

The gap between `nemo_fp64` and `nemo_fp32` is the **storage inflation factor**
and it is printed next to every number.

## Registered verdict rule

Primary statistic: **pair-flip rate**, averaged over the three member pairs.
Primary ratio: `R(d) = F_lego(d) / F_nemo_fp32(d)` at horizon `d`
(storage-matched). Secondary: `R64(d) = F_lego(d) / F_nemo_fp64(d)`.

* **CONFIRM** — `R(d) >= 5` at the earliest horizon that clears the
  resolvability bar below, i.e. legoESM's trigger resolves differently between a
  perturbed member and its control at least five times as often as NEMO's does
  on a storage-matched footing.
* **REFUTE** — `R(d) <= 2` there. The rectification asymmetry is then NOT
  measured, the chain needs a different explanation for the amplification
  asymmetry, and the 2dt-flicker link weakens. This is to be said loudly.
* **INDETERMINATE** — `2 < R(d) < 5`.
* **UNMEASURABLE at horizon d** — legoESM's stored perturbation is non-zero on
  fewer than **0.1%** of wet cells. Below that bar the float32 snapshot has
  barely separated the pair at all and any census or flip count computed from it
  is measuring the file format. No upper bound is substituted for the missing
  measurement.

**Disclosed before scoring:** `dino_kick_asymmetry_result.md` already records
that legoESM's stored members differ from their controls in 2 cells at day 10, 8
at day 30, 26,691 at day 60 and 59,768 at day 90 (denominators unmasked). Days
10 and 30 are therefore EXPECTED to fail the resolvability bar and be returned
UNMEASURABLE; days 60 and 90 are expected to clear it. That expectation is
registered here so that returning UNMEASURABLE at day 10 cannot be read as a
result invented after the fact. No flip count, census count or ratio was known
when this was written.

## Instrument defect found while registering, and fixed here

The shipped census in `switch_rectifier.py::census` derives its wet mask from
`np.isfinite(N2)`, which is true everywhere: its reported denominator, 362,180,
is exactly 199*52*35, the **entire** grid. That census applied **no dry-cell
masking at all**, and the same is true of the 372,528 denominator quoted in the
kick document. About 8% of the T-grid is land. This probe takes its mask from
NEMO's own `mesh_mask.nc` `tmask`, with the interface mask
`tmask[k] & tmask[k+1]` (NEMO's `wmask`), and proves the mask is live with a
planted violation in both directions: a violation planted in a dry cell must not
move any statistic, and the same violation planted in a wet cell must move one.

The affected prior numbers are percentages with a slightly-too-large
denominator. The load-bearing prior conclusion — **zero** interfaces within reach
at 1e-14 — is a count of zero and is unaffected by the denominator.

## Other rules this probe runs under

NaN anywhere in an input field or an intermediate is FATAL, never nan-reduced.
Every artifact is stamped with its path, mtime, launch SHA and the probe's own
git SHA. The verdict text is computed from the values, never typed. Each
statistic carries its own bar. The probe does not edit a card, a recipe or a
floor, and it does not interpret its own numbers beyond applying the rule above.

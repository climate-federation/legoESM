# PHASE 2 — alignment row R6, built as an option, and the pre-registered A/B

Written and committed BEFORE either arm ran.  Nothing above the RESULT heading
is edited afterwards; git history proves the order.

NEMO paths are `cfgs/DINO/MY_SRC` (the build that produced every oracle number
in this campaign).  `src/OCE` differs and its line numbers do not apply.

---

## The ordered alignment rows

### R6-a — NEMO SITE 1, the NOW-level advective-velocity swap

    dynspg_ts.F90:1171-1173, inside dyn_spg (stpmlf.F90:332)

    puu(:,:,jk,Kmm) = ( puu(:,:,jk,Kmm) + un_adv(:,:)*r1_hu(:,:,Kmm)
       &              - puu_b(:,:,Kmm) ) * umask(:,:,jk)

| | |
|---|---|
| thickness | `hu(Kmm)` — the NOW level |
| installed mean | `un_adv/hu(Kmm)`, the SECONDARY transport-weighted substep average (accumulated `:736`, normalised `:999`) |
| removed mean | `puu_b(:,:,Kmm)`, the stored NOW barotropic velocity |
| lifetime | WITHIN-STEP.  Consumed by `tra_adv` (`stpmlf.F90:528`) and the rest of the now-level block, then UNDONE at `stpmlf.F90:787-790` (the `.NOT.ln_bt_fw` branch; `ln_bt_fw=.false.`, `RUN_90D_TWIN/namelist_cfg:353`) before `dyn_atf_qco` (`:613`).  Never reaches the committed state. |

**legoESM under the option: NOT BUILT, deliberately.**  This is an advecting-
velocity convention for the within-step advection, not a state update, and
building it honestly means building its removal too — two changes, and a
different question from the one this option asks.  Which substep average is
installed is the separate `barotropic_reconcile_target` lane, already tested
and REFUTED as a lever on the 90-day gate (`PREREG_gate90_reconcile_ab.md`).

### R6-b — NEMO SITE 2, the AFTER-level committed reconciliation

    stpmlf.F90:754-765, in mlf_baro_corr, called at :578
    — AFTER dyn_zdf (:396) and BEFORE dyn_atf_qco (:613)

    zue(ji,jj) = SUM_k e3u(ji,jj,jk,Kaa) * puu(ji,jj,jk,Kaa) * umask(ji,jj,jk)
    puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - zue(ji,jj)*r1_hu(ji,jj,Kaa)
       &                + uu_b(ji,jj,Kaa) ) * umask(ji,jj,jk)

| | |
|---|---|
| thickness | `e3u(Kaa)` in the sum AND `hu(Kaa)` in the divisor — the AFTER level, both |
| installed mean | `uu_b(:,:,Kaa)`.  DINO sets `ln_dynadv_vec=.TRUE.` (`namelist_cfg:321`), so `dynspg_ts.F90:978-980` accumulates VELOCITIES (`za1*ua_e`) and `:1001` divides by `r1_wgt1s`.  `puu_b(Kaa)` is therefore ALREADY a velocity, and the transport-to-velocity division at `:1164` does NOT run — it sits in the `ELSE` branch (`:1139-1166`) that `:1124`'s `ln_dynadv_vec` test skips.  So `uu_b(Kaa)` is the PRIMARY velocity-weighted boxcar average, exactly. |
| removed mean | the column's OWN after-thickness mean, RECOMPUTED here — not a stored field |
| what it accomplishes | it annihilates whatever column mean `dyn_zdf` deposited.  Measured on the 90-day DINO twin: `+17.916 m3/s2` per southern u-row, 12x the realized spin-up rate, discarded every step |

**legoESM under the option:** `barotropic_common.after_level_column_mean_reconcile`,
called from BOTH outer-step paths (`_leapfrog_step`, `_nemo_mlf_step`) at NEMO's
position — after `_apply_implicit_vertical_mixing`, before the conservation
fixer and the Asselin filter.

| | |
|---|---|
| thickness | `min_cell_to_uface(compute_layer_thickness(naa.eta, ...))` — `naa.eta` is the barotropic solve's after ssh, the same level `e3t_aft` is built from |
| installed mean | `btu_exp`/`btv_exp`, the depth mean the barotropic solve produced, i.e. whichever average `barotropic_reconcile_target` selected.  Under `"velocity_avg"` that IS `uu_b(Kaa)` and this row is NEMO's line verbatim; under the card's `"transport_avg"` it is the secondary average in NEMO's slot. |
| removed mean | the column's own after-thickness mean, recomputed — same as NEMO |

The two config fields are ORTHOGONAL on purpose: one picks WHICH average, the
other picks WHERE and at which time level's thickness it is enforced.

### R6-c — legoESM WITHOUT the option, for contrast

ONE reconciliation, inside the barotropic solve (the shared `_reconcile_targets`
helper, both entry points), at the NOW thickness; the leap-frog combine then
pins the after-level velocity's NOW-thickness column mean to `btu_exp`.  Nothing
runs after the implicit vertical solve.

### R6-d — the two halves of R6 do NOT carry the same weight

**(i) The THICKNESS half (now -> after) is STRUCTURALLY INERT under pure
z-star.**  Every layer of a column rescales by the same `(H+eta)/H`, and the
min-rule face depth inherits that factor, so a thickness-weighted column mean is
IDENTICAL at the two levels.  Measured `1.1e-16 m/s` on a flat-bottom z-star
channel and gated by `tests/ocean/unit/test_barotropic_after_reconcile.py::
test_inert_on_pure_zstar_without_partial_cells`.  It is live only through
partial cells — which the DINO card does have.

**(ii) The SECOND-SITE half is already MOSTLY done, from the other side.**
`zdf_baroclinic_only=True` (on for this card) strips the depth mean BEFORE the
implicit solve and re-adds it unchanged after, which achieves NEMO's discard
without a post-solve corrector.  What survives is the difference between that
strip/re-add weighting (the solve's `dz_u`, floored at `min_water_column_m`) and
the combine's own (`h_u`, floor `1e-10`).  On the valid-stack 90-day twin that
residue is the retained `ZDF bt = +0.296 m3/s2` per southern u-row.

So R6 is a REAL fidelity row and a SMALL one, and it was named as the last
untested structural DIFF before either of those two facts was measured.

---

## The pre-registered A/B

### Arms — one variable

| | arm A (baseline) | arm B (R6 option) |
|---|---|---|
| `barotropic_after_reconcile` | `"off"` (card default) | `"nemo_mlf_baro_corr"` |
| selected by | no env var | `DINO_AFTER_RECONCILE=nemo_mlf_baro_corr` |
| GPU | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=1` |

Held identical and stamped in every log: HEAD (this commit), clean tracked
tree, recipe `nemo_dino_kamm_mlf`, `--bridge-before`, IC = NEMO
`DINO_00005760_restart.nc` (day 180), `LEGOESM_NEMO_E3T=both`, corrected
seasonal clock (`t0 = 15552000 s` from the restart's own `adatrj`), fp64
control dtype via `run_fp64.py`, dt 2700 s, 32 steps/day, 90 days.
`barotropic_reconcile_target` stays at the card's `transport_avg` on BOTH arms —
flipping it too would be two variables.

### The three numbers, and where each comes from

1. **southern-band realized spin-up rate**, band mean over u-rows 1..13,
   90-day, `[m3/s2 per u-row]` — `southern_term_torque_accum.py`, compared
   against NEMO's `+1.430` from `nemo_accum_torque.py`.
2. **day-90 south-group transport deficit** `[Sv]` and the fraction of NEMO's
   `+3.097 Sv` spin-up captured — `southern_circulation_budget.py`.
3. **the five-metric acceptance gate** at 5x the noise floor, including the
   full-section ACC gap — `acceptance_gate_90d.py`, unmodified, with its own
   two fatal self-checks (NEMO y10 ACC 121.07 Sv; band volume 2.694775e16 m3).

Noise floors (#1492 2.1 micro-ensemble, n=3): ACC **0.091 Sv**, upper contrast
1.1e-4, deep contrast 4.5e-5, southern surface sigma max/mean 9.5e-5 kg/m3.

### Valid-stack arm-A values already measured (this branch, HEAD's parent)

    band realized dR/dt      +1.069   (NEMO +1.430, gap +0.361)
    stage row ZDF bt         +0.296   <- what the option removes
    day-90 south-group gap   -0.772 Sv, 75% of NEMO's spin-up captured
    full-section ACC gap     -0.5965 Sv   (channel band +0.2874)

### PREDICTION, quantitative and falsifiable

Turning the option ON removes legoESM's retained `ZDF bt` row from the realized
rate and re-pins the after-level column mean at the after thickness.  The
thickness half is a partial-cell-only effect and is expected to be small.  So:

* **Expected**: the band realized rate FALLS by roughly **0.30** (from `+1.069`
  toward `+0.77`), and the NEMO-minus-lego band gap therefore GROWS from
  `+0.361` to roughly **`+0.66`** — the WRONG direction for the deficit.
* **CONFIRMS R6 owns the deficit**: the band gap closes by more than 50%
  (`+0.361 -> < +0.18`) AND the full-section ACC gap shrinks toward the
  `0.091 Sv` floor.
* **REFUTES, null**: every metric moves by less than its own noise floor.
* **REFUTES, adverse** (what is expected): the band gap GROWS beyond the
  window scatter (`sd 0.225` over the nine matched windows).

A refutation is a real outcome, not a failed run.  R6 is being tested because
it was the last named untested structural DIFF, and R6-d already bounds how
much of the deficit it could plausibly carry.

### Also recorded whatever the verdict

The full five-metric gate table on BOTH arms, each arm's day-0 gate line,
`tau_x` range, resolved ladder and clock, and the stability tail.  **A density
metric degrading past its floor on arm B is a Rule-8 finding to be reported,
not a reason to revert the option** — no gate constant and no card default is
edited by this test.

### Implementation gates already passed

`tests/ocean/unit/test_barotropic_after_reconcile.py`: 23 passed.  Proven
NON-VACUOUS by reverting the model-side insertion — the four tests that gate it
(behaviour on both step paths, dispatch raise on both step paths) then FAIL
while the 19 kernel/config tests still pass.
`tests/test_dispatch_hardening.py`: 24 passed, with the new guard pinned in the
grow-only baseline.

---

# AMENDMENT — written after the first A/B ran, BEFORE the replacement 2x2

Committed separately from everything above; the pre-registration text is
unedited (git history proves the order).  This amendment exists because DUAL
ADVERSARIAL REVIEW found two defects in the EXPERIMENT, not in the kernel, and
both change what the arms mean.  The first A/B's numbers are reported as
SUPERSEDED, not quietly dropped.

## What the reviews found

**(1) NEITHER of the first two arms is NEMO.**  The card resolves
`barotropic_reconcile_target="transport_avg"`, and the pre-registration pinned
it there on both arms to keep one variable.  But `transport_avg` is legoESM's
analogue of NEMO's `un_adv/hu` — the average NEMO installs only TRANSIENTLY at
`dynspg_ts.F90:1172` and then DELETES again at `stpmlf.F90:788`.  What NEMO
COMMITS at site 2 is `uu_b(Kaa)`, its primary velocity-weighted boxcar, which
is legoESM's `velocity_avg`.  So arm B put NEMO's placement around the average
NEMO throws away, and the faithful pair — `velocity_avg` +
`nemo_mlf_baro_corr` — was never run.  The two fields COMPOSE; treating them as
orthogonal was the error.

**(2) THE "AFTER-LEVEL THICKNESS" IS A NO-OP IN NEMO, ALGEBRAICALLY.**  DINO
builds with `key_qco` (`cpp_DINO.fcm`).  Its substitutions
(`WORK/domzgr_substitute.h90:127,137,46,51`) are

    e3u(i,j,k,t)  ->  e3u_0(i,j,k) * (1 + r3u(i,j,t)*umask(i,j,k))
    r1_hu(i,j,t)  ->  r1_hu_0(i,j) / (1 + r3u(i,j,t))

so in `zue * r1_hu(Kaa)` the `(1+r3u(Kaa))` appears once in the sum and once
inverted in the divisor and CANCELS EXACTLY.  NEMO's reconciliation is
TIME-LEVEL INDEPENDENT and weights by the fixed reference ladder `e3u_0/hu_0`.
Verified in the macro file directly, not taken from the review.

Consequences, all recorded as RETRACTIONS:

* **RETRACTED** — "legoESM ... uses the now-level column thickness for an
  after-level update" as a FIDELITY GAP.  There is no after-level thickness in
  NEMO's site 2 to be wrong about.
* **RETRACTED** — the kernel's original "AFTER-level thickness" weighting.  It
  has been changed to the REFERENCE ladder, which is what NEMO uses.  Under
  pure z-star the two agree to roundoff (a column-uniform rescale leaves a
  weighted mean invariant — now gated by its own test); under PARTIAL CELLS
  they differ, because `min`-of-scaled is not `scale`-of-min, and that
  difference was a legoESM-only artifact rather than fidelity.
* **STANDS** — the option's real content: NEMO DISCARDS the implicit vertical
  solve's column-mean deposit every step and legoESM keeps a residue of it.
  Independently measured at ~99.8% of the option's effect.

**(3) The `+0.296` residue's CAUSE is not established.**  R6-d(ii) named the
floor/weighting difference as if it were measured.  At least three mechanisms
contribute and no measurement here separates them: (a) the implicit solve
interpolates cell thicknesses to faces with an ARITHMETIC mean where the
repo's own operator docstring says a partial-cell model must use the MIN rule;
(b) the floor difference as named; (c) in-matrix bottom drag, which makes the
solve genuinely non-mean-preserving rather than merely re-weighted.  The
attribution is downgraded to PLAUSIBLE and (a) is named as a candidate
legoESM defect in its own right, not fixed here.

## SUPERSEDED first-A/B result, recorded in full

Arms at `92b2420c0`, kernel weighting by the live after-level thickness,
`transport_avg` on both.  Both stable to day 90.  Arm A was BIT-IDENTICAL
(max diff exactly 0.0 over every saved field) to the pre-option twin, so
"default off is inert" is measured, not argued.

    band-mean deficit      -0.361  ->  -0.186   (48.5% closure)
    day-90 transport gap   -0.772  ->  -0.394 Sv
    spin-up captured          75%  ->     87%
    full-section ACC gap   -0.5965 ->  -0.3042 Sv
    acceptance gate     PASS 4/FAIL 1 -> PASS 5/FAIL 0

The stage decomposition showed the mechanism exactly: a `POST fixer` row
appeared that cancels the retained `ZDF bt` row identically in every one of the
nine windows (+0.210/-0.210 ... +0.335/-0.335), i.e. legoESM reproducing NEMO's
own discard identity.

These numbers are SUPERSEDED because the kernel's weight changed.  The change
is expected to move them by ~0.2% (the partial-cell-only part), but "expected"
is not "measured", which is why the arms are re-run rather than re-labelled.

## The replacement design: a 2x2, and the prediction for it

| arm | `barotropic_reconcile_target` | `barotropic_after_reconcile` | what it is |
|---|---|---|---|
| A | `transport_avg` | `off` | the card as shipped — the baseline |
| B | `transport_avg` | `nemo_mlf_baro_corr` | NEMO's placement, the average NEMO deletes |
| C | `velocity_avg` | `off` | NEMO's average, no second site |
| D | `velocity_avg` | `nemo_mlf_baro_corr` | **the faithful pair — this is NEMO** |

All four at the same SHA, same IC, same ladder (`both`), corrected clock, fp64,
90 days.  A/B and C/D are each a one-variable comparison; A/C and B/D are each
a one-variable comparison on the other axis.

PREDICTION, recorded before the four arms run:

* **A -> B reproduces** the superseded numbers to within ~0.05 Sv on ACC.  If
  it does not, the reference-ladder change is bigger than the 0.2% argued and
  that is itself the finding.
* **D is the best arm on the acceptance gate**, because it is the only one that
  is NEMO on both rows.  Falsified if D is worse than B.
* **The second site (off -> on) dominates the average (transport -> velocity)**:
  |B-A| and |D-C| are each larger than |C-A| and |D-B|.  This is the direct
  test of the claim that the discard, not the choice of average, is the
  mechanism.  Falsified if the averages axis moves more.
* Both axes are expected to be READABLE (> 0.091 Sv on ACC).  The earlier
  reconcile-target A/B found the averages axis ALONE was climate-inert at
  0.087 Sv on a withdrawn stack; if `C-A` reproduces that null on the valid
  stack, the two lanes finally agree.

---

# RESULT — the 2x2

Committed separately; everything above is unedited (git history proves the
order).  Four arms at `d27dc0909`, clean tracked tree, same IC (NEMO
`DINO_00005760_restart.nc`, day 180), `LEGOESM_NEMO_E3T=both`, corrected
seasonal clock (`t0 = 15552000 s`), fp64 control dtype printed by every log,
dt 2700 s, 90 days = 2880 steps.  All four STABLE.  Every arm's ablation line
is stamped in its own log; arm A carries only the `transport_avg` line, arm D
carries both.  Gate self-checks passed before every comparison (NEMO y10 ACC
121.07 Sv; band volume 2.694775e16 m3, rel 5.0e-08).

## The four arms

| | reconcile target | second site | ACC gap [Sv] | gate | band deficit [m3/s2/row] | day-90 transport gap [Sv] | spin-up captured |
|---|---|---|---|---|---|---|---|
| **A** | transport_avg | off | **0.5965** | 4 PASS / 1 FAIL | -0.361 | -0.772 | 75% |
| **B** | transport_avg | **on** | **0.3042** | **5 PASS / 0 FAIL** | **-0.186** | **-0.394** | **87%** |
| **C** | velocity_avg | off | 0.5694 | 4 PASS / 1 FAIL | -0.294 | -0.628 | 80% |
| **D** | velocity_avg | **on** | 0.4107 | **5 PASS / 0 FAIL** | -0.206 | -0.441 | 86% |

Arm D is the FAITHFUL pair — the only one that is NEMO on both rows.
Arm A is the card as shipped.  Noise floor on ACC: 0.091 Sv.

## The four pre-registered predictions, scored

**1. "A -> B reproduces the superseded numbers to within ~0.05 Sv on ACC."
CONFIRMED**, and far tighter than the bar: the superseded arm B (live
after-level thickness) gave an ACC gap of 0.3042 Sv and the re-run
(reference ladder) gives 0.3042 Sv — a difference of 2e-05 Sv, and the band
deficit and captured fraction are identical to their printed precision.  The
two kernels are NOT bit-identical (day-90 peak differences: 4.3e-02 K in T,
6.3e-03 m/s in u, 4.5e-05 m in eta; at day 30 the trajectories still agree to
3e-08), so the correction is real and its integrated effect on every reported
metric is far below every floor.  The thickness half of R6 is confirmed
negligible on this card by measurement, not only by the algebra.

**2. "D is the best arm on the acceptance gate."  REFUTED on the headline
metric, CONFIRMED on the density metrics.**  This is the uncomfortable result
of the set and it is reported as-is:

| metric | floor | B (transport+on) | D (velocity+on, FAITHFUL) | better |
|---|---|---|---|---|
| ACC gap [Sv] | 9.1e-02 | **0.3042** | 0.4107 | B, by 0.107 (1.2x floor) |
| band deficit | — | **-0.186** | -0.206 | B |
| upper contrast | 1.1e-04 | 2.501e-04 | **2.367e-04** | D |
| deep contrast | 4.5e-05 | 6.216e-06 | **1.544e-06** | D, by 4x |
| S-band sigma MAX | 9.5e-05 | 1.227e-04 | **1.072e-04** | D |
| S-band sigma MEAN | 9.5e-05 | **2.995e-04** | 2.943e-04 | D |

The more faithful configuration is WORSE on the headline transport metric and
BETTER on all four density metrics.  Both pass 5/5.  Two errors partially
cancelling in arm B is the obvious reading and it is PLAUSIBLE, not confirmed —
nothing here isolates which second error the transport average is compensating.

**3. "The second site dominates the choice of average."  CONFIRMED**, on both
metrics and along both slices of the square:

| axis | comparison | ACC move [Sv] | band move |
|---|---|---|---|
| second site | A -> B | **0.2923** | 0.175 |
| second site | C -> D | **0.1587** | 0.088 |
| average | A -> C | 0.0271 | 0.067 |
| average | B -> D | 0.1065 | 0.020 |

Every second-site move is larger than the average move it is paired against.

**4. "Both axes are readable (> 0.091 Sv on ACC)."  PARTIALLY REFUTED.**  The
second-site axis is readable on both slices (0.292, 0.159 = 3.2x and 1.7x the
floor).  The average axis alone is NOT: A -> C moves ACC by 0.027 Sv, below the
floor — which REPRODUCES, on the valid stack, the earlier finding that
`barotropic_reconcile_target` is climate-inert by itself
(`PREREG_gate90_reconcile_ab.md` measured 0.087 Sv on the withdrawn stack, also
below floor).  The two lanes now agree.

## THE STRUCTURAL FINDING: the two knobs INTERACT, and the interaction FLIPS SIGN

The square is strongly non-additive:

    with the second site OFF, switching to NEMO's average IMPROVES ACC by 0.027 Sv (unreadable)
    with the second site ON,  switching to NEMO's average DEGRADES ACC by 0.107 Sv (readable)

Additivity would have predicted arm D at an ACC gap of 0.570 - 0.292 = 0.278;
it measures 0.411.  So the pre-registration's original framing — that the two
fields are "orthogonal" — is refuted twice over: they compose, and their
composition is not even monotone.  Any future single-knob A/B on either field
must state which value of the other it held.

## R6 OWNERSHIP VERDICT — PARTIAL; the pre-registered bar is MISSED, narrowly

The criterion was: the southern-band gap closes by MORE THAN 50% and the
full-section ACC gap shrinks toward the floor.

    best arm (B)      band closure 48.5%   ACC 0.5965 -> 0.3042  gate 4/5 -> 5/5
    faithful arm (D)  band closure 43.0%   ACC 0.5965 -> 0.4107  gate 4/5 -> 5/5

The closure figure is stable across three independent reductions of the band —
48.5% on the per-row torque, 49.0% on the day-90 transport, 48.6% on the
band-summed circulation — so the shortfall against the 50% bar is real and not
a reduction artifact.

**NOT CONFIRMED** on the pre-registered threshold.  Also emphatically **NOT
REFUTED**: the effect is 3.2x the ACC noise floor, moves five independent
metrics in the same direction, is reproducible across a kernel change, and
takes the 90-day acceptance gate from FAILING to PASSING for the first time in
this campaign.  R6's second site is the largest single lever this campaign has
measured on the southern-basin deficit — and it owns about half of it, not all.
The remaining ~51% is still unowned.

## MECHANISM — measured, not inferred

The stage decomposition shows legoESM adopting NEMO's own discard identity.
With the second site ON a `POST fixer` row appears that cancels the retained
vertical-mixing deposit IDENTICALLY in every one of the nine matched windows:

    window        0-10   10-20  20-30  30-40  40-50  50-60  60-70  70-80  80-90
    ZDF bt      +0.210  +0.230 +0.232 +0.237 +0.263 +0.279 +0.289 +0.307 +0.335
    POST fixer  -0.210  -0.230 -0.232 -0.237 -0.263 -0.279 -0.289 -0.307 -0.335

Arm A's POST fixer row is 0.000 in every window.  This is NEMO's identity I2
(`mlf_baro_corr` discards the implicit solve's column mean), which NEMO
satisfies to 3.3e-10 on a 1.1e+04 cancellation scale, now holding in legoESM.

The realized band rate did NOT simply lose the discarded row.  It ROSE, from
+1.069 to +1.244 (NEMO +1.430), because the barotropic-solve row itself rose
from +0.773 to +1.244 — a trajectory response, not a bookkeeping subtraction.
The pre-registration predicted the opposite (see the RETRACTION below).

## GATE TABLE ON THE OPTION ARMS — no Rule-8 finding

No constant was modified and no card default was changed.  Against NEMO day 90,
level 5x, all four density metrics on arms B and D:

* upper contrast, deep contrast and southern surface sigma MAX all IMPROVE
  relative to arm A on both option arms.
* southern surface sigma MEAN DEGRADES slightly on both: 2.821e-04 (A) ->
  2.995e-04 (B), 2.943e-04 (D), i.e. by 1.7e-05 and 1.2e-05 against its own
  9.5e-05 floor.  That is BELOW the floor and therefore UNREADABLE in either
  direction — it is recorded because it moved, not claimed as a degradation.
  **No metric degrades past its floor, so there is no Rule-8 finding here.**

## RETRACTIONS from this result

**RETRACTED — my own pre-registered quantitative prediction, which was wrong in
DIRECTION.**  The first pre-registration predicted that turning the option on
would REMOVE the retained `ZDF bt` row and so drop the band rate from +1.069
toward +0.77, GROWING the NEMO-minus-lego gap to about +0.66.  Measured: the
rate ROSE to +1.244 and the gap SHRANK to +0.186.  The error was treating the
stage rows as independent bookkeeping when the arms' trajectories diverge and
the barotropic solve responds.  A stage decomposition of one arm does not
predict another arm's stage decomposition.

**RETRACTED — "D is the best arm".**  On the headline ACC metric and on the
southern band, the LESS faithful arm B is better than the faithful arm D.

**RETRACTED — "the two fields are orthogonal"** (already withdrawn in the
AMENDMENT on source-reading grounds; now also refuted by measurement, with the
interaction changing sign between the two slices).

## What is still open

* ~51% of the southern-band deficit is unowned, and no named structural DIFF
  remains for it.  The barotropic-solve row still wears it: arm D's band rate
  is +1.244 against NEMO's +1.430.
* Why the faithful pair is worse than the unfaithful one on ACC.  PLAUSIBLE:
  a second error that the transport average partially cancels.  Nothing here
  isolates it, and the honest label is "cause unknown".
* The `+0.296` residue's own cause remains one of three candidates (see the
  AMENDMENT), one of which is a legoESM defect in its own right: the implicit
  solve interpolates cell thicknesses to faces with an ARITHMETIC mean where
  the repo's own operator docstring says a partial-cell model must use the MIN
  rule.  Named, not fixed, not measured.
* No card default is changed by this work.  The option ships off.

## Stage decomposition, both option arms — the discard identity holds on each

Band mean over u-rows 1..13, per 10-day interval, `[m3/s2 per u-row]`.

| stage row | A (card) | B (transport+on) | D (velocity+on, FAITHFUL) | NEMO |
|---|---|---|---|---|
| BARO solve | +0.773 | +1.244 | +1.223 | +1.430 |
| BCLIN expl+diss | +0.000 | +0.000 | +0.000 | — |
| ZDF bt | +0.296 | +0.265 | +0.257 | +17.916 (discarded) |
| ZDF bc | -0.000 | -0.000 | -0.000 | — |
| **POST fixer** | **0.000** | **-0.265** | **-0.257** | (= -ZDF, identity I2) |
| = realized dR/dt | +1.068 | +1.244 | +1.223 | +1.430 |
| **NEMO - lego** | **+0.361** | **+0.186** | **+0.207** | — |

On BOTH option arms the `POST fixer` row equals minus the `ZDF bt` row to the
printed precision in EVERY one of the nine windows — legoESM reproducing NEMO's
identity I2, which arm A does not have at all (its POST fixer is 0.000
throughout).  NEMO's realized row IS its barotropic solve for the same reason.

The independent NEMO-side estimate of the barotropic shortfall moves the same
way and stays **PLAUSIBLE** (its systematic common-mode bound is of order
1 m3/s2 and does not average down): `-0.951 +- 0.211` (A) -> `-0.479 +- 0.184`
(B) -> `-0.500 +- 0.162` (D).

## Artifacts

    /tmp/dino_valid/q_{A,B,C,D}.npz            the four 90-day twins
    /tmp/dino_valid/q_{A,B,C,D}_gate.log       the five-metric acceptance gate
    /tmp/dino_valid/q_{A,B,C,D}_budget.log     the southern circulation budget
    /tmp/dino_valid/q_D_accum.npz              faithful-arm stage decomposition
    /tmp/dino_valid/ab_{A,B}_accum.npz         arm A + superseded arm B stages

Outside the repo, not tracked; each is regenerated by the command recorded in
its own log beside it.

---

# CORRECTION TO THE RESULT — the headline ranking INVERTS, and the verdict changes

Written after dual adversarial re-review of the RESULT above.  The RESULT text
is left unedited; this section supersedes it where they disagree.  **The commit
subject of `e1acaa685` — "the FAITHFUL pair is not the best arm" — IS WRONG and
cannot be edited, so it is retracted here.**

Every number below was re-read from the same artifacts the RESULT used
(`/tmp/dino_valid/q_*_budget.log`, `*_accum.log`); nothing was re-run.

## THE DEFECT IN MY REDUCTION: a band MEAN over 13 rows, five of which carry no physics

The band mean weights all 13 u-rows equally.  It should not have been the only
reduction reported, because the rows are not comparable:

    wall rows 1-5   NEMO's own 90-day spin-up torque  +0.119   (0.6% of the band)
    main rows 6-13  NEMO's own 90-day spin-up torque +18.465   (99.4%)

Rows 1-5 sit hard against the southern wall.  They carry **72%** of the
band-mean deficit while carrying **0.6%** of the circulation NEMO actually
spins up.  Every headline in the RESULT is therefore dominated by five rows
that are nearly inert in the oracle.  The probe printed this split for all four
arms and I did not use it.

## The same four arms, reduced over the rows that carry the circulation

`deficit` = lego minus NEMO, summed over the named rows, `[m3/s2]`
(from each arm's own `V0` block, `q_*_budget.log`):

| arm | | wall rows 1-5 | **main rows 6-13** | **closure vs A, main rows** |
|---|---|---|---|---|
| A | transport + off | -3.376 | **-1.319** | — |
| B | transport + **on** | -1.590 | **-0.824** | **37.5%** |
| C | **velocity** + off | -3.046 | **-0.771** | **41.5%** |
| D | **velocity** + **on** (FAITHFUL) | -2.181 | **-0.503** | **61.9%** |

**The ranking inverts.**  On the rows carrying 99.4% of NEMO's spin-up the
FAITHFUL arm D is the best by a wide margin and arm B is the worst of the three
non-baseline arms.  B's band-mean advantage came entirely from the wall rows.

## The highest-tier evidence in the run directory also points at D, and I omitted it

`southern_term_torque_accum.py` reports a MATCHED-STATE comparison: legoESM's
FIRST step, taken on the state that is bit-identical to NEMO's restart, against
NEMO's own first step.  Same state, same step, one variable — a truth tier above
any 90-day trajectory metric.  Band-mean barotropic-solve torque
`[m3/s2 per u-row]`:

| reconcile target | lego | NEMO | diff |
|---|---|---|---|
| `transport_avg` (arms A and B) | 0.524 | 0.936 | **-0.413** |
| `velocity_avg` (arms C and D) | 0.993 | 0.936 | **+0.057** |

NEMO's own average matches NEMO **7.2x better** on a matched state.  The log's
"barotropic cold start" caveat applies identically to both arms, so the
DIFFERENCE between them is a clean controlled comparison even though neither
absolute number is.

Internal consistency check, unplanned and reassuring: arms A and B give the
IDENTICAL matched-state number (0.524).  They must, because the option's site
is not on the forward-Euler first step — which is exactly the scope boundary
documented on the config field.  The instrument agrees with the code.

## Predictions RE-SCORED on the main-row reduction

**Prediction 3 ("the second site dominates the choice of average") — RETRACTED,
it INVERTS.**  Main-row moves:

| axis | comparison | move |
|---|---|---|
| second site | A -> B | 0.495 |
| second site | C -> D | 0.268 |
| **average** | A -> C | **0.548** |
| **average** | B -> D | **0.321** |

The AVERAGE axis moves MORE on both slices.  The RESULT scored this CONFIRMED
on the band mean; on the reduction that weights the physics it is refuted.

**Prediction 2 ("D is the best arm") — the RESULT's refutation is itself
RETRACTED.**  D is the best arm on the main-row reduction and on the
matched-state comparison.  It remains behind B on the band mean and on
full-section ACC.

## THREE FLOOR ERRORS IN THE RESULT, all in the direction of over-claiming

1. **A difference of two runs was compared against a single-run bar.**  The B-vs-D
   ACC gap of 0.107 Sv was called "1.2x the floor".  The floor for a DIFFERENCE
   is `sqrt(2) x 0.091 = 0.129 Sv`.  **0.107 < 0.129, so B and D are NOT
   separable on ACC.**  The RESULT's "B is better on the headline metric" is
   downgraded to UNREADABLE.
2. **"D is better on all four density metrics, by 4x on deep contrast" is
   UNREADABLE too**, by the RESULT's own standard.  Every B-vs-D density
   difference is 0.05x-0.16x its own floor (upper 1.34e-05 vs 1.1e-04; deep
   4.67e-06 vs 4.5e-05; sigma max 1.55e-05 vs 9.5e-05; sigma mean 5.2e-06 vs
   9.5e-05).  The RESULT refused to call a 1.7e-05 sigma move a degradation
   *because* it was below floor, then read a 4x ratio off numbers further below
   floor.  Both directions retracted: **B and D are indistinguishable on all
   five gate metrics.**
3. **The 0.091 Sv floor is transported from a regime nobody checked.**  It is the
   range of a 3-member NEMO ensemble over 10-YEAR branches from a year-20
   restart, saturating by years 3-5.  These arms are 90-DAY runs from a day-180
   restart.  Whether 0.091 bounds a 90-day spread was never measured, and every
   "Nx the floor" in this document inherits that unvalidated transfer.
   Consequence worth stating: **arm D's fifth gate pass has only 0.044 Sv of
   margin** (0.4107 against a 0.455 threshold), less than one floor.  The
   4/5 -> 5/5 headline is solid for B (0.151 margin) and MARGINAL for D.
   The band deficit and the closure percentage have NO floor quoted at all,
   while the probe prints a 30-day-window scatter of 0.252 m3/s2 per row.

## RETRACTED — the "interaction flips sign" claim, on its own internal contradiction

The RESULT declared A->C (0.027 Sv) below floor and UNREADABLE, then read a
SIGN off that same 0.027 to claim the interaction changes sign.  A number
called unreadable cannot carry a sign.  The interaction contrast itself,
`(A - B - C + D) = 0.1336 Sv`, sits at 1.03x the difference bar of 0.129 —
i.e. right at the edge and not resolved.  **The two knobs are still NOT
orthogonal — that stands on the source reading — but "the interaction flips
sign" is withdrawn as unsupported.**

## RETRACTED — "which the DINO card does have" (partial cells)

The AMENDMENT says the thickness half is "live only through partial cells --
which the DINO card does have".  **It does not.**  DINO runs a pure
z-coordinate with NO partial steps (`namelist_cfg:70-72`, `ln_zco_nam=.true.`,
`ln_zps_nam=.false.`) and legoESM's bridge builds a matching full-step
coordinate.  On this card the thickness half is inert OUTRIGHT, and the kernel
is bit-faithful to NEMO's `SUM_k e3u_0*u*umask / hu_0` rather than merely
faithful-under-z-star.  Also corrected in the kernel docstring: NEMO builds
`e3u_0` as an ARITHMETIC mean of adjacent `e3t_0`, not a minimum; the two
coincide only on a horizontally uniform ladder, which is what DINO has.

## THE CORRECTED VERDICT

**R6's second reconciliation is CONFIRMED as the owner of the majority of the
southern-basin deficit, on the faithful configuration and on the reduction that
weights the physics.**

    main rows 6-13, 99.4% of NEMO's spin-up torque:
        faithful arm D closes 61.9%  -- CLEARS the pre-registered 50% bar
        arm B          closes 37.5%
    band mean over all 13 rows (5 of them near-inert in the oracle):
        arm B          closes 48.5%  -- misses the bar
        faithful arm D closes 43.0%

The pre-registered criterion is MET on the main-row reduction with the faithful
arm and MISSED on the band mean.  Which reduction is right is not a judgement
call: a closure fraction weighted toward rows where the oracle changes by
+0.119 out of +18.584 is measuring the wrong thing.

**The faithful pair is the best configuration measured in this campaign** — best
main-row closure, best matched-state agreement by 7.2x, and 5/5 on the
acceptance gate — and it is not separable from arm B on any gate metric.  The
RESULT's headline, that fidelity and skill point in opposite directions here,
is WITHDRAWN.  They point the same way once the deficit is weighted by where
the circulation actually is.

## What this does NOT change

* ~38% of the main-row deficit is still unowned (arm D, -0.503 against A's
  -1.319), and no named structural DIFF remains for it.
* The mechanism is unchanged and still cleanly measured: both option arms
  reproduce NEMO's discard identity window by window.
* My original pre-registered prediction is still wrong in direction, for the
  reason already recorded.
* No card default changed. The option still ships off.

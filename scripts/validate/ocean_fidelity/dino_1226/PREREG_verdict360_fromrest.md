# PREREG — the FROM-REST spread floor

Written BEFORE any member runs. Companion to `PREREG_verdict360.md`, which
covers the RESTART-seeded twin; this one covers the from-rest year, and it is a
different measurement with a different floor.

## The question

legoESM and NEMO, both started from rest on the same DINO card, differ by a
3-D temperature rms that grows

| day | 30 | 60 | 90 | 180 | 360 |
|---|---|---|---|---|---|
| gap [K] | 2.061e-3 | 2.160e-3 | 2.412e-3 | 4.736e-3 | 4.283e-3 |

Is that evidence of a remaining operator mismatch, or is it inside the spread
each model makes on its own from an infinitesimally perturbed initial state?

## §1 — THE PRE-CHECK THAT CAN CANCEL THE EXPERIMENT

A claim review (fresh independent Claude agent, run on this design before any
code) found the premise may not hold. On the restart-seeded record the
ensemble spread grows by a factor ~2200 between day 90 and day 360, while the
from-rest gap above grows by a factor 1.8 — essentially flat. **A gap that is
flat while chaos grows 2200x is the signature of a bounded deterministic
offset, not of a diverging trajectory.** If that is what is happening, a
day-360 yes/no verdict is a coin-flip on when a growing floor overtakes a flat
gap and says nothing about fidelity.

So the harness runs **PHASE 0 first, and PHASE 0 can refuse PHASE 1**:

* PHASE 0 costs **no NEMO time at all**: four legoESM members from rest, the
  NEMO perturbation applied to the initial temperature, 360 days. It reports
  the ensemble spread at days 30/90/180/360 in the SAME 3-D T rms metric as
  the gap.
* Using legoESM's own spread as the floor estimate is justified by the
  restart-seeded record's own numbers: NEMO's spread there is 9–16600x
  TIGHTER, so 77–99% of the combined `sqrt(spread_lego² + spread_nemo²)` IS
  legoESM's spread.

**PREREGISTERED WINDOW for the day-360 floor, in 3-D T rms [K]: `[2.0e-3,
1.0e-2]`.** It is a constant in the harness (`FLOOR_WINDOW_K`), not a
judgement:

| measured day-360 floor | verdict | action |
|---|---|---|
| `< 2.0e-3` | DEAD ON ARRIVAL — `2*floor` is already below the 4.283e-3 K gap, so "distinguishable" is fixed before the members run | do NOT spend PHASE 1; the next measurement is an operator, not an ensemble |
| `2.0e-3 … 1.0e-2` | informative | PHASE 1 is worth its NEMO time |
| `> 1.0e-2` | the floor overtakes the gap INSIDE the year | the deliverable is the CROSSING DAY, not a day-360 binary |

`nemo_dino_fromrest_members/run.sh` **refuses to run** unless
`phase0_floor.json` exists and its day-360 floor clears the low edge. That is a
hard gate in the script, not an intention.

## §2 — The perturbation is NEMO's own, and it needs no source patch

DINO already carries it. `cfgs/DINO/MY_SRC/usrdef_istate.F90:177-183`, compiled
at `BLD/ppsrc/nemo/usrdef_istate.f90:188-194`:

```fortran
IF( nn_pert_seed /= 0 ) THEN
   DO jk = 1, jpk
      pts(:,:,jk,jp_tem) = pts(:,:,jk,jp_tem) + 1.e-10_wp                  &
         * SIN( REAL( NINT(pdept(:,:,jk))*73 + NINT(gphit(:,:)*1000._wp)*179 &
                    + nn_pert_seed*997, wp ) ) * ptmask(:,:,jk)
```

`nn_pert_seed` is declared at `MY_SRC/usrdef_nam.F90:81` and is already in the
`namusr_def` list (`:126`). DINO from rest runs `usr_def_istate`
(`ln_rstart=.false.`, `nn_initcase=4`), so **each NEMO member is one namelist
line on the certified binary.** No MY_SRC patch, no second source tree.

Two properties of it are easy to get wrong and are therefore asserted by the
harness rather than assumed:

1. it is **1e-10 ABSOLUTE**, not relative (≈5e5 ulp on a ~10 K field — far
   above fp64 rounding, physically meaningless, never written to a file);
2. its argument uses only DEPTH and LATITUDE, so the perturbation is **ZONALLY
   UNIFORM**. `verdict360_fromrest.py` prints `max over lon of (max-min) within
   a row` on WET cells and fails if it is not exactly 0. (The first version of
   that check measured the land MASK instead and reported the whole amplitude;
   corrected.)

legoESM reproduces the same expression statement for statement in
`nemo_istate_perturbation`, including Fortran `NINT`'s round-half-AWAY-FROM-ZERO
(`np.round` is round-half-to-EVEN, and `gphit*1000` lands on exact halves on a
regular grid, so that is not pedantic).

## §3 — Scoring: pairwise distances, not a difference of means

`verdict360.py`'s `|mean_A − mean_B|` against a spread is valid for a SIGNED
SCALAR — a transport, a density contrast. For a positive-definite FIELD NORM it
is a category error: the difference of two rms values is not a distance. The
field rows therefore use the two-sample distance statistic on

* `within` = `d(A_i, A_j)` pooled with `d(B_i, B_j)` — the FLOOR
* `across` = `d(A_i, B_j)` — the GAP

with `d` the wet-masked 3-D rms difference. No mean field is ever formed.
The scalar rows (ACC, the density contrasts the verdict used) keep the
mean-difference form, which is valid for them.

**The permutation test is a SUPPORTING column, never the claim.** 4 vs 4 gives
`C(8,4)/2 = 35` distinct splits, so the smallest achievable two-sided p is
`2/70 = 0.0286`, over ~10 correlated metrics with no multiplicity correction.
It cannot carry a verdict and is not asked to.

## §4 — The deliverable is a CROSSING DAY

`INDISTINGUISHABLE at day 360` is reported, but the headline number is the
**first day at which the gap falls inside `2*floor`**, or "never within the
year". A binary at one arbitrary day hides the shape, and the shape is the
finding.

## §5 — ASKED

**ASKED, not taken:** producing the legoESM members needs the perturbed initial
state to reach `run_dino.py`. A `--member-seed` / `--initial-T-perturbation`
flag would be a NEW PRODUCTION KNOB, so it is not added.

*Recommendation: do not add the flag.* The harness can build the card's initial
state, apply the perturbation, and call the model directly — it already builds
that state to write the perturbation fields — which needs no production
surface at all and keeps the ensemble machinery out of the driver every DINO
run uses. The alternative (add the flag) is one CLI argument plus a round-trip
test, and is the right call only if members are wanted from the shipped driver.
`verdict360_fromrest.py --phase0` prints the command it WOULD use and says, in
its own output, that the flag does not exist yet.

## §6 — What this cannot see

* The floor is estimated from legoESM's members alone in PHASE 0. The 77–99%
  share quoted above comes from the RESTART-seeded record; whether NEMO is
  equally tight FROM REST is unmeasured until PHASE 1.
* Four members is a small ensemble; the floor itself has sampling error that is
  not propagated into the window comparison.
* Only the 3-D T rms is windowed. The transports are scored in PHASE 1.

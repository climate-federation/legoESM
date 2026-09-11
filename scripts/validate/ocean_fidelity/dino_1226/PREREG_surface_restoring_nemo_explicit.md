# PREREG — the DINO surface restoring moves to NEMO's EXPLICIT form (decision 30)

Written BEFORE the code change.  Companion to the kt=1 surface gate
(`kt1_surface_gate.py`), which produced the statement this file acts on.

## §0 — What the previous round established, and what it did not

MEASURED, and it is the whole reason this file exists: the step-1 temperature
residual (4.7912e-06 K rms, all at level 0) is the surface restoring's
implicit-Euler denominator.  The salinity row is the clean test because salt has
no solar member, and the TEMPERATURE ratio is a BLEND, not the factor — a claim
review caught this file over-claiming it.  The measured T level-0 ratio was
0.996917045 while `tau_T/(tau_T+dt)` is 0.997405894; the two differ because the
`Q_sr` subtraction term never carried the denominator, so the covariance weights
a damped member against an undamped one.  Only the salt row lands on the
predicted factor:

    predicted S ratio  tau_S/(tau_S+dt) = 0.998999633
    MEASURED  S ratio                   = 0.998999633   (difference 5.6e-16)
    rms(lego - pred*nemo)               = 2.6704e-22 K/s  = 2.5e-15 of NEMO's rms

so the deficit is the factor, not merely correlated with it.

NOT established, and not claimed: that the denominator is the ONLY level-0
statement that differs.  §2 lists the ones that remain and registers them.

## §1 — NEMO, statement by statement, as COMPILED

All citations are `cfgs/DINO/BLD/ppsrc/nemo/<file>.f90:LINE` — the DINO build's
own preprocessed source, not `src/OCE`.  The card runs `nn_forcingtype = 4`
(`RUN_TRAJ/namelist_cfg:28`), so `usrdef_sbc`'s `CASE(4)` is the live branch.

| # | NEMO statement | cite |
|---|---|---|
| S1 | `sfx(ji,jj) = ( rn_srp * ( ts(ji,jj,1,jp_sal,Kbb) - zsstar(ji,jj) ) ) * tmask(ji,jj,1)` | `usrdef_sbc.f90:223` |
| S2 | `emp(:,:) = 0._wp` — `rn_emp_prop=0` (namelist `:36`), `ln_emp_field=F` (namelist `:47`), `ln_qns_field=F` (**absent from the namelist**, so the module default at `usrdef_nam.f90:55` holds), so the ELSE branch runs | `:270` |
| S3 | `qtot(ji,jj) = rn_trp * ( ts(ji,jj,1,jp_tem,Kbb) - ztstar(ji,jj) ) - emp(ji,jj)*ts(ji,jj,1,jp_tem,Kbb)*rcp*tmask` | `:272-273` |
| S4 | `zqsr_dayMean = MAX( 230*COS( rpi*(gphit - 23.5*zcos_sais1)/180 )*tmask, 0 )`; `ln_diu_cyc=F` so `qsr = zqsr_dayMean` | `:283`, `:605` |
| S5 | `qns(ji,jj) = ( qtot(ji,jj) - zqsr_dayMean(ji,jj) ) * tmask(ji,jj,1)` | `:294` |
| S6 | `sbc_tsc(:,:,jp_tem) = r1_rho0_rcp * qns` ; `sbc_tsc(:,:,jp_sal) = r1_rho0 * sfx` | `trasbc.f90:153-154` |
| S7 | the `emp * pts(...,Kmm)` concentration/dilution term is inside `IF( lk_linssh )` | `trasbc.f90:156-165` |
| S8 | `lk_linssh = .FALSE.` — DINO compiles `key_qco key_vco_3d` and no `key_linssh` | `dom_oce.f90:137`, `cfgs/DINO/cpp_DINO.fcm` |
| S9 | `zfact = 1.0` on the no-restart Euler start with `sbc_tsc_b = 0`; `zfact = 0.5` with the swapped `sbc_tsc_b` thereafter | `trasbc.f90:138-150` |
| S10 | `pts(ji,jj,1,jn,Krhs) += zfact*(sbc_tsc_b + sbc_tsc) / ( e3t_3d(ji,jj,1) * (1 + r3t(ji,jj,Kmm)*tmask(ji,jj,1)) )` | `trasbc.f90:169-170` |
| S11 | `rho0 = 1026.`, `rcp = 3991.86795711963`, `r1_rho0_rcp = 1/(rho0*rcp)` | `eosbn2.f90:1947-1948,2380-2383` |

**There is NO damping denominator anywhere in S1..S10.**  NEMO's restoring is a
flux evaluated on the BEFORE tracer and applied forward; the only division is by
the live first-level thickness.

`fraqsr_1lev` (`traqsr.f90:270-271`, written to the restart at `:294`) is the
fraction of `qsr` absorbed in level 1.  **WAIVED, with reason:** it feeds the
ice-ocean `qsr_hc` bookkeeping only; on an ice-free DINO it never re-enters the
tracer tendency, and the tracer-side solar term is `traqsr`'s two-band ladder,
which this gate already scores separately (`shortwave_penetration_ladder`).

## §2 — legoESM, and the four rows that differ

Resolved from the card, printed not assumed (Rule 10):
`A_theta = 40.0`, `A_S = 0.003858`, `rho_0 = 1026.0`, `c_p = 3991.86795711963`
— bit-equal to `|rn_trp|`, `|rn_srp|`, `rho0`, `rcp`.

| # | row | disposition |
|---|---|---|
| L1 | coefficients and the W/m^2 -> K/s conversion | **MATCH** — `tau_T = rho_0*c_p*dz_0/A_theta` is S6 divided by S10's thickness, algebraically identical |
| L2 | the divisor: `surface_flux_divisor='nemo_live'` gives `dz_0*(1+r3t)` | **MATCH** to S10 |
| L3 | `emp`/concentration-dilution | **MATCH** — neither model applies it (S2 and S8) |
| L4 | **`RestoringConfig(implicit=True)` -> denominator `(tau + dt)`** | **DIFF — this round's change.**  NEMO has no such factor (§1). |
| L5 | the flux reads the NOW tracer; NEMO reads `ts(...,Kbb)` (S1, S3) | **DIFF — REGISTERED, NOT FIXED.**  Invisible at kt=1 from rest (`Kbb == Knn`).  Fixing it needs the applicator to be handed the before-level tracer. |
| L6 | legoESM applies `sbc_tsc` alone; NEMO applies `0.5*(sbc_tsc_b + sbc_tsc)` from kt=2 on (S9) | **DIFF — REGISTERED, NOT FIXED.**  Invisible at kt=1 (`zfact=1`, `sbc_tsc_b=0`).  Fixing it adds a carried 2-D surface-flux memory, i.e. new model state, which is an ASKED item. |

L5 and L6 are the reason the year-long gap is NOT predicted to close to the
floor by this change alone, and this file says so BEFORE the year is re-run.

**How big are they, and are they the same KIND of error?**  A claim review sized
them and the answer changes how the year should be read.  Together they place
legoESM's flux at step *n* where NEMO's is at *n - 1.5*, an instantaneous flux
offset of `1.5 x dt/tau_T = 3.9e-03` — larger than the `2.6e-03` this change
removes.  But they are **LAGS**, and a lag vanishes at steady state (all three
time levels coincide), whereas the implicit denominator was a multiplicative
GAIN error that biases the fixed point permanently.  The seasonal phase shift a
1.5-step lag buys on a 360-day cycle is 68 minutes.  So the year-long headline is
not made misleading by leaving them — but **no kt >= 2 surface gate may be called
AT BAR while they stand**, and the check that would size them directly is this
same gate run against a kt=2 or kt=3 record (a level-0 ratio near 0.9961 confirms
the arithmetic).

## §3 — THE CHANGE

One statement, in the one shared DINO surface path:

    packages/ocean/legoesm/ocean/experiments/dino.py
      apply_dino_lat_lon_surface_forcing:  RestoringConfig(..., implicit=True)
                                        -> RestoringConfig(..., implicit=False)

No knob, no new config field, no default that keeps the implicit form
(user decision 30, under the standing "do exactly what NEMO does" directive).
`box_heat_budget.py`'s mirror of the same statement moves with it, because a
diagnostic that silently disagrees with the model it diagnoses is worse than no
diagnostic.

**Rule 9 pre-check, so the change cannot be mistaken for removing a needed
stabiliser:** `dt/tau_T = 2700 / 1.038e6 = 2.6e-03` and
`dt/tau_S = 2700 / 2.696e6 = 1.0e-03`.  Explicit relaxation is stable for
`dt < 2*tau`; the card is four orders inside that.  The implicit denominator was
never load-bearing on this card.

## §4 — PREREGISTERED step-1 table, and the falsifier

`kt1_surface_gate.py`, unchanged, run before and after.  The gate's own
sub-surface row gives the instrument floor: **3.0136e-20 K/s rms**, which is
what "exact" costs in fp64 here.

| row | BEFORE (measured) | PREDICTED AFTER | falsified if |
|---|---|---|---|
| `S: level 0 only` cells != | 9920 / 9920 | **0** | any nonzero cell count |
| `S: level 0 only` res rms | 1.0786e-10 K/s | **<= 3.0e-20 K/s** | `> 1e-18` |
| `S: level 0 only` ratio | 0.998999633 | **1.000000000** | departs by `> 1e-12` |
| `T: level 0 only` res rms | 1.0806e-08 K/s | **<= 3.0e-20 K/s** | `> 1e-18` |
| `T: level 0 only` ratio | 0.996917045 | **1.000000000** | departs by `> 1e-12` |
| `T: sub-surface only` | 3.0136e-20, ratio 1.0 | **unchanged** | it moves at all |
| `level 0: vs sbc_hc_b/e3t` | 2.9415e-06, ratio 0.677 | ratio **1.000000000** | departs by `> 1e-12` |
| GATE verdict | `GATE FAIL (6 rows)` | **`GATE PASS`, exit 0** | any row DEBT |

**The falsifier that matters:** if T or S level 0 stays above `1e-18` K/s after
the change, the implicit denominator was NOT the level-0 statement and the
remainder belongs to something else in §1 — most likely S10's thickness or S4's
solar term.  A PARTIAL fall (say 1e-8 -> 1e-12) would be the most informative
outcome of all and is explicitly allowed to happen.

The `T: rate vs ttrd_nsr alone` / `ttrd_qsr alone` rows are NOT predicted to
reach the bar: they compare legoESM's single combined level-0 rate against one
half of NEMO's split, and the split is `traqsr`'s, not `trasbc`'s.  They stay
DEBT by construction and the gate is expected to keep saying so.

## §5 — Rule 12: which cards move

Every caller of `apply_dino_lat_lon_surface_forcing` moves, because the
statement is unconditional in the shared applicator.  The sweep enumerates them
and MEASURES the size of the move rather than asserting it.

The DINO paper's own reference run **is NEMO** (Kamm, Deshayes & Madec 2025 run
the NEMO configuration this oracle is built from), so NEMO's explicit form IS
the paper's form: the paper cards are moving TOWARD their reference, not away.
That is the argument for letting them move; the decision is still the user's and
it is in the ASKED table.

`neverworld2_lite` is the one caller whose reference is NOT NEMO.  It reuses the
DINO applicator through `_to_dino_cfg`.  Its move is reported with a number and
placed in the ASKED table.

## §6 — MEASURED (filled in after the runs)

`kt1_surface_gate.py`, same commit, one variable, the model rebuilt for each arm:

| row | BEFORE | AFTER | predicted? |
|---|---|---|---|
| `S: level 0` cells != | 9920 / 9920 | 6048 / 9920 | **NO — see below** |
| `S: level 0` res rms [K/s] | 1.0786e-10 | **2.6717e-22** | yes (`<= 3.0e-20`) |
| `S: level 0` ratio | 0.998999633 | **1.000000000** | yes |
| `T: level 0` cells != | 9920 / 9920 | 8518 / 9920 | **NO** |
| `T: level 0` res rms [K/s] | 1.0806e-08 | **4.2669e-21** | yes (`<= 3.0e-20`) |
| `T: level 0` ratio | 0.996917045 | **1.000000000** | yes |
| `T: sub-surface` res rms | 3.0136e-20, ratio 1.0 | 3.0136e-20, ratio 1.0 | yes, unchanged |
| gate verdict | GATE FAIL (6 rows) | GATE FAIL (6 rows) | **NO** |

**The two rows the preregistration got WRONG, and they are mine, not the
model's.**

1. I predicted `cells != -> 0` and `GATE PASS`. That cannot happen and I should
   have seen it: NEMO computes `r1_rho0_rcp * qns / e3t` while legoESM computes
   `-(T - T*)/tau` with `tau = rho_0*c_p*dz_0/A_theta`. Those are algebraically
   identical and DIFFERENTLY ASSOCIATED, so they differ in the last bits of
   fp64 — 6048 and 8518 cells at 1e-21..1e-22 K/s, i.e. 1.4e-13 of NEMO's own
   level-0 rms and BELOW the gate's own sub-surface floor. The bar is exact and
   is NOT relaxed (Rule 1b): those rows stay DEBT and the gate keeps exiting
   non-zero. What changed is that the DEBT is now rounding rather than a
   statement.
2. I predicted the `level 0: vs sbc_hc_b/e3t` row would reach ratio 1. It reads
   0.678615997. That row compares legoESM's single COMBINED level-0 rate
   against NEMO's NON-SOLAR HALF alone, and legoESM has no such split — the
   combined row (`T: rate vs nsr+qsr`) is the comparable object and it is at
   ratio 1.000000000 on every level. The prereg named the `ttrd_nsr alone` /
   `ttrd_qsr alone` rows as blends-by-construction and then failed to apply the
   same reasoning to this third row. No model finding; a prereg error.

`nemo_dino_step1_gate.py`, the same two arms, each a fresh one-step
`run_dino.py` on the shipped card (the BEFORE arm reproduced the frozen
4.7912e-06 literal exactly, which is what makes it a valid control):

| field | BEFORE rms | AFTER rms | factor |
|---|---|---|---|
| T | 4.7912e-06 K | **1.1760e-07 K** | 40.7x |
| S | 4.9646e-08 | **1.0245e-08** | 4.85x |
| eta | 6.8206e-07 | 6.8206e-07 | **byte-identical** |
| u | 1.0816e-08 | 1.0816e-08 | **byte-identical** |
| v | 6.4589e-08 | 6.4589e-08 | **byte-identical** |

The three unchanged rows are a NO-COLLATERAL check, **not a control** — a claim
review was right that they are structurally incapable of moving: at kt=1 the
tracers are updated after momentum and the free surface, so a tracer-RHS change
cannot reach `eta`/`u`/`v` inside the step no matter how large it is.  They rule
out a stray edit; they cannot falsify anything about the restoring.

**The floor the level-0 rows are read against.**  The prereg used the gate's
sub-surface residual (3.0136e-20 K/s) as "the floor", and a claim review was
right that it is borrowed from the wrong magnitude class: NEMO's level-0 rate rms
is 3.5e-06 K/s, so the honest fp64 bar there is `eps x 3.5e-06 = 7.8e-22 K/s`.
The measured 4.2669e-21 K/s is **5.5 x that bar**, not below it — consistent with
a handful of last-bit operations in a differently-associated expression, and
reported as such rather than as "at the floor".

## §7 — Rule 12 card sweep, MEASURED

`surface_restoring_card_sweep.py`, all 8 callers, one dt (2700 s), each card's
own grid/state/forcing, the applicator called through its own path in both arms:

| card | tau_T [d] | tau_S [d] | T max rel move | S max rel move | dt/(2 tau) |
|---|---|---|---|---|---|
| legoesm_default | 11.9948 | 31.1542 | 3.1913e-03 | 1.0021e-03 | 1.303e-03 |
| mitgcm | 11.9948 | 31.1542 | 3.1913e-03 | 1.0021e-03 | 1.303e-03 |
| nemo_dino_kamm | 12.0153 | 31.2073 | 3.0763e-03 | 1.0004e-03 | 1.300e-03 |
| nemo_dino_kamm_mlf | 12.0153 | 31.2073 | 3.0763e-03 | 1.0004e-03 | 1.300e-03 |
| nemo_paper | 11.9948 | 31.1542 | 3.1913e-03 | 1.0021e-03 | 1.303e-03 |
| oceananigans | 11.9948 | 31.1542 | 3.1913e-03 | 1.0021e-03 | 1.303e-03 |
| veros | 11.9948 | 31.1542 | 3.1913e-03 | 1.0021e-03 | 1.303e-03 |
| neverworld2_lite | 11.9948 | 31.1542 | 3.1991e-03 | 1.0021e-03 | 1.303e-03 |

Every salt row equals the closed form `dt/(tau_S+dt)` to 1e-9, which is what
makes the sweep a measurement rather than a restatement. Largest stability
margin 1.3e-03, i.e. every card is ~770x inside the explicit bound.

## §7b — THE YEAR, measured

360 days from rest on the shipped card
(`nemo_faithful_kamm_mlf.yaml`, `--days 360 --snapshot-every-days 30`), scored
by `twin_nemo_ts_maps.py --run-dino-dir` against NEMO's own trajectory at the
matched steps (days 30/60/90/180 vs `RUN_TRAJ` kt 960/1920/2880/5760; day 360 vs
`RUN_FROMREST_Y1` kt 11520).  3-D wet-masked T rms, fp64 both sides, 339744 of
354600 cells wet.

| day | 30 | 60 | 90 | 180 | 360 |
|---|---|---|---|---|---|
| BEFORE [K] | 2.039e-03 | 2.157e-03 | 2.304e-03 | 4.590e-03 | 3.924e-03 |
| **AFTER [K]** | **9.138e-04** | **1.212e-03** | **1.808e-03** | **3.133e-03** | **3.690e-03** |
| after / before | 0.448 | 0.562 | 0.785 | 0.683 | 0.940 |
| Phase-0 spread floor [K] | 1.2e-05 | — | 8.8e-04 | 1.4e-03 | 5.5e-03 |
| gap / (2 x floor) | 81.8 -> **38.1** | — | 1.31 -> 1.03 | 1.64 -> 1.12 | 0.36 -> 0.34 |

Read the LEFT of that table, not the right.  Phase 0 already established that
the floor grows 437x over the year while the gap grows ~1.9x, so by day 180 the
floor is the same size as the gap and days 180/360 cannot discriminate anything.
Day 30, where the floor is 1.2e-05 K, is the informative column: **the gap fell
55%**, from 81.8 to 38.1 times the distinguishability threshold.  It did not
reach the floor, and §2's L5/L6 (the Kbb time level and the two-step flux
average) are the registered reason it was not expected to.

The model that produced this year is BIT-IDENTICAL to the committed tree: a
one-step re-run after every later edit reproduces
`snapshot_00001.npz` sha256 `5d0256a12d03c3fcbf53bb458b652b8314fd8437478fdda3082358696d643feb`,
so the operand-diagnostics and gate edits that landed after the run are inert on
the model, as claimed rather than assumed.

## §8 — ASKED / UNASKED

**ASKED, one line, and the work does not depend on the answer.**
Seven of the eight cards above are DINO cards whose reference run IS NEMO, so
the explicit form is their reference's form and they move toward it. The eighth
is not:

> `neverworld2_lite` reuses the DINO surface applicator and its reference is
> NOT NEMO. Its surface restoring moved from the implicit form to NEMO's
> explicit form, changing its trajectory by 3.2e-03 (T) and 1.0e-03 (S) of one
> step's surface increment. **Does neverworld2_lite follow the DINO card to
> NEMO's form (X, what this diff does), or does it keep the implicit form
> through its own recipe (Y)?** My recommendation is X: the implicit
> denominator was never a NeverWorld2 choice, it was inherited from a DINO
> default, and at `dt/(2 tau) = 1.3e-03` it buys nothing. But it is a change to
> a non-NEMO card's physics and the decision is the user's.

**UNASKED: none.** No config field was added, no default was moved beyond
decision 30, no YAML key changed, and the two registered statements in §2
(the Kbb time level and the two-step flux average) were deliberately NOT
implemented, because carrying `sbc_tsc_b` is new model state.

## §9 — What the two adversarial reviews changed

Two fresh independent Claude agents (codex is on the GYRE lane, GLM
unavailable), one on the CLAIMS and one on the DIFF.  The findings that changed
this file or the code are folded in above; the ones that changed something else
are listed here so they are not lost:

* the Rule-12 sweep **exited 0 with the restoring's SIGN FLIPPED**, with the
  salt restoring **deleted** (NaN counted as a pass), and with the heat flux
  doubled — it only ever compared the two arms, so any defect common to both
  cancelled.  It now scores an ABSOLUTE independent transcription of the salt
  statement plus both arm-to-arm closed forms, treats NaN as failure, and both
  of the reviewer's plants now make it exit 1 (measured).
* **two committed tests were red** and are fixed: two independent
  transcriptions in `test_dino_experiment.py` hard-coded the `(tau + dt)`
  denominator, and `test_box_heat_budget.py`'s dt-conflation guard used the
  FORCING term as its probe — a probe that only worked BECAUSE of the implicit
  denominator.  That guard now pins FORCING's dt-INVARIANCE (which goes red if
  the denominator returns) and its calibration half is an explicit
  `xfail(strict=True)` rather than a quietly dead assertion.
* the corrected `ah_wslp2` row is **blind to the roll it fixed**, and a first
  attempt to close that in the gate was itself vacuous (rolling BOTH sides is a
  permutation and every statistic is identical).  The blind spot is now written
  into the gate and the roll direction is pinned by a unit test that goes red on
  the flip.
* **"akz is zero, so the implicit path is inert" is a FROM-REST ACCIDENT**, and
  the gate now says so with the threshold: `akz` fires once
  `|slope| > e3w*sqrt(0.5/(rDt*aht)) ~ 2.6e-03` at `e3w = 10 m`, well inside
  `rn_slpmax = 0.01`.  DINO's initial slopes are ~1e-04.  A Y1/Y5 record is
  required before that row is quoted for a spun-up state.
* the kt=2 attribution is **not isolated**: `eta=state.eta_before` feeds the
  whole Nbb `_step_impl` pass, so lateral viscosity and the GM/eiv trend move
  their `e3` too.  The discriminating check is to inject the now-height into
  the isoneutral operator alone.
* the `box_heat_budget` mirror **already drifts** from the applicator in a
  second field (static `dz_ref[0]` vs the applicator's `nemo_live` divisor on
  the two oracle cards); recorded in the module, not fixed here.

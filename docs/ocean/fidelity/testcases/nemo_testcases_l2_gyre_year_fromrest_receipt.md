# GYRE from-rest YEAR, round 1: PHASE 0 IS BLOCKED — the TKE closure diverges at day 6

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_fromrest.md`
(written and independently reviewed twice BEFORE any member ran).
Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/`,
SHA-256 in `phase0_blocker_artifacts.sha256`.

## VERDICT

**The GYRE year cannot be run.**  legoESM's certified GYRE card
(`GYRE-zco`, `build_gyre_zco_card`) aborts at **step 48 of 2160**, and the
reason is not the abort's message.

The owner is identified, not merely localised: in ONE column -- the south-east
corner of the wet domain, at the deepest interior interface -- the model's
turbulent kinetic energy obeys

```
e(n+1) = e(n) + (0.5 * rn_ediss * dt / L) * e(n)^1.5
```

with the coefficient matching `0.5 * 0.7 * 14400 / 306.5 = 16.4437` to five
significant figures over five consecutive steps.  That is the EXPLICIT half of
NEMO's semi-implicit dissipation split running as a SOURCE with nothing on the
implicit diagonal to balance it.  `e` reaches `3.5e40 m2/s2` by step 47, the
eddy viscosity reaches `5.6e21 m2/s`, temperature and salinity go infinite and
the velocities go NaN.  The sea surface inherits the NaN inside step 48 and the
vertical-geometry guard in `eos.py:736-740` is what finally reports it.

This is a **NEW, previously unmeasurable finding**: the kt=1..10 ladder that
certified this card stops 38 steps before the instability starts.  "Never
measured beyond ten steps" turns out to mean "cannot currently be measured
beyond 47".

CONFIRMED.  Three readings, all stopping at the same step: the year harness
itself (all four members), a bare stepping walk, and the geometry trace.  They
are three READINGS OF ONE CODE PATH, not three independent instruments -- all
call the certified gate's `_surface_forcings` and `LatLonCGridOceanModel.step`
-- so they establish reproducibility, not independence.  (Corrected after
review; the first version of this line claimed independence.)

## The measurement

The probe is COMMITTED, not a heredoc: `--census` on the year harness
(`nemo_testcase_l2_gyre_year_fromrest.py`).  It is fp64 / CPU / scalar-libm and
steps the card through the SAME functions the ten-step gate uses
(`nemo_testcase_l2_gyre_phase3_gate._surface_forcings` +
`LatLonCGridOceanModel.step`), and it EXITS NON-ZERO on the first non-finite
value rather than printing a table someone has to read:

```
STATUS NONFINITE at step 47: u (17250 cells), v (16950 cells),
                             T (16350 cells), S (16350 cells)
```

It reports one step EARLIER than the model's own abort, and it names the
fields.

| step | max `tke` [m2/s2] | max `tke_avm` [m2/s] | eta, u, v, T |
|---:|---:|---:|---|
| 1 | `5.538e-03` | `8.188e-02` | healthy |
| 20 | `2.208e-03` | `1.136e-01` | healthy |
| 30 | `2.205e-03` | `6.237e-01` | healthy |
| 36 | `5.779e-03` | `2.286e+00` | healthy |
| 37 | `1.298e-02` | `3.426e+00` | healthy |
| 38 | `3.726e-02` | `5.805e+00` | healthy |
| 39 | `1.554e-01` | `1.185e+01` | healthy |
| 40 | `1.161e+00` | `3.241e+01` | healthy |
| 41 | `2.173e+01` | `1.402e+02` | healthy |
| 42 | `1.687e+03` | `1.235e+03` | healthy |
| 43 | `1.141e+06` | `3.212e+04` | healthy |
| 44 | `2.004e+10` | `4.257e+06` | healthy |
| 45 | `4.665e+16` | `6.495e+09` | healthy |
| 46 | `1.657e+26` | `3.871e+14` | healthy |
| 47 | `3.507e+40` | `5.631e+21` | **T, S = inf; u, v = NaN (17250 faces)** |
| 48 | — | — | abort |

**The dynamics are healthy the entire time.**  Right up to step 46:
`eta` in `[-4.90e-2, 4.77e-2] m`, `u` in `[-8.11e-2, 5.44e-2] m/s`,
`v` in `[-2.97e-2, 1.52e-1] m/s`, `T` in `[0, 23.83] C`,
`uu_b` in `[-1.11e-2, 1.98e-2] m/s`.  The advective CFL is about `0.011`.
There is no barotropic mode, no CFL violation and no front.  Only `tke`,
`tke_avm`, `tke_avt` and `tke_dissl` move, and they move together.

`tke` FALLS from `5.5e-3` at step 1 to `2.2e-3` by step 20-30, then TURNS
AROUND between steps 30 and 36 and runs away, ACCELERATING.  An earlier draft
of this receipt read that acceleration as a THRESHOLD crossing and pointed at
the convective switch.  **That reading is RETRACTED**: the acceleration is
fully explained by the recursion identified below, which needs no threshold,
and two independent reviews said so before any further compute was spent.  An
adjacent paragraph claiming `tke_avt` tracked `tke_avm/10` and then became
equal is **also RETRACTED** -- the two are maxima taken independently and need
not sit in the same cell, and the log contradicts the claim anyway (they are
EQUAL at steps 10 and 20 and are NOT equal at step 46).

## THE OWNER, and it is not a mystery

The census already contains NEMO's mixing length.  `dissl` is a RATE,
`sqrt(en)/L` (`zdftke.F90:717`), not a length -- so `L = sqrt(tke)/tke_dissl`
is readable straight off the table:

| step | 36 | 38 | 40 | 41 | 42 | 43 | 44 | 45 | 46 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L` [m] | 306.93 | 306.86 | 306.71 | 306.60 | 306.52 | 306.50 | 306.50 | 306.50 | 306.51 |
| `e(n+1)/e(n)^1.5` | 29.5 | 21.6 | 17.4 | 16.65 | **16.4669** | **16.4426** | **16.4439** | **16.4454** | **16.4419** |

and `0.5 * rn_ediss * dt / L = 0.5 * 0.7 * 14400 / 306.5 = ` **`16.4437`**.

**The recursion is `e(n+1) = e(n) + (0.5 * rn_ediss * dt / L) * e(n)^1.5`**,
matched to five significant figures over five consecutive steps.  That is the
EXPLICIT half of NEMO's semi-implicit dissipation split (`zdftke.F90:419`)
running as a SOURCE with no implicit counterpart on the diagonal (`:414`).
Shear, buoyancy and TKE diffusion are numerically absent from the balance at
that cell: nothing else in the closure can produce an `e^1.5` map with that
coefficient.

This identification came from an independent review of this receipt, from the
numbers already in it.  **The arithmetic was re-derived here before it was
adopted** -- a reviewer's finding is a hypothesis, not an instruction -- and it
reproduces: the measured ratios are `16.4419` to `16.4669`, the prediction is
`16.4437`.

### WHERE, measured rather than inferred

The reviewer predicted the peak would sit on the deepest active interface, and
named the one-line measurement that would settle it: print `argmax(tke)`.  Run
(`phase0_blocker_census_argmax.log`):

| steps | argmax(tke) | |
|---|---|---|
| 30-34 | wanders, `k=0`, `tke ~ 2e-3` | surface, decaying |
| **35-47** | **`j=1, i=30, k=28`**, pinned | the runaway |

`(j=1, i=30)` is the **south-east CORNER column of the wet domain** -- wet `j`
runs 1..20 and wet `i` runs 1..30, and that column has 2 wet neighbours of 4.
`k=28` is the **second-deepest of 30 levels** (`bottom_level = 29`), at
`gdept = 3849.90 m`.  So the runaway is ONE COLUMN, at the deepest interior
interface, i.e. the last interior row of the vertical tridiagonal solve.

`L` pins at `306.50 m` there from step 42, which is LARGER than any thickness
in GYRE's ladder (`max e3t_1d = 301.10 m`, `max e3w_1d = 300.98 m`), so `L` is
at a geometric ceiling built from more than one spacing.  Which ceiling, and
whether the bottom TKE boundary condition is the reason the last row takes a
dissipation add-back without an implicit counterpart, is **PLAUSIBLE, not
measured**: the reviewer's reading is that NEMO applies its bottom TKE
Dirichlet unconditionally while legoESM gates it on a `bottom_tke_bc` flag that
defaults to False and that this card never sets.  That is a code reading, and
it is the next round's first measurement.

## Why the reported error names the wrong thing

The abort message is `raw-mesh e3w_int must contain only finite values > 0`
(`eos.py:736-740`).  That guard is correct and it is not the owner:

* at step 47's END the live `e3w` computed from the prognostic state is
  perfectly healthy — stretch in `[0.9999877, 1.0000119]`, minimum `e3w`
  `10.1210 m`, zero non-positive cells;
* inside step 48, the trace of every live-`e3w` producer
  (`phase0_blocker_e3w_trace.log`) shows call sites 11-19 healthy with the
  same stretch, and only the LAST TWO calls of the step returning `NaN`;
* the `NaN` is in the **stretch**, i.e. in `eta`
  (`nemo_r3t_stretch`, `eos.py:863-923`, floors at `1e-6` and guards dry
  columns, so it cannot manufacture a non-finite value from a finite `eta`).

So the chain is: TKE runs away -> the eddy coefficients run away -> `T`, `S`
go infinite and `u`, `v` go NaN at step 47 -> `eta` inherits the NaN inside
step 48 -> the vertical-geometry guard fires.  **Attributing this to the
geometry, to the free surface, or to the barotropic solver would be wrong**;
each was checked and each is healthy at the last finite step.

## What this does and does not say

* **CONFIRMED**: the card aborts at step 48; TKE is the diverging field; the
  dynamics are healthy through step 46; the guard that reports it is
  downstream.
* **CONFIRMED**: the owning term is the explicit half of the TKE dissipation,
  acting as an `e^1.5` source with no implicit counterpart; the coefficient
  matches `0.5 * rn_ediss * dt / L` to five significant figures over five
  steps; the runaway is one corner column at the deepest interior interface.
* **RETRACTED**: the enhanced-vertical-diffusion convective switch as the
  trigger.  Two independent reviews refuted it -- the implied stratification at
  that cell is a strong pycnocline, not a convecting one, and the diffusivity
  ratio that was offered as evidence does not say what the earlier draft said
  it said.
* **PLAUSIBLE, NOT MEASURED**: that the missing implicit counterpart is the
  bottom TKE boundary condition, which legoESM gates behind a flag defaulting
  to off and which NEMO applies unconditionally.  Code reading only.
* **NOT ATTEMPTED**: no fix.  A change to a certified card's closure is a
  numerics change requiring both adversarial reviews and, almost certainly, a
  configuration decision that is not the agent's to make.  It is reported, not
  patched.
* **NEMO's own side is UNMEASURED, and probably need not be spent here.**  No
  NEMO GYRE year has been run in this campaign.  GYRE is NEMO's standard
  demonstration configuration and is routinely integrated for years, and
  NEMO's semi-implicit split gives `e_new/e_old -> 1/3` at large `e`, which
  analytically cannot produce this map.  So the one-sided conclusion stands --
  legoESM cannot reach a year on this card -- and the NEMO members are better
  spent after the closure is fixed than before.

## The operator script was dry-run, and it was broken

`run.sh` cannot be executed by the agent, so the part of it that can silently
refuse a CORRECT input was extracted and run by hand against the certified
namelist.  It refused, with `REFUSE: 197 changed rows`.

The bug: the namelist check compared the two files LINE BY LINE with a `zip`,
so the moment `nn_pert_seed` was inserted every subsequent line shifted by one
and the whole file read as changed.  The operator would have hit that on the
first member.  It also inserted the new row immediately after the
`&namusr_def` header, ahead of the group's own comment line.

Fixed: the check now compares PARSED ASSIGNMENTS (`key = value`, skipping
comment and group lines), so it is insensitive to position, and the row is
inserted just before the group's closing `/`.  Re-run against the certified
namelist it reports exactly

```
    nn_itend       10 -> 2160
    nn_stock       10 -> 180
    nn_write       10 -> 2160
    nn_pert_seed   (new) -> 1
```

and a plant that additionally flips `rn_Uv` from `2.0` to `4.0` exits
non-zero naming `rn_Uv`.  It also refuses if FEWER than all three rows change,
so a writer that silently failed to reach `nn_itend` cannot produce a ten-step
run wearing a year's name.

## Consequence for the round

| phase | status |
|---|---|
| preregistration | **DONE**, and revised after two independent reviews |
| harness + tests | **DONE**; `--self-check` green, 18 direct tests pass, three of five plants exercised |
| NEMO `run.sh` + MY_SRC patches | **WRITTEN**, not run.  Its PHASE-0 gate correctly refuses, because `phase0_floor.json` does not and cannot exist |
| PHASE 0 (legoESM ensemble) | **BLOCKED at step 48 of 2160** |
| PHASE 1 (NEMO ensemble) | not started |
| verdict, floor, maps | **UNMEASURED** |

Every preregistered expectation P1-P5 is therefore **UNMEASURED**, not held and
not refuted.  The 40-hour entering condition from the preregistration's section
0 (`T3D` rms `2.768e-3 K` at ten steps) stands, and remains the only
legoESM-vs-NEMO GYRE number beyond kt=2.

## Reproduce

```
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:\
packages/ice:packages/land:packages/ml:packages/tools:src \
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_fromrest.py \
  --census 50 --census-start 36
```

about 6 minutes on CPU, exit status 1.  The card requires
`transcendentals='libm'`, which the precision policy rejects on GPU, so CPU is
not a preference here.  Output preserved as
`phase0_blocker_census_committed.log`.

## Reviews of this round, and what they changed

Two independent fresh reviews ran on this diff and this receipt, on top of the
two that ran on the preregistration before any simulation.  Codex was occupied
on this branch's kt=2 ladder and GLM-5.2 was unavailable, so all four reviewers
were Claude instances with no shared context.

| finding | what changed |
|---|---|
| the owning term is determined by the receipt's own table: `dissl` is a rate, so `L = sqrt(tke)/dissl`, and `e(n+1)/e(n)^1.5` matches `0.5*rn_ediss*dt/L` | the receipt now IDENTIFIES the term instead of calling it UNMEASURED.  The arithmetic was re-derived here before adoption |
| the convective-switch trigger is refuted; the acceleration needs no threshold | that attribution RETRACTED |
| the `tke_avt` / `tke_avm` ratio paragraph is contradicted by the log it cites, and compares two independently-taken maxima | paragraph DELETED |
| "three independent probes" is one code path read three ways | corrected to "three readings of one path" |
| the committed census reduced min/max over the FINITE subset, so a half-infinite field printed a healthy range | fixed to raw min/max; a test pins it by source |
| the floor is a max over six within-model pairs while the gap was a single control pair -- an extreme of six against one draw, inflating the floor ~2x toward INDISTINGUISHABLE | the headline ratio now uses the max over the sixteen cross-model pairs, the same order statistic; the control-pair ratio is reported beside it |
| NEMO's restart fields were never checked for finiteness while legoESM's were | finiteness gate added to the NEMO reader |
| a non-finite ratio was reported as `UNMEASURED_ZERO_FLOOR` | distinct `UNMEASURED_NONFINITE` verdict |
| only `T3D` reached the exit status; salinity, SSH, PSI and QNET had verdicts nothing could see | every row's day-360 verdict is now printed |
| the "no sqrt(2)" test grepped one exact spelling, and the constant existed only to be absent | the constant is DELETED; the check is now a property (no root factor multiplies a floor) with its own non-vacuity probe |
| the time-level citation is off by 43 lines | it named the shipped `stprk3.F90`; the card compiles its MY_SRC copy.  Both are now cited |
| "15 direct tests, every plant fires" over-claims | corrected below |

**Plant coverage, stated honestly.**  Eighteen direct tests pass.  Three of the
five plants are exercised (`perturbation-zero`, `perturbation-relative`,
`operand-mismatch`), each shown to raise.  `floor-inflate` and `gap-zero` live
inside `score()` and are exercised only when member data exists, which it does
not while PHASE 0 is blocked.  They are UNEXERCISED, and that is a debt, not a
detail.

## Artifacts, SHA-256

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/`.

| file | sha256 |
|---|---|
| `phase0_blocker_tke_census.log` | `6c62febdec708ff092e39052b3f9b497c653edb54ae9940568f0f52588181349` |
| `phase0_blocker_e3w_trace.log` | `c7d35b0fc893997a8665dee8aaf75e06749b3ad72816e68db773c6fa0aa91f02` |
| `phase0_blocker_census.py` | `1bd66258160d21f7c1b064e4f8c975d9032bf01d67b210253e4b85aacda8b626` |
| `phase0_blocker_census_committed.log` | `d29bc20d4cae9a451ecba38ac89741f4e4fc1f594f76542637fe52765242a0a9` |
| `phase0_blocker_census_argmax.log` | `8488369f94b3ce3ee8cf7e8f541b88b90ed97f7a06ff88976aad431395e7da37` |
| `lego_seed0.log` | `ff005d91ef3659cb359c3404ce560afcb4d673ea48c6d14c79ef9916ad4adad3` |

`phase0_blocker_census.py` is the throwaway that FOUND this; it is superseded
by the harness's `--census` mode and kept only so the first reading is
reproducible.  `lego_seed0.log` is the year harness's own first failure, i.e.
the blocker as it appeared in production rather than in a probe.

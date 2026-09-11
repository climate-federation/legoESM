# GYRE from-rest YEAR, round 1: PHASE 0 IS BLOCKED — the TKE closure diverges at day 6

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_year_fromrest.md`
(written and independently reviewed twice BEFORE any member ran).
Artifacts: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/`,
SHA-256 in `phase0_blocker_artifacts.sha256`.

## VERDICT

**The GYRE year cannot be run.**  legoESM's certified GYRE card
(`GYRE-zco`, `build_gyre_zco_card`) aborts at **step 48 of 2160**, and the
reason is not the abort's message.  The model's **prognostic TKE closure
diverges super-exponentially from about step 38**, reaches `3.5e40 m2/s2` at
step 47, drives the eddy viscosity to `5.6e21 m2/s`, and destroys the tracer
and momentum fields.  The sea-surface height inherits the NaN, and the
vertical-geometry guard in `eos.py:736-740` is what finally reports it.

This is a **NEW, previously unmeasurable finding**: the kt=1..10 ladder that
certified this card stops 38 steps before the instability starts.  "Never
measured beyond ten steps" turns out to mean "cannot currently be measured
beyond 47".

CONFIRMED.  Three independent probes, three different instruments, all stop at
the same step: the year harness itself (all four members), a bare stepping
walk, and the geometry trace.

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
AROUND between steps 30 and 36 and runs away, ACCELERATING: about `1.2x` per
step over steps 30-36, `2.7x` over 36-38, `8x` over 38-41, `1.5e5` over 41-46.
A constant-rate instability does not accelerate; a threshold crossing followed
by a positive feedback does.

**One more structural clue, recorded without interpretation.**  `tke_avt`
tracks `tke_avm/10` exactly from step 1 to step 44 and then becomes EQUAL to it
at steps 45-46 (`6.495e+09` both).  Whatever sets that ratio stops setting it
at step 45; which branch or limiter that is, is UNMEASURED.

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
* **UNMEASURED**: which term of the closure owns it.  The certified namelist
  runs `ln_zdftke = .true.` with `nn_etau = 0`, plus `ln_zdfevd = .true.`,
  `rn_evd = 100.`, `nn_evdm = 1`.  The step-38 turnaround is consistent with a
  threshold process, and the enhanced-vertical-diffusion switch is exactly
  such a process, but nothing here demonstrates that it is the trigger.  The
  next round's job is to isolate the term, given NEMO's own inputs, in the
  manner Rule 12 requires.
* **NOT ATTEMPTED**: no fix.  A change to a certified card's closure is a
  numerics change requiring both adversarial reviews and, almost certainly, a
  configuration decision that is not the agent's to make.  It is reported, not
  patched.
* **NEMO's own side is UNMEASURED.**  No NEMO GYRE year has been run in this
  campaign.  GYRE is NEMO's standard demonstration configuration and is
  routinely integrated for years, so the one-sided conclusion is that legoESM
  cannot reach a year on this card; whether NEMO's TKE behaves differently on
  the same card is exactly what `run.sh` would measure, and it is not measured
  here.

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
| harness + tests | **DONE**; `--self-check` green, 15 direct tests pass, every plant fires |
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

## Artifacts, SHA-256

All under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/`.

| file | sha256 |
|---|---|
| `phase0_blocker_tke_census.log` | `6c62febdec708ff092e39052b3f9b497c653edb54ae9940568f0f52588181349` |
| `phase0_blocker_e3w_trace.log` | `c7d35b0fc893997a8665dee8aaf75e06749b3ad72816e68db773c6fa0aa91f02` |
| `phase0_blocker_census.py` | `1bd66258160d21f7c1b064e4f8c975d9032bf01d67b210253e4b85aacda8b626` |
| `phase0_blocker_census_committed.log` | `d29bc20d4cae9a451ecba38ac89741f4e4fc1f594f76542637fe52765242a0a9` |
| `lego_seed0.log` | `ff005d91ef3659cb359c3404ce560afcb4d673ea48c6d14c79ef9916ad4adad3` |

`phase0_blocker_census.py` is the throwaway that FOUND this; it is superseded
by the harness's `--census` mode and kept only so the first reading is
reproducible.  `lego_seed0.log` is the year harness's own first failure, i.e.
the blocker as it appeared in production rather than in a probe.

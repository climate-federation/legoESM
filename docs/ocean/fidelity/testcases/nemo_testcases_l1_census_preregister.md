# OVERFLOW-zps `final_water_mass_census`: preregistration for the k=24 ownership arm

Frozen before any arm.  The instrument (`nemo_testcase_census_map_probe.py`)
was committed before its own numbers, at `7e2ff4e91` (the `map` command) and
`45dd9ea` -> see the receipt for the exact SHAs (the `faces` and `variance`
commands); the probe refuses to write a map until it has reproduced the
committed scorer's candidate / floor / spread for this row to the last digit.
Every number is fp64 (`PrecisionPolicy.fp64()` set BEFORE the card is built,
plus `JAX_ENABLE_X64=1`), CPU.

## 0. What the statistic actually is (read from the scorer, not from its name)

`nemo_testcase_full_statistics.py:826-845` and `:1133-1143`:

* REGION -- wet cells whose column bathymetry satisfies `500 m < H < 2000 m`
  (`slope = (bathy > 500.0) & (bathy < 2000.0)`), i.e. the continental SLOPE.
  Not the shelf, not the abyssal plain.
* TIME -- the FINAL state alone, 6120 steps = 61200 s.  No earlier time enters
  the row.
* QUANTITY -- live-partial-cell volume fractions of that region in
  `cold [10,12)`, `mixed [12,18)`, `ambient [18,20]`, normalised to sum 1.
* REDUCTION -- `_curve_distance`, `max_i |left_i - right_i|` over the three
  classes.  (The sibling histogram row uses total variation; this one does
  not.)
* Candidate `L64` vs `N2`; precision floor `L32` vs `L64`; scheme spread `N4`
  vs `N2`, where `N4` is NEMO with `nn_fct_h=4, nn_fct_v=4` and is otherwise
  the same executable and namelist (the scorer's own namelist diff shows only
  those two assignments and the experiment name change).

Standing figures after the 3-D umask round: candidate `0.009437984095819751`,
floor `0.0005206303447481894`, spread `0.003362090947063974` -- the one row
still `OUTSIDE`, by 2.81x its spread.

## 1. The claim under test

> The `k=24` shoreward injection family at the shelf break -- the per-step `u`
> difference measured at gate faces 20/21 level 24, untouched (exactly `1.00x`)
> by the 3-D umask fix -- owns the `final_water_mass_census` gap.

## 2. Refute condition, stated BEFORE the arm

The claim is REFUTED, and no arm is run, if EITHER holds:

* **(R1) the calibration pair kills it.**  `N4` differs from `N2` in the
  TRACER advection order only -- no momentum-operator difference of any kind.
  If `N4` reaches a pointwise decorrelation from `N2` at least as large as
  `L64` does (`temperature_linf` and `instantaneous_u_linf`) while moving the
  census and the domain tracer variance by far less than `L64` does, then
  decorrelation of the observed size does not produce the observed census gap,
  and a momentum injection whose only channel is decorrelation cannot own it.
* **(R2) the per-step perturbation bound kills it.**  The `L32` floor arm
  injects a relative perturbation of order `eps(f32) = 1.19e-7` into every wet
  cell at every one of 6120 steps.  If that perturbation -- larger per step
  than the `k=24` injection and spread over the whole domain rather than two
  faces -- moves the census by less than a third of the gap, the `k=24`
  family's chaotic channel is bounded far below the gap.

## 3. Predictions, if the arm is run

| # | prediction |
|---|---|
| P1 | the `k=24` injection at gate faces 20/21 (kt=2 `7.064252e-12`, kt=3 `4.1754e-09`, both at exactly `1.00x` across the umask arm) drops by `>= 2x` |
| P2 | the kt=3..60 trajectory `u` rows improve, with kt=3 `4.181444e-09` and kt=60 `3.238532e-05` each dropping `>= 1.5x`, and no `T`, `S` or `SSH` row worsening by more than `1.0001x` |
| P3 | `final_water_mass_census` `0.009437984` moves to `<= 0.003362091` (its NEMO scheme spread), i.e. the row leaves `OUTSIDE` |
| P4 | the volume-weighted domain tracer-variance ratio `L64/N2` at 6120 steps moves from `1.1316` to within `[0.994, 1.006]` (the `N4/N2` scheme spread) |
| P5 | LOCK_EXCHANGE-zco stays BIT-IDENTICAL on every trajectory row (its flat bottom has no `k=24` staircase family) |
| P6 | the new unit test FAILS on the reverted code |

## 4. The arm, if run

One variable, a private `_NEMOWSRK3TestHooks` field (harness only; NEMO has no
such switch), holding the eval protocol byte-identical to
`phantom_velocity/after` -- same scorer, same registered NEMO `N2` and `N4`
runs, same 6120-step duration, same fp64/fp32 pair.

## 5. What is NOT in scope, and why

* Changing `nn_fct_h`/`nn_fct_v`, the BBL options, or `ln_zad_Aimp` on either
  side.  Those are scheme SELECTIONS; they are ASK items, not arms.
* Adding any limiter, filter or damping legoESM does not already share with
  NEMO (Rule 9).
* Re-running NEMO with different diagnostics.  The certified executable,
  namelist and restart hashes gate every row in the scorer, and a re-run to
  obtain a variance time series would invalidate them.

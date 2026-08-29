# Preregistration: ZDF row-30 `zbu` operand peel

Date: 2026-08-29. CPU-only matched day-180 state. Frozen before measurement.
This continues immediately after the verified `zau/zav` reciprocal fix and
uses existing dumps first; it requests no NEMO rebuild or run.

## Oracle order and time levels

The running standard-slope branch is `stpmlf.F90:219,234`:
`eos(ts,Nbb,rhd)` followed by `ldf_slp(kstp,rhd,rn2b,Nbb,Nnn)`. Thus `prd`
is BEFORE-level S-EOS density and `pn2` is the step-entry BEFORE `rn2b`
computed at `stpmlf.F90:205-207` on `Nnn` geometry. NEMO then writes:

```fortran
! ldfslp.F90:226-230
zdzr = zm1_g * (prd + 1._wp) * (pn2(jk) + pn2(jk+1)) \
     * (1._wp - 0.5_wp*tmask(jk+1))
! ldfslp.F90:244
zbu = 0.5_wp * (zdzr(i,j) + zdzr(i+1,j))
```

The frozen order is: `pn2(jk)` -> `pn2(jk+1)` -> their ordered sum ->
`prd+1` -> mask factor -> `zm1_g=-1/grav` product -> east-face pair sum ->
literal `0.5_wp` multiply. No later limiter or slope is inspected before all
eight stages pass.

## Existing operands, bars, and adjudication

The probe SHA-binds `tke_dump_rn2b.bin`, `eiv_dump_prd_arg.bin`,
`eiv_dump_zbu_pre.bin`, the row-30 binary/restart/source receipts, focus map,
clean repository HEAD, CPU backend, and fp64 policy. The first three arrays
are independently written NEMO operands/output; no value is inferred from the
final slope.

Each array stage uses the normalized whole-column bar `1.0e-15`, reports its
own wet-column denominator, and scores all four southern focus columns.
`zbu` face stages must be `0/9758`. Intermediate T/W stages must have zero
failed wet columns. The first nonzero stage is `DIVERGED`; later stages are
not dispositioned.

A mandatory discrimination compares two legoESM paths before assigning a
physics owner:

1. the historical `ldf_slp_per_element.build_state()` reconstruction used by
   the row-30 scorer; and
2. the actual `LatLonCGridOceanModel._tke_step_entry_n2_bundle()` production
   path, including its below-seafloor extrapolation and carried `rn2b` slot.

If the production bundle and all eight literal stages pass but the historical
capture fails, disposition is `DIVERGED-HARNESS`: repair the scorer/helper,
not production physics, then rerun row 30. If production `pn2(jk)` is first
red, it owns the production interval and the fix must be selectable, faithful
by default on only the two DINO NEMO cards, with every other card byte-exact.
Any later first red owns only that exact arithmetic interval.

Controls must fail: bar-scale perturbation, zonal roll, wet NaN, one-ULP exact
identity, swapping `jk/jk+1`, replacing `rn2b` by `rn2`, and using current
instead of BEFORE `prd`. Missing/changed dump SHA, wrong lane, backend, dtype,
focus map, or dirty tracked tree is fatal.

Rows 31--32 and climate remain ordered-blocked until row 30 is fully VERIFIED
or admissibly waived; there is no focus-only exception.

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

## Frozen continuation: post-bound `zbu` limiter

Date: 2026-08-29. Frozen before the post-bound substitution measurement. The
eight-stage peel above has repaired the historical scorer and the complete
row-30 scorer now verifies the raw `zbu` at `ldfslp.F90:244` (`0/9758`) but
first diverges after the stability bounds at `ldfslp.F90:248`
(`5496/9758`, all four focus columns). This continuation uses only the existing
`zau`, `zbu_pre`, and `zbu_post` dumps plus the SHA-bound NOW restart and raw
mesh; no oracle rebuild is authorized before this ladder is exhausted.

The running source is evaluated literally, in this order:

```fortran
! ldfslp.F90:247-248
zbu = MIN( zbu, -z1_slpmax * ABS( zau ),   &
     &          -7.e+3_wp / e3u(ji,jj,jk,Kmm) * ABS( zau ) )
```

The registered stages are: captured `zbu_pre`; captured `zau`; stored
`z1_slpmax`; raw-mesh `e3u_0`; `r3u(Kmm)` reconstructed from the NOW SSH using
`domqco.F90:166-169`; live `e3u(Kmm)=e3u_0*(1+r3u*umask)` from
`domzgr_substitute.h90:129`; slope cap; live-metric cap; inner ordered `MIN`;
outer ordered `MIN`; captured `zbu_post`. Each stage uses the whole wet-U
column bar `1.0e-15` and reports the four southern focus columns.

The discrimination is frozen as follows. If substituting the literal live
`e3u(Kmm)` makes the final `MIN` `0/9758`, while the production static partial-
cell face thickness reproduces the measured `5496/9758`, the owner is the
`e3u(Kmm)` operand at `ldfslp.F90:248`. The production repair is a selectable
live-QCO face-thickness construction, faithful by default only on the two DINO
NEMO cards; the historical static face metric remains the global default and
must be byte-identical on every other card. If the live substitution is still
red, the first red literal stage owns the row and no later stage is scored.

Controls must fail: replacing live `e3u` by static partial-cell thickness,
using BEFORE rather than NOW SSH in `r3u`, a zonal face roll, a one-ULP exact-
identity perturbation, wet NaN, and bar-scale perturbation. The raw-mesh,
restart, dump, source, binary, parent-artifact, focus-map, clean-HEAD, CPU, and
fp64 gates are fatal. Rows 31--32 and climate remain blocked until the row is
closed under the original no-focus-exception registration.

# DINO twin-vs-NEMO day-360 state maps

Date: 2026-08-30. Session: `01a04e34-d1fb-73e0-b25a-177641f0a246`.

The user-requested map comparator is committed at
`scripts/validate/ocean_fidelity/dino_1226/twin_nemo_ts_maps.py`. The admitted
instrument commit is `c0ce4389f0aaf704a41b0e263cd9ad49de247524`; its source
SHA-256 is
`eb6fe57efc77673d851e77858cbf141ec499a3dbe6d21011c4cf5fe526d1126a`.
Four direct oracle-independent unit tests pass.

## Run and provenance

The clean CPU invocation was:

```bash
python scripts/validate/ocean_fidelity/dino_1226/twin_nemo_ts_maps.py \
  --arm-npz /tmp/dino-climate-rebattery-01a04e34/arms/climate_a.npz \
  --day 360 --nemo-kt 17280 \
  --output-dir /tmp/dino-twin-nemo-ts-maps-day360-c0ce4389
```

The JSON sidecar is
`/tmp/dino-twin-nemo-ts-maps-day360-c0ce4389/twin_nemo_ts_maps_day360_kt17280.json`,
SHA-256
`9a0bc012ad8cff46882d4e97601b273839ebc04b0d795433ced1218e8578cb83`.
It records the clean comparator git SHA, arm producer/session, fp64 scored
dtypes, arm/mesh paths and SHA-256 values, and all 16 NEMO restart-tile paths
and SHA-256 values. The arm SHA-256 is
`e598d01e3baf69c1af48df3b3b8df66150052820bc13a782a5b8be413c16cb20`;
the mesh SHA-256 is
`3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622`.

## Alignment and wet population

The arm and rebuilt restart are asserted full `(j,i)=(199,52)` arrays. The
map-only comparison explicitly takes `[1:-1,1:-1]`, producing `(197,50)`.
This is a diagnostic interior crop, not a claim that DINO 5.x restart files
carry file halos; `packages/ocean/legoesm/ocean/fidelity/nemo_io.py:3-18`
defines those files as canonical halo-free arrays and the NEMO-to-legoESM
axis move as `(k,j,i) -> (j,i,k)`. Full-frame gates remain authoritative for
boundary-ring claims.

| population | wet | total | fraction |
|---|---:|---:|---:|
| surface | 9,850 | 9,850 | 1.000000 |
| 3-D T mask | 339,744 | 354,600 | 0.958105 |

The arm land mask is exactly equal to the NEMO surface mask on the scored
interior. The planted control moves one wet cell by `0.25`, changing max error
from zero to `0.25` and RMS from zero to `0.1767767`; an independent dry-cell
plant leaves both statistics unchanged.

## Day-360 statistics

All rows compare fp64 arm values against fp64 rebuilt NEMO values on the wet
population.

| field | RMS | maximum absolute | mean legoESM - NEMO |
|---|---:|---:|---:|
| SSH [m] | `1.039384226e-4` | `9.853308317e-4` | `2.031883745e-5` |
| SST [degC] | `5.021561930e-3` | `5.426575943e-2` | `-1.007343515e-3` |
| SSS [psu] | `4.125609644e-4` | `6.158186042e-3` | `5.816970715e-5` |
| 3-D T [degC] | `3.436364194e-3` | `2.521433832e-1` | `9.259068254e-5` |
| 3-D S [psu] | `3.021247471e-4` | `3.477877315e-2` | `3.942240649e-5` |

The figures are `surface_day360.png`, SHA-256
`31ba37c8e5ddc1b03d31f4b70c27af3c86022fc81eca749547315fabd1ff8dfb`,
and `subsurface_day360.png`, SHA-256
`12fd05863896f32f9d54956364c1547b9a43b98dfeace7dfc5296af6465b5dd8`.

## Localized 3-D temperature maximum

The maximum is at zero-based map-interior `(j,i,k)=(107,5,4)`, corresponding
to full-frame `(108,6,4)`, latitude `8.963215662388489 deg N`, longitude
`5.5 deg E`, and local T-point depth `48.57634758945642 m`. legoESM is
`25.777104660254285 degC`; NEMO is `26.02924804347644 degC`; the signed
difference is `-0.2521433832221547 degC`. Thus the reported `0.25 degC` cells
are a localized near-surface northern-tropical feature in this map frame,
not an unidentified global maximum.

This comparator is descriptive and introduces no new acceptance bar or
climate-owner attribution.

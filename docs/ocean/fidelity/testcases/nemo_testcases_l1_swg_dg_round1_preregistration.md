# SWG double-gyre round 1 preregistration

Date: 2026-10-12. Incoming tip: `1d7fae76681ecd6e551bebca6892a1953d2e9c79`.
No SWG_DG NEMO or legoESM fidelity output has been opened or produced in this
round. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/`.

## Retraction and case boundary

The earlier 4,000 km smoke run is withdrawn as a double-gyre run. Its 32
`mesh_mask` rank files showed a 320 by 320 wet basin spanning about 9--2003
km because shipped `usrdef_zgr.F90` fixes both upper mask limits at 2,010 km.
The old three-year run was cancelled. No output from it may be used.

The replacement oracle case is `tests/SWG_DG`: shipped `tests/SWG` plus
only the committed, SHA-pinned two-line `usrdef_zgr_domsiz.patch`. It replaces
the two fixed upper limits by `rn_domsiz + 10000._wp`. This changes one piece
of physics: basin extent. The wind formula is unchanged.

## Resolved-deck diff

Only differing scientific and clock fields are listed.

| field | shipped SWG | Maddison SWG_DG stage deck | Maddison SWG_DG month deck |
|---|---:|---:|---:|
| source case | `SWG` | `SWG_DG` | `SWG_DG` |
| `nn_SWG` | 4 | 16 | 4 |
| spacing | 25 km | 6.25 km | 25 km |
| `rn_domsiz` | 2,000 km | 4,000 km | 4,000 km |
| wet basin | 80×80 | 640×640 | 160×160 |
| `rn_tau` | 0.20 N m-2 | 0.10 N m-2 | 0.10 N m-2 |
| `rn_modified_grav` | 0.02 m s-2 | 9.80665 m s-2 | 9.80665 m s-2 |
| `rn_Uv`, `rn_Lv` | 2 m s-1, 500 m | 2 m s-1, 10 m | 2 m s-1, 10 m |
| `ahm=rn_Uv*rn_Lv/2` | 500 m2 s-1 | 10 m2 s-1 | 10 m2 s-1 |
| `rn_Dt` | 3,600 s | 50 s | 200 s |
| acquisition steps | 10 / 720 | 10 | 12,960 |
| daily cadence | 24 steps | 1,728 steps | 432 steps |

`rn_dx=100000`, `rn_rfr=1e-7`, `rn_beta=2e-11`,
`rn_f0=0.7e-4`, the single 500 m layer, zero rotation, and free-slip agree.
The compiled wind divisor remains exactly 2,000 km. Thus the 4,000 km basin
spans one full cosine period: easterlies at both walls and westerlies at the
middle.

## Parameterised-card prediction

One `build_swg_card(namelist=...)` implementation will consume the resolved
namelist. Resolution, nominal spacing, basin size, wind amplitude, modified
gravity, the viscosity pair, layer drag, beta, f0, timestep, and run length
must reach the executed card. Any unsupported resolved switch must raise.

CONFIRM only if:

1. The no-argument shipped card digest stays `f9d46bfd08f23702`.
2. Fresh shipped ladders remain bit-identical: given-entry 338/338, chained
   338/338, independent 50/50, carried state 29/29, and month 690/690.
3. The parameterised 4,000 km / `nn_SWG=16` geometry has 644 by 644 stored
   points and exactly 640 by 640 wet points.
4. The analytic wind uses the fixed 2,000 km divisor, not `rn_domsiz`.

Any digest movement, shipped unequal cell, wrong wet count, basin-scaled wind,
or accepted unsupported namelist value REFUTES the implementation.

## New oracle ladders

The ten-step stage ladder uses the run resolution: `nn_SWG=16`, 6.25 km,
640 by 640 wet cells, `rn_Dt=50 s`. Each of the 30 stage frames stores
thirteen float64 644 by 644 arrays: about 41.13 MiB per frame and 1.205 GiB
total before mesh, restart, and ordinary-output files.

The thirty-day daily ladder uses the same physical deck at `nn_SWG=4`,
25 km, 160 by 160 wet cells, `rn_Dt=200 s`: 432 steps per day and 12,960
steps. This coarser month is the operator's protocol choice because the
640 by 640 single-rank reference plus instrumented acquisition is estimated
at about ten hours. The fine-grid month remains **OPEN**. Its 30 stage-one
frames are about 2.67 MiB each, 0.078 GiB total.

For both ladders the prediction is AT-BAR and bit-identical at every scored
row, with zero unequal wet cells. The existing nonzero recorded-input one-ULP
plants must bind and produce DEBT/nonzero exit. Missing evidence is
`ACQUISITION_NEEDED`, not a failed physics prediction.

If a first difference appears, the first suspects to report without
attribution are: explicit `9.80665` versus NEMO's `phycst` gravity
association; the 4,000 km basin combined with the unchanged 2,000 km wind
period; and halo handling across 640 wet cells. No stabilizer, clipping,
alternate scheme, or relaxed bar may be introduced.

## Operator commands and evidence

```text
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run.sh --deck maddison --run
  -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1/
bash scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run_30d.sh --deck maddison --run
  -> /data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round1/oracle_swg_dg_r1_30d/
```

No acquisition will be run by this round's coding agent.

## Oracle citations

The compiled domain size and spacing derive from `rn_domsiz`, `rn_dx`,
`nn_SWG`, and the ghost-cell count at
`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:154-166`. The shipped mask's
fixed 2,010 km upper limits, U/V-point inequalities, and lonely-corner rule
are at `SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:206-235`. The one-layer
500 m vertical ladder is at
`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:158-168`. The unchanged stress
uses `rn_tau` and the fixed 2,000 km cosine divisor at
`SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_sbc.f90:99-106`.

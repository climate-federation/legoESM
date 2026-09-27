# Round 186 receipt — pre-day-180 process ranking

**Status: STOPPED_FOR_RECORD.**  The admitted steps 1--1080 process budget
closes exactly and ranks penetrative shortwave first at
`+6.755462019716337e-05 K` of the fixed day-240 error direction.  Its largest
cancelling partner is the surface boundary at
`-2.5093847717433246e-05 K`.  The existing compatible shortwave record stores
only the completed smooth-step increment, so no internal shortwave statement
is claimed.  A passive developed-step operand/replay acquisition is requested.
No model physics, configuration, carried state, default, or certified
trajectory changed.

Preregistration:
`PREREG_nemo_testcases_l2_gyre_round186.md`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round186/`.

## Record admission and compiled order

The Round-185 compiled writer opens one frame for each stage-3 step 1--1080
and records entry temperature and all three QCO ratios at
`GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-831`.
It then records the advection and surface-boundary writes in compiled order at
`GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`, and
shortwave, lateral diffusion, and the completed vertical solve at
`GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-970`.

The admission defect was exactly the hard-coded old endpoint.  The repaired
gate admitted 1,080 contiguous 1,415,300-byte frames, totaling
`1,528,524,000` bytes, with zero chain mismatches.  Day-30 and day-180 restart
SHA-256 values remain respectively
`853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6`
and `6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976`.
The stamp, truncation, SBC-ULP, SBC-effect, and trajectory-ULP plants all
printed `STATUS PLANT-FIRED` and exited nonzero.

The production-JIT legoESM trace likewise contains 1,080 frames.  Its observer
moved zero carried-state bytes, and its generated day-180 T/S/u/v/SSH snapshot
is bit-identical to the immutable Round-183 arm.  The row-ULP and
production-effect controls fired; the latter moved its intended one boundary
cell while the returned ordinary state remained unchanged.

## Day-240 magnitude ranking

All signed carries are projections onto the independent fixed Round-183
day-240 temperature error.  The seven non-incoming rows sum exactly to the
measured day-180 carry `3.879177135308106e-05 K`; the reconstructed error field
has maximum residual `0`.  Incoming error is exactly zero.

| rank | owner | signed carry (K) | component RMS (K) | strongest ten-day block / carry | first bitwise-nonzero block | largest depth / longitude / latitude |
|---:|---|---:|---:|---|---|---|
| 1 | shortwave | `+6.755462019716337e-05` | `1.659976115199776e-04` | 170--180 / `+6.152312286933793e-06` | 0--10 | 0--100 m / west / south <=37.2 N |
| 2 | surface boundary | `-2.5093847717433246e-05` | `1.0158252945517908e-04` | 170--180 / `-3.6843403354711616e-06` | 0--10 | 0--100 m / west / south <=37.2 N |
| 3 | vertical diffusion | `-9.363809435702184e-06` | `1.9273599533939755e-04` | 120--130 / `-2.082027175040126e-06` | 0--10 | 0--100 m / east / north >37.2 N |
| 4 | lateral diffusion | `+5.011122476554521e-06` | `1.3823531131687154e-04` | 170--180 / `+4.347738367507402e-06` | 0--10 | 0--100 m / east / north >37.2 N |
| 5 | advection | `+6.889995532588958e-07` | `8.492015056550564e-05` | 110--120 / `-5.028195203284445e-07` | 0--10 | 100--1000 m / west / south <=37.2 N |
| 6 | geometry | `-5.313720434327574e-09` | `8.632870500747533e-09` | 10--20 / `-9.022460388411128e-10` | 0--10 | 0--100 m / west / south <=37.2 N |
| 7 | rounding closure | `-3.259713155872930e-16` | `5.950217847049836e-14` | 110--120 / `-1.633418403071361e-16` | 0--10 | 100--1000 m / interior / north >37.2 N |
| 8 | incoming | `0` | `0` | before day 0 | before day 0 | none |

The preregistered shortwave-first and surface-cancellation predictions are
confirmed.  The original `<=4e-15 K` closure prediction is **REFUTED** for the
initial scorer: subtracting two independently accumulated model closures gave
`6.039613253960852e-14 K`.  That is bookkeeping rounding, not an omitted
physical row.  The committed paired explicit-rounding component evaluates the
same interval difference directly and closes with zero field residual; the
separate NEMO and legoESM accumulated closures remain reported as
`2.541358032111431e-14` and `5.395373089197061e-14 K`.

## Why the statement walk stops

The existing compatible kt=1 shortwave record brackets `CALL tra_qsr` and
stores only surface `qsr` plus the completed `Krhs` increment at
`GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-945`.
The executed two-band routine forms the coefficients and surface attenuation,
then evaluates live-depth attenuation, the live `ze3t` divisor, and the
near-surface/deep updates at
`GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/traqsr.f90:615-645`.
None of those developed-state operands or intermediate statement boundaries
is in the existing record.  Promoting the completed process row would confuse
inherited state error with a transcription error, so the first non-bit
shortwave statement remains **UNMEASURED**.

The new operator-run acquisition uses a new target, copies the admitted year
card file by file, adds a post-call replay, and records step 1080 in the
strongest days-170--180 interval.  Its source patches pass
`gfortran -fsyntax-only`.  Admission requires byte-identical day-30/day-180
restarts, bit-identical replay versus the actual increment, a stamped clean
producer commit, and a firing actual-increment ULP plant.  It does not change
NEMO production source in place.

## Certified rows, scope, review, and tests

No executable legoESM statement changed.  Certified values therefore remain:

- kt2 T/S/U/V: `1.4210854715202004e-14`, `2.1316282072803006e-14`,
  `8.326672684688674e-17`, `9.714451465470120e-17`;
- kt3 T/S: `4.9403105251144552e-07`, `4.0085410546453204e-08`;
- day-30/day-240/day-360 T3D RMS: `2.3276772050683987e-06`,
  `6.5861718814795174e-05`, `2.6709923853294689e-03 K`;
- first over bar: kt3.

GYRE, generic NEMO-GYRE, DINO, LOCK_EXCHANGE, OVERFLOW, and ORCA2 execute no
changed model statement.  ORCA2 transfer of this ranking remains
**UNMEASURED-WITH-SPEC**: acquire its native process rows, masks, independent
production trace, and endpoint projection before transferring ownership.

The required separate read-only Codex review could not initialize in this
sandbox.  Its complete verdict text is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only
> file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)
>
> codex_exit_code=1

This is **independent review unavailable in-sandbox**, not a SHIP verdict.

The first focused instrument run reported **`1 failed, 53 passed in 34.37s`**;
the sole failure was the pre-existing Round-179 source-layout test observing
the deliberately dirty uncommitted instrument tree and exiting at its clean
tree guard.  The final committed-tree rerun and citation results are reported
below after completion.

## OPEN — round 187

Run and admit
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round186_qsr_walk/run.sh`.
Using its step-1080 live operands and bit-exact post-call replay, walk
`qsr_2BD` in compiled order under production JIT and eager execution.  Name
the first non-bit statement or continue upstream to its first non-bit operand.
Only a one-variable NEMO-source-exact candidate may enter the full
Decision-43/45/55/59 trajectory and card gates.

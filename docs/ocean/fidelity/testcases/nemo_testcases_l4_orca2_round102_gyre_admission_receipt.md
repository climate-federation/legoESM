# ORCA2 round 102 — merged-tree GYRE admission

**Date:** 2026-10-02  
**Base:** `87279760b351e7a95284b7f47631ae68a955d3a1`  
**Preregistration:** `a7ccf2abe0307337077ebd026d722091cc02c4b2`  
**Year measurement:** `a7ccf2abe0307337077ebd026d722091cc02c4b2`  
**Disposition:** **LANDED — round-100 merge admitted under Decisions 43/59/AW**

## Claim boundary

This round adjudicates the shared stage-one tracer-thickness-ratio statement
already named in round 101. NEMO forms the nonlinear after-level ratios and
then performs the HYB interpolation in
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`.
No production model statement, card selector, deck, carried state, threshold,
stabilizer, sea-ice selector, or ORCA2 `unmeasured_features` entry changed.

The GYRE numbers are **independent** from-rest measurements. ORCA2's rung-7
read-out is **given NEMO's recorded entry**. These claim classes are not mixed.

## GYRE landing gate — independent

The fresh CPU/fp64 seed-0 member completed 2,160 steps and 360 daily
snapshots. All 1,800 saved field arrays are bit-identical to round 100's
merged member. R102-P1 is **CONFIRMED**. The current ten-step gate also
reproduces the live-source certificate: 70/70 rows, zero moved rows, no kt=1
AT-BAR loss, first-over-bar kt=3 unchanged, and byte-identical residual
archive SHA-256
`377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.

Every registered year row is below. Negative delta moves toward NEMO.

| day | certified T rms (K) | merged T rms (K) | delta (K) | floor units | direction |
|---:|---:|---:|---:|---:|---|
| 30 | `2.3432510206121264e-06` | `2.3432465132112266e-06` | `-4.507400899784753e-12` | `-0.0225370` | toward |
| 60 | `1.4793247420315405e-05` | `1.4793247973304582e-05` | `+5.529891763955425e-13` | `+0.00276495` | away, admitted |
| 90 | `1.6332637650962138e-05` | `1.633271203963844e-05` | `+7.438867630335435e-11` | `+0.371943` | away, admitted |
| 120 | `1.0965898339728307e-04` | `1.0965907352116351e-04` | `+9.012388044508067e-11` | `+0.450619` | away, admitted |
| 180 | `6.115333823687429e-05` | `6.115335539300055e-05` | `+1.715612626463671e-11` | `+0.0857806` | away, admitted |
| 240 | `6.581707093530567e-05` | `6.58170609494473e-05` | `-9.985858372293065e-12` | `-0.0499293` | toward |
| 300 | `5.466049869672802e-05` | `5.466049845187051e-05` | `-2.4485750612420615e-13` | `-0.00122429` | toward |
| 360 | `5.407735418221895e-05` | `5.4077419367442036e-05` | `+6.518522308436797e-11` | `+0.325926` | away, admitted |

All away rows are below Decision 59's strict ten-floor-unit (`2.0e-9 K`)
allowance. The mechanical gate reports PASS; its planted 20-floor-unit day-240
regression reports `STATUS PLANT-FIRED`. R102-P2 and R102-P3 are **CONFIRMED**.

The merged member's registered snapshot SHA-256 values are day 30
`4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`,
day 240 `8b9cd60475626373d9a2fec0baa508f06c1f91aed4c3d7a24fc5d8b3f5878c8a`,
and day 360
`e3e0a068346c7866f0a318bb32141f2fb326cc05c158a1c95bc39b091f585e25`.
The GYRE lane should adopt the merged column above and those digests when it
imports the shared statement.

## Rung-7 entry refusal — given NEMO's entry

The pre-existing refusal consisted of exactly 77,662 T cells, all `+0.0` in
the card and `-0.0` in the record; zero cells differed numerically. NEMO masks
the z/zps initial T and S by multiplication at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:308-309`, so a
negative dry input preserves a negative zero. The gate now classifies only
temperature zero signs at this card-input precondition; S/u/v/ssh and every
nonzero T value remain strict-bit comparisons, as do all executed trajectory
rows. A nextafter nonzero plant and an S signed-zero plant both refuse.

The rung-7 step-1 gate now reaches `LADDER_MEASURED`, registers the 77,662 dry
T zero signs, and leaves the first numerical non-bit boundary at kt=1 stage-1
T. R102-P4 is **CONFIRMED**. The full ten-step rerun was started but did not
produce a new terminal artifact before the round cutoff; no ten-step movement
claim is made from it.

## Validation

- Decision-43/59 gate tests: 19 passed.
- ORCA2 phase-1 gate tests: 19 passed.
- GYRE trajectory: 70 rows unchanged; residual artifacts array-identical.
- Fresh year: 360/360 snapshot files and 1,800/1,800 arrays reproduce round 100.
- Separate read-only Codex review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).

## OPEN

1. R102-P5 is **UNMEASURED**: commit the rung-0 kt=1..10 gate over the admitted
   round-90 frames. The existing committed gate still covers only kt=1 entry
   and complete stage 1; no record acquisition is requested.
2. Re-run rung 7 through kt=10 with the corrected input precondition and
   register its complete 200-row movement table.
3. Resume the rung-0 barotropic source-order walk from round 99 only after item
   1 supplies the required ladder predicate.

## UNVERIFIED

- Rung-0 kt=2..10 bar/status rows remain unmeasured because its gate is not yet
  committed.
- No NEMO acquisition was run or requested.

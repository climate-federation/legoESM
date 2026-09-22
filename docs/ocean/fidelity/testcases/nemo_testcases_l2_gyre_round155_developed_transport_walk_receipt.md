# NEMO testcase L2 GYRE Round 155 receipt — developed transport walk

Date: 2026-09-22  
Status: **HELD** — no physics or configuration changed.  The developed
stage-3 U transport first inherits a non-bit external time-mean transport;
therefore no statement inside the stage transport is an admissible owner.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round155.md`, committed as
`06ebee850`.  The authoritative measurement commit is `c29c3e995`.  Evidence
is under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round155/`; the final
measurement is `developed_transport_walk_c29c3e995.json`.

No physics, configuration, default, carried state, restart schema, stabilizer,
year harness, reconciliation gate, freshwater pair, or #1484 guard changed.
No configuration choice was made.  Without a source-exact candidate there is
no ladder, month, year, DINO, LOCK_EXCHANGE, or OVERFLOW candidate arm.

## Compiled program

This receipt cites only the compiled branch that produced the acquired
record.  Stage 3 binds Kmm to N+1/2 and restores the stage fields at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:261-277`.
The active barotropic-update dispatch forms `zub`/`zvb` from `un_adv`/`vn_adv`,
the live inverse depth, and `uu_b`/`vv_b(Kmm)` at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:300-309`.
The next loop writes `zFu`/`zFv` from metric, reference thickness, live QCO
ratio, mask, Kmm velocity, and that correction at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:313-315`.
The additive writer reads those operands only after both production loops at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:317-321`.

## Admission and method

The Round-154 record's normal gate reports `STATUS PASS`: 20 finite registered
fields and all seven restarts plus `mesh_mask.nc` bit-identical to the
Round-153 baseline.  Its stamp, truncation, restart-byte, and operand-ULP
plants each print their named `STATUS PLANT-FIRED` marker and exit 1.  The
daily record audit at the measurement commit admits 360/360 boundaries and
12/12 monthly overlaps.

The existing year-owner developed-step harness loads NEMO's exact step-1080
restart and executes `LatLonCGridOceanModel.step` through production JIT.  It
observes the shared QCO geometry, corrected-velocity, and metric-transport
helpers without changing their return values.  The observer's full returned
state is bit-identical to an ordinary step.  Production eager and isolated
JIT are reported separately; production-step JIT is authoritative.

The source-ordered production-JIT U table is below.  `unequal` counts every
registered face; `active unequal` counts only faces on which NEMO's 3-D U mask
is one.  Two-dimensional rows use a column with any active U level.

| row | unequal / scored | max abs | active unequal / scored | active max abs |
|---|---:|---:|---:|---:|
| `un_adv` | 580 / 726 | 1.3413024362307624e-6 | 580 / 580 | 1.3413024362307624e-6 |
| `r1_hu_0` | 0 / 726 | 0 | 0 / 580 | 0 |
| `1+r3u(Kmm)` | 580 / 726 | 6.522560269672795e-13 | 580 / 580 | 6.522560269672795e-13 |
| live inverse depth | 580 / 726 | 1.5165277887640993e-16 | 580 / 580 | 1.5165277887640993e-16 |
| `uu_b(Kmm)` | 617 / 726 | 6.134203041352482e-10 | 580 / 580 | 6.134203041352482e-10 |
| written `zub` | 580 / 726 | 3.0153192095982995e-10 | 580 / 580 | 3.0153192095982995e-10 |
| `e2u` | 0 / 726 | 0 | 0 / 580 | 0 |
| reference `e3u_0` | 4,380 / 21,780 | 3.007100172156124e2 | 0 / 17,400 | 0 |
| `umask` | 0 / 21,780 | 0 | 0 / 17,400 | 0 |
| live `e3u(Kmm)` | 21,780 / 21,780 | 3.007100172156124e2 | 17,400 / 17,400 | 1.9616663848864846e-10 |
| `uu(Kmm)` | 17,693 / 21,780 | 1.30926020461275e-6 | 17,400 / 17,400 | 1.30926020461275e-6 |
| corrected U | 17,400 / 21,780 | 1.3093558326737753e-6 | 17,400 / 17,400 | 1.3093558326737753e-6 |
| completed `zFu` | 17,400 / 21,780 | 1.4246544619672932 | 17,400 / 17,400 | 1.4246544619672932 |

All three execution modes have 3/13 bit-exact rows and name `un_adv` first.
Production eager has the same `un_adv` value; its later `uu_b`, `zub`, Kmm
velocity, corrected velocity and completed transport differ in last-bit
rounding from production JIT.  Isolated JIT reproduces the captured production
JIT correction and product exactly, but is not relabelled production.

## Instrument correction and retraction

The first committed instrument reconstructed zero-ssh reference thickness
from the bridge mesh instead of capturing the masked reference operand passed
to the production stage.  That made the live-thickness row appear unequal in
all 21,780 cells with a 300.710017 m maximum.  The internal reconstruction
control exposed 4,380 dry-cell failures.  That row is **RETRACTED**.

The corrected instrument captures the exact production reference operand.
Its model-side reconstruction of live `e3u(Kmm)` from reference thickness,
`1+r3u`, and mask is bit-exact in both production JIT and eager execution.
The remaining 4,380 reference-thickness differences are all dry and the
active reference row is bit-exact.  The corrected active live-thickness
difference is 17,400 cells, maximum 1.9616663848864846e-10 m, inherited from
the already non-bit live ratio.

## Predictions and verdict

1. Record admission and all four acquisition plants: **CONFIRMED**.
2. Reference geometry bit-exact on every registered cell: **REFUTED**.  The
   reference thickness differs in 4,380 dry cells; its 17,400 active cells,
   plus `e2u`, `umask`, and `r1_hu_0`, are bit-exact.
3. First production-JIT non-bit U row is `un_adv`: **CONFIRMED**, 580/580
   active columns, maximum 1.3413024362307624e-6 m2/s.
4. At least one later Kmm state operand is non-bit: **CONFIRMED**; all three
   preregistered state rows are non-bit.
5. Production `un_adv` ULP plant: **CONFIRMED**.  One ULP at active index
   `[1,2]` moves 16 completed `zFu` levels, maximum
   7.275957614183426e-12 m3/s, prints `STATUS PLANT-FIRED`, and exits 1.

**First non-bit statement:** none inside the measured transport block.  The
first unequal input is the external/barotropic time-mean transport `un_adv`.
The stage transport therefore inherits its mismatch and nothing lands.
Round 152 measured the completed FCT contribution at only
1.0974591404090626e-8 K one-step RMS; this walk localises that measured row but
does not claim that `un_adv` carries the 1.6447e-2 K day-240 gap.

The campaign headline rows remain inherited and unmodified: kt2 T/S at the
bar, kt2 U/V approximately 2.7377e-12 / 3.2849e-12, kt3 T approximately
8.60e-7 K, day-30 6.888194e-5 K, day-240 1.644671864e-2 K, and day-360
1.122345086e-2 K.  ORCA2 is **UNMEASURED-WITH-SPEC**: repeat this developed
stage-3 transport and upstream external-average walk on its ocean-only card
before transferring the verdict.

## Controls, review, and tests

The synthetic first-operand registry test passes and fails when the registered
`zFu` row is removed.  The final production-path ULP plant is described above.

The required read-only Codex review was attempted after the implementation.
It did not start a reviewer and emitted verbatim:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Its exit was 1.  Independent review was unavailable in-sandbox; no `SHIP` or
`DO NOT SHIP` verdict is fabricated.  This round lands no physics.

The citation gate, shifted-citation plant, and focused-suite summary are
recorded after this receipt's clean commit.

No unasked scientific or configuration choice was made.

## OPEN — Round 156

Follow `un_adv` upstream through the already acquired developed external-step
record `oracle_bt_step_operands_kt00001081.bin`.  Reuse the Round-81/82/83
external-step harness rather than creating another implementation.  Compare
the 50 substeps and the final accumulated U average under production JIT,
starting with the first source-ordered row that feeds the final `un_adv`.
The first non-bit input owns the next walk.  No new acquisition is needed
unless the existing record lacks that final accumulation boundary; if it does,
write a fail-closed operator-run script rather than inferring it.

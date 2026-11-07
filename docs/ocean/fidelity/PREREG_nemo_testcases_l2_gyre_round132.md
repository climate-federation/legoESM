# Preregistration — NEMO testcase L2 GYRE round 132

Date: 2026-09-20

Incoming lane tip: `c7d13151d95cdbd05fbf7e9d9dee1fa7133f685f`

This document is frozen before the Round-132 acquisition script is written or
its preflight is run. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round132/`.

Round 131 stopped before a legoESM step because the certified daily NEMO
record ends at day 30. Round 132 acquires only the missing oracle input: one
unperturbed, from-rest NEMO GYRE trajectory through day 360 with a restart at
every completed day. It lands no model physics, changes no configuration
choice beyond the operator-specified duration and output cadences, and does
not touch the ORCA2 integration.

## P0 — frozen source card and destination

The only source card is the build that produced the admitted Round-131 daily
record:

* NEMO root: `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`;
* source configuration: `GYRE_OMIP_L2_P3_SM_R41ADVSP`;
* new configuration: `GYRE_OMIP_L2_P3_SM_R132DAILY`;
* existing daily control: `phase3/year_owners/nemo_seed0`;
* existing monthly control: `phase3/year_fromrest/nemo_seed0`;
* new run root: `phase3/round132/oracle_daily_restarts`.

The source executable SHA-256 is frozen as
`a759e8b478e3bda5ba731009fd353159db892c5ac2ba4c48351f5136411960cd`.
The source card contains 55 top-level `EXP00` files or links and 14 top-level
`MY_SRC` files or links. The SHA-256 of the sorted, path-bearing checksum
manifest over those two directories is
`977818735d03095ffbbac145014db463b781e543a0d77bc3ae6228ce79e7887e`;
the source cpp-key file SHA-256 is
`54bd2cead92cee1eaa8cb257c7f7cdea8fe39cc9fb0f47c7c0d5853b909b34b7`.

Frozen prediction: all hashes and counts reproduce. A mismatch is a named
`REFUSE` and stops before `makenemo`.

## P1 — clone and controlled namelist delta

The operator runs one committed `run.sh`. It clones `GYRE_PISCES` with
`makenemo -r`, copies every source-card `EXP00` and `MY_SRC` file or link
file-by-file, copies the cpp keys under the new target name, proves the copied
manifests identical, and builds with the scalar-math architecture. Canonical
NEMO `src/` and the source card remain read-only. There is no Fortran writer
patch: NEMO's existing restart program already writes every required family.

Only these parsed `namelist_cfg` assignments may differ from the source card:

| assignment | source | Round 132 | reason |
|---|---:|---:|---|
| `nn_itend` | 10 | 2160 | operator-specified 360 days |
| `nn_stock` | 10 | 6 | operator-specified daily restarts |
| `nn_write` | 10 | 2160 | retain the certified year run's single diagnostic-output cadence |

The time step remains `rn_Dt=14400`, the calendar remains `nn_leapy=30`, and
the run remains from rest with no `nn_pert_seed` field. Added, removed, or any
other changed assignment is a named `REFUSE`. The compiled source must retain
the scalar-math toolchain, the RK3 restart call, the six barotropic-history
writes, and the four TKE writes.

Frozen prediction: the copied card builds under the new name and its compiled
restart statements match the source build byte-for-byte. A vector-math symbol,
changed cpp key, missing compiled statement, or source-card mutation refutes
the acquisition before `mpirun`.

## P2 — syntax and run controls

Before the build, the script runs `gfortran -fsyntax-only` against the source
build's compiled `restart`, `stprk3`, `dynspg_ts`, and `zdftke` programs using
the admitted build's module include directory. This is a syntax proof of the
exact compiled branch being copied; it is not a substitute for the subsequent
new-target build.

The run uses one MPI rank, one thread, the staged new binary, and no restart
input. It must terminate with process status zero and an exact `STOP 0` line.
No `RUN_DONE` marker is written until all record and admission checks pass.
An ERR trap prints a named `REFUSE` line for any otherwise-unhandled nonzero
exit.

Frozen prediction: NEMO completes 2,160 steps. A nonzero process status,
missing `STOP 0`, changed resolved cadence, or premature marker is a refusal.

## P3 — complete daily record and non-vacuous gate

The run must produce exactly the 360 filenames
`GYRE_OMIP_L2_P3_00000006_restart.nc` through
`GYRE_OMIP_L2_P3_00002160_restart.nc` at stride six, with no unexpected
restart filename. Existing controls establish the frozen restart size as
`1,466,328` bytes; therefore the exact total is `527,878,080` bytes.

The committed Round-131 admission gate must then verify every boundary's
`kt` and `adatrj`, the 16-family schema, `nn_itend=2160`, `nn_stock=6`, and all
twelve monthly overlaps. Its complete-inventory control must pass first. Its
missing-boundary and required-variable plants must each print
`STATUS PLANT-FIRED` and exit nonzero.

Frozen prediction: the inventory has 360 files, every file has the frozen
size and schema, the total byte count is exact, both plants fire, and the gate
prints `STATUS ADMITTED`. Any discrepancy is retained and the round remains
stopped for record.

## P4 — bytewise twin admission

The new run changes restart frequency, so it is admitted only if that cadence
does not change the trajectory. All twelve 30-day boundaries at
`kt=180,360,...,2160` must be byte-identical to the existing monthly seed-0
control. The day-30 boundary must additionally be byte-identical to the
existing daily Round-131 control. The two old day-30 files already share the
frozen SHA-256
`853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6`.

Frozen prediction: all twelve monthly comparisons and the independent old
daily day-30 comparison are byte-identical. One unequal byte refutes the
claim that this is the certified trajectory and prevents `RUN_DONE`.

## P5 — outcome and scope

If P0–P4 pass, the script writes provenance manifests, the Round-131 audit,
an exact restart digest manifest and its clean-commit stamp, then prints
`ROUND132_DAILY_RESTART_RECORD_READY`. Round 132 remains
`STOPPED_FOR_RECORD`: the acquisition has been prepared but the operator has
not run it. `ACQUISITION_NEEDED` names this script; `DECISION_NEEDED` is
`NONE`.

Round 133 must run the Round-131 gate again at its own clean commit before any
nudged model step, then execute all four daily-reset families. No three-step
cadence is inferred from daily data. A separate read-only Codex pass must try
to refute source-card identity, namelist isolation, count/size coverage,
bytewise twin admission, and the handoff verdict. The receipt citation gate
and its shifted-citation plant remain mandatory.

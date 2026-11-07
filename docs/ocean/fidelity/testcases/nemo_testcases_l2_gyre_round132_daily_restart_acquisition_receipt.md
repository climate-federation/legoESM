# NEMO testcase L2 GYRE phase 3 — round 132 daily-restart acquisition receipt

Date: 2026-09-20

Incoming tip: `c7d13151d95cdbd05fbf7e9d9dee1fa7133f685f`

Status: **STOPPED_FOR_RECORD — the fail-closed 360-day daily-restart NEMO
acquisition is committed and preflight-proved, but no NEMO acquisition was
run in this agent round; Round 133 remains blocked until the operator runs the
named script and it admits all 360 boundaries.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round132/`

## Outcome first

Round 132 supplies the missing acquisition requested by the Round-131 OPEN
section and operator note AG:

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round132_daily_restarts/run.sh`

The default mode is the operator-only acquisition. It creates the new NEMO
configuration `GYRE_OMIP_L2_P3_SM_R132DAILY`, runs 2,160 four-hour steps from
rest, and targets
`phase3/round132/oracle_daily_restarts`. This agent ran only its safe
`--preflight` mode. It did not invoke `makenemo` or `mpirun`, did not create the
target configuration or target record, and did not alter NEMO source.

The preflight reproduced the frozen source-card identity, the old-control
inventory, the only three permitted namelist changes, and the syntax of all
four compiled programs that define the record. It printed:

```text
CONTROL_CENSUS_PASS daily=30 monthly=12 bytes=1466328
namelist assignments changed from the source card:
  nn_itend: 10 -> 2160
  nn_stock: 10 -> 6
  nn_write: 10 -> 2160
SYNTAX_PROOF_PASS restart.f90
SYNTAX_PROOF_PASS stprk3.f90
SYNTAX_PROOF_PASS dynspg_ts.f90
SYNTAX_PROOF_PASS zdftke.f90
ROUND132_DAILY_RESTART_PREFLIGHT_READY /tmp/gyre-r132-provenance.b7kkm12g
```

There is no acquisition result to overstate: record count, bytewise monthly
twins, schema admission, and the final ready marker are **UNMEASURED** until
the operator runs the default mode. `RUN_DONE` is written only after all of
those checks pass.

## Compiled-source record contract

The frozen source card is the compiled
`GYRE_OMIP_L2_P3_SM_R41ADVSP` branch that produced the already-admitted daily
control. At the completed-step boundary NEMO swaps the `Naa` state into `Nbb`
and forms the next-step extrapolated SSH at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:222-226`; after its
diagnostics it calls the restart writer with those resolved levels at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:249-260`.

The compiled writer stores `sshn`, `un`, `vn`, `tn`, `sn`, the two depth-mean
vectors, and `ssha` at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/restart.f90:176-184`. The active
time-split external-mode branch stores the six carried barotropic histories at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynspg_ts.f90:974-980`. The active
TKE branch stores `en`, `avt_k`, `avm_k`, and `dissl` at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90:895-901`.

Those are the same five compiled statements admitted in Round 131. Round 132
adds no writer or instrument. The acquisition instead copies the complete
source card, rebuilds it under a new target name, and refuses the run unless
all four target compiled files are byte-identical to these source-build files.

## Frozen source and run identity

| item | frozen value | preflight |
|---|---|---|
| source configuration | `GYRE_OMIP_L2_P3_SM_R41ADVSP` | confirmed |
| source executable SHA-256 | `a759e8b478e3bda5ba731009fd353159db892c5ac2ba4c48351f5136411960cd` | confirmed |
| `EXP00` top-level files/links | 55 | confirmed |
| `MY_SRC` top-level files/links | 14 | confirmed |
| path-bearing card-manifest SHA-256 | `977818735d03095ffbbac145014db463b781e543a0d77bc3ae6228ce79e7887e` | confirmed |
| source cpp-key SHA-256 | `54bd2cead92cee1eaa8cb257c7f7cdea8fe39cc9fb0f47c7c0d5853b909b34b7` | confirmed |
| scalar architecture SHA-256 | `132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561` | confirmed |
| old daily boundaries | `kt=6..180` by 6, 30 files | confirmed |
| old monthly boundaries | `kt=180..2160` by 180, 12 files | confirmed |
| restart bytes | 1,466,328 each | confirmed on all old controls |
| old daily/monthly day-30 SHA-256 | `853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6` | confirmed and byte-identical |

The script performs `makenemo -r GYRE_PISCES` under the new target name, then
copies every top-level source-card `EXP00` and `MY_SRC` entry file-by-file and
copies the cpp keys under that target name. It compares the complete
link-aware path-bearing entry manifest before building: regular files compare
by SHA-256, while symbolic links compare by their recorded link target. This
distinction is required for `EXP00/nemo`, whose identical `../BLD/bin/nemo.exe`
link resolves to the preliminary target binary before the intended rebuild.
The script then rebuilds with `conda-scalarmath`, rejects vector-math symbols,
records the actual new binary hash, and compares the four relevant compiled
files byte-for-byte with the frozen source build.

The operator specified 2,160 steps and a restart every six steps. The only
additional cadence row is `nn_write=2160`, inherited from the certified
360-day year protocol so that frequent diagnostics are not introduced. This
is an output-volume choice, not a physics, forcing, initial-state, or carried-
state change. The parsed whole-namelist comparison permits exactly those
three rows, rejects additions/removals, and explicitly refuses
`nn_pert_seed`; all other source-card settings remain identical.

## Fail-closed acquisition and admission

The operator run must satisfy every row below before it can write `RUN_DONE`
or `ROUND132_DAILY_RESTART_RECORD_READY`:

| gate | exact requirement | current disposition |
|---|---|---|
| clean provenance | committed clean worktree and exact producer commit | enforced; operator run pending |
| target isolation | new config and run roots do not already exist | enforced; operator run pending |
| process result | one MPI rank, status 0, exact `STOP 0` | UNMEASURED |
| resolved card | `rn_Dt=14400`, `nn_itend=2160`, `nn_stock=6`, `nn_write=2160` | UNMEASURED |
| file census | exactly `kt=6,12,...,2160`, no extras | UNMEASURED |
| size census | 360 files × 1,466,328 bytes = 527,878,080 bytes | UNMEASURED |
| monthly twin | all 12 `kt=180,360,...,2160` restarts byte-identical to the year control | UNMEASURED |
| independent day-30 twin | new `kt=180` byte-identical to the old daily control | UNMEASURED |
| complete gate | all 360 metadata rows and all 16 required fields admitted | UNMEASURED |
| controls | complete synthetic inventory passes; both plants exit 1 with fired markers | confirmed in preflight; repeated after acquisition |
| final stamp | 360-row digest manifest bound to the clean producer commit | written only after admission |

Every explicit refusal calls a named `REFUSE` line. An ERR trap covers an
otherwise-unhandled shell failure. As a direct control, the unknown-mode arm
exited 64 and printed:

```text
REFUSE: unknown mode --definitely-invalid; use --preflight or no argument
```

No partial acquisition can masquerade as ready: the run marker is the final
write after process, census, byte-twin, schema, plant, and commit-stamp checks.

## Frozen prediction ledger

* **P0 CONFIRMED in preflight.** All source-card hashes and counts reproduce.
  No target configuration was created.
* **P1 PARTIALLY CONFIRMED.** The parsed dry namelist changes exactly
  `nn_itend`, `nn_stock`, and `nn_write`; the compiled new-target identity and
  scalar binary remain UNMEASURED until the operator build.
* **P2 PARTIALLY CONFIRMED.** All four source-build compiled programs passed
  `gfortran -fsyntax-only`. The 2,160-step process result is UNMEASURED.
* **P3 UNMEASURED.** No new daily files exist. The exact 360-file census,
  527,878,080-byte total, metadata, and 16-field schema predictions remain
  frozen.
* **P4 UNMEASURED.** The twelve monthly byte twins and independent old-daily
  day-30 twin remain acquisition admission criteria, not results.
* **P5 CONFIRMED.** The committed artifact is an acquisition handoff, not a
  physics landing. Round 132 stops for the record and names the script in its
  machine-readable handoff.

No trajectory row, day-240 score, DINO result, tank result, or ORCA2 result
moved: legoESM production, its cards, its harnesses, and its carried state are
unchanged. The ORCA2 assembly present at the incoming tip was not touched.

## Mechanical controls and evidence

The safe preflight log is `phase3/round132/preflight.log` (SHA-256
`0e79f30fa7d57d8c53e3ffc30b2cc3adcb4f6a7522bcb9ae9b5f8ace7d83f3f4`).
It includes the exact control census, three-row delta, four syntax proofs, and
the preflight-ready marker. It contains no `RUN_DONE` marker.

The reused Round-131 admission gate first passed its complete 360-boundary,
16-field virtual control. Starting from that passing control, the
missing-boundary plant removed only `kt=2160`, and the required-variable plant
removed only `dissl`. Both exited exactly 1 and printed:

```text
STATUS PLANT-FIRED: missing-boundary
STATUS PLANT-FIRED: required-variable
```

Their logs are `phase3/round132/missing-boundary_plant.log` and
`phase3/round132/required-variable_plant.log`; the passing control is
`phase3/round132/record_gate_self_check.log`. The named-refusal control is
`phase3/round132/named_refuse_control.log`.

## Independent adversarial review

The required separate read-only Codex pass was invoked against the Round-132
commits, this receipt, the Round-131 gate, and the cited compiled source. Its
prompt explicitly tried to refute source-card identity, exact namelist
isolation, count/size coverage, bytewise twin admission, control non-vacuity,
commit stamping, final-marker ordering, and the STOPPED_FOR_RECORD handoff.
It could not initialize in the sandbox and returned status 1. Its verdict,
verbatim, was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Thus, **independent review unavailable in-sandbox**. The invocation and full
output are retained in `phase3/round132/codex_review.log`; it issued no
`DO NOT SHIP` verdict to override.

## Verification

All Python commands used the required CPU/fp64 environment.

* `bash -n` accepted the acquisition script.
* The direct Round-131 admission-gate test suite reported exactly
  `7 passed in 0.65s`.
* The real-source safe preflight completed all four syntax proofs and returned
  status zero without invoking `makenemo` or `mpirun`.
* The unknown-mode control returned 64 with its named `REFUSE` line.
* The combined Round-131 admission and receipt-citation suites reported
  exactly `23 passed in 2.65s`.
* The unplanted citation run found five citations, zero unmapped citations,
  zero failures, an empty whole-map audit, all nine internal controls firing,
  and `status: PASS`.
* Shifting the registered completed-step citation by two lines returned exit
  1 with `status: FAIL` and `SYMBOL-NOT-AT-LINE`. The retained artifacts are
  `phase3/round132/citation_gate.json` and
  `phase3/round132/citation_gate_shifted_plant.json`.

No full model or all-tree pytest battery ran: this round changes one operator
acquisition shell script and documentation, not model code, a card, a Python
gate, or NEMO source. The syntax proof, real-input preflight, admission-gate
suite, citation gate, and adversarial review cover the committed scope.

## OPEN — round 133

1. The operator runs the committed Round-132 acquisition script. Any named
   refusal leaves the record unadmitted; do not repair or infer missing state.
2. Round 133 reruns the Round-131 audit against
   `phase3/round132/oracle_daily_restarts` from its own clean stamped commit.
   Do not start a legoESM reset arm unless it prints `STATUS ADMITTED`.
3. Run the four independent 360-day daily-reset families: T/S; u/v plus all
   six carried barotropic histories; SSH plus `ssha` and both SSH histories;
   and TKE state. Score every field at days
   30/60/90/120/180/240/300/360 against the Round-130 immutable free arm and
   rank measured day-240 T3D removal.
4. For the owner family, halve the reset cadence to three steps only if an
   independently recorded matching NEMO boundary exists at every such step.
   Daily restarts do not support that inference.
5. Re-score `r99_wclock` alone only if all four owner-family arms and their
   attribution table fit first.

The Round-132 acquisition does not select a scientific configuration, add a
stabilizer, or change carried state. `DECISION_NEEDED` is therefore `NONE`.

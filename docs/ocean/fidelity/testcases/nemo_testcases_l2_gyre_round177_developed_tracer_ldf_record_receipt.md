# Round 177 receipt — developed tracer-LDF operand record

**Status: STOPPED_FOR_RECORD.**  Round 176 selected lateral tracer diffusion by
magnitude, but no admitted record contains its developed direct operands or
intermediates.  Round 177 preregistered and syntax-proved a passive step-1081
NEMO record.  No NEMO run occurred in the sandbox, no scientific row was
measured, and no production physics, configuration, carried state, restart
schema, card default, or trajectory changed.

Preregistration: `PREREG_nemo_testcases_l2_gyre_round177.md`, commit
`ddc5cbcd9`.  Final acquisition preflight commit: `447b1aedb`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round177/`.

## Why a record is required

The admitted Round-132 compiled dispatcher enters `tra_ldf` and selects the
standard iso-neutral Laplacian call at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf.f90:69-110`.
The selected compiled routine calls `traldf_iso_a33`, evaluates the masked
tracer gradients, face coefficients and fluxes, and adds their divergence to
temperature `Krhs` at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf_iso.f90:154-300`.

The existing Round-123 record stores only the completed before/after `Krhs`
boundary.  The Round-40 and Round-64 records are from-rest, and Round 148 is a
momentum-LDF record.  None contains developed step-1081 tracer, QCO geometry,
slope, diffusivity, or flux operands.  Therefore no internal statement can be
named honestly this round.

## Acquisition contract

The new target is `GYRE_OMIP_L2_P3_SM_R177TRALDF`; its output root is
`round177/oracle_tracer_ldf_walk`.  The script copies the Round-132 EXP00,
MY_SRC, and preprocessor card file by file, adds only the two writer patches,
and runs through step 1440.  It records one 8,903,548-byte stream at step 1081:

* 38 full 3-D arrays: temperature `Kbb`, `Krhs` before/after/increment, live
  thicknesses, masks, diffusivities, slopes, masked gradients, matrix
  coefficients, and horizontal/vertical fluxes;
* 11 full 2-D QCO/metric arrays; and
* the 31-level `e3w_1d` reference thickness.

The fixed header is 16 bytes plus 13 four-byte integers, followed by
`(38*36*26*31 + 11*36*26 + 31)` eight-byte values.  The admission parser pins
magic, header, dimensions, time levels, field counts, dtype, and final offset.
It also requires 18,000 active cells, finite arrays, non-vacuous intermediate
rows, a bit-exact stored increment, and a bit-exact match to Round 123's
after-LDF minus after-QSR increment.

Because the observer stores values inside the compiled loop, passivity is not
assumed.  The candidate step-1080 and step-1440 restarts must each be
byte-identical to the uninstrumented Round-132 restart.  Either moved restart
refuses the record.  This implements the developed-record admission policy;
there is no inherited-stream waiver in the primary admission.

## Frozen controls and preflight

The acquisition's source ancestry is fixed by the Round-132 binary, canonical
`traldf_iso.F90`, include file, two reference restart hashes, and the
Round-123 process-record hash.  Both source patches are additive.  The dry
source was preprocessed with the card's `key_qco`, `key_vco_1d3d`, and
`key_RK3` state and passed:

> `SYNTAX_PROOF_PASS traldf_iso.f90`

> `ROUND177_TRACER_LDF_PREFLIGHT_READY /tmp/gyre-r177-tracer-ldf-source.Iw3Lsvl5`

The source-layout plant removes the recorded upper vertical flux.  It printed
`STATUS PLANT-FIRED: source-layout`, a named `REFUSE`, and exited 69.  The
record-magic plant is implemented but cannot execute until the record exists;
`--plant-admission` must exit nonzero before the record is used in Round 178.

The first preflight attempt correctly refused a nonexistent
`nn_pert_seed=0` assertion.  The admitted Round-132 member-zero namelist has no
such row.  Commit `447b1aedb` removes only that invalid assertion; binary,
namelist, source, process-record, and restart hashes remain binding.  This is a
recorded instrument correction, not a hidden configuration choice.

## Independent review

The required separate `codex exec --sandbox read-only` review could not start
in this sandbox.  Its complete terminal disposition is quoted verbatim:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
>
> codex_exit_code=1

This is **independent review unavailable in-sandbox**, not a SHIP verdict.
The acquisition remains operator-gated and cannot produce a scientific claim
until every admission and plant passes.

## Trajectory and card census

There is no production candidate, so Decisions 43/45/55/59, the year run, and
the card census do not run.  The immutable Round-163 trajectory remains:

| headline | unchanged value |
|---|---:|
| first over bar | kt=3 |
| kt2 U max abs | `8.326672684688674e-17 m s-1` |
| kt2 V max abs | `9.714451465470120e-17 m s-1` |
| kt3 T max abs | `4.940071072212504e-07 K` |
| kt3 S max abs | `4.0086298724872904e-08 psu` |
| day-30 T3D RMS | `6.572574374770603e-05 K` |
| day-240 T3D RMS | `1.644836070117868e-02 K` |
| day-360 T3D RMS | `1.122566001855131e-02 K` |

No GYRE, generic, DINO, tank, ORCA2, or MPAS production row moved.  No
configuration or carried-state decision is requested.

## Evidence

| artifact | SHA-256 |
|---|---|
| `preflight.log` | `dd901e9e13dcf14a6b3e29a45d67ad46c54293a2986541211d78acdb6f8b41f6` |
| `layout_plant.log` | `831c68020af81934d5c23f7fa3d311cc859ede03422abc1b86cbbaaadecad29d` |
| `codex_review.log` | `52e37095a196d69ea605b6b132ddf8ece1a9087cb75fdc3c3909f62877501da8` |

## OPEN — round 178

The operator runs the committed Round-177 `run.sh`.  Round 178 first runs
`--admit-existing` and `--plant-admission`.  If both step-1080 and step-1440
restarts are byte-identical and every calibration passes, extend the existing
year-owner harness to compare the recorded developed tracer-LDF rows under the
production JIT step and the complete eager step.  Name the first non-bit
statement in compiled order.  If either restart differs, the writer perturbs
NEMO: refuse it and move the observer to a passive routine boundary before any
scientific interpretation.

# SI3 round 18 preregistration — O8–O16 master-record acquisition

## Claim and run

One instrumented, one-rank, 100-step ORCA2_SAS_ICE run will acquire the exact
stage views needed by O8–O16. The ice namelist is byte-for-byte the production
namelist. There is no `ln_zdf_chkcvg` or other physical overlay. Production
dynamics remains on; dynamics outputs are explicitly outside every O8–O16
projection. The O7-completed surface products are the forcing boundary.

The reference and record executables are built from the same source/configuration;
the latter adds only marked observer calls and the stream writer. The run manifest,
including every source, deck and executable digest, is written before either run.

## Registered boundaries and projections

The master schema has 20 ordered record kinds: frazil entry/exit; category entry;
ZDF, DH, first TEMP, salinity, second TEMP and `ice_thd_da` exits; category/pond,
pond/remap, remap/open-water, open-water/correction and correction boundaries;
`ice_update_flx` entry/exit (the latter is the first ocean-format boundary); and
the final BL99 Picard state; the pre-aggregation category state; and
`ice_update_tau` entry/exit. Category frames are recorded at every step and all
five categories. Picard final state is recorded only at steps 1, 2 and 100.

Each admission reader exposes only these record kinds:

| rung | kinds | required records |
|---|---|---:|
| O8 | frazil entry/exit | 200 |
| O9 | category, ZDF, DH, TEMP1, Picard final | 2,015 |
| O10 | TEMP1/salinity, salinity exit, TEMP2 | 1,500 |
| O11 | `ice_thd_da` entry/exit | 1,000 |
| O12 | category-complete/pond exit | 200 |
| O13 | pond exit/remap exit | 200 |
| O14 | remap exit/open-water exit | 200 |
| O15 | open-water exit/correction exit | 200 |
| O16 | pre-aggregation/`ice_update_flx` entry/ocean-format exit/`ice_update_tau` entry/exit | 500 |

## Frozen acquisition predictions

- P1 CONFIRM: exactly 4,715 rank-0 records exist with valid headers, finite
  binary64 payloads, exact declared extents and no trailing bytes. REFUTE: any
  missing/extra record, invalid extent, nonfinite value or byte-count mismatch.
- P2 CONFIRM: reference and observer-run restart filename sets and bytes are
  identical. REFUTE: any differing byte or file set.
- P3 CONFIRM: both logs contain `STOP 0`, contain the expected dynamics and
  thermodynamics banners, and contain no spaced `E R R O R` or abort token.
  REFUTE: any missing completion marker or fatal token.
- P4 CONFIRM: each per-rung reader sees exactly the table projection; no reader
  admits a foreign record kind. REFUTE: missing, foreign or wrong-category entry.
- P5 CONFIRM: the real-flag census covers all eight patched units;
  observer-free and instrumented counts agree mnemonic by mnemonic, and an
  injected FP operation changes each named routine's nonzero count. REFUTE: a
  zero census, any passive-arm difference or a plant that does not fire.

These are acquisition predictions, not physics scores. No O8–O16 fidelity verdict
is authorized until the retained record exists and the respective scorer runs.

## Storage, retention and rerun

Worst-case sizing uses ORCA2 local dimensions 182×149, interior 180×147, five
categories and 10+5 layers, pessimistically treating every interior cell as
gathered in every category. The master is 133,386,394,340 bytes (124.23 GiB).
The largest retained projection is O9 at 34.21 GiB; the
other projections are references into the same immutable files, never copies.
The run refuses below 150,000,000 KiB free disk or 150 GiB available RAM, and
rejects final evidence above 150,000,000 KiB or total acquisition time above
14,400 seconds.

Raw records, logs, production decks, source/binary manifests, per-rung relative
manifests and builds are retained until every O8–O16 admission completes. A
refusal moves the evidence directory aside and keeps both builds. The documented
rerun is `run.sh --move-aside-builds`, then `run.sh --run`; nothing is deleted.

## Controls before operator execution

The default read-only preflight applies every patch to the original NEMO `.F90`
source and preprocesses exactly once with the production CPP command. It compiles
at production flags, runs the per-function census and its plants, and executes
all 20 writer kinds through the real writer and all eight patched units under
`-fcheck=bounds` with distinct values and `-0.0`. It then compares to an
independently copied observer-free build,
validates namelist group ownership with misspelling/misplacement plants, checks
source-derived SAS time level, validates A2D(0) handling, and fires schema,
checksum, truncation, swapped-field and every projection plant. A stub NEMO
exercises the generated 100-step driver's syntax, run/admission/refusal,
retention and move-aside paths. The stub replaces only `nemo.exe` and, because
the validation sandbox forbids PMIx sockets, `mpirun`; the same
preflight, patching, same-program and `mpirun` call paths execute as for NEMO.

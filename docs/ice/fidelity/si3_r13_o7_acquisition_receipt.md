# SI3 round 13 receipt — O7 operator acquisition tool

## Outcome

The O7 NEMO acquisition tool is complete and its read-only preflight passes.
No NEMO build or integration was run. The operator must run
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o7_orca2_sas_acquisition/run.sh --run`;
until that happens, O7 remains UNMEASURED against NEMO.

The tool uses DECISION 165 exactly: the shipped, unmodified production
`ORCA2_SAS_ICE` physics deck, one MPI rank, and `kt=1..2`. Only the run-control
end/stock/write values are shortened for the two-step acquisition. The gate
reconstructs every O7 scalar from the run's resolved namelists and requires
the resulting configuration to equal the four shipped production namelists.
`nn_write=-1` disables SAS output safely; zero is forbidden because non-XIOS
`diawri` divides by it. Every staged assignment is checked against its exact
reference-namelist group, with misspelling and misplacement plants. No
campaign bar or production model default changed.

## Source-derived observation contract

Four observer patches bracket the preregistered running call points in
`blk_ice_1`, the post-exchange stress, `ice_alb`, `blk_ice_2`, and
`ice_flx_other`. The patches apply with zero fuzz to NEMO 5.0.2, are
preprocessed with the record build's real keys and include files, and compile
against the real NEMO module interfaces under the production flags.
The global `ice_alb` observer records only the named pre-thermodynamics S4
call; the later `src/ICE/iceupdate.F90:208` call is independently pinned after
`ice_flx_other` and cannot overwrite S4. A finish observer requires the complete
eleven-position call order and exactly two albedo entry/exit pairs per step.

The observed statements include the wind magnitude
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1082`), category shortwave
flux (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1231`), the category
sum for `emp_ice` (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1290`),
the stress T exchange
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:167`), production-dynamics
bottom friction (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:396`), the
bottom-flux cap (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:436`), and
the lead heat flux (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:463`).
The preflight moves each of the twelve calls in turn and requires every
misplacement plant to fail.

The writer emits nine kinds at each of two steps: `blk_ice_1` entry/exit,
post-exchange stress, albedo entry/exit, `blk_ice_2` entry/exit, and
`ice_flx_other` entry/exit. S4 has distinct `o7_s4_alb_entry...` and
`o7_s4_alb_exit...` filenames. Its 148-byte header declares global and filled
extents, category count, halo, decomposition, step, `Kmm`, `nit000`, and the
seven run scalars. Interior records use NEMO's `ntsi:ntei,ntsj:ntej` filled
ranges; only preregistered full-halo fields use `1:jpi,1:jpj`. Direct stream
writes retain signed zeros. Admission requires exactly 18 frames, with no
missing or extra file.

## Passivity and same-program proof

The shared same-program helper strips observer blocks from actual record
`ppsrc`, requires byte identity with reference `ppsrc`, and separately
compares every nonblank marked line with a fresh preprocessing of the
committed patch. Outside-marker and inside-marker plants both fail. Preflight
and admission use the build's `%CPP`/`%FPPFLAGS`, not `gfortran -E`.

The per-function passivity census reads `%FCFLAGS` from the record build and
requires the production `-O3`, real-size, unrolled, non-vectorized flags. It
compiles real observer-free and instrumented NEMO `ppsrc` and compares
nonzero floating-point mnemonic counts for `blk_ice_1`, `blk_ice_2`,
`ice_alb`, `ice_sbc_tau`, `ice_sbc_flx`, and `ice_stp`. Private
`ice_flx_other` is inlined and is never required as a symbol. A real-source
injected operation must change the census. One `-fcheck=bounds` executable
compiles and runs the real patched, preprocessed NEMO `sbcblk`, `icealb`,
`icesbc`, and `icestp` observer path against minimal conforming stubs for two
pseudo-ranks. It verifies distinct per-rank records, signed zero, all 18 frame
layouts, and both ordered `ice_alb` calls.

## Admission and controls

The retained manifest pins the producer commit, reference/record executable
digests, every record checksum, all staged input files, the byte copies of the
production deck, and the same-program and passivity reports. Paths are
evidence-root-relative. `--admit-existing` re-runs preflight and admission in
place against that manifest; failure does not discard the retained evidence.
Reference and record restarts must be present and byte-identical, and both
runs must have a clean `STOP 0` plus the expected production selectors in
their real `ocean.output`; any `E R R O R` line is quoted in the refusal.
Refused builds are never deleted. The documented rerun is
`run.sh --move-aside-builds` followed by `run.sh --run`; the preflight proves
both builds remain below `refused/builds_*` and the target names become free.

Admission sets fp64, CPU production JIT, and the glibc/libm transcendental
policy before scoring. For both steps it requires zero ULP, including signed
zero, on wet cells and on full arrays. It reports seam and fold maxima for
all nine `blk_ice_1` outputs, both full-halo stresses, albedo, all 19
`blk_ice_2` outputs and 11 statement intermediates, all three
`ice_flx_other` outputs, and the retained `frq_m`. The composed surface step
is independently required to reproduce its 45 returned leaves. The native
transcendental policy must make the retained-record control fail; a vacuous
native control refuses admission.

The swapped-field, composition-feed, and SUM-seed controls write record-shaped
fixtures, load them through `_load_step`, and fail through `_score_step`; they
therefore exercise the same scoring path as admission. The fold-sign control
changes only the north-fold row rather than negating an entire stress array.

Controls also cover magic/version, truncation, declared versus filled
extents, relative checksums, missing frames, source-derived SAS time level
(the plant changes the binary header), first-wet-cell one-ULP changes, swapped
fields, `Cd_ia +1 ULP`, wrong fold sign, wrong pond branch, changed
`nn_snwfra`, wrong SUM seed, a composition-only P6 perturbation, strict wet
signed zero, selector drift, producer stamp, binary/input/record digests, and
restart divergence. Because the retained real excerpt is an O1 dynamics-off
run, preflight exercises `verify_o7_log` on a synthetic dynamics-on echo; the
real production-deck log check occurs at admission.

## Verification

- Read-only preflight: PASS, eleven sections and 36 planted violations; no NEMO
  build or integration.
- Focused O7 acquisition tests: 18 passed.
- NEMO acquisition result: ACQUISITION_NEEDED.
- independent review: operator (Claude)

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: raw patches land; marker-stripped real preprocessed sources equal
  their references; interfaces compile under real production flags; the
  bounds, passivity, placement, record-reader, wet-cell, retention, and log
  controls all fire in read-only preflight.
- PLAUSIBLE: zero-ULP agreement with NEMO for O7. No O7 NEMO record was read,
  so the frozen round-11 predictions remain unmeasured.

## Choices

ASKED: unmodified production deck, one rank, `kt=1..2`, the preregistered call
points and arrays, CPU production JIT with glibc/libm, exact wet/full ULP
admission, and operator-only NEMO execution.

UNASKED: the shared same-program comparator gained an optional marked-`ppsrc`
mode so O7 can compare like with like. Existing callers retain their previous
raw-diff behavior. No production model or configuration default changed.

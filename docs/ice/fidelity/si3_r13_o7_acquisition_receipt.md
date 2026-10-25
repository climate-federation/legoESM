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
No campaign bar or production model default changed.

## Source-derived observation contract

Four observer patches bracket the preregistered running call points in
`blk_ice_1`, the post-exchange stress, `ice_alb`, `blk_ice_2`, and
`ice_flx_other`. The patches apply with zero fuzz to NEMO 5.0.2, are
preprocessed with the record build's real keys and include files, and compile
against the real NEMO module interfaces under the production flags.

The observed statements include the wind magnitude
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1082`), category shortwave
flux (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1231`), the category
sum for `emp_ice` (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1290`),
the stress T exchange
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:167`), production-dynamics
bottom friction (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:396`), the
bottom-flux cap (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:436`), and
the lead heat flux (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:463`).
The preflight moves each of the eleven calls in turn and requires every
misplacement plant to fail.

The writer emits nine kinds at each of two steps: `blk_ice_1` entry/exit,
post-exchange stress, albedo entry/exit, `blk_ice_2` entry/exit, and
`ice_flx_other` entry/exit. Its 148-byte header declares global and filled
extents, category count, halo, decomposition, step, `Kmm`, `nit000`, and the
seven run scalars. Interior records use NEMO's `ntsi:ntei,ntsj:ntej` filled
ranges; only preregistered full-halo fields use `1:jpi,1:jpj`. Direct stream
writes retain signed zeros. Admission requires exactly 18 frames, with no
missing or extra file.

## Passivity and same-program proof

The shared same-program helper now has a marker mode for real preprocessed
sources. It strips explicit observer blocks from the actual record `ppsrc`
and requires byte identity with reference `ppsrc`; it does not compare raw
patch hunks with preprocessed sources. A change outside a marker is planted
and must fail. The operator path repeats this proof over the two real builds
before either executable is launched.

The per-function passivity census reads `%FCFLAGS` from the record build and
requires the production `-O3`, real-size, unrolled, non-vectorized flags. It
compares floating-point instruction counts by mnemonic, with a nonzero total,
for `blk_ice_1`, `blk_ice_2`, `ice_alb`, and `ice_flx_other`. The preflight
compiles the census fixture with those flags and proves an injected floating
operation changes the result. One `-fcheck=bounds` executable invokes every
writer interface for both steps and reads all resulting layouts; a second
compiles and runs the real patched, preprocessed NEMO `ice_alb` routine against
conforming stubs, including an interior array whose lower bound is not one.

## Admission and controls

The retained manifest pins the producer commit, reference/record executable
digests, every record checksum, all staged input files, the byte copies of the
production deck, and the same-program and passivity reports. Paths are
evidence-root-relative. `--admit-existing` re-runs preflight and admission in
place against that manifest; failure does not discard the retained evidence.
Reference and record restarts must be present and byte-identical, and both
runs must have a clean `STOP 0` plus the expected production selectors in
their real `ocean.output`.

Admission sets fp64, CPU production JIT, and the glibc/libm transcendental
policy before scoring. For both steps it requires zero ULP, including signed
zero, on wet cells and on full arrays. It reports seam and fold maxima for
all nine `blk_ice_1` outputs, both full-halo stresses, albedo, all 19
`blk_ice_2` outputs and 11 statement intermediates, all three
`ice_flx_other` outputs, and the retained `frq_m`. The composed surface step
is independently required to reproduce its 45 returned leaves. The native
transcendental policy must make the retained-record control fail; a vacuous
native control refuses admission.

Controls also cover magic/version, truncation, declared versus filled
extents, relative checksums, missing frames, SAS time level, first-wet-cell
one-ULP changes, strict wet signed zero, selector drift, producer stamp,
binary/input/record digests, and restart divergence. The default path has RAM,
disk, timeout, error-trap, and failure-retention gates. A committed excerpt
from a real NEMO ORCA2_SAS_ICE `ocean.output` is checked byte-for-byte against
its source and exercises the whitespace-tolerant production log parser.

## Verification

- Read-only preflight: PASS, ten sections and 24 planted violations; no NEMO
  build or integration.
- Focused O7 acquisition tests: 14 passed.
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

# SI3 round 12 receipt — O5 operator acquisition tool

## Outcome

The O5 NEMO acquisition tool is complete and its read-only preflight passes.
No NEMO build or integration was run. The operator must run
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o5_orca2_sas_acquisition/run.sh --run`;
until that happens, O5 remains UNMEASURED against NEMO.

The frozen overlay is one MPI rank, `nn_it000=1`, `nn_itend=3`,
`ln_dynALL=T`, `ln_dynRHGADV=F`, `ln_ridging=F`, `ln_rafting=F`,
`ln_icethd=F`, H79 strength with smoothing, and aEVP with 100 inner
iterations. No campaign bar was relaxed.

## Source-derived observation contract

The observer patches quote the running ORCA2_SAS_ICE arm, not a dead branch.
A1 brackets NEMO's strength call
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:268`). A2 is
after H79's raw strength statement
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1081`), while A3
is after smoothing and its T exchange
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1114`). B1 follows
the levitating SSH statement
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:281`), whose active
arm returns ocean SSH (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1090`).

B2 follows the one-time F-mask exchange
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:238`). Each selected
inner iteration 1, 2, 3, 50, 99, and 100 has separate records after the
delta exchange (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:440`),
after the stress update, and after the parity-dependent velocity exchange
(`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:647` and `:754`).
The entry and exit around `ice_dyn_rdgrft` retain the preregistered
no-redistribution control.

The writer produces direct unformatted stream writes with a 112-byte common
header. In addition to `jpi`, `jpj`, `jpl`, halo, decomposition, step,
iteration, rank, and SAS time-level metadata, it stores `Nis0`, `Nie0`,
`Njs0`, and `Nje0`. The three A2D(0) diagnostics are consequently read at
their declared 180-by-148 interior extent. Fast and iceberg U/V masks are
recorded only on the 183-by-151 rectangle actually filled by their common
NEMO loop, and inner stresses use the exact runtime `ntsi`/`ntei` and
`ntsj`/`ntej` bounds of their two producing loops. There are 82 records for
the three-step, one-rank acquisition window; signed-zero bits are retained.
The required `Kmm` is derived from the SAS `Nnn` assignment and `sbc` call,
then checked in every record rather than hard-coded by the reader.

## Admission and controls

The admission gate imports O1's wet-mask, ULP, signed-zero, and real-log
helpers and round 328's config-root-normalized same-program comparator. It
requires the producer commit to exist in this repository and match the run
manifest, verifies the two retained executables against relative manifest
paths and digests, verifies every raw record and individual payload checksum,
and refuses a retained run produced by any commit other than the current
manifest producer. It requires `RUN_COMPLETE` and `STOP 0` independently for
the reference and record runs, then byte-compares both step-3 ocean and ice
restart files before scoring. A failed admission leaves a completed evidence
directory in place.

Strength and aEVP replay run under CPU production JIT with fp64/libm policy.
Strength, B2 `fimask`, all delta/stress/velocity fields at B3 iterations 1,
2, 3, 50, 99, and 100, final stresses and velocities, SSH, fast-mask filled
ranges, metric areas and reciprocals, and the no-redistribution entry/exit are
required to be 0 ULP on wet masks and full recorded arrays; seam and fold
maxima are reported. The shared production solver supplies the trace, so
inner-frame scoring and final-state scoring cannot diverge into two solvers.

Preflight plants cover each of the twelve observer landing sites, header,
truncation, checksum, SAS time level, swapped fields, signed zero, extents,
rank collapse, and the first wet-cell one-ULP change. The bounds-checked
Fortran fixture is generated from the observer CALL statements copied
verbatim out of the patched ppsrc, then compiled with bounds checking and
minimal declarations/stubs. It executes every patched observer path at real
extents with distinct two-rank payloads, planted unfilled NaN edges, a `-0.0`
cell, and checks all 668 payload digests and finiteness. The patched sources
are preprocessed with the real O2 keys, syntax-checked against the real NEMO
modules, and the placement checks inspect that ppsrc rather than raw `.F90`.

In the operator path, same-program proof occurs after both builds and before
either launch. The disassembly census parses complete objdump instruction
lines, reports nonzero per-function equality and totals of 2928, 2231, and
309 for `icedyn_rhg_evp`, `icedyn_rdgrft`, and `icedyn`, respectively. It
examines only the Fortran flags used for those objects, so the arch file's
unrelated `%CFLAGS -O0` is not a refusal. A genuinely compiled extra
binary64 multiply/add makes the census fail in preflight. All preregistered
fold-sign labels, exchange-before-smooth, unmasked-smooth, parity-swap,
swapped-field, and wet-one-ULP admission controls are wired into retained-run
admission.

The committed provenance-bearing real excerpt is from the earlier
dynamics-off acquisition. Preflight therefore checks its provenance and also
tests the O5 log parser against an explicit synthetic dynamics-on overlay;
admission checks the complete real O5 `ocean.output`. The default path remains
read-only, while `--run` has RAM, disk, timeout, ERR and EXIT gates.

## Verification

- Read-only preflight: PASS, nine checklist sections, 12 landing controls and
  12 misplacement plants, the nonzero real-object census, and its injected-op
  plant; no NEMO build/run.
- O1/O2/O5 acquisition plus O5 model regressions: 61 passed, 1 deselected
  (the final O5 gate subset was rerun separately: 13 passed).
- Private-import ratchet: 6 passed against a complete archived checkout (the
  operator work clone omits two tracked ML sentinel files).
- Citation gate: all nine SI3 receipts and the default gate PASS with zero
  failures, unmapped citations, or stale full-map entries; gate tests: 17
  passed.
- NEMO acquisition result: ACQUISITION_NEEDED.
- independent review: operator (Claude)

## Choices

ASKED: three steps, one rank, the six preregistered inner frames, exact-input
strength plus aEVP admission, and operator-only acquisition.

UNASKED: the model now exposes a trace view of the same aEVP execution so the
preregistered inner frames can be scored when evidence arrives; the existing
final-state solver API and every configuration default are unchanged. No
NEMO agreement is claimed before operator evidence exists.

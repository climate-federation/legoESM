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

The writer produces direct unformatted stream writes with a 100-byte common
header and declared `jpi`, `jpj`, `jpl`, halo, decomposition, step, iteration,
rank, and SAS time-level metadata. There are 82 records for the three-step,
one-rank window; signed zero bits are not normalized. The SAS `Kmm` admission
is fixed to source-derived level 2 at all three steps.

## Admission and controls

The admission gate imports O1's wet-mask, ULP, signed-zero, and real-log
helpers and round 328's config-root-normalized same-program comparator. It
requires the producer commit to exist in this repository and match the run
manifest, verifies retained reference/record executable digests and every raw
record checksum, then runs strength and final aEVP replay under CPU production
JIT with fp64/libm policy. Strength, final stresses and velocities, SSH, metric
areas and reciprocals, and the no-redistribution entry/exit are required to be
0 ULP on both wet masks and full arrays; seam and fold maxima are reported.
All six intermediate delta/stress/velocity frames are checksum-bound and
schema-validated for every step and retained in the admission JSON.

Preflight plants cover each of the twelve observer landing sites, header,
truncation, checksum, SAS time level, and the first wet-cell one-ULP change.
The bounds-checked Fortran fixture executes every writer routine. In the
operator path, same-program proof occurs after both builds and before either
launch; an instruction census compares floating-point opcodes in all three
patched objects using the record build's actual non-`-O0` flags. The default
path is read-only, while `--run` has RAM, disk, timeout, ERR and EXIT gates and
retains refused evidence. `--admit-existing` is bound to the manifest producer
commit and copied binary digests.

## Verification

- Read-only preflight: PASS, eight checklist sections, 12 landing controls and
  12 misplacement plants; no NEMO build/run.
- Focused O5 acquisition tests: 9 passed.
- O1/O2/O5 acquisition regressions: 36 passed; private-import ratchet: 6 passed.
- Citation gate: all nine SI3 receipts PASS with zero failures, unmapped
  citations, or stale full-map entries.
- NEMO acquisition result: ACQUISITION_NEEDED.
- independent review: operator (Claude)

## Choices

ASKED: three steps, one rank, the six preregistered inner frames, exact-input
strength plus aEVP admission, and operator-only acquisition.

UNASKED: the intermediate frames are retained and schema/checksum admitted;
the existing public model API returns only the final aEVP state, so this round
does not claim a separately scored per-frame verdict before operator evidence
exists. No production model or configuration default changed.

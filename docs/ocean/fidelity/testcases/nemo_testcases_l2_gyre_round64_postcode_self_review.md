# Round 64 post-code self-review

Date: 2026-09-12

## Findings and disposition

- Corrected the task premise: round63 admission has zero changed consumed
  fields. The 16-count exact-record loss is the 16 missing kt=3--10 records,
  caused by the two-step horizon. All 26 raw changed field-record pairs and
  first-index magnitudes are retained in the receipt.
- Diffed all four R63 patched preprocessed units against R46. Every executable
  delta is a recorder call; the original physical statements are unchanged.
  Common consumed fields prove these calls inert. The raw halo/undefined
  changes are memory-layout-sensitive uninitialised payload, not model state.
- Kept the writer and patches unchanged. The R64 runner removes the horizon
  rewrite, copies the R46 namelist byte-for-byte, and verifies it with `cmp`.
  This is the smallest correction and introduces no second instrument.
- Did not add an admission plant: missing inherited records are an existing
  failure class with a direct non-vacuity test. The owned-bit plant remains in
  the operator runner.
- The provisional mode refuses a passing admission, any changed consumed
  field, any failure beyond missing inherited records, a mismatched record
  directory, a bad artifact hash/producer, a dirty/current-code stamp, non-fp64
  execution, failed exact content calibration, or model T/S inputs unequal to
  recorded Kbb.
- The provisional result is labelled NOT ADMISSIBLE in both JSON and stdout.
  It stops at the first unequal boundary and cannot emit a confirmed owner.

## Controls

- Shell syntax: pass.
- cpp plus gfortran syntax-only for writer and four patched units: exit 0,
  empty output; retained under `/tmp/gyre-r64-check-syntax.tzjjig`.
- Focused parser/runner/admission tests: 27 passed.
- Provisional metric: exact synthetic baseline; one-ULP consumed-value plant
  produces exactly one unequal cell.
- Citation gate: 13/13 citations pass; all map entries pass; shifted-line plant
  exits 1.
- No NEMO build, acquisition, or integration run.

## ASKED / UNASKED

| kind | item | disposition |
|---|---|---|
| ASKED | Admission diagnosis, preview, repaired operator runner | Complete |
| UNASKED | Production physics and round-62 candidates | Unchanged |
| UNASKED | Resolved configuration or carried state | Unchanged |

Open: the complete-FCT preview must be repeated against an R64 record whose
admission passes before any owner claim is made.

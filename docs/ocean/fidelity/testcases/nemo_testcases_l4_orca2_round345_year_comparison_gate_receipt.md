# ORCA2 round 345 receipt — Decision-188 year-comparison gate

Round: 345 (Codex). Base: `d3a0a329c692`. Preregistration commit:
`8bb1ab115`. Implementation commit: `304cad7ea`.
Disposition: **STOPPED_FOR_RECORD** — the gate is committed and the one extra
NEMO year requires operator acquisition.

## Result

Decision 188 is now executable month by month. The committed gate consumes a
candidate-vs-NEMO Decision-160 metric report and a NEMO-vs-perturbed-NEMO
envelope report. It applies the frozen temperature, salinity, SSH, streamfunction,
Drake, and MLD bars exactly; surface KE instead uses
`max(0.01, 3 * envelope_relative_RMS)` for the matching month. Complete year
reports must contain all 12 no-leap months, use the same NEMO baseline root,
and have identical day and saved-step windows. The original ORCA2X 30-day pilot
format remains explicitly admitted as one partial month.

No scientific year verdict is claimed in this round because the perturbed-NEMO
envelope does not exist yet. No NEMO process was run.

## NEMO arm and perturbation

The first non-bit source statement is that the active K0 cold-start branch
loads T/S data into `Kbb` through `dta_tsd`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/istate.f90:118`), while
the active damping branch calls that same reader for its interpolated target
(`ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/tradmp.f90:181`). The
round-345 tool therefore records, rather than conceals, that the staged
temperature file supplies both the cold-start state and the configured T/S
damping target.

The operator-only acquisition script reuses round 342's admitted observer-free
K0 binary and deck for one 2,920-step arm. Before launch it replaces only the
staged `data_1m_potential_temperature_nomask.nc` symlink by a regular NetCDF
copy. In every one of its 12 records, every finite wet
`votemper[:, 0, :, :]` float32 is advanced by exactly one representable value
toward positive infinity. This is one uniform deterministic wet-surface
pattern. Land, deeper levels, other variables, attributes, and every other
input remain bit-identical.

`initial_sst_ulp_perturbation.json` records the resolved source path, source,
perturbed, and domain SHA-256 digests, variable and dtype, selection and
direction, selected/changed counts, exact ULP distance, and dual NEMO use. The
admission gate re-derives the bitwise pattern and digests; a two-ULP corruption
plant fails. The run inherits round 342's real-flags census, source/build,
namelist/forcing, bounds-fixture, timeout, clean completion, retention,
manifest, and stub pipeline controls. It adds shell-syntax, generated-runner,
overwrite-refusal, exact-ULP, and one-arm admission checks.

Acquisition command:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_sst_ulp_acquisition/run.sh --run`

## Decision-188 gate controls

- A report exactly at the SST bar of `0.01 K` passes; the next representable
  float above it fails.
- KE relative RMS `0.02` passes against an envelope of `0.007`, because the
  monthly bar is `0.021`.
- KE relative RMS `0.011` fails against a zero envelope, proving the `0.01`
  floor applies; `0.01` passes.
- Candidate and envelope reports with different saved-step windows refuse.
- The committed ORCA2X pilot artifact at
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2x_rounds/round1/pilot/metrics/pilot_metrics.json`
  ran through the CLI as one partial month and returned PASS when used as both
  schema arms. That is a compatibility check only, not a chaos-envelope claim.

## Verification

- Round-345, round-342, and ORCA2X pilot metric tests: **18 / 18 PASS**.
- Ruff on all three new Python files: **PASS**.
- Decision-188 fixed-bar, 3x-envelope, floor, window, and exact-bit ULP plants:
  **PASS**.
- Certified-card fingerprint gate: **PASS**, all six card fingerprints
  unchanged.
- Full ten-file source ratchet: **12,380 passed, 4 skipped, 3 failed**. The
  failures are outside this round's diff: two audits encounter the pre-existing
  indentation error in round 336's `make_stub_records.py`, and the known
  shortwave parameter-spec failure. There is no round-345 ratchet failure.
- The clean-tree inherited acquisition preflight reached its MPI stub, then the
  managed sandbox refused PMIx listener socket creation (`errno=1`, rc 213).
  It did not start NEMO. The operator must run the same default preflight outside
  this socket-restricted sandbox before `--run`.

## Choices and review

- ASKED: option-1 fixed bars, monthly 3x KE envelope with a 1% floor, one
  observer-free perturbed K0 year, reuse of Decision-160 metrics, controls, and
  ORCA2X pilot compatibility.
- UNASKED: none.
- independent review: operator (Claude)

## Open

The operator runs the acquisition script, then evaluates the admitted
perturbed-NEMO year against the unperturbed NEMO baseline with round 342's
Decision-160 metric reporter. That report is the envelope input for the
committed Decision-188 gate; the existing legoESM-vs-NEMO report is the
candidate input.

# ORCA2 round 345 receipt — Decision-188 year-comparison gate

Round: 345 (Codex). Base: `d3a0a329c6925786815c4684593e88ff7d58167d`.
Review-resume base: `acef9cc04e314d10ce6e1bb7cbb795d32c5ffacb`.
Round-342 tool HEAD: `c983dc345a0b30c3ac5287d796986a069d981289`.
Admitted arm-A binary SHA-256:
`835f03b99a3c96e80d23f35c415d099fde292678ae30572e9c5412be4b67e16d`.
Admitted arm-A day-1 restart SHA-256 values:
`3fbed2281e66f7d51cd34247e799d1b76c8a102d36d5b64c0a3b80aba56dd2e3` and
`a5660f4c44b70a7cf7d70500b2df8e2d2145f08a205b266301c7312517413023`.
Disposition: **STOPPED_FOR_RECORD** — corrected preflight is pending; the
perturbed NEMO continuation remains operator acquisition.

## Result

Decision 188 remains executable month by month with its frozen fixed bars and
surface-KE rule `max(0.01, 3 * envelope_relative_RMS)`. The gate now refuses an
identically zero envelope and requires the envelope report to bind to an
admitted round-345 fp64 restart perturbation, admitted arm-A forcing/damping,
and the arm-A binary. The original ORCA2X 30-day schema remains a single
partial-month input.

No scientific year verdict is claimed because the perturbed-NEMO envelope has
not been acquired. No NEMO process was run.

## Corrected NEMO perturbation

The active restart reader loads `un`, `vn`, `tn`, and `sn` into `Kbb` and then
copies that state to `Kmm`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y/BLD/ppsrc/nemo/restart.f90:290-306`).
The acquisition tool copies admitted arm A's `kt = 8` restart pair, advances
only finite wet `tn[0, 0, :, :]` values by one float64 ULP toward positive
infinity, and starts the continuation at step 9. It keeps the perturbed pair as
day 1 and writes the remaining daily restart pairs through step 2920.

The perturbation manifest records both restart source and perturbed SHA-256
digests, variable, double dtype, selected/changed counts, direction, and exact
ULP distance. Its verifier compares every restart variable, dimension, and
attribute, proving all unselected state and metadata are bit-identical. The
cold-start temperature input is no longer modified; all forcing and damping
files are re-hashed against admitted arm A.

## Safety and retention

The shell claims a new arm with an atomic directory creation and a per-attempt
token. Quarantine requires that exact token. A target that already exists
refuses before the attempt is claimed, so admitted evidence from round 342 or a
previous round-345 attempt cannot be moved.

The stub scratch is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work`. The preflight covers a
successful pipeline, safe overwrite refusal, a missing-STOP refusal retaining
the binary and build snapshot, in-place admission, and direct re-admission of a
completed arm retained beneath `refused/`.

Acquisition command:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_sst_ulp_acquisition/run.sh --run`

Re-admit a completed retained arm without deleting it:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_sst_ulp_acquisition/run.sh --admit-existing /absolute/path/to/refused/<retained-arm>`

## Integrity

The base and round-342 git SHAs above are full values obtained with
`git rev-parse`. The document integrity test rejects abbreviated or unresolved
git SHAs and resolves citations against this clone or the round-342 clone. File
digests in the eventual acquisition manifest and receipt are emitted by the
gate from `sha256sum`-equivalent byte hashing and immediately re-hashed.

## Verification

Verification and the default preflight will be recorded after the corrected
implementation is committed on a clean worktree.

## Choices and review

- ASKED: option-1 fixed bars; monthly 3x KE envelope with 1% floor; perturb the
  admitted day-1 NEMO double-precision restart SST by one ULP; leave forcing and
  damping untouched; compare from day 1; refuse zero/non-perturbed envelopes.
- UNASKED: none.
- independent review: operator (Claude Sonnet; HOLD under repair)

## Open

Run the corrected clean-tree preflight. If it passes, the operator runs the
acquisition command, evaluates admitted arm A versus the perturbed continuation
with the Decision-160 reporter, and supplies that report plus the round-345
admission JSON to the Decision-188 gate.

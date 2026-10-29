# ORCA2 round 345 receipt — Decision-188 year-comparison gate

Round: 345 (Codex). Base: `d3a0a329c6925786815c4684593e88ff7d58167d`.
Review-resume base: `acef9cc04e314d10ce6e1bb7cbb795d32c5ffacb`.
Round-342 tool HEAD: `c983dc345a0b30c3ac5287d796986a069d981289`.
Admitted arm-A binary SHA-256:
`835f03b99a3c96e80d23f35c415d099fde292678ae30572e9c5412be4b67e16d`.
Admitted arm-A day-1 restart SHA-256 values:
`3fbed2281e66f7d51cd34247e799d1b76c8a102d36d5b64c0a3b80aba56dd2e3` and
`a5660f4c44b70a7cf7d70500b2df8e2d2145f08a205b266301c7312517413023`.
Deterministic perturbed day-1 restart SHA-256 values:
`bb25179365e776ae73e6ead9d2bad42e222f2094f687c22ede3b0eb3bcae1f01` and
`0279bb4021ef1b557e178ef01b5c4cac732479102404871b557b984433fff8cf`.
Disposition: **STOPPED_FOR_RECORD** — the corrected round-345 pipeline is
verified; the full preflight must be repeated outside this sandbox before the
control and perturbed NEMO continuations are acquired by the operator.

## Result

Decision 188 remains executable month by month with its frozen fixed bars and
surface-KE rule `max(0.01, 3 * envelope_relative_RMS)`. The gate now requires a
strictly positive envelope in every evaluated month and binds the report to an
admitted round-345 fp64 restart perturbation, admitted arm-A forcing/damping,
and the arm-A binary. Before using that envelope, it also requires a second NEMO
continuation from the exact same unperturbed restart to be byte-identical to arm
A at all 365 daily restart pairs. The original ORCA2X 30-day schema remains a
single partial-month input.

No scientific year verdict is claimed because the perturbed-NEMO envelope has
not been acquired. No NEMO process was run.

## Corrected NEMO perturbation

The active restart reader loads `un`, `vn`, `tn`, and `sn` into `Kbb` and then
copies that state to `Kmm`
(`ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y/BLD/ppsrc/nemo/restart.f90:291-296,311-316`).
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

`--admit-existing` validates producer-commit ancestry and relative manifest and
binary digests. A failing in-place admission quarantines both new arms and logs
the refusal; a failure already in `refused/` stays there. A passed retained pair
is promoted to the canonical paths, and the passed admission JSON is written
only under the run root.

The stub scratch is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work`. The preflight covers a
successful paired pipeline, safe overwrite refusal, a missing-STOP refusal
retaining the binary and build snapshot, a failing in-place admission trap,
ancestry refusal, and promotion of a passed pair from `refused/`.

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

The safety/restart implementation is committed as
`4e175e60e1223eb2f98188d0f102ac037aa27ceb`; the retained-arm relocation fix is
`95eb6b45a12508e4b2d32eef1f89a3af938a76cc`; and the paired-control, admission,
and all-month envelope repair is
`8e91ec01953e048586c296094361d9fd37df346c`. These values were copied from
`git rev-parse`, and the document gate resolves every quoted git object.

Focused round-345, inherited round-342 year, and ORCA2X pilot tests pass 23/23.
The seven Decision-188 controls fire, including independent refusal of a zero
month, a non-perturbed envelope, and a missing restart control. A real-arm staging probe verifies
16,433 changed wet SST values, 59 unchanged forcing/deck inputs, the admitted
binary, and the step-9 restart deck. A direct end-to-end stub flow passes
staging, completion, admission, in-place re-admission, failed-admission
quarantine, and promotion from `refused/`. A separate existing-target probe
refuses while retaining both targets in place and creating no refused copy.

The clean default preflight was re-run at
`b8db53c7d2fc60f1aa2615d83f6dcb742b1600bd` with evidence rooted under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/r345-resume2-preflight`.
Its inherited round-342 MPI stub stops
before the round-345 stub because this managed sandbox denies the Open MPI
PMIx listener socket (launcher exit 213). The same sandbox failure occurs for
`mpirun --version`; it is not a scientific or admission result. The
round-345-specific paired pipeline above was therefore exercised directly
under `work/` with four control restart files compared in the two-day stub, a
failing in-place admission quarantined, and both retained arms promoted, but
the committed default preflight remains fail-closed and must be run by the
operator in the acquisition environment.

The per-receipt citation gate passes its one citation with zero failures and
its shifted-line plant refuses. The exact default ORCA2 landing battery passes
155/155 in 1,104.74 seconds; the explicit private-import ratchet passes 6/6.

## Choices and review

- ASKED: option-1 fixed bars; monthly 3x KE envelope with 1% floor; perturb the
  admitted day-1 NEMO double-precision restart SST by one ULP; leave forcing and
  damping untouched; compare from day 1; refuse zero/non-perturbed envelopes;
  require the exact unperturbed restart continuation before using the envelope.
- UNASKED: none.
- independent review: Claude Sonnet RUN confirmed the prior repair; Argon HOLD
  received; all four findings are repaired here; operator re-review remains open

## Open

The operator runs the acquisition command (which first runs the same
fail-closed preflight), acquires both the exact restart control and perturbed
continuation, evaluates admitted arm A versus the perturbed continuation with
the Decision-160 reporter, and supplies that report plus the paired round-345
admission JSON to the Decision-188 gate.

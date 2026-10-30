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
Disposition: **STOPPED_FOR_RECORD** — both continuations completed with `STOP
0` and are retained under `round345/s4_year/refused/`; the operator must run the
reviewed re-admission command after this comparator repair lands.

## Result

Decision 188 remains executable month by month with its frozen fixed bars and
surface-KE rule `max(0.01, 3 * envelope_relative_RMS)`. The gate now requires a
strictly positive envelope in every evaluated month and binds the report to an
admitted round-345 fp64 restart perturbation, admitted arm-A forcing/damping,
and the arm-A binary. Before using that envelope, it also requires a second NEMO
continuation from the exact same unperturbed restart to have bitwise-identical
data for every common variable in both ranks at all 364 continuation-output
restart pairs. The original ORCA2X 30-day schema remains a single partial-month
input.

No scientific year verdict is claimed because the completed pair is not yet
admitted or scored. This repair did not run NEMO and killed no process.

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

## Restart continuation control correction

The operator's reviewed acquisition completed both arms with `STOP 0`, then the
old whole-file SHA-256 comparison refused at
`ORCA2_00000016_restart_0000.nc`. Both complete arms were retained at
`round345/s4_year/refused/20261030T023228Z_orca2_k0_year1_restart_control_np2`
and
`round345/s4_year/refused/20261030T023228Z_orca2_k0_year1_sst_plus_1ulp_np2`.

The operator's committed measurement script
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/pending/r345_restart_data_compare.py`
checked all 728 output restart files (`kt = 16..2920`, stride 8, ranks 0 and 1).
All 39 variables present in both arms had equal dtype, shape, and raw data bytes:
28,392 variable-data comparisons, zero unequal. The exact inventory difference
in every file was `DELAY_re8_fwb2` and `DELAY_im8_fwb2`, present only in arm A.

This difference is NEMO restart bookkeeping, not model-state drift. The
`nn_fwb = 2` annual freshwater-budget branch labels its delayed global sum
`fwb2` (`src/OCE/SBC/sbcfwb.F90:342-365`). Delayed communication registers the
label and allocates its output buffer only when that call executes
(`src/OCE/LBC/mpp_allreduce_generic.h90:113-149`). On restart, NEMO reads the
delayed list and complex buffer from the `DELAY_re8_fwb2` and
`DELAY_im8_fwb2` variables carried by arm A's `kt = 8` restart and both staged
continuation inputs (`src/OCE/IOM/iom.F90:1925-1966`). The restart writer skips
a delayed buffer that was read but not used in the restarted run
(`src/OCE/IOM/iom.F90:1972-1984`, specifically line 1974). The next annual
`fwb2` delayed call is at `kt = 2921` (`src/OCE/SBC/sbcfwb.F90:342-365`), after
this continuation ends, so the buffer cannot affect state and is not written
in the `kt = 16..2920` outputs.

The repaired control therefore compares the 728 continuation outputs variable
by variable. It requires exactly those two variables to be arm-A-only, requires
no control-only variable, and compares every common variable's dtype, shape, and
raw C-order data bytes. Any other inventory difference or any unequal byte
refuses. Per-file SHA-256 values remain provenance fields, not equality tests.
Plants prove refusal after one byte is flipped in `tn`, after an unknown variable
is added, and after a non-delay variable is omitted.

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

The refused-arm route validates both retained manifests and both scientific
arms before moving either path, then rechecks the canonical pair and writes the
admission JSON only under the run root. The two retained operator arms above
match the route's exact naming and pairing rules; either retained-arm path may
be supplied to `--admit-existing`.

The operator's resume-3 preflight REFUSED in the retained control arm. This
CONFIRMED two stub defects that invalidate the earlier direct-stub claim: both
MPI processes entered the all-ranks writer concurrently, and the restart-arm
writer overwrote the staged `kt = 8` restart that the continuation reads. The
stub now requires an unambiguous launcher rank, gives shared outputs and the
clean-stop signal to rank 0 only, and gives each mesh/restart file to its own
rank. In both restart arms it hashes the staged `kt = 8` input before and after,
leaves it unchanged, and writes only cadence outputs from `kt = 16` through
`kt = 2920`. A no-MPI regression launches the generated executable as two
concurrent processes with Open MPI and PMIx rank variables for both the control
and perturbed arms; it separately proves rank 1 cannot change shared/rank-0
files and applies the same inventory, completion, restart-content, control, and
perturbation checks used by admission.

Acquisition command:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_sst_ulp_acquisition/run.sh --run`

Re-admit the completed retained pair without deleting or re-running it:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_sst_ulp_acquisition/run.sh --admit-existing /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year/refused/20261030T023228Z_orca2_k0_year1_sst_plus_1ulp_np2`

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
The rank-local, input-preserving stub repair is
`527dd75f1e043278d41693dc11a0cd9781cd65ec`.

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

Resume 3 passes the complete focused round-342/345 battery (18/18), including
the new concurrent rank-env regression for both arms. The receipt citation gate
passes 1/1 with zero failures and its shifted-line plant refuses. The new test
reproduces both confirmed failure mechanisms against the pre-repair writer
without invoking MPI or NEMO. The operator must re-run the complete
control-plus-perturbed preflight because this sandbox still cannot launch
`mpirun`.

The per-receipt citation gate passes its one citation with zero failures and
its shifted-line plant refuses. The exact default ORCA2 landing battery passes
155/155 in 1,104.74 seconds; the explicit private-import ratchet passes 6/6.

Resume 5 is committed as `96d29d324f2c767932bddff0950a4cf15f89c4c7`.
The repaired comparator was run read-only on the actual retained control: all
728 restart files passed, with 28,392 common-variable data comparisons, zero
unequal, no control-only variables, and exactly the two required arm-A-only
delay buffers. The fp64 focused round-342/345/ORCA2X battery passes 26/26 in
87.92 seconds; round 345 alone passes 12/12, including both re-admission routes
and the three requested variable-data plants. The per-receipt citation gate
passes 5 citations with zero failures/unmapped entries, and its shifted-line
plant exits 1. The full citation-gate suite passes 20/20. The private-import
ratchet passes 5/6; its sole failure is the pre-existing missing-sentinel census
in five unrelated package files, and this resume changes no package file.

## Choices and review

- ASKED: option-1 fixed bars; monthly 3x KE envelope with 1% floor; perturb the
  admitted day-1 NEMO double-precision restart SST by one ULP; leave forcing and
  damping untouched; compare from day 1; refuse zero/non-perturbed envelopes;
  require the unperturbed restart continuation before using the envelope;
  compare all common restart-variable data bitwise and require exactly the two
  arm-A-only `fwb2` delay buffers measured by the operator.
- UNASKED: none.
- independent review: Claude Sonnet RUN confirmed the prior repair; Argon HOLD
  received; all four findings are repaired here; operator re-review remains open

## Open

The operator re-admits the retained pair with the command above. No NEMO rerun
is needed. After admission, the operator evaluates arm A versus the perturbed
continuation with the Decision-160 reporter and supplies that report plus the
paired round-345 admission JSON to the Decision-188 gate.

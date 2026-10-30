# ORCA2 round 345 receipt — Decision-188 year-comparison gate

Round: 345 (Codex). Base: `d3a0a329c6925786815c4684593e88ff7d58167d`.
Review-resume base: `fe98377a856078fd9cef30d842061d9f2ce49628`
(pre-rebase `acef9cc04`).
Round-342 tool HEAD: `c983dc345a0b30c3ac5287d796986a069d981289`.
Admitted arm-A binary SHA-256:
`835f03b99a3c96e80d23f35c415d099fde292678ae30572e9c5412be4b67e16d`.
Admitted arm-A day-1 restart SHA-256 values:
`3fbed2281e66f7d51cd34247e799d1b76c8a102d36d5b64c0a3b80aba56dd2e3` and
`a5660f4c44b70a7cf7d70500b2df8e2d2145f08a205b266301c7312517413023`.
Deterministic perturbed day-1 restart SHA-256 values:
`bb25179365e776ae73e6ead9d2bad42e222f2094f687c22ede3b0eb3bcae1f01` and
`0279bb4021ef1b557e178ef01b5c4cac732479102404871b557b984433fff8cf`.
Disposition: **PASS_DECISION_188** — both real continuations are admitted;
the year comparison passes all 12 months with a nonzero envelope in all 12.

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

Resume 7 closes the scientific year verdict: **PASS**, 12/12 months, with a
strictly positive KE envelope in every evaluated month. This work did not run
NEMO and killed no process.

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
`88a294214393b7a1da74f06db1be49a40226de7c` (pre-rebase `4e175e60e`); the
retained-arm relocation fix is
`eb28fac1b7026a93c718d2a64fd82b7bf8f0aca2` (pre-rebase `95eb6b45a`); and the
paired-control, admission, and all-month envelope repair is
`15af7c3b09014792848a99f5cfbacbe28fb29ed5` (pre-rebase `8e91ec019`). These
values were copied from `git rev-parse`, and the document gate resolves every
quoted git object.
The rank-local, input-preserving stub repair is
`2a6c1d842e83a326028d6843cb10f57d81666034` (pre-rebase `527dd75f1`).

Focused round-345, inherited round-342 year, and ORCA2X pilot tests pass 23/23.
The seven Decision-188 controls fire, including independent refusal of a zero
month, a non-perturbed envelope, and a missing restart control. A real-arm staging probe verifies
16,433 changed wet SST values, 59 unchanged forcing/deck inputs, the admitted
binary, and the step-9 restart deck. A direct end-to-end stub flow passes
staging, completion, admission, in-place re-admission, failed-admission
quarantine, and promotion from `refused/`. A separate existing-target probe
refuses while retaining both targets in place and creating no refused copy.

The clean default preflight was re-run at
`c023d7ababcc494e785ff7798bc321e2d135ef1a` (pre-rebase `b8db53c7d`) with
evidence rooted under
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

Resume 5 is committed as `8f4de8ac7ac7f9f35a7a79e9bdd642d4cf5fc640`
(pre-rebase `96d29d324`).
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

## Resume 7: admitted envelope and Decision-188 result

### Exact commands and inputs

All Python commands that can reach the reporter import graph were pinned to CPU.
The operator admitted the retained real pair with:

```bash
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_sst_ulp_acquisition/run.sh --admit-existing /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year/refused/20261030T023228Z_orca2_k0_year1_sst_plus_1ulp_np2
```

It returned zero and promoted both arms. The metric command was:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu .venv/bin/python scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_metrics.py --nemo-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2 --candidate-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step --nemo-twin-b-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year/orca2_k0_year1_sst_plus_1ulp_np2 --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics
```

The Decision-188 command was:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu .venv/bin/python scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_decision188_gate.py --candidate-metrics /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics/round342_s4_candidate_vs_nemo_metrics.json --envelope-metrics /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics/round342_s4_nemo_twin_b_vs_a_metrics.json --envelope-admission /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year/round345_s4_year_sst_ulp_admission.json --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics/decision188_verdict.json
```

Inputs were the admitted round-342 arm A, the completed round-342 legoESM K0
year, the admitted round-345 one-ULP NEMO continuation, and the paired
round-345 admission JSON named by those commands. The admission status is
`PASS_R345_S4_YEAR_SST_PLUS_ONE_ULP_ARM`: 16,433 selected wet surface `tn`
values changed by exactly one float64 ULP toward positive infinity; all
unselected restart values and metadata are bit-identical; all 59 deck,
forcing, and damping inputs match arm A; the binary matches arm A; and the
unperturbed continuation control compares 728 files with zero unequal common
variable arrays.

### Computed provenance

The following values were computed with `sha256sum -- <path>` rather than
copied from prose:

- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year/round345_s4_year_sst_ulp_admission.json` SHA-256 `e98db5c288964138ac97ae9d80444d8f83e14d24bfa3607b488f084c8077d494`.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics/round342_s4_candidate_vs_nemo_metrics.json` SHA-256 `6be9341fea3d89edbf735661e0b5e7bdd7fc72faa4c3b08efee3a30a1ccf5ac7`.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics/round342_s4_nemo_twin_b_vs_a_metrics.json` SHA-256 `d191184fab8e4de011b9ee691dcad21c43c115a1f58dea04cbb78a6184509640`.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics/decision188_verdict.json` SHA-256 `a14601770432d3480743f890b22145d03c91ca1dc1a92830fa3de94edd8383b0`.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics/eval.log` SHA-256 `ca74a5a05f13d0c99e3286cdb7c1be1e18ef972514433dd85bbbe94c76157cc2`.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345/year1_envelope_metrics/gate.log` SHA-256 `a14601770432d3480743f890b22145d03c91ca1dc1a92830fa3de94edd8383b0`.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/year1_metrics_comparison/round342_s4_candidate_vs_nemo_metrics.json` SHA-256 `7ec864d4da7fb28a67671ab895405e43a00124e5ef5ed8a497b1d1ccbb4c85ec`.
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2x_rounds/round1/pilot/metrics/pilot_metrics.json` SHA-256 `26cf06800b1a7e08535722aa83540ad2f415a5ce5ac4d415f84303db1b758308`.

The whole recomputed candidate JSON is **not** byte-identical to the earlier
round-342 JSON: `cmp -s` returns 1. The only differences are the regenerated
figure path/digest and the worktree commit/root provenance. After deleting
exactly `daily_sst_rms_curve_figure` and `worktree` from both JSON objects and
sorting with `jq -S`, `cmp -s` returns zero. Thus the scientific payload is
byte-identical; the complete provenance-bearing files are correctly different.

Full values computed with `git rev-parse` are: reporter execution HEAD
`d2ba162780b9f2b48eaeb640d9a524d8a5c0191d`, imported round-342 tool HEAD
`c983dc345a0b30c3ac5287d796986a069d981289`, post-rebase lane base
`bf3ad4b42442900686648d9c432ed4e247e57663`, and post-rebase pre-receipt HEAD
`02c37e14463f0d84883784131e418555204ec435`.

### Decision-188 month tables

Verdict: **PASS**, 12/12 complete months, and a strictly positive KE envelope
in all 12. `C/E/bar` means legoESM candidate / one-ULP NEMO envelope / frozen
option-1 bar. Under-ice SST/SSS are reporter diagnostics but are not registered
Decision-188 gate rows.

| Month | KE candidate rel RMS | KE envelope rel RMS | KE bar | SST candidate RMS (K) | SST envelope RMS (K) | SST bar (K) |
|---|---:|---:|---:|---:|---:|---:|
| `M01_Jan` | `0.0000e+00` | `1.8632e-10` | `1.0000e-02` | `0.0000e+00` | `8.8920e-11` | `1.0000e-02` |
| `M02_Feb` | `9.5620e-11` | `2.8232e-10` | `1.0000e-02` | `5.4642e-11` | `4.3331e-10` | `1.0000e-02` |
| `M03_Mar` | `8.1798e-10` | `1.2008e-09` | `1.0000e-02` | `2.1069e-10` | `4.5284e-10` | `1.0000e-02` |
| `M04_Apr` | `3.2995e-09` | `6.4596e-09` | `1.0000e-02` | `9.2933e-08` | `1.2333e-09` | `1.0000e-02` |
| `M05_May` | `2.5481e-09` | `1.3476e-08` | `1.0000e-02` | `7.2534e-08` | `2.1616e-09` | `1.0000e-02` |
| `M06_Jun` | `9.7334e-09` | `4.8931e-08` | `1.0000e-02` | `3.2197e-08` | `5.9077e-09` | `1.0000e-02` |
| `M07_Jul` | `1.1447e-07` | `3.7440e-07` | `1.0000e-02` | `2.2401e-08` | `2.6684e-08` | `1.0000e-02` |
| `M08_Aug` | `7.6776e-07` | `2.6294e-06` | `1.0000e-02` | `7.2797e-08` | `1.9897e-07` | `1.0000e-02` |
| `M09_Sep` | `2.6068e-05` | `5.4988e-03` | `1.6496e-02` | `1.2176e-06` | `3.5322e-04` | `1.0000e-02` |
| `M10_Oct` | `1.4613e-03` | `3.3248e-02` | `9.9745e-02` | `7.0776e-05` | `1.9268e-03` | `1.0000e-02` |
| `M11_Nov` | `1.2823e-02` | `3.4522e-02` | `1.0357e-01` | `8.1962e-04` | `3.0700e-03` | `1.0000e-02` |
| `M12_Dec` | `3.1441e-02` | `4.6904e-02` | `1.4071e-01` | `1.8368e-03` | `4.5631e-03` | `1.0000e-02` |

| Month | T 0-100m C/E/bar (K) | T 100-1000m C/E/bar (K) | T >1000m C/E/bar (K) |
|---|---:|---:|---:|
| `M01_Jan` | `0.0000e+00 / 1.1556e-10 / 1.0000e-02` | `0.0000e+00 / 5.3429e-11 / 1.0000e-02` | `0.0000e+00 / 3.8415e-13 / 1.0000e-02` |
| `M02_Feb` | `6.7943e-11 / 4.8057e-10 / 1.0000e-02` | `3.5125e-11 / 1.6903e-10 / 1.0000e-02` | `1.9818e-13 / 1.0794e-12 / 1.0000e-02` |
| `M03_Mar` | `2.4682e-10 / 5.8958e-10 / 1.0000e-02` | `1.4116e-10 / 2.1240e-10 / 1.0000e-02` | `1.1338e-12 / 3.3653e-12 / 1.0000e-02` |
| `M04_Apr` | `7.7286e-08 / 3.4558e-09 / 1.0000e-02` | `3.9743e-08 / 3.0074e-10 / 1.0000e-02` | `3.0567e-11 / 7.9998e-12 / 1.0000e-02` |
| `M05_May` | `6.6198e-08 / 6.8736e-09 / 1.0000e-02` | `2.5164e-08 / 1.1327e-09 / 1.0000e-02` | `9.7508e-11 / 2.7710e-11 / 1.0000e-02` |
| `M06_Jun` | `3.1558e-08 / 2.5346e-08 / 1.0000e-02` | `1.4221e-08 / 9.3774e-09 / 1.0000e-02` | `9.3142e-11 / 1.6842e-10 / 1.0000e-02` |
| `M07_Jul` | `3.9706e-08 / 1.1890e-07 / 1.0000e-02` | `1.5554e-08 / 3.0991e-08 / 1.0000e-02` | `1.7949e-10 / 6.4499e-10 / 1.0000e-02` |
| `M08_Aug` | `2.7153e-07 / 2.7842e-06 / 1.0000e-02` | `7.2004e-08 / 6.9773e-07 / 1.0000e-02` | `9.0244e-10 / 6.7089e-09 / 1.0000e-02` |
| `M09_Sep` | `2.5010e-06 / 4.7883e-04 / 1.0000e-02` | `1.0038e-06 / 9.5081e-05 / 1.0000e-02` | `1.7161e-08 / 3.0627e-06 / 1.0000e-02` |
| `M10_Oct` | `1.1113e-04 / 1.9645e-03 / 1.0000e-02` | `3.8344e-05 / 4.6575e-04 / 1.0000e-02` | `6.7040e-07 / 1.3231e-05 / 1.0000e-02` |
| `M11_Nov` | `7.4306e-04 / 3.2809e-03 / 1.0000e-02` | `2.3959e-04 / 1.1762e-03 / 1.0000e-02` | `5.2469e-06 / 2.5080e-05 / 1.0000e-02` |
| `M12_Dec` | `1.7365e-03 / 4.5588e-03 / 1.0000e-02` | `5.7897e-04 / 1.7501e-03 / 1.0000e-02` | `1.2812e-05 / 3.8155e-05 / 1.0000e-02` |

| Month | SSS C/E/bar (psu) | S 0-100m C/E/bar (psu) | S 100-1000m C/E/bar (psu) | S >1000m C/E/bar (psu) |
|---|---:|---:|---:|---:|
| `M01_Jan` | `0.0000e+00 / 6.8828e-11 / 2.0000e-03` | `0.0000e+00 / 5.8476e-11 / 2.0000e-03` | `0.0000e+00 / 8.9487e-12 / 2.0000e-03` | `0.0000e+00 / 4.8381e-14 / 2.0000e-03` |
| `M02_Feb` | `2.5010e-11 / 1.1488e-10 / 2.0000e-03` | `2.1400e-11 / 1.0567e-10 / 2.0000e-03` | `6.5279e-12 / 2.7211e-11 / 2.0000e-03` | `2.3595e-14 / 9.7905e-14 / 2.0000e-03` |
| `M03_Mar` | `8.9433e-11 / 1.1638e-10 / 2.0000e-03` | `7.3257e-11 / 1.0602e-10 / 2.0000e-03` | `2.4882e-11 / 3.4433e-11 / 2.0000e-03` | `1.2945e-13 / 1.8958e-13 / 2.0000e-03` |
| `M04_Apr` | `4.5327e-09 / 3.5089e-10 / 2.0000e-03` | `3.5933e-09 / 5.3120e-10 / 2.0000e-03` | `1.4380e-09 / 3.6112e-11 / 2.0000e-03` | `6.3281e-13 / 6.4839e-13 / 2.0000e-03` |
| `M05_May` | `3.8584e-09 / 8.3422e-10 / 2.0000e-03` | `3.5144e-09 / 8.5858e-10 / 2.0000e-03` | `9.9537e-10 / 7.1081e-11 / 2.0000e-03` | `2.3168e-12 / 1.9396e-12 / 2.0000e-03` |
| `M06_Jun` | `2.1403e-09 / 1.1387e-09 / 2.0000e-03` | `2.0453e-09 / 2.2715e-09 / 2.0000e-03` | `6.0585e-10 / 7.4226e-10 / 2.0000e-03` | `3.5091e-12 / 5.7000e-12 / 2.0000e-03` |
| `M07_Jul` | `2.0273e-09 / 3.6650e-09 / 2.0000e-03` | `3.4018e-09 / 9.8268e-09 / 2.0000e-03` | `1.0347e-09 / 2.5272e-09 / 2.0000e-03` | `6.2572e-12 / 2.1206e-11 / 2.0000e-03` |
| `M08_Aug` | `8.8039e-09 / 2.6979e-08 / 2.0000e-03` | `2.2045e-08 / 6.2435e-07 / 2.0000e-03` | `6.0199e-09 / 7.9964e-08 / 2.0000e-03` | `2.8761e-11 / 1.8783e-10 / 2.0000e-03` |
| `M09_Sep` | `1.7826e-07 / 2.7968e-05 / 2.0000e-03` | `3.3050e-07 / 3.6094e-05 / 2.0000e-03` | `8.1153e-08 / 5.4990e-06 / 2.0000e-03` | `5.8044e-10 / 1.0558e-07 / 2.0000e-03` |
| `M10_Oct` | `1.4015e-05 / 1.5586e-04 / 2.0000e-03` | `2.1845e-05 / 1.6193e-04 / 2.0000e-03` | `3.5027e-06 / 2.3368e-05 / 2.0000e-03` | `2.1885e-08 / 4.5088e-07 / 2.0000e-03` |
| `M11_Nov` | `1.0454e-04 / 2.9473e-04 / 2.0000e-03` | `9.2015e-05 / 3.1090e-04 / 2.0000e-03` | `2.1122e-05 / 7.0412e-05 / 2.0000e-03` | `1.7708e-07 / 8.6912e-07 / 2.0000e-03` |
| `M12_Dec` | `2.0464e-04 / 4.5864e-04 / 2.0000e-03` | `1.8289e-04 / 4.3029e-04 / 2.0000e-03` | `4.0078e-05 / 9.1877e-05 / 2.0000e-03` | `4.3713e-07 / 1.2935e-06 / 2.0000e-03` |

| Month | SSH C/E/bar (m) | PSI RMS C/E/bar (Sv) | PSI corr C/E/min | Drake abs diff C/E/bar (Sv) | MLD C/E/bar (m) |
|---|---:|---:|---:|---:|---:|
| `M01_Jan` | `0.0000e+00 / 4.0477e-12 / 1.0000e-03` | `0.0000e+00 / 3.0155e-10 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `0.0000e+00 / 8.3276e-12 / 5.0000e-01` | `0.0000e+00 / 1.6217e-12 / 1.0000e+00` |
| `M02_Feb` | `2.1439e-12 / 9.0496e-12 / 1.0000e-03` | `1.7408e-10 / 1.2040e-09 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `1.3046e-11 / 1.1426e-11 / 5.0000e-01` | `1.0742e-12 / 4.8856e-12 / 1.0000e+00` |
| `M03_Mar` | `5.7041e-12 / 7.9614e-12 / 1.0000e-03` | `3.9657e-10 / 6.1146e-10 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `1.2278e-11 / 1.3017e-11 / 5.0000e-01` | `2.6923e-12 / 4.0003e-12 / 1.0000e+00` |
| `M04_Apr` | `2.5839e-10 / 4.7845e-11 / 1.0000e-03` | `1.6213e-08 / 3.4110e-09 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `3.5183e-10 / 2.3320e-10 / 5.0000e-01` | `5.6797e-11 / 1.8938e-11 / 1.0000e+00` |
| `M05_May` | `5.1005e-10 / 5.1188e-11 / 1.0000e-03` | `1.2448e-08 / 8.3308e-09 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `2.9840e-10 / 1.1059e-10 / 5.0000e-01` | `5.2974e-11 / 1.6759e-11 / 1.0000e+00` |
| `M06_Jun` | `3.9972e-10 / 1.4616e-10 / 1.0000e-03` | `9.2615e-09 / 2.4952e-08 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `1.6408e-10 / 1.0638e-10 / 5.0000e-01` | `2.8645e-11 / 8.4624e-12 / 1.0000e+00` |
| `M07_Jul` | `7.4963e-10 / 7.4250e-10 / 1.0000e-03` | `1.3547e-07 / 1.2362e-07 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `1.0764e-08 / 1.2616e-09 / 5.0000e-01` | `3.7524e-10 / 4.1084e-11 / 1.0000e+00` |
| `M08_Aug` | `1.5256e-09 / 7.3214e-09 / 1.0000e-03` | `1.9063e-07 / 7.6416e-07 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `2.2698e-10 / 2.6584e-08 / 5.0000e-01` | `1.1965e-10 / 9.5355e-11 / 1.0000e+00` |
| `M09_Sep` | `2.8938e-08 / 3.8971e-06 / 1.0000e-03` | `4.3547e-06 / 2.0839e-04 / 1.0000e-01` | `1.000000000 / 1.000000000 / 0.999` | `5.1503e-08 / 6.6768e-07 / 5.0000e-01` | `2.1952e-10 / 3.6437e-02 / 1.0000e+00` |
| `M10_Oct` | `9.4832e-07 / 1.5137e-05 / 1.0000e-03` | `7.1743e-05 / 1.7883e-03 / 1.0000e-01` | `1.000000000 / 0.999999999 / 0.999` | `2.0399e-06 / 8.9886e-06 / 5.0000e-01` | `3.5275e-02 / 1.7816e-01 / 1.0000e+00` |
| `M11_Nov` | `6.8366e-06 / 2.8338e-05 / 1.0000e-03` | `7.5224e-04 / 4.3799e-03 / 1.0000e-01` | `1.000000000 / 0.999999996 / 0.999` | `1.8753e-06 / 1.4911e-05 / 5.0000e-01` | `1.2348e-01 / 2.3075e-01 / 1.0000e+00` |
| `M12_Dec` | `1.6333e-05 / 4.7813e-05 / 1.0000e-03` | `2.6756e-03 / 9.1371e-03 / 1.0000e-01` | `0.999999998 / 0.999999981 / 0.999` | `3.0384e-05 / 8.9517e-05 / 5.0000e-01` | `2.0259e-01 / 3.6758e-01 / 1.0000e+00` |

### Interpretation, ORCA2X pilot, and rebase

- **CONFIRMED:** from September through December, the one-ULP NEMO twin's KE
  relative RMS and SST RMS both exceed legoESM's. In December, SST RMS is
  `4.5631e-03 K` (4.6 mK) for the twin versus `1.8368e-03 K` (1.8 mK) for
  legoESM; KE relative RMS is `4.6904e-02` (4.7%) versus `3.1441e-02` (3.1%).
- **PLAUSIBLE, not confirmed:** the twin separates faster because the one-ULP
  perturbation touches all 16,433 wet surface cells simultaneously. No
  localization or dose-response experiment was run, so this is not causal.
- **ORCA2X pilot not scored:** the existing 30-day candidate report is
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2x_rounds/round1/pilot/metrics/pilot_metrics.json`
  with format `orca2x-r1-pilot-metrics-v1`, but no matching ORCA2X perturbed-
  versus-baseline envelope report of that format exists. That is the missing
  gate input; substituting the ORCA2 envelope would be a cross-grid confound.
- **Rebase:** fetched the current ORCA2 lane and rebased the 33-commit round
  stack from `d2ba162780b9f2b48eaeb640d9a524d8a5c0191d` onto
  `bf3ad4b42442900686648d9c432ed4e247e57663`. Replayed pre-receipt HEAD is
  `02c37e14463f0d84883784131e418555204ec435`. Conflict hunks: **none**.

## Producer commits pinned under refs/keep/

These producer identities are immutable fields in admitted evidence and must
remain verbatim. Before landing, pin each object in the ORCA2 lane at the named
keep ref:

- `d6b2a3342c0c006cacbcd3289d5598f7d8d893ce` at
  `refs/keep/orca2_r345_d6b2a3342` (admitted continuation manifests and
  admission JSON).
- `d2ba162780b9f2b48eaeb640d9a524d8a5c0191d` at
  `refs/keep/orca2_r345_d2ba16278` (candidate and envelope metric JSON
  worktree provenance).
- `c983dc345a0b30c3ac5287d796986a069d981289` at
  `refs/keep/orca2_r345_c983dc345a` (imported round-342 tool producer).

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

Round-345 year gate is closed. The ORCA2X pilot remains unscored until a
matching ORCA2X perturbation-envelope report exists.

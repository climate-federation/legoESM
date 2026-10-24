# SI3 round 7 receipt — O2 acquisition needed

## Outcome and provenance

Round `si3_r7` implements the O2 scalar initial state and the initial
levitating-ice SSH adjustment, supplies a fail-closed acquisition/admission
tool, runs only its read-only preflight, and stops at `ACQUISITION_NEEDED`.
Neither `makenemo` nor `mpirun` was run.

The O2 commits were rebased from the pre-fix O1 head onto the landed SI3 lane
tip `042c7bd74c2944dab8c39318687ab0cef54f9d2b`. This receipt and the operator
amendment in `docs/ice/fidelity/si3_r7_o2_preregister.md` supersede stale
pre-rebase statements without changing the preregistered scope.

The implementation was checked against the configuration's own compiled
source, not only the OMIP example:

- `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iceistate.f90:174-175`
  defines the halo-free-interior `t_bo` freezing-point operation.
- The same file at `:308-337` defines the TEOS-10 SST mask and
  hemisphere-dependent scalar initialization.
- The same file at `:441-459` defines mass, DDPDD global reductions, and the
  levitating-ice SSH branch. Lines `:456-459` subtract the same adjustment from
  the pre-existing Kmm and Kbb fields independently; there is no inner
  `ln_rstart` guard in this compiled branch.

## Model implementation

`packages/ice/legoesm/ice/si3_orca2_state.py` requires an explicit
`SI3O2InitConfig`, explicit `ssmask`, `smask0_i`, both pre-branch SSH levels,
and explicit halo policy. It refuses the wrong initialization mode, EOS,
embedded-ice mode, and restart mode. No missing O2 field is inferred.

The scalar operator implements:

- TEOS-10 `eos_fzp` for `t_bo` on the NEMO-defined interior and `ztfrz` through
  the NEMO-defined halos (`src/OCE/TRA/eosbn2.F90:1650-1710`);
- the SST threshold mask and northern/southern namelist scalars
  (`src/ICE/iceistate.F90:262-304`);
- pond-area scaling before category fitting; and
- the SSH branch by calling the existing public `orca2_initial_ice_ssh` recipe
  from `packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`.

The ORCA2 recipe retains its production-dispatcher NumPy behavior for existing
callers and adds the explicit TEOS-10, unique-mask, DDPDD, diagnostic-returning
path used by O2. The parent/current production-dispatcher control covers real
non-SI3, lane SI3, O1, and ORCA2 recipe callers. Code-path plants prove the O1
and O2 dispatcher branches are actually reached. O2 remains fail-closed in
`step_sea_ice`: it exposes initialization through `si3_o2_step` but refuses to
invent the still-unowned `ice_update_flx` heat/freshwater/salt response.

## Acquisition and admission contract

The committed tool is
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh`.
Its default is `--preflight-only`; `--run` is operator-only. It uses scratch
space under the evidence root, requires `free -g` available memory of at least
150 GB immediately before builds and runs, checks disk, wraps NEMO in
`timeout`, guards `cfgs/work_cfgs.txt`, and retains failed or admission-refused
logs through `ERR` and `EXIT` handling.

Before any possible NEMO launch, operator order is statically planted and then
requires both real executables to pass the shared landed same-program helper.
The producer manifest records both executable digests. `--admit-existing`
requires the recorded producer to be a real commit object, an ancestor of
current HEAD, and equal to the requested pin when supplied. It checks input and
binary SHA-256 manifests, same-program build paths, per-run producer stamps,
completion/log markers, and the optional gate-change marker.

The restart run refuses unless `rec_2step` produced both the SI3 and SAS ocean
restart. It copies them into `rst_1step/input_restart`, writes step-2 output to
`rst_1step/output_restart`, and never aliases an input with an output. The
restart reader consumes NEMO's halo-free NetCDF payload; rank-local binary
records retain the NEMO-defined seam/fold halos.

The O2 writer uses `USE, INTRINSIC :: iso_fortran_env`. Its compiled four-rank
fixture links the real writer, writes different values on every rank, and
contains a dry negative-zero cell whose sign must survive the writer. Scoring
uses wet cells only: `t_bo` uses the defined interior; other O2 fields include
defined wet seam/fold halos. Plants perturb real wet interior, seam/fold,
rank-local, SSH, provenance, binary, log, and restart-dependency paths and must
emit `REFUSE`/`STATUS PLANT-FIRED`.

The committed `ocean.output.excerpt` contains source path, complete-source
SHA-256, and exact source line provenance. Preflight checks those headers and
the deck deviations rather than trusting hand-written log values.

## Verification and stop condition

The focused O2 tests cover the compiled writer fixture, parent/current
production dispatchers, exact model formulas, JIT/JVP/VJP behavior,
same-program plants, halo-free restart layout, and producer-pinned admission.
The final preflight receipt is written to
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round7/o2_preflight_receipt.json`.
It reports `PREFLIGHT_OK`; its SHA-256 is
`e506a4144d5841736a298174c3b23acfd1f856eca832ad73adb2f9e876ceda30`.
Quantitative NEMO fidelity results are deliberately absent until the operator
runs acquisition and admission on the committed producer.

## Choices and review disposition

- `ASKED`: implement O2 on the landed O1 lane, reuse
  `orca2_initial_ice_ssh`, run preflight only, and stop at
  `ACQUISITION_NEEDED`.
- `UNASKED`: none.
- Independent review: operator.

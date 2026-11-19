# NEMO SWG double-gyre reproduction package: Round C receipt

Date: 2026-10-13. Status: **HELD FOR FRESH LONG-RUN/GPU ACQUISITION**.
Round C repairs the stranger-facing package on top of Round-B tip
`6aea0af1b`. No physics, deck value, or acceptance bar changed. This agent did
not run `makenemo`, MPI, NEMO, or a GPU. The operator's independent fresh NEMO
run remains read-only and in progress; exact GPU commands are retained below.

Work and commits are in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgdgsa-1049327816`.
Round-C evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round3`.

## Round-C commits

- `62ce3e27d`: bind package entry points to their owning checkout; add exact
  environment files, portable fixtures, cross-deck guard, short ladder, and
  runtime/memory manifests.
- `e9221475f`: launch ffmpeg before JAX worker threads and atomically publish
  a complete movie, after a measured Python 3.13 fork-after-JAX deadlock.

- `22dd7e2e8`: reconstructable environment guide, stranger runbook, cost
  artifact, review-disposition receipt, and runbook completeness guard.
- `d2a4e061f`: re-anchor the shifted provenance citation, record the completed
  movie/first battery, and correct the clean-wrapper decoy assertion.
- `7d6555aa3`: finalize the Round-C evidence and requested battery receipt.
- `fd67594ba`: select and verify the explicit JAX backend before checkout
  verification imports JAX; extend the decoy subprocess assertion.

## Independent-review findings

| Finding | Disposition | Evidence |
|---|---|---|
| 1. Package not executed end to end | **PARTIAL, acquisition still open.** The operator's clean run completed build, case identity, both rank arms, and smoke. The original rank wrapper then failed on the wrong checkout import; a manual scorer rerun passed. The three-year NEMO run is in progress and the fresh GPU half remains requested. | Fresh-run log SHA-256 `58c984116f91c080f0fff39c5d996ac5367d332f9c7f73868d4386ec4a9ec650`; acquisition table below. |
| 2. False single-source statement and unguarded duplicate decks | **FIXED.** The guide now says there are two independent definitions. A new guard compares NEMO's actual resolved echo with the resolved legoESM run card. It compares 137 common values and binds an exhaustive two-key output-control allowlist: `sn_cfctl.l_runstat` and `nn_stock`. A physical-value plant is rejected. Completed smoke/long NEMO launchers invoke the guard. | `deck_guard.json`, SHA-256 `3f875d1663510a75356516510b1cd0cab5aed6a15859a1c75fa5d9ff275aabfc`; committed fixture tests. |
| 3. Environment cannot be reconstructed | **FIXED.** The package names the NEMO repository/tag/SHA and clone/verification commands. It includes the measured linux-64 conda explicit lock, a readable no-build-string YAML, restricted pinned Python/CUDA requirements, compiler/runtime build strings, and hardware/storage pins. | `nemo-build-linux-64-explicit.txt`, `nemo-build-environment.yml`, `python-gpu-requirements.txt`, environment JSON/guide, syntax test. |
| 4. Developer paths, silent skips, bare Python, optional roots | **FIXED.** Tests use byte-exact committed NEMO fixtures; the patch headers are relative; Round 6/7 roots are required; old stage/month launchers also require explicit paths/interpreter/MPI. Every package entry point binds and prints its owning checkout, interpreter, and NumPy and refuses a mismatched import. | Decoy-checkout subprocess matrix and fixture tests: 23/23 passed in the focused shell/guard run. |
| 5. Missing failure path, runtime/resource/movie evidence | **FIXED except fresh GPU peak is acquisition-bound.** A committed command creates one-rank restarts at steps 1–10 and a scorer localizes every step/field. Runtime evidence is machine-readable; disk and tested GPU capacity are stated. The long driver now records wall time and JAX allocator `peak_bytes_in_use`, refusing unsupported GPU statistics. A year-one movie is executed in this round and hashed below. | Short-ladder JSONs, `swg_double_gyre_reference_costs.json`, driver tests, environment pin, movie artifact/log. The fresh GPU manifest must supply the new peak measurement. |
| 6. Headline scope too broad | **FIXED.** Guide and PR wording now state only end-of-day `ssh_m/u_m_s/v_m_s` versus `sshn[0]/un[0,0]/vn[0,0]` under T/U/V wet masks at days 365/730/1095. They explicitly exclude full-restart and dry-cell identity and do not use `uu_n/vv_n` as same-instant evidence. | Runbook headline and annual-comparison section; the three committed annual JSONs remain the citable artifacts. |

## “A stranger would get stuck at” dispositions

| Stuck point | Disposition | Evidence |
|---|---|---|
| 1. No NEMO URL/clone/tag proof | URL, tag, detached SHA checkout, and equality check added. | Environment guide and runbook step 1. |
| 2. No exact NEMO environment | Exact conda explicit lock plus readable YAML committed. Exact lock is normative. | Machine-readable syntax/build tests. |
| 3. No Python/CUDA creation | Python 3.13.0 assertion, venv command, fully pinned requirements, CUDA/PJRT wheel builds, and tested driver recorded. | Python requirements and environment documents. |
| 4. No disk/GPU sizing | Measured 56,985,134,712-byte NEMO and 21,803,971,913-byte legoESM roots; 100 GiB reservation; 32 GiB is the only tested GPU capacity, not an inferred minimum. | Environment JSON and guide. |
| 5. Portable build success unproved | Operator fresh build exited zero. | Fresh log `STEP build exit=0`; build-manifest inventory SHA-256 `8ca97cf52469f7d85e3e6e522fd79c7c352934b8a24241b6f1810328a9edfb5e`. |
| 6. Identity/rank/smoke/long execute paths unproved | Case identity and smoke passed. Both rank runs passed; wrapper scoring failed from the wrong checkout, then passed manually. Long NEMO is still in progress. | Fresh table below; no completion is implied. |
| 7. Launcher uses bare `python3` | All launchers require `--python`; GPU wrapper requires the documented `PYTHON` environment variable. | Shell tests and source audit. |
| 8. No GPU memory/CUDA/cost artifact | CUDA wheel builds and tested 32 GiB card pinned. Old 8,117 s log is hashed in the committed cost artifact. New driver records wall and peak allocator bytes; fresh values remain acquisition. | Costs JSON and `device_memory_statistics` tests. |
| 9. No short-ladder command/scorer | `--short-ladder` makes steps 1–10; committed scorer supports outer scan or materialized composition on explicitly selected CPU/GPU. | Reference short-ladder results below. |
| 10. No movie artifact | A full 365-frame reference movie is executed from pinned existing daily means with a completion log and hash below. The first attempt also found and caused repair of an actual fork-after-JAX deadlock. | `swg_dg_year1_daily_mean.mp4` and `movie_year1.log`. |

## Measured checkout-import defect and repair

The operator's clean rank command is the decisive reproduction failure:
both NEMO runs completed (`WALL_SECONDS=8` on one rank and `2` on 32 ranks),
then the comparison exited 1 with `ModuleNotFoundError` for
`legoesm.ocean.fidelity.provenance`. Its bare interpreter had selected another
checkout. With this checkout placed first, the unchanged outputs scored
`all_bit_identical=True variables=21`; report SHA-256 is
`113fe89f7e3530bff1232ec84eb244423ff857b6baa37334d2f2fdb6dd6ebb44`.

The repair is fail-closed rather than advisory. Shell launchers prepend package
roots computed from their own Git tree. Python entry points do the same before
importing legoESM. Both call `require_checkout_import`, print the selected
source/interpreter/NumPy, and refuse if `legoesm.ocean.__file__` is outside the
owning checkout. Subprocess tests put a decoy `legoesm` first on the incoming
path for every entry point and prove the real checkout wins.

A final import-order audit found that checkout verification itself imports
`legoesm.ocean`, which imports JAX. Commit `fd67594ba` therefore makes platform
selection part of the bootstrap: analysis entry points explicitly select and
verify CPU, while the long driver and short ladder apply their requested
CPU/GPU backend and physical GPU index before the verification import. Every
entry point prints `JAX_PLATFORM_OK`; the same decoy subprocess matrix asserts
the CPU selection. The NEMO-only shell bootstrap also pins its verification
process to CPU. This closes the possibility that a correct checkout could
still initialize JAX on an implicit platform before CLI validation.

## One guarded deck

The guard ran against the reference NEMO `output.namelist.dyn` and legoESM's
resolved 1,095-day namelist. All physical and numerical values were equal.
The only differences were:

| Field | NEMO | legoESM | Explicit reason |
|---|---:|---:|---|
| `sn_cfctl.l_runstat` | true | false | NEMO completion output has no legoESM writer and cannot change state. |
| `nn_stock` | 630720 | 1728 | annual NEMO tiles versus daily legoESM NPZ; output cadence cannot change state. |

The guard compares the resolved lego run, so the raw constant's shorter
`nn_itend` is not hidden or allowlisted: it is replaced by the requested days
before comparison and must equal NEMO. A planted `rn_tau` difference refuses.

## First-difference return path

Clean commit `62ce3e27d9ce1ceca3985f56ab9744525daf5cdc` produced both reference
reports from existing one-rank NEMO restarts, with a planted one-ULP control:

- outer scan: all 30 completed-state step/field rows exact through step 10;
  final line `SWG_DG_SHORT_LADDER EXACT ... first_difference=NONE`; JSON
  SHA-256 `8088ad908e92b1173cda0405d31f6a8c9376a233ae35b0a75107727c736582f7`;
- materialized loop: expected exit 1, first difference step 2 `ssh_m`, 189
  wet words, maximum absolute difference `1.3552527156068805e-20`; JSON
  SHA-256 `31e7be2862f2cccb78c800d388becd9e842b4df05dfd63603b4ed856454ef258`.

The 189-word completed-state statistic does not retract the distinct formal
stage-entry debt: its first step-2 stage-1 SSH row has 1,278 unequal words.
The 640 V-stress signed-zero words also remain DEBT. No bar or exception was
changed.

## Costs and resources

The committed cost record binds each quoted runtime to a hashed source:

- NEMO three years: 22,918 s on 32 Xeon 6242R ranks;
- legoESM daily-mean three years: 8,117 s on one V100S 32 GB;
- legoESM formal CPU fallback: 233.93357684463263 s for one day.

The older GPU log lacks allocator memory. The new driver reads the installed
JAX `Device.memory_stats()` API, requires `peak_bytes_in_use` on GPU, and adds
start/completion memory dictionaries plus wall seconds to the manifest. The
fresh GPU acquisition is required before quoting a measured peak. CPU JAX
correctly reports the API as unsupported and remains usable because its peak
is not claimed.

## Fresh package run, measured so far

All rows are **MEASURED** from the operator's clean run log/artifacts, not this
agent's inference:

| Step | Result | Artifact |
|---|---|---|
| Build | exit 0 | fresh log; build manifest above |
| Shipped-case identity | `BIT_IDENTICAL variables=41 differing=0` | `swg_dg_identity.json`, SHA-256 `faa618f78cb5037c8f806d4851d15648f00349b0b507eed8a0a11ab82309b937` |
| Rank command | exit 1 after both NEMO runs; measured packaging import failure | fresh log and 8 s/2 s `run.log` files |
| Rank scorer rerun | `all_bit_identical=True variables=21` | report SHA-256 `113fe89f7e3530bff1232ec84eb244423ff857b6baa37334d2f2fdb6dd6ebb44` |
| 30-day smoke | exit 0, `WALL_SECONDS=504` | `dg_smoke/run.log`, SHA-256 `97a87bca3c2035b1764e7baf21422aa271f7257b14d276bd2f56df0705c2ad81` |
| Three-year NEMO | **in progress**, not claimed complete; step 1,035,291 observed during the final audit | `operator_acquisition/nemo/dg_3yr/time.step` |
| Fresh GPU/comparisons/movie | not started | acquisition commands below |

## Tests and evidence boundary

Portable fixture tests do not skip when the oracle tree is absent. Tests that
exercise external operator evidence remain visibly classified as such and are
not counted as package guards. The combined focused package run passed 56/56
in 178.50 seconds.

The first complete requested battery ran from clean commit `22dd7e2e8` and
recorded 349 passed, one registered six-step LLVM-memory skip, and four failed
in 3215.77 seconds; log SHA-256 is
`d52b05bf28e86889885f93efb063b665e2be34763b63f4bf787d9c570222ccdf`.
Three failures had one cause: the new checkout helper shifted `git_sha` from
line 107 to 149 without re-anchoring its campaign receipt/map entry. The
fourth caught a test-harness error: the decoy test expected the clean GPU
wrapper's `--help` to fail, although success is correct after it binds this
checkout. Both are corrected in `d2a4e061f`; final clean rerun results follow
below. No model/card/gradient/comparator assertion failed.

After `d2a4e061f`, the citation gate reported PASS for 274 citations, zero
failures, zero unmapped citations, zero map-audit failures, and all nine
controls firing; log SHA-256 is
`5c2bb03e06c8e38a3b793b0eee123e3543117e3656db5f0df05796079503fb3e`.
The 18 directly affected tests passed. Following the repository instruction
to split compiler-heavy suites, the already-passed clean gradient matrix was
retained (11 passed, one registered six-step LLVM-memory skip), and every
non-matrix requested battery file was rerun at `d2a4e061f`: **342 passed** in
806.99 seconds, log SHA-256
`8d63a51d613d60234ad8e6d12d7a4b41ea690eb26eb373745a768298bd7bf7e7`.
Combined final requested coverage is **353 passed, one registered skip** over
all 354 tests. No full-repository suite or optional 172,800-step scorer ran.

After the platform-before-import audit, the complete reproduction-package set
ran once more from clean commit `fd67594ba`: **67 passed** in 192.64 seconds.
Its log SHA-256 is
`fd2530fc14dd9884d970507c6bd417a802dc96a72da48f3e8610d23b5bf06a1b`.
The citation gate at that commit again passed all 274 citations and nine
controls; log SHA-256
`08b0a8d570835cf71fbfb3d92ffbbf19d419206dac057d7806bdc0c1a6040399`.

The movie repair arose during this round's permitted CPU execution. The first
attempt left only a 48-byte partial after Python warned that ffmpeg's fork
could deadlock with JAX threads. It was stopped and retained as
`swg_dg_year1_daily_mean.deadlock-partial.mp4`. Commit `e9221475f` starts
ffmpeg before any JAX work, writes to a hidden partial path, and renames only
after all frames close successfully. A test proves encoder start precedes
provider/model work. The rerun printed
`SWG_DG_MOVIE COMPLETE frames=365`, took 1,794 seconds including fail-closed
input hashing, and produced a 5,348,065-byte MP4 with SHA-256
`a7b06ec4c01de35b1ee35cb96f7ee08050e9953effc49e540156feb0faee1eaa`.
The completion log SHA-256 is
`7c154a1b6d9aac0653da2f52ccb9b78580cbc72e7c3a47664231cca1260d1351`.
An independent ffmpeg decode count reported 365 frames and 30.41 seconds;
that log's SHA-256 is
`2c8231eec25db5fc953bebc448a0d601b1d7cecc0524931fd629ad30bf9ad9b6`.

## Remaining operator acquisition

Run only from a detached worktree at the final Round-C commit and only when
the selected GPU is idle. The exact final commands and required terminal lines
are repeated in the handoff message. They produce:

1. a fresh 1,095-day daily-mean legoESM root whose manifest reports
   `wall_seconds` and positive `peak_bytes_in_use`;
2. three annual identity JSONs against the fresh NEMO root;
3. two matched-daily-mean climate reports; and
4. a 365-frame MP4 and completion log.

Until those complete, Round C remains **HELD**, not “landed end to end.”

## Required stacked-PR wording

The PR must say:

- this branch is stacked on `fidelity/nemo-testcases-swg-standalone`; Round A
  faithfully replays 58 commits with unchanged author/message metadata;
- the six non-equal range-diff situations are citation relocation/context
  effects, including the sole citation-map conflict, and all 13 final
  package/test patches are hunk-equivalent;
- base merge `ac8e49fab` carries the selector-order and SWG-libm fixes, tested
  for shipped, fine-double-gyre, and another accepted parameterized deck;
- SWG has no remaining `SWG_UNMEASURED` item and removal of `key_xios` is the
  only declared NEMO build deviation;
- the exact annual claim uses only the three named completed-state fields and
  wet masks above—not full restarts or dry cells—and remains conditional on
  the pinned compiler/JAX/GPU composition;
- the NEMO and legoESM decks are independent definitions guarded across 137
  resolved values with the two explained output-control differences;
- fresh build/case-identity/smoke passed; the measured rank-wrapper import
  failure is fixed and manually rescored, while long NEMO/GPU acquisition
  status is stated exactly as it stands;
- the climate comparison uses matched daily means and its measured residuals
  are not a cross-toolchain/post-decorrelation acceptance bar; and
- the materialized step-2 and 640 forcing signed-zero debts remain open with
  no relaxed exactness bar or renewed policy question.

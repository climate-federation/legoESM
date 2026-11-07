# NEMO SWG double-gyre reproduction-package receipt

Date: 2026-10-13. Status: **HELD FOR OPERATOR ACQUISITION**. The portable
Round-B package and locally permitted comparisons are complete. The remaining
commands are the explicitly prohibited `makenemo`, MPI, and GPU executions
listed under “Acquisition boundary”; no exactness bar or configuration was
changed.

All work was done in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgdgsa-242520746`.
Evidence is outside Git under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round2`.

## Branch and base synchronization

The already-pushed Round-A replay tip was `9572421ad`. Commit `ac8e49fab`
merged the advanced standalone base at `0db49ce5b` without conflict or history
rewrite. That brought in the reviewed selector-order and shared SWG-libm
guards. Parameterized tests exercise shipped, fine double-gyre, and another
accepted deck: each retains the libm precondition, and an unsupported r3t
selector is rejected before the linear-free-surface shortcut. The post-merge
package diff adds no new validation-bypassing shortcut. The inspection log is
`validation_order_audit.log` (SHA-256
`69e1856a7f2fbaacf01fd9b6d39867fd2ab131de4b67b2dfd1979a4393cd6eec`).

Round-B implementation commits through the final evidence commit are:

- `d568cd30f`, portable build/run/compare/movie/environment tools;
- `22ee701db`, committed program-composition discriminator and fp64 climate
  assembly repair;
- `537c137e6`, fail-closed deck, dtype, mask, revision, completion, rank, and
  path controls;
- `cf66741d6`, annual reports regenerated from the daily-mean run with the
  final checker.

## Portable package

The package contains:

- a path-argument-only NEMO `SWG_DG` builder, a provenance-stamped reference
  scalar-math architecture, the two-line `usrdef_zgr.F90` patch, and a
  self-contained, directly checkable build-hash inventory;
- shipped-case identity, 1-versus-32-rank, smoke, and three-year launchers
  whose mutating modes require `--execute`, validate safe output labels, and
  refuse incomplete final-step stamps;
- one committed Maddison namelist patch as the physical-deck source of truth;
- a legoESM driver with an explicit required physical GPU, a manifest-stamped
  device list, one-JIT-per-day production composition, daily means, and a
  measured certified-CPU fallback;
- a yearly restart checker, climate instrument, composition discriminator,
  and two-row NEMO/legoESM movie tool, each with synthetic controls;
- machine-readable and prose environment pins plus the colleague runbook
  `swg_double_gyre_reproduction.md`.

The build preflight succeeded against a fresh shared clone at the exact NEMO
revision without invoking `makenemo`. Its log SHA-256 is
`5d3a9786ab9dbc4e6ed3bc5a8a774592ad3fef04d5609558aa7149548292ea1d`.

## Re-measured environment

The round command re-read NEMO revision
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`, GNU Fortran 15.2.0
(conda-forge gcc 15.2.0-19), Open MPI 5.0.10, Python 3.13.0, JAX/JAXlib/CUDA
plugin 0.10.0, NumPy 2.4.4, Equinox 0.13.7, imageio-ffmpeg 0.6.0, Intel Xeon
Gold 6242R, and Linux 5.14.0-611.36.1.el9_7.x86_64. The reference arch hash is
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`.
The command log SHA-256 is
`bebaa9e1c1cdd064ce908dc25e5783843c7eaf25396afdd72ab00e0643b8da50`.
The unavailable GPU facts remain the operator's same-day measurement: Tesla
V100S-PCIE-32GB and driver 580.126.20.

## NEMO controls rechecked from existing output

The final case-identity checker admitted `time.step=10` in both arms and
compared all 41 output variables: zero differed. Its JSON SHA-256 is
`2879718e9f8907ccd892340460a86adc00fb00ad6bb5e19eb1ed4ac59ebaf32f`.
The final rank checker compared all 21 assembled restart variables from one
and 32 ranks and reported `all_bit_identical=True`; its JSON SHA-256 is
`60d55fd29cdf0625c565a1597186aeb99122e5c192df9d51f824b8f3d8551c10`.
These commands compared existing NEMO artifacts; they did not run NEMO.

## Annual completed-state identity

The final checker binds the complete physical deck, native fp64 finite
storage, every NEMO T/U/V wet mask from 32 mesh tiles, both content manifests,
and the following completed-step mapping:

| legoESM field | NEMO restart field | Capture instant |
|---|---|---|
| `ssh_m` | `sshn[0]` | completed step N |
| `u_m_s` | `un[0,0]` | completed SWE layer state after step N |
| `v_m_s` | `vn[0,0]` | completed SWE layer state after step N |

`uu_n/vv_n` are hashed zero-valued barotropic workspace arrays in these NEMO
restarts. They are explicitly excluded rather than presented as same-instant
layer velocities.

The checker ran against the same `dg_gpu_1095d_daily_mean` root prescribed by
the runbook. Years 1, 2, and 3 each reported **PASS**, 3 fields, 1,227,520 wet
words, zero unequal words, zero maximum absolute difference, and zero
prognostic signed-zero words. The external and committed JSONs are byte-for-
byte identical, with SHA-256 values:

- year 1: `bd2bee45181c1495f83cc52139e2ddd84fbcda7bf6cddb33063a96bee99f5438`;
- year 2: `1f07f73f05c4d166e9cc3eacce8aa83b549d6896b4c8f8cfec9cb275ac0f655e`;
- year 3: `6cf0dbd0141366810ed008cdcb4dcf80e36f24a189d843a7a97d0855deced08e`.

Those three reports are the citable support for “bit-identical at years 1,
2, and 3.”

## Program composition and preserved debt

The committed ten-step discriminator uses the production step function. On
the reference toolchain, the outer-jitted scan has zero prognostic unequal
words; its only 640 unequal words are signed zeros on V stress. The formal
materialized Python loop has 414,545 prognostic unequal words and 415,185
total unequal words at step 10. The planted one-ULP control binds. The
committed and external report SHA-256 is
`8e87166f0a9edd286d797e598cccb4f8656835d475b04fc07d4dfed3599387a8`.

The real-data per-step audit was rerun after adding the signed-zero scope.
Every observed signed-zero row is in the forcing-field allowlist,
`forcing_only_observation=true`, and
`prognostic_or_unknown_violations=[]`; the planted prognostic sign change is
rejected. The report remains DEBT pending the user's policy ruling. Its
SHA-256 is
`3177de6e4d5c26b928e0ad0bdc43ece9b1f13a38cf3b58d64aab473a28b20daf`.

No debt is hidden or excepted. The formal per-step composition still differs
from NEMO from step 2 in last-bit wall-adjacent SSH words; the 640 forcing
signed-zero words also remain reported. The available future choices are
explicitly non-exhaustive: carry that formal debt, separately certify the
daily scan, or pin/reshape the formal XLA composition. This round makes none
of those policy decisions.

## Climate and CPU fallback

The final fp64 instrument ran with `JAX_PLATFORMS=cpu` on both registered
windows and printed the expected PASS lines. For year 1, the basin-mean KE
and EKE maximum absolute differences are respectively
`2.286230926795696e-11` and `1.1921534939784983e-11`; the report SHA-256 is
`6919fc38b4b2a0b975226676e32dc512b5f90584954f59e7ae62e892cbcc2e53`.
For days 366--1095, they are `2.2225409013199027e-11` and
`1.402270660699756e-11`; the report SHA-256 is
`a9ca415707301708a41acbeec8edc7dd13a06b4ae0df8fe15bb9c1ca1954f2c8`.
These reproduce the reference-trajectory scale. They are not promoted into a
cross-toolchain or post-decorrelation acceptance bar.

The full-resolution certified CPU fallback completed one 1,728-step model day
in 233.93357684463263 seconds (233.934 seconds rounded for the runbook). Its
complete manifest SHA-256 is
`60d38083ff1134ff514562d18643aa8bce5267b3b5eafd8ac5ed7cd60bbf452c`.
It is a debugging fallback, not a substitute for the operational daily-scan
composition.

## Disposition of the prior six-part HOLD

1. The broad “carried U/V” wording is superseded: the annual checker states
   the exact `un/vn` mapping and capture instant and excludes `uu_n/vv_n`.
2. Three final, committed yearly reports now cite the annual identity claim.
3. The outer-scan versus materialized composition discriminator and its real
   report are committed.
4. Pinning or reshaping the formal XLA composition is retained as a third
   possible future action; the decision record is not reduced to two choices.
5. Signed-zero scope is machine checked with a forcing allowlist and a planted
   prognostic violation; no exception or bar change was applied.
6. The Round-7 receipt's test-coverage sentence is not relied upon. The
   standalone replay receipt already narrows that coverage, and this receipt
   separately identifies synthetic tests versus real-data executions.

## Tests and gates

Focused reproduction tests recorded 50/50 passed after the fail-closed
repairs, including the mounted no-run NEMO preflights. Shell syntax and the
focused new-file Ruff checks passed; the older acquisition checker retains
pre-existing style exceptions and was not represented as newly Ruff-clean.

The first requested battery collected 335 items and recorded 332 passed, one
registered skip, and two refusals in 638.39 seconds. Both refusals were the
gradient matrix's clean-worktree guard: the two new documentation files were
still untracked. No underlying gradient or package assertion failed. This
precommit log has SHA-256
`d0450432feb69c00f83841774f448dd932ef4464ac3ec501f7c6fc2bc7bd9653`;
the same battery was then rerun from the clean documentation commit. The
authoritative run recorded **334 passed, 1 registered skip** in 1926.56
seconds. The skip is the existing six-step CPU GYRE row's LLVM section-memory
limit; every SWG and reproduction-package test passed. The clean battery log
SHA-256 is
`e474781733e733aa25a3636bab6c22ec7fb67e908ed42990acb0c3a0b2d817a4`.

The citation gate was rerun after this receipt was written.
It reported PASS for 274 mapped citations, no failures, no unmapped
citations, no map-audit failures, and all nine built-in controls firing. The
clean-tree gate at `c27fe2a37` has log SHA-256
`1a95ae4faee744ee245ce07bc6748aea71eba00e8a0fa3b95a0e1c9179b512bf`.

## Acquisition boundary

This round obeyed the prohibition on `makenemo`, MPI, NEMO engine, and GPU
commands. The build success path, shipped-case identity run, rank-control run,
NEMO smoke/three-year runs, and a fresh GPU run therefore remain explicit
operator acquisitions. Existing outputs were sufficient to execute the
portable comparators and establish the annual, rank, composition, and climate
results above.

The exact operator sequence is below. The final GPU command must run only
after physical device 0 is observed idle; if it is not idle, defer rather than
pre-empting it.

```bash
CLONE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-swgdgsa-242520746
NEMO=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round2/nemo_5.0.2_clean_preflight
PREFIX=/home/dbalwada/miniconda3/envs/nemo-build
OP=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdgsa_rounds/round2/operator_acquisition
PY=/home/dbalwada/legoESM/.venv/bin/python
export PATH="$PREFIX/bin:$PATH"
mkdir -p "$OP"
cd "$CLONE"
bash scripts/data/nemo_swg_double_gyre/build_swg_dg.sh --nemo-root "$NEMO" --conda-prefix "$PREFIX" --manifest-dir "$OP/build_manifest" --execute
bash scripts/data/nemo_swg_double_gyre/check_swg_dg_identity.sh --nemo-root "$NEMO" --arch-name swg-dg-scalarmath --output "$OP/case_identity" --python "$PY" --execute
bash scripts/data/nemo_swg_double_gyre/swg_dg_rankcheck.sh --nemo-cfg "$NEMO/tests/SWG_DG" --output "$OP/rank_control" --python "$PY" --execute
bash scripts/data/nemo_swg_double_gyre/swg_md_run.sh --nemo-cfg "$NEMO/tests/SWG_DG" --output-root "$OP/nemo" --smoke dg_smoke 32 --execute
bash scripts/data/nemo_swg_double_gyre/swg_md_run.sh --nemo-cfg "$NEMO/tests/SWG_DG" --output-root "$OP/nemo" --three-year dg_3yr 32 --execute
PYTHON="$PY" scripts/run/run_swg_double_gyre_gpu.sh --gpu-device 0 --days 1095 --daily-means --output "$OP/lego/dg_gpu_1095d_daily_mean"
```

## Required stacked-PR description

The pull-request description must say all of the following:

- this stacked branch adds explicit `swe_rk3`, total-ENS vorticity, and
  constant NEMO-viscosity paths; valid pre-existing defaults are unchanged;
- the advanced base's r3t-selector and shared-libm fixes are merged and tested
  across every accepted parameterized SWG deck;
- Round A faithfully replayed 58 source commits plus its receipt, with the
  sole citation-map conflict and all range/package differences explained in
  the standalone replay receipt;
- SWG has no remaining `SWG_UNMEASURED` item and removing `key_xios` is its
  only declared build deviation;
- the yearly exact claim is supported by the three committed final-checker
  JSONs, while exactness remains conditional on the pinned compiler/JAX/GPU
  program composition;
- the formal materialized composition and 640 forcing signed-zero words remain
  open DEBT with no relaxed bar, and the user's policy decision remains
  separate from this reproduction package;
- this round did not run `makenemo`, MPI, NEMO, or a GPU, and the exact
  operator acquisitions remain listed rather than implied complete;
- the requested focused battery was run on CPU/x64; no full-repository test
  suite or optional 172,800-step shipped scorer was run.

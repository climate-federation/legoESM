# Reproducing the NEMO–legoESM shallow-water double gyre

This is a clean-machine runbook for a colleague who knows NEMO and Python but
not legoESM. It builds the `SWG_DG` case, proves the two-line case variant is
inert on shipped SWG, runs the 4,000 km eddying double gyre for three years in
both models, compares annual states, computes matched daily-mean climate
statistics, and makes a side-by-side movie. The workflow has nine numbered
steps; the environments and the package self-test are step 1.

The exact reference claim is deliberately narrow: on the pinned composition,
legoESM end-of-day `ssh_m`, `u_m_s`, and `v_m_s` at days 365, 730, and 1095
are IEEE-word identical to NEMO `sshn[0]`, `un[0,0]`, and `vn[0,0]` over the
T, U, and V wet masks. This is not full-restart identity and says nothing
about dry/storage cells.

## Before you start

You need:

- network access to GitHub, the NEMO Forge, conda-forge, and the Python
  wheel index;
- the legoESM repository `https://github.com/climate-federation/legoESM`,
  branch `fidelity/swg-double-gyre-standalone`, at the full commit the
  maintainer sends with this guide (the hand-off commit; step 1 checks it),
  and a separate writable NEMO tree;
- GNU/Linux x86-64 with **glibc 2.34 or newer**; check with
  `ldd --version | head -1`. The exact NEMO environment needs it: three of its
  conda packages (`sysroot_linux-64` 2.34, `libsystemd0`, `libudev1`) declare
  `__glibc >=2.34`, and the reference NEMO executable uses `GLIBC_2.34`
  symbols. That means RHEL/Rocky 9, Ubuntu 22.04, Debian 12 or newer; on an
  older host use a container with a newer glibc;
- conda, Git, `patch`, the SHA-256 tools, and GNU binutils `ar` on the host:
  the architecture file names a plain `ar` and the conda environment provides
  only prefixed archivers, so the host's `ar` builds the NEMO library (the
  build refuses with `REFUSE_ARCHIVER` when there is none);
- 32 CPU cores for NEMO and an idle NVIDIA GPU. The only tested GPU is a
  Tesla V100S-PCIE-32GB (32 GiB, compute capability 7.0) with driver
  580.126.20 (measured on the reference host). The pinned CUDA wheels are
  CUDA 12.9 Update 1, whose NVIDIA release notes list Linux driver
  `>=575.57.08`; NVIDIA's CUDA 12 minor-version compatibility covers drivers
  `>=525` and `<580`, and JAX documents `>=525` for its CUDA 12 wheels. Those
  minimums are documented by NVIDIA and JAX, not measured here. Unset
  `LD_LIBRARY_PATH` so a system CUDA cannot shadow the pinned wheels;
- about 1.2 GiB of free memory for the package self-test of step 1 (its
  measured peak);
- at least 100 GiB free for outputs (the reference NEMO directory is
  56,985,134,712 bytes and the legoESM daily-mean directory is
  21,803,971,913 bytes); and
- a scheduling window of about 7 hours on 32 CPU cores and 2.5 hours on one
  GPU, plus about half an hour for the movie and storage-bound comparison
  time.

The reference host CPU is an Intel Xeon Gold 6242R. The three-year NEMO run
took 22,918 s there for the reference build and 17,404 s for a fresh rebuild
on the same host (byte-identical output); NEMO time depends on machine load.
The GPU run took 8,117 s and the year-1 movie 1,794 s. These four times come
from `swg_double_gyre_reference_costs.json`, which names each source log and
its SHA-256. The current manifests and logs, not those planning values, are
authoritative for your machine. Never pre-empt an occupied GPU.

GPU generation: NVIDIA's cuDNN 9.21.0 support matrix lists compute capability
7.5 and newer for the CUDA 12 build (the pin is cuDNN 9.21.1.3), so it does
not list the V100S used here. The legoESM GPU path calls no convolution (a
search of the ocean and core packages and of the driver finds none), and the
V100S reference run completed, so that support change does not affect this
package; JAX itself documents compute capability 5.2 and newer on CUDA 12.
Another GPU generation may still change the last bit of the result (see
"Honest expectation").

Keep `$REPRO` (below) outside the legoESM checkout: the GPU wrapper refuses a
checkout with untracked files. Put the `export` lines of step 1 in a file and
`source` it in every new shell; no step relies on other shell state. Every
tool that needs MPI or the compilers takes `--conda-prefix` and puts that
prefix first on `PATH` itself, and every Python tool selects float64 itself.

## What you should see

The 5-minute case, 8/2-second rank arms, and 504-second smoke measurements are
bound to the fresh-run artifacts in the Round-C receipt. Long-run timings are
bound to `swg_double_gyre_reference_costs.json`.

| Step | Command below | Required final line | Reference time |
|---|---|---|---|
| Import check | Python environment check | `CHECKOUT_IMPORT_OK ...` | seconds |
| Package self-test | eleven test files | the per-file lines of step 1 | 177 s |
| Build variant | `build_swg_dg.sh --execute` | `BUILD_DONE .../tests/SWG_DG` | not benchmarked |
| Namelist inventory | inventory command | `SWG_DG_NAMELIST_INVENTORY PASS groups=27 fields=336 ...` | seconds |
| Case identity | `check_swg_dg_identity.sh --execute` | `SWG_DG_IDENTITY BIT_IDENTICAL variables=41 differing=0` | about 5 min in fresh operator run |
| Rank control | `swg_dg_rankcheck.sh --execute` | `all_bit_identical=True variables=21` | 1-rank 8 s; 32-rank 2 s |
| NEMO smoke | `swg_md_run.sh --smoke ... --execute` | `NEMO_RUN_COMPLETE label=dg_smoke steps=51840 ranks=32 ...` | 504 s in fresh operator run |
| NEMO long run | `swg_md_run.sh --three-year ... --execute` | `NEMO_RUN_COMPLETE label=dg_3yr steps=1892160 ranks=32 ...` | 22,918 s and 17,404 s |
| NEMO tile check | tile-hash command | `NEMO_RESTART_TILES_MATCH_REFERENCE tiles=96` | seconds |
| legoESM long run | GPU wrapper | `COMPLETED days=1095 backend=gpu ...` | 8,117 s (2.255 h) |
| Annual checks | restart checker, three times | `SWG_DG_RESTART_IDENTITY PASS ... unequal=0 ...` | not benchmarked |
| Climate statistics | climate instrument, twice | `SWG_DG_CLIMATE_REPORT_WRITTEN window=... agreement_test=none ...` | storage-bound; not benchmarked |
| Movie | movie tool | `SWG_DG_MOVIE COMPLETE frames=365 ...` | 1,794 s |

“Not benchmarked” is intentional: this guide does not invent a cost where no
timing artifact exists.

## 1. Reconstruct the environments

Choose only paths on your machine, and the hand-off commit:

```bash
export LEGOESM_COMMIT=full-40-hex-commit-sent-with-this-guide
export LEGOESM_ROOT=/path/to/clean/legoesm
export NEMO_ROOT=/path/to/new/nemo-5.0.2
export NEMO_PREFIX=/path/to/new/nemo-build-environment
export PYTHON_PREFIX=/path/to/new/python-3.13.0
export PYTHON_ENV=/path/to/new/legoesm-swg-dg-python
export PY="$PYTHON_ENV/bin/python"
export REPRO=/path/to/new/swg-dg-reproduction
export TOOLS="$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre"
export NEMO_RUN="$REPRO/nemo/dg_3yr"
export LEGO_RUN="$REPRO/lego/dg_gpu_1095d_daily_mean"
```

Clone and verify legoESM and NEMO exactly:

```bash
git clone --branch fidelity/swg-double-gyre-standalone \
  https://github.com/climate-federation/legoESM.git "$LEGOESM_ROOT"
git -C "$LEGOESM_ROOT" checkout --detach "$LEGOESM_COMMIT"
test "$(git -C "$LEGOESM_ROOT" rev-parse HEAD)" = "$LEGOESM_COMMIT"
test -z "$(git -C "$LEGOESM_ROOT" status --porcelain)"
git clone --branch 5.0.2 --single-branch \
  https://forge.nemo-ocean.eu/nemo/nemo.git "$NEMO_ROOT"
git -C "$NEMO_ROOT" checkout --detach \
  dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
test "$(git -C "$NEMO_ROOT" rev-parse HEAD)" = \
  dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
```

Recreate the exact linux-64 NEMO environment. Use the readable YAML only if
the explicit build file cannot be solved; that fallback is no longer an exact
build-composition reproduction (it also pins `sysroot_linux-64=2.34`).

```bash
conda create --prefix "$NEMO_PREFIX" --file \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/nemo-build-linux-64-explicit.txt"
```

The exact NEMO lock has 122 package/build URLs and an MD5 on every URL. Create
the exact Python 3.13.0 interpreter from conda-forge build
`h9ebbce0_101_cp313`, then make the isolated virtual environment and install
the model pins and the self-test pins:

```bash
conda create --prefix "$PYTHON_PREFIX" --override-channels \
  --channel conda-forge 'python=3.13.0=h9ebbce0_101_cp313' pip
"$PYTHON_PREFIX/bin/python" -c \
  'import sys; assert sys.version_info[:3] == (3, 13, 0)'
"$PYTHON_PREFIX/bin/python" -m venv "$PYTHON_ENV"
"$PY" -m pip install -r \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/python-gpu-requirements.txt"
"$PY" -m pip install -r \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/python-selftest-requirements.txt"
mkdir -p "$REPRO"
```

The 40 model pins include JAX, jaxlib, CUDA plugin and PJRT 0.10.0, every
required CUDA wheel, `typing-extensions==4.15.0`, and `certifi==2026.4.22`.
The separate self-test file pins only the test tools at the reference
versions: `pytest==9.0.3`, `pluggy==1.6.0`, `iniconfig==2.3.0`,
`pygments==2.20.0`, and `PyYAML==6.0.3`. Full values and the readable conda
alternative are in `swg_double_gyre_environment.md` and `.json`.

Which of these commands were executed when this guide was written, and which
were not: the conda lock is byte-identical to `conda list --explicit --md5` of
the reference NEMO environment, every pin equals the version installed in the
reference Python environment, and the package self-test below was run in an
environment holding exactly the 45 pinned distributions (built from the
reference environment's installed files, not downloaded). Nobody has yet run
the two `conda create` commands or the two `pip install` commands from
scratch, and the fresh reproduction of 2026-10-13 reused the reference
environments.

Every command below binds imports to the checkout containing the script. Its
first line names that checkout, interpreter, NumPy version, and source file.
Verify it before continuing:

```bash
cd "$LEGOESM_ROOT"
"$PY" scripts/run/run_swg_double_gyre.py --help
```

Required prefix:

```text
CHECKOUT_IMPORT_OK root=/your/legoesm ... numpy=2.4.4 ...
```

The tools refuse if `legoesm.ocean` resolves outside this checkout. They also
set CPU/GPU selection and float64 explicitly before importing JAX.

On the GPU host, a one-line driver check before anything long:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda "$PY" -c \
  'import jax; print(jax.default_backend(), jax.devices())'
```

It must print `gpu` and one CUDA device without an error. This check was not
executed when this guide was written (no GPU was available to that round).

### Package self-test

This self-test covers the reproduction package itself: the deck guard and its
namelist inventory, the shell tools, the long-run driver, the annual,
climate, rank and composition checkers, the short-ladder scorer, the movie
tool, the environment files, and the legoESM card. Run each file in its own
process:

```bash
cd "$LEGOESM_ROOT"
for test_file in \
  tests/ocean/fidelity/test_swg_dg_reproduction_guards.py \
  tests/ocean/fidelity/test_swg_dg_reproduction_shell.py \
  tests/ocean/fidelity/test_swg_double_gyre_movie.py \
  tests/ocean/fidelity/test_run_swg_double_gyre.py \
  tests/ocean/fidelity/test_nemo_swg_dg_climate.py \
  tests/ocean/fidelity/test_nemo_swg_dg_restart_identity.py \
  tests/ocean/fidelity/test_nemo_swg_dg_composition.py \
  tests/ocean/fidelity/test_nemo_swg_dg_round6.py \
  tests/ocean/fidelity/test_nemo_swg_dg_round7.py \
  tests/ocean/fidelity/test_nemo_swg_check_records.py \
  tests/ocean/unit/test_nemo_swg_card.py; do
  PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
    "$PY" -m pytest -q -p no:cacheprovider "$test_file"
done
```

Measured result on a machine with only the repository (2026-10-14, reference
host): the loop above, in an environment holding only the 45 pinned
distributions, with every directory of external evidence hidden in a private
mount namespace and the checkout mounted read-only. The committed record is
`swg_dg_selftest_reference.json`.

| Test file | Result | Wall time | Peak memory |
|---|---|---|---|
| `test_swg_dg_reproduction_guards.py` | 33 passed, 1 skipped | 27 s | 1.2 GiB |
| `test_swg_dg_reproduction_shell.py` | 23 passed | 37 s | 0.6 GiB |
| `test_swg_double_gyre_movie.py` | 2 passed | 12 s | 1.1 GiB |
| `test_run_swg_double_gyre.py` | 8 passed | 57 s | 1.2 GiB |
| `test_nemo_swg_dg_climate.py` | 9 passed | 10 s | 0.8 GiB |
| `test_nemo_swg_dg_restart_identity.py` | 10 passed | 3 s | 0.6 GiB |
| `test_nemo_swg_dg_composition.py` | 3 passed | 3 s | 0.6 GiB |
| `test_nemo_swg_dg_round6.py` | 6 passed | 11 s | 0.9 GiB |
| `test_nemo_swg_dg_round7.py` | 6 passed | 2 s | 0.2 GiB |
| `test_nemo_swg_check_records.py` | 22 passed | 3 s | 0.6 GiB |
| `test_nemo_swg_card.py` | 157 passed | 11 s | 1.0 GiB |
| Total | 279 passed, 1 skipped | 177 s | 1.2 GiB (largest file) |

The one skip is the optional live re-derivation of the namelist inventory;
pytest prints its reason:

```text
tests/ocean/fidelity/test_swg_dg_reproduction_guards.py:441: set SWG_DG_PPSRC to a built tests/SWG_DG/BLD/ppsrc/nemo tree to re-derive the committed compiled-read inventory
```

Set `SWG_DG_PPSRC` to your built `tests/SWG_DG/BLD/ppsrc/nemo` after step 2
to run it. Run with the external evidence visible on the reference machine,
the same eleven files give the same result except that this optional test
also runs when `SWG_DG_PPSRC` is set.

The guard and environment tests also check that this Python environment holds
exactly the pinned versions. The heavier gradient-matrix test
(`tests/ocean/validation/test_step_gradient_matrix.py`) is not part of this
self-test: it checks gradient health across many ocean configurations (one
row is the SWG card's one-step gradient), needs `optax`, `xarray` and
`pandas` (outside these pins), and one of its rows peaked at about 34 GB of
memory in the Round-D review. It runs in the
maintainers' lane battery, whose reference-machine log is cited in the
Round-D receipt; its one row that reads external SWG records now skips with
a stated reason when those records are absent.

## 2. Build `SWG_DG`

The builder starts from shipped `tests/SWG`, removes only `key_xios`, renders
the committed scalar-math architecture, and applies the committed two-line
`usrdef_zgr.F90` patch so the wet extent follows `rn_domsiz`.

```bash
bash "$TOOLS/build_swg_dg.sh" \
  --nemo-root "$NEMO_ROOT" --conda-prefix "$NEMO_PREFIX" \
  --manifest-dir "$REPRO/build-manifest" --python "$PY"
bash "$TOOLS/build_swg_dg.sh" \
  --nemo-root "$NEMO_ROOT" --conda-prefix "$NEMO_PREFIX" \
  --manifest-dir "$REPRO/build-manifest" --python "$PY" --execute
(cd "$REPRO/build-manifest" && \
  sha256sum -c swg_dg_build_manifest.sha256)
```

Preflight ends `PREFLIGHT_OK` and names the `ar` it will use; the build ends
`BUILD_DONE`. Preserve the self-contained build manifest. It pins NEMO
revision, compiler composition, architecture, archiver, patch, every case
source, the executable, and this build's derived namelist inventory.

Before `BUILD_DONE`, the build derives which namelist variables your
executable can read and refuses unless that equals the committed inventory
(`swg_dg_compiled_namelist_reads.json`). You can rerun that check on your
build at any time:

```bash
"$PY" -I "$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_namelist_inventory.py" \
  --source-root "$NEMO_ROOT/tests/SWG_DG/BLD/ppsrc/nemo"
```

Required line (measured on the reference build, the fresh rebuild, and the
shipped-SWG build of the reference host):

```text
SWG_DG_NAMELIST_INVENTORY PASS groups=27 fields=336 excluded_reads=2 matches_committed_reference=True
```

The inventory is derived, not hand-listed: every namelist `READ` in a routine
reachable from `nemo_gcm` in the preprocessed sources, whatever switch guards
the call, so it includes groups this deck switches off. The two excluded
reads are NEMO's own namelist self-test on a literal string.

## 3. Prove the variant is inert at 2,000 km

```bash
bash "$TOOLS/check_swg_dg_identity.sh" \
  --nemo-root "$NEMO_ROOT" --arch-name swg-dg-scalarmath \
  --output "$REPRO/case-identity" --python "$PY" \
  --conda-prefix "$NEMO_PREFIX"
bash "$TOOLS/check_swg_dg_identity.sh" \
  --nemo-root "$NEMO_ROOT" --arch-name swg-dg-scalarmath \
  --output "$REPRO/case-identity" --python "$PY" \
  --conda-prefix "$NEMO_PREFIX" --execute
```

The required result is:

```text
SWG_DG_IDENTITY BIT_IDENTICAL variables=41 differing=0
```

It proves the variant reproduces all 41 shipped-case output variables over ten
steps when `rn_domsiz=2000000`.

## 4. Prove NEMO rank-count independence

```bash
bash "$TOOLS/swg_dg_rankcheck.sh" \
  --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output "$REPRO/rank-control" --python "$PY" \
  --conda-prefix "$NEMO_PREFIX"
bash "$TOOLS/swg_dg_rankcheck.sh" \
  --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output "$REPRO/rank-control" --python "$PY" \
  --conda-prefix "$NEMO_PREFIX" --execute
```

Expected final block:

```text
SWG-DG-ROUND6-NEMO-RANK-CHECK
all_bit_identical=True variables=21
```

The checker stitches rank tiles by coordinates and compares every word of all
21 restart variables. The 2026-10-13 fresh run initially failed after both
NEMO arms completed because its bare interpreter imported another checkout;
that measured packaging defect is fixed by the mandatory `--python` plus the
checkout-binding preamble.

All MPI steps launch with `$NEMO_PREFIX/bin/mpirun`, the Open MPI 5.0.10 of
the lock, and add `--oversubscribe` (an Open MPI option). `--mpiexec` exists
for a site launcher of that same Open MPI; another MPI is untested.

## 5. Run NEMO smoke and three-year decks

The NEMO namelist patch and legoESM's `SWG_MADDISON_NAMELIST` are independent
definitions, not one source of truth. On completion, `swg_md_run.sh` feeds
the complete NEMO run directory to a cross-model deck guard. The guard takes
every one of the 336 fields of the derived inventory (27 namelist groups) and
resolves its value from `output.namelist.dyn` when NEMO echoes the group,
otherwise from NEMO's `namelist_ref` then `namelist_cfg` override order on the
two content-hashed run inputs (`nambdy` is read but never echoed; ten more
groups are switched off by this deck and so are not echoed either). Then:

- 147 fields have a legoESM card counterpart and are compared with the card,
  158 rows in all because `sn_cfctl` has 12 members;
- the other 189 fields have no card counterpart (MPI layout, T/S input files,
  runoff, and the switched-off groups). Each has a reason in the committed
  registry `swg_dg_nemo_only_namelist_fields.json` and is pinned to the
  reference run's value, so any change of the value the guard resolves
  refuses. For an echoed group that value is the EFFECTIVE value NEMO wrote
  after reading, not necessarily the raw input: NEMO adjusts some inputs
  before echoing them (for example `nn_hls = MAX(2, nn_hls)` in `mppini.f90`),
  so a raw `nn_hls = 1` in `namelist_cfg` echoes as 2 and passes, which is
  correct because the model then runs with 2. Groups NEMO does not echo are
  compared at the raw values read from `namelist_ref` and `namelist_cfg`;
- the guard refuses a field that is neither compared nor registered, a field
  it cannot resolve, an echoed group or field outside the inventory, an
  echoed group missing a field (a disappearing comparison row), a set of
  echoed groups different from the reference run's, a line inside a read
  group it cannot parse, a read group written twice in one namelist file
  (NEMO reads only the first block), and a card field that is neither
  matched by a read nor registered as card-only.

Only these two output controls may differ:

- `sn_cfctl.l_runstat`: NEMO completion output with no legoESM analogue; and
- `nn_stock`: annual NEMO restart tiles versus daily legoESM NPZ output.

What no namelist carries, and how it is bound: the card takes the reference
density 1026 kg/m³, the Earth radius 6,371,229 m and the rotation rate
(2π over NEMO's sidereal day) from legoESM's NEMO constants preset, and the
500 m layer thickness from the card builder; NEMO hard-codes the same values
in its SWE `nemogcm` initialisation, `phycst` and `usrdef_zgr`. Gravity is
`rn_modified_grav`, a compared namelist field. The 19 card fields with no
namelist counterpart are the four CPP keys (fixed by the build checks), the
two periodicity flags (hard-coded closed in SWG `usr_def_nam`), and 13 values
of the `namdrg`, `namdyn_spg`, `namzdf` and `namzdf_tke` groups, which the
SWE executable never reads. None of these is compared by the guard; they are
bound only by the pinned NEMO revision, the build manifest, and the annual
end-to-end identity.

```bash
NEMO_RUNS="$REPRO/nemo"
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --python "$PY" --conda-prefix "$NEMO_PREFIX" \
  --smoke dg_smoke 32
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --python "$PY" --conda-prefix "$NEMO_PREFIX" \
  --smoke dg_smoke 32 --execute

bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --python "$PY" --conda-prefix "$NEMO_PREFIX" \
  --three-year dg_3yr 32
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$NEMO_RUNS" --python "$PY" --conda-prefix "$NEMO_PREFIX" \
  --three-year dg_3yr 32 --execute
(cd "$NEMO_RUN" && sha256sum -c manifest.sha256)
```

The smoke is 51,840 steps. The three-year run is 1,892,160 steps with daily
means and restarts at 630,720, 1,261,440, and 1,892,160. Each command prints

```text
SWG_DG_DECK_GUARD PASS compiled_reads=336 groups=27 card_compared=147 nemo_only_pinned=189 comparison_rows=347 allowed_differences=2
```

before its final `NEMO_RUN_COMPLETE` line. That guard line was measured by
running the guard on the reference three-year run and on the fresh smoke and
three-year runs; no launcher run has yet printed it, because the fresh runs
predate the guard. Retain `output.namelist.dyn`, `deck_guard.json`,
`run.log`, `run.stat`, `time.step`, and both manifests.

Before spending GPU time, compare your 96 NEMO restart tiles with the hashes
committed in the three annual reports. A match means your NEMO side is the
reference NEMO side; a mismatch means a later departure starts on the NEMO
side (build, toolchain or deck), not in legoESM:

```bash
for year in 1 2 3; do
  "$PY" -c 'import json, sys
tiles = json.load(open(sys.argv[1]))["provenance"]["NEMO"]["restart_files_sha256"]
print("".join(f"{tiles[name]}  {name}\n" for name in sorted(tiles)), end="")' \
    "$LEGOESM_ROOT/docs/ocean/fidelity/testcases/swg_dg_year${year}_restart_identity.json"
done > "$REPRO/reference_nemo_restart_tiles.sha256"
(cd "$NEMO_RUN" && sha256sum --quiet -c "$REPRO/reference_nemo_restart_tiles.sha256") \
  && echo "NEMO_RESTART_TILES_MATCH_REFERENCE tiles=$(grep -c . "$REPRO/reference_nemo_restart_tiles.sha256")"
```

The fresh 2026-10-13 rebuild on the reference host prints
`NEMO_RESTART_TILES_MATCH_REFERENCE tiles=96`; a tile changed by one byte
prints `FAILED` for that tile and no match line.

## 6. Run legoESM

Pick an idle physical GPU explicitly; the example uses device 0.

```bash
PYTHON="$PY" "$LEGOESM_ROOT/scripts/run/run_swg_double_gyre_gpu.sh" \
  --gpu-device 0 --days 1095 --daily-means --output "$LEGO_RUN"
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); m=p["device_memory_at_completion"]; assert p["status"]=="complete" and p["completed_days"]==1095 and p["wall_seconds"]>0 and m["supported"] and m["peak_bytes_in_use"]>0; print("GPU_MANIFEST PASS wall_seconds=%.3f peak_bytes_in_use=%d"%(p["wall_seconds"],m["peak_bytes_in_use"]))' \
  "$LEGO_RUN/manifest.json"
```

Required prefixes:

```text
COMPLETED days=1095 backend=gpu certified_policy=False wall_seconds=
GPU_MANIFEST PASS wall_seconds=... peak_bytes_in_use=...
```

The daily-mean run also writes end-of-day states. The reference wall time was
8,117 s; the fresh manifest is the source for your runtime and peak allocator
memory.

The debugging fallback is explicit CPU fp64/libm and uses the formal
materialized-step composition. It cannot create daily means and is not the GPU
composition used for annual identity:

```bash
"$PY" scripts/run/run_swg_double_gyre.py --mode cpu-certified \
  --days 1 --output "$REPRO/lego/dg_cpu_1d"
```

The measured one-day reference manifest records 233.93357684463263 s. Linear
extrapolation is about 71.2 hours for 1,095 days; this is an estimate, not a
measured long CPU run.

## 7. Compare annual completed states

The comparison is exactly:

| legoESM | NEMO | Instant and scope |
|---|---|---|
| `ssh_m` | `sshn[0]` | completed step N, T wet mask |
| `u_m_s` | `un[0,0]` | completed SWE layer velocity, U wet mask |
| `v_m_s` | `vn[0,0]` | completed SWE layer velocity, V wet mask |

NEMO `uu_n`/`vv_n` are zero barotropic work arrays here, not evidence of the
same capture instant. The checker hashes them for provenance but does not
compare them to layer velocities. It also refuses wrong steps/days, incomplete
manifests, wrong masks/decks/dtypes, and non-finite data.

```bash
CHECK="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_restart_identity.py"
JAX_PLATFORMS=cpu "$PY" "$CHECK" --nemo-root "$NEMO_RUN" \
  --legoesm-npz "$LEGO_RUN/day0365.npz" --day 365 --step 630720 \
  --json "$REPRO/year1_restart_identity.json"
JAX_PLATFORMS=cpu "$PY" "$CHECK" --nemo-root "$NEMO_RUN" \
  --legoesm-npz "$LEGO_RUN/day0730.npz" --day 730 --step 1261440 \
  --json "$REPRO/year2_restart_identity.json"
JAX_PLATFORMS=cpu "$PY" "$CHECK" --nemo-root "$NEMO_RUN" \
  --legoesm-npz "$LEGO_RUN/day1095.npz" --day 1095 --step 1892160 \
  --json "$REPRO/year3_restart_identity.json"
```

The pinned reference composition prints:

```text
SWG_DG_RESTART_IDENTITY PASS day=365 step=630720 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
SWG_DG_RESTART_IDENTITY PASS day=730 step=1261440 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
SWG_DG_RESTART_IDENTITY PASS day=1095 step=1892160 fields=3 wet_words=1227520 unequal=0 max_abs=0.00000000000000000e+00
```

The three committed reports beside this guide are the citable artifacts for
that statement. They prove 1,227,520 wet words at each checkpoint, not a full
restart or dry-cell comparison.

Each report (and each climate report) records
`matches_2026_10_13_reference_build`. It is true only when your NEMO
executable, `namelist_cfg` and `namelist_ref` are byte-identical to the files
of the 2026-10-13 reference run. Expect `false` on your machine: even the
fresh rebuild on the reference host has a different executable hash and a
`namelist_cfg` that differs in comments only, while its restart tiles are
byte-identical. The flag records provenance; it is not a pass/fail result.
The tile check of step 5 and the annual word counts are.

## 8. Compute matched daily-mean climate statistics

```bash
CLIMATE="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_climate.py"
JAX_PLATFORMS=cpu "$PY" "$CLIMATE" \
  --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 1 --days 365 --output "$REPRO/climate_year1"
JAX_PLATFORMS=cpu "$PY" "$CLIMATE" \
  --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 366 --days 730 --output "$REPRO/climate_years2_3"
```

Each command ends with a line of this form:

```text
SWG_DG_CLIMATE_REPORT_WRITTEN window=1-365 label='year 1, spin-up' agreement_test=none output=...
SWG_DG_CLIMATE_REPORT_WRITTEN window=366-1095 label='years 2-3' agreement_test=none output=...
```

The instrument computes the statistics of both models and their differences
and writes them; it applies no agreement test, which `agreement_test=none`
states. No cross-toolchain acceptance bar exists yet: NEMO daily means are
fp32, and no sampling-uncertainty or post-decorrelation bar has been
registered. Retain and report every metric.

The reference statistics come from the two committed reference reports,
`swg_dg_climate_year1_reference_report.json` and
`swg_dg_climate_years2_3_reference_report.json` (basin-mean daily values in
m²/s²; mean, then minimum–maximum over the window; four significant digits):

| Window | Model | KE mean (min–max) | EKE mean (min–max) |
|---|---|---|---|
| days 1–365 | NEMO | 0.01827 (5.521e-06–0.02257) | 0.01195 (0.001571–0.01566) |
| days 1–365 | legoESM | 0.01827 (5.521e-06–0.02257) | 0.01195 (0.001571–0.01566) |
| days 366–1095 | NEMO | 0.02252 (0.02182–0.02309) | 0.0147 (0.01201–0.02035) |
| days 366–1095 | legoESM | 0.02252 (0.02182–0.02309) | 0.0147 (0.01201–0.02035) |

The minimum–maximum range is the reference trajectory's own variability over
the window, not an uncertainty bar. On the pinned reference trajectory the
two models agree to the digits shown; the measured maxima of the day-by-day
differences were:

| Window | Basin-mean KE | Basin-mean EKE |
|---|---:|---:|
| days 1–365 | `2.286230926795696e-11` | `1.1921534939784983e-11` |
| days 366–1095 | `2.2225409013199027e-11` | `1.402270660699756e-11` |

Across both windows, the measured maxima are
`2.286230926795696e-11` for kinetic energy and
`1.402270660699756e-11` for eddy kinetic energy.
These are measured reference values, not a cross-toolchain acceptance bar.

## 9. Render the matched movie

```bash
export MPLCONFIGDIR="$REPRO/matplotlib-cache"
JAX_PLATFORMS=cpu "$PY" \
  "$LEGOESM_ROOT/scripts/plot/swg_double_gyre_movie.py" \
  --nemo-root "$NEMO_RUN" --legoesm-root "$LEGO_RUN" \
  --start-day 1 --days 365 --fps 12 \
  --output "$REPRO/swg_dg_year1.mp4"
```

Expected line:

```text
SWG_DG_MOVIE COMPLETE frames=365 output=.../swg_dg_year1.mp4
```

The reference movie took 1,794 s. The two rows are NEMO and legoESM; columns
are relative vorticity/f, SSH, and speed with shared per-frame scales. The
tool verifies/hashes its inputs, starts ffmpeg before JAX threads, streams
frames, and atomically publishes the MP4. It never presents a partial encode
as complete.

## If annual identity fails: make and score steps 1–10

Do not change a tolerance. First run the NEMO tile check of step 5: if it
fails, the departure is on the NEMO side. Then make a one-rank NEMO restart at
every step:

```bash
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$REPRO/nemo" --python "$PY" --conda-prefix "$NEMO_PREFIX" \
  --short-ladder dg_ladder
bash "$TOOLS/swg_md_run.sh" --nemo-cfg "$NEMO_ROOT/tests/SWG_DG" \
  --output-root "$REPRO/nemo" --python "$PY" --conda-prefix "$NEMO_PREFIX" \
  --short-ladder dg_ladder --execute
```

Score the same outer-scan composition used by the long GPU driver, first on
CPU for a portable diagnostic:

```bash
LADDER="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_short_ladder.py"
"$PY" "$LADDER" --nemo-root "$REPRO/nemo/dg_ladder" \
  --backend cpu --composition outer-scan \
  --json "$REPRO/short_ladder_outer_scan.json"
```

On the reference machine this prints 30 zero-unequal rows and ends:

```text
SWG_DG_SHORT_LADDER EXACT composition=outer-scan backend=cpu first_difference=NONE
```

Score the operational device composition with a separate output path:

```bash
"$PY" "$LADDER" --nemo-root "$REPRO/nemo/dg_ladder" \
  --backend gpu --gpu-device 0 --composition outer-scan \
  --json "$REPRO/short_ladder_outer_scan_gpu.json"
```

The GPU form scores the long-run driver's own case under its native
transcendental policy. It has been tested only by a CPU emulation of an
initialised GPU backend (it reaches its first input), never on GPU hardware,
and no reference GPU ladder exists yet. The committed baseline below was
measured on CPU, so a GPU report that departs from it is information about
the GPU composition, not by itself a failure.

The report identifies the first differing step/field, unequal words, maximum
residual, region, hashes, and a binding one-ULP control. Never reuse the CPU
JSON path; the scorer refuses overwrites.

The formal materialized composition is a useful discriminator and is expected
to return exit 1 on the reference composition:

```bash
"$PY" "$LADDER" --nemo-root "$REPRO/nemo/dg_ladder" \
  --backend cpu --composition materialized \
  --json "$REPRO/short_ladder_materialized.json" || test "$?" -eq 1
```

Its first completed-state difference is:

```text
SWG_DG_SHORT_LADDER DEBT composition=materialized backend=cpu first_difference=step2.ssh_m bits=189
```

Compare both reports with the committed reference behavior:

```bash
"$PY" "$LADDER" --compare-reference \
  --outer-report "$REPRO/short_ladder_outer_scan.json" \
  --materialized-report "$REPRO/short_ladder_materialized.json"
```

The required reference-machine lines are:

```text
SWG_DG_SHORT_LADDER_BASELINE outer-scan same as reference
SWG_DG_SHORT_LADDER_BASELINE materialized same as reference
```

On another composition, the command instead prints the first step and field
whose counts or maximum residual depart from the reference. To compare the GPU
form, substitute `short_ladder_outer_scan_gpu.json` for `--outer-report`.

Three often-confused debt counts describe different reductions:

- **189** is the number of unequal wet SSH words in the materialized
  completed state at step 2. The short-ladder command above produces it.
- **1,278** is the number of unequal wet SSH words between legoESM's
  materialized stage-1 output and NEMO's stage-2 entry at step 2. It needs a
  NEMO build instrumented to write every stage entry, which this package does
  not make for colleagues, so it is reference-only evidence: comparison (iv)
  of `nemo_testcases_l1_swg_dg_round7_receipt.md`. The maintainers produced
  it with the operator-only acquisition tool
  `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg/run.sh --deck maddison --run`
  (which builds that instrumented executable with `makenemo`) followed by
  `nemo_testcase_l1_swg_dg_round7.py --root` on its instrumented output.
- **414,545** is the step-10 sum of materialized completed-state unequal words
  over SSH, U, and V (`278,318 + 3,143 + 133,084`). It is the accumulated
  endpoint of the same materialized-program debt, not a third comparison
  boundary. This command produces that total:

  ```bash
  COMPOSITION="$LEGOESM_ROOT/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_composition.py"
  "$PY" "$COMPOSITION" --root "$REPRO/nemo/dg_ladder" \
    --json "$REPRO/step10_composition_discriminator.json"
  ```

The committed compact baseline records all 30 rows for both program forms.

## Exactness policy (decided by the maintainer on 2026-10-13)

- The bar is exact (zero unequal words) for the prognostic fields `ssh`, `u`
  and `v`. No tolerance exists or is introduced.
- The one exception is the sign of zero in forcing fields: the 640 V-stress
  forcing words that differ only by the sign of zero. A machine check
  enforces that the exception covers forcing fields only; it refuses a
  signed-zero difference in a prognostic field.
- The last-bit difference of the formal step-by-step replay (the materialized
  composition: 189 words at step 2, 1,278 at the stage-2 entry, 414,545 by
  step 10) is recorded debt, carried openly. The compiled-day program used for
  the three-year comparison is exact at all three year ends.

The committed composition discriminator JSON was produced before this ruling,
so its `signed_zero_policy` field still carries the earlier
`reported_as_DEBT_pending_user_ruling` label; the tools now write the ruling.

## Honest expectation across machines

Bit identity depends on the physical transcription together with GNU Fortran
version and flags, the scalar-math architecture, `key_nosignedzero`, NEMO rank
composition (controlled above), JAX/jaxlib/XLA versions, and the GPU versus CPU
program composition. The compiled-day outer scan matches NEMO on this deck;
the formal per-step program differs in the last bit. On shipped SWG, the
opposite composition choice was the certified exact one.

A compiler, XLA, or GPU change may therefore introduce a last-bit difference.
This flow is chaotic, so individual eddies can decorrelate within weeks. After
that, annual word comparison is not informative; the matched daily-mean
climate statistics of step 8 are the comparison, read against the reference
values above. No cross-toolchain acceptance bar exists yet.

If identity fails, report the exact environment/build/run manifests, the NEMO
tile-check line, the first failing annual JSON, the 1–10-step outer-scan JSON
(and GPU device), the materialized discriminator JSON, and both climate
reports. That evidence separates deck, capture alignment, compiler, and
composition differences without moving a bar.

## What the 2026-10-13 fresh reproduction covered

The maintainers re-ran the NEMO half from a clean NEMO tree on the reference
host with the tools at lane commit `6aea0af1b`, an earlier state of this
package. It covered the build, the case identity, the rank control (whose
wrapper exited 1 on the import defect described in step 4; the comparison
was rescored by hand to `all_bit_identical=True`), the smoke run, and the
three-year run, whose 96 restart tiles are byte-identical to the reference.
It did not cover: creating either environment (it reused the reference conda
prefix and Python environment), the package self-test, the deck guard inside
the launcher (that launcher predated the guard, which was run on its outputs
afterwards), the namelist inventory, the tile-check command (run on its
outputs afterwards), or the current `--conda-prefix` interface.

The GPU half followed on 2026-10-14 with the tools at commit `d7d2e21f6` (the
platform fix on top of Round D), against the fresh NEMO run above, on the
idle reference V100S: the three-year legoESM run printed `COMPLETED
days=1095 backend=gpu` in 8,320.989 s with a peak of 437,028,608 bytes of GPU
memory (`GPU_MANIFEST PASS`); the deck guard of that commit printed `PASS
compiled_reads=131/131` (the 336-field Round-E guard was run on the same
NEMO run separately and also passes); the three annual checks printed
`SWG_DG_RESTART_IDENTITY PASS` at days 365, 730 and 1095 with 1,227,520 wet
words, 0 unequal and maximum difference 0 each (report SHA-256 `b325b65d...`,
`2c44440b...`, `35eef943...`); both climate reports were written; and the
movie printed `SWG_DG_MOVIE COMPLETE frames=365`. The first attempt at this
half stopped at start-up because JAX 0.10.0 refuses `JAX_PLATFORMS=gpu` on a
CUDA-only install (fixed in `d7d2e21f6`), and its first guard run refused
because the shell lacked `JAX_ENABLE_X64` (the guard now selects float64
itself). This half did not cover the Round-E tools themselves, the
environment creation, or a GPU other than the reference V100S.

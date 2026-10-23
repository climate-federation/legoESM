# SWG double-gyre reference environment

Bit identity was measured with the complete composition below. The
machine-readable inventory is `swg_double_gyre_environment.json`; measured
runtime evidence is in `swg_double_gyre_reference_costs.json`.

## Reconstruct NEMO

The source is the public NEMO repository, tag `5.0.2`, at full revision
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`:

```bash
git clone --branch 5.0.2 --single-branch \
  https://forge.nemo-ocean.eu/nemo/nemo.git "$NEMO_ROOT"
git -C "$NEMO_ROOT" checkout --detach \
  dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
test "$(git -C "$NEMO_ROOT" rev-parse HEAD)" = \
  dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796
```

For Linux x86-64, reproduce the exact conda-forge package builds used here:

```bash
conda create --prefix "$NEMO_PREFIX" --file \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/nemo-build-linux-64-explicit.txt"
```

That explicit file is normative for bit-identity attempts. It contains the
122 measured package/build URLs and the MD5 supplied by conda for every URL;
its SHA-256 is
`9aa5e72d56547edeaf27ad6514a58457d39940aa3eb09514b8526381e10c81f5`.
The companion
`nemo-build-environment.yml` omits build strings and is a readable, more
portable fallback when exact linux-64 artifacts are unavailable:

```bash
conda env create --prefix "$NEMO_PREFIX" --file \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/nemo-build-environment.yml"
```

The reference environment has GNU Fortran 15.2.0 build 19, Open MPI 5.0.10
`h67ed482_1`, HDF5 2.1.0 `mpi_openmpi_hb23c26a_9`, libnetcdf 4.10.0
`mpi_openmpi_h75687c8_4`, and netcdf-fortran 4.6.3
`mpi_openmpi_h8f6c4f5_0`. XIOS is not used. The committed architecture uses
scalar math and these material flags:

```text
FCFLAGS -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer
        -ffree-line-length-none -fallow-argument-mismatch
        -fno-tree-vectorize
CPP     cpp -Dkey_nosignedzero
CFLAGS  -O0
```

`build_swg_dg.sh` renders the conda prefix into the architecture and records
the resulting architecture, binary, case sources, compiler versions, flags,
and hashes in its self-contained build manifest.

## Reconstruct Python/JAX

First create the measured Python interpreter from conda-forge. The exact
reference build is `python 3.13.0 h9ebbce0_101_cp313`; do not rely on a
distribution-provided `python3.13`:

```bash
conda create --prefix "$PYTHON_PREFIX" --override-channels \
  --channel conda-forge 'python=3.13.0=h9ebbce0_101_cp313' pip
"$PYTHON_PREFIX/bin/python" -c \
  'import sys; assert sys.version_info[:3] == (3, 13, 0)'
"$PYTHON_PREFIX/bin/python" -m venv "$PYTHON_ENV"
PY="$PYTHON_ENV/bin/python"
"$PY" -m pip install -r \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/python-gpu-requirements.txt"
"$PY" -m pip install -r \
  "$LEGOESM_ROOT/scripts/data/nemo_swg_double_gyre/python-selftest-requirements.txt"
```

The 40-package fully pinned closure includes JAX, jaxlib, `jax-cuda12-plugin`, and
`jax-cuda12-pjrt` 0.10.0; NumPy 2.4.4; Equinox 0.13.7; every CUDA wheel build;
NetCDF4 plus certifi 2026.4.22; typing-extensions 4.15.0; SciPy; Matplotlib;
and imageio-ffmpeg. The separate self-test file pins the test tools of the
package self-test (pytest 9.0.3, pluggy 1.6.0, iniconfig 2.3.0, pygments
2.20.0, PyYAML 6.0.3); the two files together are a closed set. Every pin of
both files, and each file's SHA-256, is recorded in
`swg_double_gyre_environment.json`, and the environment test refuses a pin
that is not `name==version` at those versions. The package tools add their own
checkout's package roots before imports and print `CHECKOUT_IMPORT_OK` with
the selected Python, NumPy, and `legoesm.ocean` paths. They refuse an import
from another checkout. No editable legoESM install is required.

Neither `conda create` nor the two `pip install` commands above has been run
from scratch yet. What was checked: the explicit lock is byte-identical to
`conda list --explicit --md5` of the reference NEMO environment; every pin
equals the version installed in the reference Python environment; and the
package self-test ran in an environment holding exactly the 45 pinned
distributions, assembled from the reference environment's installed files.

## Host requirements

| Requirement | Value | Basis |
|---|---|---|
| glibc | 2.34 or newer (`ldd --version \| head -1`) | measured: `sysroot_linux-64` 2.34, `libsystemd0` and `libudev1` in the lock declare `__glibc >=2.34`; the reference NEMO executable uses `GLIBC_2.34` symbols |
| Archiver | host GNU binutils `ar` | read: the architecture's `%AR` is a plain `ar`; the conda environment provides only prefixed archivers |
| NVIDIA driver, tested | 580.126.20 | measured on the reference host |
| NVIDIA driver, CUDA 12.9 Update 1 | `>=575.57.08` | documented: NVIDIA release-note table of toolkit and driver versions |
| NVIDIA driver, CUDA 12 minor-version compatibility | `>=525` and `<580` | documented: NVIDIA minor-version compatibility table; JAX documents `>=525` |

The pinned cuDNN 9.21 is documented (cuDNN 9.21.0 support matrix) for compute
capability 7.5 and newer only, while the reference GPU is compute capability
7.0. The package's GPU path calls no convolution and the reference V100S run
completed, so this does not affect it.

## Reference machine and capacity

| Component | Measured reference |
|---|---|
| GPU | Tesla V100S-PCIE-32GB; driver 580.126.20 |
| CPU | Intel Xeon Gold 6242R |
| OS | Linux 5.14.0-611.36.1.el9_7.x86_64 |
| NEMO three-year storage | 56,985,134,712 bytes |
| legoESM daily-mean storage | 21,803,971,913 bytes |

Reserve at least 100 GiB of free output space. A 32 GiB GPU is the minimum
tested capacity, not a measured lower bound: smaller GPUs may work, but have
not been admitted. The Round-C driver records both `wall_seconds` and JAX's
`peak_bytes_in_use` allocator statistic in `manifest.json` and refuses a GPU
backend that cannot report that high-water value.

Exact floating-point identity is conditional on compiler versions and flags,
the scalar-math architecture, `key_nosignedzero`, JAX/XLA versions, and the
selected CPU/GPU program composition. Always retain build/run manifests with
comparison reports.

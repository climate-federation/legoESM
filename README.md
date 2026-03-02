# legoESM

**A Differentiable Earth System Model in JAX**

legoESM is a next-generation, fully differentiable Earth System Model spanning weather-to-climate timescales. Built from scratch in JAX, it enables end-to-end gradient computation for data assimilation, parameter estimation, and hybrid AI-physics modeling.

## Features

- **Differentiable-first**: End-to-end `jax.grad` through the full coupled model
- **Conservation as hard constraint**: Mass, energy, and momentum conserved via projection
- **Modular & swappable**: Standard tensor-in/tendency-out interface for AI or physics modules
- **Hardware-portable**: CPU, GPU (multi-GPU), TPU, and Apple Silicon
- **Multi-grid**: Cubed-sphere (primary), with support for icosahedral, lat-lon, spectral element

## Quick Start

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run Williamson Test Case 2
legoesm test williamson --case 2 --resolution 48 --days 5

# Run tests
pytest tests/
```

## Platform Notes

### Tested Compatibility Matrix

| Path | Hardware / Backend | MPI Runtime | Tested JAX | Tested mpi4jax | Status / Notes |
|---|---|---|---|---|---|
| Finite-volume dycores + ocean (single-process) | CPU (`jax` CPU backend) | N/A | `>=0.8,<0.10` | N/A | Regular unit/regression path |
| Finite-volume dycores + ocean (single-process) | Apple Silicon Metal (`jax-metal`) | N/A | `>=0.8,<0.10` | N/A | Supported for FV solvers (`float32`) |
| Spectral solvers (atmosphere/ocean) | CPU (`JAX_PLATFORMS=cpu`) | N/A | `>=0.8,<0.10` | N/A | Requires `float64`/`complex128` |
| Distributed MPI halo/reductions | CPU + OpenMPI (`mpirun`) | OpenMPI (4.x/5.x) | `>=0.8,<0.10` | `>=0.8,<0.9` | Validated with `mpirun -np 2/3/6` on `tests/distributed/test_halo_mpi.py` |
| Nightly long-run MPI ocean conservation | CPU + OpenMPI (`mpirun`) | OpenMPI (4.x/5.x) | `>=0.8,<0.10` | `>=0.8,<0.9` | Validated on `tests/distributed/test_ocean_mpi_conservation.py` |
| Multi-device scaling suite | CPU/GPU (if available) | Optional (for distributed checks) | `>=0.8,<0.10` | `>=0.8,<0.9` (MPI mode) | Use `scripts/run_parallel_validation.py` for strong/weak scaling and thresholds |

`legoesm.parallel.reductions` enforces MPI compatibility guardrails:
- Hard fail for legacy `mpi4jax<0.8` (incompatible token semantics).
- Warning for versions outside tested range; set `LEGOESM_MPI_STRICT_COMPAT=1` to make this a hard fail.

### Apple Silicon (Metal/MPS backend)

The **spectral solver** (Gaussian grid + spherical harmonic transforms) requires
`float64` and `complex128` arithmetic, which Apple's Metal backend does not support.

If you are on Apple Silicon and want to use the spectral solver, force the CPU backend:

```bash
JAX_PLATFORMS=cpu python your_script.py
```

The **finite-volume solvers** (cubed-sphere shallow water, hydrostatic primitive
equations) work on all backends including Metal, using `float32` precision.

To check your current JAX backend:

```bash
python -c "import jax; print(jax.default_backend())"
```

## Acknowledgments

legoESM bundles [jax-rrtmgp](https://github.com/climate-analytics-lab/jax-rrtmgp)
(Apache 2.0 license) for correlated-k radiation. jax-rrtmgp was developed by
Jeff Parker (Google), Duncan Watson-Parris (UCSD), and Juan Nathaniel (Columbia
University).

## License

MIT

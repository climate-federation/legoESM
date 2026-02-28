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

## License

MIT

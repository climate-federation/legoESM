"""Login-node-safe dtype probe for the RRTMGP fp32 optics fixes.

Verifies that the three float64 leaks fixed on branch ``fp32_mpas`` no
longer re-promote the optics computation to float64 when the inputs are
float32:

  1. ``gas_optics._mixing_fraction_interpolant`` — the relative-abundance
     reference grid (``jnp.linspace``) used to be ``dtype=jnp.float_``
     (float64 under ``jax_enable_x64``); it now follows ``f.dtype``.
  2. ``gas_optics.get_vmr`` — the global-mean VMR stack used to be
     ``dtype=jnp.float_``; it now follows ``lookup_gas_optics.vmr_ref.dtype``.
  3. ``cloud_optics._particle_size_interpolant`` — the particle-size
     reference grid (``jnp.linspace``) had no explicit dtype (float64
     under x64); it now follows ``f.dtype``.

This does NOT load the RRTMGP NetCDF/Zarr tables and does NOT JIT the full
g-point scan, so it is cheap enough for the Ginsburg login node
(single-core, < ~5 s).  ``jax_enable_x64`` is turned ON deliberately so
that the *default* (strong) dtype of a bare ``jnp`` allocation would be
float64 — this is the regime in which the old code leaked.  We then pass
float32 reference arrays and assert the interpolant weights / gathered
VMR come back float32.  A float64 control confirms the fp64 path is
unchanged.

The heavy end-to-end check (real tables, full ``RRTMGP.solve_columns``
with ``compute_fp32=True``, finite heating-rate / flux dtype assertions)
must run on a compute node via sbatch — it loads the gas/cloud tables and
JIT-compiles the 256+224 g-point scan, which is forbidden on the login
node.
"""

from __future__ import annotations

import os

# CPU only; keep XLA single-threaded so this is safe on a login node.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import dataclasses

import jax

# Turn x64 ON: this is the regime where a bare/`jnp.float_` allocation is
# strong-typed float64 and would re-promote a float32 input.  The fix must
# keep float32 inputs in float32 even here.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cloud_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import gas_optics


def _check_mixing_fraction_interpolant(dtype):
    """gas_optics._mixing_fraction_interpolant must follow f.dtype."""
    f = jnp.asarray([[0.1, 0.4], [0.7, 0.9]], dtype=dtype)
    interp = gas_optics._mixing_fraction_interpolant(f, n_mixing_fraction=9)
    w_low = interp.interp_low.weight
    w_high = interp.interp_high.weight
    assert w_low.dtype == dtype, (
        f"mixing-fraction interp_low weight dtype {w_low.dtype} != {dtype}"
    )
    assert w_high.dtype == dtype, (
        f"mixing-fraction interp_high weight dtype {w_high.dtype} != {dtype}"
    )
    assert bool(jnp.all(jnp.isfinite(w_low))) and bool(jnp.all(jnp.isfinite(w_high)))


def _check_particle_size_interpolant(dtype):
    """cloud_optics._particle_size_interpolant must follow f.dtype."""
    f = jnp.asarray([[2.5, 5.0], [10.0, 20.0]], dtype=dtype)
    d = cloud_optics._particle_size_interpolant(
        f, lower_bnd=2.5, upper_bnd=21.5, n_size=13
    )
    interp = d["r"]()
    w_low = interp.interp_low.weight
    w_high = interp.interp_high.weight
    assert w_low.dtype == dtype, (
        f"particle-size interp_low weight dtype {w_low.dtype} != {dtype}"
    )
    assert w_high.dtype == dtype, (
        f"particle-size interp_high weight dtype {w_high.dtype} != {dtype}"
    )
    assert bool(jnp.all(jnp.isfinite(w_low))) and bool(jnp.all(jnp.isfinite(w_high)))


@dataclasses.dataclass
class _FakeGasOptics:
    """Minimal stand-in exposing only what ``get_vmr`` touches."""

    idx_gases: dict
    vmr_ref: jax.Array


@dataclasses.dataclass
class _FakeVmrLib:
    global_means: dict


def _check_get_vmr(dtype):
    """gas_optics.get_vmr must follow lookup_gas_optics.vmr_ref.dtype."""
    idx_gases = {"h2o": 0, "co2": 1, "o3": 2}
    # vmr_ref dtype is what the stacked global-means must adopt.  After the
    # production cast (``_cast_optics_f64_to_f32``) this is float32 in fp32
    # mode; here we set it directly to exercise both dtypes.
    vmr_ref = jnp.ones((1, 1, 1), dtype=dtype)
    lookup = _FakeGasOptics(idx_gases=idx_gases, vmr_ref=vmr_ref)
    vmr_lib = _FakeVmrLib(global_means={"co2": 400e-6, "o3": 1e-7})
    species_idx = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    vmr = gas_optics.get_vmr(lookup, vmr_lib, species_idx, vmr_fields=None)
    assert vmr.dtype == dtype, f"get_vmr dtype {vmr.dtype} != {dtype}"
    assert bool(jnp.all(jnp.isfinite(vmr)))


def main() -> int:
    for dtype in (jnp.float32, jnp.float64):
        _check_mixing_fraction_interpolant(dtype)
        _check_particle_size_interpolant(dtype)
        _check_get_vmr(dtype)
        print(f"[ok] all dtype probes passed for {jnp.dtype(dtype).name}")
    print("PASS: fp32 optics dtype threading verified (x64 enabled).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

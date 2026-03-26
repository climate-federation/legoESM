"""Multi-GPU scaling tests using JAX virtual devices.

Simulates 6 CPU devices via ``--xla_force_host_platform_device_count=6``
to verify that face-sharded dynamics produces identical results to
single-device execution.

Because JAX device count must be set before initialization, each
multi-device test runs in a subprocess with the appropriate XLA flag.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

# The project root (for PYTHONPATH)
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _run_virtual_gpu_script(script: str, n_devices: int = 6) -> dict:
    """Run a Python script in a subprocess with virtual JAX devices.

    Sets ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` so that
    JAX sees N virtual CPU devices, enabling multi-device testing on a
    single machine.

    The script must print a JSON dict on its last line.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    existing_flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = (
        f"{existing_flags} --xla_force_host_platform_device_count={n_devices}"
    ).strip()
    env["PYTHONPATH"] = (
        os.path.join(_PROJECT_ROOT, "src")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    if result.returncode != 0:
        pytest.fail(
            f"Virtual GPU subprocess failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    # Parse the last non-empty line as JSON
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        pytest.fail(
            f"No output from virtual GPU subprocess.\n"
            f"--- stderr ---\n{result.stderr}"
        )
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        pytest.fail(
            f"Could not parse JSON from subprocess output:\n"
            f"{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


# ======================================================================
# Test 1: Device mesh creation with 6 virtual devices
# ======================================================================

class TestVirtualDeviceMesh:
    def test_6_device_mesh_creation(self):
        """create_device_mesh(6) produces a mesh with 6 devices."""
        out = _run_virtual_gpu_script("""
            import json
            import jax
            from legoesm.parallel.mesh import create_device_mesh

            n_dev = jax.device_count()
            config = create_device_mesh(n_devices=6)
            result = {
                "n_devices": n_dev,
                "config_n_devices": config.n_devices,
                "has_mesh": config.mesh is not None,
                "tiling": list(config.tiling),
                "mesh_shape": list(config.mesh.devices.shape) if config.mesh else [],
            }
            print(json.dumps(result))
        """)
        assert out["n_devices"] == 6
        assert out["config_n_devices"] == 6
        assert out["has_mesh"] is True
        assert out["tiling"] == [1, 1]
        assert out["mesh_shape"] == [6]


# ======================================================================
# Test 2: Shard/gather roundtrip preserves data on 6 devices
# ======================================================================

class TestShardGatherMultiDevice:
    def test_shard_gather_roundtrip_6_devices(self):
        """Shard state to 6 devices, gather back: exact roundtrip."""
        out = _run_virtual_gpu_script("""
            import json
            import jax
            import jax.numpy as jnp
            import numpy as np
            from legoesm.parallel.mesh import create_device_mesh
            from legoesm.parallel.sharded_dynamics import shard_state, gather_state

            config = create_device_mesh(n_devices=6)
            N = 8
            data = jnp.arange(6 * N * N, dtype=jnp.float64).reshape(6, N, N)
            state = {"field": data, "scalar": jnp.array(42.0)}

            sharded = shard_state(state, config)
            gathered = gather_state(sharded, config)

            field_match = bool(np.allclose(
                np.array(gathered["field"]), np.array(data)
            ))
            scalar_match = float(gathered["scalar"]) == 42.0

            # Check that sharded data is actually distributed
            sharding = sharded["field"].sharding
            is_sharded = not sharding.is_fully_replicated

            result = {
                "field_match": field_match,
                "scalar_match": scalar_match,
                "is_sharded": is_sharded,
                "n_devices": config.n_devices,
            }
            print(json.dumps(result))
        """)
        assert out["field_match"] is True
        assert out["scalar_match"] is True
        assert out["is_sharded"] is True
        assert out["n_devices"] == 6


# ======================================================================
# Test 3: Sharded shallow water step matches single-device
# ======================================================================

class TestShardedDynamics:
    def test_sharded_jit_operators_match_single_device(self):
        """Gradient/divergence on face-sharded data matches single-device."""
        out = _run_virtual_gpu_script("""
            import json
            import jax
            import jax.numpy as jnp
            from legoesm.grids.cubed_sphere import create_cubed_sphere
            from legoesm.core.field import Field
            from legoesm.core.operators import gradient_x, divergence
            from legoesm.parallel.mesh import create_device_mesh, shard_pytree

            N = 8
            grid = create_cubed_sphere(N)

            # Smooth test field
            data = jnp.sin(grid.lon) * jnp.cos(grid.lat)
            field = Field(data, name="f", dims=("face","x","y"), units="1")

            # Single-device gradient
            gx_single = gradient_x(field, grid)

            # Sharded gradient
            dev_config = create_device_mesh(n_devices=6)
            data_sharded = shard_pytree(data, dev_config)
            field_sharded = Field(data_sharded, name="f",
                                  dims=("face","x","y"), units="1")

            @jax.jit
            def compute_grad(f):
                return gradient_x(f, grid)

            gx_sharded = compute_grad(field_sharded)

            err = float(jnp.max(jnp.abs(gx_sharded.data - gx_single.data)))
            is_sharded = not data_sharded.sharding.is_fully_replicated

            result = {
                "grad_err": err,
                "is_sharded": is_sharded,
                "finite": bool(jnp.all(jnp.isfinite(gx_sharded.data))),
            }
            print(json.dumps(result))
        """)
        assert out["finite"] is True
        assert out["is_sharded"] is True
        assert out["grad_err"] < 1e-10, f"gradient mismatch: {out['grad_err']}"


# ======================================================================
# Test 4: Sharded halo exchange matches single-device
# ======================================================================

class TestShardedHalo:
    def test_pad_halo_sharded_matches_local(self):
        """pad_halo on face-sharded data matches single-device result."""
        out = _run_virtual_gpu_script("""
            import json
            import jax
            import jax.numpy as jnp
            import numpy as np
            from legoesm.grids.cubed_sphere import create_cubed_sphere
            from legoesm.grids.halo import pad_halo
            from legoesm.parallel.mesh import create_device_mesh, shard_pytree

            N = 8
            grid = create_cubed_sphere(N)
            dev_config = create_device_mesh(n_devices=6)

            # Smooth test field (face index + gradient)
            data = jnp.zeros((6, N, N))
            for f in range(6):
                data = data.at[f].set(
                    (f + 1.0) + 0.1 * jnp.arange(N)[:, None]
                )

            # Single-device halo
            padded_single = pad_halo(data, interp_offsets=grid.halo_interp_offsets)

            # Sharded halo
            data_sharded = shard_pytree(data, dev_config)
            @jax.jit
            def sharded_pad(d):
                return pad_halo(d, interp_offsets=grid.halo_interp_offsets)
            padded_sharded = sharded_pad(data_sharded)

            err = float(jnp.max(jnp.abs(padded_sharded - padded_single)))
            is_sharded = not data_sharded.sharding.is_fully_replicated

            result = {
                "max_error": err,
                "is_sharded": is_sharded,
                "shapes_match": padded_sharded.shape == padded_single.shape,
            }
            print(json.dumps(result))
        """)
        assert out["shapes_match"] is True
        assert out["is_sharded"] is True
        assert out["max_error"] < 1e-10, f"halo mismatch: {out['max_error']}"


# ======================================================================
# Test 5: Global integral with psum across 6 devices
# ======================================================================

class TestShardedReductions:
    def test_global_integral_sharded_matches_single(self):
        """global_integral on sharded data uses psum correctly."""
        out = _run_virtual_gpu_script("""
            import json
            import jax
            import jax.numpy as jnp
            from legoesm.grids.cubed_sphere import create_cubed_sphere
            from legoesm.core.field import Field
            from legoesm.core.operators import global_integral
            from legoesm.parallel.mesh import create_device_mesh, shard_pytree

            N = 8
            grid = create_cubed_sphere(N)

            # Constant field = 1 → integral should be total_area (= 4*pi*R^2)
            ones = Field(
                jnp.ones((6, N, N), dtype=jnp.float64),
                name="ones", dims=("face","x","y"), units="1",
            )

            # Single device
            integral_single = float(global_integral(ones, grid))

            # Create mesh and shard
            dev_config = create_device_mesh(n_devices=6)
            ones_sharded = Field(
                shard_pytree(ones.data, dev_config),
                name="ones", dims=("face","x","y"), units="1",
            )

            # Sharded integral (should use psum)
            integral_sharded = float(global_integral(ones_sharded, grid))

            expected = 4.0 * 3.141592653589793 * grid.radius ** 2
            rel_err_single = abs(integral_single - expected) / expected
            rel_err_sharded = abs(integral_sharded - expected) / expected
            match_err = abs(integral_sharded - integral_single) / abs(integral_single)

            result = {
                "integral_single": integral_single,
                "integral_sharded": integral_sharded,
                "rel_err_single": rel_err_single,
                "rel_err_sharded": rel_err_sharded,
                "match_err": match_err,
            }
            print(json.dumps(result))
        """)
        assert out["rel_err_single"] < 0.01, f"single: {out['rel_err_single']}"
        assert out["rel_err_sharded"] < 0.01, f"sharded: {out['rel_err_sharded']}"
        assert out["match_err"] < 1e-10, f"single vs sharded: {out['match_err']}"


# ======================================================================
# Test 6: Compiled segment on 6 devices
# ======================================================================

class TestShardedCompiledSegment:
    def test_sharded_lax_scan_matches_single_device(self):
        """jax.lax.scan on face-sharded state matches single-device."""
        out = _run_virtual_gpu_script("""
            import json
            import jax
            import jax.numpy as jnp
            from legoesm.grids.cubed_sphere import create_cubed_sphere
            from legoesm.grids.halo import pad_halo
            from legoesm.parallel.mesh import create_device_mesh, shard_pytree
            from legoesm.parallel.sharded_dynamics import gather_state

            N = 8
            grid = create_cubed_sphere(N)

            # Simple diffusion step using pad_halo (tests cross-face comm)
            def diffusion_step(state, _):
                padded = pad_halo(state, interp_offsets=grid.halo_interp_offsets)
                lap = (
                    padded[:, 2:, 1:-1] + padded[:, :-2, 1:-1]
                    + padded[:, 1:-1, 2:] + padded[:, 1:-1, :-2]
                    - 4.0 * state
                )
                return state + 0.1 * lap, None

            # Initial condition
            data = jnp.sin(grid.lon) * jnp.cos(grid.lat)

            # Single-device scan (10 steps)
            result_single, _ = jax.lax.scan(diffusion_step, data, None, length=10)

            # Sharded scan (6 devices)
            dev_config = create_device_mesh(n_devices=6)
            data_sharded = shard_pytree(data, dev_config)

            @jax.jit
            def sharded_scan(d):
                return jax.lax.scan(diffusion_step, d, None, length=10)[0]

            result_sharded = sharded_scan(data_sharded)

            err = float(jnp.max(jnp.abs(result_sharded - result_single)))
            is_sharded = not data_sharded.sharding.is_fully_replicated
            finite = bool(
                jnp.all(jnp.isfinite(result_single))
                and jnp.all(jnp.isfinite(result_sharded))
            )

            result = {
                "err": err,
                "is_sharded": is_sharded,
                "finite": finite,
            }
            print(json.dumps(result))
        """)
        assert out["finite"] is True
        assert out["is_sharded"] is True
        # Allow float32-level tolerance: cross-device reduction order
        # can differ, causing ~1e-7 differences from accumulation reordering.
        assert out["err"] < 1e-6, f"scan mismatch: {out['err']}"

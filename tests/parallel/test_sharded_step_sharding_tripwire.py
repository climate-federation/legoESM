"""Sharding tripwire for the compiled sharded step + timed scan runner.

Mechanical guard for the single-process multi-GPU replication bug: the
Levante scaling bench showed exactly 0.500 efficiency from 1 -> 2 GPUs
at identical wall time because the timed executable ran fully
replicated — every device computed the whole globe — while the
warning-only sharding check stayed silent.  The fix pins explicit
``in_shardings``/``out_shardings`` on both the inner compiled step
(``CompiledShardedStep``) and the outer timed scan runner, and adds the
loud ``_assert_expected_sharding`` tripwire in the bench.

This test forces 2 CPU devices and asserts, with the SAME tripwire
function the benchmark runs (imported from
``scripts.bench.run_levante_gpu_scaling``), that:

* ``make_sharded_step``'s compiled executable keeps face-leading state
  leaves face-sharded after stepping (2 addressable shards, per-shard
  leading dim 3 = 6 faces / 2 devices);
* a jitted 3-step ``lax.scan`` runner with explicit in/out shardings —
  what the benchmark actually times — also keeps the carry face-sharded;
* the SPMD halo backend is active after ``make_sharded_step`` (the
  bench asserts this instead of re-activating it);
* the tripwire is non-vacuous: it RAISES on a deliberately replicated
  face-leading leaf (synthetic violation), and is a no-op on
  single-device configs (zero behavior change for 1 GPU).

Run standalone (canonical, deterministic device count)::

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \\
        .venv/bin/python -m pytest \\
        tests/parallel/test_sharded_step_sharding_tripwire.py -v

In a full-suite run JAX may already be initialized before this module
imports (the ``setdefault`` below is then a no-op) — the multi-device
tests skip unless >=2 CPU devices are visible, same pattern as
``tests/parallel/test_ppermute_halo_exchange.py``.
"""

import os

# Must be set BEFORE the first jax backend initialization in the process
# (same pattern as tests/parallel/test_ppermute_halo_exchange.py).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

N_DEVICES = 2
N_GRID = 24
N_LEV = 4
DT = 450.0  # CFL-stable at C24 with adaptive nu4 (same as SPMD step tests)


def _need_devices(n: int):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices "
            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
        )


def _build_tiny_cubed_sphere_case():
    """Tiny C24/L4 CD-grid baroclinic-wave case (mirrors the bench setup)."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.core.cfl import (
        adaptive_hyperdiff_coeff, estimate_min_dx_cubed_sphere,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
        hydrostatic_to_fv3,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init

    grid = create_cubed_sphere(N_GRID)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(N_LEV)
    dx_min = estimate_min_dx_cubed_sphere(N_GRID)
    nu4 = adaptive_hyperdiff_coeff(dx_min, DT, order=4, safety=0.5)
    cfg = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=nu4, hyperdiff_ps_coeff=nu4,
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=True, zero_mean_ps_tendency=False,
        time_integrator="ssp_rk3",
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
    return model, hydrostatic_to_fv3(state_cc, cdgrid)


class TestShardedStepShardingTripwire:
    """The compiled step and the timed scan runner must keep face sharding."""

    def test_compiled_step_and_scan_runner_stay_face_sharded(self):
        _need_devices(N_DEVICES)
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree
        from legoesm.parallel.sharded_dynamics import (
            CompiledShardedStep, create_output_shardings, make_sharded_step,
        )
        from legoesm.parallel.cubesphere_exchange import (
            deactivate_spmd_halo_backend,
        )
        from legoesm.grids.halo import get_halo_backend
        from scripts.bench.run_levante_gpu_scaling import (
            _assert_expected_sharding,
        )

        model, s0 = _build_tiny_cubed_sphere_case()
        # Anchor target mass on the CONCRETE initial state. step() lazily
        # computes it on first call, but here the first call runs under
        # CompiledShardedStep's jit, so the lazy path would write a traced
        # float64[] onto the model — a leaked tracer that explodes with
        # UnexpectedTracerError when the scan runner re-traces step().
        model.set_target_mass(model.compute_mass(s0))
        dev_config = create_device_mesh(n_devices=N_DEVICES)
        assert dev_config.mesh is not None and dev_config.n_devices == N_DEVICES

        step_fn = make_sharded_step(model, dev_config, n=N_GRID, nlev=N_LEV)
        try:
            # Multi-device path must yield the cached compiled step and
            # must have activated the explicit SPMD halo backend (the
            # bench asserts this instead of re-activating it).
            assert isinstance(step_fn, CompiledShardedStep)
            assert get_halo_backend() == "spmd"

            state = shard_pytree(s0, dev_config)

            # --- inner compiled step (now jitted with in_shardings) ---
            state = step_fn(state, DT)
            jax.block_until_ready(jax.tree.leaves(state))
            _assert_expected_sharding(
                state, dev_config, where="after compiled step",
            )

            # Belt-and-braces: every face-leading leaf is genuinely
            # split — 2 addressable shards of 3 faces each, never a
            # full-globe replica per device.
            face_leaves = [
                leaf for leaf in jax.tree.leaves(state)
                if isinstance(leaf, jax.Array)
                and leaf.ndim >= 1 and leaf.shape[0] == 6
            ]
            assert face_leaves, "state has no face-leading leaves?"
            for leaf in face_leaves:
                shards = leaf.addressable_shards
                assert len(shards) == N_DEVICES
                for shard in shards:
                    assert shard.data.shape[0] == 6 // N_DEVICES

            # --- outer timed scan runner (what the bench measures) ---
            shardings = create_output_shardings(state, dev_config)
            input_dtypes = jax.tree.map(
                lambda x: x.dtype if hasattr(x, "dtype") else None, state)

            def _run(st):
                def _body(carry, _):
                    new = step_fn(carry, DT)
                    new = jax.tree.map(
                        lambda x, d: x.astype(d)
                        if d is not None and hasattr(x, "astype") else x,
                        new, input_dtypes,
                    )
                    return new, None
                return jax.lax.scan(_body, st, None, length=3)[0]

            scan_runner = jax.jit(
                _run, in_shardings=(shardings,), out_shardings=shardings,
            )
            out = scan_runner(state)
            jax.block_until_ready(jax.tree.leaves(out))
            _assert_expected_sharding(
                out, dev_config, where="after 3-step scan",
            )
        finally:
            deactivate_spmd_halo_backend()

    def test_tripwire_raises_on_replicated_state(self):
        """Synthetic violation: the tripwire is non-vacuous.

        A face-leading leaf deliberately replicated across the mesh
        (the exact signature of the 1->2 GPU bug) must raise
        RuntimeError; the correctly face-sharded control must pass.
        """
        _need_devices(N_DEVICES)
        from legoesm.parallel.mesh import create_device_mesh
        from scripts.bench.run_levante_gpu_scaling import (
            _assert_expected_sharding,
        )

        dev_config = create_device_mesh(n_devices=N_DEVICES)
        mesh = dev_config.mesh
        arr = jnp.ones((6, 4, 4, 3))

        good = {
            "T": jax.device_put(arr, NamedSharding(mesh, P("face"))),
            "scalar": jax.device_put(
                jnp.asarray(1.0), NamedSharding(mesh, P()),
            ),
        }
        # Control: face-sharded leaves + replicated scalar pass.
        _assert_expected_sharding(good, dev_config, where="synthetic-good")

        bad = dict(good)
        bad["T"] = jax.device_put(arr, NamedSharding(mesh, P()))  # replicated
        with pytest.raises(RuntimeError, match="does not match expected"):
            _assert_expected_sharding(bad, dev_config, where="synthetic-bad")

    def test_tripwire_noop_on_single_device(self):
        """Zero behavior change for 1 GPU: mesh-less configs early-return."""
        from legoesm.parallel.mesh import create_device_mesh
        from scripts.bench.run_levante_gpu_scaling import (
            _assert_expected_sharding,
        )

        cfg = create_device_mesh(n_devices=1)
        assert cfg.mesh is None
        # Unsharded leaves would fail every check were the guard absent.
        _assert_expected_sharding(
            {"x": jnp.ones((6, 4, 4))}, cfg, where="single-device",
        )
        _assert_expected_sharding(None, None, where="no-config")

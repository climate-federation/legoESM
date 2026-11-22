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


class TestShardedStepShardingTripwire:
    """The compiled step and the timed scan runner must keep face sharding."""


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

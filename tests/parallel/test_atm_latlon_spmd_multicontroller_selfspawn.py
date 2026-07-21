"""SELF-SPAWNING multi-controller equivalence gate for the atm lat-band SPMD step.

Companion to tests/parallel/test_atm_latlon_spmd_multicontroller.py (the
launcher-gated variant that needs `srun -n 2` + LEGOESM_JAX_DISTRIBUTED_TEST=1
and therefore never runs in plain pytest CI): THIS variant spawns its own two
worker processes, so the federation path is exercised on every CI run, with
port-race retries via multihost_harness.

Federates TWO OS processes into one JAX program (jax.distributed, CPU/Gloo —
the same federation layer NCCL uses on GPU nodes) and asserts the 2-band
shard_map trajectory matches the single-device serial reference to the SAME
tolerances as the single-process gate (test_atm_latlon_spmd_step.py:
rtol=1e-6 / atol=1e-9 over 3 RK3 steps — the FV-PPM cut-truncation bound; a
real decomposition bug is O(1e-3)).

What this pins beyond the single-process gate: the multi-controller
construction path — ``shard_state_atm_latlon``'s ``device_put`` onto a mesh
spanning NON-addressable devices, cross-process ppermute/psum inside the
jitted shard_map, and ``gather_state_atm_latlon``'s replication gather — i.e.
exactly the pieces a Derecho/Levante multi-node NCCL run exercises.

DUAL-USE FILE: run under pytest it spawns two subprocesses of ITSELF
(``python this_file.py <rank> <port>``), each executing ``_worker``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

N_PROC = 2
N_LAT = 16      # divisible by N_PROC
N_LON = 16
NLEV = 4
N_STEPS = 3
DT = 100.0
RTOL, ATOL = 1e-6, 1e-9   # single-process gate tolerances


def _worker(rank: int, port: int) -> int:
    import jax

    jax.distributed.initialize(
        coordinator_address=f"localhost:{port}",
        num_processes=N_PROC,
        process_id=rank,
    )
    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np

    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState,
        CGridLatLonPrimitiveEquationConfig,
        CGridLatLonPrimitiveEquationModel,
    )
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        gather_state_atm_latlon,
        make_sharded_atm_latlon_step,
        shard_state_atm_latlon,
    )
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    assert jax.process_count() == N_PROC, jax.process_count()
    assert len(jax.devices()) == N_PROC, jax.devices()

    # Deterministic identical build on every process (the multi-controller
    # contract: device_put slices the same host-global array per process).
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,                # exercises the cross-process psum
        use_polar_filter=True,
        use_ppm_transport=True,
        time_integrator="ssp_rk3",
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(31337)
    eps = 1.0e-3
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(eps * rng.standard_normal((N_LAT + 1, N_LON, NLEV))),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((N_LAT, N_LON, NLEV))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((N_LAT, N_LON))),
        phis=jnp.zeros((N_LAT, N_LON)),
    )
    v0 = np.array(state.v)
    v0[0] = 0.0
    v0[-1] = 0.0
    state = state._replace(v=jnp.asarray(v0))

    # Serial reference: identical on every process (no collectives).
    s = state
    for _ in range(N_STEPS):
        s, _ = model._step_cgrid(s, DT, target_mass=None, physics_fn=None)
    serial_out = s

    # Multi-controller SPMD: 2 bands, one device per PROCESS.
    mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=("lat",))
    sharded_step = make_sharded_atm_latlon_step(model, mesh)
    sc = shard_state_atm_latlon(state, mesh)
    for _ in range(N_STEPS):
        sc = sharded_step(sc, DT)
    spmd_out = gather_state_atm_latlon(sc, mesh)  # fully replicated

    failures = []
    for field in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(spmd_out, field))   # replicated -> host OK
        b = np.asarray(getattr(serial_out, field))
        if a.shape != b.shape:
            failures.append(f"{field}: shape {a.shape} vs {b.shape}")
            continue
        if not np.allclose(a, b, rtol=RTOL, atol=ATOL):
            failures.append(
                f"{field}: max|diff|={float(np.max(np.abs(a - b))):.3e}")
    if failures:
        print(f"rank{rank} MULTIHOST PARITY FAIL: " + "; ".join(failures),
              flush=True)
        return 1
    print(f"rank{rank} multihost parity OK", flush=True)
    return 0


def test_atm_latlon_spmd_two_process_selfspawn_matches_serial(tmp_path):
    from multihost_harness import run_federated

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env.pop("XLA_FLAGS", None)  # 1 real CPU device per process, no virtuals

    def build_cmd(rank, port):
        return [sys.executable, str(Path(__file__).resolve()), str(rank),
                str(port)]

    rcs, outs = run_federated(build_cmd, N_PROC, env, timeout_s=420)
    for rank, (rc, out) in enumerate(zip(rcs, outs)):
        assert rc == 0, (
            f"rank {rank} exited {rc}\n--- output ---\n{out[-4000:]}"
        )
        assert "multihost parity OK" in out


if __name__ == "__main__":
    raise SystemExit(_worker(int(sys.argv[1]), int(sys.argv[2])))

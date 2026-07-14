"""#921 regression: multi-process MIXED-CLIQUE (psum + ppermute) rendezvous gate.

Guards the collective pattern that deadlocked the 24-GPU closed-loop tiled-cube
lane (`scripts/cluster/scaling_derecho/cube_tiled_step.pbs`).  The single-shot
tiled step is halo-only — cliques built solely from ``jax.lax.ppermute``.  The
CLOSED-LOOP step adds ONE extra collective: the dry-mass fixer's GLOBAL
all-reduce ``jax.lax.psum(..., axis_name=("face","tile_i","tile_j"))``
(`packages/core/legoesm/parallel/tiled_production_cdgrid.py`
:func:`_tile_fix_ps_mass_delta`, ~line 2477) layered on top of the halo
``jax.lax.ppermute(send_buf, ("face","tile_i","tile_j"), perm)``
(`packages/core/legoesm/parallel/cubesphere_exchange.py`, ~line 581).  Under XLA
GPU defaults the all-reduce clique and the permute cliques init concurrently and
can be ordered differently per rank -> NCCL comm-init deadlock (issue #921).

This test compiles+runs the SAME mesh (``(6, kt, kt)`` with kt=2 -> 24 devices,
axes ``("face","tile_i","tile_j")``) and the SAME collective mix — 4 halo
ppermute rounds + 1 global psum — federated across TWO OS processes launched by
``mpiexec -n 2``, each holding 12 virtual CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=12`` -> 24 global).  It
asserts the mixed-collective program compiles, the cross-process rendezvous
completes, and the result is finite.

HONEST CAVEAT (why this is necessary, not sufficient): CPU JAX uses a NON-NCCL
collectives backend (gloo / MPI), so this guards the multi-process mixed-clique
RENDEZVOUS + ordering path — a real "does the extra global all-reduce clique
break multi-controller execution?" gate — but it does NOT reproduce the
NCCL-specific concurrent comm-init race the ``XLA_FLAGS`` serialize-flags fix
addresses.  The definitive gate remains a real multi-GPU run of
``cube_tiled_step.pbs`` (confirm the mechanism with ``NCCL_DEBUG_SUBSYS=INIT``).

DUAL-USE FILE: run under pytest it launches ``mpiexec -n 2`` on ITSELF
(``python this_file.py worker``), each rank executing :func:`_worker`.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

# Real closed-loop tiled-cube mesh: (6, kt, kt) with kt=2 -> 24 devices, the
# exact axis names the fix_mass psum and the halo ppermute both reduce over.
MESH_SHAPE = (6, 2, 2)
AXES = ("face", "tile_i", "tile_j")
N_GLOBAL_DEVICES = 6 * 2 * 2  # 24
DEVICES_PER_PROC = 12         # -> 2 processes
N_PROC = N_GLOBAL_DEVICES // DEVICES_PER_PROC
NH = 6                        # per-tile halo-exchange face size
N_STEPS = 4                   # repeated collective rounds


def _shift_perm(n: int, stride: int) -> list[tuple[int, int]]:
    """A cyclic-shift ppermute perm: device ``i`` sends to ``(i+stride) % n``.

    A valid permutation (bijection) for ANY stride.  Its cycle structure
    depends on ``gcd(n, stride)``: strides coprime to ``n`` (e.g. +-1) give one
    ``n``-cycle; ``stride=6`` on ``n=24`` gives six disjoint 4-cycles.  The
    mixed-clique step below deliberately uses BOTH kinds so the concurrent
    permute cliques span a range of sizes (mirroring the real halo's many
    small device-pair cliques) alongside the single 24-way psum clique.
    """
    return [(i, (i + stride) % n) for i in range(n)]


def _install_macos_gloo_shim() -> None:
    """macOS-only: make JAX's gloo CPU collectives bind loopback.

    On macOS the machine hostname (``mac.lan``) is not resolvable by gloo's
    ``getaddrinfo`` (only by the higher-level mDNS resolver), so the default
    gloo device init raises ``Unable to find address for: <host>`` and EVERY
    multi-process CPU-collectives test dies before any collective runs (the
    repo's existing ``*_multicontroller_selfspawn.py`` gates hit this too).
    JAX's ``make_gloo_tcp_collectives`` accepts ``hostname``/``interface`` but
    ``xla_bridge`` never passes them; we wrap it to bind ``127.0.0.1`` on the
    loopback iface.  This changes only the gloo BIND address (network
    plumbing), never the collective semantics or the rendezvous ordering under
    test.  Linux CI resolves its hostname fine and never installs the shim.
    """
    if sys.platform != "darwin":
        return
    from jax._src.lib import xla_client

    orig = xla_client._xla.make_gloo_tcp_collectives

    def _wrapped(distributed_client, hostname=None, interface=None):
        return orig(distributed_client=distributed_client,
                    hostname="127.0.0.1", interface="lo0")

    xla_client._xla.make_gloo_tcp_collectives = _wrapped


def _worker(rank: int, size: int, port: int) -> int:
    _install_macos_gloo_shim()  # before any backend init

    import jax

    jax.distributed.initialize(
        coordinator_address=f"localhost:{port}",
        num_processes=size,
        process_id=rank,
    )
    jax.config.update("jax_enable_x64", True)  # f64 mass-fixer accumulation

    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from legoesm.parallel.shard_map_compat import shard_map

    # 12 local virtual devices per process x 2 processes = 24 global.
    if jax.device_count() != N_GLOBAL_DEVICES:
        print(f"rank{rank} FAIL: global device_count={jax.device_count()} "
              f"(want {N_GLOBAL_DEVICES}); check "
              f"XLA_FLAGS=--xla_force_host_platform_device_count", flush=True)
        return 2
    if jax.local_device_count() != DEVICES_PER_PROC:
        print(f"rank{rank} FAIL: local device_count="
              f"{jax.local_device_count()} (want {DEVICES_PER_PROC})",
              flush=True)
        return 2

    # Mesh spans NON-addressable devices (the multi-controller path a Derecho
    # NCCL run exercises): each process owns 12 of the 24 mesh cells.
    mesh = Mesh(np.array(jax.devices()).reshape(MESH_SHAPE), AXES)

    def _mixed_clique_step(t):
        """One tiled step: 4 halo ppermute rounds + 1 global mass-fixer psum.

        ``t`` is a device-local shard of shape ``(1, 1, 1, NH, NH)`` (the three
        mesh axes are size-1 per device).  Mirrors the closed-loop lane: the
        four edge exchanges are distinct ``ppermute`` cliques; the ``psum`` is
        the ``_tile_fix_ps_mass_delta`` GLOBAL all-reduce over ALL mesh axes —
        the one collective single-shot lacks.
        """
        # (a) halo-like exchanges: 4 distinct permute clique sets (+-1 span the
        # full 24-cycle; +-6 decompose into six 4-cycles -> a MIX of sizes).
        top = jax.lax.ppermute(t[:, :, :, 0, :], AXES, _shift_perm(24, 1))
        bot = jax.lax.ppermute(t[:, :, :, -1, :], AXES, _shift_perm(24, -1 % 24))
        left = jax.lax.ppermute(t[:, :, :, :, 0], AXES, _shift_perm(24, 6))
        right = jax.lax.ppermute(t[:, :, :, :, -1], AXES, _shift_perm(24, -6 % 24))
        # (b) fix_mass-like GLOBAL all-reduce over the SAME axes (24-way clique).
        g = jax.lax.psum(jnp.sum(t), AXES)
        # Combine so XLA cannot DCE either collective.
        t = (t.at[:, :, :, 0, :].add(0.1 * top)
              .at[:, :, :, -1, :].add(0.1 * bot)
              .at[:, :, :, :, 0].add(0.1 * left)
              .at[:, :, :, :, -1].add(0.1 * right))
        return t + 1e-12 * g

    sm_step = shard_map(_mixed_clique_step, mesh=mesh,
                        in_specs=P(*AXES), out_specs=P(*AXES), check_vma=False)
    # Replicated (fully addressable on every process) global L2 for the check.
    sm_norm = shard_map(lambda t: jax.lax.psum(jnp.sum(t * t), AXES),
                        mesh=mesh, in_specs=P(*AXES), out_specs=P(),
                        check_vma=False)

    @jax.jit
    def run(x0):
        x_final, _ = jax.lax.scan(
            lambda x, _: (sm_step(x), None), x0, None, length=N_STEPS)
        return sm_norm(x_final)

    # Multi-controller idiom: the SAME host-global array is built identically
    # on every process (deterministic np.arange), and device_put slices out
    # each process's addressable shards (matches test_atm_latlon_spmd_step.py
    # and shard_state_atm_latlon; no cross-process transfer of the global
    # array). The sharding spans all 24 (incl. non-addressable) mesh devices.
    g0 = (1.0 + np.arange(N_GLOBAL_DEVICES * NH * NH, dtype=np.float64)
          ).reshape(*MESH_SHAPE, NH, NH)
    x0 = jax.device_put(g0, NamedSharding(mesh, P(*AXES)))
    # ``run`` returns a fully-REPLICATED (out_specs=P()) global scalar; read
    # THIS process's local replica directly (unambiguously addressable) rather
    # than relying on np.asarray of a multi-process global array.
    out = float(jax.block_until_ready(run(x0)).addressable_data(0))

    if not (np.isfinite(out) and out > 0.0):
        print(f"rank{rank} FAIL: mixed-clique norm={out!r} not finite/positive",
              flush=True)
        return 1
    print(f"rank{rank}: MIXED-CLIQUE OK norm={out:.4f}", flush=True)
    return 0


@pytest.mark.skipif(shutil.which("mpiexec") is None,
                    reason="mpiexec (MPI launcher) not available")
def test_tiled_mixed_clique_two_process_mpiexec_completes():
    """#921: the closed-loop psum+ppermute mix survives 2-process rendezvous."""
    from multihost_harness import run_federated  # reuse retry/timeout/kill

    repo_root = Path(__file__).resolve().parents[2]
    pkg_path = os.pathsep.join(
        str(p) for p in sorted((repo_root / "packages").glob("*"))
        if p.is_dir())
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = pkg_path + (os.pathsep + existing if existing else "")

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={DEVICES_PER_PROC}"
    env["PYTHONPATH"] = pythonpath

    # n_proc=1: ONE mpiexec launcher that itself fans out to N_PROC ranks.
    # ``-x VAR`` forwards from this env; ``-x VAR=VALUE`` sets the (per-retry)
    # coordinator port explicitly so run_federated's fresh-port retry works.
    def build_cmd(_rank, port):
        return [
            "mpiexec", "-n", str(N_PROC),
            "-x", "JAX_PLATFORMS",
            "-x", "XLA_FLAGS",
            "-x", "PYTHONPATH",
            "-x", f"LEGOESM_COORD_PORT={port}",
            sys.executable, str(Path(__file__).resolve()), "worker",
        ]

    rcs, outs = run_federated(build_cmd, n_proc=1, env=env, timeout_s=420)
    rc, out = rcs[0], outs[0]
    assert rc == 0, f"mpiexec exited {rc}\n--- output ---\n{out[-6000:]}"
    n_ok = out.count("MIXED-CLIQUE OK")
    assert n_ok == N_PROC, (
        f"expected {N_PROC} 'MIXED-CLIQUE OK' markers, got {n_ok}\n"
        f"--- output ---\n{out[-6000:]}")


if __name__ == "__main__":
    # mpiexec-spawned worker: rank/size from Open MPI, coordinator port from env.
    _rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    _size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    _port = int(os.environ["LEGOESM_COORD_PORT"])
    raise SystemExit(_worker(_rank, _size, _port))

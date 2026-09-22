"""Is ``SPMD_HALO_DEPTH`` (3 rings per tendency evaluation) the exact reach of
the benchmark dycore stencil, or does it carry a spare ring?

The wide-halo GPU step fetches ``evals x SPMD_HALO_DEPTH`` rings once per
step (9 at RK3); at 128 GPUs the exchanged bytes are the dominant term of
the 4.65 ms step (atm_mpas_halo_attrib: ballast x2 = +2.66 ms), so one
spare ring per evaluation is ~1/3 of the wire.  A ring the stencil never
reads cannot change an owned row: owned rows at depth d and d+1 must be
BIT-identical.  So bitwise equality of owned rows across depths is the
discriminator: depth 3 == depth 4 pins today's sufficiency; depth 2 == 3
would mean the margin is spare (and a candidate lever, owner decision).

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""

import jax
import numpy as np
import pytest


def _need_multi_device(n: int):
    if len(jax.devices("cpu")) < n:
        pytest.skip(f"need {n} CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def _owned_after(depth: int, n_steps: int, monkeypatch, devices: int = 4):
    """Owned-cell rows of (u, T, p_s) after ``n_steps`` of the benchmark
    dycore config (hyperdiffusion on, SSP-RK3, per-evaluation fills) at
    partition halo depth ``depth``."""
    from legoesm.parallel import sharded_dynamics as sd
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    from legoesm.parallel.mesh import create_voronoi_device_mesh, replicate_pytree
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    monkeypatch.setenv("LEGOESM_MPAS_WIDE_HALO", "0")
    monkeypatch.setattr(sd, "SPMD_HALO_DEPTH", depth)
    mesh = create_voronoi_mesh(subdivision_level=4)
    mesh = reorder_voronoi_for_sharding(mesh, devices)
    sigma = create_sigma_coordinate(8)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3",
    )
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
        n_devices=devices)
    model = MPASPrimitiveEquationModel(replicate_pytree(mesh, dev_config), sigma, cfg)
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    step = sd.make_voronoi_sharded_step(model, dev_config, halo_strategy="ppermute")
    assert step._halo_depth_effective == depth
    for _ in range(n_steps):
        state = step(state, 600.0)
    return {k: np.asarray(getattr(state, k).data) for k in ("u", "T", "p_s")}


def _report(a, b, tag):
    for k in a:
        d = np.abs(a[k] - b[k]); n_bad = int(np.count_nonzero(d))
        print(f"[reach] {tag} {k}: max|diff|={float(d.max()):.3e} "
              f"nonzero={n_bad}/{d.size}")


@pytest.mark.parametrize("n_steps", [1, 3])
def test_depth3_is_sufficient_and_report_depth2(n_steps, monkeypatch):
    _need_multi_device(4)
    d4 = _owned_after(4, n_steps, monkeypatch)
    d3 = _owned_after(3, n_steps, monkeypatch)
    d2 = _owned_after(2, n_steps, monkeypatch)
    _report(d3, d4, f"steps={n_steps} depth3-vs-4")
    _report(d2, d3, f"steps={n_steps} depth2-vs-3")
    for k in d3:
        assert np.array_equal(d3[k], d4[k]), f"{k}: depth 3 != depth 4 — SPMD_HALO_DEPTH too small"

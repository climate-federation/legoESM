"""How many halo rings does the MPAS sharded step actually read per fill?

``_build_voronoi_partition_infra`` collects ``halo_depth`` rings by BFS and
then ALWAYS runs two ``_close_halo_under_cellsOnEdge`` passes.  Measured
2026-09-22 (subdivision 4, 4 devices, 3-5 steps, benchmark config with
hyperdiffusion), max rel err in u against the single-device ``model.step``:

    path      depth  passes  err        note
    default     1      1     1.0e-3     one closure pass is NOT enough at ANY depth
    default     3      1     1.0e-3       (it repairs edge completeness; not a ring)
    default     1      2     3.4e-6     == depth 3 + 2 passes TO THE BIT
    default     3      2     3.4e-6     production per-evaluation fill
    wide        1      2     2.9e-4     but 3e-5 off production: stage validity
    wide        2      2     2.9e-4       shrinks by the TRUE reach (3 rings) per
    wide        3      2     2.9e-4       evaluation; depth 3 x 3 evals is exact

So SPMD_HALO_DEPTH = 3 is the per-evaluation reach, the wide GPU step's
depth 9 is NOT spare (at most the two closure rings on top of it are), and
the per-evaluation path carries two spare rings it never reads.  The
assertions pin exactly that; both are non-vacuous (one closure pass must
fail; wide depth 2 must differ from 3).

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""


import os

import jax
import numpy as np
import pytest

from legoesm.parallel import sharded_dynamics as sd

_ORIG_CLOSE = sd._close_halo_under_cellsOnEdge   # captured ONCE; a per-call capture chains wrappers


def _need_multi_device(n: int):
    if len(jax.devices("cpu")) < n:
        pytest.skip(f"need {n} CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def _setup(devices: int):
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
    )
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
    mesh = create_voronoi_mesh(subdivision_level=4)
    mesh = reorder_voronoi_for_sharding(mesh, devices)
    sigma = create_sigma_coordinate(8)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
        pv_scheme="energy", time_integrator="ssp_rk3",
    )
    return mesh, sigma, cfg


def _fields(state):
    return {k: np.asarray(getattr(state, k).data) for k in ("u", "T", "p_s")}


def _truth(devices: int, n_steps: int, dt: float):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationModel
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
    mesh, sigma, cfg = _setup(devices)
    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
    s = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    for _ in range(n_steps):
        s = model.step(s, dt)
    return _fields(s)


def _sharded(devices: int, n_steps: int, dt: float, *, depth: int, passes: int,
             wide: bool, monkeypatch):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationModel
    from legoesm.parallel.mesh import create_voronoi_device_mesh, replicate_pytree
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
    monkeypatch.setenv("LEGOESM_MPAS_WIDE_HALO", "1" if wide else "0")
    monkeypatch.setattr(sd, "SPMD_HALO_DEPTH", depth)
    monkeypatch.setattr(sd, "_close_halo_under_cellsOnEdge",
                        lambda o, h, c, n_passes: _ORIG_CLOSE(o, h, c, passes))
    mesh, sigma, cfg = _setup(devices)
    dev = create_voronoi_device_mesh(nCells=mesh.nCells, nEdges=mesh.nEdges,
                                     nVertices=mesh.nVertices, n_devices=devices)
    model = MPASPrimitiveEquationModel(replicate_pytree(mesh, dev), sigma, cfg)
    s = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    step = sd.make_voronoi_sharded_step(model, dev, halo_strategy="ppermute")
    assert step._halo_depth_effective == (3 * depth if wide else depth)
    for _ in range(n_steps):
        s = step(s, dt)
    return _fields(s)


def _rel_err(got, truth):
    return {k: float(np.max(np.abs(got[k] - truth[k])) / max(float(np.max(np.abs(truth[k]))), 1e-30))
            for k in truth}


def test_default_path_carries_two_spare_rings(monkeypatch):
    _need_multi_device(4)
    n_steps, dt = 3, 600.0
    truth = _truth(4, n_steps, dt)
    prod = _sharded(4, n_steps, dt, depth=3, passes=2, wide=False, monkeypatch=monkeypatch)
    lean = _sharded(4, n_steps, dt, depth=1, passes=2, wide=False, monkeypatch=monkeypatch)
    short = _sharded(4, n_steps, dt, depth=3, passes=1, wide=False, monkeypatch=monkeypatch)
    print(f"[reach] default: prod={_rel_err(prod, truth)} lean={_rel_err(lean, truth)} "
          f"one-pass={_rel_err(short, truth)}")
    assert _rel_err(short, truth)["u"] > 1e-4, "one closure pass matched single-device: instrument blind"
    for k in prod:
        assert np.array_equal(lean[k], prod[k]), f"{k}: depth-1 halo != depth-3 on owned rows"
    assert _rel_err(lean, truth)["u"] < 1e-5


def test_wide_path_needs_three_rings_per_evaluation(monkeypatch):
    _need_multi_device(4)
    n_steps, dt = 5, 600.0
    prod = _sharded(4, n_steps, dt, depth=3, passes=2, wide=True, monkeypatch=monkeypatch)
    d2 = _sharded(4, n_steps, dt, depth=2, passes=2, wide=True, monkeypatch=monkeypatch)
    e = _rel_err(d2, prod)
    print(f"[reach] wide depth 2 vs 3: {e}")
    assert e["u"] > 1e-6, "wide depth 2 == depth 3: the reach would be < 3 rings per evaluation"

"""Resolution scaling of the MPAS biharmonic vorticity damping coefficient.

``K_zeta_bih`` damps the vorticity checkerboard null mode of the
energy-conserving potential-vorticity flux.  The OMIP NEMO-match recipe tuned
it at 1e14 m^4/s on the ico6 mesh (~120 km).  Held fixed, that coefficient
exceeds the biharmonic operator's explicit stability limit on a finer mesh --
an ico8 (~28 km) cold start diverges at step 11 with it and survives without
it (measured on GPU, jobs 27330187 / 27330586).  The coefficient is therefore
DERIVED from the mesh the model is built on, as ``K_ref*(dx/dx_ref)**3``.

These tests pin the three properties that make that safe: the anchor mesh is
unchanged (bit-identical), a finer mesh gets the cubed law, and an explicit
value still pins (including 0.0 = off).
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.mpas_config import (
    MPASOceanConfig,
    resolution_scaled_k_zeta_bih,
)


def _derive_cfg(**kw) -> MPASOceanConfig:
    return MPASOceanConfig(K_zeta_bih=None, **kw)


def test_anchor_mesh_is_bit_identical_to_the_tuned_value():
    """The ico6 mesh the value was tuned on gets EXACTLY the tuned value.

    Not the tautology ``(x/x)**3 == 1``: this builds the real anchor mesh and
    checks the coefficient the model derives from it is bit-identical to the
    1e14 that mesh was tuned with, so the change is a no-op there.
    """
    from legoesm.grids.voronoi import create_voronoi_mesh

    cfg = _derive_cfg()
    assert cfg.K_zeta_bih_ref == 1.0e14       # the recipe's tuned value
    mesh = create_voronoi_mesh(6)             # ico6 at the DEFAULT 50 Lloyd its
    dc, dv = np.asarray(mesh.dcEdge), np.asarray(mesh.dvEdge)
    dx = float(dc[(dv > 0) & (dc > 0)].mean(dtype=np.float64))
    assert resolution_scaled_k_zeta_bih(dx, cfg) == 1.0e14   # bit-identical


def test_scaling_is_cubed_in_the_spacing():
    """Halving the mesh spacing divides the coefficient by eight."""
    cfg = _derive_cfg()
    dx = cfg.K_zeta_bih_ref_dx_m
    coarse = resolution_scaled_k_zeta_bih(dx, cfg)
    fine = resolution_scaled_k_zeta_bih(dx / 2.0, cfg)
    assert fine == pytest.approx(coarse / 8.0, rel=1e-12)
    # ... and the damping's velocity scale K/dx^3 is what stays constant.
    assert (fine / (dx / 2.0) ** 3) == pytest.approx(coarse / dx**3, rel=1e-12)


def test_icosahedral_ladder_values():
    """The mesh spacings this model runs at, and the coefficients they get.

    Mean ``dcEdge`` measured from the model's own mesh builder (Lloyd 0):
    ico6 120.32 km, ico7 60.16 km, ico8 30.08 km, ico9 15.04 km.
    """
    cfg = _derive_cfg()
    dx = {6: 120.32e3, 7: 60.16e3, 8: 30.08e3, 9: 15.04e3}
    k = {lvl: resolution_scaled_k_zeta_bih(v, cfg) for lvl, v in dx.items()}
    assert k[6] == pytest.approx(1.0e14, rel=5e-3)   # anchor, unchanged
    assert k[7] == pytest.approx(1.25e13, rel=5e-3)
    assert k[8] == pytest.approx(1.56e12, rel=5e-3)
    assert k[9] == pytest.approx(1.95e11, rel=5e-3)
    for lvl in (7, 8, 9):                            # strictly decreasing
        assert k[lvl] < k[lvl - 1]


def test_explicit_value_pins_and_zero_means_off():
    """A float is an override: the mesh spacing must not touch it."""
    for pinned in (0.0, 1.0e14, 4.0e11):
        cfg = MPASOceanConfig(K_zeta_bih=pinned)
        assert resolution_scaled_k_zeta_bih(1.0e3, cfg) == pinned
        assert resolution_scaled_k_zeta_bih(1.0e9, cfg) == pinned


def test_degenerate_spacing_raises():
    cfg = _derive_cfg()
    with pytest.raises(ValueError, match="spacing must be positive"):
        resolution_scaled_k_zeta_bih(0.0, cfg)
    bad = MPASOceanConfig(K_zeta_bih=None, K_zeta_bih_ref_dx_m=0.0)
    with pytest.raises(ValueError, match="K_zeta_bih_ref_dx_m"):
        resolution_scaled_k_zeta_bih(1.0e4, bad)


def test_variable_resolution_mesh_is_refused_not_guessed():
    """One mean spacing describes a quasi-uniform mesh; a refined mesh raises."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_voronoi_mesh(3, lloyd_iterations=0)
    dc = np.asarray(mesh.dcEdge)
    stretched = dc.at[: dc.size // 4].set(dc.max() * 10.0) if hasattr(dc, "at") \
        else np.concatenate([dc[: dc.size // 4] * 10.0, dc[dc.size // 4:]])
    vr_mesh = mesh._replace(dcEdge=np.asarray(stretched))
    z = create_ocean_z_star(n_levels=3, H_max=1000.0)
    with pytest.raises(ValueError, match="quasi-uniform"):
        MPASOceanModel(vr_mesh, z, MPASOceanConfig(K_zeta_bih=None))
    # ... and a PINNED coefficient is still accepted on that same mesh.
    assert MPASOceanModel(
        vr_mesh, z, MPASOceanConfig(K_zeta_bih=1.0e14)).config.K_zeta_bih == 1.0e14


def test_derived_coefficient_records_the_mesh_it_came_from():
    """A derived value carries the spacing it was derived at; a pin does not."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_voronoi_mesh(3, lloyd_iterations=0)
    z = create_ocean_z_star(n_levels=3, H_max=1000.0)
    dc, dv = np.asarray(mesh.dcEdge), np.asarray(mesh.dvEdge)
    derived = MPASOceanModel(mesh, z, MPASOceanConfig(K_zeta_bih=None)).config
    assert derived.K_zeta_bih_dx_m == pytest.approx(
        float(dc[(dv > 0) & (dc > 0)].mean(dtype=np.float64)))
    pinned = MPASOceanModel(mesh, z, MPASOceanConfig(K_zeta_bih=1.0e14)).config
    assert pinned.K_zeta_bih_dx_m == 0.0


def test_model_derives_from_its_own_mesh_and_a_pin_survives():
    """The model constructor is the choke point: mesh in, coefficient out."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_voronoi_mesh(3, lloyd_iterations=0)
    z = create_ocean_z_star(n_levels=3, H_max=1000.0)
    dc, dv = np.asarray(mesh.dcEdge), np.asarray(mesh.dvEdge)
    dx_mean = float(dc[(dv > 0) & (dc > 0)].mean(dtype=np.float64))

    derived = MPASOceanModel(mesh, z, MPASOceanConfig(K_zeta_bih=None))
    cfg = MPASOceanConfig(K_zeta_bih=None)
    assert derived.config.K_zeta_bih == resolution_scaled_k_zeta_bih(dx_mean, cfg)
    assert derived.config.K_zeta_bih > 0.0

    pinned = MPASOceanModel(mesh, z, MPASOceanConfig(K_zeta_bih=1.0e14))
    assert pinned.config.K_zeta_bih == 1.0e14
    off = MPASOceanModel(mesh, z, MPASOceanConfig(K_zeta_bih=0.0))
    assert off.config.K_zeta_bih == 0.0


def test_sharding_padded_mesh_derives_the_same_coefficient():
    """The distributed lane pads the mesh with ghost edges carrying dvEdge=0
    and dcEdge=1 m.  Selecting real edges by dcEdge would keep every ghost at
    1 m -- skewing the mean and tripping the uniformity guard on EVERY sharded
    run -- so the padded mesh must derive exactly what the unpadded one does.
    """
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding

    mesh = create_voronoi_mesh(3, lloyd_iterations=0)
    z = create_ocean_z_star(n_levels=3, H_max=1000.0)
    plain = MPASOceanModel(mesh, z, MPASOceanConfig(K_zeta_bih=None)).config

    padded = reorder_voronoi_for_sharding(mesh, 7, edge_order="owner")  # 7 does not divide this mesh
    n_ghost = int(np.asarray(padded.dvEdge).size) - int(np.asarray(mesh.dvEdge).size)
    assert n_ghost > 0, "this mesh/device count must actually pad, or the gate is vacuous"
    assert float(np.asarray(padded.dcEdge).min()) == pytest.approx(1.0), \
        "ghost edges are the dcEdge=1 m entries the real-edge predicate must drop"
    shard = MPASOceanModel(padded, z, MPASOceanConfig(K_zeta_bih=None)).config
    assert shard.K_zeta_bih == pytest.approx(plain.K_zeta_bih, rel=1e-12)


def test_omip_recipe_defaults_to_derived():
    """The OMIP MPAS recipe ships the derived coefficient, not a fixed one."""
    from legoesm.ocean.fidelity.nemo_match_recipe import (
        NEMOMatchMPASRecipeConfig,
        nemo_match_mpas_model_config,
    )
    assert NEMOMatchMPASRecipeConfig().K_zeta_bih is None
    assert nemo_match_mpas_model_config().K_zeta_bih is None

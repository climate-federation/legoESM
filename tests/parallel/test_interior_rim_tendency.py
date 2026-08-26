"""Numerical-equivalence core for the MPAS interior/rim halo-overlap split.

The overlap step will compute the tendency for interior cells (all-owned
stencil) while the halo exchange is in flight, then recompute the rim on a
compact submesh once the halo lands and scatter it back. This proves the
two-pass result equals a single full-mesh pass BEFORE any of it touches the
hot path.

The test is non-vacuous by construction: the interior pass reads a
deliberately WRONG (but finite) halo pad. If the interior/rim boundary
(rim_width == the RHS stencil radius) were too small, some halo-affected
owned cell would be classified interior, computed from the wrong pad, and
never overwritten by the rim scatter — and the owned tendency would not
match the filled single pass.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _reference_and_split(n_dev=6, subdivision=6, nlev=4, stencil_depth=2,
                         halo_depth=4, nu4=0.0, n_rounds=3):
    from legoesm.core.field import Field
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import (
        build_local_mesh, reorder_voronoi_for_sharding,
    )
    from legoesm.parallel.sharded_dynamics import (
        InteriorPad, _WIDE_RING_FAR, _build_rim_plan,
        _build_rim_rings_by_dependency, _build_rim_rings_structural,
        _build_voronoi_partition_infra, interior_rim_tendency,
    )
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, mpas_hydrostatic_tendencies,
    )

    mesh = create_voronoi_mesh(subdivision_level=subdivision)
    mesh = reorder_voronoi_for_sharding(mesh, n_dev, method="metis")
    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
     _owner) = _build_voronoi_partition_infra(mesh, n_dev,
                                              halo_depth=halo_depth)

    sigma = create_sigma_coordinate(nlev)
    cfg = (MPASPrimitiveEquationConfig(nu_del4=nu4, nu_del4_ps=nu4)
           if nu4 > 0 else MPASPrimitiveEquationConfig())
    dt = 100.0

    # PRIMARY rim: STRUCTURAL stencil closure (state-independent, provably
    # safe). rim_width=1 because the structural builder emits distance 1 for
    # rim and FAR for interior; stencil_depth is the submesh closure depth.
    cell_rim, edge_rim = _build_rim_rings_structural(mesh, partitions,
                                                     max_lc, max_le,
                                                     n_rounds=n_rounds)
    # VALIDATOR: the data-dependency probe (varied owned state) must never flag
    # an entity the structural closure calls interior. If it does, the
    # structural connectivity union or its round count missed a real
    # dependence -- a bug, not a tolerance.
    dep_c, dep_e = _build_rim_rings_by_dependency(
        mpas_hydrostatic_tendencies, sigma, cfg, dt, mesh, partitions,
        max_lc, max_le)
    for d in range(n_dev):
        struct_c = cell_rim[d] != _WIDE_RING_FAR
        struct_e = edge_rim[d] != _WIDE_RING_FAR
        prob_c = dep_c[d] != _WIDE_RING_FAR
        prob_e = dep_e[d] != _WIDE_RING_FAR
        assert not (prob_c & ~struct_c).any(), (
            f"device {d}: dependency probe flagged a CELL the structural "
            f"closure calls interior -- closure under-covers")
        assert not (prob_e & ~struct_e).any(), (
            f"device {d}: dependency probe flagged an EDGE the structural "
            f"closure calls interior -- closure under-covers")
    plans = _build_rim_plan(mesh, partitions, cell_rim, edge_rim,
                            rim_width=1, stencil_depth=stencil_depth)

    rng = np.random.default_rng(0)
    nC, nE = mesh.nCells, mesh.nEdges
    # Smooth-ish global reference fields so the RHS is well posed.
    u_g = jnp.asarray(rng.normal(scale=5.0, size=(nE, nlev)))
    T_g = jnp.asarray(250.0 + rng.normal(scale=5.0, size=(nC, nlev)))
    ps_g = jnp.asarray(1.0e5 + rng.normal(scale=1.0e3, size=(nC,)))
    phis_g = jnp.asarray(rng.normal(scale=100.0, size=(nC,)))

    # A pad the interior pass must be immune to: values far from the truth.
    pad = InteriorPad(u=7.0, T=999.0, p_s=5.0e4, phis=-500.0)

    checked = 0
    for d, (part, plan) in enumerate(zip(partitions, plans)):
        if plan["n_rim_cells"] == 0:
            continue
        local_mesh = build_local_mesh(mesh, part)
        lc = np.asarray(part.local_cells)
        le = np.asarray(part.local_edges)
        n_oc = part.n_owned_cells
        n_oe = part.n_owned_edges

        # Filled device-local buffers (owned + halo), UNPADDED: this
        # standalone core runs on the unpadded local mesh from
        # build_local_mesh (n_local rows), so the field buffers must be
        # n_local too. (The shard_map wiring later pads to the across-device
        # max; the core function reads the row count off the buffer, so the
        # same code serves both.)
        u_full = u_g[le]
        T_full = T_g[lc]
        ps_full = ps_g[lc]
        phis_full = phis_g[lc]

        # Reference: a single pass on the FULLY-FILLED local mesh.
        ref_state = MPASHydrostaticState(
            u=Field(data=u_full, name="u", dims=("nEdges", "nlev"),
                    units="m/s", long_name="normal velocity",
                    staggering="edge"),
            T=Field(data=T_full, name="T", dims=("nCells", "nlev"),
                    units="K", long_name="temperature", staggering="cell"),
            p_s=Field(data=ps_full, name="p_s", dims=("nCells",), units="Pa",
                      long_name="surface pressure", staggering="cell"),
            phis=Field(data=phis_full, name="phis", dims=("nCells",),
                       units="m^2/s^2", long_name="surface geopotential",
                       staggering="cell"),
            tracers=None,
        )
        ref = mpas_hydrostatic_tendencies(ref_state, local_mesh, sigma, cfg,
                                          dt=dt)
        ref_du = np.asarray(ref.du_dt.data[:n_oe])
        ref_dT = np.asarray(ref.dT_dt.data[:n_oc])
        ref_dps = np.asarray(ref.dp_s_dt.data[:n_oc])

        # Split: interior pass sees ONLY the owned rows + the wrong pad.
        du, dT, dps = interior_rim_tendency(
            mpas_hydrostatic_tendencies, sigma, cfg, dt,
            u_g[le][:n_oe], T_g[lc][:n_oc], ps_g[lc][:n_oc], phis_g[lc][:n_oc],
            u_full, T_full, ps_full, phis_full,
            local_mesh, plan["sub_mesh"],
            jnp.asarray(plan["cell_gather"]), jnp.asarray(plan["edge_gather"]),
            jnp.asarray(plan["cell_scatter"]),
            jnp.asarray(plan["edge_scatter"]),
            pad=pad,
        )
        yield (d, np.asarray(du), np.asarray(dT), np.asarray(dps),
               ref_du, ref_dT, ref_dps)
        checked += 1
    if checked == 0:
        pytest.skip("no device carried a non-empty rim on this fixture")


@pytest.mark.parametrize("nu4,n_rounds", [
    (0.0, 3),      # bare dry core: 2-hop stencil, 3 incidence rounds cover it
    (1e16, 5),     # production hyperdiffusion (del4): 4 incidence hops, +1 margin
])
def test_interior_rim_equals_single_pass(nu4, n_rounds):
    # float64; the only allowed difference is stencil-sum re-association
    # between the submesh order and the local-mesh order. The del4 case is the
    # production configuration (component_factory sets nu_del4>0); n_rounds is
    # sized to the composed-stage count with a margin, and the dependency
    # validator (inside _reference_and_split) guards that the structural
    # closure never calls a halo-dependent entity interior.
    tol = 1e-8
    any_checked = False
    for (d, du, dT, dps, ref_du, ref_dT, ref_dps) in _reference_and_split(
            nu4=nu4, n_rounds=n_rounds):
        any_checked = True
        assert np.all(np.isfinite(du)) and np.all(np.isfinite(dT)), (
            f"device {d}: non-finite tendency leaked from the pad")
        for name, got, ref in (("du", du, ref_du), ("dT", dT, ref_dT),
                               ("dps", dps, ref_dps)):
            err = np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30)
            assert err < tol, (
                f"device {d}: {name} interior/rim split differs from the "
                f"single pass by rel {err:.2e} (> {tol:.0e}) — the rim_width "
                f"may be smaller than the RHS stencil radius")
    assert any_checked

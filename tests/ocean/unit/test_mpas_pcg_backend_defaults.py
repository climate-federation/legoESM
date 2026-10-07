"""The MPAS distributed barotropic PCG bundle is resolved per JAX backend
(owner decision 2026-10-04) and recorded in the model's config."""
from __future__ import annotations

import pytest

from legoesm.ocean.mpas_config import (
    MPAS_BAROTROPIC_PCG_DEFAULTS,
    MPASOceanConfig,
    resolve_barotropic_pcg_defaults,
)

KEYS = ("barotropic_implicit_pcg_variant", "barotropic_implicit_pcg_precond",
        "barotropic_implicit_pcg_fixed_iters")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_unset_fields_take_the_backend_bundle(backend):
    cfg = resolve_barotropic_pcg_defaults(MPASOceanConfig(), backend)
    assert {k: getattr(cfg, k) for k in KEYS} == MPAS_BAROTROPIC_PCG_DEFAULTS[backend]
    assert resolve_barotropic_pcg_defaults(cfg, "gpu" if backend == "cpu" else "cpu") is cfg


def test_pinned_bundle_is_kept_on_any_backend():
    pinned = MPASOceanConfig(barotropic_implicit_pcg_variant="standard",
                             barotropic_implicit_pcg_precond="poly",
                             barotropic_implicit_pcg_fixed_iters=20)
    for backend in ("cpu", "gpu", "tpu"):
        assert resolve_barotropic_pcg_defaults(pinned, backend) is pinned


def test_pinning_the_backend_preconditioner_fills_the_rest():
    cfg = resolve_barotropic_pcg_defaults(
        MPASOceanConfig(barotropic_implicit_pcg_precond="jacobi"), "cpu")
    assert cfg.barotropic_implicit_pcg_fixed_iters == 30
    assert cfg.barotropic_implicit_pcg_variant == "single_reduce_deep"


def test_another_preconditioner_must_pin_its_count_and_recurrence():
    with pytest.raises(ValueError, match="also pin"):
        resolve_barotropic_pcg_defaults(
            MPASOceanConfig(barotropic_implicit_pcg_precond="jacobi"), "gpu")


def test_unknown_backend_needs_a_pinned_bundle():
    with pytest.raises(ValueError, match="no MPAS barotropic PCG default"):
        resolve_barotropic_pcg_defaults(MPASOceanConfig(), "tpu")


def test_model_records_the_resolved_bundle():
    import jax

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.vertical import create_ocean_z_star

    model = MPASOceanModel(create_voronoi_mesh(2), create_ocean_z_star(n_levels=3),
                           MPASOceanConfig())
    want = MPAS_BAROTROPIC_PCG_DEFAULTS[jax.default_backend()]
    assert {k: getattr(model.config, k) for k in KEYS} == want


def test_layout_halo_sizing_refuses_a_mixed_bundle():
    """The SPMD halo depth is chosen from the resolved bundle, so a
    preconditioner pinned without its count and recurrence fails at layout
    time, not later inside the model build."""
    import jax

    from legoesm.parallel.voronoi_spmd_ocean import halo_depth_for_config

    other = {"cpu": "gpoly", "gpu": "jacobi"}[jax.default_backend()]
    with pytest.raises(ValueError, match="also pin"):
        halo_depth_for_config(MPASOceanConfig(barotropic_implicit_pcg_precond=other))

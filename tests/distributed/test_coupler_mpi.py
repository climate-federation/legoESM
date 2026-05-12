"""MPI coupler regression tests with partitioned face masks.

Run with:
    mpirun -np 3 python -m pytest tests/distributed/test_coupler_mpi.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# Guard: skip all tests if mpi4jax/mpi4py are not installed.
mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm import constants
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.coupler.coupler import init_surface_state, make_coupler
from legoesm.coupler.coupling_fields import AtmToSurface, SurfaceToAtm
from legoesm.coupler.lake.config import LakeConfig
from legoesm.grids.halo import set_halo_backend
from legoesm.ice.config import SeaIceConfig
from legoesm.land.config import LandConfig
from legoesm.parallel.comm import build_comm_topology
from legoesm.parallel.distributed import gather_to_global


@pytest.fixture(autouse=True)
def reset_halo_backend():
    """Reset halo backend to local after each test."""
    yield
    set_halo_backend("local")


@pytest.fixture
def topology():
    """Build topology for this MPI rank."""
    rank = MPI.COMM_WORLD.Get_rank()
    n_processes = MPI.COMM_WORLD.Get_size()
    return build_comm_topology(rank, n_processes)


def _pattern(shape, *, base: float, face_scale: float, x_scale: float, y_scale: float):
    """Deterministic spatial pattern with face/x/y dependence."""
    _, n_x, n_y = shape
    face = jnp.arange(6, dtype=jnp.float32)[:, None, None]
    x = jnp.arange(n_x, dtype=jnp.float32)[None, :, None]
    y = jnp.arange(n_y, dtype=jnp.float32)[None, None, :]
    return base + face_scale * face + x_scale * x + y_scale * y


def _make_forcing(shape: tuple[int, int, int]) -> AtmToSurface:
    """Create nontrivial but stable atmospheric forcing fields."""
    sw_down = _pattern(shape, base=220.0, face_scale=6.0, x_scale=0.8, y_scale=-0.4)
    lw_down = _pattern(shape, base=310.0, face_scale=2.0, x_scale=0.25, y_scale=0.15)
    precip_total = _pattern(
        shape, base=1.0e-5, face_scale=1.0e-6, x_scale=2.0e-7, y_scale=1.0e-7,
    )
    T_lowest = _pattern(shape, base=279.0, face_scale=0.3, x_scale=0.07, y_scale=-0.05)
    q_lowest = _pattern(shape, base=4.5e-3, face_scale=1.0e-4, x_scale=2.0e-5, y_scale=-1.0e-5)
    u_lowest = _pattern(shape, base=4.0, face_scale=0.5, x_scale=0.1, y_scale=-0.05)
    v_lowest = _pattern(shape, base=-2.0, face_scale=-0.2, x_scale=0.03, y_scale=0.04)
    p_surface = _pattern(shape, base=1.0e5, face_scale=15.0, x_scale=1.0, y_scale=-0.5)
    p_lowest = 0.95 * p_surface
    rho_lowest = p_lowest / (constants.R_d * jnp.maximum(T_lowest, 200.0))

    zero = jnp.zeros(shape, dtype=jnp.float32)
    return AtmToSurface(
        sw_down=sw_down,
        lw_down=lw_down,
        precip_total=precip_total,
        precip_snow=zero,
        T_lowest=T_lowest,
        q_lowest=q_lowest,
        u_lowest=u_lowest,
        v_lowest=v_lowest,
        p_lowest=p_lowest,
        p_surface=p_surface,
        rho_lowest=rho_lowest,
        cos_zenith=jnp.full(shape, 0.6, dtype=jnp.float32),
        co2_ppmv=jnp.array(410.0, dtype=jnp.float32),
        has_radiation=jnp.array(1.0, dtype=jnp.float32),
        has_precipitation=jnp.array(1.0, dtype=jnp.float32),
    )


def _make_tile_config(shape: tuple[int, int, int]) -> TileConfig:
    """Create deterministic land/lake masks with valid static sums."""
    f_land = _pattern(shape, base=0.25, face_scale=0.02, x_scale=0.01, y_scale=-0.005)
    f_lake = _pattern(shape, base=0.05, face_scale=0.005, x_scale=-0.002, y_scale=0.003)
    f_land = jnp.clip(f_land, 0.0, 0.8)
    f_lake = jnp.clip(f_lake, 0.0, 0.3)
    return TileConfig(f_land=f_land, f_lake=f_lake)


def _face_mask(local_face_ids: tuple[int, ...] | list[int], dtype: jnp.dtype) -> jnp.ndarray:
    """Build a face-only mask (6, 1, 1) for the local MPI rank."""
    return jnp.array(
        [1.0 if f in set(local_face_ids) else 0.0 for f in range(6)],
        dtype=dtype,
    )[:, None, None]


def _partition_tile_config(
    tile_cfg: TileConfig,
    local_face_ids: tuple[int, ...] | list[int],
) -> TileConfig:
    """Partition tile masks by face ownership.

    Non-local faces are set to all-land (f_land=1, f_lake=0) so water tiles
    are inactive there. Local faces keep the original fractions.
    """
    mask = _face_mask(local_face_ids, tile_cfg.f_land.dtype)
    one = jnp.ones_like(tile_cfg.f_land)
    zero = jnp.zeros_like(tile_cfg.f_lake)
    return TileConfig(
        f_land=mask * tile_cfg.f_land + (1.0 - mask) * one,
        f_lake=mask * tile_cfg.f_lake + (1.0 - mask) * zero,
    )


def _mask_face_leading_pytree(
    pytree,
    local_face_ids: tuple[int, ...] | list[int],
):
    """Zero non-local face-leading leaves in a pytree."""
    local_set = set(local_face_ids)

    def _mask_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        if leaf.ndim < 1 or leaf.shape[0] != 6:
            return leaf
        mask_1d = jnp.array([f in local_set for f in range(6)], dtype=bool)
        shape = (6,) + (1,) * (leaf.ndim - 1)
        mask = mask_1d.reshape(shape)
        return jnp.where(mask, leaf, jnp.zeros_like(leaf))

    return jax.tree.map(_mask_leaf, pytree)


def _assert_surface_to_atm_close(
    got: SurfaceToAtm,
    ref: SurfaceToAtm,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> None:
    """Assert two blended coupler outputs are numerically identical."""
    for name in SurfaceToAtm._fields:
        g = getattr(got, name)
        r = getattr(ref, name)
        assert jnp.all(jnp.isfinite(g)), f"{name}: non-finite in distributed output"
        assert jnp.all(jnp.isfinite(r)), f"{name}: non-finite in reference output"
        assert jnp.allclose(g, r, atol=atol, rtol=rtol), f"{name}: mismatch"


class TestCouplerMPIRegression:
    """Distributed coupler regression checks against single-rank reference."""

    def test_partitioned_step_matches_single_rank_reference(self, topology):
        """One coupler step should match reference after gather."""
        set_halo_backend("mpi", topology)

        shape = (6, 4, 4)
        coupler_cfg = CouplerConfig(coupling_dt=3600.0)
        step_fn = make_coupler(
            coupler_cfg,
            LandConfig(),
            SeaIceConfig(),
            LakeConfig(),
        )

        state_ref = init_surface_state(shape)
        forcing = _make_forcing(shape)
        tile_cfg = _make_tile_config(shape)
        ocean_sst = _pattern(shape, base=286.0, face_scale=0.4, x_scale=0.1, y_scale=-0.05)
        ocean_u = _pattern(shape, base=0.05, face_scale=0.01, x_scale=0.003, y_scale=-0.002)
        ocean_v = _pattern(shape, base=-0.03, face_scale=-0.008, x_scale=0.002, y_scale=0.003)
        dt = 600.0

        ref_state_new, ref_blended = step_fn(
            state_ref, forcing, tile_cfg, ocean_sst, ocean_u, ocean_v, dt,
        )

        tile_cfg_part = _partition_tile_config(tile_cfg, topology.local_face_ids)

        part_state_new, part_blended = step_fn(
            state_ref,
            forcing,
            tile_cfg_part,
            ocean_sst,
            ocean_u,
            ocean_v,
            dt,
        )
        part_state_new = _mask_face_leading_pytree(
            part_state_new, topology.local_face_ids,
        )
        part_blended = _mask_face_leading_pytree(
            part_blended, topology.local_face_ids,
        )

        got_state = gather_to_global(part_state_new)
        got_blended = gather_to_global(part_blended)

        if topology.rank == 0:
            _assert_surface_to_atm_close(got_blended, ref_blended)
            assert jnp.allclose(
                got_state.land.T_soil.data,
                ref_state_new.land.T_soil.data,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            assert jnp.allclose(
                got_state.ice.h_ice.data,
                ref_state_new.ice.h_ice.data,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            assert jnp.allclose(
                got_state.lake.T_epi.data,
                ref_state_new.lake.T_epi.data,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
            assert float(got_state.accumulator.total_dt) == pytest.approx(
                float(ref_state_new.accumulator.total_dt),
                abs=1.0e-9,
            )

    def test_partitioned_multistep_flush_and_carry_matches_reference(self, topology):
        """Flush/carry behavior with coupling_dt should match reference exactly."""
        set_halo_backend("mpi", topology)

        shape = (6, 4, 4)
        coupler_cfg = CouplerConfig(coupling_dt=900.0)
        step_fn = make_coupler(
            coupler_cfg,
            LandConfig(),
            SeaIceConfig(),
            LakeConfig(),
        )

        ref_state = init_surface_state(shape)
        part_state = ref_state

        forcing = _make_forcing(shape)
        tile_cfg = _make_tile_config(shape)
        tile_cfg_part = _partition_tile_config(tile_cfg, topology.local_face_ids)
        ocean_sst = _pattern(shape, base=287.0, face_scale=0.35, x_scale=0.08, y_scale=-0.04)
        ocean_u = _pattern(shape, base=0.08, face_scale=0.012, x_scale=0.003, y_scale=-0.001)
        ocean_v = _pattern(shape, base=-0.04, face_scale=-0.006, x_scale=0.001, y_scale=0.002)

        dt_sequence = (600.0, 600.0, 300.0)
        for dt in dt_sequence:
            ref_state, ref_blended = step_fn(
                ref_state, forcing, tile_cfg, ocean_sst, ocean_u, ocean_v, dt,
            )
            part_state, part_blended = step_fn(
                part_state,
                forcing,
                tile_cfg_part,
                ocean_sst,
                ocean_u,
                ocean_v,
                dt,
            )
            part_state = _mask_face_leading_pytree(
                part_state, topology.local_face_ids,
            )
            part_blended = _mask_face_leading_pytree(
                part_blended, topology.local_face_ids,
            )
            got_state = gather_to_global(part_state)
            got_blended = gather_to_global(part_blended)

            if topology.rank == 0:
                _assert_surface_to_atm_close(got_blended, ref_blended)
                assert float(got_state.accumulator.total_dt) == pytest.approx(
                    float(ref_state.accumulator.total_dt),
                    abs=1.0e-9,
                )

        if topology.rank == 0:
            # After 600 + 600 + 300 with coupling_dt=900:
            # flush at second step leaves carry 300, then +300 => 600.
            assert float(ref_state.accumulator.total_dt) == pytest.approx(600.0, abs=1.0e-9)

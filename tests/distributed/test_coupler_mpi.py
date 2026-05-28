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


def _assert_surface_to_atm_close(
    got: SurfaceToAtm,
    ref: SurfaceToAtm,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
    face_ids: tuple[int, ...] | list[int] | None = None,
) -> None:
    """Assert two blended coupler outputs match on the given face ids.

    If ``face_ids`` is ``None``, compare every face (legacy single-rank
    behaviour).  Under MPI, pass the rank's local face ids so the
    comparison ignores non-local faces whose ``tile_cfg`` was modified
    to all-land (which legitimately changes their physics output).
    """
    for name in SurfaceToAtm._fields:
        g = getattr(got, name)
        r = getattr(ref, name)
        if face_ids is None:
            g_sel, r_sel = g, r
        else:
            idx = jnp.asarray(list(face_ids), dtype=jnp.int32)
            g_sel = g[idx]
            r_sel = r[idx]
        assert jnp.all(jnp.isfinite(g_sel)), f"{name}: non-finite in distributed output"
        assert jnp.all(jnp.isfinite(r_sel)), f"{name}: non-finite in reference output"
        assert jnp.allclose(g_sel, r_sel, atol=atol, rtol=rtol), f"{name}: mismatch"


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
        # FV3_3D iter-1060: ``part_state_new`` is shape (6, n, n) on
        # every rank (step_fn runs on full state).  The pre-iter-1060
        # test called ``gather_to_global(part_state_new)`` which is
        # designed for scattered (n_local, n, n) input — the heuristic
        # in ``gather_pytree`` skipped gathering when shape[0] != n_local,
        # returning the masked array unchanged so faces non-owned by
        # rank 0 stayed zero.  Direct ref-vs-got comparison then failed.
        #
        # Correct contract: each rank's locally-owned faces must match
        # the single-rank reference bit-for-bit.  Non-owned faces ran
        # under modified tile_cfg (all-land) and legitimately differ.
        # Drop the broken gather and compare per local face on every
        # rank (matches the FV3 step-fidelity test pattern).
        local_ids = topology.local_face_ids
        idx = jnp.asarray(list(local_ids), dtype=jnp.int32)
        _assert_surface_to_atm_close(
            part_blended, ref_blended, face_ids=local_ids,
        )
        assert jnp.allclose(
            part_state_new.land.T_soil.data[idx],
            ref_state_new.land.T_soil.data[idx],
            atol=1.0e-6, rtol=1.0e-6,
        )
        assert jnp.allclose(
            part_state_new.ice.h_ice.data[idx],
            ref_state_new.ice.h_ice.data[idx],
            atol=1.0e-6, rtol=1.0e-6,
        )
        assert jnp.allclose(
            part_state_new.lake.T_epi.data[idx],
            ref_state_new.lake.T_epi.data[idx],
            atol=1.0e-6, rtol=1.0e-6,
        )
        assert float(part_state_new.accumulator.total_dt) == pytest.approx(
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
            # FV3_3D iter-1060: see sibling test for why gather +
            # full-state comparison was broken under the iter-aa707bda
            # gather API.  Per-face coupler processing is independent
            # across faces, so feeding non-local-face junk forward
            # does not pollute local-face outputs at the next step.
            local_ids = topology.local_face_ids
            idx = jnp.asarray(list(local_ids), dtype=jnp.int32)
            _assert_surface_to_atm_close(
                part_blended, ref_blended, face_ids=local_ids,
            )
            # FV3_3D iter-1062 (codex iter-1061 WARN #2): also compare
            # the persisted state fields across steps, not just the
            # current-step blended output + total_dt scalar.  A
            # regression in carried-state mutation (e.g., a typo in
            # T_soil update during flush) would otherwise be silent.
            assert jnp.allclose(
                part_state.land.T_soil.data[idx],
                ref_state.land.T_soil.data[idx],
                atol=1.0e-6, rtol=1.0e-6,
            ), f"land.T_soil drifted at dt={dt}"
            assert jnp.allclose(
                part_state.ice.h_ice.data[idx],
                ref_state.ice.h_ice.data[idx],
                atol=1.0e-6, rtol=1.0e-6,
            ), f"ice.h_ice drifted at dt={dt}"
            assert jnp.allclose(
                part_state.lake.T_epi.data[idx],
                ref_state.lake.T_epi.data[idx],
                atol=1.0e-6, rtol=1.0e-6,
            ), f"lake.T_epi drifted at dt={dt}"
            assert float(part_state.accumulator.total_dt) == pytest.approx(
                float(ref_state.accumulator.total_dt),
                abs=1.0e-9,
            )

        if topology.rank == 0:
            # After 600 + 600 + 300 with coupling_dt=900:
            # flush at second step leaves carry 300, then +300 => 600.
            assert float(ref_state.accumulator.total_dt) == pytest.approx(600.0, abs=1.0e-9)

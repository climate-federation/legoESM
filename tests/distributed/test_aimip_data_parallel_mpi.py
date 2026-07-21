"""#985 item 1: data-parallel AIMIP chunked spectral trainer (REAL MPI). Run:
    mpirun -np 2 <py> -m pytest tests/distributed/test_aimip_data_parallel_mpi.py -v

Drives the REAL ``_train_spectral_loop`` chunked path with
``config.data_parallel=True`` under 2 ranks, a tiny SFNO, and a synthetic
2-sample chunk (no GCS).  Rank 0 trains sample 0, rank 1 trains sample 1; each
step averages the two gradients across ranks and applies the SAME update, so
the replicas must end BYTE-IDENTICAL — the core DP invariant (they never
diverge).  Also checks the run actually moved the weights (non-vacuous) and the
loss stayed finite.  Skips unless nranks == 2.

Requires JAX_ENABLE_X64=1 (spectral grid) and, on Apple Silicon, JAX_PLATFORMS=cpu.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.driver.compiled_segments import pack_carry
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.ml.channel_packing import PE3DChannelSpec
from legoesm.ml.sfno import SFNO, SFNOConfig

N_MAX, NLEV = 10, 3


def _nranks():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    except Exception:
        return 0, 1


def _carry(grid, T_val, q_v_val):
    n_lat, n_lon = grid.lat.shape[0], grid.lon.shape[0]
    s3, s2 = (n_lat, n_lon, NLEV), (n_lat, n_lon)
    state = HydrostaticState(
        u=Field(jnp.zeros(s3), name="u", dims=("lat", "lon", "lev"), units="m/s"),
        v=Field(jnp.zeros(s3), name="v", dims=("lat", "lon", "lev"), units="m/s"),
        T=Field(jnp.full(s3, T_val), name="T", dims=("lat", "lon", "lev"), units="K"),
        p_s=Field(jnp.full(s2, 101325.0), name="p_s", dims=("lat", "lon"), units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=("lat", "lon"), units="m2/s2"),
    )
    return pack_carry(
        state, q_v=jnp.ones(s3) * q_v_val, q_c=jnp.zeros(s3), q_r=jnp.zeros(s3),
        held_dT_rad=jnp.zeros(s3), held_sw_net_sfc=jnp.zeros(s2),
        held_lw_net_sfc=jnp.zeros(s2), held_sw_up_toa=jnp.zeros(s2),
        held_lw_up_toa=jnp.zeros(s2), held_sw_down_toa=jnp.zeros(s2), step_index=0,
    )


def _small_sfno(grid):
    from legoesm.training.neural_gcm_spectral import N_SFNO_FORCING_CHANNELS
    spec = PE3DChannelSpec(nlev=NLEV)
    cfg = SFNOConfig(
        in_channels=spec.n_channels + N_SFNO_FORCING_CHANNELS,
        out_channels=spec.n_channels, embed_dim=16, n_blocks=1,
        mlp_expansion=2, residual_prediction=False,
    )
    return SFNO(cfg, grid, key=jax.random.PRNGKey(42))


def _cfg(ckpt_dir):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
    return NeuralGCMSpectralConfig(
        n_max=N_MAX, n_levels=NLEV, dt=1800.0,
        pe_config=SpectralPEConfig(time_integrator="ssp_rk3"),
        lr=1e-4, warmup_steps=0, optimizer="adamw",
        rollout_curriculum=((1, 1),),   # 1 h lead x 1 epoch (one synced update)
        loss_config=LossConfig(multi_step_hours=(1,), multi_step_weights=(1.0,)),
        log_every=1, checkpoint_dir=str(ckpt_dir), data_parallel=True,
    )


def _loader_2samples(grid):
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
    # One chunk, two DISTINCT samples -> rank 0 gets sample 0, rank 1 gets 1.
    s0 = carry_to_spectral_state(_carry(grid, 280.0, 0.004), grid)
    s1 = carry_to_spectral_state(_carry(grid, 284.0, 0.005), grid)
    t0 = _carry(grid, 281.0, 0.0042)
    t1 = _carry(grid, 285.0, 0.0052)

    def _loader(start_chunk=0):
        if start_chunk > 0:
            return
        yield [s0, s1], [(t0,), (t1,)], None

    _loader.n_chunks = 1
    _loader.chunk_sizes = [2]   # required by the DP schedule/guard contract
    return _loader


def test_dp_replicas_stay_synced(tmp_path):
    rank, nproc = _nranks()
    if nproc != 2:
        pytest.skip("needs mpirun -np 2")
    from legoesm.training.neural_gcm_spectral import (
        _train_spectral_loop,
        make_sfno_spectral_physics,
    )
    from mpi4py import MPI

    grid = create_gaussian_grid(N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(NLEV, sigma_top=0.1)
    model0 = _small_sfno(grid)

    model, hist = _train_spectral_loop(
        model0, make_sfno_spectral_physics, grid, sigma, None, None,
        _cfg(tmp_path / "dp"),
        start_epoch=0, chunk_loader=_loader_2samples(grid),
        n_samples_total=2, resume_from_dir=None,
    )

    fin = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    init = jax.tree_util.tree_leaves(eqx.filter(_small_sfno(grid), eqx.is_array))
    # (a) non-vacuous: training actually moved the weights.
    assert any(not jnp.array_equal(i, f) for i, f in zip(init, fin)), (
        "DP run did not update the weights"
    )
    assert np.isfinite(hist[-1]), hist

    # (b) replica sync: the two ranks trained on DIFFERENT samples but, applying
    # the SAME cross-rank-averaged gradient, must end BYTE-IDENTICAL.
    flat = np.concatenate([np.asarray(leaf).ravel() for leaf in fin])
    gathered = MPI.COMM_WORLD.allgather(flat)
    assert np.array_equal(gathered[0], gathered[1]), (
        f"DP replicas diverged: max|delta|={np.max(np.abs(gathered[0]-gathered[1]))}"
    )

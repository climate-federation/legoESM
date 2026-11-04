"""The WB trainer's data-parallel step on the REAL WeatherBench loss (real MPI).

    mpirun -np 2 <py> -m pytest tests/distributed/test_wb_dp_gradient_mpi.py

N ranks (2 or 4), one sample each, through the trainer's own pieces (multi-lead loss
with the near-surface head, ``build_dp_value_and_grad``,
``mpi_data_parallel_train_step``): the cross-rank gradient and the updated
parameters must equal a single process's batch-mean gradient and update over
the same N samples, to float64 round-off (rtol 1e-12).  Skips at one
rank.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.ml.test_scale_build_spectral_core import _cfg, _synthetic_carry, _yml
from tests.unit.test_near_surface_head import _scales

jax.config.update("jax_enable_x64", True)   # rtol 1e-12 is a float64 claim

RTOL = 1e-12


def _nranks():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    except Exception:
        return 0, 1


def test_wb_data_parallel_step_equals_single_process_batch_mean():
    rank, nproc = _nranks()
    if nproc < 2:
        pytest.skip("needs mpirun -np >= 2")
    import optax
    from legoesm.training.data_parallel import (
        all_reduce_grad_mean,
        build_dp_value_and_grad,
        mpi_data_parallel_train_step,
    )
    from legoesm.training.losses import LossConfig
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
    from legoesm.training.scale_build import (
        build_mode_components,
        lead_diagnose_fn,
        multi_lead_loss,
    )

    from evaluations.wb_forecast import diagnose_headline_fields

    i, o = _scales()
    yml = {**_yml(), "near_surface_head": {"seed": 0, "inputs": i, "outputs": o},
           "learned_position_embedding": True, "loss": {"w_t2m": 1.0}}
    _, grid, sigma, params, mk, _, dt = build_mode_components(_cfg("neural_gcm"), yml)
    params = jax.tree.map(lambda x: x + 1e-2 if eqx.is_inexact_array(x) else x, params)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    cfg = LossConfig(w_t2m=1.0, t2m_resid_scale=2.0)

    def sample(k):
        ic = _synthetic_carry(grid, 8)
        ic = ic._replace(T=ic.T + 0.5 * k)
        f = {"T_sfc": jnp.full((ncol,), 300.0 - k), "sic": jnp.zeros((ncol,)),
             "land_frac": jnp.zeros((ncol,)), "day_of_year": jnp.asarray(80.0 + k),
             "seconds_of_day": jnp.asarray(21600.0 * k)}
        out = mk(params).raw_multi(ic, 1, 1, f)
        sfc = {key: diagnose_headline_fields(carry_to_spectral_state(out[0], grid),
                                             grid, sigma).fields[key] + 1.0 + k
               for key in ("t2m", "u10", "v10", "mslp")}
        return ic, ((out[0], sfc),), f

    arr, static = eqx.partition(params, eqx.is_inexact_array)

    def loss(a, s):
        p = eqx.combine(a, static)
        ic, tg, f = s
        return multi_lead_loss(
            mk(p), ic, tg, f, n_segs=1, seg_steps=1,
            sigma_full=jnp.asarray(sigma.sigma_full), grid=grid, sigma=sigma,
            loss_config=cfg, lead_segs=(1,),
            diagnose_fn=lead_diagnose_fn(diagnose_headline_fields, p, f, dt))

    vg = build_dp_value_and_grad(loss)
    samples = [sample(k) for k in range(nproc)]
    grads = [vg(arr, s)[1] for s in samples]
    g_ref = jax.tree.map(lambda *g: sum(g) / nproc, *grads)
    mine = samples[rank]
    g_dp = all_reduce_grad_mean(vg(arr, mine)[1], nproc)
    for a, b in zip(jax.tree.leaves(g_dp), jax.tree.leaves(g_ref)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=0.0)
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(g_ref)) > 0.0

    opt = optax.adam(1e-3)
    new, _, _ = mpi_data_parallel_train_step(vg, arr, opt.init(arr), opt, mine, nproc)
    upd, _ = opt.update(g_ref, opt.init(arr), arr)
    ref = optax.apply_updates(arr, upd)
    for a, b in zip(jax.tree.leaves(new), jax.tree.leaves(ref)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=0.0)

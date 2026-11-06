"""Gradient-access gate across ranks (real MPI).

    mpirun -np 2 <py> -m pytest tests/distributed/test_gradient_access_gate_mpi.py

A gradient that cancels across ranks must abort; a failure on ONE rank (before
or after the gradient allreduce) must abort EVERY rank with the same error and
no hang.  Skips at one rank.
"""
import jax
import jax.numpy as jnp
import pytest

from legoesm.ml.training import TrainingConfig, create_optimizer, make_lr_schedule
from legoesm.training.inert_params import GradientAccessError, gradient_access_gate

jax.config.update("jax_enable_x64", True)

TC = TrainingConfig(lr=1e-2, optimizer="adamw", warmup_steps=4, total_steps=40,
                    weight_decay=0.1, grad_clip_norm=1.0)
X = jnp.array([0.3, -0.7, 1.1])


def _rank_size():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    except Exception:
        return 0, 1


RANK, NPROC = _rank_size()
pytestmark = pytest.mark.skipif(NPROC < 2, reason="needs mpirun -np >= 2")


def _loss(p, x):
    return jnp.sum((p["a"] * jnp.tanh(p["w"] * x) - 1.0) ** 2)


def _params():
    return {"w": jnp.array([0.5, -0.2, 0.8]), "a": jnp.array([1.5, 0.4, 0.9])}


def _run(loss=_loss, opt=None):
    opt = create_optimizer(TC) if opt is None else opt
    p = _params()
    vg = jax.value_and_grad(loss)
    return gradient_access_gate([(vg, X)] * 10, [p, opt.init(p)], opt,
                                make_lr_schedule(TC), optimizer_name="adamw",
                                num_processes=NPROC)


def test_healthy_passes_on_every_rank():
    assert _run()["k"] == 4


def test_gradient_cancelling_across_ranks_aborts():
    sign = 1.0 if RANK % 2 == 0 else -1.0   # +g on even ranks, -g on odd

    def loss(p, x):
        return _loss(p, x) + sign * 5.0 * jnp.sum(p["extra"] * x)
    opt = create_optimizer(TC)
    p = {**_params(), "extra": jnp.array([0.2, 0.1, -0.3])}
    vg = jax.value_and_grad(loss)
    if NPROC % 2:
        pytest.skip("cancellation needs an even rank count")
    with pytest.raises(GradientAccessError, match=r"\['extra'\].*no_grad=3"):
        gradient_access_gate([(vg, X)] * 10, [p, opt.init(p)], opt,
                             make_lr_schedule(TC), optimizer_name="adamw",
                             num_processes=NPROC)


def test_exception_on_one_rank_before_allreduce_aborts_all():
    def loss(p, x):
        if RANK == 1:
            raise MemoryError("injected RESOURCE_EXHAUSTED")
        return _loss(p, x)
    with pytest.raises(GradientAccessError, match="rehearsal step 1 failed"):
        _run(loss=loss)


def test_non_finite_loss_on_one_rank_aborts_all():
    def loss(p, x):
        return _loss(p, x) * (jnp.nan if RANK == 1 else 1.0)
    with pytest.raises(GradientAccessError, match="rehearsal step 1 failed"):
        _run(loss=loss)


def test_exception_on_one_rank_after_allreduce_aborts_all():
    real = create_optimizer(TC)

    class Boom:
        init = staticmethod(real.init)

        @staticmethod
        def update(g, s, p):
            if RANK == 1:
                raise MemoryError("injected RESOURCE_EXHAUSTED in update")
            return real.update(g, s, p)
    with pytest.raises(GradientAccessError, match="rehearsal step 1 update failed"):
        _run(opt=Boom)


def test_tree_mismatch_between_ranks_aborts_all():
    opt = create_optimizer(TC)
    p = _params() if RANK == 0 else {**_params(), "z": jnp.ones(2)}
    with pytest.raises(GradientAccessError, match="differs between ranks"):
        gradient_access_gate([(jax.value_and_grad(_loss), X)] * 10,
                             [p, opt.init(p)], opt, make_lr_schedule(TC),
                             optimizer_name="adamw", num_processes=NPROC)


def test_ranks_running_out_at_different_points_abort_everywhere():
    """A short rank must not let the gate pass while a longer one still has
    steps (or a failure) pending."""
    opt = create_optimizer(TC)
    p = _params()
    vg = jax.value_and_grad(_loss)
    n = 3 if RANK == 0 else 10
    with pytest.raises(GradientAccessError, match="different points"):
        gradient_access_gate([(vg, X)] * n, [p, opt.init(p)], opt,
                             make_lr_schedule(TC), optimizer_name="adamw",
                             num_processes=NPROC)


def test_every_rank_running_out_together_passes():
    opt = create_optimizer(TC)
    p = _params()
    vg = jax.value_and_grad(_loss)
    rep = gradient_access_gate([(vg, X)] * 3, [p, opt.init(p)], opt,
                               make_lr_schedule(TC), optimizer_name="adamw",
                               num_processes=NPROC)
    assert rep["k"] == 3

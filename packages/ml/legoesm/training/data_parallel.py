"""Data-parallel training: replicate the model across ranks, shard the ERA5
batch, average gradients each step.

Wraps the single-device per-sample training step (``training_driver``) in a
data-parallel layer so N ranks (1 GPU each) train the SAME replicated model on
DISTINCT ERA5 shards, averaging gradients every step. Because every rank applies
the identical averaged gradient, the replicas stay in sync without broadcasting
weights.

Multi-process is via ``jax.distributed`` (armed in the entry BEFORE any
``jax.numpy`` import). The cross-device gradient average uses the same collective
idiom the repo already uses spatially: ``jax.lax.pmean(x, axis_name)`` inside a
``jax.pmap(..., axis_name="data")`` — process-transparent under ``jax.distributed``.

The gradient-average helper is validated by a 2-CPU-device equivalence test
(``tests/ml/test_data_parallel.py``) BEFORE any GPU-hours: the data-parallel mean
gradient must match a serial batch-mean gradient bit-closely.
"""
from __future__ import annotations

__all__ = [
    "mpi_rank_size",
    "shard_samples",
    "all_reduce_grad_mean",
    "mpi_data_parallel_train_step",
    "mpi_data_parallel_training_loop",
    "data_parallel_value_and_grad",
    "data_parallel_training_loop",
    "mpi_abort_on_uncaught",
]


def _abort_multirank_job(comm, *, code=1):
    """``MPI_Abort`` the whole job iff ``comm`` has more than one rank.

    ``comm=None`` resolves ``MPI.COMM_WORLD`` when ``mpi4py`` is importable, else
    no-ops (single process / laptop). Any failure to abort is swallowed so the
    caller's original exception is the one that propagates. Kept separate from the
    decorator so a fake comm (``.Get_size()``/``.Abort()``) can unit-test the
    rank-gate without a real MPI runtime.
    """
    if comm is None:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        except Exception:
            return
    try:
        if comm.Get_size() > 1:
            import sys

            sys.stderr.flush()  # don't lose the traceback when Abort SIGKILLs us
            comm.Abort(code)
    except Exception:
        pass


def mpi_abort_on_uncaught(fn=None, *, comm=None, code=1):
    """Decorator: if the wrapped call raises under a MULTI-rank MPI job,
    ``MPI_Abort`` the whole job BEFORE propagating.

    Prevents a dead rank (OOM, NaN, a failed chunk load) from leaving its peers
    hung forever in the next gradient allreduce — the failure mode reported on the
    first 4xA100 T106 data-parallel run (#985), where an OOM on 3 of 4 ranks left
    rank 0 blocked in the allreduce until walltime (~11 h). Aborting on any rank
    converts that hang into an immediate, clean job death.

    Single-rank / no ``mpi4py`` -> transparent passthrough (the exception just
    propagates), so serial training and unit tests are unaffected. Usable bare
    (``@mpi_abort_on_uncaught``) or parameterised (``@mpi_abort_on_uncaught(comm=c)``).
    """
    import functools

    def _deco(f):
        @functools.wraps(f)
        def _wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except BaseException:
                _abort_multirank_job(comm, code=code)
                raise

        return _wrapped

    return _deco if fn is None else _deco(fn)


def mpi_rank_size():
    """``(rank, num_processes)`` from MPI.

    RAISES if a multi-rank launcher IS present (SLURM/PMI/OMPI/MPICH env) but
    ``mpi4py`` init fails — otherwise every rank would silently train
    independently with NO cross-rank gradient average (16 diverging replicas,
    not one data-parallel model). Returns ``(0, 1)`` only when no multi-rank
    launcher is detected (single process / laptop), so ``num_processes <= 1``
    callers take their identity/serial path unchanged.
    """
    import os

    launcher = 1
    for v in ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALNRANKS"):
        val = os.environ.get(v, "")
        if val.isdigit():
            launcher = max(launcher, int(val))
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except Exception as exc:
        if launcher > 1:
            raise RuntimeError(
                f"multi-rank launcher detected (size={launcher}) but mpi4py init "
                f"failed ({exc}); gradients would NOT be averaged across ranks -- "
                f"aborting"
            ) from exc
        return 0, 1


def shard_samples(items, process_id, num_processes, *, drop_remainder=True):
    """Contiguous, disjoint, equal split of a sample list across ranks.

    With ``drop_remainder`` (default) every rank gets ``len(items)//num_processes``
    items so collectives stay balanced (an unequal split would deadlock a pmean).
    Single process -> identity.
    """
    n = len(items)
    if num_processes <= 1:
        return list(items)
    if drop_remainder:
        per = n // num_processes
    else:
        per = -(-n // num_processes)  # ceil
    start = process_id * per
    return list(items[start:start + per])


# ======================================================================
# PRIMARY multi-node path: 1 GPU per rank, cross-RANK gradient average via MPI.
# (The pmap helper below is the multi-GPU-PER-PROCESS sub-case.)
# ======================================================================

def all_reduce_grad_mean(grad, num_processes, *, comm=None, bucket=True):
    """Cross-RANK (process) gradient MEAN via mpi4jax ``allreduce(SUM) / N``.

    The primary multi-node data-parallel reduction: each rank (1 GPU) computes a
    local gradient on its ERA5 shard; this averages the gradient pytree across
    all ranks so every rank applies the identical update (replicas stay in sync).
    Reuses ``legoesm.parallel.reductions.global_sum_mpi`` (AD-safe: full VJP via
    ``allreduce(SUM)``). ``num_processes <= 1`` -> identity (single-rank / laptop).

    ``bucket`` (default True): pack all inexact leaves of a given dtype into
    ONE flat buffer and issue a SINGLE ``allreduce`` per dtype, instead of one
    collective PER LEAF. A big model (SFNO, neural-GCM) has hundreds of
    parameter leaves; the per-leaf reduction pays hundreds of full network
    latencies per step, and the multi-GPU leg is latency-bound (the halo-count
    analysis and the Derecho f64≈f32 curves), so bucketing collapses that to
    one (or a few, if dtypes are mixed) message per step. The reduction stays
    in each leaf's NATIVE dtype (grouping BY dtype — no lossy promotion), so
    the averaged gradient is NUMERICALLY equivalent to the per-leaf path: it
    reduces the identical values in the identical dtype. It is not bit-for-bit
    guaranteed across all rank counts — floating ``SUM`` is non-associative and
    packing changes each message's size, which can lead MPI to pick a
    different reduction tree, so at ≥3 ranks the two paths may differ in the
    last ULP (exactly as two ``allreduce`` calls on different buffer sizes
    already can). Set ``bucket=False`` to force the legacy per-leaf reduction.

    Only INEXACT (float/complex) leaves are reduced; non-inexact leaves (int
    counters) and non-array leaves pass through untouched — the same leaves a
    real ``jax.grad`` pytree never carries as gradients. Python float scalars
    ARE reduced (normalized to arrays, matching the legacy ``tree_map`` which
    also arrays them).
    """
    if num_processes <= 1:
        return grad
    import jax
    import jax.numpy as jnp
    from legoesm.parallel.reductions import global_sum_mpi

    inv = 1.0 / float(num_processes)
    if not bucket:
        return jax.tree_util.tree_map(
            lambda g: global_sum_mpi(g, comm=comm) * inv, grad)

    def _as_inexact_array(leaf):
        """Return ``leaf`` as an array iff it is inexact (float/complex) —
        including a Python float scalar, which the legacy per-leaf path also
        reduced (and arrayed). Returns ``None`` for anything not reducible
        (ints, None, exotic leaves), which then passes through untouched."""
        try:
            arr = jnp.asarray(leaf)
        except (TypeError, ValueError):
            return None
        return arr if jnp.issubdtype(arr.dtype, jnp.inexact) else None

    leaves, treedef = jax.tree_util.tree_flatten(grad)
    normalized = [_as_inexact_array(leaf) for leaf in leaves]
    # Bucket the reducible leaves by dtype, preserving first-seen order for
    # determinism across ranks (every rank has the identical pytree, so the
    # bucket order agrees).
    buckets: dict[object, list[int]] = {}
    for i, arr in enumerate(normalized):
        if arr is not None:
            buckets.setdefault(arr.dtype, []).append(i)

    out = list(leaves)
    for dtype, idxs in buckets.items():
        arrs = [normalized[i] for i in idxs]
        shapes = [a.shape for a in arrs]
        sizes = [int(a.size) for a in arrs]
        packed = jnp.concatenate([a.reshape(-1) for a in arrs], axis=0)
        reduced = global_sum_mpi(packed, comm=comm) * inv   # ONE allreduce
        offset = 0
        for i, shape, size in zip(idxs, shapes, sizes):
            out[i] = reduced[offset:offset + size].reshape(shape)
            offset += size
    return jax.tree_util.tree_unflatten(treedef, out)


def mpi_data_parallel_train_step(loss_fn, params, opt_state, optimizer, sample,
                                 num_processes, *, comm=None):
    """One data-parallel step: local ``value_and_grad`` on this rank's sample,
    mean the gradient across ranks, then ``optimizer.update`` + ``apply_updates``.

    ``loss_fn(params, sample) -> scalar``. Because every rank applies the same
    cross-rank-averaged gradient, the replicas remain identical without any weight
    broadcast. Returns ``(params, opt_state, mean_loss)``.
    """
    import jax
    import optax

    loss, grad = jax.value_and_grad(loss_fn)(params, sample)
    grad = all_reduce_grad_mean(grad, num_processes, comm=comm)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    if num_processes > 1:
        from legoesm.parallel.reductions import global_sum_mpi
        loss = global_sum_mpi(loss, comm=comm) / float(num_processes)
    return params, opt_state, loss


@mpi_abort_on_uncaught
def mpi_data_parallel_training_loop(loss_fn, params, opt_state, optimizer,
                                    local_samples, n_epochs, num_processes, *,
                                    comm=None, on_epoch=None):
    """Per-rank loop over this rank's ERA5 shard, gradients averaged across ranks
    each step. All ranks run lockstep (balanced shards from ``shard_samples`` with
    ``drop_remainder``), so the per-step ``allreduce`` never deadlocks. Rank-0
    logging/checkpointing belongs in ``on_epoch(epoch, mean_loss)``. Returns
    ``(params, opt_state, history)``.
    """
    import jax

    history = []
    for epoch in range(n_epochs):
        losses = []
        for sample in local_samples:
            # #1286 fix A: samples are kept HOST-resident (numpy leaves) so the
            # whole training set is never parked in device memory; move ONE
            # sample onto the device here and let it free at the next iteration.
            # A no-op (cheap) when the sample is already device-resident (the
            # legacy eager path), so this is safe for both.
            sample = jax.device_put(sample)
            params, opt_state, loss = mpi_data_parallel_train_step(
                loss_fn, params, opt_state, optimizer, sample, num_processes, comm=comm)
            losses.append(float(loss))
        mean_loss = sum(losses) / max(len(losses), 1)
        history.append(mean_loss)
        if on_epoch is not None:
            on_epoch(epoch, mean_loss, params, opt_state)   # CURRENT params (not a stale closure)
    return params, opt_state, history


# ======================================================================
# Multi-GPU-PER-PROCESS sub-case: average across a process's LOCAL devices.
# STRICTLY local-device only -- does NOT average across MPI ranks/processes. For
# multi-node (1 GPU/rank) use the mpi_data_parallel_* path above. Composing the
# two (local pmean THEN cross-rank all_reduce_grad_mean) is possible but unused.
# ======================================================================

def data_parallel_value_and_grad(loss_fn, params, batched_args):
    """Per-device ``value_and_grad`` of ``loss_fn(params, x)`` over the leading
    axis of ``batched_args``, mean-reduced across devices.

    ``batched_args`` has a leading axis of size ``jax.local_device_count()`` (one
    sample slice per device). Returns ``(mean_loss, mean_grad)`` with the gradient
    averaged across this process's LOCAL devices only -- it does NOT reduce across
    MPI ranks/processes (use ``all_reduce_grad_mean`` for that).

    Single-device -> serial mean over the batch axis (no pmap), so the same code
    path runs on one GPU or a laptop.
    """
    import jax
    import jax.numpy as jnp

    n_dev = jax.local_device_count()

    if n_dev == 1:
        # serial: vmap value_and_grad over the batch axis, mean-reduce
        losses, grads = jax.vmap(
            jax.value_and_grad(loss_fn), in_axes=(None, 0))(params, batched_args)
        mean_grad = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads)
        return jnp.mean(losses), mean_grad

    def _step(p, x):
        loss, grad = jax.value_and_grad(loss_fn)(p, x)
        return jax.lax.pmean(loss, "data"), jax.lax.pmean(grad, "data")

    # params replicated across the device axis; x already sharded along it
    p_rep = jax.tree_util.tree_map(
        lambda a: jnp.broadcast_to(a, (n_dev,) + jnp.shape(a)), params)
    losses, grads = jax.pmap(_step, axis_name="data")(p_rep, batched_args)
    # pmean made every device identical -> take device 0
    return losses[0], jax.tree_util.tree_map(lambda g: g[0], grads)


def data_parallel_training_loop(loss_fn, params, opt_state, optimizer,
                                sample_batches, n_epochs, *, on_epoch=None):
    """Data-parallel training over ``sample_batches`` for ``n_epochs``.

    Each batch has a leading device axis (``jax.local_device_count()``). Per step:
    average the gradient across devices/ranks, then ``optimizer.update`` +
    ``apply_updates`` — identical on every rank, so replicas stay in sync.

    ``on_epoch(epoch, mean_loss)`` is called once per epoch (rank-0 logging /
    checkpointing belongs there). Returns ``(params, opt_state, history)``.
    """
    import optax

    history = []
    for epoch in range(n_epochs):
        epoch_losses = []
        for batched_args in sample_batches:
            mean_loss, mean_grad = data_parallel_value_and_grad(
                loss_fn, params, batched_args)
            updates, opt_state = optimizer.update(mean_grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_losses.append(float(mean_loss))
        mean_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        history.append(mean_epoch_loss)
        if on_epoch is not None:
            on_epoch(epoch, mean_epoch_loss, params, opt_state)   # CURRENT params
    return params, opt_state, history

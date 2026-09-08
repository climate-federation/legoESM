"""No trainable parameter may sit at a zero gradient unnoticed."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from legoesm.training.inert_params import (
    freeze_unreachable,
    measure_leaf_reachability,
    mpi_max_reduce,
    probe_indices,
    trainable_filter_spec,
)


class _Params(eqx.Module):
    live: jax.Array
    dead: jax.Array
    label: str = eqx.field(static=True)


def _params():
    return _Params(live=jnp.asarray([1.0, 2.0]), dead=jnp.asarray(3.0),
                   label="x")


def _vg(loss):
    """The probe's contract: ``(arr, sample) -> (loss, grad)``."""
    return eqx.filter_jit(jax.value_and_grad(loss))


def _grad_fn(scale_dead=0.0):
    """Loss reads ``live`` always and ``dead`` only through ``scale_dead``."""
    def loss(a, sample):
        return jnp.sum(a.live * sample) + scale_dead * a.dead

    return _vg(loss)


def test_an_unreachable_leaf_is_identified():
    absmax, names, n_used = measure_leaf_reachability(
        _grad_fn(), _params(), [1.0, 2.0, 3.0], n_probe=3)
    assert n_used == 3
    dead = [n for n, m in zip(names, absmax) if m == 0.0]
    assert len(dead) == 1 and "dead" in dead[0]


def test_a_reachable_leaf_is_not_frozen():
    absmax, names, _ = measure_leaf_reachability(
        _grad_fn(scale_dead=0.5), _params(), [1.0, 2.0], n_probe=2)
    assert all(m > 0.0 for m in absmax), dict(zip(names, absmax))


def test_the_filter_spec_freezes_exactly_the_unreachable_leaf():
    p = _params()
    absmax, _names, _ = measure_leaf_reachability(
        _grad_fn(), p, [1.0, 2.0], n_probe=2)
    spec = trainable_filter_spec(p, absmax)
    arr, static = eqx.partition(p, spec)
    assert arr.live is not None and arr.dead is None
    # The frozen leaf keeps its VALUE — it is still applied, just not updated.
    assert float(eqx.combine(arr, static).dead) == 3.0


def test_the_filter_spec_refuses_a_mismatched_reachability_vector():
    with pytest.raises(ValueError, match="array leaves"):
        trainable_filter_spec(_params(), [1.0])


def test_a_poisoned_sample_is_not_evidence():
    """A scene with a non-finite gradient must be skipped, not treated as
    proof that every parameter is unreachable."""
    def loss(a, sample):
        return jnp.sum(a.live * sample) + 0.5 * a.dead / sample

    g = _vg(loss)
    absmax, _names, n_used = measure_leaf_reachability(
        g, _params(), [0.0, 2.0], n_probe=2)
    assert n_used == 1
    assert all(m > 0.0 for m in absmax)


def test_every_probe_sample_poisoned_is_an_error():
    def loss(a, sample):
        return jnp.sum(a.live * sample) / sample

    g = _vg(loss)
    with pytest.raises(ValueError, match="non-finite"):
        measure_leaf_reachability(g, _params(), [0.0, 0.0], n_probe=2)


def test_the_reduce_hook_runs_before_the_all_poisoned_check():
    """The cross-rank combine must happen even when THIS rank saw nothing
    usable — a collective skipped on the way to a rank-local raise hangs the
    job instead of failing it."""
    def loss(a, sample):
        return jnp.sum(a.live * sample) / sample

    g = _vg(loss)
    called = []

    def _reduce(absmax, n_used):
        called.append(n_used)
        return [1.0] * len(absmax), 3          # another rank had usable scenes

    absmax, _names, n_used = measure_leaf_reachability(
        g, _params(), [0.0, 0.0], n_probe=2, reduce=_reduce)
    assert called == [0] and n_used == 3 and all(m > 0.0 for m in absmax)


def test_single_process_needs_no_reduction():
    assert mpi_max_reduce(1) is None
    assert mpi_max_reduce(4) is not None


def test_freeze_unreachable_splits_and_recombines():
    p = _params()
    trainable, frozen, names, n_used = freeze_unreachable(
        p, _grad_fn(), [1.0, 2.0, 3.0, 4.0], n_probe=2, device_put=False)
    assert n_used == 2
    assert len(names) == 1 and "dead" in names[0]
    assert trainable.dead is None and frozen.live is None
    assert float(eqx.combine(trainable, frozen).dead) == 3.0


def test_freeze_unreachable_probes_across_the_record_not_the_front():
    """The first samples of a WeatherBench shard are the same weather; a leaf
    that only fires later must not be frozen on their evidence."""
    def loss(a, sample):
        return jnp.sum(a.live * sample) + jnp.where(sample > 3.0, 1.0, 0.0) * a.dead

    g = _vg(loss)
    samples = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
    _t, _f, names, _n = freeze_unreachable(_params(), g, samples, n_probe=2,
                                           device_put=False)
    assert names == []


def test_freeze_unreachable_needs_samples():
    with pytest.raises(ValueError, match="no samples"):
        freeze_unreachable(_params(), _grad_fn(), [], device_put=False)


def test_the_filter_spec_rejects_a_vector_measured_on_another_tree():
    """Same LENGTH, different leaves — the length check alone would pass it and
    freeze the wrong parameter."""
    absmax, names, _ = measure_leaf_reachability(
        _grad_fn(), _params(), [1.0, 2.0], n_probe=2)
    with pytest.raises(ValueError, match="different pytree"):
        trainable_filter_spec(_params(), absmax,
                              ["." + n.strip(".") + "_elsewhere" for n in names])


@pytest.mark.parametrize("n_probe", [0, -1])
def test_freeze_unreachable_rejects_a_nonsense_probe_count(n_probe):
    with pytest.raises(ValueError, match="n_probe"):
        freeze_unreachable(_params(), _grad_fn(), [1.0, 2.0], n_probe=n_probe,
                           device_put=False)


def test_a_frozen_leaf_survives_a_checkpoint_round_trip(tmp_path):
    """`on_epoch` serialises eqx.combine(trainable, frozen) — the frozen value
    must be the one that lands in the checkpoint."""
    trainable, frozen, _names, _n = freeze_unreachable(
        _params(), _grad_fn(), [1.0, 2.0], n_probe=2, device_put=False)
    path = tmp_path / "epoch.eqx"
    eqx.tree_serialise_leaves(str(path), eqx.combine(trainable, frozen))
    back = eqx.tree_deserialise_leaves(str(path), _params())
    assert float(back.dead) == 3.0
    assert [float(x) for x in back.live] == [1.0, 2.0]


def test_the_probe_does_not_retrace_on_host_resident_samples():
    """Samples arrive host-resident; a bare numpy scalar leaf would be frozen
    as a static argument and recompile the probe on every call."""
    import numpy as np

    traces = []

    def loss(a, sample):
        traces.append(1)
        return jnp.sum(a.live * sample["x"]) * sample["k"]

    g = _vg(loss)
    # A bare PYTHON float is the leaf that bites: equinox counts a numpy
    # scalar as an array, but not this one, so filter_jit freezes it as a
    # static argument.  Without the device_put this probe traces 4 times.
    samples = [{"x": np.asarray([1.0, 2.0]), "k": float(i + 1)}
               for i in range(4)]
    freeze_unreachable(_params(), g, samples, n_probe=4, device_put=True)
    assert len(traces) == 1, f"probe retraced {len(traces)} times"


def test_a_nan_loss_with_a_finite_gradient_is_not_evidence():
    """The dangerous poisoned scene: the loss is NaN but the gradient comes
    back finite (here exactly zero).  Counting it would freeze every parameter
    while looking perfectly clean."""
    def loss(a, sample):
        return jnp.sum(a.live * sample) + jnp.where(sample == 0.0, jnp.nan, 0.0)

    with pytest.raises(ValueError, match="non-finite loss or gradient"):
        measure_leaf_reachability(_vg(loss), _params(), [0.0], n_probe=1)


@pytest.mark.parametrize("n_samples,n_probe,expected", [
    (6, None, [0, 1, 2, 3, 4, 5]),   # the default: every sample, no subsampling
    (6, 2, [0, 5]),            # both ends, not [0, 3]
    (6, 3, [0, 2, 5]),
    (6, 1, [0]),
    (3, 8, [0, 1, 2]),         # more probes than samples
    (2, 3, [0, 1]),            # rounding duplicates collapse
    (1, 4, [0]),
])
def test_probe_indices_span_the_record(n_samples, n_probe, expected):
    assert probe_indices(n_samples, n_probe) == expected


@pytest.mark.parametrize("n_samples,n_probe", [(0, 4), (4, 0), (4, -1)])
def test_probe_indices_reject_nonsense(n_samples, n_probe):
    with pytest.raises(ValueError):
        probe_indices(n_samples, n_probe)


def test_the_probe_stages_one_sample_at_a_time(monkeypatch):
    """Samples are host-resident on purpose; the probe must move them one at a
    time, not park the whole set on the device before the first evaluation."""
    import numpy as np

    order = []
    real_put = jax.device_put
    monkeypatch.setattr(jax, "device_put",
                        lambda x, *a, **k: (order.append("stage"),
                                            real_put(x, *a, **k))[1])

    def loss(a, sample):
        order.append("evaluate")               # once — the trace
        return jnp.sum(a.live * sample["x"])

    samples = [{"x": np.asarray([1.0, 2.0])} for _ in range(4)]
    freeze_unreachable(_params(), _vg(loss), samples, n_probe=4)
    assert order.count("stage") == 4
    # The first evaluation happens after ONE staging, not after all four.
    assert order.index("evaluate") == 1, order


def test_a_non_finite_gradient_proves_the_leaf_is_connected():
    """A poisoned scene is unusable for MAGNITUDE, but a leaf whose gradient
    came back non-finite is manifestly wired to the loss.  Dropping that
    evidence is how a live parameter gets frozen on a bad scene: here ``dead``
    is inf on the skipped scene and exactly zero on the usable one."""
    def loss(a, sample):
        return jnp.sum(a.live * sample) + jnp.where(sample == 0.0,
                                                    a.dead / sample,
                                                    0.0 * a.dead)

    trainable, _frozen, names, n_used = freeze_unreachable(
        _params(), _vg(loss), [0.0, 2.0], n_probe=2, device_put=False)
    assert n_used == 1
    assert names == [], f"froze a leaf the poisoned scene proved live: {names}"
    assert trainable.dead is not None


def test_the_freeze_reads_every_sample_by_default():
    """The frozen set must cover the WHOLE shard: a leaf that is live on one
    interior scene and dead on the rest must survive, which a subsample
    misses."""
    def loss(a, sample):
        return jnp.sum(a.live * sample) + jnp.where(sample > 3.0, 1.0, 0.0) * a.dead

    samples = [1.0] * 20 + [5.0] + [1.0] * 3      # live on ONE interior scene
    _t, _f, names, n_used = freeze_unreachable(
        _params(), _vg(loss), samples, device_put=False)
    assert n_used == len(samples)
    assert names == [], "the default probe must read every sample"
    # ... and the cheap 4-scene subsample is what would have got it wrong.
    _t, _f, names_sub, _ = freeze_unreachable(
        _params(), _vg(loss), samples, n_probe=4, device_put=False)
    assert names_sub, "control: a subsample does miss the one live scene"


def test_a_poisoned_scene_still_proves_the_other_leaves_live():
    """The scene is unusable, but the leaves whose gradients came back FINITE
    on it were measured correctly.  Here ``dead``'s only non-zero gradient is
    on the scene that ``live`` poisons; discarding the scene wholesale would
    freeze a parameter the loss plainly moves."""
    def loss(a, sample):
        return (jnp.sum(a.live * sample) / sample) + jnp.where(
            sample == 0.0, a.dead, 0.0 * a.dead)

    _t, _f, names, n_used = freeze_unreachable(
        _params(), _vg(loss), [0.0, 2.0], device_put=False)
    assert n_used == 1                      # only the good scene is usable
    assert names == [], f"froze a leaf the poisoned scene measured live: {names}"

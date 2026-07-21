"""Direct tests for the lat-band SCATTER primitive ``shard_leaf`` (issue #1100).

``shard_leaf`` replaces the ``jax.device_put(full_global, NamedSharding)`` scatter
that all-gathered a transient second global copy per process (rc=137 OOM under
128-proc CPU packing). The multiprocess branch instead contributes each process's
addressable lat-band via ``make_array_from_process_local_data`` — no all-gather.

Run with 4 forced host devices so the sharding actually splits the leading axis:
    XLA_FLAGS=--xla_force_host_platform_device_count=4
The true cross-PROCESS collective elision is covered by the
``*_multicontroller_selfspawn`` tests (which spawn real processes); here
``process_count()==1`` so the multiprocess branch is exercised for its index
math + assembly and must match ``device_put`` byte-for-byte.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from legoesm.parallel.latlon_spmd import shard_leaf


def _mesh4():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]), axis_names=("lat",))


def test_single_process_branch_is_device_put():
    mesh = _mesh4()
    arr = np.arange(8 * 3 * 2, dtype=np.float64).reshape(8, 3, 2)
    sh = NamedSharding(mesh, P("lat", None, None))
    out = shard_leaf(arr, sh, multiprocess=False)
    assert out.sharding == sh
    assert np.array_equal(np.asarray(out), arr)


def test_multiprocess_branch_matches_device_put():
    # process_count()==1 here: the make_array_from_process_local_data path must
    # reassemble the SAME global array (and same sharding) as a plain device_put.
    mesh = _mesh4()
    arr = np.arange(8 * 3 * 2, dtype=np.float64).reshape(8, 3, 2)
    sh = NamedSharding(mesh, P("lat", None, None))
    ref = jax.device_put(arr, sh)
    out = shard_leaf(arr, sh, multiprocess=True)
    assert out.sharding == ref.sharding
    assert np.array_equal(np.asarray(out), np.asarray(ref))
    assert np.array_equal(np.asarray(out), arr)  # exact global roundtrip


def test_multiprocess_2d_leading_shard():
    # A 2-D leaf (e.g. p_s: (n_lat, n_lon)) scatters on its leading axis too.
    mesh = _mesh4()
    arr = np.arange(8 * 5, dtype=np.float64).reshape(8, 5)
    sh = NamedSharding(mesh, P("lat", None))
    out = shard_leaf(arr, sh, multiprocess=True)
    assert np.array_equal(np.asarray(out), arr)

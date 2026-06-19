"""shard_map version-compat shim (ponytail dedup 2026-06-17): the 6 parallel
backends import shard_map from one place instead of copy-pasting the
try/except across modules."""
from __future__ import annotations


def test_shard_map_importable_and_callable():
    from legoesm.parallel.shard_map_compat import shard_map
    assert callable(shard_map)


def test_is_the_jax_shard_map():
    import jax
    from legoesm.parallel.shard_map_compat import shard_map
    expected = getattr(jax, "shard_map", None)
    if expected is None:  # older JAX fallback path
        from jax.experimental.shard_map import shard_map as expected
    assert shard_map is expected

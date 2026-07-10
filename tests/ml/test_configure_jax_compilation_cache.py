"""Fast, pure unit test for the shared persistent-compilation-cache helper.

``legoesm.ml.training.configure_jax_compilation_cache`` is the ONE implementation
of the JAX persistent on-disk compilation-cache policy, reused by the correction
campaign and the Stage-B carbon calibration to amortize a large cold XLA compile
across launches.  These checks touch only ``jax.config`` (no model / no compile),
so the whole file runs in well under a second and needs no GPU / x64.

Contract under test:
- empty ``cache_dir`` => NO-OP: returns ``None`` and mutates NOTHING, so the
  default (cache-disabled) JAX behavior is preserved exactly;
- a non-empty ``cache_dir`` => sets ``jax_compilation_cache_dir`` and
  ``jax_persistent_cache_min_compile_time_secs`` and returns ``str(cache_dir)``;
- the ``min_compile_secs`` argument (default 30.0) is honored.
"""
from __future__ import annotations

import jax

from legoesm.ml.training import configure_jax_compilation_cache


def test_empty_dir_is_noop_and_returns_none():
    """An empty dir must NOT touch jax.config (default behavior preserved)."""
    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs

    assert configure_jax_compilation_cache("") is None
    assert configure_jax_compilation_cache("", 5.0) is None
    # No mutation on the no-op path.
    assert jax.config.jax_compilation_cache_dir == before_dir
    assert jax.config.jax_persistent_cache_min_compile_time_secs == before_secs


def test_nonempty_dir_sets_config_and_returns_dir(tmp_path):
    """A dir sets both persistent-cache config knobs and returns str(dir)."""
    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        d = str(tmp_path / "jax_cache")
        # default threshold (30 s)
        assert configure_jax_compilation_cache(d) == d
        assert jax.config.jax_compilation_cache_dir == d
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 30.0
        # explicit threshold is honored
        assert configure_jax_compilation_cache(d, 45.0) == d
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 45.0
    finally:  # restore global JAX state so sibling tests are unaffected
        jax.config.update("jax_compilation_cache_dir", before_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", before_secs)


def test_pathlike_dir_is_stringified(tmp_path):
    """A PathLike cache_dir is accepted and returned as its string form."""
    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        p = tmp_path / "jax_cache_pathlike"
        assert configure_jax_compilation_cache(p) == str(p)
        assert jax.config.jax_compilation_cache_dir == str(p)
    finally:
        jax.config.update("jax_compilation_cache_dir", before_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", before_secs)

"""Fast, pure unit test for the shared persistent-compilation-cache helper.

``legoesm.ml.training.configure_jax_compilation_cache`` is the ONE implementation
of the JAX persistent on-disk compilation-cache policy, reused by the correction
campaign and the Stage-B carbon calibration to amortize a large cold XLA compile
across launches.  These checks touch only ``jax.config`` (no model / no compile),
so the whole file runs in well under a second and needs no GPU / x64.

Contract under test:
- empty ``cache_dir`` => NO-OP: returns ``None`` and mutates NOTHING, so the
  default (cache-disabled) JAX behavior is preserved exactly;
- the CPU backend is a SAFETY-GATED NO-OP even with a non-empty ``cache_dir``:
  JAX's persistent cache key is not microarch-specific, so a cross-node AOT load
  on a heterogeneous CPU partition risks SIGILL / silently-wrong results -> the
  function returns ``None`` and mutates NOTHING;
- a NON-CPU (GPU/TPU) backend with a non-empty ``cache_dir`` ENABLES the cache:
  it sets ``jax_compilation_cache_dir`` + ``jax_persistent_cache_min_compile_time_secs``
  and returns ``str(cache_dir)`` (the device arch is in the key => safe);
- ``LEGOESM_ALLOW_CPU_COMPILE_CACHE=1`` opts a PROVEN-homogeneous CPU pool back
  into the cache (enables even on CPU);
- a failed backend probe fails SAFE (treated as CPU => disabled);
- the ``min_compile_secs`` argument (default 30.0) is honored.

The backend is mocked via ``jax.default_backend`` so every assertion is
deterministic regardless of the machine the test runs on.
"""
from __future__ import annotations

import jax

from legoesm.ml.training import (
    _CPU_COMPILE_CACHE_OPT_IN_ENV,
    configure_jax_compilation_cache,
)


def _force_backend(monkeypatch, name):
    """Pin ``jax.default_backend`` so the CPU gate is deterministic."""
    monkeypatch.setattr(jax, "default_backend", lambda: name)


def test_empty_dir_is_noop_and_returns_none(monkeypatch):
    """An empty dir must NOT touch jax.config (default behavior preserved).

    The empty-dir path returns before the backend is even probed, so it is a
    no-op on EVERY backend -- pin CPU to prove it does not depend on the gate.
    """
    _force_backend(monkeypatch, "cpu")
    monkeypatch.delenv(_CPU_COMPILE_CACHE_OPT_IN_ENV, raising=False)
    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs

    assert configure_jax_compilation_cache("") is None
    assert configure_jax_compilation_cache("", 5.0) is None
    # No mutation on the no-op path.
    assert jax.config.jax_compilation_cache_dir == before_dir
    assert jax.config.jax_persistent_cache_min_compile_time_secs == before_secs


def test_cpu_backend_is_noop_even_with_dir(monkeypatch, tmp_path):
    """CPU SAFETY GATE: a non-empty dir is a DISABLED no-op on CPU.

    JAX's persistent-cache key omits the CPU microarchitecture, so serving an
    AOT binary compiled on a different CPU risks SIGILL.  On CPU (and without
    the homogeneous-pool opt-in) the function must return ``None`` and mutate
    NOTHING, exactly like the cache-disabled default.
    """
    _force_backend(monkeypatch, "cpu")
    monkeypatch.delenv(_CPU_COMPILE_CACHE_OPT_IN_ENV, raising=False)
    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs

    d = str(tmp_path / "jax_cache")
    assert configure_jax_compilation_cache(d) is None
    assert configure_jax_compilation_cache(d, 45.0) is None
    # The disabled path must not have written EITHER config knob.
    assert jax.config.jax_compilation_cache_dir == before_dir
    assert jax.config.jax_persistent_cache_min_compile_time_secs == before_secs


def test_probe_failure_fails_safe_disabled(monkeypatch, tmp_path):
    """A failed backend probe fails SAFE: treated as CPU => disabled no-op."""
    def _boom():
        raise RuntimeError("backend probe unavailable")

    monkeypatch.setattr(jax, "default_backend", _boom)
    monkeypatch.delenv(_CPU_COMPILE_CACHE_OPT_IN_ENV, raising=False)
    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs

    d = str(tmp_path / "jax_cache_probe_fail")
    assert configure_jax_compilation_cache(d) is None
    assert jax.config.jax_compilation_cache_dir == before_dir
    assert jax.config.jax_persistent_cache_min_compile_time_secs == before_secs


def test_non_cpu_backend_enables_cache(monkeypatch, tmp_path):
    """A GPU/TPU backend ENABLES the cache (device arch is in the key => safe).

    Sets both persistent-cache config knobs and returns ``str(dir)``; the
    ``min_compile_secs`` argument (default 30 s) is honored.
    """
    _force_backend(monkeypatch, "gpu")
    monkeypatch.delenv(_CPU_COMPILE_CACHE_OPT_IN_ENV, raising=False)
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


def test_cpu_opt_in_env_enables_cache(monkeypatch, tmp_path):
    """``LEGOESM_ALLOW_CPU_COMPILE_CACHE=1`` re-enables the cache on CPU.

    For a user whose CPU pool is PROVEN homogeneous, the opt-in overrides the
    safety gate: on CPU the function then ENABLES the cache exactly like the
    accelerator path (sets the dir, returns ``str(dir)``).
    """
    _force_backend(monkeypatch, "cpu")
    monkeypatch.setenv(_CPU_COMPILE_CACHE_OPT_IN_ENV, "1")
    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        d = str(tmp_path / "jax_cache_optin")
        assert configure_jax_compilation_cache(d, 45.0) == d
        assert jax.config.jax_compilation_cache_dir == d
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 45.0
    finally:
        jax.config.update("jax_compilation_cache_dir", before_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", before_secs)


def test_cpu_opt_in_env_falsey_stays_disabled(monkeypatch, tmp_path):
    """A falsey opt-in value ("0") does NOT override the CPU safety gate."""
    _force_backend(monkeypatch, "cpu")
    monkeypatch.setenv(_CPU_COMPILE_CACHE_OPT_IN_ENV, "0")
    before_dir = jax.config.jax_compilation_cache_dir

    d = str(tmp_path / "jax_cache_optin_off")
    assert configure_jax_compilation_cache(d, 45.0) is None
    assert jax.config.jax_compilation_cache_dir == before_dir


def test_pathlike_dir_is_stringified(monkeypatch, tmp_path):
    """A PathLike cache_dir is accepted and returned as its string form.

    Uses the enabled (non-CPU) path so the string round-trip is observable.
    """
    _force_backend(monkeypatch, "gpu")
    monkeypatch.delenv(_CPU_COMPILE_CACHE_OPT_IN_ENV, raising=False)
    before_dir = jax.config.jax_compilation_cache_dir
    before_secs = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        p = tmp_path / "jax_cache_pathlike"
        assert configure_jax_compilation_cache(p) == str(p)
        assert jax.config.jax_compilation_cache_dir == str(p)
    finally:
        jax.config.update("jax_compilation_cache_dir", before_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", before_secs)

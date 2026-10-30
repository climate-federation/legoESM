"""Tests for the public persistent-compile-cache entrypoint and its wiring.

The RRTMGP radiation graph cold-compiles in ~2600 s; ``run_amip`` and the
``training`` package do NOT go through ``bootstrap()``/``configure_backend()``,
so they call ``enable_persistent_compile_cache()`` directly to get the
cross-process XLA cache.  These tests pin (a) the public wrapper actually turns
the cache on and is idempotent, and (b) the two drivers keep calling it (a
cheap regression guard so the wiring is not silently dropped).
"""

import importlib
import inspect

import jax
import pytest

from legoesm.runtime import backend as bk


def test_enable_persistent_compile_cache_sets_dir(tmp_path, monkeypatch):
    """Calling the wrapper sets jax_compilation_cache_dir from the env."""
    target = tmp_path / "jit_cache"
    monkeypatch.setenv("LEGOESM_JIT_CACHE_DIR", str(target))
    # This test runs on CPU, where the cache is opt-in (the CPU gate below).
    monkeypatch.setenv(bk.CPU_COMPILE_CACHE_OPT_IN_ENV, "1")

    prev_dir = jax.config.jax_compilation_cache_dir
    prev_done = getattr(bk._configure_persistent_jit_cache, "_done", False)
    try:
        # Force a fresh configuration despite the process-global idempotency.
        bk._configure_persistent_jit_cache._done = False
        bk.enable_persistent_compile_cache()
        assert jax.config.jax_compilation_cache_dir == str(target)
        assert target.is_dir()
        # Second call is a no-op (idempotent), must not raise.
        bk.enable_persistent_compile_cache()
    finally:
        bk._configure_persistent_jit_cache._done = prev_done
        if prev_dir is not None:
            jax.config.update("jax_compilation_cache_dir", prev_dir)


@pytest.mark.parametrize("backend, enabled", [("cpu", False), ("gpu", True)])
def test_cpu_backend_cache_is_opt_in(tmp_path, monkeypatch, backend, enabled):
    """The persistent cache key is not CPU-microarch-specific: on the CPU
    backend the cache stays OFF unless LEGOESM_ALLOW_CPU_COMPILE_CACHE=1 (a
    coupled test segfaulted on a cross-node AOT load and passed with the
    cache off); a GPU backend is unaffected."""
    target = tmp_path / "jit_cache"
    monkeypatch.setenv("LEGOESM_JIT_CACHE_DIR", str(target))
    monkeypatch.delenv(bk.CPU_COMPILE_CACHE_OPT_IN_ENV, raising=False)
    prev_dir = jax.config.jax_compilation_cache_dir
    prev_secs = jax.config.jax_persistent_cache_min_compile_time_secs
    prev_done = getattr(bk._configure_persistent_jit_cache, "_done", False)
    try:
        bk._configure_persistent_jit_cache._done = False
        bk._configure_persistent_jit_cache(backend)
        assert (jax.config.jax_compilation_cache_dir == str(target)) is enabled
        assert target.is_dir() is enabled
    finally:
        # restore exactly (None included): no cache dir leaks to later tests
        bk._configure_persistent_jit_cache._done = prev_done
        jax.config.update("jax_compilation_cache_dir", prev_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", prev_secs)


def test_disable_via_empty_env(monkeypatch):
    """LEGOESM_JIT_CACHE_DIR="" opts out without raising (escape hatch)."""
    monkeypatch.setenv("LEGOESM_JIT_CACHE_DIR", "")
    prev_done = getattr(bk._configure_persistent_jit_cache, "_done", False)
    try:
        bk._configure_persistent_jit_cache._done = False
        bk.enable_persistent_compile_cache()  # must be a clean no-op
    finally:
        bk._configure_persistent_jit_cache._done = prev_done


@pytest.mark.parametrize(
    "module_name",
    [
        "legoesm.training.training_driver",
        None,  # run_amip: loaded from source path below
    ],
)
def test_drivers_wire_the_cache(module_name):
    """run_amip.main and the shared training loop both call the wrapper.

    Source-level guard: removing the call (re-introducing the ~2600 s
    re-compile on every launch) fails here.
    """
    if module_name is None:
        import pathlib
        import legoesm  # noqa: F401 — locate the repo
        # run_amip lives under scripts/, not importable as a package; read it.
        repo = pathlib.Path(__file__).resolve().parents[2]
        src = (repo / "scripts" / "run" / "run_amip.py").read_text()
    else:
        mod = importlib.import_module(module_name)
        src = inspect.getsource(mod)
    assert "enable_persistent_compile_cache()" in src

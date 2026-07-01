"""Tests for MPI runtime version-guard behaviour.

These tests exercise :func:`legoesm.parallel.reductions._validate_mpi_runtime_versions`
directly — no MPI stack required.
"""

import warnings

import pytest
from legoesm.parallel import reductions
from legoesm.parallel.reductions import (
    _TESTED_JAX_MAX_EXCL,
    _TESTED_JAX_MIN,
    _TESTED_MPI4JAX_MAX_EXCL,
    _TESTED_MPI4JAX_MIN,
    _format_range,
    _mpi4jax_transport_action,
    _parse_version_triplet,
    _validate_mpi_runtime_versions,
    check_mpi4jax_transport,
)

# -----------------------------------------------------------------------
# _parse_version_triplet
# -----------------------------------------------------------------------

class TestParseVersion:
    def test_simple(self):
        assert _parse_version_triplet("0.8.1") == (0, 8, 1)

    def test_major_minor_only(self):
        assert _parse_version_triplet("0.9") == (0, 9, 0)

    def test_four_part(self):
        assert _parse_version_triplet("0.9.0.1") == (0, 9, 0)

    def test_empty(self):
        assert _parse_version_triplet("") == (0, 0, 0)

    def test_dev_suffix(self):
        assert _parse_version_triplet("0.8.0.dev123") == (0, 8, 0)


# -----------------------------------------------------------------------
# _format_range
# -----------------------------------------------------------------------

class TestFormatRange:
    def test_basic(self):
        s = _format_range((0, 8, 0), (0, 9, 0))
        assert s == ">=0.8.0, <0.9.0"


# -----------------------------------------------------------------------
# _validate_mpi_runtime_versions
# -----------------------------------------------------------------------

class TestValidate:
    """Test the three outcomes: pass, warn, hard-fail."""

    def test_within_tested_range_passes_silently(self):
        """Versions inside the tested range produce no warning or error."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → exception
            _validate_mpi_runtime_versions("0.8.5", "0.8.0")

    def test_old_mpi4jax_hard_fails(self):
        """mpi4jax < 0.8 must raise RuntimeError regardless of strict."""
        with pytest.raises(RuntimeError, match="incompatible token semantics"):
            _validate_mpi_runtime_versions("0.8.0", "0.7.9")

    def test_old_mpi4jax_error_contains_fix(self):
        """The hard-error message must contain a remediation command."""
        with pytest.raises(RuntimeError, match="pip install"):
            _validate_mpi_runtime_versions("0.8.0", "0.7.0")

    def test_jax_outside_range_warns(self):
        """JAX outside tested range produces a RuntimeWarning."""
        with pytest.warns(RuntimeWarning, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.7.0", "0.8.0")

    def test_mpi4jax_above_range_warns(self):
        """mpi4jax above tested range produces a RuntimeWarning."""
        with pytest.warns(RuntimeWarning, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.8.0", "0.10.0")

    def test_warning_contains_fix(self):
        """Warning message must point to remediation."""
        with pytest.warns(RuntimeWarning, match="LEGOESM_MPI_STRICT_COMPAT"):
            _validate_mpi_runtime_versions("0.7.0", "0.8.0")

    def test_strict_mode_raises(self):
        """strict=True upgrades the warning to RuntimeError."""
        with pytest.raises(RuntimeError, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.7.0", "0.8.0", strict=True)

    def test_strict_with_mpi4jax_above(self):
        """strict=True with mpi4jax above range also raises."""
        with pytest.raises(RuntimeError, match="mpi4jax==0.10.0"):
            _validate_mpi_runtime_versions("0.8.0", "0.10.0", strict=True)


# -----------------------------------------------------------------------
# Consistency: constants match pyproject.toml
# -----------------------------------------------------------------------

class TestConstantsConsistency:
    """Ensure runtime guard constants stay aligned with packaging metadata.

    If someone changes the pyproject.toml MPI extra version range,
    these tests will fail — reminding them to also update the runtime
    guards (and vice-versa).
    """

    def test_mpi4jax_min(self):
        assert _TESTED_MPI4JAX_MIN == (0, 8, 0), (
            "_TESTED_MPI4JAX_MIN must match pyproject.toml mpi4jax>=0.8"
        )

    def test_mpi4jax_max_excl(self):
        assert _TESTED_MPI4JAX_MAX_EXCL == (0, 10, 0), (
            "_TESTED_MPI4JAX_MAX_EXCL must match pyproject.toml mpi4jax<0.10 "
            "(mpi4jax 0.9.0 = the FFI rewrite, issue #567)"
        )

    def test_jax_tested_range(self):
        assert _TESTED_JAX_MIN == (0, 8, 0)
        assert _TESTED_JAX_MAX_EXCL == (0, 10, 0)


# -----------------------------------------------------------------------
# mpi4jax GPU transport (device-direct vs silent host-staging)
# -----------------------------------------------------------------------

class _FakeMpi4jax:
    """Stand-in exposing only ``has_cuda_support`` (raise if ``cuda is _RAISE``)."""

    _RAISE = object()

    def __init__(self, cuda):
        self._cuda = cuda

    def has_cuda_support(self):
        if self._cuda is self._RAISE:
            raise RuntimeError("cannot introspect build")
        return self._cuda


class TestTransportAction:
    """The pure decision policy: 'ok' | 'warn' | 'raise'."""

    def test_cpu_backend_is_always_ok(self):
        # CPU/TPU: the device-direct toggle is irrelevant, never warn or raise.
        for use in (True, False):
            for built in (True, False, None):
                assert _mpi4jax_transport_action("cpu", use, built) == "ok"

    def test_gpu_direct_without_cuda_extension_raises(self):
        # Asked for GPU-direct, but the library can't do it -> would segfault.
        for backend in ("gpu", "cuda", "rocm"):
            assert _mpi4jax_transport_action(backend, True, False) == "raise"

    def test_gpu_cuda_capable_but_toggle_unset_warns(self):
        # The user's bug: silent host-staging that caps multi-GPU scaling.
        for backend in ("gpu", "cuda", "rocm"):
            assert _mpi4jax_transport_action(backend, False, True) == "warn"

    def test_gpu_direct_correctly_configured_is_ok(self):
        assert _mpi4jax_transport_action("gpu", True, True) == "ok"

    def test_gpu_no_extension_and_not_requested_is_ok(self):
        # No CUDA extension AND not requested: nothing actionable to say.
        assert _mpi4jax_transport_action("gpu", False, False) == "ok"

    def test_unintrospectable_build_with_request_raises(self):
        # GPU-direct requested but support cannot be proven (None) -> fail closed.
        for backend in ("gpu", "cuda", "rocm"):
            assert _mpi4jax_transport_action(backend, True, None) == "raise"

    def test_unintrospectable_build_without_request_is_ok(self):
        # Host-staged (toggle unset) and support unprovable -> nothing to say.
        assert _mpi4jax_transport_action("gpu", False, None) == "ok"


class TestCheckTransport:
    """The stateful wrapper: raises on misconfig, warns once, latches cleanly."""

    @pytest.fixture(autouse=True)
    def _reset_and_force_gpu(self, monkeypatch):
        # Pretend we are on a GPU backend and reset the once-per-process warn latch.
        monkeypatch.setattr(reductions.jax, "default_backend", lambda: "gpu")
        monkeypatch.setattr(reductions, "_mpi4jax_transport_warned", False)

    def test_cpu_backend_silent(self, monkeypatch):
        monkeypatch.setattr(reductions.jax, "default_backend", lambda: "cpu")
        monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_mpi4jax_transport(_FakeMpi4jax(cuda=False))  # no raise, no warn

    def test_requested_but_no_extension_raises(self, monkeypatch):
        monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
        with pytest.raises(ImportError, match="usable CUDA support"):
            check_mpi4jax_transport(_FakeMpi4jax(cuda=False))

    def test_raise_is_not_latched(self, monkeypatch):
        # An impossible config must keep failing, not be swallowed after once.
        monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
        fake = _FakeMpi4jax(cuda=False)
        with pytest.raises(ImportError):
            check_mpi4jax_transport(fake)
        with pytest.raises(ImportError):
            check_mpi4jax_transport(fake)

    def test_host_staging_warns_once(self, monkeypatch):
        monkeypatch.delenv("MPI4JAX_USE_CUDA_MPI", raising=False)
        fake = _FakeMpi4jax(cuda=True)
        with pytest.warns(RuntimeWarning, match="caps multi-GPU scaling"):
            check_mpi4jax_transport(fake)
        # Second call latched -> no further warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_mpi4jax_transport(fake)

    def test_correct_gpu_direct_is_silent(self, monkeypatch):
        monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_mpi4jax_transport(_FakeMpi4jax(cuda=True))

    def test_unintrospectable_build_with_request_raises(self, monkeypatch):
        # has_cuda_support() raising + GPU-direct requested -> fail closed,
        # do NOT swallow the misconfig (Codex [high]).
        monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
        with pytest.raises(ImportError, match="usable CUDA support"):
            check_mpi4jax_transport(_FakeMpi4jax(cuda=_FakeMpi4jax._RAISE))

    def test_unintrospectable_build_without_request_is_silent(self, monkeypatch):
        # has_cuda_support() raising but host-staged (toggle unset) -> no segfault
        # risk, so stay silent rather than nag with an unprovable suggestion.
        monkeypatch.delenv("MPI4JAX_USE_CUDA_MPI", raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check_mpi4jax_transport(_FakeMpi4jax(cuda=_FakeMpi4jax._RAISE))

    def test_latch_does_not_mask_later_raise(self, monkeypatch):
        # Codex [medium] regression: a safe/warn first call must NOT latch away a
        # genuine GPU-direct misconfig that appears on a later call.
        monkeypatch.delenv("MPI4JAX_USE_CUDA_MPI", raising=False)
        with pytest.warns(RuntimeWarning, match="caps multi-GPU scaling"):
            check_mpi4jax_transport(_FakeMpi4jax(cuda=True))  # host-staged warn
        # Now GPU-direct is enabled against a no-CUDA build -> must still raise.
        monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
        with pytest.raises(ImportError, match="usable CUDA support"):
            check_mpi4jax_transport(_FakeMpi4jax(cuda=False))


class TestSendrecvChokePointEnforcesPreflight:
    """The halo sendrecv choke point must run the transport preflight on every
    call, so a GPU-direct misconfig fails closed even on the many halo paths that
    reach mpi4jax by a direct ``import mpi4jax`` rather than ``require_mpi_stack``
    (Codex pass-2 [high]: paths bypassing the preflight).
    """

    def test_get_sendrecv_vjp_fails_closed_on_gpu_direct_misconfig(self, monkeypatch):
        from legoesm.parallel.halo_exchange import get_sendrecv_vjp

        monkeypatch.setattr(reductions.jax, "default_backend", lambda: "gpu")
        monkeypatch.setattr(reductions, "_mpi4jax_transport_warned", False)
        monkeypatch.setenv("MPI4JAX_USE_CUDA_MPI", "1")
        # mpi4jax with no CUDA extension + GPU-direct requested: the choke point
        # must raise HERE (before any sendrecv touches a device buffer), not let
        # the exchange segfault.
        with pytest.raises(ImportError, match="usable CUDA support"):
            get_sendrecv_vjp(_FakeMpi4jax(cuda=False))

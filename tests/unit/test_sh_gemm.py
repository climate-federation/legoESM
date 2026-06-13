"""Batched-GEMM spectral Legendre path (scaling review 2026-06-13 lever #2).

``LEGOESM_SH_GEMM=1`` swaps the 3-D SH analysis/synthesis Legendre step from
the memory-bound broadcast-multiply + jnp.sum/segment_sum to a batched
``dot_general`` over the zonal wavenumber m (the tensor-core GEMM form; the
per-device win on Ampere+ fp64).  It must be NUMERICALLY EQUIVALENT to the
default path (same products + reduction, only the summation grouping changes)
and AD-safe.

Pins: the (m,n,lat) layout transform is a strided re-index of the flat SH
matrix; analysis + synthesis GEMM match the default to fp round-off; the
truncation round-trip holds; jax.grad flows.  Correctness is validated here
on CPU (the Ginsburg default backend); the SPEEDUP is Ampere-only.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis_3d,
    sh_synthesis_3d,
    _flat_to_bym,
    _sh_idx,
    _sh_gemm_enabled,
)


def _grid(n_max=12):
    return create_gaussian_grid(n_max)


def _rand_field(grid, nlev=4, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal((grid.n_lat, grid.n_lon, nlev)))


def _rand_coeffs(grid, nlev=4, seed=1):
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((grid.n_sh, nlev))
    im = rng.standard_normal((grid.n_sh, nlev))
    return jnp.asarray(re + 1j * im)


def test_flat_to_bym_strided_equivalence():
    grid = _grid()
    bym = _flat_to_bym(grid.Pnm, grid.ms, grid.ls, grid.n_max)
    assert bym.shape == (grid.n_max + 1, grid.n_max + 1, grid.n_lat)
    Pnm = np.asarray(grid.Pnm)
    bym_np = np.asarray(bym)
    for n in range(grid.n_max + 1):
        for m in range(n + 1):
            k = _sh_idx(n, m)
            assert np.array_equal(bym_np[m, n, :], Pnm[:, k]), (n, m)
    # zero where n < m (the unused lower triangle)
    for m in range(grid.n_max + 1):
        for n in range(m):
            assert np.all(bym_np[m, n, :] == 0.0), (n, m)


def test_analysis_gemm_matches_default(monkeypatch):
    grid = _grid()
    field = _rand_field(grid)
    monkeypatch.delenv("LEGOESM_SH_GEMM", raising=False)
    c_def = sh_analysis_3d(grid, field)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    assert _sh_gemm_enabled()
    c_gemm = sh_analysis_3d(grid, field)
    err = float(jnp.max(jnp.abs(c_gemm - c_def)))
    scale = float(jnp.max(jnp.abs(c_def)))
    assert err <= 1e-11 * max(scale, 1.0), f"analysis parity {err} (scale {scale})"


def test_synthesis_gemm_matches_default(monkeypatch):
    grid = _grid()
    coeffs = _rand_coeffs(grid)
    monkeypatch.delenv("LEGOESM_SH_GEMM", raising=False)
    f_def = sh_synthesis_3d(grid, coeffs)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    f_gemm = sh_synthesis_3d(grid, coeffs)
    err = float(jnp.max(jnp.abs(f_gemm - f_def)))
    scale = float(jnp.max(jnp.abs(f_def)))
    assert err <= 1e-11 * max(scale, 1.0), f"synthesis parity {err} (scale {scale})"


def test_gemm_round_trip(monkeypatch):
    """analysis∘synthesis is the spectral-truncation projector (idempotent on
    band-limited coeffs); the GEMM path reproduces it, matching the default
    round-trip to fp round-off."""
    grid = _grid()
    coeffs = _rand_coeffs(grid)
    monkeypatch.delenv("LEGOESM_SH_GEMM", raising=False)
    rt_def = sh_analysis_3d(grid, sh_synthesis_3d(grid, coeffs))
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    rt_gemm = sh_analysis_3d(grid, sh_synthesis_3d(grid, coeffs))
    err = float(jnp.max(jnp.abs(rt_gemm - rt_def)))
    scale = float(jnp.max(jnp.abs(rt_def)))
    assert err <= 1e-10 * max(scale, 1.0), f"round-trip parity {err}"


def test_gemm_analysis_grad_finite(monkeypatch):
    grid = _grid()
    field = _rand_field(grid)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")

    def loss(x):
        c = sh_analysis_3d(grid, x)
        return jnp.sum(jnp.abs(c) ** 2)

    g = jax.grad(loss)(field)
    assert bool(jnp.all(jnp.isfinite(g))), "GEMM analysis grad non-finite"
    assert g.shape == field.shape


def test_gemm_synthesis_grad_matches_default(monkeypatch):
    """grad through the synthesis scatter+GEMM (w.r.t. complex coeffs) is
    finite and MATCHES the default segment_sum path — exercises the
    coeffs->(m,n) scatter VJP, not just analysis (codex 2026-06-13)."""
    grid = _grid()
    coeffs = _rand_coeffs(grid)

    def loss(c):
        return jnp.sum(sh_synthesis_3d(grid, c) ** 2)

    monkeypatch.delenv("LEGOESM_SH_GEMM", raising=False)
    g_def = jax.grad(loss, holomorphic=False)(coeffs)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    g_gemm = jax.grad(loss, holomorphic=False)(coeffs)
    assert bool(jnp.all(jnp.isfinite(g_gemm))), "GEMM synthesis grad non-finite"
    err = float(jnp.max(jnp.abs(g_gemm - g_def)))
    scale = float(jnp.max(jnp.abs(g_def)))
    assert err <= 1e-10 * max(scale, 1.0), f"synthesis grad parity {err}"


def test_sh_gemm_env_falsey(monkeypatch):
    """Falsey env values must NOT enable the opt-in path (case-insensitive)."""
    for val in ("0", "false", "False", "FALSE", "no", "off", ""):
        monkeypatch.setenv("LEGOESM_SH_GEMM", val)
        assert _sh_gemm_enabled() is False, f"{val!r} wrongly enabled SH-GEMM"
    for val in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("LEGOESM_SH_GEMM", val)
        assert _sh_gemm_enabled() is True, f"{val!r} should enable SH-GEMM"

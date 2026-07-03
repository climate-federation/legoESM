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
    sh_synthesis_H_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    sh_analysis_oc2_dmu_3d,
    uv_from_vordiv_3d,
    vordiv_from_uv_3d,
    _flat_to_bym,
    _sh_idx,
    _sh_gemm_enabled,
    _sh_gemm_bf16_enabled,
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


# ----------------------------------------------------------------------
# A3: variant coverage — H-synthesis, oc2/dmu analysis, the fused
# oc2+dmu, and the vector composites must ALL match the legacy path to
# ~1e-12 (f64) with LEGOESM_SH_GEMM=1, and flow finite, matching grads.
# ----------------------------------------------------------------------

def _parity_off_on(monkeypatch, fn):
    """Return (default_result, gemm_result) for a zero-arg ``fn``."""
    monkeypatch.delenv("LEGOESM_SH_GEMM", raising=False)
    res_def = fn()
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    res_gemm = fn()
    return res_def, res_gemm


def _assert_close(a, b, tol=1e-11, label=""):
    err = float(jnp.max(jnp.abs(a - b)))
    scale = float(jnp.max(jnp.abs(b)))
    assert err <= tol * max(scale, 1.0), f"{label} parity {err} (scale {scale})"


def test_synthesis_H_gemm_matches_default(monkeypatch):
    grid = _grid()
    coeffs = _rand_coeffs(grid)
    f_def, f_gemm = _parity_off_on(
        monkeypatch, lambda: sh_synthesis_H_3d(grid, coeffs))
    _assert_close(f_gemm, f_def, label="H-synthesis")


def test_analysis_oc2_gemm_matches_default(monkeypatch):
    grid = _grid()
    field = _rand_field(grid)
    c_def, c_gemm = _parity_off_on(
        monkeypatch, lambda: sh_analysis_oc2_3d(grid, field))
    _assert_close(c_gemm, c_def, label="oc2-analysis")


def test_analysis_dmu_gemm_matches_default(monkeypatch):
    grid = _grid()
    field = _rand_field(grid)
    c_def, c_gemm = _parity_off_on(
        monkeypatch, lambda: sh_analysis_dmu_3d(grid, field))
    _assert_close(c_gemm, c_def, label="dmu-analysis")


def test_analysis_oc2_dmu_fused_gemm_matches_default(monkeypatch):
    """The fused GEMM path must (a) match the legacy fused output AND (b)
    equal the two standalone GEMM analyses — catches a shared-``f_m`` /
    wrong-matrix regression in the fused kernel."""
    grid = _grid()
    field = _rand_field(grid)
    (oc2_def, dmu_def), (oc2_g, dmu_g) = _parity_off_on(
        monkeypatch, lambda: sh_analysis_oc2_dmu_3d(grid, field))
    _assert_close(oc2_g, oc2_def, label="fused-oc2")
    _assert_close(dmu_g, dmu_def, label="fused-dmu")
    # With GEMM on, the fused outputs equal the standalone GEMM variants.
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    _assert_close(oc2_g, sh_analysis_oc2_3d(grid, field), label="fused==oc2_3d")
    _assert_close(dmu_g, sh_analysis_dmu_3d(grid, field), label="fused==dmu_3d")


def test_uv_from_vordiv_3d_gemm_matches_default(monkeypatch):
    """Vector synthesis (composes Pnm- and Hnm-synthesis) parity — the
    integration check that pole-safe 1/cos² weighting is unchanged."""
    grid = _grid()
    vor = _rand_coeffs(grid, seed=2)
    div = _rand_coeffs(grid, seed=3)
    (u_def, v_def), (u_g, v_g) = _parity_off_on(
        monkeypatch, lambda: uv_from_vordiv_3d(grid, vor, div))
    _assert_close(u_g, u_def, label="uv-from-vordiv u")
    _assert_close(v_g, v_def, label="uv-from-vordiv v")


def test_vordiv_from_uv_3d_gemm_matches_default(monkeypatch):
    """Vector analysis (composes oc2- and dmu-analysis) parity."""
    grid = _grid()
    u = _rand_field(grid, seed=4)
    v = _rand_field(grid, seed=5)
    (vor_def, div_def), (vor_g, div_g) = _parity_off_on(
        monkeypatch, lambda: vordiv_from_uv_3d(grid, u, v))
    _assert_close(vor_g, vor_def, label="vordiv-from-uv vor")
    _assert_close(div_g, div_def, label="vordiv-from-uv div")


def test_synthesis_H_grad_matches_default(monkeypatch):
    """grad through the Hnm scatter+GEMM (w.r.t. complex coeffs) is finite
    and matches the default segment_sum path."""
    grid = _grid()
    coeffs = _rand_coeffs(grid)

    def loss(c):
        return jnp.sum(sh_synthesis_H_3d(grid, c) ** 2)

    monkeypatch.delenv("LEGOESM_SH_GEMM", raising=False)
    g_def = jax.grad(loss, holomorphic=False)(coeffs)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    g_gemm = jax.grad(loss, holomorphic=False)(coeffs)
    assert bool(jnp.all(jnp.isfinite(g_gemm))), "H-synthesis grad non-finite"
    _assert_close(g_gemm, g_def, tol=1e-10, label="H-synthesis grad")


def test_analysis_dmu_grad_matches_default(monkeypatch):
    """grad through the weighted (wDnm) analysis GEMM is finite and matches
    the default latitude-sum path."""
    grid = _grid()
    field = _rand_field(grid)

    def loss(x):
        return jnp.sum(jnp.abs(sh_analysis_dmu_3d(grid, x)) ** 2)

    monkeypatch.delenv("LEGOESM_SH_GEMM", raising=False)
    g_def = jax.grad(loss)(field)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    g_gemm = jax.grad(loss)(field)
    assert bool(jnp.all(jnp.isfinite(g_gemm))), "dmu-analysis grad non-finite"
    _assert_close(g_gemm, g_def, tol=1e-10, label="dmu-analysis grad")


# ----------------------------------------------------------------------
# A3 bf16: the Legendre contraction in bfloat16 (fp32 accumulate) — the
# MXU/tensor-core reduced-precision throughput path. A BOUND vs f64, not
# equality; default OFF must be byte-identical; forward/inference-only.
# ----------------------------------------------------------------------

def test_sh_gemm_bf16_env_parse(monkeypatch):
    for val in ("0", "false", "False", "no", "off", ""):
        monkeypatch.setenv("LEGOESM_SH_GEMM_BF16", val)
        assert _sh_gemm_bf16_enabled() is False, f"{val!r} wrongly enabled bf16"
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv("LEGOESM_SH_GEMM_BF16", val)
        assert _sh_gemm_bf16_enabled() is True, f"{val!r} should enable bf16"


def test_sh_gemm_bf16_default_off_is_byte_identical(monkeypatch):
    """bf16 OFF (default) leaves the f64 GEMM path byte-for-byte unchanged."""
    grid = _grid()
    coeffs = _rand_coeffs(grid)
    field = _rand_field(grid)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    monkeypatch.delenv("LEGOESM_SH_GEMM_BF16", raising=False)
    s_a, a_a = sh_synthesis_3d(grid, coeffs), sh_analysis_3d(grid, field)
    monkeypatch.setenv("LEGOESM_SH_GEMM_BF16", "0")
    s_b, a_b = sh_synthesis_3d(grid, coeffs), sh_analysis_3d(grid, field)
    assert float(jnp.max(jnp.abs(s_a - s_b))) == 0.0, "bf16=0 changed synthesis"
    assert float(jnp.max(jnp.abs(a_a - a_b))) == 0.0, "bf16=0 changed analysis"


def test_sh_gemm_bf16_relative_accuracy(monkeypatch):
    """bf16 matches the f64 GEMM to a BOUND (~1e-2 rel), and is genuinely
    lossy (not silently exact). A sanity bound that catches a gross error
    (wrong matrix, broken complex split) — NOT a parity claim."""
    grid = _grid()
    field = _rand_field(grid)
    coeffs = _rand_coeffs(grid)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    monkeypatch.delenv("LEGOESM_SH_GEMM_BF16", raising=False)
    a_f64 = sh_analysis_3d(grid, field)
    s_f64 = sh_synthesis_3d(grid, coeffs)
    monkeypatch.setenv("LEGOESM_SH_GEMM_BF16", "1")
    a_bf16 = sh_analysis_3d(grid, field)
    s_bf16 = sh_synthesis_3d(grid, coeffs)
    for got, ref, lbl in ((a_bf16, a_f64, "analysis"), (s_bf16, s_f64, "synthesis")):
        err = float(jnp.max(jnp.abs(got - ref)))
        scale = float(jnp.max(jnp.abs(ref)))
        assert err <= 5e-2 * max(scale, 1.0), f"bf16 {lbl} {err} (scale {scale})"
        assert err > 0.0, f"bf16 {lbl} suspiciously exact — bf16 not engaged"
        assert got.dtype == ref.dtype, f"bf16 {lbl} changed output dtype"


def test_sh_gemm_bf16_grad_finite(monkeypatch):
    """grad through the bf16 contraction is finite. bf16 is NOT for training
    (lossy gradients), but differentiating it must not produce NaN/Inf."""
    grid = _grid()
    field = _rand_field(grid)
    monkeypatch.setenv("LEGOESM_SH_GEMM", "1")
    monkeypatch.setenv("LEGOESM_SH_GEMM_BF16", "1")

    def loss(x):
        return jnp.sum(jnp.abs(sh_analysis_3d(grid, x)) ** 2)

    g = jax.grad(loss)(field)
    assert bool(jnp.all(jnp.isfinite(g))), "bf16 analysis grad non-finite"

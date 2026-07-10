"""Equivalence tests for the turbulence dedup refactors (2026-07-07).

Three copy-pasted numerics blocks were factored into single shared homes with
NO intended behavior change; each test here pins the new shared implementation
against an inline reference copy of the OLD numerics (helper extraction can
shift XLA fusion by ~1e-12, so tolerances are rtol 1e-10 under x64):

1. ``physics._shared.louis_stability_functions`` — the Louis (1979/1982)
   f_stable/f_unstable + sigmoid Ri blend previously verbatim in louis.py and
   ysu.py.
2. ``turbulence.pbl_height.first_crossing_height`` — the differentiable
   bulk-Ri first-crossing PBL-height kernel previously triplicated in
   ysu.py / holtslag_boville.py.
3. ``turbulence.vertical_diffusion.implicit_vertical_diffusion`` — now
   assembling bands for the SHARED ``timestepping.tridiagonal
   .thomas_solve_batched`` instead of hand-rolled lax.scan sweeps; verified
   against a dense NumPy solve of the same backward-Euler flux-form system.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics._shared import louis_stability_functions
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    first_crossing_height,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)


# ---------------------------------------------------------------------------
# 1. Louis stability functions
# ---------------------------------------------------------------------------

def _louis_reference(Ri, l_mix, dz, b, c, d, blend_sharpness, b_heat):
    """Inline copy of the OLD louis.py stability-function block (pre-dedup)."""
    Ri_neg = jnp.minimum(Ri, 0.0)
    denom_unstable = (
        1.0 + 3.0 * b * c * l_mix ** 2
        * jnp.sqrt(jnp.abs(Ri_neg) + 1e-10) / (dz ** 2 + 1e-10)
    )
    f_unstable_m = 1.0 - 2.0 * b * Ri_neg / denom_unstable
    f_unstable_h = 1.0 - 2.0 * b_heat * Ri_neg / denom_unstable
    Ri_pos = jnp.maximum(Ri, 0.0)
    sqrt_stable = jnp.sqrt(1.0 + d * Ri_pos)
    f_stable_m = 1.0 / (1.0 + 2.0 * b * Ri_pos / sqrt_stable)
    f_stable_h = 1.0 / (1.0 + 2.0 * b_heat * Ri_pos / sqrt_stable)
    blend = jax.nn.sigmoid(blend_sharpness * Ri)
    f_m = (1.0 - blend) * f_unstable_m + blend * f_stable_m
    f_h = (1.0 - blend) * f_unstable_h + blend * f_stable_h
    return f_m, f_h


def _stability_inputs():
    rng = np.random.default_rng(7)
    ncol, nhalf = 4, 18
    # Mixed-sign Ri spanning strongly unstable to strongly stable + values
    # straddling 0 (the sigmoid blend region).
    Ri = jnp.asarray(rng.uniform(-5.0, 5.0, (ncol, nhalf)))
    Ri = Ri.at[:, 0].set(0.0)
    l_mix = jnp.asarray(rng.uniform(5.0, 100.0, (ncol, nhalf)))
    dz = jnp.asarray(rng.uniform(50.0, 800.0, (ncol, nhalf)))
    return Ri, l_mix, dz


def test_louis_stability_matches_old_louis_inline():
    """Shared helper == the old louis.py inline block (momentum AND heat,
    with the LTG82 b_heat = 1.5*b split)."""
    Ri, l_mix, dz = _stability_inputs()
    b, c, d, sharp = 5.0, 5.0, 5.0, 100.0
    b_heat = 1.5 * b
    f_m, f_h = louis_stability_functions(
        Ri, l_mix, dz, b, c, d, sharp, b_heat=b_heat)
    f_m_ref, f_h_ref = _louis_reference(Ri, l_mix, dz, b, c, d, sharp, b_heat)
    np.testing.assert_allclose(np.asarray(f_m), np.asarray(f_m_ref), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(f_h), np.asarray(f_h_ref), rtol=1e-10)


def test_louis_stability_matches_old_ysu_inline():
    """Shared helper (default b_heat=b => f_h == f_m) == the old ysu.py inline
    momentum-only block (which used the single Louis-1979 function)."""
    Ri, l_mix, dz = _stability_inputs()
    b, c, d, sharp = 5.0, 5.0, 5.0, 100.0
    f_m, f_h = louis_stability_functions(Ri, l_mix, dz, b, c, d, sharp)
    # Old YSU inline expression (verbatim pre-dedup):
    Ri_pos = jnp.maximum(Ri, 0.0)
    f_stable = 1.0 / (1.0 + 2.0 * b * Ri_pos / jnp.sqrt(1.0 + d * Ri_pos))
    Ri_neg = jnp.minimum(Ri, 0.0)
    f_unstable = 1.0 - 2.0 * b * Ri_neg / (
        1.0 + 3.0 * b * c * l_mix ** 2
        * jnp.sqrt(jnp.abs(Ri_neg) + 1e-10) / (dz ** 2 + 1e-10)
    )
    blend_ri = jax.nn.sigmoid(sharp * Ri)
    f_m_ref = (1.0 - blend_ri) * f_unstable + blend_ri * f_stable
    np.testing.assert_allclose(np.asarray(f_m), np.asarray(f_m_ref), rtol=1e-10)
    # b_heat defaulted to b: heat function must equal momentum function.
    np.testing.assert_allclose(np.asarray(f_h), np.asarray(f_m), rtol=1e-12)


def test_louis_stability_neutral_limit_and_grad():
    """Ri = 0 -> f_m = f_h = 1 exactly (up to the 0.5 blend of two branches
    that both equal 1); gradient finite through the helper."""
    ncol, nhalf = 2, 6
    Ri = jnp.zeros((ncol, nhalf))
    l_mix = jnp.full((ncol, nhalf), 30.0)
    dz = jnp.full((ncol, nhalf), 200.0)
    f_m, f_h = louis_stability_functions(Ri, l_mix, dz, 5.0, 5.0, 5.0, 100.0)
    np.testing.assert_allclose(np.asarray(f_m), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f_h), 1.0, rtol=1e-12)

    def loss(Ri_in):
        fm, fh = louis_stability_functions(
            Ri_in, l_mix, dz, 5.0, 5.0, 5.0, 100.0, b_heat=7.5)
        return jnp.sum(fm ** 2 + fh ** 2)

    g = jax.grad(loss)(jnp.linspace(-1.0, 1.0, ncol * nhalf).reshape(ncol, nhalf))
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# 2. First-crossing PBL-height kernel
# ---------------------------------------------------------------------------

def _old_ysu_crossing(Ri_b, z_full, ricr, sharpness):
    """Inline copy of the OLD ysu.py ``_crossing_pbl_height`` (pre-dedup),
    WITHOUT the trailing 100 m clip (that clip stayed at the YSU call site)."""
    ncol, nlev = Ri_b.shape
    Ri_s = Ri_b[:, ::-1]
    z_s = z_full[:, ::-1]
    Ri_lo = Ri_s[:, :-1]
    Ri_hi = Ri_s[:, 1:]
    z_lo = z_s[:, :-1]
    z_hi = z_s[:, 1:]
    crossed = jax.nn.sigmoid(sharpness * (Ri_hi - ricr))
    not_crossed = jnp.cumprod(
        jnp.concatenate(
            [jnp.ones((ncol, 1), Ri_b.dtype), 1.0 - crossed[:, :-1]], axis=1,
        ),
        axis=1,
    )
    w = not_crossed * crossed
    dRi = Ri_hi - Ri_lo
    frac = jnp.clip(
        (ricr - Ri_lo) / jnp.where(jnp.abs(dRi) > 1.0e-12, dRi, 1.0e-12),
        0.0, 1.0,
    )
    z_cross = z_lo + frac * (z_hi - z_lo)
    z_top = z_full[:, 0]
    fallback_w = jnp.prod(1.0 - crossed, axis=1)
    num = jnp.sum(w * z_cross, axis=1) + fallback_w * z_top
    den = jnp.sum(w, axis=1) + fallback_w + 1.0e-30
    return num / den


def _old_hb_crossing(rino, z_full, ricr, sharpness, search_ok, z_top_search):
    """Inline copy of the OLD holtslag_boville.py ``_crossing_height``."""
    ncol, nlev = rino.shape
    rino_s = rino[:, ::-1]
    z_s = z_full[:, ::-1]
    search_s = search_ok[:, ::-1]
    rino_lo = rino_s[:, :-1]
    rino_hi = rino_s[:, 1:]
    z_lo = z_s[:, :-1]
    z_hi = z_s[:, 1:]
    pair_ok = search_s[:, :-1] * search_s[:, 1:]
    crossed = jax.nn.sigmoid(sharpness * (rino_hi - ricr)) * pair_ok
    not_crossed = jnp.cumprod(
        jnp.concatenate(
            [jnp.ones((ncol, 1), rino.dtype), 1.0 - crossed[:, :-1]], axis=1,
        ),
        axis=1,
    )
    w = not_crossed * crossed
    dRi = rino_hi - rino_lo
    frac = jnp.clip(
        (ricr - rino_lo) / jnp.where(jnp.abs(dRi) > 1.0e-12, dRi, 1.0e-12),
        0.0, 1.0,
    )
    z_cross = z_lo + frac * (z_hi - z_lo)
    fallback_w = jnp.prod(1.0 - crossed, axis=1)
    num = jnp.sum(w * z_cross, axis=1) + fallback_w * z_top_search
    den = jnp.sum(w, axis=1) + fallback_w + 1.0e-30
    return num / den


def _crossing_inputs():
    rng = np.random.default_rng(11)
    ncol, nlev = 5, 16
    # Top-first heights, surface z ~ 30 m.
    z_1d = np.linspace(12000.0, 30.0, nlev)
    z_full = jnp.asarray(np.broadcast_to(z_1d[None, :], (ncol, nlev)).copy())
    # Bulk-Ri-like profile: negative near surface, increasing upward through
    # ricr, plus one column that never crosses (fallback path) and one with a
    # very sharp inversion (tight straddle).
    base = np.linspace(3.0, -0.3, nlev)[None, :]  # top-first: large aloft
    Ri = np.broadcast_to(base, (ncol, nlev)).copy()
    Ri += rng.normal(0.0, 0.05, (ncol, nlev))
    Ri[2] = -0.5  # never reaches ricr -> fallback
    Ri[3] = np.linspace(0.05, -0.2, nlev)  # crosses very low
    Ri = jnp.asarray(Ri)
    return Ri, z_full


def test_first_crossing_matches_old_ysu_kernel():
    Ri, z_full = _crossing_inputs()
    got = first_crossing_height(Ri, z_full, 0.25, 20.0)
    ref = _old_ysu_crossing(Ri, z_full, 0.25, 20.0)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-10)


def test_first_crossing_matches_old_hb_kernel():
    Ri, z_full = _crossing_inputs()
    ncol, nlev = Ri.shape
    # HB-style smooth search region (pressure-limited): allow the lower
    # two-thirds of the column, smooth edge.
    kk = jnp.arange(nlev, dtype=Ri.dtype)[None, :]
    search_ok = jnp.broadcast_to(
        jax.nn.sigmoid((kk - nlev / 3.0)), (ncol, nlev))
    z_top_search = jnp.full((ncol,), 8000.0, dtype=Ri.dtype)
    got = first_crossing_height(
        Ri, z_full, 0.25, 20.0, search_ok=search_ok, z_top=z_top_search)
    ref = _old_hb_crossing(Ri, z_full, 0.25, 20.0, search_ok, z_top_search)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-10)


def test_first_crossing_differentiable():
    Ri, z_full = _crossing_inputs()

    def loss(Ri_in):
        return jnp.sum(first_crossing_height(Ri_in, z_full, 0.25, 20.0) ** 2)

    g = jax.grad(loss)(Ri)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


# ---------------------------------------------------------------------------
# 3. Implicit vertical diffusion via the shared batched Thomas solver
# ---------------------------------------------------------------------------

def _diffusion_inputs():
    rng = np.random.default_rng(3)
    ncol, nlev = 4, 20
    phi = jnp.asarray(280.0 + rng.normal(0.0, 5.0, (ncol, nlev)))
    K_half = jnp.asarray(rng.uniform(0.1, 60.0, (ncol, nlev - 1)))
    rho = jnp.asarray(rng.uniform(0.3, 1.2, (ncol, nlev)))
    dz = jnp.asarray(rng.uniform(100.0, 900.0, (ncol, nlev)))
    dz_half = jnp.asarray(rng.uniform(100.0, 900.0, (ncol, nlev - 1)))
    sflx = jnp.asarray(rng.normal(0.0, 0.05, (ncol,)))
    dt = 600.0
    return phi, K_half, rho, dz, dz_half, dt, sflx


def _dense_reference(phi, K_half, rho, dz, dz_half, dt, sflx):
    """Dense NumPy solve of the SAME backward-Euler flux-form tridiagonal
    system (independent of any Thomas implementation, old or new)."""
    phi = np.asarray(phi, dtype=np.float64)
    K_half = np.asarray(K_half, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    dz_half = np.asarray(dz_half, dtype=np.float64)
    sflx = np.asarray(sflx, dtype=np.float64)
    ncol, nlev = phi.shape
    out = np.zeros_like(phi)
    for i in range(ncol):
        rho_half = 0.5 * (rho[i, :-1] + rho[i, 1:])
        a = np.zeros(nlev)
        c = np.zeros(nlev)
        a[1:] = dt * rho_half * K_half[i] / (rho[i, 1:] * dz[i, 1:] * dz_half[i])
        c[:-1] = dt * rho_half * K_half[i] / (rho[i, :-1] * dz[i, :-1] * dz_half[i])
        b = 1.0 + a + c
        rhs = phi[i].copy()
        rhs[-1] += dt * sflx[i] / (rho[i, -1] * dz[i, -1])
        A = np.diag(b) + np.diag(-a[1:], -1) + np.diag(-c[:-1], 1)
        out[i] = np.linalg.solve(A, rhs)
    return out


def test_implicit_diffusion_matches_dense_solve():
    """The shared-thomas_solve_batched path must solve the identical
    backward-Euler system: compare against an independent dense solve
    (rtol 1e-10, fp64)."""
    phi, K_half, rho, dz, dz_half, dt, sflx = _diffusion_inputs()
    got = np.asarray(
        implicit_vertical_diffusion(phi, K_half, rho, dz, dz_half, dt, sflx))
    ref = _dense_reference(phi, K_half, rho, dz, dz_half, dt, sflx)
    np.testing.assert_allclose(got, ref, rtol=1e-10)


def test_implicit_diffusion_zero_flux_conserves_mass_weighted_integral():
    """No-flux BCs: Σ rho·dz·phi conserved to machine precision (flux form),
    unchanged from the pre-dedup solver contract."""
    phi, K_half, rho, dz, dz_half, dt, _ = _diffusion_inputs()
    ncol = phi.shape[0]
    out = implicit_vertical_diffusion(
        phi, K_half, rho, dz, dz_half, dt, jnp.zeros((ncol,)))
    before = jnp.sum(rho * dz * phi, axis=1)
    after = jnp.sum(rho * dz * out, axis=1)
    np.testing.assert_allclose(
        np.asarray(after), np.asarray(before), rtol=1e-12)


def test_implicit_diffusion_grad_finite():
    """Reverse-mode AD flows through the shared solver path."""
    phi, K_half, rho, dz, dz_half, dt, sflx = _diffusion_inputs()

    def loss(K_in):
        out = implicit_vertical_diffusion(phi, K_in, rho, dz, dz_half, dt, sflx)
        return jnp.sum(out ** 2)

    g = jax.grad(loss)(K_half)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0

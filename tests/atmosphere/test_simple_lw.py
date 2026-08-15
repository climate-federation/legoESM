"""Idealized tests for the Stevens (2005) simple longwave.

The module's ``__physics_contract__`` names this file, and each test below is
one of the claims that contract makes: the clear-column limit, the sign of the
cloud-top cooling, and the telescoping column integral. One further test pins
the fact that extracting the kernel out of ``run_dycoms_les.py`` changed no
number, which is the only thing that makes the existing DYCOMS LES reference
still comparable to runs made after the refactor.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from legoesm.atmosphere.physics.radiation.simple_lw import (  # noqa: E402
    DYCOMS_SPEC_CP_J_KG_K,
    SimpleLWConfig,
    simple_lw_inversion_height,
    simple_lw_net_upward_flux,
    simple_lw_temperature_tendency,
)

NZ = 40
LZ = 1500.0


def _grid(nz: int = NZ, lz: float = LZ):
    """Uniform bottom-up column: z_half[0] = 0 is the surface."""
    z_half = jnp.linspace(0.0, lz, nz + 1)
    dz = jnp.diff(z_half)
    # A plausible Boussinesq-ish density profile; the scheme only needs rho>0.
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    rho = 1.2 * jnp.exp(-z_full / 8000.0)
    return z_half, dz, rho


def _cloud_column(q_c_max=5.0e-4, z_base=400.0, z_top=800.0):
    """A slab cloud between z_base and z_top, moist below the inversion."""
    z_half, dz, rho = _grid()
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    q_cond = jnp.where((z_full >= z_base) & (z_full <= z_top), q_c_max, 0.0)
    # q_t above 8 g/kg below the cloud top, dry above: puts the inversion at
    # the cloud top, which is the RF01 geometry.
    q_v = jnp.where(z_full <= z_top, 9.0e-3, 2.0e-3)
    return z_half, dz, rho, q_cond, q_v + q_cond


def test_clear_column_is_the_two_exponential_floor_and_does_not_heat():
    """No condensate and no divergence: F is uniform, so dT/dt is exactly 0."""
    z_half, dz, rho = _grid()
    cfg = SimpleLWConfig(divergence_s=0.0)
    q = jnp.zeros(NZ)
    flux = simple_lw_net_upward_flux(q, q, rho, dz, z_half, cfg)
    assert np.allclose(np.asarray(flux), cfg.f0_w_m2 + cfg.f1_w_m2, rtol=1e-12)
    tend = simple_lw_temperature_tendency(q, q, rho, dz, z_half, cfg)
    assert np.allclose(np.asarray(tend), 0.0, atol=1e-15)


def test_cloud_top_cools_and_cloud_base_warms():
    """The sign claim in the contract, on the geometry the scheme was fit to."""
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    cfg = SimpleLWConfig(divergence_s=0.0)   # isolate the cloud terms
    tend = np.asarray(
        simple_lw_temperature_tendency(q_cond, q_tot, rho, dz, z_half, cfg))
    z_full = np.asarray(0.5 * (z_half[:-1] + z_half[1:]))
    in_cloud = np.asarray(q_cond) > 0.0
    top_idx = np.max(np.flatnonzero(in_cloud))
    base_idx = np.min(np.flatnonzero(in_cloud))
    assert tend[top_idx] < 0.0, "cloud top must cool"
    assert tend[base_idx] > 0.0, "cloud base must warm"
    # and the cooling is the dominant signal, not a rounding-level wiggle
    assert tend[top_idx] < -1.0e-4, f"cooling too weak: {tend[top_idx]:.3e} K/s"
    assert z_full[top_idx] > z_full[base_idx]


def test_column_heating_telescopes_to_the_flux_difference():
    """conserves-claim: sum(rho cp dz dT/dt) == -(F_top - F_surface)."""
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    cfg = SimpleLWConfig()
    flux = simple_lw_net_upward_flux(q_cond, q_tot, rho, dz, z_half, cfg)
    tend = simple_lw_temperature_tendency(q_cond, q_tot, rho, dz, z_half, cfg)
    lhs = float(jnp.sum(rho * cfg.cp_j_kg_k * dz * tend))
    rhs = -float(flux[-1] - flux[0])
    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_inversion_is_the_face_above_the_highest_moist_cell():
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    cfg = SimpleLWConfig()
    z_i = float(simple_lw_inversion_height(q_tot, z_half, cfg)[..., 0])
    k_top = int(np.max(np.flatnonzero(np.asarray(q_tot) >= cfg.qt_inversion_kg_kg)))
    assert z_i == pytest.approx(float(z_half[k_top + 1]))


def test_dry_column_puts_the_inversion_at_the_surface():
    """gSAM initialises itop=1, i.e. zi=0, when nothing crosses the isoline."""
    z_half, dz, rho = _grid()
    cfg = SimpleLWConfig()
    q_dry = jnp.full(NZ, 1.0e-3)          # below the 8 g/kg isoline
    z_i = float(simple_lw_inversion_height(q_dry, z_half, cfg)[..., 0])
    assert z_i == pytest.approx(0.0)


def test_clear_sky_term_is_off_below_the_inversion():
    """dz_i is clipped at 0, which is how gSAM's k=itop+1 loop bound reads."""
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    on = SimpleLWConfig()
    off = SimpleLWConfig(divergence_s=0.0)
    f_on = np.asarray(simple_lw_net_upward_flux(q_cond, q_tot, rho, dz, z_half, on))
    f_off = np.asarray(simple_lw_net_upward_flux(q_cond, q_tot, rho, dz, z_half, off))
    z_i = float(simple_lw_inversion_height(q_tot, z_half, on)[..., 0])
    below = np.asarray(z_half) <= z_i
    assert np.allclose(f_on[below], f_off[below], rtol=1e-12)
    assert np.any(f_on[~below] > f_off[~below])


def test_gsam_local_density_differs_from_the_specification_form():
    """The two Faithfulness options must actually be two different schemes."""
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    spec = simple_lw_net_upward_flux(
        q_cond, q_tot, rho, dz, z_half, SimpleLWConfig(density_at_inversion=True))
    gsam = simple_lw_net_upward_flux(
        q_cond, q_tot, rho, dz, z_half, SimpleLWConfig(density_at_inversion=False))
    assert not np.allclose(np.asarray(spec), np.asarray(gsam), rtol=1e-6)


def test_spec_cp_scales_the_heating_rate():
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    a = SimpleLWConfig(divergence_s=0.0)
    b = SimpleLWConfig(divergence_s=0.0, cp_j_kg_k=DYCOMS_SPEC_CP_J_KG_K)
    ta = np.asarray(simple_lw_temperature_tendency(q_cond, q_tot, rho, dz, z_half, a))
    tb = np.asarray(simple_lw_temperature_tendency(q_cond, q_tot, rho, dz, z_half, b))
    # With divergence off, cp enters only the 1/(rho cp dz) divisor.
    nz_mask = np.abs(ta) > 0.0
    ratio = tb[nz_mask] / ta[nz_mask]
    assert np.allclose(ratio, a.cp_j_kg_k / b.cp_j_kg_k, rtol=1e-10)


def test_matches_the_inline_les_implementation_bit_for_bit():
    """The extraction must not have changed a number.

    Transcribes run_dycoms_les.py::make_stevens_lw as it stood before the
    refactor. If this drifts, the existing DYCOMS LES reference stops being
    comparable to anything produced afterwards.
    """
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    cfg = SimpleLWConfig()
    z_c = 0.5 * (z_half[:-1] + z_half[1:])
    z_f = z_half
    dz_l = dz

    # --- the original inline formula ---
    dq = cfg.kappa_m2_kg * rho * q_cond * dz_l
    Q_from_bot = jnp.cumsum(dq, axis=-1)
    Q_bot_f = jnp.pad(Q_from_bot, (1, 0))
    Q_tot = Q_bot_f[-1:]
    Q_from_top_f = Q_tot - Q_bot_f
    below = q_tot >= cfg.qt_inversion_kg_kg
    k_top = jnp.max(jnp.where(below, jnp.arange(below.shape[-1]), -1))
    z_i = z_f[k_top + 1][..., None]
    rho_i = jnp.interp(z_i[..., 0], z_c, rho)[..., None]
    dz_i = jnp.clip(z_f - z_i, 0.0, None)
    term3 = (rho_i * cfg.cp_j_kg_k * cfg.divergence_s
             * (0.25 * dz_i ** (4.0 / 3.0) + z_i * dz_i ** (1.0 / 3.0)))
    F_ref = (cfg.f0_w_m2 * jnp.exp(-Q_from_top_f)
             + cfg.f1_w_m2 * jnp.exp(-Q_bot_f) + term3)
    # ------------------------------------

    F_new = simple_lw_net_upward_flux(q_cond, q_tot, rho, dz_l, z_half, cfg)
    assert np.array_equal(np.asarray(F_ref).ravel(), np.asarray(F_new).ravel())


def test_flux_is_differentiable_in_its_coefficients():
    """The contract claims the empirical coefficients carry a gradient."""
    z_half, dz, rho, q_cond, q_tot = _cloud_column()

    def loss(f0, kappa):
        cfg = SimpleLWConfig(f0_w_m2=f0, kappa_m2_kg=kappa)
        return jnp.sum(
            simple_lw_temperature_tendency(q_cond, q_tot, rho, dz, z_half, cfg) ** 2)

    g0, gk = jax.grad(loss, argnums=(0, 1))(70.0, 85.0)
    assert np.isfinite(float(g0)) and abs(float(g0)) > 0.0
    assert np.isfinite(float(gk)) and abs(float(gk)) > 0.0


def test_batched_columns_match_the_single_column_result():
    """The kernel is written on the last axis; a batch must not mix columns."""
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    q_cond2 = jnp.stack([q_cond, 0.5 * q_cond])
    q_tot2 = jnp.stack([q_tot, q_tot])
    cfg = SimpleLWConfig()
    batched = np.asarray(
        simple_lw_net_upward_flux(q_cond2, q_tot2, rho, dz, z_half, cfg))
    single_a = np.asarray(
        simple_lw_net_upward_flux(q_cond, q_tot, rho, dz, z_half, cfg))
    single_b = np.asarray(
        simple_lw_net_upward_flux(0.5 * q_cond, q_tot, rho, dz, z_half, cfg))
    assert np.allclose(batched[0], single_a, rtol=1e-12)
    assert np.allclose(batched[1], single_b, rtol=1e-12)


# --- the gradient path the SCM actually uses --------------------------------

def _finite(x) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x))))


def test_gradient_through_the_COLUMN_GEOMETRY_is_finite():
    """`test_flux_is_differentiable_in_its_coefficients` cannot catch this.

    f0 and kappa never reach `dz_i`, so differentiating in them never touches
    the fractional power at its zero. The SCM's path does: its level heights
    move with temperature, so `z_half` is live, `dz_i = clip(z_half - z_i, 0,
    None)` is live, and `dz_i ** (1/3)` has an INFINITE derivative at the zeros
    that clip creates below the inversion. inf times the clip's zero cotangent
    is NaN.

    That NaN is what froze every parameter of all NINE closures in the
    seven-case turbulence campaign while the loss stayed finite -- and only in
    the two arms that select this kernel.
    """
    z_half, dz, rho, q_cond, q_tot = _cloud_column()

    def loss(stretch):
        # A column stretched about the surface: exactly the dependence a
        # temperature change induces, reduced to one scalar.
        return jnp.sum(simple_lw_temperature_tendency(
            q_cond, q_tot, rho, dz * stretch, z_half * stretch) ** 2)

    g = jax.grad(loss)(1.0)
    assert _finite(g), (
        f"d(loss)/d(column stretch) = {g}; the clear-sky fractional power is "
        "differentiating at zero")
    assert abs(float(g)) > 0.0, "the geometry must actually reach the flux"


def test_gradient_through_the_STATE_is_finite():
    """The other live input in the SCM: condensate moves with the closure."""
    z_half, dz, rho, q_cond, q_tot = _cloud_column()

    def loss(qc, qt):
        return jnp.sum(simple_lw_temperature_tendency(
            qc, qt, rho, dz, z_half) ** 2)

    g_c, g_t = jax.grad(loss, argnums=(0, 1))(q_cond, q_tot)
    assert _finite(g_c) and _finite(g_t)


def test_gradient_through_density_is_finite():
    z_half, dz, rho, q_cond, q_tot = _cloud_column()

    def loss(scale):
        return jnp.sum(simple_lw_temperature_tendency(
            q_cond, q_tot, rho * scale, dz, z_half) ** 2)

    assert _finite(jax.grad(loss)(1.0))


def test_the_clear_sky_term_is_still_exactly_zero_below_the_inversion():
    """The guard masks the RESULT rather than flooring dz_i, so the forward
    value must be untouched -- a floored base would leak a small clear-sky
    flux into the sub-inversion column, where gSAM's loop bound puts none."""
    z_half, dz, rho, q_cond, q_tot = _cloud_column()
    cfg = SimpleLWConfig()
    z_i = float(np.asarray(simple_lw_inversion_height(q_tot, z_half, cfg))
                .ravel()[0])
    flux = np.asarray(simple_lw_net_upward_flux(
        q_cond, q_tot, rho, dz, z_half, cfg))
    zh = np.asarray(z_half)

    # Below the inversion the flux is exactly the two-exponential floor.
    dq = np.asarray(cfg.kappa_m2_kg * rho * q_cond * dz)
    q_below = np.concatenate([[0.0], np.cumsum(dq)])
    q_above = q_below[-1] - q_below
    floor = (cfg.f0_w_m2 * np.exp(-q_above) + cfg.f1_w_m2 * np.exp(-q_below))
    below = zh <= z_i
    assert below.sum() > 5, "the fixture must have faces below the inversion"
    np.testing.assert_array_equal(flux[below], floor[below])

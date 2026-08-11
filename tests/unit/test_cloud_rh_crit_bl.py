import numpy as np, jax.numpy as jnp, pytest
from legoesm.atmosphere.physics.clouds.config import build_cloud_config
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    sundqvist_cloud_fraction, compute_cloud_properties)


def test_bl_rh_crit_default_is_bit_identical():
    """sigma_bl=1.0 (default) must reproduce the scalar-rh_crit result exactly."""
    cfg = build_cloud_config("sundqvist", rh_crit=0.85)
    RH = jnp.linspace(0.5, 1.05, 40)[None, :]
    sig = jnp.linspace(0.05, 1.0, 40)[None, :]
    np.testing.assert_array_equal(
        np.asarray(sundqvist_cloud_fraction(RH, cfg)),
        np.asarray(sundqvist_cloud_fraction(RH, cfg, sigma=sig)))


def test_bl_rh_crit_applies_below_sigma_bl_only():
    """A HIGHER bl threshold must REDUCE cf in the BL and leave aloft alone."""
    base = build_cloud_config("sundqvist", rh_crit=0.85)
    split = build_cloud_config("sundqvist", rh_crit=0.85,
                               rh_crit_bl=0.95, sigma_bl=0.85)
    RH = jnp.full((1, 40), 0.90)
    sig = jnp.linspace(0.05, 1.0, 40)[None, :]
    cf_b = np.asarray(sundqvist_cloud_fraction(RH, base, sigma=sig))
    cf_s = np.asarray(sundqvist_cloud_fraction(RH, split, sigma=sig))
    bl = np.asarray(sig)[0] >= 0.85
    assert (cf_s[0][bl] < cf_b[0][bl] - 1e-6).all(), "BL cf did not drop"
    np.testing.assert_allclose(cf_s[0][~bl], cf_b[0][~bl], rtol=1e-12)


def test_knobs_reach_compute_cloud_properties():
    """End-to-end through the function radiation calls: the split must change
    the published cloud fraction. Non-vacuous — equality here would mean the
    knobs are dangling again (they were, until 2026-08-11)."""
    from legoesm.thermo import saturation_mixing_ratio
    T = jnp.full((4, 10), 285.0)
    p_full = jnp.linspace(2e4, 1.0e5, 10)[None, :] * jnp.ones((4, 1))
    dp = jnp.full((4, 10), 8.0e3)
    # RH = 0.93 at EVERY level (shipped saturation, not a re-derivation):
    # above both rh_crit values, so cf > 0 in both configs and the split is
    # visible rather than being compared between two all-zero fields.
    q_v = 0.93 * saturation_mixing_ratio(T, p_full)
    kw = dict(T=T, p_full=p_full, q_v=q_v, dp=dp,
              q_cloud=jnp.full((4, 10), 1e-5), q_ice=jnp.zeros((4, 10)))
    a = compute_cloud_properties(config=build_cloud_config(
        "sundqvist", rh_crit=0.85), **kw)
    b = compute_cloud_properties(config=build_cloud_config(
        "sundqvist", rh_crit=0.85, rh_crit_bl=0.99, sigma_bl=0.8), **kw)
    assert not np.allclose(np.asarray(a.cloud_fraction),
                           np.asarray(b.cloud_fraction)), \
        "rh_crit_bl/sigma_bl did not reach compute_cloud_properties"

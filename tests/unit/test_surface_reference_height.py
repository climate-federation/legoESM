"""The surface reference-height and sea-water-humidity corrections.

Two long-standing defects, both measured before being fixed: the similarity
solver was told its inputs sat at 10 m when they sit at ~135 m, and the
atmosphere built a fresh-water surface humidity while the model's own coupler
ocean tile applies the standard 0.98 sea-water factor.  Both were behind
defaults that preserved them.

Every test here failed at some point during the review of that fix, and each
one pins a specific way it went wrong rather than restating the feature.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
    surface_fluxes_at_lowest_level,
)

Z_LOW = 135.0


def _state(n=4):
    u = np.full(n, 8.0)
    v = np.zeros(n)
    T = np.full(n, 298.5)
    q_v = np.full(n, 0.014)
    T_sfc = np.full(n, 300.0)
    q_sfc = np.full(n, 0.0215)
    rho = np.full(n, 1.17)
    return u, v, T, q_v, T_sfc, q_sfc, rho


def _cfg(**kw):
    base = dict(bulk_scheme="coare3", z_ref_model_level=True)
    base.update(kw)
    return SurfaceLayerConfig(**base)


def test_height_correction_lowers_the_ocean_latent_heat_flux():
    """The whole point: labelling a 135 m level as 10 m inflates the flux.

    Not merely "the number changes" -- the SIGN is the claim.  Telling the
    solver the true height must REDUCE the flux, because the same air-sea
    difference is then spread over a deeper layer.
    """
    u, v, T, q_v, T_sfc, q_sfc, rho = _state()
    z = np.full_like(T, Z_LOW)
    _, _, _, lh_10, _ = compute_surface_fluxes(
        u, v, T, q_v, T_sfc, q_sfc, rho, _cfg())
    _, _, _, lh_true, _ = surface_fluxes_at_lowest_level(
        u, v, T, q_v, T_sfc, q_sfc, rho, _cfg(), z)
    assert float(lh_true[0]) < float(lh_10[0])
    ratio = float(lh_10[0]) / float(lh_true[0])
    assert 1.05 < ratio < 1.6, ratio


def test_constant_scheme_gets_both_halves_or_neither():
    """A constant-coefficient config must not receive the warmed air alone.

    ``compute_surface_fluxes`` honours ``z_ref`` on the iterative schemes
    only.  When the height is dropped but the dry-adiabatic temperature
    adjustment survives, the surface becomes ~1.3 K colder than the air for
    free and the scheme reports a downward sensible heat flux made of
    nothing.  This is the defect that reached review twice.
    """
    u, v, T, q_v, T_sfc, q_sfc, rho = _state()
    z = np.full_like(T, Z_LOW)
    cfg = _cfg(bulk_scheme="constant")
    plain = compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, cfg)
    routed = surface_fluxes_at_lowest_level(
        u, v, T, q_v, T_sfc, q_sfc, rho, cfg, z)
    for a, b in zip(plain, routed):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)


def test_missing_height_is_not_a_partial_correction():
    """No height available means no correction, not half of one."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _state()
    plain = compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, _cfg())
    routed = surface_fluxes_at_lowest_level(
        u, v, T, q_v, T_sfc, q_sfc, rho, _cfg(), None)
    for a, b in zip(plain, routed):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)


def test_switch_off_is_byte_identical_to_the_old_path():
    """With the switch off the corrected helper must change nothing."""
    u, v, T, q_v, T_sfc, q_sfc, rho = _state()
    z = np.full_like(T, Z_LOW)
    cfg = _cfg(z_ref_model_level=False)
    plain = compute_surface_fluxes(u, v, T, q_v, T_sfc, q_sfc, rho, cfg)
    routed = surface_fluxes_at_lowest_level(
        u, v, T, q_v, T_sfc, q_sfc, rho, cfg, z)
    for a, b in zip(plain, routed):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)


def test_temperature_is_brought_down_to_the_reference_height():
    """Height and temperature move together, with the COARE sign.

    Passing the height while leaving the temperature at its level value reads
    the stability off a dry adiabat; it is a different bug from the one being
    fixed, so the adjustment is pinned explicitly here.
    """
    u, v, T, q_v, T_sfc, q_sfc, rho = _state()
    z = np.full_like(T, Z_LOW)
    routed = surface_fluxes_at_lowest_level(
        u, v, T, q_v, T_sfc, q_sfc, rho, _cfg(), z)
    explicit = compute_surface_fluxes(
        u, v, T + (constants.g / constants.c_pd) * z, q_v,
        T_sfc, q_sfc, rho, _cfg(), z_ref=z)
    for a, b in zip(routed, explicit):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)
    # and that adjustment is upward by ~1.3 K at this height
    assert 1.0 < float((constants.g / constants.c_pd) * Z_LOW) < 1.6


def test_sea_water_humidity_factor_is_the_shared_constant():
    """The 0.98 lives in one place, and both old private names point at it."""
    from legoesm.coupler.coupler import _Q_SAT_SALINE_FACTOR as coupler_factor
    from legoesm.atmosphere.physics.turbulence.integration import (
        _Q_SAT_SALINE_FACTOR as atmos_factor,
    )
    assert coupler_factor == constants.q_sat_saline_fraction
    assert atmos_factor == constants.q_sat_saline_fraction
    assert 0.97 < constants.q_sat_saline_fraction < 0.99


def test_sea_water_reduces_the_flux_by_far_more_than_two_percent():
    """A 2% cut in surface humidity is ~10% of the flux, not 2%.

    The flux scales with (q_sfc - q_air), so the fractional effect is
    0.02 * q_sfc / (q_sfc - q_air).  This was mis-stated as "a 2% correction"
    for some time; the arithmetic is pinned so nobody restates it.
    """
    u, v, T, q_v, T_sfc, q_sfc, rho = _state()
    z = np.full_like(T, Z_LOW)
    _, _, _, lh_fresh, _ = surface_fluxes_at_lowest_level(
        u, v, T, q_v, T_sfc, q_sfc, rho, _cfg(), z)
    _, _, _, lh_sea, _ = surface_fluxes_at_lowest_level(
        u, v, T, q_v, T_sfc, q_sfc * constants.q_sat_saline_fraction, rho,
        _cfg(), z)
    cut = 1.0 - float(lh_sea[0]) / float(lh_fresh[0])
    assert cut > 0.05, cut
    predicted = (1.0 - constants.q_sat_saline_fraction) * q_sfc[0] / (
        q_sfc[0] - q_v[0])
    assert predicted == pytest.approx(cut, rel=0.25), (predicted, cut)


@pytest.mark.parametrize("stated", [None, True, False])
def test_experiment_switch_is_tri_state(stated):
    """None leaves the scheme alone; True and False both reach it.

    Every case starts from a scheme that already carries True, because the
    earlier version of this test started from False and therefore passed even
    when the override was dropped entirely -- it could not fail for the reason
    it named (codex).  Starting from True, each of the three outcomes is
    distinguishable: None must PRESERVE it, False must overturn it, True must
    leave it set.
    """
    from legoesm.driver.physics_pipeline import apply_surface_flux_config
    from legoesm.atmosphere.physics.turbulence.config import (
        LouisConfig, TurbulenceConfig,
    )

    class _Cfg:
        surface_bulk_scheme = "constant"
        surface_gustiness_zi = None
        surface_thermo_convention = "legoesm"
        surface_stability_scheme = "dyer1974"
        surface_ocean_q_sfc_saline = None
        surface_tiled = False
        grid = None

        def __init__(self, zml):
            self.surface_z_ref_model_level = zml

    on = SurfaceLayerConfig(z_ref_model_level=True)
    tc = TurbulenceConfig(scheme="louis", louis=LouisConfig(surface=on))
    out = apply_surface_flux_config(tc, _Cfg(stated))
    got = getattr(out, out.scheme).surface.z_ref_model_level
    assert got is (True if stated is None else stated)


def test_explicit_true_reaches_a_scheme_that_has_it_off():
    """The complement: explicit True must turn it ON where it was off.

    The tri-state test above starts from a scheme carrying True, where None
    and True expect the same answer -- so deleting the explicit-True
    propagation alone leaves it passing (codex). This is the case that
    distinguishes them.
    """
    from legoesm.driver.physics_pipeline import apply_surface_flux_config
    from legoesm.atmosphere.physics.turbulence.config import (
        LouisConfig, TurbulenceConfig,
    )

    class _Cfg:
        surface_bulk_scheme = "constant"
        surface_gustiness_zi = None
        surface_thermo_convention = "legoesm"
        surface_stability_scheme = "dyer1974"
        surface_z_ref_model_level = True
        surface_ocean_q_sfc_saline = None
        surface_tiled = False
        grid = None

    off = SurfaceLayerConfig(z_ref_model_level=False)
    tc = TurbulenceConfig(scheme="louis", louis=LouisConfig(surface=off))
    out = apply_surface_flux_config(tc, _Cfg())
    assert getattr(out, out.scheme).surface.z_ref_model_level is True


def test_explicit_saline_false_overturns_a_scheme_level_true():
    """An explicit opt-out must survive the otherwise-default fast path.

    This exact combination -- everything else default, the run asking for
    False, the scheme carrying True -- returned early and kept the True four
    separate times during review.
    """
    from legoesm.driver.physics_pipeline import apply_surface_flux_config
    from legoesm.atmosphere.physics.turbulence.config import (
        LouisConfig, TurbulenceConfig,
    )

    class _Cfg:
        surface_bulk_scheme = "constant"
        surface_gustiness_zi = None
        surface_thermo_convention = "legoesm"
        surface_stability_scheme = "dyer1974"
        surface_z_ref_model_level = None
        surface_ocean_q_sfc_saline = False
        surface_tiled = True
        grid = None

    on = SurfaceLayerConfig(ocean_q_sfc_saline=True)
    tc = TurbulenceConfig(scheme="louis", louis=LouisConfig(surface=on))
    out = apply_surface_flux_config(tc, _Cfg())
    assert getattr(out, out.scheme).surface.ocean_q_sfc_saline is False


def test_saline_is_on_by_default_where_the_lane_can_honour_it():
    """Unstated means on wherever ocean and land can be told apart.

    The tiled mosaic carries an ocean fraction, so it qualifies; keying this
    off the grid type instead left the correction unreachable on exactly the
    lane it was written for.
    """
    from legoesm.driver.physics_pipeline import apply_surface_flux_config
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    class _Cfg:
        surface_bulk_scheme = "constant"
        surface_gustiness_zi = None
        surface_thermo_convention = "legoesm"
        surface_stability_scheme = "dyer1974"
        surface_z_ref_model_level = None
        surface_ocean_q_sfc_saline = None
        surface_tiled = True
        grid = None

    out = apply_surface_flux_config(TurbulenceConfig(scheme="louis"), _Cfg())
    assert getattr(out, out.scheme).surface.ocean_q_sfc_saline is True


def test_half_level_convention_puts_the_surface_at_the_last_index():
    """The whole correction rests on z_half[:, -1] being the surface edge.

    Ten call sites compute the reference height as z_full[:, -1] minus
    z_half[:, -1].  If the last half level were a layer midpoint instead, that
    subtraction silently returns a half-layer offset -- tens of metres, right
    shape, plausible magnitude, no error anywhere.  If the ordering were
    reversed it returns a NEGATIVE height and the log in the similarity solver
    produces NaN in a lane that previously gave finite answers.  Neither is
    caught by any other test here, so the convention is pinned rather than
    assumed (GLM).
    """
    import jax.numpy as jnp
    from legoesm.atmosphere.physics._shared import compute_heights_from_sigma

    ncol, nlev = 3, 12
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    p_s = jnp.asarray([101300.0, 97000.0, 85000.0])   # incl. a high-terrain column
    p_half = sigma_half[None, :] * p_s[:, None]
    T = jnp.full((ncol, nlev), 280.0)
    z_full, z_half = compute_heights_from_sigma(T, p_half)

    # the surface edge is the LAST half level, and it is exactly zero there
    np.testing.assert_allclose(np.asarray(z_half[:, -1]), 0.0, atol=1e-9)
    # heights increase upward, so the lowest full level sits above it
    z_low = np.asarray(z_full[:, -1] - z_half[:, -1])
    assert np.all(z_low > 0.0), z_low
    # and it is a plausible lowest-level height, not a half-layer sliver
    assert np.all(z_low < 1000.0), z_low


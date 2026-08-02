"""Land<->atmosphere turbulent-flux interface consistency.

A flux exchanged across a coupling must carry the SAME value at BOTH ends.
These tests drive the two REAL entry points with matched inputs:

* land end       — ``legoesm.land.surface_scheme.simple_seb.compute_simple_seb_fluxes``
* atmosphere end — ``legoesm.atmosphere.physics.turbulence.surface_layer
                    .compute_tiled_surface_fluxes`` with ``frac_land = 1``

and assert they agree to round-off across a sweep that INCLUDES the
condensation regime (where the interface condensation floor binds) and the
strongly-unstable light-wind regime (where the interface transfer-coefficient
ceiling binds).

Anti-vacuity: :func:`test_sweep_exercises_the_condensation_floor` and
:func:`test_sweep_exercises_the_exchange_coeff_ceiling` prove that the sweep
actually reaches both limits, so the agreement assertions are not passing
trivially.  Removing either limit from ONE end makes the agreement tests red.

Sign convention throughout: turbulent fluxes are POSITIVE UPWARD (surface ->
atmosphere); ``lhflx < 0`` is condensation onto the surface.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    SurfaceTileSpec,
    compute_tiled_surface_fluxes,
)
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.land_interface_flux import (
    LAND_INTERFACE_FLUX,
    LandInterfaceFluxConfig,
    apply_condensation_floor,
    land_interface_most_fluxes,
)
from legoesm.land.config import LandConfig
from legoesm.land.surface_scheme.simple_seb import compute_simple_seb_fluxes
from legoesm.thermo import saturation_mixing_ratio

# Interface geometry shared by both ends (matched by construction below):
# the AMIP production land roughness/reference height and MOST iteration count.
_Z0_LAND = 0.1        # surface_z0_land, config/amip/amip_production.yaml
_Z_REF = 10.0
_N_ITER = 5
_P_S = 1.0e5
_BETA_SOIL = 0.8      # root-zone soil-moisture stress (not 1, so the alpha
                      # method is genuinely exercised)


def _land_config() -> LandConfig:
    """Land config on the MOST land path with the interface geometry."""
    return LandConfig(
        bulk_scheme="most", z_ref=_Z_REF, bulk_n_iter=_N_ITER,
        z0_land=_Z0_LAND, snow_albedo_feedback=False,
    )


def _surface_layer_config() -> SurfaceLayerConfig:
    """Atmospheric LAND-tile surface-layer config, matched to the land end.

    ``gustiness_w_zi=0.0`` matches the land end, which calls
    ``compute_most_fluxes`` without gustiness (scheme-native 0 for ``most``).
    """
    return SurfaceLayerConfig(
        bulk_scheme="most", z0=_Z0_LAND, z_ref=_Z_REF, bulk_n_iter=_N_ITER,
        gustiness_w_zi=0.0, thermo_convention="legoesm",
    )


def _both_ends(T_air_K, T_sfc_K, wind, rh_air):
    """Latent + sensible flux from BOTH ends for one matched column.

    Returns ``(lh_land, sh_land, lh_atm, sh_atm)`` in [W/m^2], positive upward.
    """
    T_air = jnp.asarray([T_air_K])
    T_sfc = jnp.asarray([T_sfc_K])
    u = jnp.asarray([wind])
    v = jnp.zeros_like(u)
    q_air = rh_air * saturation_mixing_ratio(T_air, _P_S)
    rho = _P_S / (constants.R_d * T_air)
    zero = jnp.zeros_like(T_air)

    # --- LAND end -----------------------------------------------------------
    forcing = AtmToSurface(
        sw_down=zero, lw_down=zero, precip_total=zero, precip_snow=zero,
        T_lowest=T_air, q_lowest=q_air, u_lowest=u, v_lowest=v,
        p_lowest=0.99 * _P_S * jnp.ones_like(T_air),
        p_surface=_P_S * jnp.ones_like(T_air), rho_lowest=rho,
        cos_zenith=0.5 * jnp.ones_like(T_air),
        co2_ppmv=412.0 * jnp.ones_like(T_air),
        has_radiation=jnp.ones_like(T_air),
        has_precipitation=jnp.ones_like(T_air),
    )
    out = compute_simple_seb_fluxes(
        T_surface=T_sfc, snow=zero, snow_age=zero,
        beta_soil=_BETA_SOIL * jnp.ones_like(T_air), forcing=forcing,
        land_config=_land_config(), U_min=1.0, lat=None, carbon_state=None,
        dt=1800.0, land_params=None, albedo_land=0.2, emissivity=0.97,
        z0=_Z0_LAND,
    )

    # --- ATMOSPHERE end -----------------------------------------------------
    # Snow-free column, so the land SEB's surface humidity is
    # beta_soil * q_sat_liquid(T_sfc) -- the SAME shared thermo helper.
    q_sfc_land = _BETA_SOIL * saturation_mixing_ratio(T_sfc, _P_S)
    tiles = SurfaceTileSpec(
        frac_ocean=jnp.zeros_like(T_air), frac_ice=jnp.zeros_like(T_air),
        frac_land=jnp.ones_like(T_air),
        T_ocean=T_sfc, T_ice=T_sfc, T_land=T_sfc,
        q_sfc_ocean=q_sfc_land, q_sfc_ice=q_sfc_land, q_sfc_land=q_sfc_land,
    )
    cfg = _surface_layer_config()
    _, _, sh_atm, lh_atm, _ = compute_tiled_surface_fluxes(
        u, v, T_air, q_air, rho, tiles, cfg, cfg, cfg,
    )
    return (float(out.lhflx[0]), float(out.shflx[0]),
            float(lh_atm[0]), float(sh_atm[0]))


# Sweep: (T_air [C], T_sfc [C], |U| [m/s], RH_air).
# Rows 1-6 are the CONDENSATION regime (warm moist air over a cold dry skin --
# spring warm advection over frozen ground / marine air over snow); rows 7-10
# are the strongly-unstable light-wind regime (hot arid daytime surface layer);
# rows 11-13 are ordinary near-neutral evaporation.
_SWEEP = [
    (5.0, -30.0, 2.0, 0.95),
    (5.0, -30.0, 10.0, 0.95),
    (20.0, -10.0, 10.0, 0.95),
    (20.0, 0.0, 10.0, 0.95),
    (30.0, -10.0, 25.0, 0.95),
    (30.0, 0.0, 25.0, 0.95),
    (10.0, 40.0, 0.5, 0.20),
    (10.0, 40.0, 2.0, 0.20),
    (25.0, 40.0, 2.0, 0.20),
    (20.0, 50.0, 1.0, 0.10),
    (10.0, 20.0, 2.0, 0.50),
    (25.0, 20.0, 10.0, 0.50),
    (15.0, 15.0, 5.0, 0.70),
]


def _sweep_points():
    for T_air_c, T_sfc_c, wind, rh in _SWEEP:
        yield (constants.T_freeze + T_air_c, constants.T_freeze + T_sfc_c,
               wind, rh)


@pytest.mark.parametrize("T_air_K,T_sfc_K,wind,rh", list(_sweep_points()))
def test_latent_flux_agrees_at_both_ends(T_air_K, T_sfc_K, wind, rh):
    """LH is identical at both ends of the interface (round-off tolerance).

    Requires ``JAX_ENABLE_X64=1``; the tolerance below is float64 round-off on
    an O(1e3 W/m^2) flux.
    """
    lh_land, _, lh_atm, _ = _both_ends(T_air_K, T_sfc_K, wind, rh)
    assert lh_land == pytest.approx(lh_atm, rel=1e-10, abs=1e-8), (
        f"latent flux disagrees across the land<->atmosphere interface at "
        f"T_air={T_air_K:.2f} K, T_sfc={T_sfc_K:.2f} K, |U|={wind} m/s, "
        f"RH={rh}: land={lh_land:.4f} W/m^2 vs atmosphere={lh_atm:.4f} W/m^2 "
        f"(gap {lh_atm - lh_land:.4f} W/m^2 = "
        f"{(lh_atm - lh_land) / constants.L_v * 86400.0:.3f} mm/day)"
    )


@pytest.mark.parametrize("T_air_K,T_sfc_K,wind,rh", list(_sweep_points()))
def test_sensible_flux_agrees_at_both_ends(T_air_K, T_sfc_K, wind, rh):
    """SH is identical at both ends (it shares the transfer-coefficient cap)."""
    _, sh_land, _, sh_atm = _both_ends(T_air_K, T_sfc_K, wind, rh)
    assert sh_land == pytest.approx(sh_atm, rel=1e-10, abs=1e-8), (
        f"sensible flux disagrees at T_air={T_air_K:.2f} K, "
        f"T_sfc={T_sfc_K:.2f} K, |U|={wind} m/s: "
        f"land={sh_land:.4f} vs atmosphere={sh_atm:.4f} W/m^2"
    )


def test_sweep_exercises_the_condensation_floor():
    """Anti-vacuity: the floor BINDS somewhere in the sweep.

    Without this the agreement tests could pass on a sweep that never reaches
    the limit, and deleting the floor from one end would go unnoticed.
    """
    floor = LAND_INTERFACE_FLUX.condensation_floor_w
    hits = [pt for pt in _sweep_points()
            if _both_ends(*pt)[0] == pytest.approx(floor, rel=1e-9)]
    assert hits, (
        "no sweep point reaches the condensation floor "
        f"({floor} W/m^2) -- the interface-agreement tests would be vacuous "
        "for the floor"
    )


def test_sweep_exercises_the_exchange_coeff_ceiling():
    """Anti-vacuity: the transfer-coefficient ceiling BINDS somewhere.

    Compares the land end against an UNCAPPED MOST solve with otherwise
    identical arguments; a point where they differ proves the ceiling is live
    (it was previously believed inert).
    """
    from legoesm.core.bulk_flux import compute_most_fluxes

    binds = []
    for T_air_K, T_sfc_K, wind, rh in _sweep_points():
        T_air = jnp.asarray([T_air_K])
        T_sfc = jnp.asarray([T_sfc_K])
        u = jnp.asarray([wind])
        q_air = rh * saturation_mixing_ratio(T_air, _P_S)
        q_sfc = _BETA_SOIL * saturation_mixing_ratio(T_sfc, _P_S)
        rho = _P_S / (constants.R_d * T_air)
        common = dict(z_ref=_Z_REF, z0_init=_Z0_LAND, scheme="most",
                      n_iter=_N_ITER, L_latent=constants.L_v)
        _, _, _, lh_capped, _ = land_interface_most_fluxes(
            u, jnp.zeros_like(u), T_air, q_air, T_sfc, q_sfc, rho, **common)
        _, _, _, lh_free, _ = compute_most_fluxes(
            u, jnp.zeros_like(u), T_air, q_air, T_sfc, q_sfc, rho, **common)
        if not np.isclose(float(lh_capped[0]), float(lh_free[0]), rtol=1e-9):
            binds.append((T_air_K, T_sfc_K, wind,
                          float(lh_free[0]) / float(lh_capped[0])))
    assert binds, (
        "the max_exchange_coeff ceiling never binds in the sweep -- add an "
        "unstable light-wind point or the ceiling half of the interface "
        "contract is untested"
    )


def test_condensation_floor_sign_convention():
    """The floor bounds condensation ONLY; evaporation passes through.

    Convention: POSITIVE UPWARD.  ``lhflx > 0`` is evaporation and must be
    untouched; ``lhflx < floor < 0`` is condensation and must be raised TO the
    floor (never past zero, never sign-flipped).
    """
    floor = LAND_INTERFACE_FLUX.condensation_floor_w
    lh = jnp.asarray([500.0, 1.0, 0.0, -10.0, floor, -1e4])
    out = np.asarray(apply_condensation_floor(lh))
    np.testing.assert_allclose(out[:4], np.asarray(lh)[:4], rtol=0, atol=0)
    assert out[4] == pytest.approx(floor)
    assert out[5] == pytest.approx(floor)
    assert np.all(out <= np.maximum(np.asarray(lh), 0.0) + abs(floor) + 1e-9)
    # Never turns condensation into evaporation.
    assert np.all(out[np.asarray(lh) < 0.0] <= 0.0)


def test_land_interface_rejects_a_one_sided_ceiling_override():
    """``max_exchange_coeff`` is owned by the interface, not by a caller."""
    args = (jnp.asarray([2.0]), jnp.asarray([0.0]), jnp.asarray([290.0]),
            jnp.asarray([0.005]), jnp.asarray([300.0]), jnp.asarray([0.02]),
            jnp.asarray([1.2]))
    with pytest.raises(ValueError, match="owned by LandInterfaceFluxConfig"):
        land_interface_most_fluxes(*args, max_exchange_coeff=0.5)


def test_land_interface_flux_jit_parity_and_grad():
    """JIT parity + a finite reverse-mode gradient through the shared entry."""
    # Warm moist air (20 C, RH 95 %) over a -10 C skin at 10 m/s: the same
    # condensation column as the sweep, which the floor DOES reach.
    T_air = jnp.asarray([constants.T_freeze + 20.0])
    T_sfc = jnp.asarray([constants.T_freeze - 10.0])
    u = jnp.asarray([10.0])
    q_sfc = _BETA_SOIL * saturation_mixing_ratio(T_sfc, _P_S)
    rho = _P_S / (constants.R_d * T_air)
    kw = dict(z_ref=_Z_REF, z0_init=_Z0_LAND, scheme="most", n_iter=_N_ITER,
              L_latent=constants.L_v)

    def _lh(q_atm):
        return land_interface_most_fluxes(
            u, jnp.zeros_like(u), T_air, q_atm, T_sfc, q_sfc, rho,
            **kw)[3].sum()

    # Dry air -> evaporation, well above the floor: finite, non-zero
    # sensitivity (more vapour in the air => less evaporation, so
    # d(lhflx)/d(q_atm) < 0 with the positive-upward convention).
    q_dry = jnp.asarray([1e-4])
    eager = float(_lh(q_dry))
    jitted = float(jax.jit(_lh)(q_dry))
    assert eager == pytest.approx(jitted, rel=1e-12, abs=1e-12)
    assert eager > LAND_INTERFACE_FLUX.condensation_floor_w
    g = float(jax.grad(_lh)(q_dry)[0])
    assert np.isfinite(g) and g < 0.0

    # Deep inside the floored regime the clamp zeroes the gradient exactly --
    # finite and well defined, which is what AD-safety requires.
    q_wet = 0.95 * saturation_mixing_ratio(T_air, _P_S)
    assert float(_lh(q_wet)) == pytest.approx(
        LAND_INTERFACE_FLUX.condensation_floor_w, rel=1e-12)
    assert float(jax.jit(_lh)(q_wet)) == pytest.approx(
        LAND_INTERFACE_FLUX.condensation_floor_w, rel=1e-12)
    assert float(jax.grad(_lh)(q_wet)[0]) == pytest.approx(0.0, abs=0.0)


def test_custom_limits_thread_to_both_ends():
    """A non-default config changes the floor the shared entry point applies."""
    cfg = LandInterfaceFluxConfig(max_exchange_coeff=0.02,
                                  condensation_floor_w=-42.0)
    lh = jnp.asarray([-1000.0])
    assert float(apply_condensation_floor(lh, cfg)[0]) == pytest.approx(-42.0)

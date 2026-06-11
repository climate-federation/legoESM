"""Integration tests for the aerosol → microphysics → cloud → radiation chain.

Each test perturbs ONE upstream input and asserts a nonzero, correctly
signed response downstream — wiring tests, not climate validation:

  1. AOD → specified N_c (Andreae 2009 inversion, ``effective_Nc``)
  2. N_c → Morrison warm-rain autoconversion (KK2000 ``Nc^-1.79``:
     more droplets → slower rain formation, the cloud-lifetime effect)
  3. N_c → cloud-optics effective radius (Twomey: more droplets →
     smaller r_eff)
  4. aerosol_od → RRTMGP shortwave flux (direct effect: more AOD →
     less SW reaching the surface)
"""

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
    ccn_from_aod,
)
from legoesm.atmosphere.physics.microphysics._warm_rain import effective_Nc

jax.config.update("jax_enable_x64", True)


def test_effective_nc_specified_field_mode():
    """nc_specified_field uses the caller-filled N_c, not constant Nc_0."""
    N_c = jnp.full((4, 5), 3.0e8)
    out = effective_Nc(N_c, 1.0e8, predict_Nc=False, nc_specified_field=True)
    assert jnp.allclose(out, 3.0e8)
    # Zero-filled field falls back to Nc_0 (no garbage Nc in the PSD).
    out0 = effective_Nc(jnp.zeros((4, 5)), 1.0e8,
                        predict_Nc=False, nc_specified_field=True)
    assert jnp.allclose(out0, 1.0e8)
    # Legacy specified-Nc (flag off) ignores the field entirely.
    legacy = effective_Nc(N_c, 1.0e8, predict_Nc=False)
    assert jnp.allclose(legacy, 1.0e8)


def test_aod_to_nc_to_autoconversion_sign():
    """Polluted column (high AOD → high N_c) autoconverts SLOWER (KK2000)."""
    from legoesm.atmosphere.physics.microphysics._warm_rain import (
        autoconversion_kk2000,
    )
    rho = jnp.full((1, 3), 1.0)
    q_c = jnp.full((1, 3), 5.0e-4)
    dt = 60.0
    n_clean = ccn_from_aod(jnp.full((1, 3), 0.05))   # remote marine
    n_polluted = ccn_from_aod(jnp.full((1, 3), 0.8))  # smoky/urban
    assert float(n_polluted[0, 0]) > 3.0 * float(n_clean[0, 0])
    au_clean, _, _ = autoconversion_kk2000(q_c, n_clean, rho, dt)
    au_polluted, _, _ = autoconversion_kk2000(q_c, n_polluted, rho, dt)
    # Cloud-water sink: |au_polluted| < |au_clean| (lifetime effect).
    assert jnp.all(jnp.abs(au_polluted) < jnp.abs(au_clean))
    assert jnp.all(jnp.abs(au_clean) > 0.0)


def test_nc_to_reff_twomey_sign():
    """More droplets at fixed LWC → smaller liquid effective radius."""
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties,
    )
    ncol, nlev = 2, 6
    T = jnp.full((ncol, nlev), 280.0)
    p_full = jnp.broadcast_to(
        jnp.linspace(3.0e4, 9.5e4, nlev)[None, :], (ncol, nlev))
    dp = jnp.full((ncol, nlev), 1.0e4)
    q_v = jnp.full((ncol, nlev), 8.0e-3)
    q_c = jnp.full((ncol, nlev), 3.0e-4)
    cfg = CloudConfig(scheme="sundqvist")
    n_clean = jnp.full((ncol, nlev), float(ccn_from_aod(jnp.array(0.05))))
    n_polluted = jnp.full((ncol, nlev), float(ccn_from_aod(jnp.array(0.8))))
    props_clean = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
        q_cloud=q_c, n_cloud=n_clean)
    props_polluted = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
        q_cloud=q_c, n_cloud=n_polluted)
    r_clean = props_clean.to_rrtmg_kwargs()["cloud_r_eff_liq"]
    r_polluted = props_polluted.to_rrtmg_kwargs()["cloud_r_eff_liq"]
    assert jnp.all(r_polluted < r_clean), (
        f"Twomey sign violated: r_polluted={float(r_polluted.max()):.2e} "
        f"!< r_clean={float(r_clean.min()):.2e}")


def test_aerosol_od_to_sw_flux_direct_effect():
    """More aerosol optical depth → less downward SW at the surface."""
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    ncol, nlev = 2, 8
    solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
    T = jnp.broadcast_to(
        jnp.linspace(220.0, 290.0, nlev)[None, :], (ncol, nlev))
    p_half = jnp.broadcast_to(
        jnp.linspace(1.0e3, 1.0e5, nlev + 1)[None, :], (ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    q_v = jnp.full((ncol, nlev), 2.0e-3)
    kw = dict(
        T=T, p_full=p_full, p_half=p_half,
        sfc_temperature=jnp.full((ncol,), 295.0),
        q_v=q_v, cos_zenith=jnp.full((ncol,), 0.6),
    )
    out_clear = solver.solve_columns(**kw)
    out_aer = solver.solve_columns(
        **kw, aerosol_optical_depth=jnp.full((ncol, nlev), 0.05))
    sw_sfc_clear = out_clear.sw_flux_down[:, -1]
    sw_sfc_aer = out_aer.sw_flux_down[:, -1]
    assert jnp.all(sw_sfc_aer < sw_sfc_clear), (
        f"aerosol direct effect missing: {float(sw_sfc_aer[0]):.2f} "
        f"!< {float(sw_sfc_clear[0]):.2f}")

"""Marine-Sc cloud-top entrainment term in the Louis turbulence scheme.

The Louis scheme is a LOCAL down-gradient K scheme: at a stratocumulus
inversion Ri>>0 => f_h->0 => Kh~0, so the boundary layer is capped and its
moisture is trapped (the supply-limited marine-Sc liquid-cloud bias that drives
the AMIP albedo error).  ``cloudtop_entrainment_efficiency > 0`` adds a flux-matched
interior diffusivity ``K_ent`` that vents BL-top moisture UP into the dry free
troposphere.  It is localized INTRINSICALLY at the cloud top by the product of
three gates (drying dq/dz<0, inversion, cloudy layer below) -- no PBL-height
diagnosis -- which also fixes the sign (can only DRY the BL) and keeps the BL
interior + surface layer untouched.

These tests prove the physical claims WITHOUT a full SCM run, at the scheme
level with a constructed marine-Sc column and ZERO surface flux (closed column,
entrainment isolated):

* the term DRIES the cloud top and MOISTENS the free troposphere just above it;
* the SURFACE layer is untouched (no evaporation trade);
* the column water AND potential temperature are CONSERVED (rho*dz weighted);
* efficiency=0 / default-off is byte-identical to no entrainment;
* the standalone helper is >=0 and needs ALL THREE gates (drying, inversion,
  cloudy-below) to be non-zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.turbulence.config import LouisConfig
from legoesm.atmosphere.physics.turbulence.louis import (
    _cloudtop_entrainment_diffusivity,
    louis_turbulence,
)

jax.config.update("jax_enable_x64", True)


def _marine_sc_column(nlev: int = 40):
    """Well-mixed saturated marine-Sc BL capped by a sharp inversion.

    Surface-last convention (index -1 = surface, index 0 = model top; z DECREASES
    with index).  BL 0-800 m: well-mixed theta, SATURATED.  Sharp inversion at
    ~800 m: +12 K theta jump, moisture drop.  Free troposphere above: warm, DRY.
    """
    z_full = jnp.linspace(3000.0, 40.0, nlev)[None, :]  # (1, nlev) top->surface
    p_sfc = 1.0e5
    p_full = p_sfc * jnp.exp(-z_full / 8000.0)
    z_inv = 800.0
    theta = jnp.where(
        z_full <= z_inv, 288.0,
        288.0 + 12.0 + 0.004 * (z_full - z_inv),  # +12 K jump, 4 K/km FT lapse
    )
    exner = (p_full / constants.p_ref) ** constants.kappa
    T = theta * exner
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = jnp.where(z_full <= z_inv, q_sat, 0.1 * q_sat)  # saturated BL, dry FT
    rho = p_full / (constants.R_d * T)
    u = 5.0 + 0.002 * z_full
    v = jnp.zeros_like(z_full)
    z_half = jnp.concatenate(
        [z_full[:, :1] + 100.0, 0.5 * (z_full[:, :-1] + z_full[:, 1:]),
         z_full[:, -1:] - 40.0], axis=1)
    p_half = p_sfc * jnp.exp(-z_half / 8000.0)
    zero = jnp.zeros((1,))
    # ZERO surface flux => closed column: any q/theta change is interior only.
    surface_flux = (zero, zero, zero, zero, zero + 0.1)
    return dict(
        u=u, v=v, T=T, q_v=q_v, p_full=p_full, p_half=p_half,
        z_full=z_full, z_half=z_half, T_sfc=T[:, -1], q_sfc=q_sat[:, -1],
        rho=rho, dt=300.0, surface_flux=surface_flux,
    )


def _cloud_top_half_index(col):
    """Half-level index of the sharpest upward moisture DROP (the cloud top)."""
    dz = jnp.abs(col["z_full"][:, :-1] - col["z_full"][:, 1:])
    dq_v_dz = (col["q_v"][:, :-1] - col["q_v"][:, 1:]) / dz
    return int(jnp.argmin(dq_v_dz[0]))  # most negative => cloud top


def test_helper_needs_all_three_gates():
    """K_ent >= 0 and vanishes unless drying AND inversion AND cloudy-below."""
    n = 20
    dz = jnp.full((1, n), 50.0)
    dq_drop = jnp.full((1, n), -2.0e-4)  # moisture DROPS upward (cloud top)
    inv = jnp.full((1, n), 5.0e-2)       # strong inversion
    rh_sat = jnp.ones((1, n))
    cfg = LouisConfig(cloudtop_entrainment_efficiency=0.3)

    k = _cloudtop_entrainment_diffusivity(dz, dq_drop, inv, rh_sat, cfg)
    assert jnp.all(k >= 0.0)
    assert float(jnp.max(k)) > 0.0

    # Each gate individually off => negligible (< 2% of the on-regime max).
    k_no_dry = _cloudtop_entrainment_diffusivity(
        dz, jnp.full((1, n), +2.0e-4), inv, rh_sat, cfg)  # moisture INCREASES up
    k_no_inv = _cloudtop_entrainment_diffusivity(
        dz, dq_drop, jnp.zeros((1, n)), rh_sat, cfg)
    k_dry = _cloudtop_entrainment_diffusivity(
        dz, dq_drop, inv, jnp.zeros((1, n)), cfg)
    for koff in (k_no_dry, k_no_inv, k_dry):
        assert float(jnp.max(koff)) < 2e-2 * float(jnp.max(k))
    # efficiency = 0 => exactly zero.
    k0 = _cloudtop_entrainment_diffusivity(
        dz, dq_drop, inv, rh_sat,
        cfg._replace(cloudtop_entrainment_efficiency=0.0))
    assert float(jnp.max(jnp.abs(k0))) == 0.0


def test_entrainment_dries_cloud_top_not_surface_and_conserves():
    """OFF->ON: cloud top dries, FT moistens, surface untouched, column conserved."""
    col = _marine_sc_column()
    off = louis_turbulence(**col, config=LouisConfig(
        cloudtop_entrainment_efficiency=0.0))
    on = louis_turbulence(**col, config=LouisConfig(
        cloudtop_entrainment_efficiency=0.5))
    # Isolate the ENTRAINMENT's contribution: delta = ON - OFF (removes the
    # baseline Louis diffusion, which per step is tiny + can slightly moisten).
    dq = on.dq_v_dt[0] - off.dq_v_dt[0]
    i_ct = _cloud_top_half_index(col)  # cloud-top half index

    # (1) The entrainment DRIES the cloud-top BL level (the full level just below
    #     the cloud-top interface) by a measurable amount.
    assert float(dq[i_ct + 1]) < 0.0
    assert float(dq[i_ct + 1]) < -1e-9
    # (2) The moisture is exported ALOFT: net moistening above the cloud top.
    assert float(jnp.sum(dq[: i_ct + 1])) > 0.0
    # (3) Surface layer untouched (no evaporation trade).
    assert abs(float(dq[-1])) < 1e-3 * abs(float(dq[i_ct + 1]))

    # (4) Conservation (closed column): rho*dz-weighted integral of the
    #     entrainment moisture AND potential-temperature contribution ~ 0.
    z_half = col["z_half"][0]
    dz_layer = jnp.abs(z_half[:-1] - z_half[1:])          # (nlev,)
    dm = col["rho"][0] * dz_layer                          # layer mass [kg/m^2]
    exner_inv = (constants.p_ref / col["p_full"][0]) ** constants.kappa
    dtheta = (on.dT_dt[0] - off.dT_dt[0]) * exner_inv
    for tend in (dq, dtheta):
        col_int = float(jnp.sum(tend * dm))
        scale = float(jnp.sum(jnp.abs(tend) * dm)) + 1e-30
        assert abs(col_int) < 1e-6 * scale


def test_efficiency_zero_byte_identical_to_off():
    """The default (efficiency 0.0) is OFF and byte-identical to explicit off."""
    col = _marine_sc_column()
    assert LouisConfig().cloudtop_entrainment_efficiency == 0.0  # off by default
    default_off = louis_turbulence(**col, config=LouisConfig())
    explicit_off = louis_turbulence(**col, config=LouisConfig(
        cloudtop_entrainment_efficiency=0.0))
    assert float(jnp.max(jnp.abs(default_off.dq_v_dt - explicit_off.dq_v_dt))) == 0.0
    assert float(jnp.max(jnp.abs(default_off.dT_dt - explicit_off.dT_dt))) == 0.0

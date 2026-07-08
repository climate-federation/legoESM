"""Opt-in adiabatic static-stability N^2 for the Richardson (PP81) vertical
mixing and the enhanced-diffusion convective trigger.

BUG (ocean param oracle review): both schemes measured static stability from
the *in-situ* density gradient, which carries the compressibility bias.  On a
deep column that is statically UNSTABLE in potential density, the in-situ N^2
can still read > 0, so the Richardson scheme keeps background mixing and the
convective trigger is MISSED.

FIX: an opt-in ``n2_mode`` config field.  Default ``"insitu"`` preserves the
legacy forward result bit-for-bit; ``"adiabatic"`` measures the TRUE static
stability (both interface parcels displaced to the upper cell's pressure,
``eos.compute_buoyancy_frequency_adiabatic``), which is SIGNED and unbiased by
compressibility.

This module pins:
  (a) Richardson: ``n2_mode="adiabatic"`` gives higher K_v/A_v than "insitu"
      on a compressibility-masked unstable column;
  (b) enhanced-diffusion: the convective trigger FIRES (flag~1, K~K_conv) with
      ``n2_mode="adiabatic"`` where "insitu" does not (flag~0, K~K_bg);
  (c) an unknown ``n2_mode`` raises ValueError at function entry (both schemes);
  (d) the DEFAULT config is "insitu" and reproduces the legacy N^2 math exactly.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.eos import (
    compute_buoyancy_frequency,
    compute_hydrostatic_pressure,
    rho_0,
    wright_eos,
)
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.physics.vertical_mixing._shared import richardson_number
from legoesm.ocean.physics.vertical_mixing.config import (
    RichardsonVerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.richardson import (
    richardson_vertical_mixing,
)
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.enhanced_diffusion import (
    convective_K_A_flag,
    enhanced_diffusion_convection,
)

_EPS = float(jnp.finfo(jnp.float32).eps)


def _masked_unstable_column():
    """Build a deep column that is statically UNSTABLE in potential density
    (upper cell saltier -> denser at a common pressure) but reads STABLE in
    in-situ N^2 because the deep compression of the lower parcel dominates.

    Returns ``(z_coord, J, T, S, rho_insitu, p_cell)`` with a small (2, 2)
    horizontal footprint.
    """
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    horiz = (2, 2)
    J = jnp.ones(horiz, dtype=jnp.float64)
    shape = horiz + (z.n_levels,)
    # Uniform T; salinity DECREASING with depth -> each upper cell is saltier
    # (denser potential density) than the cell below it -> potential-unstable.
    T = jnp.broadcast_to(jnp.full(z.n_levels, 4.0, dtype=jnp.float64), shape)
    S = jnp.broadcast_to(
        jnp.array([35.5, 35.0, 34.5, 34.0], dtype=jnp.float64), shape,
    )
    eta = jnp.zeros(horiz, dtype=jnp.float64)
    # Self-consistent in-situ density + hydrostatic pressure (two-pass).
    rho = wright_eos(T, S, jnp.zeros_like(T))
    p_cell = compute_hydrostatic_pressure(rho, eta, z.dz_ref, J, rho_0)
    rho = wright_eos(T, S, p_cell)
    p_cell = compute_hydrostatic_pressure(rho, eta, z.dz_ref, J, rho_0)
    rho = wright_eos(T, S, p_cell)
    return z, J, T, S, rho, p_cell


def test_column_is_compressibility_masked():
    """Sanity: the fixture column is in-situ-STABLE but adiabatic-UNSTABLE."""
    z, J, T, S, rho, p_cell = _masked_unstable_column()
    from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic

    n2_insitu = compute_buoyancy_frequency(rho, z.dz_ref, J)
    n2_adia = compute_buoyancy_frequency_adiabatic(T, S, p_cell, z.dz_ref, J)
    assert bool(jnp.all(n2_insitu > 0.0))   # in-situ: masked stable
    assert bool(jnp.all(n2_adia < 0.0))     # adiabatic: truly unstable


# ---------------------------------------------------------------------------
# (a) Richardson: adiabatic mixes more than in-situ on the masked column.
# ---------------------------------------------------------------------------
def test_richardson_adiabatic_mixes_more_than_insitu():
    z, J, T, S, rho, p_cell = _masked_unstable_column()
    u = jnp.zeros_like(T)
    v = jnp.zeros_like(T)

    cfg_insitu = RichardsonVerticalMixingConfig()  # default n2_mode="insitu"
    cfg_adia = cfg_insitu._replace(n2_mode="adiabatic")

    out_insitu = richardson_vertical_mixing(
        u, v, T, S, rho, z, J, cfg_insitu, apply_diffusion=False,
    )
    out_adia = richardson_vertical_mixing(
        u, v, T, S, rho, z, J, cfg_adia, apply_diffusion=False,
        p_cell=p_cell,
    )
    # In-situ reads stable -> Ri >> 0 -> K_v/A_v pinned near background.
    # Adiabatic reads unstable (N^2<0 -> Ri clipped to 0) -> K_v/A_v jump to
    # the PP81 maximum (K_0 + backgrounds).
    assert bool(jnp.all(out_adia.K_v > out_insitu.K_v))
    assert bool(jnp.all(out_adia.A_v > out_insitu.A_v))
    # Adiabatic hits the unstable ceiling: A_v -> K_0 + A_bg.
    assert jnp.allclose(out_adia.A_v, cfg_adia.K_0 + cfg_adia.A_bg)


# ---------------------------------------------------------------------------
# (b) Enhanced diffusion: adiabatic FIRES where in-situ does not.
# ---------------------------------------------------------------------------
def test_enhanced_diffusion_adiabatic_trigger_fires():
    z, J, T, S, rho, p_cell = _masked_unstable_column()
    cfg_insitu = EnhancedDiffusionConfig(
        K_conv=1.0, K_bg=1e-5, smooth_transition=False,
    )  # default n2_mode="insitu"
    cfg_adia = cfg_insitu._replace(n2_mode="adiabatic")

    K_i, _, flag_i = convective_K_A_flag(rho, z.dz_ref, J, cfg_insitu)
    K_a, _, flag_a = convective_K_A_flag(
        rho, z.dz_ref, J, cfg_adia, T=T, S=S, p_cell=p_cell,
    )
    # In-situ: masked stable -> no convection.
    assert jnp.allclose(flag_i, 0.0)
    assert jnp.allclose(K_i, cfg_insitu.K_bg)
    # Adiabatic: truly unstable -> convective trigger fires everywhere.
    assert jnp.allclose(flag_a, 1.0)
    assert jnp.allclose(K_a, cfg_adia.K_conv)


def test_enhanced_diffusion_convection_threads_pcell_end_to_end():
    """The public kernel forwards p_cell/eos_fn to the adiabatic trigger."""
    z, J, T, S, rho, p_cell = _masked_unstable_column()
    cfg = EnhancedDiffusionConfig(
        K_conv=1.0, K_bg=1e-5, smooth_transition=False, n2_mode="adiabatic",
    )
    out = enhanced_diffusion_convection(
        T, S, rho, z, J, cfg, apply_diffusion=False, p_cell=p_cell,
    )
    assert jnp.allclose(out.convection_flag, 1.0)
    assert jnp.allclose(out.K_v, cfg.K_conv)


# ---------------------------------------------------------------------------
# (c) Unknown n2_mode raises ValueError at function entry (both schemes).
# ---------------------------------------------------------------------------
def test_richardson_unknown_n2_mode_raises():
    z, J, T, S, rho, p_cell = _masked_unstable_column()
    u = jnp.zeros_like(T)
    cfg = RichardsonVerticalMixingConfig()._replace(n2_mode="bogus")
    with pytest.raises(ValueError, match="n2_mode"):
        richardson_vertical_mixing(u, u, T, S, rho, z, J, cfg,
                                   apply_diffusion=False)


def test_enhanced_diffusion_unknown_n2_mode_raises():
    z, J, T, S, rho, p_cell = _masked_unstable_column()
    cfg = EnhancedDiffusionConfig(n2_mode="bogus")
    with pytest.raises(ValueError, match="n2_mode"):
        convective_K_A_flag(rho, z.dz_ref, J, cfg)


def test_adiabatic_without_pcell_raises():
    """Selecting adiabatic but not threading p_cell must fail loudly."""
    z, J, T, S, rho, p_cell = _masked_unstable_column()
    u = jnp.zeros_like(T)
    rcfg = RichardsonVerticalMixingConfig()._replace(n2_mode="adiabatic")
    with pytest.raises(ValueError, match="p_cell"):
        richardson_vertical_mixing(u, u, T, S, rho, z, J, rcfg,
                                   apply_diffusion=False)
    ecfg = EnhancedDiffusionConfig(n2_mode="adiabatic")
    with pytest.raises(ValueError, match="p_cell"):
        convective_K_A_flag(rho, z.dz_ref, J, ecfg, T=T, S=S)  # p_cell missing


# ---------------------------------------------------------------------------
# (d) Default config is "insitu" and reproduces the legacy N^2 math exactly.
# ---------------------------------------------------------------------------
def test_default_is_insitu_and_bit_identical_to_legacy():
    # Defaults are opt-out-safe.
    assert RichardsonVerticalMixingConfig().n2_mode == "insitu"
    assert EnhancedDiffusionConfig().n2_mode == "insitu"

    z, J, T, S, rho, p_cell = _masked_unstable_column()
    u = jnp.zeros_like(T)
    v = jnp.zeros_like(T)
    cfg = RichardsonVerticalMixingConfig()

    out = richardson_vertical_mixing(u, v, T, S, rho, z, J, cfg,
                                     apply_diffusion=False)

    # Independent recomputation of the LEGACY path (in-situ N^2 -> PP81 K/A).
    N2 = compute_buoyancy_frequency(rho, z.dz_ref, J)
    dz_actual = z.dz_ref * J[..., jnp.newaxis]
    Ri = richardson_number(N2, u, v, dz_actual, eps=_EPS, clip_negative=True)
    one_plus_aRi = 1.0 + cfg.alpha * Ri
    A_v_ref = cfg.K_0 / one_plus_aRi ** cfg.n + cfg.A_bg
    K_v_ref = A_v_ref / one_plus_aRi + cfg.K_bg

    assert jnp.array_equal(out.A_v, A_v_ref)
    assert jnp.array_equal(out.K_v, K_v_ref)

    # Enhanced-diffusion default (in-situ) also matches the legacy trigger:
    # masked-stable column => no convection.
    ecfg = EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5,
                                   smooth_transition=False)
    K_i, _, flag_i = convective_K_A_flag(rho, z.dz_ref, J, ecfg)
    assert jnp.allclose(flag_i, 0.0)
    assert jnp.allclose(K_i, ecfg.K_bg)

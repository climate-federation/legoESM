"""Deep-ocean ventilation fix — direct unit tests.

Covers the multi-part fix that closes the legoESM-Veros ACC abyssal warm bias
(the abyss never convected because the static-stability N² was the in-situ
density difference, biased too stable and never negative):

1. Shared adiabatic-displacement N² helper
   (:func:`legoesm.ocean.eos.compute_buoyancy_frequency_adiabatic`) — SIGNED,
   goes negative for a statically unstable column; agrees with the in-situ
   form for an incompressible (linear) EOS.
2. TKE ``n2_mode="adiabatic"`` — detects instability and convects (K_M
   saturates toward kappaM_max in unstable columns); ``"insitu"`` default is
   BIT-IDENTICAL to the legacy path.
3. Prandtl chain ``prandtl_mode in {"unit","constant","richardson"}`` — abyssal
   K_H drops below the legacy ``max(K_M, kappaH_min)``; ``"unit"`` default
   BIT-IDENTICAL.
4. Bryan-Lewis ``enable_kappaH_profile`` arctan floor on K_H.
5. Linear IC (``rest_state_latlon_cgrid_ocean(stratification="linear")``) —
   matches the Veros ACC ``(1 - z/z_bottom)*15`` profile; default exponential
   BIT-IDENTICAL.
6. Dispatch: unknown literals raise ``ValueError``.
7. Differentiability: ``jax.grad`` flows through the adiabatic N² + convective
   K closure.
8. The ACC recipe opts every piece in.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    compute_buoyancy_frequency,
    compute_buoyancy_frequency_adiabatic,
    linear_eos,
    make_eos_fn,
    wright_eos,
)
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    _bryan_lewis_kappaH_floor,
    _compute_N2,
    _prandtl_number,
    _veros_buoyancy_length,
    compute_K_from_tke,
    compute_mixing_lengths,
    tke_vertical_mixing,
)

jax.config.update("jax_enable_x64", True)

RHO0 = 1024.0
G = 9.81


# ---------------------------------------------------------------------------
# Helpers: a small column with a known statically-unstable interface.
# ---------------------------------------------------------------------------


def _column(nlev=6, unstable=False):
    """Single-column (...,nlev) T/S/dz with an optionally unstable interface."""
    # z* style: surface k=0 warm, deep k=nlev-1 cold (stable). For the
    # "unstable" case, place denser (colder) water on top of one interface.
    dz_ref = jnp.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0])[:nlev]
    jacobian = jnp.array(1.0)
    T = jnp.linspace(15.0, 2.0, nlev)
    S = jnp.full((nlev,), 35.0)
    if unstable:
        # Make cell 2 colder than cell 3 (denser water above lighter) -> the
        # interface between 2 and 3 is statically unstable.
        T = T.at[2].set(0.0)
    return T, S, dz_ref, jacobian


# ---------------------------------------------------------------------------
# 1. Shared adiabatic-N² helper
# ---------------------------------------------------------------------------


def test_adiabatic_N2_signed_negative_for_unstable_column():
    """Adiabatic N² goes NEGATIVE for a statically unstable interface, where
    the in-situ form (which is also signed before clipping) may not."""
    T, S, dz_ref, J = _column(unstable=True)
    # cell-centre pressure: hydrostatic-ish (just needs to be a smooth p(z)).
    depth = jnp.cumsum(dz_ref) - 0.5 * dz_ref
    p_cell = RHO0 * G * depth
    N2 = compute_buoyancy_frequency_adiabatic(
        T, S, p_cell, dz_ref, J, eos_fn=wright_eos, rho_ref=RHO0, g=G)
    # Interface k=2 is the unstable one (cold cell 2 over warmer cell 3).
    assert N2[2] < 0.0, f"expected unstable interface N2<0, got {N2[2]}"
    # The stable interfaces remain positive.
    assert N2[0] > 0.0 and N2[1] > 0.0


def test_adiabatic_N2_matches_insitu_for_incompressible_eos():
    """With a pressure-INDEPENDENT (linear) EOS the adiabatic displacement is a
    no-op, so adiabatic N² == in-situ N² (both reference the same density)."""
    T, S, dz_ref, J = _column(unstable=False)
    depth = jnp.cumsum(dz_ref) - 0.5 * dz_ref
    p_cell = RHO0 * G * depth

    def lin(Tt, Ss, pp):
        return linear_eos(Tt, Ss, pp, rho_ref=RHO0, alpha_T=2e-4,
                          beta_S=7.6e-4, T_ref=10.0, S_ref=35.0)

    rho = lin(T, S, p_cell)
    N2_insitu = compute_buoyancy_frequency(rho, dz_ref, J, rho_ref=RHO0, g=G)
    N2_adiab = compute_buoyancy_frequency_adiabatic(
        T, S, p_cell, dz_ref, J, eos_fn=lin, rho_ref=RHO0, g=G)
    np.testing.assert_allclose(np.asarray(N2_adiab), np.asarray(N2_insitu),
                               rtol=1e-12, atol=1e-18)


def test_adiabatic_N2_less_stable_than_insitu_compressible():
    """For a compressible (Wright) EOS the in-situ N² is MORE stable than the
    adiabatic N² (the compressibility bias the fix removes)."""
    T, S, dz_ref, J = _column(unstable=False)
    depth = jnp.cumsum(dz_ref) - 0.5 * dz_ref
    p_cell = RHO0 * G * depth
    rho = wright_eos(T, S, p_cell)
    N2_insitu = compute_buoyancy_frequency(rho, dz_ref, J, rho_ref=RHO0, g=G)
    N2_adiab = compute_buoyancy_frequency_adiabatic(
        T, S, p_cell, dz_ref, J, eos_fn=wright_eos, rho_ref=RHO0, g=G)
    # In-situ includes compressibility -> larger N² (more stable).
    assert np.all(np.asarray(N2_insitu) >= np.asarray(N2_adiab) - 1e-12)
    assert np.mean(np.asarray(N2_insitu)) > np.mean(np.asarray(N2_adiab))


# ---------------------------------------------------------------------------
# 2. TKE n2_mode
# ---------------------------------------------------------------------------


def test_tke_compute_N2_insitu_is_clipped_nonnegative():
    """Default n2_mode='insitu' clips N² >= 0 (legacy)."""
    T, S, dz_ref, J = _column(unstable=True)
    dz_half = 0.5 * (dz_ref[:-1] + dz_ref[1:])
    rho = wright_eos(T, S, jnp.zeros_like(T))
    N2 = _compute_N2(rho, dz_half, RHO0, G, n2_mode="insitu")
    assert np.all(np.asarray(N2) >= 0.0)


def test_tke_compute_N2_adiabatic_signed():
    """n2_mode='adiabatic' returns a SIGNED N² (negative on unstable)."""
    T, S, dz_ref, J = _column(unstable=True)
    dz_half = 0.5 * (dz_ref[:-1] + dz_ref[1:])
    depth = jnp.cumsum(dz_ref) - 0.5 * dz_ref
    p_cell = RHO0 * G * depth
    rho = wright_eos(T, S, p_cell)
    N2 = _compute_N2(rho, dz_half, RHO0, G, T_cell=T, S_cell=S, p_cell=p_cell,
                     dz_ref=dz_ref, jacobian=J, eos_fn=wright_eos,
                     n2_mode="adiabatic")
    assert np.any(np.asarray(N2) < 0.0)


def test_tke_compute_N2_adiabatic_requires_inputs():
    T, S, dz_ref, J = _column()
    dz_half = 0.5 * (dz_ref[:-1] + dz_ref[1:])
    rho = wright_eos(T, S, jnp.zeros_like(T))
    with pytest.raises(ValueError, match="adiabatic"):
        _compute_N2(rho, dz_half, RHO0, G, n2_mode="adiabatic")


def test_tke_compute_N2_unknown_mode_raises():
    T, S, dz_ref, J = _column()
    dz_half = 0.5 * (dz_ref[:-1] + dz_ref[1:])
    rho = wright_eos(T, S, jnp.zeros_like(T))
    with pytest.raises(ValueError, match="Unknown n2_mode"):
        _compute_N2(rho, dz_half, RHO0, G, n2_mode="bogus")


def test_tke_convects_in_adiabatic_mode():
    """End-to-end: an unstable column under n2_mode='adiabatic' produces a
    large convective K_M (toward kappaM_max), unlike the legacy insitu path."""
    nlev = 6
    T, S, dz_ref, J = _column(nlev=nlev, unstable=True)
    dz_half = 0.5 * (dz_ref[:-1] + dz_ref[1:])
    depth = jnp.cumsum(dz_ref) - 0.5 * dz_ref
    p_cell = RHO0 * G * depth
    u = jnp.zeros((nlev,))
    v = jnp.zeros((nlev,))

    cfg_conv = TKEConfig(n2_mode="adiabatic", prandtl_mode="richardson")
    rho = wright_eos(T, S, p_cell)
    out_conv = tke_vertical_mixing(
        u, v, T, S, rho, dz_half, tke_old=None,
        tau_x_surface=None, tau_y_surface=None,
        dt=86400.0, cfg=cfg_conv, rho_0=RHO0, g=G, n_iterations=3,
        p_cell=p_cell, dz_ref=dz_ref, jacobian=J, eos_fn=wright_eos,
        z_interface=jnp.array([-30., -90., -210., -450., -930.])[:nlev - 1],
    )
    # The unstable interface (k=2) develops a large convective K_M.
    assert float(out_conv.K_M[2]) > 1e-2, (
        f"convective K_M too small: {out_conv.K_M[2]}")
    assert np.all(np.isfinite(np.asarray(out_conv.K_M)))
    # Capped by kappaM_max.
    assert np.all(np.asarray(out_conv.K_M) <= cfg_conv.kappaM_max + 1e-9)

    # Legacy insitu path on the SAME column does NOT see the instability
    # (no large convective K from the N² trigger).
    cfg_legacy = TKEConfig()  # insitu, unit
    out_legacy = tke_vertical_mixing(
        u, v, T, S, rho, dz_half, tke_old=None,
        tau_x_surface=None, tau_y_surface=None,
        dt=86400.0, cfg=cfg_legacy, rho_0=RHO0, g=G, n_iterations=3)
    assert float(out_legacy.K_M[2]) < float(out_conv.K_M[2])


def test_veros_buoyancy_length_blows_up_on_unstable():
    """The Veros buoyancy length is huge where N²<=0 (the convective trigger),
    far larger than the legacy 2-cell cap."""
    e = jnp.full((5,), 1e-3)
    dz_int = jnp.full((5,), 100.0)
    N2 = jnp.array([1e-4, 1e-4, -1e-7, 1e-4, 1e-4])  # one unstable interface
    l = _veros_buoyancy_length(e, N2, dz_int, mxl_min=1e-8)
    # Around the unstable interface the length is at least a cell thickness
    # (not collapsed to the legacy ~2*dz of the stable closed form).
    assert float(l[2]) > 50.0
    assert np.all(np.asarray(l) >= 1e-8)


# ---------------------------------------------------------------------------
# 3. Prandtl chain
# ---------------------------------------------------------------------------


def test_prandtl_richardson_high_in_stratified_interior():
    """Stratified interior (large Ri) -> Prandtl saturates at 10."""
    N2 = jnp.full((4,), 1e-4)
    shear = jnp.full((4,), 1e-8)   # tiny shear -> huge Ri
    K_M = jnp.full((4,), 1e-3)
    cfg = TKEConfig(prandtl_mode="richardson")
    Pr = _prandtl_number(N2, shear, K_M, cfg)
    np.testing.assert_allclose(np.asarray(Pr), 10.0, rtol=1e-12)


def test_prandtl_richardson_unit_in_convection():
    """Unstable (N²<0 -> Ri<0) -> Prandtl floors at 1 (K_H tracks K_M)."""
    N2 = jnp.full((4,), -1e-6)
    shear = jnp.full((4,), 1e-6)
    K_M = jnp.full((4,), 1e-1)
    cfg = TKEConfig(prandtl_mode="richardson")
    Pr = _prandtl_number(N2, shear, K_M, cfg)
    np.testing.assert_allclose(np.asarray(Pr), 1.0, rtol=1e-12)


def test_prandtl_constant_mode():
    N2 = jnp.full((4,), 1e-4)
    shear = jnp.full((4,), 1e-6)
    K_M = jnp.full((4,), 1e-3)
    cfg = TKEConfig(prandtl_mode="constant", Prandtl_tke0=10.0)
    Pr = _prandtl_number(N2, shear, K_M, cfg)
    np.testing.assert_allclose(np.asarray(Pr), 10.0, rtol=1e-12)


def test_prandtl_unknown_mode_raises():
    cfg = TKEConfig(prandtl_mode="nope")
    with pytest.raises(ValueError, match="Unknown prandtl_mode"):
        _prandtl_number(jnp.ones(3), jnp.ones(3), jnp.ones(3), cfg)


# ---------------------------------------------------------------------------
# T8 (Phase-2 #1317): "nemo_ri" — NEMO's EXACT zri=rn2b*avm/(sh2+bshear)
# form (zdftke.F90:381-401), distinct from "richardson" (Veros's own
# Ri=N2/shear_sq, missing the avm numerator factor).
# ---------------------------------------------------------------------------


def test_nemo_ri_matches_hand_derivation():
    """zri = N2*kappaM/(shear_sq+bshear_floor); Pr = clamp(coeff*zri,1,10)
    — re-derived by hand from the F90 line, not copy-pasted."""
    N2 = jnp.full((4,), 2e-5)
    shear = jnp.full((4,), 1e-6)
    K_M = jnp.full((4,), 3.0)   # kappaM factor NEMO's zri carries, "richardson" lacks
    cfg = TKEConfig(prandtl_mode="nemo_ri", prandtl_ri_coeff=4.5)
    Pr = _prandtl_number(N2, shear, K_M, cfg)
    zri = 2e-5 * 3.0 / (1e-6 + cfg.bshear_floor)
    expected = max(1.0, min(10.0, 4.5 * zri))
    np.testing.assert_allclose(np.asarray(Pr), expected, rtol=1e-10)


def test_nemo_ri_differs_from_richardson_via_kappaM_factor():
    """With kappaM != 1, "nemo_ri" and "richardson" diverge — the avm
    numerator factor is the fidelity content of T8. Values chosen so
    NEITHER mode saturates against the [1,10] clamp (which would mask the
    difference)."""
    N2 = jnp.full((4,), 4e-6)
    shear = jnp.full((4,), 1e-5)
    K_M = jnp.full((4,), 5.0)   # kappaM far from 1 -> the two modes MUST differ
    cfg_nemo = TKEConfig(prandtl_mode="nemo_ri", prandtl_ri_coeff=1.0)
    cfg_veros = TKEConfig(prandtl_mode="richardson", prandtl_ri_coeff=1.0)
    Pr_nemo = _prandtl_number(N2, shear, K_M, cfg_nemo)
    Pr_veros = _prandtl_number(N2, shear, K_M, cfg_veros)
    assert not np.allclose(np.asarray(Pr_nemo), np.asarray(Pr_veros))
    np.testing.assert_allclose(np.asarray(Pr_veros), 1.0, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(Pr_nemo), 2.0, rtol=1e-6)


def test_nemo_ri_saturates_at_10_in_stratified_interior():
    N2 = jnp.full((4,), 1e-4)
    shear = jnp.full((4,), 1e-8)
    K_M = jnp.full((4,), 1e-3)
    cfg = TKEConfig(prandtl_mode="nemo_ri")
    Pr = _prandtl_number(N2, shear, K_M, cfg)
    np.testing.assert_allclose(np.asarray(Pr), 10.0, rtol=1e-12)


def test_nemo_ri_floors_at_1_in_convection():
    """Unstable (N²<0) -> zri<0 -> Pr floors at 1 (K_H tracks K_M)."""
    N2 = jnp.full((4,), -1e-6)
    shear = jnp.full((4,), 1e-6)
    K_M = jnp.full((4,), 1e-1)
    cfg = TKEConfig(prandtl_mode="nemo_ri")
    Pr = _prandtl_number(N2, shear, K_M, cfg)
    np.testing.assert_allclose(np.asarray(Pr), 1.0, rtol=1e-12)


def test_prandtl_chain_drops_abyssal_KH():
    """In a stratified interior the richardson Prandtl chain gives K_H ~ K_M/10,
    far below the legacy K_H = max(K_M, kappaH_min)."""
    e = jnp.full((4,), 1e-4)
    l_k = jnp.full((4,), 50.0)
    N2 = jnp.full((4,), 1e-4)
    shear = jnp.full((4,), 1e-8)   # huge Ri -> Pr=10
    z_int = jnp.array([-100., -300., -600., -1000.])
    cfg_r = TKEConfig(prandtl_mode="richardson", enable_kappaH_profile=False)
    _, K_H_r = compute_K_from_tke(e, l_k, cfg_r, N2=N2, shear_sq=shear,
                                  z_interface=z_int)
    cfg_u = TKEConfig()  # unit
    K_M_u, K_H_u = compute_K_from_tke(e, l_k, cfg_u)
    # richardson K_H is ~K_M/10, strictly below the unit-path K_H = K_M.
    assert np.all(np.asarray(K_H_r) < np.asarray(K_H_u))


# ---------------------------------------------------------------------------
# 4. Bryan-Lewis
# ---------------------------------------------------------------------------


def test_bryan_lewis_floor_shape_and_range():
    z = jnp.array([-100.0, -1000.0, -2500.0, -4000.0])
    # The Bryan-Lewis fit coefficients moved to TKEConfig (#518 item 10,
    # ee0ad0c38); the formula and defaults are unchanged so the default cfg
    # reproduces the published profile (bg_diff_amp=0.8, bg_diff_scale=1e-4).
    floor = _bryan_lewis_kappaH_floor(z, TKEConfig())
    # At ~2500 m the arctan argument is 0 -> 0.8e-4. Deep -> larger; shallow
    # -> smaller. All within a sensible O(1e-4) band.
    assert np.all(np.asarray(floor) > 0.0)
    np.testing.assert_allclose(float(floor[2]), 0.8e-4, rtol=1e-9)
    assert float(floor[3]) > float(floor[0])  # deeper has larger floor


def test_bryan_lewis_raises_abyssal_KH():
    """enable_kappaH_profile lifts K_H above kappaH_min in the abyss.

    Use a quiescent abyssal interface (tiny TKE + mixing length) so K_M/Pr
    falls to ~kappaH_min and the Bryan-Lewis depth floor (~4e-5 below 2 km)
    dominates."""
    cfg_base = TKEConfig()
    e = jnp.full((3,), cfg_base.tke_background)
    l_k = jnp.full((3,), cfg_base.mxl_min)
    N2 = jnp.full((3,), 1e-4)
    shear = jnp.full((3,), 1e-8)   # Pr=10 -> K_H from K_M tiny -> floor dominates
    z_int = jnp.array([-1500.0, -1800.0, -2050.0])
    cfg_on = TKEConfig(prandtl_mode="richardson", enable_kappaH_profile=True,
                       kappaH_min=2e-5)
    _, K_H_on = compute_K_from_tke(e, l_k, cfg_on, N2=N2, shear_sq=shear,
                                   z_interface=z_int)
    cfg_off = TKEConfig(prandtl_mode="richardson", enable_kappaH_profile=False,
                        kappaH_min=2e-5)
    _, K_H_off = compute_K_from_tke(e, l_k, cfg_off, N2=N2, shear_sq=shear,
                                    z_interface=z_int)
    assert np.all(np.asarray(K_H_on) >= np.asarray(K_H_off) - 1e-18)
    assert np.any(np.asarray(K_H_on) > np.asarray(K_H_off))


# ---------------------------------------------------------------------------
# 5. BIT-IDENTICAL default regression (THE hard constraint)
# ---------------------------------------------------------------------------


def _default_tke_inputs(nlev=7):
    rng = np.random.default_rng(0)
    leading = (3, 4)
    u = jnp.asarray(rng.standard_normal(leading + (nlev,)))
    v = jnp.asarray(rng.standard_normal(leading + (nlev,)))
    T = jnp.asarray(15.0 - np.cumsum(np.abs(rng.standard_normal(leading + (nlev,))), axis=-1))
    S = jnp.full(leading + (nlev,), 35.0)
    dz_ref = jnp.array([20., 40., 80., 120., 160., 200., 240.])[:nlev]
    J = jnp.ones(leading)
    dz_half = jnp.broadcast_to(0.5 * (dz_ref[:-1] + dz_ref[1:]),
                               leading + (nlev - 1,))
    rho = wright_eos(T, S, jnp.zeros_like(T))
    return u, v, T, S, rho, dz_half


def test_default_tke_orchestrator_bit_identical():
    """The DEFAULT TKEConfig (n2_mode='insitu', prandtl_mode='unit') produces
    BYTE-IDENTICAL K_M/K_H/tke to a frozen reference computed by the legacy
    code path (re-derived here to lock byte equality)."""
    u, v, T, S, rho, dz_half = _default_tke_inputs()
    cfg = TKEConfig()  # all defaults

    out = tke_vertical_mixing(
        u, v, T, S, rho, dz_half, tke_old=None,
        tau_x_surface=jnp.ones(u.shape[:-1]) * 0.05,
        tau_y_surface=jnp.zeros(u.shape[:-1]),
        dt=3600.0, cfg=cfg, rho_0=RHO0, g=G, n_iterations=3)

    # Re-derive the legacy result directly from the building blocks WITHOUT
    # any of the new optional args -> the byte-reference for the default path.
    from legoesm.ocean.physics.vertical_mixing.tke import (
        _vertical_shear_squared, _solve_tke_backward_euler)
    shear = _vertical_shear_squared(u, v, dz_half)
    N2 = _compute_N2(rho, dz_half, RHO0, G)  # legacy clipped insitu
    surf = (jnp.sqrt((jnp.ones(u.shape[:-1]) * 0.05) ** 2) / RHO0) ** 1.5
    e = jnp.full(u.shape[:-1] + (u.shape[-1] - 1,), cfg.tke_background)
    for _ in range(3):
        l_k, l_eps = compute_mixing_lengths(e, N2, dz_half, cfg)
        K_M, K_H = compute_K_from_tke(e, l_k, cfg)
        e = _solve_tke_backward_euler(
            e_old=e, K_M_old=K_M, K_H_old=K_H, P_s=K_M * shear, N2=N2,
            l_eps=l_eps, dz_half=dz_half, surface_flux=surf, dt=3600.0, cfg=cfg)
    l_kf, _ = compute_mixing_lengths(e, N2, dz_half, cfg)
    K_M_ref, K_H_ref = compute_K_from_tke(e, l_kf, cfg)

    assert np.array_equal(np.asarray(out.K_M), np.asarray(K_M_ref))
    assert np.array_equal(np.asarray(out.K_H), np.asarray(K_H_ref))
    assert np.array_equal(np.asarray(out.tke_new), np.asarray(e))


def test_default_compute_K_from_tke_bit_identical():
    """compute_K_from_tke with no optional args == the historical formula."""
    e = jnp.asarray(np.linspace(1e-5, 1e-2, 12)).reshape(3, 4)
    l_k = jnp.asarray(np.linspace(1.0, 100.0, 12)).reshape(3, 4)
    cfg = TKEConfig()
    K_M, K_H = compute_K_from_tke(e, l_k, cfg)
    # Historical: K_M = c_k*l_k*sqrt(2e), floor kappaM_min; K_H=max(K_M,kappaH_min)
    K_M_ref = cfg.c_k * l_k * jnp.sqrt(2.0 * jnp.maximum(e, cfg.tke_background))
    K_M_ref = jnp.maximum(K_M_ref, cfg.kappaM_min)
    K_H_ref = jnp.maximum(K_M_ref, cfg.kappaH_min)
    assert np.array_equal(np.asarray(K_M), np.asarray(K_M_ref))
    assert np.array_equal(np.asarray(K_H), np.asarray(K_H_ref))


def test_default_compute_N2_bit_identical():
    """_compute_N2 default == the historical clipped in-situ formula."""
    _, _, T, S, rho, dz_half = _default_tke_inputs()
    N2 = _compute_N2(rho, dz_half, RHO0, G)
    dz_safe = jnp.maximum(dz_half, float(jnp.finfo(jnp.float32).eps))
    drho_dz = (rho[..., :-1] - rho[..., 1:]) / dz_safe
    N2_ref = jnp.maximum(-G / RHO0 * drho_dz, 0.0)
    assert np.array_equal(np.asarray(N2), np.asarray(N2_ref))


# ---------------------------------------------------------------------------
# 6. Linear IC
# ---------------------------------------------------------------------------


def _small_grid_and_z():
    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    grid, _ = create_regional_latlon_grid(
        n_lat=4, n_lon=4, lat_south=-10.0, lat_north=10.0,
        lon_west=0.0, lon_east=40.0, periodic_x=True)
    z = create_ocean_z_star(n_levels=5, H_max=1000.0)
    return grid, z


def test_linear_ic_matches_veros_profile():
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    grid, z = _small_grid_and_z()
    st = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=15.0, T_deep=0.0, S_uniform=35.0,
        H_max=1000.0, stratification="linear")
    T = np.asarray(st.T.data)
    zc = np.asarray(z.z_full_ref)
    zb = float(np.asarray(z.z_half_ref)[-1])
    T_veros = (1.0 - zc / zb) * 15.0   # Veros ACC analytic
    # Take any column (uniform horizontally). float32 storage policy.
    np.testing.assert_allclose(T[0, 0, :], T_veros, rtol=1e-5, atol=1e-5)


def test_linear_ic_deepest_cooler_than_exponential():
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    grid, z = _small_grid_and_z()
    lin = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=15.0, T_deep=0.0, H_max=1000.0,
        stratification="linear")
    exp = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=15.0, T_deep=0.0, H_max=1000.0,
        stratification="exponential")
    assert float(lin.T.data[0, 0, -1]) < float(exp.T.data[0, 0, -1])


def test_exponential_ic_bit_identical_default():
    """The default (exponential) IC is unchanged byte-for-byte. The reference
    uses the IDENTICAL jnp expression + storage cast as the production code."""
    from legoesm.core.precision import get_policy
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
    grid, z = _small_grid_and_z()
    st = rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=20.0, T_deep=2.0, H_max=1000.0)
    T = np.asarray(st.T.data)
    # Byte-reference: the exact pre-change expression.
    T_profile = 2.0 + (20.0 - 2.0) * jnp.exp(z.z_full_ref / _SCALE_DEPTH)
    dtype = get_policy().storage
    n_lat, n_lon, nlev = T.shape
    T_ref = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev)).astype(dtype)
    np.testing.assert_array_equal(np.asarray(T), np.asarray(T_ref))


def test_linear_ic_unknown_stratification_raises():
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    grid, z = _small_grid_and_z()
    with pytest.raises(ValueError, match="Unknown stratification"):
        rest_state_latlon_cgrid_ocean(grid, z, stratification="parabolic")


# ---------------------------------------------------------------------------
# 7. Differentiability (jax.grad through the adiabatic-N² convective closure)
# ---------------------------------------------------------------------------


def test_adiabatic_N2_grad():
    """jax.grad of a scalar of the adiabatic N² wrt T is finite + nonzero."""
    T, S, dz_ref, J = _column(unstable=True)
    depth = jnp.cumsum(dz_ref) - 0.5 * dz_ref
    p_cell = RHO0 * G * depth

    def loss(Tin):
        N2 = compute_buoyancy_frequency_adiabatic(
            Tin, S, p_cell, dz_ref, J, eos_fn=wright_eos, rho_ref=RHO0, g=G)
        return jnp.sum(N2 ** 2)

    g = jax.grad(loss)(T)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.abs(np.asarray(g)) > 0)


def test_veros_buoyancy_length_grad_safe():
    """jax.grad through the Veros buoyancy length (its two fori_loop sweeps
    must use STATIC bounds; dynamic bounds break reverse-mode AD)."""
    dz_int = jnp.full((6,), 100.0)
    N2 = jnp.array([1e-4, 1e-4, -1e-7, 1e-4, -2e-7, 1e-4])

    def loss(e):
        return jnp.sum(_veros_buoyancy_length(e, N2, dz_int, mxl_min=1e-8))

    g = jax.grad(loss)(jnp.full((6,), 1e-3))
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.abs(np.asarray(g)) > 0)


def test_tke_adiabatic_convection_grad():
    """jax.grad through the full adiabatic-mode TKE convective K wrt T."""
    nlev = 6
    T, S, dz_ref, J = _column(nlev=nlev, unstable=True)
    dz_half = 0.5 * (dz_ref[:-1] + dz_ref[1:])
    depth = jnp.cumsum(dz_ref) - 0.5 * dz_ref
    p_cell = RHO0 * G * depth
    u = jnp.zeros((nlev,))
    v = jnp.zeros((nlev,))
    cfg = TKEConfig(n2_mode="adiabatic", prandtl_mode="richardson")
    z_int = jnp.array([-30., -90., -210., -450., -930.])[:nlev - 1]

    def loss(Tin):
        rho = wright_eos(Tin, S, p_cell)
        out = tke_vertical_mixing(
            u, v, Tin, S, rho, dz_half, tke_old=None,
            tau_x_surface=None, tau_y_surface=None,
            dt=86400.0, cfg=cfg, rho_0=RHO0, g=G, n_iterations=3,
            p_cell=p_cell, dz_ref=dz_ref, jacobian=J, eos_fn=wright_eos,
            z_interface=z_int)
        return jnp.sum(out.K_H) + jnp.sum(out.K_M)

    g = jax.grad(loss)(T)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.abs(np.asarray(g)) > 0)


# ---------------------------------------------------------------------------
# 8. The ACC recipe opts every piece in
# ---------------------------------------------------------------------------


def test_acc_recipe_opts_in_deep_t_fixes():
    from legoesm.ocean.fidelity.veros_acc_recipe import (
        ACC_TKE_CONFIG, build_acc_model_config, build_acc_recipe)
    assert ACC_TKE_CONFIG.n2_mode == "adiabatic"
    assert ACC_TKE_CONFIG.prandtl_mode == "richardson"
    assert ACC_TKE_CONFIG.enable_kappaH_profile is True
    mc = build_acc_model_config()
    assert float(mc.K_v) == 0.0   # dropped background tracer diffusivity
    # The recipe IC uses the linear stratification (deepest cell ~Veros).
    rec = build_acc_recipe()
    T = np.asarray(rec.initial_state.T.data)
    lm = np.asarray(rec.initial_state.land_mask.data)
    zc = np.asarray(rec.z_coord.z_full_ref)
    # The recipe IC is the LITERAL Veros ACC profile temp = (1 - zt/zw[0])*15
    # (acc.py:117), where Veros's bottom-first zw[0] is the TOP face of the
    # BOTTOM cell — legoESM (surface-first) z_half_ref[-2], NOT the bottom
    # interface -H_max (z_half_ref[-1]). Normalising by -H_max was the earlier
    # "linear" transcription that started the abyss +2.15 K warm (fixed in
    # 32e1dec41); the test must use zw0 = z_half_ref[-2] to match the model.
    zw0 = float(np.asarray(rec.z_coord.z_half_ref)[-2])
    T_veros = (1.0 - zc / zw0) * 15.0
    wet = np.argwhere(lm > 0.5)
    i, j = wet[len(wet) // 2]
    np.testing.assert_allclose(T[i, j, :], T_veros, rtol=1e-5, atol=1e-5)

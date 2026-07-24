"""TKEConfig.n2_before_advection: source the diffusivity-stage N^2 from the
BEFORE-advection (Nnow) T/S, matching NEMO's ``eosbn2`` step ordering.

BUG (BOTTOM_N2_DIAGNOSIS_FINDINGS.md): the TKE diffusivity-stage N^2 was
sampled on the POST-advection mid-step T/S. A single-step ``fct2`` tracer
drift at the deepest wet cell flipped the marginal deep interface to N^2 < 0,
firing spurious deep convection. NEMO samples ``bn2(Nnow)`` at step start,
BEFORE ``tra_adv``.

FIX: an opt-in ``TKEConfig.n2_before_advection`` flag. When set, the model
threads the step-entry T/S into the closure as an N^2 source override
(``T_n2``/``S_n2``), consumed ONLY by the adiabatic static-stability N^2
(``T_cell``/``S_cell`` are used nowhere else in the TKE orchestrator). Default
False ⇒ byte-identical legacy behaviour.

This module pins, on a synthetic column where advection tips the TOP interface
into marginal instability:
  (a) the override routes EXACTLY to the N^2 source — passing the before-state
      via ``T_n2`` is bit-identical to passing it as ``T_cell`` (K_M/K_H equal);
  (b) the override actually CHANGES the closure — default (after-state N^2) vs
      override (before-state N^2) differ at the tipped interface;
  (c) the physical direction — the after-state (unstable) N^2 gives LARGER
      convective K_H at that interface than the stable before-state N^2;
  (d) the model helper only arms when scheme=='tke' + the flag is set, and
      raises when the flag is set with a non-adiabatic n2_mode (dispatch
      hardening).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.eos import (
    compute_hydrostatic_pressure,
    rho_0,
    wright_eos,
)
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
from legoesm.ocean.vertical import create_ocean_z_star

G = 9.80665


def _column(top_T: float):
    """Stably-stratified 4-level column (warm/light on top) with a tunable
    TOP cell temperature. ``top_T`` well above the level below ⇒ stable at the
    top interface; ``top_T`` at/below it ⇒ marginally unstable there.

    Returns ``(z, J, T, S, rho, p_cell, dz_half)`` with a (1, 1) footprint.
    """
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    horiz = (1, 1)
    J = jnp.ones(horiz, dtype=jnp.float64)
    shape = horiz + (z.n_levels,)
    # Stable base column: T decreasing with depth; uniform S.
    T = jnp.broadcast_to(
        jnp.array([top_T, 10.0, 6.0, 4.0], dtype=jnp.float64), shape,
    ).copy()
    S = jnp.broadcast_to(jnp.full(z.n_levels, 35.0, dtype=jnp.float64), shape)
    eta = jnp.zeros(horiz, dtype=jnp.float64)
    rho = wright_eos(T, S, jnp.zeros_like(T))
    p_cell = compute_hydrostatic_pressure(rho, eta, z.dz_ref, J, rho_0)
    rho = wright_eos(T, S, p_cell)
    p_cell = compute_hydrostatic_pressure(rho, eta, z.dz_ref, J, rho_0)
    rho = wright_eos(T, S, p_cell)
    dz_half = z.dz_half_ref * J[..., jnp.newaxis]
    dz_half = jnp.broadcast_to(dz_half, shape[:-1] + (z.n_levels - 1,))
    return z, J, T, S, rho, p_cell, dz_half


def _run(T_cell, S_cell, rho, p_cell, dz_half, z, J, *, T_n2=None, S_n2=None):
    cfg = TKEConfig(n2_mode="adiabatic")
    tke_old = jnp.full(
        rho.shape[:-1] + (z.n_levels - 1,), cfg.tke_background,
        dtype=rho.dtype,
    )
    zeros = jnp.zeros_like(rho)
    return tke_vertical_mixing(
        zeros, zeros, T_cell, S_cell, rho, dz_half,
        tke_old=tke_old, tau_x_surface=None, tau_y_surface=None,
        dt=3600.0, cfg=cfg, rho_0=rho_0, g=G, n_iterations=1,
        p_cell=p_cell, dz_ref=z.dz_ref, jacobian=J, eos_fn=wright_eos,
        z_interface=z.z_half_ref[1:-1],
        T_n2=T_n2, S_n2=S_n2,
    )


def test_override_routes_exactly_to_n2_source():
    """Passing the before-state via ``T_n2`` == passing it as ``T_cell``.

    ``T_cell``/``S_cell`` feed ONLY the adiabatic N^2, so the override must be
    bit-identical to sourcing N^2 from that state directly (everything else —
    rho, p_cell, shear — held fixed)."""
    z, J, bT, bS, _, _, dz_half = _column(top_T=14.0)     # stable before-state
    _, _, aT, aS, arho, ap, _ = _column(top_T=5.5)        # tipped after-state

    # Everything (rho, p_cell) held to the after-state; only the N^2 T/S vary.
    r_ref = _run(bT, bS, arho, ap, dz_half, z, J)                  # N^2 from before
    r_ovr = _run(aT, aS, arho, ap, dz_half, z, J, T_n2=bT, S_n2=bS)  # override

    np.testing.assert_array_equal(np.asarray(r_ovr.K_H), np.asarray(r_ref.K_H))
    np.testing.assert_array_equal(np.asarray(r_ovr.K_M), np.asarray(r_ref.K_M))


def test_set_diffusivities_routes_exactly_to_n2_source():
    """The NEMO recipe's actual path (buoyancy_timing='post_mixing_veros' ⇒
    tke_set_diffusivities) reroutes N² the same way: the override is
    bit-identical to sourcing N² from that state directly, and differs from
    the post-advection default."""
    from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config
    from legoesm.ocean.physics.vertical_mixing.tke import tke_set_diffusivities

    cfg = _nemo_tke_config()  # prognostic, adiabatic, post_mixing, veros slots
    z, J, bT, bS, _, _, dz_half = _column(top_T=14.0)     # stable before-state
    _, _, aT, aS, arho, ap, _ = _column(top_T=5.5)        # tipped after-state
    tke_old = jnp.full(
        arho.shape[:-1] + (z.n_levels - 1,), cfg.tke_background,
        dtype=arho.dtype)
    zeros = jnp.zeros_like(arho)
    dz_surface = (-z.z_full_ref[0]) * J

    def _sd(T_cell, S_cell, *, T_n2=None, S_n2=None):
        return tke_set_diffusivities(
            zeros, zeros, T_cell, S_cell, arho, dz_half,
            tke_old, None, None, cfg, rho_0, G,
            p_cell=ap, dz_ref=z.dz_ref, jacobian=J, eos_fn=wright_eos,
            z_interface=z.z_half_ref[1:-1], dz_surface=dz_surface,
            T_n2=T_n2, S_n2=S_n2)

    ref = _sd(bT, bS)                       # N² from before-state directly
    ovr = _sd(aT, aS, T_n2=bT, S_n2=bS)     # override reroutes to before-state
    default = _sd(aT, aS)                    # legacy: N² from after-state

    np.testing.assert_array_equal(np.asarray(ovr[1]), np.asarray(ref[1]))  # K_H
    np.testing.assert_array_equal(np.asarray(ovr[0]), np.asarray(ref[0]))  # K_M
    # and the reroute genuinely changes the answer vs the post-advection default
    assert not np.array_equal(np.asarray(ovr[1]), np.asarray(default[1]))


def test_flag_changes_closure_and_direction():
    """Default (after-state N^2) fires convection at the tipped top interface;
    the before-state override does NOT — a larger K_H with the flag off."""
    z, J, bT, bS, _, _, dz_half = _column(top_T=14.0)
    _, _, aT, aS, arho, ap, _ = _column(top_T=5.5)

    r_default = _run(aT, aS, arho, ap, dz_half, z, J)                 # flag OFF
    r_override = _run(aT, aS, arho, ap, dz_half, z, J, T_n2=bT, S_n2=bS)  # ON

    kt = 0  # top interior interface (between cells 0 and 1)
    kh_off = float(np.asarray(r_default.K_H)[0, 0, kt])
    kh_on = float(np.asarray(r_override.K_H)[0, 0, kt])
    # The flag must change the answer AND in the physically-correct direction:
    # after-state unstable N^2 -> convective mixing-length blowup -> larger K_H.
    assert kh_off > kh_on
    assert kh_off > 10.0 * kh_on


def test_default_is_byte_identical():
    """No override (T_n2 = S_n2 = None) reproduces the legacy N^2 exactly."""
    z, J, aT, aS, arho, ap, dz_half = _column(top_T=5.5)
    r_none = _run(aT, aS, arho, ap, dz_half, z, J)
    r_explicit_after = _run(aT, aS, arho, ap, dz_half, z, J, T_n2=aT, S_n2=aS)
    np.testing.assert_array_equal(
        np.asarray(r_none.K_H), np.asarray(r_explicit_after.K_H))


def test_model_helper_arming_and_guard():
    """``_n2_before_advection_tracers`` arms only for tke + flag; raises on a
    non-adiabatic n2_mode with the flag set (silent-no-op guard)."""
    # Build the helper as an unbound method on a lightweight stand-in: the
    # helper reads only self.config, so a tiny namespace suffices.
    from types import SimpleNamespace

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    def make(scheme, flag, n2_mode="adiabatic"):
        vm = SimpleNamespace(
            scheme=scheme,
            tke=TKEConfig(n2_mode=n2_mode, n2_before_advection=flag),
        )
        return SimpleNamespace(
            config=SimpleNamespace(physics=SimpleNamespace(vertical_mixing=vm)))

    helper = LatLonCGridOceanModel._n2_before_advection_tracers

    class _F:
        def __init__(self, d):
            self.data = d

    class _St:
        T = _F(jnp.ones((1, 1, 3)))
        S = _F(jnp.full((1, 1, 3), 35.0))

    # Off by default / wrong scheme -> None.
    assert helper(make("tke", False), _St()) is None
    assert helper(make("constant", True), _St()) is None
    # Armed -> returns the entry-state T/S.
    out = helper(make("tke", True), _St())
    assert out is not None and out[0].shape == (1, 1, 3)
    # Armed with the NEMO card's actual N² source (nemo_bn2) -> also returns
    # the entry-state T/S (the guard must accept BOTH T/S-reading N² modes;
    # nemo_dino_kamm sets tke_n2_mode="nemo_bn2", not "adiabatic").
    out_bn2 = helper(make("tke", True, n2_mode="nemo_bn2"), _St())
    assert out_bn2 is not None and out_bn2[0].shape == (1, 1, 3)
    # Flag set with a non-T/S-reading n2_mode -> loud failure (no silent no-op).
    with pytest.raises(ValueError, match="n2_mode='adiabatic' or 'nemo_bn2'"):
        helper(make("tke", True, n2_mode="insitu"), _St())

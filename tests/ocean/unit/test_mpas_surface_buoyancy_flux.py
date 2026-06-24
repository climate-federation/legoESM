"""Direct unit test for the shared MPAS surface buoyancy-flux helper (#518 §1).

`make_kpp_physics_mpas` and `make_kpp_profiles_mpas` previously computed the
surface buoyancy flux from byte-identical ~48-LOC inline blocks.  This pins the
factored `_mpas_surface_buoyancy_flux` to be bit-identical to that formula and
guards the sign-sensitive MPAS salt convention (real brine feeds buoyancy ONLY,
never the non-local Q_sfc_S; freshwater feeds both).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.eos import (
    rho_0 as _RHO_0,
    c_sw as _C_SW,
    thermal_expansion_coeff,
    haline_contraction_coeff,
)
from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
    _mpas_surface_buoyancy_flux,
)


def _reference(q_net, fw, salt, T_3d, S_3d):
    """Exact pre-#518 inline formula, reproduced for the byte-identity gate."""
    Q_sfc_T = None
    B_f = None
    if q_net is not None:
        Q_sfc_T = q_net / (_RHO_0 * _C_SW)
        T_sfc = T_3d[..., 0]
        S_sfc = S_3d[..., 0]
        p_sfc = jnp.zeros_like(T_sfc)
        alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
        B_f = -constants.g * alpha * Q_sfc_T
    Q_sfc_S = None
    if fw is not None or salt is not None:
        S_sfc = S_3d[..., 0]
        T_sfc = T_3d[..., 0]
        p_sfc = jnp.zeros_like(T_sfc)
        beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
        B_salt = jnp.zeros_like(S_sfc)
        Q_sfc_S = jnp.zeros_like(S_sfc)
        if fw is not None:
            Q_sfc_S = -S_sfc * jnp.asarray(fw, S_sfc.dtype) / _RHO_0
            B_salt = B_salt + constants.g * beta * Q_sfc_S
        if salt is not None:
            B_salt = B_salt + constants.g * beta * (
                jnp.asarray(salt, S_sfc.dtype) * 1.0e3 / _RHO_0)
        B_f = B_salt if B_f is None else (B_f + B_salt)
    return B_f, Q_sfc_T, Q_sfc_S


def _state(seed=0):
    rng = np.random.default_rng(seed)
    n_cells, nlev = 12, 6
    T = jnp.asarray(2.0 + 20.0 * rng.random((n_cells, nlev)), dtype=jnp.float64)
    S = jnp.asarray(33.0 + 3.0 * rng.random((n_cells, nlev)), dtype=jnp.float64)
    q = jnp.asarray(rng.standard_normal(n_cells) * 50.0, dtype=jnp.float64)
    fw = jnp.asarray(rng.standard_normal(n_cells) * 1e-6, dtype=jnp.float64)
    salt = jnp.asarray(rng.standard_normal(n_cells) * 1e-5, dtype=jnp.float64)
    return q, fw, salt, T, S


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_byte_identical_to_old_inline_all_channels():
    q, fw, salt, T, S = _state()
    cases = [
        (q, fw, salt), (q, None, None), (None, fw, None), (None, None, salt),
        (q, fw, None), (None, fw, salt), (None, None, None),
    ]
    for qn, f, s in cases:
        got = _mpas_surface_buoyancy_flux(qn, f, s, T, S)
        ref = _reference(qn, f, s, T, S)
        for g_, r_ in zip(got, ref):
            assert _eq(g_, r_), (qn is not None, f is not None, s is not None)


def test_no_forcing_returns_all_none():
    _, _, _, T, S = _state()
    assert _mpas_surface_buoyancy_flux(None, None, None, T, S) == (None, None, None)


def test_real_salt_is_buoyancy_only_not_nonlocal():
    """Brine salt must NOT enter Q_sfc_S (MPAS partial-cell mass-conservation),
    but MUST destabilise B_f (positive brine -> positive B_f contribution)."""
    _, _, salt, T, S = _state()
    salt_pos = jnp.abs(salt) + 1e-6  # brine rejection (destabilising)
    B_f, Q_sfc_T, Q_sfc_S = _mpas_surface_buoyancy_flux(
        None, None, salt_pos, T, S)
    # salt-only, no freshwater -> the non-local salt flux stays at its zero
    # baseline (salt feeds buoyancy only).
    assert np.allclose(np.asarray(Q_sfc_S), 0.0)
    assert Q_sfc_T is None
    # Brine is destabilising -> B_f > 0 everywhere (beta, g > 0; salt_pos > 0).
    assert np.all(np.asarray(B_f) > 0.0)


def test_freshwater_feeds_nonlocal_salt_flux():
    """Freshwater (virtual salt) feeds Q_sfc_S; net P-E in (fw>0) -> Q_sfc_S<0."""
    _, fw, _, T, S = _state()
    fw_pos = jnp.abs(fw) + 1e-6
    _, _, Q_sfc_S = _mpas_surface_buoyancy_flux(None, fw_pos, None, T, S)
    assert np.all(np.asarray(Q_sfc_S) < 0.0)

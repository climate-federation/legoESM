"""KPP non-local flux over a partial bottom cell conserves the column content.

The explicit non-local term (LMD94 eq. 19) is a flux divergence.  With live layer thicknesses h_k
(``h_k`` given, the lat-lon path) the divergence is taken over h_k and the flux is kept only between
two wet cells, so sum_k h_k dT/dt = 0: the term moves heat and salt within the boundary layer and
creates none.  The former divisor dz_ref * jacobian is thicker than a partial bottom cell, and the
flux at the seafloor interface was kept, so a column whose boundary layer reaches a partial bottom
cell gained heat (the nest worker's residual, docs/89 5.10a of the ENSO project).
"""
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.ocean.eos import wright_eos  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing  # noqa: E402
from legoesm.ocean.vertical import (compute_layer_thickness, compute_ocean_jacobian,  # noqa: E402
                                    create_partial_cell_coordinate, create_z_star_from_thicknesses)


def _column_budget(with_live_thickness):
    zc = create_z_star_from_thicknesses(np.full(10, 10.0))
    H = jnp.asarray([[95.0, 100.0, 63.0]])                       # partial bottom cells 5 m and 3 m, and a full one
    zp = create_partial_cell_coordinate(zc, H)
    eta = jnp.zeros_like(H)
    J = compute_ocean_jacobian(eta, H, zp)
    h = compute_layer_thickness(eta, H, zp)
    shape = h.shape
    T, S = jnp.full(shape, 20.0) * (h > 0), jnp.full(shape, 35.0) * (h > 0)
    p = 1025.0 * 9.81 * (jnp.cumsum(h, -1) - 0.5 * h)
    rho = wright_eos(T, S, p)
    zero = jnp.zeros(shape)
    out = kpp_vertical_mixing(zero, zero, T, S, rho, eta, zp, J, KPPConfig(),
                              tau_x=jnp.full(H.shape, 0.02), tau_y=jnp.zeros(H.shape),
                              B_f=jnp.full(H.shape, 2e-7),            # strong convection: the layer reaches the floor
                              Q_sfc_T=jnp.full(H.shape, -1e-4), Q_sfc_S=jnp.full(H.shape, 1e-6),
                              apply_diffusion=False, h_k=h if with_live_thickness else None)
    hT = np.asarray(h * out.dT_dt)
    hS = np.asarray(h * out.dS_dt)
    return hT.sum(-1), np.abs(hT).sum(-1), hS.sum(-1), np.abs(hS).sum(-1)


def test_nonlocal_flux_conserves_column_heat_and_salt_with_partial_cells():
    netT, grossT, netS, grossS = _column_budget(True)
    assert (grossT > 0).all() and (grossS > 0).all()            # the non-local term acts in every column
    assert np.all(np.abs(netT) <= 1e-13 * grossT), (netT, grossT)
    assert np.all(np.abs(netS) <= 1e-13 * grossS), (netS, grossS)


def test_reference_thickness_divisor_creates_heat_over_partial_cells():
    """Without the live thicknesses (the former form) the 63 m column, whose boundary layer reaches its
    3 m partial bottom cell with dry levels below, does not conserve.  (In the 95 m column the boundary
    layer base is at the last interface, where the non-local shape function is zero, so it conserves
    either way.)"""
    netT, grossT, _, _ = _column_budget(False)
    assert np.abs(netT[0, 2]) > 1e-6 * grossT[0, 2], (netT, grossT)

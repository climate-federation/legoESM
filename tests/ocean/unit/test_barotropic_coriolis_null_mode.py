"""The C-grid barotropic Coriolis 2Δx rotational null mode and its faithful cure.

Root cause of the §5 eddy-resolving turbulent blow-up (docs/issues/
barotropic_mode_noise.md §A): the in-substep barotropic Coriolis interpolates V
to u-points with the 4-point average ``0.25·(V[i]+V[i+1] + V_west[i]+V_west[i+1])``
(``barotropic_latlon_cgrid.py`` L377-379), where ``V_west = roll(V, 1, axis=lon)``.
A 2Δx ZONAL checkerboard, V[:,j] = (-1)^j, makes V_west = −V, so the average
vanishes and the discrete Coriolis operator exerts NO restoring on it — the
divergent high-zonal-wavenumber Arakawa-Lamb (1977) null mode, free to grow under
the eddy field until the run blows up.

The faithful cure (``coriolis_scheme="explicit_ab2"``) routes the planetary
Coriolis to the barotropic mode through the AB2-extrapolated slow forcing F_slow
(Oceananigans split-explicit convention: ∂_tU = −gH∇η + G^U, NO in-substep
Coriolis) → ``add_barotropic_coriolis=False`` → the null-mode operator is never
applied.  No dissipation backstop.

These tests drive the REAL ``barotropic_substeps_latlon_cgrid`` (no duplicated
stencil numerics) to pin both halves: (1) the checkerboard is a Coriolis null
mode — it produces no U response while a smooth V does; (2) ``add_barotropic_
coriolis=False`` (the cure) removes the in-substep Coriolis coupling entirely.

The full 160×128×50 80-day survival of the cured stack is validated offline by
``scripts/tmp/_silvestri_eps_validate.py`` (too expensive for CI); these unit
tests pin the *mechanism* that makes that survival hold.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid)
from legoesm.ocean.experiments.silvestri_baroclinic_jet import (
    SilvestriJetConfig, build_silvestri_baroclinic_jet_setup)

set_policy(PrecisionPolicy.fp64())


def _rest_setup():
    """Small §5 setup, then quiesce: u=v=eta=0, alpha=0 (isolate Coriolis)."""
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    cfg = r.model_config._replace(barotropic_diffusion_alpha=0.0)
    s = r.initial_state
    z_u = jnp.zeros_like(s.u.data)
    z_v = jnp.zeros_like(s.v.data)
    z_e = jnp.zeros_like(s.eta.data)
    s = s._replace(
        u=Field(data=z_u, name="u", dims=s.u.dims, units=s.u.units),
        v=Field(data=z_v, name="v", dims=s.v.dims, units=s.v.units),
        eta=Field(data=z_e, name="eta", dims=s.eta.dims, units=s.eta.units))
    return r, cfg, s


def _set_v(s, v3d):
    return s._replace(v=Field(data=v3d, name="v", dims=s.v.dims, units=s.v.units))


def _checkerboard_v(s):
    """2Δx zonal checkerboard: v[:,j,:] = (-1)^j (the Coriolis null mode).

    The 4-point V→u average includes ``V_west = roll(V, 1, axis=lon)``
    (``barotropic_latlon_cgrid.py`` L377): for a zonal checkerboard
    V_west = −V, so ``V_bar_c + V_west = 0`` and V_at_u vanishes — the
    divergent high-zonal-wavenumber Arakawa-Lamb mode the Coriolis
    operator cannot see (docs/issues/barotropic_mode_noise.md §A)."""
    nlat_v, nlon, nlev = s.v.data.shape
    sign = ((-1.0) ** jnp.arange(nlon))[None, :, None]
    return jnp.broadcast_to(sign, (nlat_v, nlon, nlev)).astype(s.v.data.dtype)


def _smooth_v(s):
    """A smooth (DC) v field of the same amplitude for the control."""
    return jnp.ones_like(s.v.data)


def _u_field(r, cfg, s, add_cor):
    """One barotropic step from rest with the given v; return the u field."""
    out, _ = barotropic_substeps_latlon_cgrid(
        s, dt_s=30.0, n_substeps=10, grid=r.grid, z_coord=r.z_coord,
        config=cfg, add_barotropic_coriolis=add_cor)
    return np.asarray(out.u.data)


def _coriolis_u(r, cfg, s):
    """Coriolis-ONLY U contribution = |u(cor on) − u(cor off)| (isolates the
    Coriolis term from the continuity→PGF gravity-wave response, which both
    paths share)."""
    return float(np.max(np.abs(
        _u_field(r, cfg, s, add_cor=True) - _u_field(r, cfg, s, add_cor=False))))


def test_checkerboard_is_coriolis_null_mode():
    """A 2Δx zonal checkerboard V drives ~no Coriolis U; a smooth V drives a lot."""
    r, cfg, s = _rest_setup()
    cor_checker = _coriolis_u(r, cfg, _set_v(s, _checkerboard_v(s)))
    cor_smooth = _coriolis_u(r, cfg, _set_v(s, _smooth_v(s)))
    # The smooth field feels the Coriolis operator...
    assert cor_smooth > 1e-4, f"smooth V felt no Coriolis: {cor_smooth:.2e}"
    # ...the checkerboard is annihilated by the 4-point average → null mode:
    # the discrete Coriolis exerts no restoring on it.
    assert cor_checker < 1e-3 * cor_smooth, (
        f"checkerboard is NOT a Coriolis null mode: cor_checker={cor_checker:.2e} "
        f"vs cor_smooth={cor_smooth:.2e}")


def test_cure_removes_in_substep_coriolis():
    """add_barotropic_coriolis=False (the explicit_ab2 cure) drops the term."""
    r, cfg, s = _rest_setup()
    s_smooth = _set_v(s, _smooth_v(s))
    u_on = _u_field(r, cfg, s_smooth, add_cor=True)
    u_off = _u_field(r, cfg, s_smooth, add_cor=False)
    # With Coriolis ON a smooth (divergence-free) V drives U purely via Coriolis;
    # with it OFF (the cure) there is no in-substep Coriolis and no U at all.
    assert float(np.max(np.abs(u_on))) > 1e-4, (
        f"control: Coriolis-on must drive U, got {np.max(np.abs(u_on)):.2e}")
    assert float(np.max(np.abs(u_off))) < 1e-12, (
        f"cure: add_barotropic_coriolis=False must remove the in-substep "
        f"Coriolis (no U from a divergence-free V), got {np.max(np.abs(u_off)):.2e}")


def test_explicit_ab2_config_gates_in_substep_coriolis():
    """The §5 faithful stack wires coriolis_scheme=explicit_ab2 (=> term off)."""
    r = build_silvestri_baroclinic_jet_setup(
        n_lat=24, n_lon=16, scheme="W9V", nlev=4,
        config=SilvestriJetConfig(), stabilize=False)
    mc = r.model_config
    assert mc.coriolis_scheme == "explicit_ab2"
    assert mc.barotropic_solver == "explicit_substep"
    assert mc.barotropic_slow_forcing_ab2 is True
    assert mc.barotropic_diffusion_alpha == 0.0

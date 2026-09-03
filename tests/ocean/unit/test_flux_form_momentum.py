"""Flux-form horizontal momentum advection — truth-tier gates (build spec F3-F6).

Directly exercises ``_bc_horizontal_momentum_advection_flux_form`` on a small flat
doubly-periodic domain:
- F3 zero-velocity -> zero advective tendency,
- F4 uniform flow -> zero advective tendency (analytic),
- F5 momentum conservation (volume-weighted domain integral ~ machine-eps),
- F6 differentiability.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _bc_horizontal_momentum_advection_flux_form,
)
from legoesm.ocean.state import LatLonCGridOceanConfig

# The two 3rd-order upwind-biased arms; identical kappa=1/3 arithmetic, they
# differ ONLY in which pair sign picks the T-point upwind branch (see
# UP3_REFERENCE_SELECTOR).
_UP3_ARMS = ("nemo_up3", "oceananigans_up3")

_NLAT, _NLON, _NLEV = 8, 16, 3
_H = 100.0   # flat-bottom uniform layer thickness [m]


def _setup(u_vals, v_vals, scheme="centered"):
    """Build (du0, dv0, u, v, h_u, h_v, masks, grid, config) for a flat,
    all-ocean, doubly-periodic-in-lon domain. v is zeroed at the pole v-faces
    (wall). u_vals/v_vals are arrays of the right shape."""
    # float64 grid so the conservation gate measures the scheme's telescoping,
    # not float32 area round-off (production FV may run float32).
    grid = create_latlon_grid(_NLAT, _NLON, dtype=jnp.float64)
    nlat, nlon = grid.n_lat, grid.n_lon
    u = jnp.asarray(u_vals)
    v = jnp.asarray(v_vals)
    # Wall: no flow through the poles.
    v = v.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
    h_u = jnp.full((nlat, nlon + 1, _NLEV), _H)
    h_v = jnp.full((nlat + 1, nlon, _NLEV), _H)
    u_mask_3d = jnp.ones((nlat, nlon + 1, _NLEV))
    v_mask_3d = jnp.ones((nlat + 1, nlon, _NLEV))
    mask = jnp.ones((nlat, nlon))
    du0 = jnp.zeros((nlat, nlon + 1, _NLEV))
    dv0 = jnp.zeros((nlat + 1, nlon, _NLEV))
    cfg = LatLonCGridOceanConfig.from_flat(
        momentum_advection="flux_form", momentum_flux_scheme=scheme,
    )
    return du0, dv0, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg


def _call(du0, dv0, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg):
    return _bc_horizontal_momentum_advection_flux_form(
        du0, dv0, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg,
    )


def test_F3_zero_velocity_zero_tendency():
    """u=v=0 -> the flux-form advective tendency is exactly zero."""
    args = _setup(
        np.zeros((_NLAT, _NLON + 1, _NLEV)), np.zeros((_NLAT + 1, _NLON, _NLEV)),
    )
    du, dv, hu, hv = _call(*args)
    assert float(jnp.max(jnp.abs(hu))) == 0.0
    assert float(jnp.max(jnp.abs(hv))) == 0.0


def test_F4_uniform_flow_zero_tendency():
    """Uniform u=const, v=0 on a flat periodic domain -> advection of a constant
    is zero to round-off (div of a constant momentum flux = 0)."""
    u = np.full((_NLAT, _NLON + 1, _NLEV), 0.7)
    v = np.zeros((_NLAT + 1, _NLON, _NLEV))
    args = _setup(u, v, scheme="centered")
    du, dv, hu, hv = _call(*args)
    # Tendency scale for context: |u| * (typical 1/dt) — just assert ~0.
    assert float(jnp.max(jnp.abs(hu))) < 1e-12, float(jnp.max(jnp.abs(hu)))
    assert float(jnp.max(jnp.abs(hv))) < 1e-12, float(jnp.max(jnp.abs(hv)))


def _conservation_residual(scheme):
    rng = np.random.default_rng(3)
    u = 0.3 * rng.standard_normal((_NLAT, _NLON + 1, _NLEV))
    v = 0.3 * rng.standard_normal((_NLAT + 1, _NLON, _NLEV))
    # Interior flow: v -> 0 in the two lat rows nearest each pole, so the
    # meridional momentum flux through the boundary cell-centres vanishes. This
    # isolates the SCHEME's conservation (flux-divergence telescoping) from the
    # physical transfer of v-momentum to the N/S walls (which a walled lat-lon
    # domain legitimately does — the y-direction is not periodic). u-momentum is
    # periodic in lon and conserves regardless; this keeps v honest too.
    v[:2, :, :] = 0.0
    v[-2:, :, :] = 0.0
    args = _setup(u, v, scheme=scheme)
    du0, dv0, uu, vv, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg = args
    _, _, hadv_u, hadv_v = _call(*args)
    area = np.asarray(grid.area)
    # u-cell volume weight A_u*h_u (A_u = area avg to u-faces, periodic).
    a_uc = 0.5 * (area + np.roll(area, 1, axis=1))
    A_u = np.concatenate([a_uc, a_uc[:, 0:1]], axis=1)[..., None]
    vol_u = A_u * np.asarray(h_u)
    # Sum over UNIQUE u-points (exclude the periodic wrap column n_lon).
    mom_u = float(np.sum((vol_u * np.asarray(hadv_u))[:, :-1, :]))
    scale_u = float(np.sum(np.abs(vol_u * np.asarray(hadv_u))[:, :-1, :])) + 1e-300
    # v-cell volume weight.
    a_vc = 0.5 * (area[:-1] + area[1:])
    A_v = np.concatenate([a_vc[:1], a_vc, a_vc[-1:]], axis=0)[..., None]
    vol_v = A_v * np.asarray(h_v)
    mom_v = float(np.sum(vol_v * np.asarray(hadv_v)))
    scale_v = float(np.sum(np.abs(vol_v * np.asarray(hadv_v)))) + 1e-300
    return mom_u / scale_u, mom_v / scale_v


def test_F5_momentum_conservation_centered():
    """Volume-weighted domain-integrated flux-form advective tendency ~ 0 (the
    advection redistributes momentum; fluxes telescope on a periodic domain with
    v=0 walls). Centered scheme."""
    ru, rv = _conservation_residual("centered")
    assert abs(ru) < 1e-12, f"u-momentum not conserved: rel residual {ru:.2e}"
    assert abs(rv) < 1e-12, f"v-momentum not conserved: rel residual {rv:.2e}"


def test_F5_momentum_conservation_upwind():
    """Same conservation holds for the upwind reconstruction (telescoping is
    independent of the advected-value interpolation)."""
    ru, rv = _conservation_residual("upwind")
    assert abs(ru) < 1e-12, f"u-momentum not conserved (upwind): {ru:.2e}"
    assert abs(rv) < 1e-12, f"v-momentum not conserved (upwind): {rv:.2e}"


@pytest.mark.parametrize("scheme", _UP3_ARMS)
def test_F5_momentum_conservation_upwind3(scheme):
    """UP3 (3rd-order upwind-biased flux-form) conserves too — telescoping is
    independent of the 4-point reconstruction, and of which reference arm
    picks the upwind branch."""
    ru, rv = _conservation_residual(scheme)
    assert abs(ru) < 1e-12, f"u-momentum not conserved ({scheme}): {ru:.2e}"
    assert abs(rv) < 1e-12, f"v-momentum not conserved ({scheme}): {rv:.2e}"


@pytest.mark.parametrize("scheme", _UP3_ARMS)
def test_F4_uniform_flow_zero_tendency_upwind3(scheme):
    """Uniform u=const, v=0 -> UP3 advection of a constant is zero (UP3
    reconstructs constants exactly)."""
    u = np.full((_NLAT, _NLON + 1, _NLEV), 0.7)
    v = np.zeros((_NLAT + 1, _NLON, _NLEV))
    args = _setup(u, v, scheme=scheme)
    du, dv, hu, hv = _call(*args)
    assert float(jnp.max(jnp.abs(hu))) < 1e-12, float(jnp.max(jnp.abs(hu)))
    assert float(jnp.max(jnp.abs(hv))) < 1e-12, float(jnp.max(jnp.abs(hv)))


@pytest.mark.parametrize("scheme", _UP3_ARMS)
def test_F8_upwind3_differs_from_upwind(scheme):
    """UP3 yields a different tendency than 1st-order upwind on a structured
    field (the higher-order reconstruction changes the advected face values)."""
    rng = np.random.default_rng(7)
    u = 0.3 * rng.standard_normal((_NLAT, _NLON + 1, _NLEV))
    v = 0.3 * rng.standard_normal((_NLAT + 1, _NLON, _NLEV))
    v[:2] = 0.0
    v[-2:] = 0.0
    _, _, hu1, _ = _call(*_setup(u, v, scheme="upwind"))
    _, _, hu3, _ = _call(*_setup(u, v, scheme=scheme))
    # Difference is the same order as the (weak-flow, coarse-grid) tendency
    # itself → a genuine scheme difference, not round-off.
    assert float(jnp.max(jnp.abs(hu3 - hu1))) > 1e-9, "UP3 == 1st-order upwind?"


class TestUP3Reconstruction:
    """Unit tests for the UP3 face reconstruction _up3_reconstruct (NEMO κ=1/3)."""

    def test_constant_exact(self):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _up3_reconstruct
        c = 3.7
        for tr in (1.0, -1.0):
            r = _up3_reconstruct(jnp.array(c), jnp.array(c), jnp.array(c),
                                 jnp.array(c), jnp.array(tr))
            assert float(jnp.abs(r - c)) < 1e-14

    def test_linear_field_exact(self):
        """UP3 reconstructs a linear field exactly at the face (x=0.5 between
        the two straddling cells at x=0 and x=1)."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _up3_reconstruct
        # cells at x = -1, 0, 1, 2 ; face at x = 0.5. f(x) = 2x + 1.
        f = lambda x: 2.0 * x + 1.0
        far_pos, adv_pos, adv_neg, far_neg = (jnp.array(f(x)) for x in (-1, 0, 1, 2))
        for tr in (1.0, -1.0):
            r = _up3_reconstruct(far_pos, adv_pos, adv_neg, far_neg, jnp.array(tr))
            assert float(jnp.abs(r - f(0.5))) < 1e-13, (tr, float(r))

    def test_upwind_bias_direction(self):
        """For transport>0 the stencil leans on the upstream (far_pos) cell; for
        transport<0 it leans on far_neg — the two branches differ on a non-linear
        (curved) field."""
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _up3_reconstruct
        f = lambda x: x ** 3          # odd curvature → upwind branches differ
        fp, ap, an, fn = (jnp.array(float(f(x))) for x in (-1, 0, 1, 2))
        r_pos = _up3_reconstruct(fp, ap, an, fn, jnp.array(1.0))
        r_neg = _up3_reconstruct(fp, ap, an, fn, jnp.array(-1.0))
        assert float(jnp.abs(r_pos - r_neg)) > 1e-6


def test_F6_differentiable():
    """jax.grad of a scalar loss through the flux-form path is finite + nonzero."""
    rng = np.random.default_rng(5)
    u0 = jnp.asarray(0.3 * rng.standard_normal((_NLAT, _NLON + 1, _NLEV)))
    v0 = jnp.asarray(0.3 * rng.standard_normal((_NLAT + 1, _NLON, _NLEV)))
    args = _setup(np.asarray(u0), np.asarray(v0))
    du0, dv0, _, _, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg = args

    def loss(u):
        v = jnp.asarray(np.asarray(v0)).at[0].set(0.0).at[-1].set(0.0)
        _, _, hadv_u, _ = _bc_horizontal_momentum_advection_flux_form(
            du0, dv0, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg,
        )
        return jnp.sum(hadv_u ** 2)

    g = jax.grad(loss)(u0)
    assert jnp.all(jnp.isfinite(g)), "non-finite grad through flux-form advection"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "zero grad — path not differentiated"


def test_F7_gyre_stability_flux_form():
    """F7: a short forced-flow integration with momentum_advection='flux_form'
    stays finite and KE stays bounded (does not blow up). The upwind flux-form
    is dissipative, so KE plateaus/decays. vector_invariant runs too (baseline)."""
    import jax
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(12, 24)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=85.0,
    )
    # Active flow so horizontal momentum advection is exercised.
    up = 0.1 * jax.random.normal(jax.random.PRNGKey(11), state0.u.data.shape,
                                 dtype=jnp.float64)
    vp = 0.1 * jax.random.normal(jax.random.PRNGKey(12), state0.v.data.shape,
                                 dtype=jnp.float64)
    state0 = state0._replace(
        u=state0.u.replace(data=state0.u.data + up),
        v=state0.v.replace(data=state0.v.data + vp),
    )

    def _run(scheme):
        cfg = LatLonCGridOceanConfig.from_flat(
            momentum_advection=scheme, momentum_flux_scheme="upwind",
            A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
            n_barotropic_substeps=8, enable_runtime_checks=False,
        )
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        state = state0
        ke0 = float(jnp.sum(state.u.data ** 2) + jnp.sum(state.v.data ** 2))
        for _ in range(80):
            state = model.step(state, dt=600.0)
        ke = float(jnp.sum(state.u.data ** 2) + jnp.sum(state.v.data ** 2))
        finite = bool(
            jnp.all(jnp.isfinite(state.u.data))
            and jnp.all(jnp.isfinite(state.v.data))
            and jnp.all(jnp.isfinite(state.T.data))
        )
        return finite, ke0, ke

    fin_ff, ke0, ke_ff = _run("flux_form")
    assert fin_ff, "flux_form integration produced non-finite state"
    assert ke_ff < 10.0 * ke0, (
        f"flux_form KE grew unboundedly: {ke_ff:.3e} vs initial {ke0:.3e}"
    )
    fin_vi, _, _ = _run("vector_invariant")
    assert fin_vi, "vector_invariant baseline produced non-finite state"


def _zonal_selector_case():
    """Curved zonal profile plus a uniform ``zub`` shift large enough to flip
    the pair sign on part of the row (v == 0)."""
    nlat, nlon, nlev = _NLAT, _NLON, _NLEV
    x = np.arange(nlon + 1, dtype=np.float64)
    prof = 0.10 * np.sin(2.0 * np.pi * x / nlon) ** 3      # curved, sign-changing
    u = np.broadcast_to(prof[None, :, None], (nlat, nlon + 1, nlev)).copy()
    v = np.zeros((nlat + 1, nlon, nlev))
    u_t = u - 0.06                                          # |zub| > |u| on part of the row
    return u, v, u_t


def _hadv(u, v, tv, scheme="nemo_up3", **kw):
    args = _setup(u, v, scheme=scheme)
    du0, dv0, uj, vj, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg = args
    _, _, hadv_u, hadv_v = _bc_horizontal_momentum_advection_flux_form(
        du0, dv0, uj, vj, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, cfg,
        transport_velocity=None if tv is None else (jnp.asarray(tv[0]), jnp.asarray(tv[1])),
        **kw)
    return np.asarray(hadv_u), np.asarray(hadv_v), args


def test_F9_up3_velocity_selector_is_nemo_dynadv_up3():
    """NEMO dynadv_up3.F90:166-170: the T-point UP3 branch is chosen by the
    sign of the advected-velocity pair ``uu_i + uu_{i+1}``, not by the
    transport pair (which under WS-RK3 carries ``zub``).  ``up3_upwind_selector
    ="velocity"`` must reproduce an independent assembly of that rule.

    Non-vacuous: if the same-direction reconstruction ignored the selector
    (transport sign always), the ``velocity`` result would miss the NEMO
    reference by a first-order amount at the disagreeing faces.
    """
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _up3_reconstruct

    u, v, u_t = _zonal_selector_case()
    _, _, args = _hadv(u, v, (u_t, v))
    hadv_transport, _, _ = _hadv(u, v, (u_t, v), up3_upwind_selector="transport")
    hadv_velocity, _, _ = _hadv(u, v, (u_t, v), up3_upwind_selector="velocity")
    _, _, _, _, h_u, _, u_mask_3d, _, _, grid, _ = args

    u_core = u[:, :-1, :]
    pair_vel = u_core + u[:, 1:, :]
    pair_tr = u_t[:, :-1, :] + u_t[:, 1:, :]
    assert bool(np.any((pair_vel > 0) != (pair_tr > 0))), "selector never disagrees"

    # Independent assembly of the zonal flux with NEMO's selector (v == 0, so
    # the meridional flux is identically zero): the T-point face value picks
    # its upwind curvature by the advected-velocity pair, the flux magnitude is
    # the transport pair.
    dy_u = (np.asarray(grid.dy) * 0.5)[:, None, None]
    Q = np.asarray(h_u) * u_t * np.asarray(u_mask_3d) * dy_u
    Qx_c = 0.5 * (Q[:, :-1, :] + Q[:, 1:, :])
    u_c = np.asarray(_up3_reconstruct(
        jnp.asarray(np.roll(u_core, 1, axis=1)), jnp.asarray(u_core),
        jnp.asarray(u[:, 1:, :]), jnp.asarray(np.roll(u_core, -2, axis=1)),
        jnp.asarray(pair_vel)))
    Fx = Qx_c * u_c
    net = Fx - np.roll(Fx, 1, axis=1)
    net_u = np.concatenate([net, net[:, :1, :]], axis=1)
    area = np.asarray(grid.area)
    a_uc = 0.5 * (area + np.roll(area, 1, axis=1))
    A_u = np.concatenate([a_uc, a_uc[:, :1]], axis=1)[..., None]
    ref = -net_u / (A_u * np.asarray(h_u))

    scale = float(np.max(np.abs(ref)))
    assert scale > 0.0
    assert float(np.max(np.abs(hadv_velocity - ref))) <= 1e-13 * scale, (
        float(np.max(np.abs(hadv_velocity - ref))), scale)
    # The two rules differ by a first-order amount at the disagreeing faces
    # (the UP3 third-difference term): the selector is a live variable.
    assert float(np.max(np.abs(hadv_velocity - hadv_transport))) > 1e-3 * scale


def test_F11_up3_t_point_selector_follows_the_scheme_s_reference_arm():
    """The T-point upwind selector is keyed by the REFERENCE the scheme names,
    not by the time integrator.

    ``momentum_flux_scheme="nemo_up3"`` (NEMO dynadv_up3.F90:166,169-170)
    must pick the branch by the advected-VELOCITY pair;
    ``"oceananigans_up3"`` (Oceananigans
    ``upwind_biased_advective_fluxes.jl:18-24``, which upwind-biases by the
    sign of the interpolated TRANSPORT) must pick it by the transport pair.
    Both are exercised with NO explicit ``up3_upwind_selector`` — the arm has
    to come from the scheme name alone.  The case is built so the two pair
    signs genuinely disagree on part of the row, so the two arms cannot
    coincide by construction.

    Non-vacuous: reverting the selector to the old integrator gate (``None ->
    "transport"`` for every caller) makes the NEMO assertion fail; routing
    both names to ``"velocity"`` makes the Oceananigans assertion fail.
    """
    u, v, u_t = _zonal_selector_case()
    pair_vel = u[:, :-1, :] + u[:, 1:, :]
    pair_tr = u_t[:, :-1, :] + u_t[:, 1:, :]
    assert bool(np.any((pair_vel > 0) != (pair_tr > 0))), "selector never disagrees"

    nemo, _, _ = _hadv(u, v, (u_t, v), scheme="nemo_up3")
    ocng, _, _ = _hadv(u, v, (u_t, v), scheme="oceananigans_up3")
    by_velocity, _, _ = _hadv(u, v, (u_t, v), up3_upwind_selector="velocity")
    by_transport, _, _ = _hadv(u, v, (u_t, v), up3_upwind_selector="transport")

    scale = float(np.max(np.abs(by_velocity)))
    assert scale > 0.0
    assert float(np.max(np.abs(by_velocity - by_transport))) > 1e-3 * scale, (
        "the two selector rules coincide on this case — it cannot discriminate")
    assert np.array_equal(nemo, by_velocity), (
        "nemo_up3 must select the T-point branch by the advected-velocity pair")
    assert np.array_equal(ocng, by_transport), (
        "oceananigans_up3 must select the T-point branch by the transport pair")


def test_F12_bare_upwind3_is_refused():
    """The unqualified ``"upwind3"`` no longer names a reference, so it must
    raise rather than silently fall through to 1st-order upwind."""
    u = np.zeros((_NLAT, _NLON + 1, _NLEV))
    v = np.zeros((_NLAT + 1, _NLON, _NLEV))
    args = _setup(u, v, scheme="centered")
    cfg = args[-1]._replace(momentum_flux_scheme="upwind3")
    with pytest.raises(ValueError, match="momentum_flux_scheme must be one of"):
        _call(*args[:-1], cfg)


def test_F10_up3_cross_fluxes_keep_the_transport_selector():
    """dynadv_up3.F90:179-187: the F-point (cross) fluxes -- u advected in y
    by the v transport, v advected in x by the u transport -- choose their
    upwind curvature by the TRANSPORT pair ``zFvi``/``zFuj``, not by the
    advected-velocity pair.  Each cross term is isolated by zeroing the other
    velocity component (its own same-direction flux then vanishes and the
    remaining same-direction flux is uniform along its direction, hence
    selector-free) while its advecting transport is a uniform non-zero shift
    of the opposite sign to the advected pair.  The ``velocity`` and
    ``transport`` selections must then coincide bit-for-bit, while flipping
    the transport sign must still move the result (the cross flux is live).

    Non-vacuous: marking the cross-term reconstructions ``same_direction``
    makes the two selections differ everywhere the pair sign and the
    transport sign disagree, and the equality below fails.
    """
    nlat, nlon, nlev = _NLAT, _NLON, _NLEV
    lat_prof = 0.05 + 0.10 * np.sin(np.pi * np.arange(nlat) / (nlat - 1)) ** 3
    lon_prof = 0.05 + 0.10 * np.sin(2.0 * np.pi * np.arange(nlon) / nlon) ** 3
    zero_u = np.zeros((nlat, nlon + 1, nlev))
    zero_v = np.zeros((nlat + 1, nlon, nlev))
    # (A) u advected in y by the v transport: u > 0 varies in lat only.
    u = np.broadcast_to(lat_prof[:, None, None], (nlat, nlon + 1, nlev)).copy()
    vt_neg = np.full((nlat + 1, nlon, nlev), -0.06)
    vt_pos = -vt_neg
    a_vel = _hadv(u, zero_v, (u, vt_neg), up3_upwind_selector="velocity")[0]
    a_tr = _hadv(u, zero_v, (u, vt_neg), up3_upwind_selector="transport")[0]
    a_flip = _hadv(u, zero_v, (u, vt_pos), up3_upwind_selector="velocity")[0]
    assert np.array_equal(a_vel, a_tr), "u cross flux must not follow the velocity pair"
    assert float(np.max(np.abs(a_vel - a_flip))) > 0.0, "u cross flux is inert"
    # (B) v advected in x by the u transport: v > 0 varies in lon only.
    v = np.broadcast_to(lon_prof[None, :, None], (nlat + 1, nlon, nlev)).copy()
    ut_neg = np.full((nlat, nlon + 1, nlev), -0.06)
    ut_pos = -ut_neg
    b_vel = _hadv(zero_u, v, (ut_neg, v), up3_upwind_selector="velocity")[1]
    b_tr = _hadv(zero_u, v, (ut_neg, v), up3_upwind_selector="transport")[1]
    b_flip = _hadv(zero_u, v, (ut_pos, v), up3_upwind_selector="velocity")[1]
    assert np.array_equal(b_vel, b_tr), "v cross flux must not follow the velocity pair"
    assert float(np.max(np.abs(b_vel - b_flip))) > 0.0, "v cross flux is inert"

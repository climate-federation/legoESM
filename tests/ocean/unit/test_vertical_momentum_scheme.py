"""Stage-8 vertical momentum advection scheme — dycore-audit D3.

legoESM's stage-8 vertical momentum advection (``-d/dz(w·u)``) advects the
BAROCLINIC PERTURBATION ``u' = u - U_bar`` with FIRST-ORDER INTERFACE UPWIND.
Veros (``core/momentum.py`` ``momentum_advection``) advects the FULL velocity
``u`` with a SECOND-ORDER CENTERED, energy-conserving flux
(``flux_top = 0.25·(u[k+1]+u[k])·(w+w_east)``).  Two structural deltas:
  (a) the missing depth-integral-zero redistribution ``-d/dz(w·U_bar)``;
  (b) the upwind implicit vertical viscosity ``~|w|·dz/2`` damping shear.

``LatLonCGridOceanConfig.vertical_momentum_scheme`` selects:
  - ``"upwind_perturbation"`` (DEFAULT, bit-identical) — current scheme;
  - ``"centered_full"`` (Veros-faithful) — 2nd-order centered flux of FULL u.

This module validates: default-off bit-identity (both ab2 / forward-euler
integrators); the centered-full flux matches the hand-computed Veros form
term-for-term (incl. the w·U_bar part, and with the upwind diffusion absent);
depth-integrated-momentum conservation (telescoping); the KE-conservation sign
difference (centered ≈ 0 vs upwind strictly negative); unknown literal raises;
the centered_full + adaptive_implicit_vertadv combo is rejected; and AD is
finite + nonzero through 2 steps.

fp64 + CPU for deterministic conservation diagnostics.
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

_DT = 3600.0
_N_LAT, _N_LON = 8, 16


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


# ---------------------------------------------------------------------------
# Model harness (mirrors test_dt_mom_async._rigid_lid_basin)
# ---------------------------------------------------------------------------
def _basin(vertical_momentum_scheme="upwind_perturbation",
           outer_integrator="ab2", **cfg_kw):
    """Closed flat-bottom channel (land walls N/S, periodic x) with a
    meridional T gradient that drives a sheared geostrophic flow — so the
    vertical advection of FULL u differs measurably from that of u'."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z_coord = create_ocean_z_star(n_levels=6, H_max=4000.0)
    lm = np.ones((_N_LAT, _N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((_N_LAT, _N_LON), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=jnp.asarray(lm),
        H_bathy_override=jnp.asarray(Hb))
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=2.0e4, bottom_drag_r=1.0e-3, implicit_vertical_mixing=True,
        enable_runtime_checks=False, barotropic_solver="rigid_lid",
        outer_integrator=outer_integrator,
        vertical_momentum_scheme=vertical_momentum_scheme, **cfg_kw)
    return state, LatLonCGridOceanModel(grid, z_coord, cfg)


# ===========================================================================
# 1. DEFAULT-OFF BIT-IDENTITY (both integrator paths)
# ===========================================================================
@pytest.mark.parametrize("integrator", ["ab2", "forward_euler"])
def test_default_off_bit_identical(integrator):
    """The default scheme literal must reproduce the legacy trajectory
    byte-for-byte under both outer integrators (ab2 + forward_euler)."""
    state_d, model_d = _basin(outer_integrator=integrator)   # default field
    state_e, model_e = _basin(vertical_momentum_scheme="upwind_perturbation",
                              outer_integrator=integrator)    # explicit default
    fd, _ = model_d.integrate_scan(state_d, n_steps=6, dt=_DT)
    fe, _ = model_e.integrate_scan(state_e, n_steps=6, dt=_DT)
    for a, b in ((fd.T.data, fe.T.data), (fd.u.data, fe.u.data),
                 (fd.v.data, fe.v.data), (fd.S.data, fe.S.data),
                 (fd.eta.data, fe.eta.data)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_centered_full_changes_trajectory():
    """The knob is wired: centered_full must produce a DIFFERENT (finite)
    trajectory than the default upwind_perturbation scheme."""
    s_up, m_up = _basin("upwind_perturbation")
    s_cf, m_cf = _basin("centered_full")
    f_up, _ = m_up.integrate_scan(s_up, n_steps=8, dt=_DT)
    f_cf, _ = m_cf.integrate_scan(s_cf, n_steps=8, dt=_DT)
    assert np.all(np.isfinite(np.asarray(f_cf.u.data)))
    du = float(np.max(np.abs(np.asarray(f_cf.u.data) - np.asarray(f_up.u.data))))
    umax = float(np.max(np.abs(np.asarray(f_up.u.data))))
    assert du > 1e-6 * max(umax, 1e-12), (
        f"centered_full did not change the trajectory (max|du|={du:.3e})")


# ===========================================================================
# 2. MANUFACTURED-COLUMN FLUX == HAND-COMPUTED VEROS FORM (term-for-term)
# ===========================================================================
def _veros_centered_tendency(u, w_half, dz):
    """Hand reference: Veros vertical momentum advection on flat metrics.

    F[k] = 0.5*(u[k-1]+u[k]) * w_half[k]  for interior interfaces 1..nlev-1,
    F[0] = F[nlev] = 0 (rigid lid + bottom no-flux), tendency = -(F[k]-F[k+1])/dz.
    """
    nlev = u.shape[-1]
    F = np.zeros(nlev + 1)
    for k in range(1, nlev):
        F[k] = 0.5 * (u[k - 1] + u[k]) * w_half[k]
    return -(F[:-1] - F[1:]) / dz


def test_centered_flux_matches_veros_form():
    """The centered helper reproduces the hand-computed Veros centered flux to
    machine precision on a manufactured column — including the w·U_bar part
    (the operand is FULL u) and with the upwind diffusion ABSENT."""
    from legoesm.ocean.vertical import (
        flux_form_vertical_momentum_advection_centered,
        flux_form_vertical_momentum_advection,
    )
    nlev = 15
    ddz = np.array([50, 70, 100, 140, 190, 240, 290, 340, 390,
                    440, 490, 540, 590, 640, 690], dtype=np.float64) / 2.5
    dz = jnp.asarray(ddz)
    z_c = -(np.cumsum(ddz) - 0.5 * ddz)
    H = float(np.sum(ddz))
    U_bar = 0.20
    u_prime = 0.15 * np.exp(z_c / 800.0)
    u_prime = u_prime - np.sum(u_prime * ddz) / H        # exact zero depth-mean
    u_full = jnp.asarray(u_prime + U_bar)
    z_half = np.concatenate([[0.0], -np.cumsum(ddz)])
    w_half = jnp.asarray(-2.0e-5 * np.sin(np.pi * (-z_half) / H))  # 0 at both ends

    u3 = u_full[None, None, :]
    w3 = w_half[None, None, :]
    h3 = dz[None, None, :] * jnp.ones((1, 1, nlev))

    got = np.asarray(flux_form_vertical_momentum_advection_centered(u3, w3, h3))[0, 0]
    ref = _veros_centered_tendency(np.asarray(u_full), np.asarray(w_half), ddz)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-18)

    # The w·U_bar (barotropic redistribution) part is PRESENT: the centered
    # tendency on FULL u differs from the centered tendency on u' alone by
    # exactly the centered advection of the constant U_bar field.
    up3 = jnp.asarray(u_prime)[None, None, :]
    got_prime = np.asarray(
        flux_form_vertical_momentum_advection_centered(up3, w3, h3))[0, 0]
    ubar_part_got = got - got_prime
    # -d/dz(w·U_bar): centered flux of constant U_bar = U_bar·w at interfaces.
    F_ubar = np.zeros(nlev + 1)
    for k in range(1, nlev):
        F_ubar[k] = U_bar * np.asarray(w_half)[k]
    ubar_part_ref = -(F_ubar[:-1] - F_ubar[1:]) / ddz
    np.testing.assert_allclose(ubar_part_got, ubar_part_ref, rtol=1e-13, atol=1e-18)
    assert np.max(np.abs(ubar_part_ref)) > 0.0, "U_bar redistribution must be nonzero"

    # The UPWIND implicit-diffusion piece is absent in centered: the centered
    # tendency on a SMOOTH field carries no |w|·dz/2 diffusive signature.  The
    # difference (centered - upwind) on FULL u must be nonzero (the schemes
    # genuinely differ), confirming the diffusion is not silently reintroduced.
    got_upwind = np.asarray(
        flux_form_vertical_momentum_advection(u3, w3, h3))[0, 0]
    assert np.max(np.abs(got - got_upwind)) > 0.0


# ===========================================================================
# 3. CONSERVATION: depth-integrated momentum unchanged by vertical advection
# ===========================================================================
def test_depth_integrated_momentum_conserved():
    """For a closed column (w=0 at surface+bottom), the centered vertical-
    advection tendency has exactly zero thickness-weighted column integral
    (telescoping of the flux divergence) — to f64 round-off."""
    from legoesm.ocean.vertical import (
        flux_form_vertical_momentum_advection_centered,
        flux_form_vertical_momentum_advection,
    )
    nlev = 12
    rng = np.random.default_rng(0)
    dz = jnp.asarray(rng.uniform(40.0, 300.0, nlev))
    u = jnp.asarray(rng.normal(0.0, 0.2, nlev))   # arbitrary full velocity
    z_half = np.concatenate([[0.0], -np.cumsum(np.asarray(dz))])
    H = float(np.sum(np.asarray(dz)))
    w_half = jnp.asarray(-1e-5 * np.sin(np.pi * (-z_half) / H))  # 0 at both ends

    u3, w3, h3 = u[None, None, :], w_half[None, None, :], dz[None, None, :]
    for fn in (flux_form_vertical_momentum_advection_centered,
               flux_form_vertical_momentum_advection):
        tend = np.asarray(fn(u3, w3, h3))[0, 0]
        col_mom = float(np.sum(tend * np.asarray(dz)))  # Σ (du/dt)·h
        scale = float(np.sum(np.abs(tend) * np.asarray(dz))) + 1e-30
        assert abs(col_mom) / scale < 1e-12, (
            f"{fn.__name__}: column momentum tendency {col_mom:.3e} not ~0")


# ===========================================================================
# 4. ENERGY: centered ≈ KE-conserving; upwind strictly KE-dissipative
# ===========================================================================
def test_kinetic_energy_sign_difference():
    """KE-conservation property of the centered vs upwind vertical-advection
    flux, stated EXACTLY.

    The column-KE contribution of the flux-form vertical advection is
    ``dKE = Σ_k u[k]·(du/dt)[k]·h[k]``.  Summation-by-parts shows that the
    interface centered flux ``F[k]=0.5(u[k-1]+u[k])w[k]`` gives exactly the
    advective/continuity term ``dKE_continuity = -0.5 Σ_k u[k]²·(w[k]-w[k+1])``
    (which closes to zero once the model's continuity/metric term is added) and
    NO spurious dissipation.  The 1st-order UPWIND flux gives the same
    continuity term PLUS a strictly NEGATIVE numerical-viscosity term
    ``~|w|·dz/2·(du/dz)²`` that damps baroclinic shear.

    So: ``dKE_cen == dKE_continuity`` (to round-off) while
    ``dKE_up - dKE_continuity < 0`` (the upwind shear damping the centered
    scheme removes)."""
    from legoesm.ocean.vertical import (
        flux_form_vertical_momentum_advection_centered,
        flux_form_vertical_momentum_advection,
    )
    nlev = 15
    ddz = np.array([50, 70, 100, 140, 190, 240, 290, 340, 390,
                    440, 490, 540, 590, 640, 690], dtype=np.float64) / 2.5
    dz = jnp.asarray(ddz)
    z_c = -(np.cumsum(ddz) - 0.5 * ddz)
    H = float(np.sum(ddz))
    # Strongly sheared full velocity (so the KE budget is non-degenerate).
    u = jnp.asarray(0.30 * np.exp(z_c / 600.0) + 0.10)
    z_half = np.concatenate([[0.0], -np.cumsum(ddz)])
    w_half = jnp.asarray(-3.0e-5 * np.sin(np.pi * (-z_half) / H))

    u3, w3, h3 = u[None, None, :], w_half[None, None, :], dz[None, None, :]
    tend_cen = np.asarray(
        flux_form_vertical_momentum_advection_centered(u3, w3, h3))[0, 0]
    tend_up = np.asarray(
        flux_form_vertical_momentum_advection(u3, w3, h3))[0, 0]

    uu = np.asarray(u)
    ddz_np = np.asarray(dz)
    ww = np.asarray(w_half)
    dKE_cen = float(np.sum(uu * tend_cen * ddz_np))
    dKE_up = float(np.sum(uu * tend_up * ddz_np))
    # The advective/continuity-coupled KE term (the part that closes to zero
    # once the model adds u·(continuity divergence)).
    div_w = ww[:-1] - ww[1:]                      # w[k] - w[k+1] per cell
    dKE_continuity = float(np.sum(-0.5 * uu ** 2 * div_w))
    ke_scale = float(np.sum(uu ** 2 * ddz_np)) + 1e-30

    # Centered: NO spurious dissipation — equals the continuity term exactly.
    assert abs(dKE_cen - dKE_continuity) / ke_scale < 1e-14, (
        f"centered KE residual {dKE_cen - dKE_continuity:.3e} not ~0 "
        "(centered should carry no implicit vertical viscosity)")
    # Upwind: same continuity term PLUS a strictly NEGATIVE numerical-viscosity
    # term — the shear damping.
    dissip_up = dKE_up - dKE_continuity
    assert dissip_up < 0.0, (
        f"upwind numerical-viscosity KE term {dissip_up:.3e} should be < 0")
    assert abs(dissip_up) / ke_scale > 1e-10, (
        "upwind implicit-viscosity dissipation should be measurable")
    # The centered scheme removes exactly that damping.
    assert dKE_cen - dKE_up > 0.0, (
        "centered should remove the upwind shear damping (dKE_cen > dKE_up)")


# ===========================================================================
# 5. DISPATCH VALIDATION: unknown literal raises; combo rejected
# ===========================================================================
def test_unknown_literal_raises():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    with pytest.raises(ValueError, match="vertical_momentum_scheme must be one of"):
        LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(
            vertical_momentum_scheme="centred"))   # typo


@pytest.mark.parametrize("scheme", ["centered_full", "nemo_advective"])
def test_explicit_scheme_with_adaptive_implicit_rejected(scheme):
    """centered_full / nemo_advective + adaptive_implicit_vertadv=True is
    incompatible (the adaptive-implicit path replaces the explicit stage
    entirely) and must fail fast at config validation rather than silently
    no-op."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    grid = create_latlon_grid(_N_LAT, _N_LON)
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    with pytest.raises(ValueError, match="incompatible with adaptive_implicit_vertadv"):
        LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(
            vertical_momentum_scheme=scheme,
            adaptive_implicit_vertadv=True))


# ===========================================================================
# 7. #1226 NEMO dynzad.F90 LITERAL LOOP-PORT (independent ground truth)
# ===========================================================================
def _dynzad_reference(u, Wf, e3u):
    """Literal Python transcription of NEMO dynzad.F90:83-118 on a SINGLE
    column (Fortran 1-indexed jk in the source; this port is 0-indexed and
    mirrors every line, NOT the vectorized implementation under test).

    ``u`` : full velocity at t/u/v-levels, shape (nlev,) (NEMO ``uu``/``vv``).
    ``Wf``: e1e2t-area-weighted w ALREADY interpolated to the momentum-point
        face at each w-level interface, shape (nlev+1,) (NEMO
        ``e1e2t*ww`` averaged to the u/v-point — i.e. this port receives
        the interpolated field directly and reproduces only dynzad's OWN
        arithmetic: the ``zWfi+zWf`` sum, the ``0.25`` prefactor, and the
        carried top/bottom interface bookkeeping).  Index k here = NEMO's
        w-point index jk (1-indexed) minus 1, so ``Wf[0]`` is NEMO's
        ``ww(1)`` (surface, must be 0) and ``Wf[nlev]`` is ``ww(nlev+1)``
        (bottom, architecturally 0, never read below).
    ``e3u``: layer thickness at the momentum point, shape (nlev,).

    Returns the per-level tendency EXCLUDING the ``r1_e1e2u`` (face-area)
    factor (dynzad.F90's own arithmetic is expressed as ``0.25/e3u`` times
    the bracket; the caller applies ``r1_e1e2u`` separately, matching how
    this test isolates the u*dw/dz identity from the face-area weighting).
    """
    nlev = len(u)
    tend = np.zeros(nlev)
    zWdzU = 0.0                       # dynzad.F90:83, "surface (jk=1) = 0"
    for jk in range(0, nlev - 1):      # Fortran jk=1..jpk-2 -> Python 0..nlev-2
        zzWfu = 2.0 * Wf[jk + 1]       # zWfi+zWf = 2x the simple average
        zzWdzU = zzWfu * (u[jk] - u[jk + 1])
        tend[jk] = -0.25 / e3u[jk] * (zWdzU + zzWdzU)
        zWdzU = zzWdzU                 # dynzad.F90:109, carried to next jk
    jk = nlev - 1                      # Fortran jk = jpkm1
    tend[jk] = -0.25 / e3u[jk] * zWdzU  # dynzad.F90:113-118, bottom special case
    return tend


def _manufactured_column(nlev=10, seed=0):
    rng = np.random.default_rng(seed)
    ddz = rng.uniform(40.0, 300.0, nlev)
    z_half = np.concatenate([[0.0], -np.cumsum(ddz)])
    H = float(np.sum(ddz))
    u = rng.normal(0.0, 0.25, nlev)
    # w on half-levels: 0 at surface and bottom, smooth in between.
    Wf = -2.0e-5 * np.sin(np.pi * (-z_half) / H)
    Wf[0] = 0.0
    Wf[-1] = 0.0
    return u, Wf, ddz


def test_nemo_advective_matches_literal_dynzad_port():
    """The new 'nemo_advective' scheme must match the independent literal
    dynzad.F90 loop-port to machine precision (face_area=1 isolates the
    dynzad arithmetic from the separate e1e2u/e1e2t area-weighting)."""
    from legoesm.ocean.vertical import nemo_advective_vertical_momentum_advection
    u, Wf, ddz = _manufactured_column(nlev=12, seed=1)
    ref = _dynzad_reference(u, Wf, ddz)

    u3 = jnp.asarray(u)[None, None, :]
    Wf3 = jnp.asarray(Wf)[None, None, :]
    h3 = jnp.asarray(ddz)[None, None, :]
    face_area = jnp.ones((1, 1, 1))
    got = np.asarray(
        nemo_advective_vertical_momentum_advection(u3, Wf3, h3, face_area))[0, 0]
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-18)


def test_nemo_advective_face_area_scales_inversely():
    """face_area enters as a pure 1/area_u prefactor (NEMO's r1_e1e2u), so
    doubling it must exactly halve the tendency."""
    from legoesm.ocean.vertical import nemo_advective_vertical_momentum_advection
    u, Wf, ddz = _manufactured_column(nlev=9, seed=2)
    u3 = jnp.asarray(u)[None, None, :]
    Wf3 = jnp.asarray(Wf)[None, None, :]
    h3 = jnp.asarray(ddz)[None, None, :]
    got_1 = nemo_advective_vertical_momentum_advection(
        u3, Wf3, h3, jnp.full((1, 1, 1), 3.0e8))
    got_2 = nemo_advective_vertical_momentum_advection(
        u3, Wf3, h3, jnp.full((1, 1, 1), 6.0e8))
    np.testing.assert_allclose(np.asarray(got_1), 2.0 * np.asarray(got_2), rtol=1e-13)


def test_old_flux_form_schemes_do_not_match_dynzad(monkeypatch=None):
    """Non-vacuous check: the OLD flux-form options ('upwind_perturbation',
    'centered_full') must NOT match the literal dynzad.F90 port on a column
    where w has interior vertical structure (dw/dz != 0) — confirming the
    #1226 diagnosis (they compute d(w*u)/dz, not w*du/dz, and differ by
    u*dw/dz at every interior level)."""
    from legoesm.ocean.vertical import (
        flux_form_vertical_momentum_advection,
        flux_form_vertical_momentum_advection_centered,
    )
    u, Wf, ddz = _manufactured_column(nlev=12, seed=1)
    ref = _dynzad_reference(u, Wf, ddz)

    # The flux-form helpers take w_half directly (not area-weighted); on
    # this synthetic column area weighting is identity (face_area=1 above),
    # so w_half == Wf here (isolates the FORM difference, not area effects).
    u3 = jnp.asarray(u)[None, None, :]
    w3 = jnp.asarray(Wf)[None, None, :]
    h3 = jnp.asarray(ddz)[None, None, :]

    for fn in (flux_form_vertical_momentum_advection,
               flux_form_vertical_momentum_advection_centered):
        got = np.asarray(fn(u3, w3, h3))[0, 0]
        # Interior levels (exclude the top/bottom levels, where both forms
        # degenerate toward the same boundary zeros) must show a REAL
        # mismatch, not just round-off.
        interior = slice(1, len(u) - 1)
        resid = got[interior] - ref[interior]
        scale = np.max(np.abs(ref[interior])) + 1e-30
        assert np.max(np.abs(resid)) / scale > 1e-3, (
            f"{fn.__name__} unexpectedly matches the advective dynzad form "
            "(non-vacuous check failed)")


def test_flux_vs_advective_difference_is_u_times_dwdz():
    """The #1226 diagnosis identity: FLUX form d(w*u)/dz minus ADVECTIVE
    form w*du/dz equals -u*dw/dz at every interior level (product rule:
    d(w*u)/dz = w*du/dz + u*dw/dz, so flux_form - advective_form ==
    -(u*dw/dz)). Verified via the dynzad-vs-centered_full residual against
    a centered-difference dw/dz estimate, to the discretization's own
    truncation error."""
    from legoesm.ocean.vertical import flux_form_vertical_momentum_advection_centered
    u, Wf, ddz = _manufactured_column(nlev=14, seed=3)
    ref_advective = _dynzad_reference(u, Wf, ddz)

    u3 = jnp.asarray(u)[None, None, :]
    w3 = jnp.asarray(Wf)[None, None, :]
    h3 = jnp.asarray(ddz)[None, None, :]
    flux_form = np.asarray(
        flux_form_vertical_momentum_advection_centered(u3, w3, h3))[0, 0]

    residual = flux_form - ref_advective   # should be ~= -(u * dw/dz)

    # Discrete dw/dz at level k: (w_half[k] - w_half[k+1]) / dz[k]
    dwdz = (Wf[:-1] - Wf[1:]) / ddz
    predicted = -(u * dwdz)

    interior = slice(1, len(u) - 1)
    corr = np.corrcoef(residual[interior], predicted[interior])[0, 1]
    assert corr > 0.9, f"residual vs -(u*dw/dz) correlation too low: {corr:.4f}"
    ratio = (np.sum(residual[interior] * predicted[interior])
             / np.sum(predicted[interior] ** 2))
    assert 0.5 < ratio < 1.5, f"residual/predicted amplitude ratio off: {ratio:.4f}"


# ===========================================================================
# 8. DINO CARD SELECTION (#1226): nemo_dino_kamm(+_mlf) select nemo_advective
# ===========================================================================
@pytest.mark.parametrize("recipe", ["nemo_dino_kamm", "nemo_dino_kamm_mlf"])
def test_dino_kamm_cards_select_nemo_advective(recipe):
    from legoesm.ocean.experiments.dino import dino_config_for_recipe
    cfg = dino_config_for_recipe(recipe)
    assert cfg.vertical_momentum_scheme == "nemo_advective"


def test_dino_default_card_unchanged():
    """Non-kamm recipes keep the legacy default (bit-identical guarantee)."""
    from legoesm.ocean.experiments.dino import dino_config_for_recipe, DINOConfig
    assert DINOConfig().vertical_momentum_scheme == "upwind_perturbation"
    assert (dino_config_for_recipe("legoesm_default").vertical_momentum_scheme
            == "upwind_perturbation")


# ===========================================================================
# 9. nemo_advective WIRING: changes trajectory, differentiable, dispatch OK
# ===========================================================================
def test_nemo_advective_changes_trajectory():
    s_up, m_up = _basin("upwind_perturbation")
    s_na, m_na = _basin("nemo_advective")
    f_up, _ = m_up.integrate_scan(s_up, n_steps=8, dt=_DT)
    f_na, _ = m_na.integrate_scan(s_na, n_steps=8, dt=_DT)
    assert np.all(np.isfinite(np.asarray(f_na.u.data)))
    du = float(np.max(np.abs(np.asarray(f_na.u.data) - np.asarray(f_up.u.data))))
    umax = float(np.max(np.abs(np.asarray(f_up.u.data))))
    assert du > 1e-6 * max(umax, 1e-12), (
        f"nemo_advective did not change the trajectory (max|du|={du:.3e})")


def test_nemo_advective_differentiable():
    state, model = _basin("nemo_advective")
    T0 = state.T.data

    def loss(scale):
        st = state._replace(T=state.T.replace(data=T0 * scale))
        f, _ = model.integrate_scan(st, n_steps=2, dt=_DT)
        return jnp.sum(f.u.data ** 2) + jnp.sum(f.T.data ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0


# ===========================================================================
# 6. DIFFERENTIABILITY: jax.grad finite + nonzero through 2 steps
# ===========================================================================
def test_centered_full_differentiable():
    """end-to-end jax.grad through integrate_scan with centered_full is finite
    and nonzero (the dispersive centered flux preserves smooth gradients)."""
    state, model = _basin("centered_full")
    T0 = state.T.data

    def loss(scale):
        st = state._replace(T=state.T.replace(data=T0 * scale))
        f, _ = model.integrate_scan(st, n_steps=2, dt=_DT)
        return jnp.sum(f.u.data ** 2) + jnp.sum(f.T.data ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0

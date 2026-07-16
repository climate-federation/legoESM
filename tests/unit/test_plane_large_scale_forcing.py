"""Large-scale forcing for the plane CRM (iter-31, D3).

Covers both the shared array-level upwind operator
(:mod:`legoesm.atmosphere.forcing.idealized.large_scale_forcing`) and the plane physics_fn
(:mod:`legoesm.atmosphere.forcing.plane_large_scale_forcing`) that applies
SAM-style subsidence + prescribed advective tendencies + wind nudging to the
double-periodic f-plane state.

SAM oracle: ``subsidence.f90`` advects every scalar by ``-w_sub·∂φ/∂z`` with
first-order upwind in z (donor selected by sign of ``w_sub``); ``forcing.f90``
adds prescribed ``ttend``/``qtend`` and nudges winds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.forcing.idealized.large_scale_forcing import (  # noqa: E402
    mask_inflow_endpoint_tendency,
    subsidence_tendency_top2bottom,
    upwind_dphi_dz_top2bottom,
)
from legoesm.atmosphere.forcing.plane_large_scale_forcing import (  # noqa: E402
    make_plane_ls_forcing_physics,
)
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import PlaneNonHydrostaticState  # noqa: E402
from legoesm.grids.vertical import create_height_coordinate  # noqa: E402


# --------------------------------------------------------------------------
# Array-level upwind operator
# --------------------------------------------------------------------------

# Top-to-bottom: index 0 = top, index nlev-1 = surface; z decreases with index.
_Z = jnp.asarray([3000.0, 2000.0, 1000.0, 0.0])
_GAMMA = 0.004  # K/m stable lapse of theta
_THETA = 300.0 + _GAMMA * _Z          # [312, 308, 304, 300], increases upward


def test_upwind_gradient_linear_profile():
    """Linear θ(z): ∂θ/∂z = Γ everywhere, independent of donor side."""
    for w in (jnp.full(4, 0.01), jnp.full(4, -0.01)):
        grad = upwind_dphi_dz_top2bottom(_THETA, _Z, w)
        assert np.allclose(np.asarray(grad), _GAMMA, atol=1e-9)


def test_subsidence_warms_under_stable_stratification():
    """Subsidence (w_ls<0) under stable θ ⇒ adiabatic warming (interior)."""
    w_ls = jnp.full(4, -0.01)         # subsiding
    tend = subsidence_tendency_top2bottom(_THETA, _Z, w_ls)
    # interior k=1,2: -w_ls·Γ = 0.01·0.004 = 4e-5 K/s, warming
    assert float(tend[1]) == pytest.approx(4.0e-5, rel=1e-6)
    assert float(tend[2]) == pytest.approx(4.0e-5, rel=1e-6)
    assert float(tend[1]) > 0.0


def test_subsidence_zeros_both_endpoints_sam_parity():
    """SAM computes subsidence only on interior k=2..nzm-1; both the top and
    surface tendency stay zero (codex iter-31 C)."""
    w_ls = jnp.full(4, -0.01)
    tend = subsidence_tendency_top2bottom(_THETA, _Z, w_ls)
    assert float(tend[0]) == 0.0          # top zeroed
    assert float(tend[-1]) == 0.0         # surface zeroed (SAM parity)
    assert float(tend[1]) != 0.0          # interior active


def test_mask_surface_upward_inflow():
    """Surface with w>0 = upward inflow ⇒ masked; top kept."""
    w = jnp.full(4, 0.02)
    tend = mask_inflow_endpoint_tendency(jnp.ones(4), w)
    assert float(tend[-1]) == 0.0         # surface masked
    assert float(tend[0]) == 1.0          # top kept (w>0 has donor below)


def test_upwind_donor_selection_on_kink():
    """A kinked profile resolves different gradients for w>0 vs w<0."""
    phi = jnp.asarray([10.0, 4.0, 1.0, 0.0])   # convex kink
    # From-below at k=1 uses (phi[1]-phi[2])/(z[1]-z[2]) = 3/1000
    g_below = upwind_dphi_dz_top2bottom(phi, _Z, jnp.full(4, 1.0))
    # From-above at k=1 uses (phi[0]-phi[1])/(z[0]-z[1]) = 6/1000
    g_above = upwind_dphi_dz_top2bottom(phi, _Z, jnp.full(4, -1.0))
    assert float(g_below[1]) == pytest.approx(3.0 / 1000.0)
    assert float(g_above[1]) == pytest.approx(6.0 / 1000.0)


# --------------------------------------------------------------------------
# Plane physics_fn
# --------------------------------------------------------------------------

def _hc(nlev=12, H=15000.0):
    return create_height_coordinate(
        nlev, H, theta_ref_fn=lambda z: 300.0 + 0.004 * z,
    )


def _state(hc, ny=2, nx=2, n_tr=6, theta_p=0.0, u0=0.0, v0=0.0, qv_grad=False):
    nlev = hc.n_levels
    sd = jnp.float64
    th = jnp.full((ny, nx, nlev), theta_p, dtype=sd)
    u = jnp.full((ny, nx, nlev), u0, dtype=sd)
    v = jnp.full((ny, nx, nlev), v0, dtype=sd)
    w = jnp.zeros((ny, nx, nlev + 1), dtype=sd)
    rho_p = jnp.zeros((ny, nx, nlev), dtype=sd)
    phis = jnp.zeros((ny, nx), dtype=sd)
    tr = jnp.zeros((ny, nx, nlev, n_tr), dtype=sd)
    if qv_grad and n_tr > 0:
        # q_v decreasing upward (slot 0): a 1/z-ish moist boundary layer.
        qv = (0.015 * (hc.z_full[-1] + 1.0) / (hc.z_full + 1.0)).astype(sd)
        tr = tr.at[..., 0].set(qv.reshape(1, 1, nlev))
    f = lambda d, nm, dims=("y", "x", "z"): Field(data=d, name=nm, dims=dims)
    return PlaneNonHydrostaticState(
        u=f(u, "u"), v=f(v, "v"),
        w=f(w, "w", ("y", "x", "z_half")),
        theta_prime=f(th, "theta_prime"),
        rho_prime=f(rho_p, "rho_prime"),
        phis=f(phis, "phis", ("y", "x")),
        tracers=f(tr, "tracers", ("y", "x", "z", "tracer")),
    )


def test_plane_subsidence_warms_interior():
    hc = _hc()
    w_ls = jnp.full(hc.n_levels, -0.01)        # subsidence
    fn = make_plane_ls_forcing_physics(hc, w_ls=w_ls)
    out = fn(_state(hc), None, hc, None)
    dth = np.asarray(out.dtheta_prime_dt.data)
    # interior columns warm (stable θ_ref, subsidence): positive, ~ -w·Γ
    assert np.all(dth[:, :, 2:-1] > 0.0)
    assert dth[0, 0, 6] == pytest.approx(0.01 * 0.004, rel=1e-3)


def test_plane_advective_tendencies_passthrough():
    hc = _hc()
    th_adv = jnp.full(hc.n_levels, -1.0e-4)    # K/s cooling
    qv_adv = jnp.full(hc.n_levels, 2.0e-7)     # (kg/kg)/s moistening
    fn = make_plane_ls_forcing_physics(hc, theta_adv=th_adv, qv_adv=qv_adv)
    out = fn(_state(hc), None, hc, None)
    assert np.allclose(np.asarray(out.dtheta_prime_dt.data), -1.0e-4)
    assert np.allclose(np.asarray(out.dtracers_dt.data[..., 0]), 2.0e-7)
    # other tracer slots untouched by advection
    assert np.allclose(np.asarray(out.dtracers_dt.data[..., 1:]), 0.0)


def test_plane_wind_nudging():
    hc = _hc()
    u_tgt = jnp.full(hc.n_levels, 10.0)
    fn = make_plane_ls_forcing_physics(
        hc, u_nudge=u_tgt, v_nudge=jnp.zeros(hc.n_levels), tau_nudge=3600.0,
    )
    out = fn(_state(hc, u0=4.0, v0=0.0), None, hc, None)
    # du/dt = -(u_mean - u_tgt)/τ = -(4 - 10)/3600 = +6/3600 (uniform field)
    assert np.allclose(np.asarray(out.du_dt.data), 6.0 / 3600.0)
    assert np.allclose(np.asarray(out.dv_dt.data), 0.0)   # v already on target


def test_plane_wind_nudging_relaxes_mean_preserves_eddies():
    """SAM donudging_uv nudges the DOMAIN-MEAN wind only — the same correction
    is applied to every column, so wind PERTURBATIONS (eddies) are untouched."""
    hc = _hc(nlev=4)
    nlev = hc.n_levels
    # u with a horizontal eddy: column means are 5 m/s at every level.
    u = jnp.zeros((2, 2, nlev))
    u = u.at[0, :, :].set(2.0).at[1, :, :].set(8.0)   # mean 5, ±3 eddy
    st = _state(hc, u0=0.0)
    st = st._replace(u=st.u.replace(data=u))
    u_tgt = jnp.full(nlev, 5.0)        # target == current mean
    fn = make_plane_ls_forcing_physics(hc, u_nudge=u_tgt, tau_nudge=3600.0)
    du = np.asarray(fn(st, None, hc, None).du_dt.data)
    # mean already on target ⇒ ZERO nudging everywhere (eddies NOT damped)
    assert np.allclose(du, 0.0)
    # now shift the target: a UNIFORM correction (mean-based), identical in
    # every column regardless of the local eddy value
    fn2 = make_plane_ls_forcing_physics(
        hc, u_nudge=jnp.full(nlev, 11.0), tau_nudge=3600.0)
    du2 = np.asarray(fn2(st, None, hc, None).du_dt.data)
    assert np.allclose(du2, -(5.0 - 11.0) / 3600.0)   # = +6/3600 in ALL columns
    assert np.allclose(du2[0], du2[1])                # eddy-independent


def test_plane_subsidence_on_all_tracers():
    """SAM advects all micro fields, not just q_v: a tracer with a vertical
    gradient must receive a nonzero subsidence tendency."""
    hc = _hc()
    w_ls = jnp.full(hc.n_levels, -0.02)
    fn = make_plane_ls_forcing_physics(hc, w_ls=w_ls, subsidence_on_tracers=True)
    out = fn(_state(hc, qv_grad=True), None, hc, None)
    dqv = np.asarray(out.dtracers_dt.data[..., 0])
    assert np.any(np.abs(dqv[:, :, 1:-1]) > 0.0)


def test_plane_forcing_ad_safe():
    hc = _hc(nlev=8)
    w_ls = jnp.full(hc.n_levels, -0.01)
    fn = make_plane_ls_forcing_physics(hc, w_ls=w_ls, theta_adv=jnp.full(8, 1e-4))

    def loss(theta_p_scalar):
        st = _state(hc, theta_p=theta_p_scalar)
        out = fn(st, None, hc, None)
        return jnp.sum(out.dtheta_prime_dt.data ** 2)

    g = jax.grad(loss)(jnp.asarray(0.5))
    assert bool(jnp.isfinite(g))


def test_no_active_channel_raises():
    hc = _hc()
    with pytest.raises(ValueError, match="no forcing channel is active"):
        make_plane_ls_forcing_physics(hc)


def test_profile_shape_validation():
    hc = _hc(nlev=12)
    with pytest.raises(ValueError, match="expected"):
        make_plane_ls_forcing_physics(hc, w_ls=jnp.zeros(5))


def test_nudge_requires_tau():
    hc = _hc()
    with pytest.raises(ValueError, match="positive tau_nudge"):
        make_plane_ls_forcing_physics(hc, u_nudge=jnp.zeros(hc.n_levels))


_LSF_SYNTH = """ z[m] p[mb] tls qls uls vls wls
 0., 2, 1000.  day,levels,pres0
     0.0  1000.0   0.1000E-04   0.2000E-08   -1.0   0.0   -0.010
 20000.0  100.0    0.5000E-04   0.1000E-08   -5.0   0.0   -0.050
"""


def test_forcing_from_sam_case_file(tmp_path):
    """End-to-end: SAM lsf file → interpolated profiles → forcing physics_fn
    producing nonzero subsidence + advective tendencies on the plane state."""
    from legoesm.atmosphere.forcing.plane_large_scale_forcing import (
        make_plane_ls_forcing_from_sam_case,
    )
    p = tmp_path / "lsf"
    p.write_text(_LSF_SYNTH)
    hc = _hc()
    fn = make_plane_ls_forcing_from_sam_case(hc, p, day=0.0)
    out = fn(_state(hc, qv_grad=True), None, hc, None)
    dth = np.asarray(out.dtheta_prime_dt.data)
    dqv = np.asarray(out.dtracers_dt.data[..., 0])
    # advective + subsidence both contribute ⇒ nonzero θ and q_v tendencies
    assert np.any(np.abs(dth[:, :, 1:-1]) > 0.0)
    assert np.any(np.abs(dqv[:, :, 1:-1]) > 0.0)
    # θ advective tendency is positive (tls>0 everywhere here)
    assert np.all(dth[:, :, 1:-1] > 0.0)


# constant tls so the height interpolation is trivial (T_adv = const).
_LSF_CONST_T = """ z[m] p[mb] tls qls uls vls wls
 0., 2, 1000.  day,levels,pres0
     0.0  1000.0   0.1000E-04   0.0000E+00   0.0   0.0   0.0
 20000.0  100.0    0.1000E-04   0.0000E+00   0.0   0.0   0.0
"""


def test_forcing_from_sam_case_converts_tls_T_to_theta_via_exner(tmp_path):
    """FORCING-T (iter-55): SAM's lsf ``tls`` is an ABSOLUTE-temperature
    tendency (``forcing.f90`` adds it to the static-energy ``t``≈tabs, no
    ``/prespot``). The SAM-case adapter must convert it to legoESM's θ tendency
    via ``dθ/dt = (dT/dt)/exner_ref`` — applying it RAW (the old bug) under-cools
    by the exner factor (~22 % at 500 mb, worse aloft), biasing the GATE T
    profile warm. Subsidence OFF isolates the advective channel."""
    from legoesm.atmosphere.forcing.plane_large_scale_forcing import (
        make_plane_ls_forcing_from_sam_case,
    )
    p = tmp_path / "lsf"
    p.write_text(_LSF_CONST_T)
    hc = _hc()
    fn = make_plane_ls_forcing_from_sam_case(
        hc, p, day=0.0, include_subsidence=False, nudge_winds=False)
    out = fn(_state(hc), None, hc, None)
    dth = np.asarray(out.dtheta_prime_dt.data)[0, 0]       # (nlev,)
    exner = np.asarray(hc.exner_ref).astype(dth.dtype)     # match sim dtype
    tls = 1.0e-5                                            # 0.1000E-04 in file
    # the FIX: dθ/dt = tls / exner_ref (per level). rtol tracks the sim dtype
    # (float64 here under jax_enable_x64) so this tests the physics, not the
    # numpy-vs-jax precision config (codex iter-55 LOW).
    rtol = 1e-6 if dth.dtype == np.float32 else 1e-11
    np.testing.assert_allclose(dth, tls / exner, rtol=rtol, atol=0.0)
    # the OLD BUG would give dθ/dt = tls (raw); exner<1 aloft ⇒ the converted
    # value is strictly LARGER there, so a raw-apply is detectably wrong.
    assert np.all(dth >= tls - 1e-18)
    assert np.max(dth) > tls * 1.1           # ≥10 % larger somewhere aloft

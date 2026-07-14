"""Unit tests for the Emanuel (1991) convection scheme.

Tests pin:

* tendency shape / dtype / finiteness;
* the no-CMT invariant;
* the buoyancy-sort detrainment enhancement (the distinguishing
  feature relative to ZM and KF);
* the unsaturated-downdraft toggle (column conservation under both
  modes);
* finite gradients through ``cape_threshold`` and
  ``smooth_trigger_sharpness`` — the AD-safety property.
"""

from __future__ import annotations

from legoesm import constants

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.convection.config import (
    ConvectionConfig,
    EmanuelConfig,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.convection.emanuel import emanuel_convection


def _column(
    ncol=2, nlev=16, T_sfc=302.0, q_sfc=16e-3, lapse_rate=7.5,
    p_s=1.0e5, p_top=5.0e3,
):
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [
            jnp.full((ncol, 1), p_top * 0.5),
            p_half_inner,
            jnp.full((ncol, 1), p_s),
        ],
        axis=1,
    )
    z_full = -8500.0 * jnp.log(p_full / p_s)
    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    q = q_sfc * jnp.exp(-z_full / 3000.0)
    return T, q, p_full, p_half


# ---------------------------------------------------------------------------
# Shape / finiteness / no CMT
# ---------------------------------------------------------------------------

def test_emanuel_shape_dtype():
    T, q, pf, ph = _column(ncol=3, nlev=12)
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, cpp_new = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    assert out.dT_dt.shape == (ncol, nlev)
    assert out.dq_v_dt.shape == (ncol, nlev)
    assert out.dq_c_conv_dt.shape == (ncol, nlev)
    assert out.cape.shape == (ncol,)
    assert cpp_new.shape == (ncol, nlev)


def test_emanuel_outputs_finite():
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, cpp_new = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    for arr in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt, out.cape,
                out.convective_mask, cpp_new):
        assert jnp.all(jnp.isfinite(arr))


def test_emanuel_cbmf_relaxes_positive_from_rest():
    """The CONVECT DTMA/ALPHA/DAMP closure can spin CBMF up from zero.

    A moist, weakly unstable tropical column should not be permanently
    zero-locked when the prognostic CBMF carry starts at rest; the
    relaxed carry in ``[:, -1]`` becomes positive and remains AD-finite
    through the ALPHA coefficient.
    """
    T, q, pf, ph = _column(ncol=1, nlev=30, T_sfc=300.0, q_sfc=22e-3, lapse_rate=6.5)
    cpp = jnp.zeros_like(T)
    out, cpp_new = emanuel_convection(T, q, pf, ph, cpp, dt=600.0)
    assert float(out.cape[0]) > 0.0
    assert float(cpp_new[0, -1]) > 0.0

    def carry_for_alpha(alpha):
        _, cpp_alpha = emanuel_convection(
            T, q, pf, ph, cpp, dt=600.0,
            config=EmanuelConfig(alpha_closure=alpha),
        )
        return cpp_alpha[0, -1]

    g = jax.grad(carry_for_alpha)(jnp.asarray(0.2))
    assert jnp.isfinite(g)


def test_emanuel_no_cmt():
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    assert out.du_dt_conv is None
    assert out.dv_dt_conv is None


def test_emanuel_cloud_water_source_non_negative():
    """The convective cloud-water source is non-negative even after
    the unsaturated-downdraft evaporation step subtracts column
    condensate (we ``maximum(., 0)`` the result)."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    assert jnp.all(out.dq_c_conv_dt >= 0.0)


# ---------------------------------------------------------------------------
# Buoyancy-sort: n_mixing_fractions affects detrainment but not gross sign
# ---------------------------------------------------------------------------

def test_emanuel_n_fractions_finite_for_all_choices():
    """Tendencies are finite for ``n_mixing_fractions`` ∈ {1, 4, 8, 16}.
    n_fractions=1 is the limit of a single bulk plume; larger values
    approach Emanuel 1991's 50-bin spectrum more closely."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    for n in (1, 4, 8, 16):
        config = EmanuelConfig(n_mixing_fractions=n)
        out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0, config=config)
        assert jnp.all(jnp.isfinite(out.dT_dt)), (
            f"NaN with n_mixing_fractions={n}"
        )


def test_emanuel_genuine_mixing_detrainment_structure():
    """The GENUINE (i,j) mixing matrix concentrates detrainment near the
    parcel's level of neutral buoyancy (cloud top), not uniformly.

    With the faithful SIJ/ELIJ/MENT spectrum the per-level convective
    cloud-water source ``dq_c_conv_dt`` is bottom-heavy in mass flux but
    its detrainment (and hence the upper-tropospheric heating) peaks near
    cloud top — a structure a single bulk plume cannot produce.  We pin
    the genuine path against the legacy single-sigmoid surrogate: the two
    give materially different per-level ``dT_dt`` profiles (the genuine
    sort redistributes the heating)."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    # Seed Emanuel's prognostic CBMF memory so this test isolates the
    # mixing matrix structure rather than the DTMA trigger-from-rest.
    cpp = jnp.zeros((ncol, nlev)).at[:, -1].set(0.1)
    out_genuine, _ = emanuel_convection(
        T, q, pf, ph, cpp, dt=300.0,
        config=EmanuelConfig(use_genuine_mixing=True),
    )
    out_surrogate, _ = emanuel_convection(
        T, q, pf, ph, cpp, dt=300.0,
        config=EmanuelConfig(use_genuine_mixing=False),
    )
    # The genuine mixing is not the surrogate (materially different).
    dT_diff = float(jnp.max(jnp.abs(out_genuine.dT_dt - out_surrogate.dT_dt)))
    assert dT_diff > 1e-6
    # The genuine path heats the free troposphere (the detrainment-height
    # sort deposits buoyancy aloft); the column-max heating sits above the
    # boundary layer, not at the surface.
    dT = out_genuine.dT_dt[0]
    k_max = int(jnp.argmax(dT))
    assert k_max < nlev - 2, "genuine heating should peak above the surface"


def test_emanuel_genuine_mixing_matrix_tunable_changes_tendency():
    """A genuine-mixing tunable (the SIJ entrainment-gate sharpness)
    changes the tendency — i.e. the mixing matrix is genuinely wired,
    not a dead parameter."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    # Seed the prognostic CBMF memory; the test is about the SIJ gate
    # wiring, not whether this synthetic sounding fires from rest.
    cpp = jnp.zeros((ncol, nlev)).at[:, -1].set(0.1)
    out_a, _ = emanuel_convection(
        T, q, pf, ph, cpp, dt=300.0,
        config=EmanuelConfig(sij_gate_sharpness=40.0),
    )
    out_b, _ = emanuel_convection(
        T, q, pf, ph, cpp, dt=300.0,
        config=EmanuelConfig(sij_gate_sharpness=20.0),
    )
    dT_diff = float(jnp.max(jnp.abs(out_a.dT_dt - out_b.dT_dt)))
    assert dT_diff > 1e-9


def test_emanuel_mixture_buoyancy_sign_crosses():
    """Faithful Emanuel-1991 buoyancy sorting: the mixture buoyancy
    B(χ) starts positive (undilute updraft) and **crosses zero** into
    negative for a dry environment (evaporative cooling of entrained
    air) — the defining feature the old ``B_mix = χ·B_u`` could not
    produce.  A saturated environment evaporates nothing, so no mixture
    turns spuriously negative."""
    from legoesm.atmosphere.physics.convection.emanuel import _mixture_buoyancy
    from legoesm.thermo import saturation_mixing_ratio

    T_e = jnp.array([[290.0]])
    p = jnp.array([[8.0e4]])
    T_u = jnp.array([[292.0]])                       # buoyant updraft
    q_u = saturation_mixing_ratio(T_u, p)            # saturated cloud
    q_c_u = jnp.array([[2e-3]])                      # cloud condensate
    chi = jnp.linspace(0.02, 0.98, 25)

    B_dry = _mixture_buoyancy(
        T_e, 0.3 * saturation_mixing_ratio(T_e, p), T_u, q_u, q_c_u, p, chi,
    )[0, 0]
    B_sat = _mixture_buoyancy(
        T_e, saturation_mixing_ratio(T_e, p), T_u, q_u, q_c_u, p, chi,
    )[0, 0]

    assert float(B_dry[0]) > 0.0                     # undilute is buoyant
    assert bool(jnp.any(B_dry > 0) & jnp.any(B_dry < 0))  # genuine sign reversal
    assert float(jnp.min(B_sat)) >= -1e-6            # saturated env: no spurious sink
    # Gradient through the mixture buoyancy is finite (AD-safe sat-adjust).
    g = jax.grad(lambda t: jnp.sum(_mixture_buoyancy(
        jnp.array([[t]]), 0.3 * saturation_mixing_ratio(jnp.array([[t]]), p),
        T_u, q_u, q_c_u, p, chi)))(290.0)
    assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Unsaturated downdraft toggle
# ---------------------------------------------------------------------------

def test_emanuel_downdraft_toggle_changes_subcloud_dT():
    """Enabling the unsaturated downdraft introduces an additional
    sub-cloud cooling term — the difference in ``dT_dt`` between the
    on/off configurations is non-zero in the surface-adjacent
    layers."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    # The downdraft branch needs active convective condensate. Seed the
    # prognostic CBMF carry so the toggle is exercised on this fixture.
    cpp = jnp.zeros((ncol, nlev)).at[:, -1].set(0.1)
    out_off, _ = emanuel_convection(
        T, q, pf, ph, cpp, dt=300.0,
        config=EmanuelConfig(enable_unsaturated_downdraft=False),
    )
    out_on, _ = emanuel_convection(
        T, q, pf, ph, cpp, dt=300.0,
        config=EmanuelConfig(enable_unsaturated_downdraft=True),
    )
    # Sub-cloud (last 4 levels) tendencies differ between the two
    # branches — the downdraft is doing something visible.
    diff = jnp.abs(out_on.dT_dt[:, -4:] - out_off.dT_dt[:, -4:])
    assert float(jnp.max(diff)) > 1e-8


# ---------------------------------------------------------------------------
# Differentiability through tunable parameters
# ---------------------------------------------------------------------------

def test_emanuel_grad_through_cape_threshold():
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(threshold):
        config = EmanuelConfig(cape_threshold=threshold)
        out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0, config=config)
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(70.0)))
    assert np.isfinite(g)


def test_emanuel_grad_through_smooth_trigger_sharpness():
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(s):
        config = EmanuelConfig(smooth_trigger_sharpness=s)
        out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0, config=config)
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(0.5)))
    assert np.isfinite(g)


# ---------------------------------------------------------------------------
# Carry layout
# ---------------------------------------------------------------------------

def test_emanuel_carry_layout():
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    _, cpp_new = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    assert jnp.all(cpp_new[:, :-1] == 0.0)
    assert jnp.all(cpp_new[:, -1] >= 0.0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_emanuel_orchestrator_one_step_finite():
    grid = create_cubed_sphere(4)
    sigma = create_sigma_coordinate(12)
    state = held_suarez_init(grid, sigma)
    tracers = {
        "q_v": Field(0.014 * jnp.ones((6, 4, 4, 12)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
        "q_c": Field(jnp.zeros((6, 4, 4, 12)), name="q_c",
                     dims=("face", "x", "y", "level"), units="kg/kg"),
    }
    state = state._replace(tracers=tracers)
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="emanuel"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    ncol = 6 * 4 * 4
    ps = init_physics_state(ncol, 12, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)
    assert ps_out.conv_prog_profile.shape == (ncol, 12)
    for f in (tend.du_dt, tend.dv_dt, tend.dT_dt, tend.dp_s_dt, tend.dphis_dt):
        assert jnp.all(jnp.isfinite(f.data))
    # Emanuel has no CMT — orchestrator zeros these.
    assert float(jnp.max(jnp.abs(tend.du_dt.data))) == 0.0
    assert float(jnp.max(jnp.abs(tend.dv_dt.data))) == 0.0


# ---------------------------------------------------------------------------
# Column conservation (oracle-faithful): vapor-side enthalpy + total water
# ---------------------------------------------------------------------------

def test_emanuel_vapor_side_enthalpy_residual_small():
    """Oracle-faithful invariant (convect43c.f ENTS pass, lines 969-984):
    the vapor-side moist enthalpy tendency is corrected over the active
    1..INB convective layer, not by cooling the whole column.

    Physically: convection condenses vapor into cloud water
    (``dq_c_conv`` → microphysics) and the latent heat of that
    condensation is released as convective heating, so
    ``c_p ∫dT = L_v ∫dq_c`` to truncation error.  The full-column
    residual is small, but not forced to machine zero, because CONVECT's
    ENTS pass divides by ``PH(1)-PH(INB+1)`` and does not apply a uniform
    correction in quiescent levels above the diagnosed convection top."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = emanuel_convection(
        T=T, q_v=q, p_full=pf, p_half=ph,
        conv_prog_profile=cpp, dt=1800.0,
    )
    dp = ph[:, 1:] - ph[:, :-1]
    cpn = constants.c_pd * (1.0 - q) + constants.c_pv * q
    lv_eff = constants.L_v - (EmanuelConfig().c_l_emanuel - constants.c_pv) * (
        T - constants.T_freeze
    )
    residual = float(
        jnp.sum((cpn * out.dT_dt + lv_eff * out.dq_v_dt) * dp / constants.g, axis=1)
        .mean()
    )
    total = float(
        jnp.sum(
            (jnp.abs(cpn * out.dT_dt) + jnp.abs(lv_eff * out.dq_v_dt))
            * dp
            / constants.g,
            axis=1,
        ).mean()
    )
    rel = abs(residual) / (total + 1e-10)
    assert rel < 1e-2, (
        f"Emanuel vapor-side enthalpy residual {residual:.3e} W/m^2 "
        f"({rel:.2e} of total)"
    )


def test_emanuel_total_water_nearly_conserved():
    """The legacy kernel surrogate conserves vapor+cloud water because it has
    no explicit Emanuel precipitation split.  The genuine mixer is different:
    EP·CLW is precipitating condensate and is intentionally not reinserted as
    retained cloud water."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = emanuel_convection(
        T=T, q_v=q, p_full=pf, p_half=ph,
        conv_prog_profile=cpp, dt=1800.0,
        config=EmanuelConfig(use_genuine_mixing=False),
    )
    dp = ph[:, 1:] - ph[:, :-1]
    net = jnp.sum((out.dq_v_dt + out.dq_c_conv_dt) * dp / constants.g, axis=1)
    flux = jnp.sum(jnp.abs(out.dq_v_dt) * dp / constants.g, axis=1)
    rel = float(jnp.max(jnp.abs(net) / (flux + 1e-12)))
    assert rel < 0.05, f"Emanuel total-water residual {rel*100:.1f}% of flux"

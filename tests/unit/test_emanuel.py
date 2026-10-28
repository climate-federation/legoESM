"""Unit tests for the Emanuel (1991) convection scheme (CONVECT v4.3c port).

Fidelity to the running oracle is tests/atmosphere/test_emanuel_convect43c_oracle.py.
These pin shape / finiteness, the no-CMT invariant, the free-tropospheric
heating peak, the downdraught toggle, the refusal of retired surrogate fields,
the carry layout, the orchestrator, and column enthalpy / water closure.
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
    # Isothermal stratosphere: without one the moist static energy falls to
    # the model top, and CONVECT rejects the column (its IHMIN = NL-1 gate).
    T = jnp.maximum(T, 200.0)
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


@pytest.mark.parametrize("dd", (False, True))
def test_emanuel_condensate_goes_to_rain_not_cloud(dd):
    """As CONVECT (WDTRAIN), the mixture condensate AWAT detrained below its
    origin is precipitation: no cloud-water source, and the rain source
    carries the whole column vapour loss."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev)).at[:, -1].set(0.1)
    out, _ = emanuel_convection(
        T, q, pf, ph, cpp, dt=300.0,
        config=EmanuelConfig(enable_unsaturated_downdraft=dd))
    dpg = (ph[:, 1:] - ph[:, :-1]) / constants.g
    loss = -jnp.sum(out.dq_v_dt * dpg, axis=-1)
    assert float(jnp.max(loss)) > 0.0
    assert float(jnp.max(jnp.abs(out.dq_c_conv_dt))) == 0.0
    np.testing.assert_allclose(
        np.asarray(jnp.sum(out.dq_r_conv_dt * dpg, axis=-1)),
        np.asarray(jnp.maximum(loss, 0.0)), rtol=1e-10, atol=0.0)


def test_emanuel_downdraught_on_by_default():
    """The oracle always runs its downdraught: so does the default config."""
    assert EmanuelConfig().enable_unsaturated_downdraft is True


# ---------------------------------------------------------------------------
# Retired surrogate fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override", [
    dict(use_genuine_mixing=False), dict(n_mixing_fractions=4),
    dict(cu_coefficient=0.9), dict(delta_0=3e-4), dict(epsilon_0=1e-3),
    dict(smooth_trigger_sharpness=1.0), dict(parcel_perturb_T=0.0),
    dict(parcel_perturb_q=0.0), dict(precip_efficiency_water=0.5),
    dict(precip_efficiency_lcl=0.1), dict(mse_min_search_offset=4e4),
    dict(M_b_max=0.2), dict(cbmf_carry_max=1.0),
])
def test_emanuel_retired_surrogate_fields_are_refused(override):
    T, q, pf, ph = _column()
    with pytest.raises(ValueError, match="EmanuelConfig"):
        emanuel_convection(T, q, pf, ph, jnp.zeros_like(T), dt=300.0,
                           config=EmanuelConfig(**override))


def test_emanuel_genuine_mixing_detrainment_structure():
    """The (i,j) mixing matrix deposits its heating in the free troposphere:
    the column-maximum heating sits above the boundary layer."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    # Seed Emanuel's prognostic CBMF memory so this test isolates the
    # mixing matrix structure rather than the DTMA trigger-from-rest.
    cpp = jnp.zeros((ncol, nlev)).at[:, -1].set(0.1)
    out_genuine, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0)
    assert float(jnp.max(jnp.abs(out_genuine.dT_dt))) > 0.0
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
    """``cape_threshold`` shapes only the diagnostic convective_mask."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))

    def f(threshold):
        config = EmanuelConfig(cape_threshold=threshold)
        out, _ = emanuel_convection(T, q, pf, ph, cpp, dt=300.0, config=config)
        return jnp.sum(out.convective_mask)

    g = float(jax.grad(f)(jnp.asarray(70.0)))
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


def test_emanuel_total_water_conserved():
    """Vapour, cloud-water and rain sources integrate to zero over the column:
    every kg of vapour convection removes leaves as cloud water or rain."""
    T, q, pf, ph = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev)).at[:, -1].set(0.1)
    for dd in (False, True):
        out, _ = emanuel_convection(
            T=T, q_v=q, p_full=pf, p_half=ph, conv_prog_profile=cpp, dt=1800.0,
            config=EmanuelConfig(enable_unsaturated_downdraft=dd),
        )
        dp = ph[:, 1:] - ph[:, :-1]
        net = jnp.sum((out.dq_v_dt + out.dq_c_conv_dt + out.dq_r_conv_dt)
                      * dp / constants.g, axis=1)
        flux = jnp.sum(jnp.abs(out.dq_v_dt) * dp / constants.g, axis=1)
        assert float(jnp.max(flux)) > 0.0
        rel = float(jnp.max(jnp.abs(net) / (flux + 1e-30)))
        assert rel < 1e-12, f"Emanuel total-water residual {rel:.2e} of flux (dd={dd})"

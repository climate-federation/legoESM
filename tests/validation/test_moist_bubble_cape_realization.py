"""Behavioral test: the plane CRM REALIZES CAPE via M2005 latent heating.

Codex iter-73 flagged the M2005 -> theta' -> buoyancy coupling as the weakest
"verified faithful" component (formula-match is weaker than behavior-match) and
proposed a decisive CIN-bypass diagnostic: place a SATURATED warm bubble just
ABOVE the LFC of a conditionally-unstable (GATE) sounding, run MICROPHYSICS-ONLY
(no surface flux / large-scale forcing / radiation), and check that w GROWS.

If condensation releases latent heat that correctly enters the prognostic theta'
(and hence the density/EOS buoyancy), the bubble accelerates upward into a deep
updraft (O(several) m/s) AND its theta' is MAINTAINED/raised against adiabatic
cooling.  A coupling bug (latent heat lost in the dry-reference + separate
moist-buoyancy split) would leave w ~ 0 and theta' decaying despite condensate
forming.  The existing ``test_plane_nh_rising_thermal`` is a DRY thermal and does
NOT exercise this moist coupling.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as C
from legoesm.atmosphere.dynamics.gcm.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
)
from legoesm.atmosphere.dynamics.crm.sam_case_setup import build_gate_ideal_setup
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.thermodynamics import parcel_profile_and_cape
from legoesm.grids.plane import create_plane_grid
from legoesm.thermo import saturation_mixing_ratio

_GATE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "gSAM", "gsam1.8.7", "gSAM1.8.7", "CASES", "GATE_IDEAL",
)


def _dx_aware_hyperdiff(dx):
    # K = 1e8 * (dx/1000)^4 (the plane CRM 2Δx biharmonic; see dx_aware_hyperdiff)
    return 1.0e8 * (dx / 1000.0) ** 4


def _build_bubble_state_and_model():
    nx = ny = 24
    nlev = 48
    dx = 1000.0
    H = 20000.0
    dt = 2.0
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
        coriolis_mode="none", lat0=8.5,
    )
    setup = build_gate_ideal_setup(
        _GATE_DIR, grid, nlev=nlev, H=H, n_tracers=11, seed_amp=0.0,
    )
    hc = setup.height_coord
    tm = make_flat_plane_terrain_metric(grid, hc)
    st0 = setup.initial_state

    # --- diagnose the LFC of the GATE sounding ---
    thref = np.asarray(hc.theta_ref)
    exn = np.asarray(hc.exner_ref)
    T_env = thref * exn
    p = float(C.p_ref) * exn ** (1.0 / C.kappa)
    ph = np.empty(len(p) + 1)
    ph[1:-1] = 0.5 * (p[:-1] + p[1:])
    ph[0] = p[0] - 0.5 * (p[1] - p[0])
    ph[-1] = p[-1] + 0.5 * (p[-1] - p[-2])
    qv = np.asarray(st0.tracers.data[0, 0, :, 0])
    T_moist, cape = parcel_profile_and_cape(
        jnp.asarray(T_env)[None], jnp.asarray(p)[None],
        jnp.asarray(ph)[None], jnp.asarray(qv)[None],
    )
    assert float(cape[0]) > 200.0, f"GATE sounding not unstable (CAPE={float(cape[0]):.0f})"
    buoy = np.asarray(T_moist[0]) - T_env
    k_lfc = next((k for k in range(nlev - 1, -1, -1) if buoy[k] > 0.05), nlev // 2)

    # --- saturated +3 K warm bubble just ABOVE the LFC (bypass the CIN) ---
    z = np.asarray(hc.z_full)
    k_bub = max(0, k_lfc - 2)
    zc = z[k_bub]
    xc = (nx // 2) * dx
    yc = (ny // 2) * dx
    xx = (jnp.arange(nx) * dx)[None, :]
    yy = (jnp.arange(ny) * dx)[:, None]
    r2 = ((xx - xc) ** 2 + (yy - yc) ** 2) / (3000.0 ** 2)
    zz = (jnp.asarray(z)[None, None, :] - zc) ** 2 / (1500.0 ** 2)
    blob = jnp.exp(-r2[:, :, None] - zz)
    theta_ref3 = jnp.asarray(thref)[None, None, :]
    theta_p = st0.theta_prime.data + 3.0 * blob
    T_bub = (theta_ref3 + theta_p) * jnp.asarray(exn)[None, None, :]
    qsat = saturation_mixing_ratio(T_bub, jnp.asarray(p)[None, None, :])
    qv3 = st0.tracers.data[..., 0]
    qv_new = jnp.where(blob > 0.3, jnp.maximum(qv3, qsat), qv3)
    tracers = st0.tracers.data.at[..., 0].set(qv_new)
    rho0 = jnp.asarray(hc.rho_ref)[None, None, :]
    rho_p = -rho0 * theta_p / theta_ref3
    state = st0._replace(
        theta_prime=st0.theta_prime.replace(data=theta_p),
        rho_prime=st0.rho_prime.replace(data=rho_p),
        tracers=st0.tracers.replace(data=tracers),
    )

    hyp = _dx_aware_hyperdiff(dx)
    cfg = CompressibleEulerConfig(
        hyperdiff_coeff=hyp, hyperdiff_rho_coeff=hyp, hyperdiff_w_coeff=hyp,
        semi_implicit_acoustic=True, substep_horizontal_acoustic=True,
        use_coriolis=False, fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=0.19, smagorinsky_prandtl=1.0,
        smagorinsky_wall_damping=False, sgs_vertical_diffusion=True,
        sponge_width=0.4 * float(hc.H), sponge_w_only=True,
        sponge_profile_shape="sam_rational",
        horizontal_advection_scheme="van_leer",
        vertical_tracer_advection="van_leer",
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    micro_cfg = MicrophysicsConfig(scheme="morrison")
    micro_fn = make_microphysics_physics(micro_cfg, model_type="plane", dt=dt)
    return model, state, micro_fn, dt


@pytest.mark.skipif(
    not os.path.isdir(_GATE_DIR), reason="gSAM GATE_IDEAL case dir not available",
)
def test_moist_bubble_realizes_cape():
    """Saturated warm bubble above the LFC, microphysics-only: must grow a
    deep updraft (CAPE realized) with condensate + latent-heat-maintained
    theta' — the M2005 -> theta' -> buoyancy coupling, end-to-end."""
    model, state, micro_fn, dt = _build_bubble_state_and_model()
    theta0_max = float(jnp.max(state.theta_prime.data))
    max_w = 0.0
    max_cond = 0.0
    for _ in range(50):
        state = model.step(state, dt=dt, physics_fn=micro_fn)
        assert bool(jnp.all(jnp.isfinite(state.w.data))), "non-finite w"
        max_w = max(max_w, float(jnp.max(jnp.abs(state.w.data))))
        cond = float(jnp.max(state.tracers.data[..., 1] + state.tracers.data[..., 3]))
        max_cond = max(max_cond, cond)
    theta_f_max = float(jnp.max(state.theta_prime.data))

    # (1) condensation occurred (the saturated parcel makes cloud as it rises)
    assert max_cond > 5.0e-5, f"no condensate formed (max qc+qi={max_cond:.2e})"
    # (2) the bubble grew a DEEP updraft — CAPE realized (dry thermal ~ 0.02 m/s)
    assert max_w > 1.0, f"bubble did not realize CAPE (max|w|={max_w:.3f} m/s)"
    # (3) THE discriminator: latent heating REPLENISHES the warm core against
    #     ascent + mixing, so theta' is maintained/raised; with the heating lost
    #     (the bug), the core mixes away unreplenished and theta' DECAYS.  Note
    #     (1)+(2) alone do NOT catch the bug -- the q_v virtual buoyancy still
    #     lifts the saturated bubble (w stays O(1) m/s) and condensate still
    #     forms.  Fault-injection (iter-75, zeroing the micro theta' tendency
    #     while keeping condensation): LATENT ON -> final max(theta') 3.00->3.11;
    #     LATENT OFF -> 3.00->2.28.  The 0.9*theta0 = 2.7 threshold sits cleanly
    #     between them, so this assertion catches the "lost latent heat" coupling
    #     bug codex flagged.
    assert theta_f_max >= 0.9 * theta0_max, (
        f"theta' decayed ({theta0_max:.2f} -> {theta_f_max:.2f}) — latent "
        f"heating not feeding the prognostic theta'"
    )

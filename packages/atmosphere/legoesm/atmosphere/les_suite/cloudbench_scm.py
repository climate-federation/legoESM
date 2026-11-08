"""Single-column model driven like a CloudBench LES run (PI 2026-10-31 .. 11-03).

:func:`build_cloudbench_scm` turns a CloudBench artifact (:mod:`.cloudbench`) and the
case's GCM sounding into a column:

* grid: ``nlev`` uniform-σ levels, float64, surface pressure = the run's ``p_sfc``,
  top at the LES domain top (6 km) and level heights from the sounding's own
  pressure-height relation (``ln p`` linear in height);
* start: frame 0 (the sounding as the LES received it) interpolated linearly in
  height; θ = θ_l + L_v q_c/(c_pd Π), q_v = q_t − q_c − q_r;
* forcing (swirl-lm ``gcm_forcing.py``, profiles fixed at the starting heights):
  θ_l/q_t subsidence by the GCM w, the recorded horizontal advection, θ_l/q_t
  nudging to frame 0 at the recorded rate, winds relaxed to frame 0 with the
  recipe's τ_wind, the LES radiative heating (:func:`.bridge.rad_tendency_fn`);
* surface: swirl-lm's MOST (:mod:`~legoesm.atmosphere.physics.turbulence.swirl_most`)
  on the lowest level every stage, from the run's surface θ_li and q_t (θ_li
  converted to our θ_l through the surface temperature it implies), with the LES's
  density-free drag; every closure receives its stress, heat and moisture fluxes;
* microphysics: the artifact's recorded scheme (Kessler, CloudBench warm rain).

KNOWN DIFFERENCES: the surface law is evaluated at our lowest level (~70 m), not
the LES's first node (6.3 m), on the horizontal-mean state; the sounding's height
is taken as physical height (the LES placed each sounding value half a grid
spacing, 6.3 m, above its coordinate). The levels are placed at the sounding's
heights, but the surface law and the closures use the column's own hydrostatic
heights: measured 2026-11-03 on s000_m07, s250_m01, s499_m10, the lowest level sits
0.24-0.58 m below its sounding height (ln z changes by <= 8.2e-3, ~0.1 % of the
surface fluxes) and at most 6.6 m off anywhere at the start.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing, SurfaceFluxRecord
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma,
    compute_rho,
    exner_function,
)
from legoesm.atmosphere.physics.turbulence.swirl_most import (
    SwirlMOSTConfig,
    swirl_most_fluxes,
    validate_swirl_most,
)
from legoesm.grids.vertical import create_sigma_coordinate

from legoesm import constants

from .bridge import LESReferenceArtifact, is_cloudbench, rad_tendency_fn
from .cloudbench import (
    DOMAIN_TOP_M,
    SWIRL_G_M_S2,
    parse_case_name,
    read_sounding,
    surface_theta_l,
)
from .scm_coupling import interp_profile, liquid_water_theta
from .scm_runner import SCMGridSpec, moist_scm_microphysics, scm_physics_config, seed_rain

#: swirl-lm MOST settings of the CloudBench runs (the run config's ``most`` block:
#: beta_m 4.8, beta_h 7.8, alpha 1, gamma_m/gamma_h at the proto default 15), as found
#: in the four stores checked; they are NOT read per artifact, unlike the roughness
#: and gustiness.
_CB_MOST_BETA_M = 4.8
_CB_MOST_BETA_H = 7.8
_CB_MOST_GAMMA = 15.0


def _require_cloudbench(artifact: LESReferenceArtifact) -> dict:
    if not is_cloudbench(artifact):
        raise ValueError(f"{artifact.case_name}: not a CloudBench artifact")
    setup = artifact.les_setup
    recipe = setup.get("forcing_recipe") or {}
    for key, want in (("subsidence", "theta_l and q_t, first-order upwind"),
                      ("nudge_targets", "frame 0 theta_l and q_t"),
                      ("wind_targets", "frame 0 u, v"), ("coriolis", "none")):
        if recipe.get(key) != want:
            raise ValueError(f"{artifact.case_name}: forcing recipe {key}="
                             f"{recipe.get(key)!r}, the builder implements {want!r}")
    if float(artifact.f_c) != 0.0:
        raise ValueError(f"{artifact.case_name}: CloudBench runs have no Coriolis")
    return setup


def surface_record_fn(cfg: SwirlMOSTConfig, theta_l_s: float, q_t_s: float):
    """``SCMForcing.surface_flux_fn``: swirl-lm MOST at the column's lowest full level
    (moist hypsometric height and density, as the turbulence wrapper uses)."""
    validate_swirl_most(cfg)

    def fn(state, sigma_coord, _t):
        temp = jnp.asarray(state.T.data).reshape(-1)
        p_s = jnp.asarray(state.p_s.data).reshape(())
        tr = state.tracers or {}
        q_v = jnp.asarray(tr["q_v"].data).reshape(-1)
        q_c = (jnp.asarray(tr["q_c"].data).reshape(-1) if "q_c" in tr
               else jnp.zeros_like(temp))
        p_full = sigma_coord.pressure_at_full(p_s)
        p_half = sigma_coord.pressure_at_half(p_s)
        z_full, z_half = compute_heights_from_sigma(temp[None], p_half[None], q_v=q_v[None])
        rho = compute_rho(temp[None], p_full[None], q_v=q_v[None])[0]
        exner = exner_function(p_full)
        theta = temp[-1] / exner[-1]
        f = swirl_most_fluxes(
            jnp.asarray(state.u.data).reshape(-1)[-1], jnp.asarray(state.v.data).reshape(-1)[-1],
            theta, liquid_water_theta(theta, q_c[-1], exner[-1]), q_v[-1] + q_c[-1],
            rho[-1], z_full[0, -1] - z_half[0, -1], theta_l_s, q_t_s, cfg)
        return SurfaceFluxRecord(w_theta_l=f.w_theta_l, w_q_t=f.w_q_t, uw=f.uw, vw=f.vw,
                                 rho_s=rho[-1], exner_s=exner_function(p_half[-1]))
    return fn


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cloudbench_grid(sounding: dict, p_s: float, nlev: int) -> tuple[SCMGridSpec, float]:
    """Uniform-σ float64 grid from ``p_s`` to the pressure the sounding has at the LES
    domain top; level heights from the sounding (``ln p`` linear in height).
    Returns ``(grid, sigma_top)``."""
    z_s, ln_p = np.asarray(sounding["z"]), np.log(np.asarray(sounding["p"]))
    if not (np.all(np.diff(z_s) > 0.0) and np.all(np.diff(ln_p) < 0.0)):
        raise ValueError("sounding height must increase and pressure decrease")
    if z_s[0] != 0.0 or z_s[-1] < DOMAIN_TOP_M:
        raise ValueError(f"sounding must span 0..{DOMAIN_TOP_M} m, got {z_s[0]}..{z_s[-1]}")
    p_top = float(np.exp(np.interp(DOMAIN_TOP_M, z_s, ln_p)))
    sigma_top = p_top / p_s
    sc = create_sigma_coordinate(nlev, sigma_top=sigma_top, dtype=jnp.float64)
    p_full = np.asarray(sc.sigma_full) * p_s
    p_half = np.asarray(sc.sigma_half) * p_s
    # np.interp needs increasing abscissae: -ln p increases with height
    z_full = np.interp(-np.log(p_full), -ln_p, z_s)
    return SCMGridSpec(z_scm_m=jnp.asarray(z_full), p_full_pa=jnp.asarray(p_full),
                       p_half_pa=jnp.asarray(p_half), nlev=nlev), sigma_top


def build_cloudbench_scm(artifact: LESReferenceArtifact, turbulence, *, forcing_dir,
                         nlev: int, dt: float):
    """The CloudBench column for ``artifact`` (see module docstring). ``turbulence``
    is an explicit ``TurbulenceConfig``; ``forcing_dir`` holds the cases'
    ``<site>/<month>/<exp>/sounding.csv``, checked against the artifact's hash."""
    setup = _require_cloudbench(artifact)
    par = setup["parameters"]
    site = parse_case_name(artifact.case_name)[0]
    path = Path(forcing_dir) / str(site) / str(int(par["month"])) / par["experiment"] / "sounding.csv"
    if _sha256(path) != setup["sounding_sha256"]:
        raise ValueError(f"{artifact.case_name}: {path} is not the sounding the "
                         "artifact was built from")
    sounding = read_sounding(path)
    p_s = float(par["p_sfc"])
    grid, sigma_top = cloudbench_grid(sounding, p_s, nlev)
    z_les = jnp.asarray(artifact.heights_m)
    z_scm, p_full = grid.z_scm_m, grid.p_full_pa

    def on_scm(prof):
        return interp_profile(jnp.asarray(prof), z_les, z_scm)

    exner = exner_function(p_full)
    theta_l0 = on_scm(jnp.asarray(artifact.theta)[0])
    q_t0 = on_scm(jnp.asarray(artifact.qt)[0])
    q_c0 = jnp.clip(on_scm(jnp.asarray(artifact.qc)[0]), 0.0, None)
    micro = moist_scm_microphysics(artifact)
    q_r0 = seed_rain(artifact, micro, z_les, z_scm)
    q_v0 = q_t0 - q_c0 - (0.0 if q_r0 is None else q_r0)
    temp0 = (theta_l0 + constants.L_v / (constants.c_pd * exner) * q_c0) * exner
    u0 = on_scm(jnp.asarray(artifact.u)[0])
    v0 = on_scm(jnp.asarray(artifact.v)[0])

    def const(prof):
        return lambda _t: prof

    recipe = setup["forcing_recipe"]
    rad = rad_tendency_fn(artifact, dt_s=dt)
    most = SwirlMOSTConfig(
        z0_m=float(setup["z0_m"]), gamma_m=_CB_MOST_GAMMA, gamma_h=_CB_MOST_GAMMA,
        beta_m=_CB_MOST_BETA_M, beta_h=_CB_MOST_BETA_H,
        gustiness_m_s=float(setup["surface_gustiness_m_s"]),
        kappa=constants.kappa_von_karman, g_m_s2=SWIRL_G_M_S2, stability_width_K=0.01,
        stable_continuation_width=1.0e-3, density_free_drag=True)
    theta_l_s = surface_theta_l(float(par["theta_li_sfc"]),
                                           float(par["q_t_sfc"]), p_s)
    forcing = SCMForcing(
        f_c=0.0,
        subsidence_w=const(on_scm(artifact.subsidence_w)),
        subsidence_on="theta_l_qt",
        theta_adv=const(on_scm(artifact.theta_adv)),
        qv_adv=const(on_scm(artifact.qv_adv)),
        nudge_rate=const(on_scm(artifact.nudge_rate)),
        theta_l_target=const(theta_l0),
        qt_target=const(q_t0),
        wind_relax_rate=1.0 / float(recipe["tau_wind_s"]),
        u_target=const(u0),
        v_target=const(v0),
        rad_T_tendency=lambda t: interp_profile(rad(t), z_les, z_scm),
        prescribe="fluxes",
        flux_to_closure=True,
        surface_flux_fn=surface_record_fn(most, theta_l_s, float(par["q_t_sfc"])),
    )
    scm = SingleColumnModel.create(
        physics_config=scm_physics_config(artifact, turbulence),
        nlev=nlev, dt=dt, T_profile=temp0, q_v_profile=q_v0,
        q_c_profile=q_c0, q_r_profile=q_r0, u=u0, v=v0, p_s=p_s,
        sigma_top=sigma_top, forcing=forcing, time_integrator="forward_euler",
        dtype=jnp.float64)
    return scm, grid

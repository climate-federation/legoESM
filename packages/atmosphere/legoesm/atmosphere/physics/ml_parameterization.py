"""Joint runtime helper for ML physics parameterization."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.convection.config import MassFluxConfig
from legoesm.atmosphere.physics.convection.mass_flux import (
    MassFluxClosureDiagnostics,
    mass_flux_convection_from_closure,
)
from legoesm.atmosphere.physics.microphysics.output import MicrophysicsOutput
from legoesm.atmosphere.physics.microphysics.sundqvist import (
    diagnose_sundqvist_process_rates,
)
from legoesm.atmosphere.physics.turbulence.config import LouisConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)
from legoesm.ml.normalization import denormalize, normalize
from legoesm.ml.physics.io import load_physics_checkpoint, load_physics_stats
from legoesm.ml.physics.model import (
    PhysicsParameterizationModel,
    pack_physics_parameterization_features,
    unpack_physics_parameterization_targets,
)


# Machine-checked scheme contract (see tests/test_physics_contracts.py). A learned
# column model predicts closure inputs (Km, Kh, M_eq, microphysics tendencies)
# that are run through the physical closures; the ML predictions carry no hard
# conservation guarantee -> conserves ["none"].
__physics_contract__ = {
    "summary": (
        "Joint ML physics parameterization: a learned column model predicts "
        "eddy diffusivities (Km, Kh), an equilibrium convective mass flux "
        "(M_eq) and optional microphysics tendencies, then runs them through "
        "the mass-flux convection / implicit vertical-diffusion / Sundqvist-"
        "Kessler microphysics closures to produce standard tendencies."
    ),
    "inputs": {
        "T": "K", "u": "m/s", "v": "m/s", "q_v": "kg/kg", "q_c": "kg/kg",
        "q_r": "kg/kg", "p_full": "Pa", "p_half": "Pa", "p_s": "Pa",
        "z_full": "m", "z_half": "m", "T_sfc": "K", "q_sfc": "kg/kg",
        "lat": "rad", "rho": "kg/m^3", "M_c": "kg/m^2/s", "dt": "s",
    },
    "outputs": {
        "Km": "m^2/s", "Kh": "m^2/s", "M_eq": "kg/m^2/s",
        "M_c_new": "kg/m^2/s",
        "dT_dt": "K/s (convective + turbulent temperature tendency)",
        "dq_v_dt": "kg/kg/s", "du_dt": "m/s^2", "dv_dt": "m/s^2",
        "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "Learned mapping: no enforced conservation. Predicted diffusivities "
        "clipped to [0, max_diffusivity], M_eq >= 0; learned warm-rain tracer "
        "tendencies floored so one step cannot drive q_v/q_c/q_r negative; "
        "precipitation >= 0 (leaves the column). Downstream closures apply "
        "their own flux-form updates."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Learned column parameterization run through physical closures "
        "(mass-flux convection + Louis vertical diffusion + Sundqvist/Kessler "
        "microphysics); no single canonical reference -- see module and "
        "closure docstrings"
    ),
    "idealized_test": (
        "tests/unit/test_ml_physics_parameterization.py; zero predicted "
        "diffusivity & mass flux -> zero turbulent/convective tendency; "
        "learned warm-rain tendencies obey one-step non-negativity"
    ),
}


__param_spec__ = {
    "PhysicsParameterizationAssets": {
        "scheme_key": "atm.ml_param.PhysicsParameterizationAssets",
        "excluded": {
        },
        "params": {
            "max_diffusivity": {"units": "m^2/s", "bounds": (165.0, 1500.0), "tunable_tier": 3, "transform": "sigmoid", "category": "diffusivity", "reference": "stability clip on ML-predicted vertical eddy diffusivity", "shape": None},
        },
    },
}


class PhysicsParameterizationAssets(NamedTuple):
    """Loaded joint ML model and normalization stats."""

    model: PhysicsParameterizationModel
    stats_bundle: object
    max_diffusivity: float = 500.0


def _apply_predicted_diffusivity_turbulence(
    *,
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    Km: jax.Array,
    Kh: jax.Array,
    p_full: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    max_diffusivity: float,
    surface_config,
) -> TurbulenceOutput:
    """Convert predicted diffusivities into Louis-style turbulence tendencies."""
    ncol, nlev = T.shape

    if nlev < 2:
        zeros = jnp.zeros_like(T)
        zeros_2d = jnp.zeros((ncol,), dtype=T.dtype)
        return TurbulenceOutput(
            du_dt=zeros,
            dv_dt=zeros,
            dT_dt=zeros,
            dq_v_dt=zeros,
            Km=Km,
            Kh=Kh,
            shflx=zeros_2d,
            lhflx=zeros_2d,
            ustar=zeros_2d,
            h_pbl=zeros_2d,
        )

    Km = jnp.clip(Km, 0.0, max_diffusivity)
    Kh = jnp.clip(Kh, 0.0, max_diffusivity)
    Km_half = 0.5 * (Km[:, :-1] + Km[:, 1:])
    Kh_half = 0.5 * (Kh[:, :-1] + Kh[:, 1:])

    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    dz_half = jnp.clip(dz_half, 1.0, None)
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1],
        v[:, -1],
        T[:, -1],
        q_v[:, -1],
        T_sfc,
        q_sfc,
        rho[:, -1],
        surface_config,
    )

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion(T, Kh_half, rho, dz_layer, dz_half, dt, sflx_T)
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)
    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    return TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km,
        Kh=Kh,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
    )


def load_physics_parameterization_assets(
    *,
    nlev: int,
    hidden_dim: int,
    n_layers: int,
    seed: int,
    checkpoint_path: str,
    stats_path: str,
    microphysics_scheme: str = "none",
) -> PhysicsParameterizationAssets:
    """Load the strict joint ML checkpoint and stats bundle."""
    if not checkpoint_path or not stats_path:
        raise ValueError(
            "physics_parameterization='ml' requires both checkpoint and stats paths",
        )
    model = PhysicsParameterizationModel(
        nlev=nlev,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        microphysics_scheme=microphysics_scheme,
        key=jax.random.PRNGKey(seed),
    )
    model = load_physics_checkpoint(model, checkpoint_path)
    stats_bundle = load_physics_stats(stats_path)
    return PhysicsParameterizationAssets(model=model, stats_bundle=stats_bundle)


def predict_physics_parameterization(
    assets: PhysicsParameterizationAssets,
    *,
    T: jax.Array,
    u: jax.Array,
    v: jax.Array,
    q_v: jax.Array,
    q_c: jax.Array,
    q_r: jax.Array,
    p_full: jax.Array,
    z_full: jax.Array,
    p_s: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    lat: jax.Array,
    cape: jax.Array,
    M_c: jax.Array,
    dt: float,
) -> dict[str, jax.Array]:
    """Predict joint ML outputs for a batch of columns."""
    dt_batch = jnp.full_like(p_s, dt)
    microphysics_scheme = assets.model.microphysics_scheme
    features = jax.vmap(
        lambda T_col, u_col, v_col, q_v_col, q_c_col, q_r_col, p_full_col, z_full_col, p_s_col, T_sfc_col, q_sfc_col, lat_col, cape_col, M_c_col, dt_col: (
            pack_physics_parameterization_features(
                T=T_col,
                u=u_col,
                v=v_col,
                q_v=q_v_col,
                q_c=q_c_col,
                q_r=q_r_col,
                p_full=p_full_col,
                z_full=z_full_col,
                p_s=p_s_col,
                T_sfc=T_sfc_col,
                q_sfc=q_sfc_col,
                lat=lat_col,
                cape=cape_col,
                M_c=M_c_col,
                dt=dt_col,
                microphysics_scheme=microphysics_scheme,
            )
        ),
    )(
        T,
        u,
        v,
        q_v,
        q_c,
        q_r,
        p_full,
        z_full,
        p_s,
        T_sfc,
        q_sfc,
        lat,
        cape,
        M_c,
        dt_batch,
    )
    features_norm = normalize(features, assets.stats_bundle.input_stats)
    targets_norm = jax.vmap(assets.model)(features_norm)
    targets = denormalize(targets_norm, assets.stats_bundle.output_stats)
    unpacked = unpack_physics_parameterization_targets(
        targets,
        T.shape[1],
        microphysics_scheme=microphysics_scheme,
    )
    predicted = {
        "Km": jnp.clip(unpacked["Km"], 0.0, assets.max_diffusivity),
        "Kh": jnp.clip(unpacked["Kh"], 0.0, assets.max_diffusivity),
        "M_eq": jnp.clip(unpacked["M_eq"], 0.0, None),
    }
    if microphysics_scheme == "kessler":
        predicted.update(
            {
                "dq_v_dt_micro": unpacked["dq_v_dt_micro"],
                "dq_c_dt_micro": unpacked["dq_c_dt_micro"],
                "dq_r_dt_micro": unpacked["dq_r_dt_micro"],
                "precip_micro": jnp.clip(unpacked["precip_micro"], 0.0, None),
            },
        )
    elif microphysics_scheme == "sundqvist":
        predicted["rain_survival_fraction"] = jnp.clip(
            unpacked["rain_survival_fraction"],
            0.0,
            1.0,
        )
    return predicted


def _apply_predicted_kessler_microphysics(
    predicted: dict[str, jax.Array],
    dtype,
) -> MicrophysicsOutput:
    """Convert direct ML microphysics outputs into the common backend interface."""
    dq_v_dt = predicted["dq_v_dt_micro"]
    zeros = jnp.zeros_like(dq_v_dt, dtype=dtype)
    return MicrophysicsOutput(
        dT_dt=-constants.L_v * dq_v_dt / constants.c_pd,
        dq_v_dt=dq_v_dt,
        dq_c_dt=predicted["dq_c_dt_micro"],
        dq_r_dt=predicted["dq_r_dt_micro"],
        dq_i_dt=zeros,
        dq_s_dt=zeros,
        dq_g_dt=zeros,
        dN_c_dt=zeros,
        dN_r_dt=zeros,
        dN_i_dt=zeros,
        precipitation=predicted["precip_micro"],
    )


def _limit_predicted_kessler_microphysics_tendencies(
    predicted: dict[str, jax.Array],
    *,
    q_v: jax.Array,
    q_c: jax.Array,
    q_r: jax.Array,
    dt: float,
) -> dict[str, jax.Array]:
    """Enforce one-step non-negativity for the learned warm-rain tracers."""
    dt_safe = jnp.maximum(jnp.asarray(dt, dtype=q_v.dtype), 1.0e-6)
    limited = dict(predicted)
    limited["dq_v_dt_micro"] = jnp.maximum(
        predicted["dq_v_dt_micro"],
        -jnp.maximum(q_v, 0.0) / dt_safe,
    )
    limited["dq_c_dt_micro"] = jnp.maximum(
        predicted["dq_c_dt_micro"],
        -jnp.maximum(q_c, 0.0) / dt_safe,
    )
    limited["dq_r_dt_micro"] = jnp.maximum(
        predicted["dq_r_dt_micro"],
        -jnp.maximum(q_r, 0.0) / dt_safe,
    )
    limited["precip_micro"] = jnp.clip(predicted["precip_micro"], 0.0, None)
    return limited


def apply_predicted_sundqvist_rain_survival_fraction(
    predicted_rain_survival_fraction: jax.Array,
    *,
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config,
) -> MicrophysicsOutput:
    """Rebuild Sundqvist microphysics from a learned column rain-survival fraction."""
    rates = diagnose_sundqvist_process_rates(
        T=T,
        q_v=q_v,
        hydrometeors=hydrometeors,
        p_full=p_full,
        p_half=p_half,
        rho=rho,
        dz=dz,
        dt=dt,
        config=config,
    )
    # Both fluxes share the ``rho * dz`` weight on the level axis; fuse
    # the autoconversion and base-evaporation column reductions.
    _flux_pair = jnp.sum(
        jnp.stack([rates.autoconversion, rates.evaporation], axis=-1)
        * (rho * dz)[..., None],
        axis=-2,
    )
    generated_rain_flux = _flux_pair[..., 0]
    base_evap_flux = _flux_pair[..., 1]
    target_precip = jnp.clip(predicted_rain_survival_fraction, 0.0, 1.0) * generated_rain_flux
    target_evap_flux = jnp.clip(generated_rain_flux - target_precip, 0.0, generated_rain_flux)
    scale = jnp.where(
        base_evap_flux > 1.0e-12,
        target_evap_flux / base_evap_flux,
        1.0,
    )
    evaporation = rates.evaporation * scale[:, None]
    # Diagnostic-rain semantics (must match the non-ML Sundqvist leaf):
    # rain produced by autoconversion is treated as falling instantly;
    # any pre-existing q_r is drained to the surface in one step and
    # added to the precipitation flux.  See ``microphysics/sundqvist.py``
    # for the full rationale.  Without this the ML rebuild would emit
    # ``dq_r_dt = autoconv - evap`` and double-count rain mass — q_r
    # would accumulate while precipitation also reports it leaving.
    dt_safe = jnp.maximum(dt, 1e-10)
    q_r_in = jnp.clip(hydrometeors.q_r, 0.0, None)
    dq_r_dt = -q_r_in / dt_safe
    dp = p_half[:, 1:] - p_half[:, :-1]
    q_r_drain_flux = jnp.sum(q_r_in * dp, axis=1) / (constants.g * dt_safe)
    precipitation = jnp.clip(
        generated_rain_flux - jnp.sum(evaporation * rho * dz, axis=1)
        + q_r_drain_flux,
        0.0,
        None,
    )
    zeros = jnp.zeros_like(T)
    return MicrophysicsOutput(
        dT_dt=constants.L_v * (rates.condensation - evaporation) / constants.c_pd,
        dq_v_dt=-rates.condensation + evaporation,
        dq_c_dt=rates.condensation - rates.autoconversion,
        dq_r_dt=dq_r_dt,
        dq_i_dt=zeros,
        dq_s_dt=zeros,
        dq_g_dt=zeros,
        dN_c_dt=zeros,
        dN_r_dt=zeros,
        dN_i_dt=zeros,
        precipitation=precipitation,
    )


def apply_physics_parameterization(
    assets: PhysicsParameterizationAssets,
    *,
    closure: MassFluxClosureDiagnostics,
    T: jax.Array,
    u: jax.Array,
    v: jax.Array,
    q_v: jax.Array,
    q_c: jax.Array,
    q_r: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    p_s: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    lat: jax.Array,
    rho: jax.Array,
    M_c: jax.Array,
    dt: float,
    mass_flux_config: MassFluxConfig,
    louis_config: LouisConfig,
) -> tuple[object, jax.Array, object, MicrophysicsOutput | None, dict[str, jax.Array]]:
    """Run the joint ML parameterization through the physical closures."""
    predicted = predict_physics_parameterization(
        assets,
        T=T,
        u=u,
        v=v,
        q_v=q_v,
        q_c=q_c,
        q_r=q_r,
        p_full=p_full,
        z_full=z_full,
        p_s=p_s,
        T_sfc=T_sfc,
        q_sfc=q_sfc,
        lat=lat,
        cape=closure.cape,
        M_c=M_c,
        dt=dt,
    )
    if assets.model.microphysics_scheme == "kessler":
        predicted = _limit_predicted_kessler_microphysics_tendencies(
            predicted,
            q_v=q_v,
            q_c=q_c,
            q_r=q_r,
            dt=dt,
        )
    M_c_new = jnp.maximum(
        M_c + dt * (predicted["M_eq"] - M_c) / mass_flux_config.tau_adj,
        0.0,
    )
    conv_out = mass_flux_convection_from_closure(
        T=T,
        q_v=q_v,
        p_full=p_full,
        p_half=p_half,
        closure=closure._replace(M_eq=predicted["M_eq"], M_c_new=M_c_new),
        config=mass_flux_config,
        dt=dt,
    )
    turb_out = _apply_predicted_diffusivity_turbulence(
        u=u,
        v=v,
        T=T,
        q_v=q_v,
        Km=predicted["Km"],
        Kh=predicted["Kh"],
        p_full=p_full,
        z_full=z_full,
        z_half=z_half,
        T_sfc=T_sfc,
        q_sfc=q_sfc,
        rho=rho,
        dt=dt,
        max_diffusivity=assets.max_diffusivity,
        surface_config=louis_config.surface,
    )
    micro_out = None
    if assets.model.microphysics_scheme == "kessler":
        micro_out = _apply_predicted_kessler_microphysics(predicted, T.dtype)
    return conv_out, M_c_new, turb_out, micro_out, predicted

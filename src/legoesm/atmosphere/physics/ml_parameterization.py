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
    features = jax.vmap(pack_physics_parameterization_features)(
        T,
        u,
        v,
        q_v,
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
    unpacked = unpack_physics_parameterization_targets(targets, T.shape[1])
    return {
        "Km": jnp.clip(unpacked["Km"], 0.0, assets.max_diffusivity),
        "Kh": jnp.clip(unpacked["Kh"], 0.0, assets.max_diffusivity),
        "M_eq": jnp.clip(unpacked["M_eq"], 0.0, None),
    }


def apply_physics_parameterization(
    assets: PhysicsParameterizationAssets,
    *,
    closure: MassFluxClosureDiagnostics,
    T: jax.Array,
    u: jax.Array,
    v: jax.Array,
    q_v: jax.Array,
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
) -> tuple[object, jax.Array, object, dict[str, jax.Array]]:
    """Run the joint ML parameterization through the physical closures."""
    predicted = predict_physics_parameterization(
        assets,
        T=T,
        u=u,
        v=v,
        q_v=q_v,
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
    return conv_out, M_c_new, turb_out, predicted

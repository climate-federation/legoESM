"""Shared MPAS state, model, and diagnostic operations for the experiment."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    MPASPrimitiveEquationModel,
)
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.state import HydrostaticState
from legoesm.driver.component_factory import compute_diffusion
from legoesm.driver.config import DycoreConfig
from legoesm.grids.factory import create_grid
from legoesm.grids.vertical import create_sigma_coordinate

from evaluations.headline_diagnostics import geopotential_height_at, interp_to_pressure_level

set_policy(PrecisionPolicy.fp64())


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = (
        "inputs",
        "output",
        "model",
        "background",
        "observation",
        "background_error",
        "optimization",
        "diagnostics",
        "plot",
    )
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"Missing required configuration sections: {missing}")
    return config


def state_from_sample(path: Path, sample_state: str = "f24") -> HydrostaticState:
    if sample_state not in {"f24", "f48"}:
        raise ValueError("sample_state must be 'f24' or 'f48'")
    with np.load(path, allow_pickle=False) as data:
        return HydrostaticState(
            u=Field(
                jnp.asarray(data[f"{sample_state}_u"]),
                name="u",
                dims=("cell", "level"),
                units="m/s",
            ),
            v=Field(
                jnp.asarray(data[f"{sample_state}_v"]),
                name="v",
                dims=("cell", "level"),
                units="m/s",
            ),
            T=Field(
                jnp.asarray(data[f"{sample_state}_T"]),
                name="T",
                dims=("cell", "level"),
                units="K",
            ),
            p_s=Field(
                jnp.asarray(data[f"{sample_state}_p_s"]),
                name="p_s",
                dims=("cell",),
                units="Pa",
            ),
            phis=Field(
                jnp.asarray(data[f"{sample_state}_phis"]),
                name="phis",
                dims=("cell",),
                units="m2/s2",
            ),
            tracers={
                "q_v": Field(
                    jnp.asarray(data[f"{sample_state}_tracer_q_v"]),
                    name="q_v",
                    dims=("cell", "level"),
                    units="kg/kg",
                )
            },
        )


def state_from_analysis(path: Path) -> HydrostaticState:
    with np.load(path, allow_pickle=False) as data:
        return HydrostaticState(
            u=Field(jnp.asarray(data["u"]), name="u", dims=("cell", "level"), units="m/s"),
            v=Field(jnp.asarray(data["v"]), name="v", dims=("cell", "level"), units="m/s"),
            T=Field(jnp.asarray(data["T"]), name="T", dims=("cell", "level"), units="K"),
            p_s=Field(jnp.asarray(data["p_s"]), name="p_s", dims=("cell",), units="Pa"),
            phis=Field(jnp.asarray(data["phis"]), name="phis", dims=("cell",), units="m2/s2"),
            tracers={
                "q_v": Field(
                    jnp.asarray(data["q_v"]),
                    name="q_v",
                    dims=("cell", "level"),
                    units="kg/kg",
                )
            },
        )


def q_specific(state: HydrostaticState) -> jax.Array:
    mixing_ratio = state.tracers["q_v"].data
    return mixing_ratio / (1.0 + jnp.maximum(mixing_ratio, 0.0))


def z500(state: HydrostaticState, sigma) -> jax.Array:
    return geopotential_height_at(
        state.T.data,
        q_specific(state),
        state.p_s.data,
        state.phis.data,
        sigma,
        50_000.0,
    )


def t500(state: HydrostaticState, sigma) -> jax.Array:
    pressure = sigma.pressure_at_full(state.p_s.data)
    return interp_to_pressure_level(state.T.data, pressure, 50_000.0)


def nearest_cell(mesh, latitude_degrees: float, longitude_degrees: float) -> int:
    target_latitude = np.deg2rad(latitude_degrees)
    target_longitude = np.deg2rad(longitude_degrees % 360.0)
    latitude = np.asarray(mesh.latCell)
    longitude = np.asarray(mesh.lonCell)
    cosine = np.sin(latitude) * np.sin(target_latitude) + np.cos(latitude) * np.cos(
        target_latitude
    ) * np.cos(longitude - target_longitude)
    return int(np.argmax(cosine))


def apply_cell_increment(
    state: HydrostaticState,
    increment: HydrostaticState,
) -> HydrostaticState:
    return HydrostaticState(
        u=state.u.replace(data=state.u.data + increment.u.data),
        v=state.v.replace(data=state.v.data + increment.v.data),
        T=state.T.replace(data=state.T.data + increment.T.data),
        p_s=state.p_s.replace(data=state.p_s.data + increment.p_s.data),
        phis=state.phis,
        tracers={
            "q_v": state.tracers["q_v"].replace(
                data=state.tracers["q_v"].data + increment.tracers["q_v"].data
            )
        },
    )


def cell_state_to_edge_state(state: HydrostaticState, mesh) -> HydrostaticState:
    first_cell = mesh.cellsOnEdge[0]
    second_cell = mesh.cellsOnEdge[1]
    eastward = 0.5 * (state.u.data[first_cell] + state.u.data[second_cell])
    northward = 0.5 * (state.v.data[first_cell] + state.v.data[second_cell])
    angle = jnp.asarray(mesh.angleEdge, dtype=eastward.dtype)
    edge_wind = eastward * jnp.cos(angle)[:, None] + northward * jnp.sin(angle)[:, None]
    return HydrostaticState(
        u=Field(
            edge_wind,
            name="u_edge",
            dims=("edge", "level"),
            units="m/s",
            staggering="edge",
        ),
        v=None,
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=state.tracers,
    )


def make_grid_sigma_model(model_config: dict):
    mesh = create_grid(
        "mpas",
        int(model_config["resolution"]),
        lloyd_iterations=int(model_config["lloyd_iterations"]),
    )
    sigma = create_sigma_coordinate(int(model_config["levels"]))
    dycore = DycoreConfig(
        model_type="hydrostatic",
        discretization="mpas",
        dt=float(model_config["time_step_seconds"]),
        hyperdiff_scale=float(model_config["hyperdiffusion_scale"]),
        a_h_scale=float(model_config["horizontal_diffusion_scale"]),
        time_integrator="auto",
        mpas_nu_vert4_T=float(model_config["vertical_temperature_diffusion"]),
    )
    diffusion = compute_diffusion(mesh, dycore)
    settings = MPASPrimitiveEquationConfig(
        nu_del2=diffusion.A_h,
        nu_del4=diffusion.hyperdiff,
        nu_del4_ps=diffusion.hyperdiff,
        K_h=diffusion.K_h_A,
        fix_mass=dycore.fix_mass and dycore.conservation_fixer,
        anchor_mass_to_initial=dycore.fix_mass and dycore.conservation_fixer,
        time_integrator=MPASPrimitiveEquationConfig().time_integrator,
        nu_vert4_T=dycore.mpas_nu_vert4_T,
    )
    return mesh, sigma, MPASPrimitiveEquationModel(mesh, sigma, settings)


def rollout(model, state: HydrostaticState, dt: float, steps: int):
    step = jax.checkpoint(lambda value: model.step(value, dt))

    def body(value, _):
        return step(value), None

    final, _ = jax.lax.scan(body, state, None, length=steps)
    return final


def field_series(model, state, sigma, dt, checkpoints, field):
    output = [field(state, sigma)]
    previous = 0
    for target in checkpoints[1:]:
        state = rollout(model, state, dt, int(target - previous))
        output.append(field(state, sigma))
        previous = int(target)
    return jnp.stack(output)

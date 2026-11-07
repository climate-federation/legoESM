#!/usr/bin/env python
"""Run the MPAS single-point T500 3DVar and strong-constraint 4DVar experiment."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.da.control_vector import build_control_spec, control_to_increment
from legoesm.da.gen_be import GenBETransform, load_gen_be_params
from legoesm.da.minimizer import minimize_lbfgs

from scripts.run.mpas_4dvar_single import common


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/4DVar_single/experiment.yaml"),
    )
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--gen-be", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def minimizer_status_message(result) -> str:
    """Why L-BFGS stopped: converged, line search failed, or gradient tolerance not met."""
    if bool(result.converged):
        return "converged"
    if bool(result.line_search_failed):
        return "line search failed"
    return "gradient tolerance not met"


def main() -> None:
    args = _parse_args()
    config = common.load_config(args.config)
    sample = args.sample or Path(config["inputs"]["nmc_sample"])
    gen_be_path = args.gen_be or Path(config["inputs"]["gen_be_path"])
    output_directory = args.output_dir or Path(config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    model_config = config["model"]
    observation = config["observation"]
    be_config = config["background_error"]
    optimization = config["optimization"]
    dt = float(model_config["time_step_seconds"])
    mesh, sigma, model = common.make_grid_sigma_model(model_config)
    background = common.state_from_sample(
        sample,
        str(config["background"]["sample_state"]),
    )
    params = load_gen_be_params(gen_be_path)
    params = params._replace(
        len_scale=params.len_scale * float(be_config["horizontal_length_scale_multiplier"])
    )
    specification = build_control_spec(
        background,
        grid=mesh,
        fields=("u", "v", "T", "p_s", "tracers"),
    )
    transform = GenBETransform(
        params,
        specification,
        mesh,
        n_diffusion_iter=int(be_config["diffusion_iterations"]),
    )
    standard_deviation_scale = float(be_config["standard_deviation_multiplier"])
    observation_cell = common.nearest_cell(
        mesh,
        float(observation["latitude_degrees_north"]),
        float(observation["longitude_degrees_east"]),
    )
    four_dvar_hours = float(observation["four_dvar_time_hours"])
    four_dvar_steps = int(round(four_dvar_hours * 3600.0 / dt))
    background_edge = common.cell_state_to_edge_state(background, mesh)
    background_final = common.rollout(model, background_edge, dt, four_dvar_steps)
    innovation = float(observation["increment_k"])
    observation_error = jnp.asarray(
        float(observation["error_standard_deviation_k"]),
        dtype=jnp.float64,
    )
    future_observation = common.t500(background_final, sigma)[observation_cell] + innovation

    def increment(control):
        physical_control = standard_deviation_scale * transform.sqrt_multiply(control)
        return control_to_increment(physical_control, specification, background)

    def analysis(control):
        return common.apply_cell_increment(background, increment(control))

    def four_dvar_operator(control):
        state = common.cell_state_to_edge_state(analysis(control), mesh)
        final = common.rollout(model, state, dt, four_dvar_steps)
        return common.t500(final, sigma)[observation_cell]

    zero = jnp.zeros(specification.total_size, dtype=jnp.float64)

    def cost(control):
        residual = four_dvar_operator(control) - future_observation
        return 0.5 * jnp.sum(control * control) + 0.5 * (residual / observation_error) ** 2

    initial_cost = cost(zero)
    result = minimize_lbfgs(
        jax.value_and_grad(cost),
        zero,
        max_iter=int(optimization["maximum_iterations"]),
        gtol=float(optimization["gradient_tolerance"]),
        ftol=float(optimization["function_tolerance"]),
        m=int(optimization["history_size"]),
    )
    jax.block_until_ready(result.x)
    minimizer_status = minimizer_status_message(result)
    if minimizer_status != "converged":
        warnings.warn(
            f"4D-Var minimizer did not converge ({minimizer_status}); the saved "
            "4D-Var analysis is the last accepted iterate and may equal the "
            "background.",
            RuntimeWarning,
            stacklevel=1,
        )
    four_dvar_analysis = analysis(result.x)

    def three_dvar_operator(control):
        return common.t500(analysis(control), sigma)[observation_cell]

    background_observation, three_dvar_gradient = jax.value_and_grad(three_dvar_operator)(zero)
    three_dvar_observation = common.t500(background, sigma)[observation_cell] + innovation
    three_dvar_control = (
        three_dvar_gradient
        * (three_dvar_observation - background_observation)
        / (jnp.sum(three_dvar_gradient**2) + observation_error**2)
    )
    three_dvar_analysis = analysis(three_dvar_control)

    hours = np.asarray(config["diagnostics"]["forecast_hours"], dtype=np.float64)
    checkpoints = tuple(int(round(hour * 3600.0 / dt)) for hour in hours)
    background_z500 = common.field_series(
        model, background_edge, sigma, dt, checkpoints, common.z500
    )
    background_t500 = common.field_series(
        model, background_edge, sigma, dt, checkpoints, common.t500
    )
    three_dvar_edge = common.cell_state_to_edge_state(three_dvar_analysis, mesh)
    three_dvar_t500 = common.field_series(
        model, three_dvar_edge, sigma, dt, checkpoints, common.t500
    )
    four_dvar_edge = common.cell_state_to_edge_state(four_dvar_analysis, mesh)
    four_dvar_t500 = common.field_series(model, four_dvar_edge, sigma, dt, checkpoints, common.t500)

    np.savez_compressed(
        output_directory / "analysis_4dvar_state.npz",
        u=np.asarray(jax.device_get(four_dvar_analysis.u.data)),
        v=np.asarray(jax.device_get(four_dvar_analysis.v.data)),
        T=np.asarray(jax.device_get(four_dvar_analysis.T.data)),
        p_s=np.asarray(jax.device_get(four_dvar_analysis.p_s.data)),
        q_v=np.asarray(jax.device_get(four_dvar_analysis.tracers["q_v"].data)),
        phis=np.asarray(jax.device_get(four_dvar_analysis.phis.data)),
    )
    background_z500_np = np.asarray(jax.device_get(background_z500))
    background_t500_np = np.asarray(jax.device_get(background_t500))
    three_dvar_t500_np = np.asarray(jax.device_get(three_dvar_t500))
    four_dvar_t500_np = np.asarray(jax.device_get(four_dvar_t500))
    np.savez_compressed(
        output_directory / "t500_single_point_timeseries.npz",
        hours=hours,
        sample=str(sample),
        gen_be_path=str(gen_be_path),
        observation_variable=str(observation["variable"]),
        observation_cell=observation_cell,
        observation_latitude_degrees=np.rad2deg(np.asarray(mesh.latCell)[observation_cell]),
        observation_longitude_degrees=np.rad2deg(np.asarray(mesh.lonCell)[observation_cell])
        % 360.0,
        observation_increment_k=innovation,
        observation_error_standard_deviation_k=float(observation_error),
        latitude_cell_degrees=np.rad2deg(np.asarray(mesh.latCell)),
        longitude_cell_degrees=np.rad2deg(np.asarray(mesh.lonCell)) % 360.0,
        z500_background=background_z500_np,
        t500_background=background_t500_np,
        t500_3dvar_increment=three_dvar_t500_np - background_t500_np,
        t500_4dvar_increment=four_dvar_t500_np - background_t500_np,
        four_dvar_initial_cost=np.asarray(jax.device_get(initial_cost)),
        four_dvar_final_cost=np.asarray(jax.device_get(result.fun)),
        four_dvar_iterations=np.asarray(jax.device_get(result.n_iter)),
        four_dvar_converged=np.asarray(jax.device_get(result.converged)),
        four_dvar_minimizer_status=minimizer_status,
    )
    print(f"saved outputs in {output_directory}", flush=True)


if __name__ == "__main__":
    main()

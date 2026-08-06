#!/usr/bin/env python
"""Fit and balance-adjust MPAS GEN_BE parameters from NMC samples."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.da.gen_be import fit_gen_be, save_gen_be_params
from legoesm.grids.factory import create_grid
from legoesm.grids.vertical import create_sigma_coordinate

REQUIRED_KEYS = (
    "error_u",
    "error_v",
    "error_T",
    "error_p_s",
    "error_phis",
    "error_tracer_q_v",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/4DVar_single/nmc.yaml"),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for section in ("model", "gen_be"):
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    return config


def _state_from_sample(path: Path) -> HydrostaticState:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")
        arrays = {key: np.asarray(data[key], dtype=np.float64) for key in REQUIRED_KEYS}
    for name, array in arrays.items():
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"{path} contains invalid values in {name}")
    return HydrostaticState(
        u=Field(arrays["error_u"], name="u", dims=("cell", "level"), units="m/s"),
        v=Field(arrays["error_v"], name="v", dims=("cell", "level"), units="m/s"),
        T=Field(arrays["error_T"], name="T", dims=("cell", "level"), units="K"),
        p_s=Field(arrays["error_p_s"], name="p_s", dims=("cell",), units="Pa"),
        phis=Field(
            arrays["error_phis"],
            name="phis",
            dims=("cell",),
            units="m2/s2",
        ),
        tracers={
            "q_v": Field(
                arrays["error_tracer_q_v"],
                name="q_v",
                dims=("cell", "level"),
                units="kg/kg",
            )
        },
    )


def _horizontal_normalization(
    mesh,
    length_scale: jax.Array,
    diffusion_iterations: int,
    probes: int,
    seed: int,
) -> np.ndarray:
    """Estimate the MPAS diffusion square-root diagonal normalization."""
    columns = int(mesh.grid_n_columns)
    neighbors = jnp.clip(jnp.asarray(mesh.cellsOnCell), 0, columns - 1)
    edges = jnp.clip(jnp.asarray(mesh.edgesOnCell), 0, int(mesh.nEdges) - 1)
    edge_count = jnp.asarray(mesh.nEdgesOnCell)
    mask = jnp.arange(neighbors.shape[0])[:, None] < edge_count[None, :]
    area = jnp.asarray(mesh.areaCell, dtype=jnp.float64)
    dc_edge = jnp.asarray(mesh.dcEdge, dtype=jnp.float64)
    dv_edge = jnp.asarray(mesh.dvEdge, dtype=jnp.float64)
    weights = jnp.where(
        mask,
        dv_edge[edges] / jnp.maximum(dc_edge[edges], 1.0) / jnp.maximum(area[None, :], 1.0),
        0.0,
    )
    max_diagonal = jnp.max(jnp.sum(weights, axis=0))
    step_m2 = jnp.clip(
        length_scale**2 / (2.0 * diffusion_iterations),
        0.0,
        0.49 / jnp.maximum(max_diagonal, 1.0e-30),
    )

    @jax.jit
    def smooth(values):
        def step(current, _):
            neighbor_values = current[neighbors, :]
            laplacian = jnp.sum(
                weights[..., None] * (neighbor_values - current[None, :, :]),
                axis=0,
            )
            return current + step_m2[None, :] * laplacian, None

        return jax.lax.scan(step, values, None, length=diffusion_iterations)[0]

    variance = jnp.zeros_like(length_scale, dtype=jnp.float64)
    for probe in range(probes):
        white = jax.random.rademacher(
            jax.random.PRNGKey(seed + probe),
            (columns, int(length_scale.shape[0])),
            dtype=jnp.float64,
        )
        smoothed = smooth(white)
        variance += jnp.mean(smoothed * smoothed, axis=0)
    normalization = 1.0 / jnp.sqrt(jnp.maximum(variance / probes, 1.0e-30))
    return np.asarray(jax.device_get(normalization), dtype=np.float64)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    model_config = config["model"]
    be_config = config["gen_be"]
    paths = [Path(name) for name in sorted(glob.glob(be_config["samples_glob"]))]
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if len(paths) < 2:
        raise ValueError(f"At least two NMC samples are required; found {len(paths)}")
    errors = [_state_from_sample(path) for path in paths]
    mesh = create_grid(
        "mpas",
        int(model_config["resolution"]),
        lloyd_iterations=int(model_config["lloyd_iterations"]),
    )
    sigma = create_sigma_coordinate(int(model_config["levels"]))
    if int(mesh.grid_n_columns) != int(errors[0].p_s.data.shape[0]):
        raise ValueError("Configured MPAS grid does not match the NMC sample grid")
    base = fit_gen_be(
        errors,
        mesh,
        sigma_coord=sigma,
        vert_corr_length=float(be_config["vertical_correlation_length"]),
        default_len_scale_km=float(be_config["default_horizontal_length_scale_km"]),
        balance_n_modes=int(be_config["balance_predictor_modes"]),
        balance_lat_center_deg=float(be_config["balance_latitude_center_degrees"]),
        balance_lat_half_width_deg=float(be_config["balance_latitude_half_width_degrees"]),
    )
    levels = int(base.n_levels)
    standard_deviation_scale = float(be_config["nmc_standard_deviation_scale"])
    final_standard_deviation_scale = float(be_config["final_standard_deviation_scale"])
    length_scale = np.asarray(base.len_scale, dtype=np.float64).copy()
    length_scale[1 : 1 + 2 * levels] *= float(be_config["wind_length_scale"])
    length_scale *= float(be_config["final_horizontal_length_scale_multiplier"])
    eigenvalue_scale = (standard_deviation_scale * final_standard_deviation_scale) ** 2
    pressure_scale = standard_deviation_scale * final_standard_deviation_scale
    horizontal_normalization = _horizontal_normalization(
        mesh,
        jnp.asarray(length_scale),
        int(be_config["diffusion_iterations"]),
        int(be_config["normalization_probes"]),
        int(be_config["normalization_seed"]),
    )
    adjusted = base._replace(
        vert_eig_val=jnp.asarray(base.vert_eig_val) * eigenvalue_scale,
        std_ps=jnp.asarray(base.std_ps) * pressure_scale,
        len_scale=jnp.asarray(length_scale),
        horiz_norm=jnp.asarray(horizontal_normalization),
    )
    audit = {
        "balance_predictor_modes": int(be_config["balance_predictor_modes"]),
        "balance_latitude_center_degrees": float(be_config["balance_latitude_center_degrees"]),
        "balance_latitude_half_width_degrees": float(
            be_config["balance_latitude_half_width_degrees"]
        ),
        "nmc_standard_deviation_scale": standard_deviation_scale,
        "wind_length_scale": float(be_config["wind_length_scale"]),
        "final_horizontal_length_scale_multiplier": float(
            be_config["final_horizontal_length_scale_multiplier"]
        ),
        "final_standard_deviation_scale": final_standard_deviation_scale,
        "diffusion_iterations": int(be_config["diffusion_iterations"]),
        "normalization_probes": int(be_config["normalization_probes"]),
        "horizontal_normalization_min": float(horizontal_normalization.min()),
        "horizontal_normalization_mean": float(horizontal_normalization.mean()),
        "horizontal_normalization_max": float(horizontal_normalization.max()),
        "wind_transform": base.wind_transform,
    }
    output_path = Path(be_config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_gen_be_params(adjusted, output_path)
    manifest = {
        "sample_count": len(paths),
        "samples": [str(path) for path in paths],
        "resolution": int(model_config["resolution"]),
        "levels": int(model_config["levels"]),
        "vertical_correlation_length": float(be_config["vertical_correlation_length"]),
        "default_horizontal_length_scale_km": float(
            be_config["default_horizontal_length_scale_km"]
        ),
        **audit,
    }
    manifest_path = output_path.parent / "mpas_gen_be_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {output_path}.npz and {output_path}.json", flush=True)
    print(f"saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

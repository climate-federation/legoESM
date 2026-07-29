#!/usr/bin/env python
"""Fit and balance-adjust MPAS GEN_BE parameters from NMC samples."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import yaml
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.da.gen_be import GenBEParams, fit_gen_be, save_gen_be_params
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


def _mode_to_physical(eigenvectors: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    return eigenvectors.T * np.sqrt(np.maximum(eigenvalues, 1.0e-30))[:, None]


def _physical_to_modes(eigenvectors: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    return eigenvectors / np.sqrt(np.maximum(eigenvalues, 1.0e-30))[None, :]


def _adjust_balance(
    base: GenBEParams,
    temperature_scale: float,
    pressure_scale: float,
    minimum_eigenvalue_fraction: float,
) -> tuple[GenBEParams, dict]:
    levels = int(base.n_levels)
    eigenvectors = np.asarray(base.vert_eig_vec, dtype=np.float64).copy()
    eigenvalues = np.asarray(base.vert_eig_val, dtype=np.float64).copy()
    regression = np.asarray(base.reg_coeff, dtype=np.float64).copy()
    length_scale = np.asarray(base.len_scale, dtype=np.float64).copy()
    pressure_std = float(np.asarray(base.std_ps))
    pressure_rows = slice(0, 1)
    temperature_rows = slice(1 + 2 * levels, 1 + 3 * levels)
    temperature_regression = regression[temperature_rows].copy()
    pressure_regression = regression[pressure_rows].copy()

    old_transform = _mode_to_physical(eigenvectors[2], eigenvalues[2])
    old_covariance = old_transform.T @ old_transform
    balanced_covariance = (
        old_transform.T @ (temperature_regression @ temperature_regression.T) @ old_transform
    )
    factor = temperature_scale * (2.0 - temperature_scale)
    residual = old_covariance - factor * balanced_covariance
    residual = 0.5 * (residual + residual.T)
    residual_values, residual_vectors = np.linalg.eigh(residual)
    floor = max(
        minimum_eigenvalue_fraction * float(np.max(np.diag(old_covariance))),
        1.0e-12,
    )
    clipped = np.maximum(residual_values, floor)
    order = np.argsort(clipped)[::-1]
    new_values = clipped[order]
    new_vectors = residual_vectors[:, order]
    inverse_transform = _physical_to_modes(new_vectors, new_values)
    balanced_physical = (temperature_scale * temperature_regression.T) @ old_transform
    regression[temperature_rows] = (balanced_physical @ inverse_transform).T
    eigenvectors[2] = new_vectors
    eigenvalues[2] = new_values

    pressure_balanced_fraction = float(np.sum(pressure_regression**2))
    pressure_residual_factor = max(
        1.0 - pressure_scale * (2.0 - pressure_scale) * pressure_balanced_fraction,
        minimum_eigenvalue_fraction,
    )
    new_pressure_std = pressure_std * np.sqrt(pressure_residual_factor)
    regression[pressure_rows] = pressure_regression * (
        pressure_scale * pressure_std / max(new_pressure_std, 1.0e-30)
    )
    adjusted = GenBEParams(
        vert_eig_vec=eigenvectors,
        vert_eig_val=eigenvalues,
        std_ps=np.asarray(new_pressure_std),
        reg_coeff=regression,
        len_scale=length_scale,
        tracer_names=tuple(base.tracer_names),
        n_levels=levels,
        wind_transform=base.wind_transform,
    )
    audit = {
        "temperature_balance_scale": temperature_scale,
        "surface_pressure_balance_scale": pressure_scale,
        "minimum_eigenvalue_fraction": minimum_eigenvalue_fraction,
        "surface_pressure_std_before_pa": pressure_std,
        "surface_pressure_std_after_pa": float(new_pressure_std),
        "temperature_eigenvalues_clipped": int(np.sum(residual_values < floor)),
        "wind_transform": base.wind_transform,
    }
    return adjusted, audit


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
    )
    adjusted, audit = _adjust_balance(
        base,
        float(be_config["temperature_balance_scale"]),
        float(be_config["surface_pressure_balance_scale"]),
        float(be_config["minimum_eigenvalue_fraction"]),
    )
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

"""Plotting helpers for the joint ML physics workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import griddata


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _get_cartopy():
    import cartopy.crs as ccrs

    return ccrs


def plot_training_history(
    train_loss_history: Iterable[float],
    val_loss_history: Iterable[float],
    output_path: str | Path,
) -> None:
    """Save a simple training/validation loss curve."""
    plt = _get_pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_losses = np.asarray(tuple(train_loss_history), dtype=float)
    val_losses = np.asarray(tuple(val_loss_history), dtype=float)
    epochs = np.arange(1, train_losses.size + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(epochs, train_losses, label="train", lw=2.0)
    ax.plot(epochs, val_losses, label="validation", lw=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.set_title("ML Physics Parameterization Training History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_profile_comparison(
    p_full: np.ndarray,
    teacher: dict[str, np.ndarray],
    student: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    sample_index: int = 0,
    sample_day: float | None = None,
) -> None:
    """Save teacher-vs-student diffusivity profiles for one sampled column."""
    plt = _get_pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pressure_hpa = np.asarray(p_full[sample_index], dtype=float) / 100.0
    order = np.argsort(pressure_hpa)
    p_sorted = pressure_hpa[order]

    teacher_Km = np.asarray(teacher["Km"][sample_index], dtype=float)[order]
    student_Km = np.asarray(student["Km"][sample_index], dtype=float)[order]
    teacher_Kh = np.asarray(teacher["Kh"][sample_index], dtype=float)[order]
    student_Kh = np.asarray(student["Kh"][sample_index], dtype=float)[order]
    teacher_M_eq = float(np.asarray(teacher["M_eq"][sample_index], dtype=float))
    student_M_eq = float(np.asarray(student["M_eq"][sample_index], dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), constrained_layout=True)
    axes[0].plot(teacher_Km, p_sorted, lw=2.2, label="Teacher")
    axes[0].plot(student_Km, p_sorted, lw=2.0, ls="--", label="ML")
    axes[0].set_xlabel("$K_m$ [m$^2$ s$^{-1}$]")
    axes[0].set_ylabel("Pressure [hPa]")
    axes[0].set_title("Momentum Diffusivity")
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()

    axes[1].plot(teacher_Kh, p_sorted, lw=2.2, label="Teacher")
    axes[1].plot(student_Kh, p_sorted, lw=2.0, ls="--", label="ML")
    axes[1].set_xlabel("$K_h$ [m$^2$ s$^{-1}$]")
    axes[1].set_ylabel("Pressure [hPa]")
    axes[1].set_title("Heat Diffusivity")
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()

    title_parts = [f"Sample {sample_index} joint-physics profile comparison"]
    if sample_day is not None:
        title_parts.append(f"day={sample_day:.1f}")
    title_parts.append(f"$M_{{eq}}$ teacher={teacher_M_eq:.3e}")
    title_parts.append(f"$M_{{eq}}$ ML={student_M_eq:.3e}")
    fig.suptitle(" | ".join(title_parts), fontsize=11)
    axes[0].legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profile_rmse(
    p_full: np.ndarray,
    teacher: dict[str, np.ndarray],
    student: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Save level-wise RMSE curves for ``Km`` and ``Kh``."""
    plt = _get_pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    p_mean_hpa = np.asarray(np.mean(p_full, axis=0), dtype=float) / 100.0
    order = np.argsort(p_mean_hpa)
    p_sorted = p_mean_hpa[order]

    rmse_Km = np.sqrt(
        np.mean((np.asarray(student["Km"]) - np.asarray(teacher["Km"])) ** 2, axis=0)
    )
    rmse_Kh = np.sqrt(
        np.mean((np.asarray(student["Kh"]) - np.asarray(teacher["Kh"])) ** 2, axis=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), constrained_layout=True)
    axes[0].plot(rmse_Km[order], p_sorted, lw=2.0)
    axes[0].set_xlabel("RMSE [$m^2$ s$^{-1}$]")
    axes[0].set_ylabel("Pressure [hPa]")
    axes[0].set_title("$K_m$ Profile RMSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()

    axes[1].plot(rmse_Kh[order], p_sorted, lw=2.0)
    axes[1].set_xlabel("RMSE [$m^2$ s$^{-1}$]")
    axes[1].set_ylabel("Pressure [hPa]")
    axes[1].set_title("$K_h$ Profile RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()

    fig.suptitle("Joint ML Physics Profile-Wise RMSE", fontsize=12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_timeseries(
    baseline: dict[str, np.ndarray],
    ml: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    baseline_label: str = "Default",
    ml_label: str = "Full ML",
    comparison_title: str = "Default vs Full ML",
) -> None:
    """Save baseline-vs-ML scalar AMIP timeseries."""
    plt = _get_pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("T_atm", "T_atm"),
        ("T_low", "T_low"),
        ("CWV", "CWV"),
        ("precip", "Precip"),
        ("max_wind", "Wind"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8.2, 11.0), constrained_layout=True)
    days = np.asarray(baseline["days"], dtype=float)
    for ax, (key, label) in zip(np.atleast_1d(axes).ravel(), metrics, strict=True):
        ax.plot(days, np.asarray(baseline[key], dtype=float), lw=2.0, label=baseline_label)
        ax.plot(days, np.asarray(ml[key], dtype=float), lw=2.0, ls="--", label=ml_label)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Day")
    fig.suptitle(comparison_title, fontsize=12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_profiles(
    baseline: dict[str, np.ndarray],
    ml: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    baseline_label: str = "Default",
    ml_label: str = "Full ML",
    comparison_title: str = "Final Profiles",
) -> None:
    """Save final vertical profile comparisons from paired AMIP rollouts."""
    plt = _get_pyplot()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sigma = np.asarray(baseline["sigma"], dtype=float)
    baseline_T_final = np.asarray(baseline["profiles_T"][-1], dtype=float)
    ml_T_final = np.asarray(ml["profiles_T"][-1], dtype=float)
    baseline_q_final = np.asarray(baseline["profiles_qv"][-1], dtype=float)
    ml_q_final = np.asarray(ml["profiles_qv"][-1], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.5), constrained_layout=True)
    axes[0].plot(baseline_T_final, sigma, lw=2.0, label=baseline_label)
    axes[0].plot(ml_T_final, sigma, lw=2.0, ls="--", label=ml_label)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("T")
    axes[0].set_ylabel("sigma")
    axes[0].set_title("Temperature")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(baseline_q_final, sigma, lw=2.0, label=baseline_label)
    axes[1].plot(ml_q_final, sigma, lw=2.0, ls="--", label=ml_label)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("q_v")
    axes[1].set_ylabel("sigma")
    axes[1].set_title("Humidity")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()

    fig.suptitle(comparison_title, fontsize=12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _interpolate_to_latlon(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    data_cs: np.ndarray,
    *,
    nlon: int = 361,
    nlat: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate cubed-sphere cell-center data to a regular lat-lon grid."""
    lon_flat = np.asarray(lon_deg, dtype=float).ravel()
    lat_flat = np.asarray(lat_deg, dtype=float).ravel()
    data_flat = np.asarray(data_cs, dtype=float).ravel()

    points = np.column_stack([
        np.concatenate([lon_flat - 360.0, lon_flat, lon_flat + 360.0]),
        np.concatenate([lat_flat, lat_flat, lat_flat]),
    ])
    values = np.concatenate([data_flat, data_flat, data_flat])

    lon_grid = np.linspace(-180.0, 180.0, nlon)
    lat_grid = np.linspace(-90.0, 90.0, nlat)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    field = griddata(points, values, (lon_mesh, lat_mesh), method="linear")
    if np.isnan(field).any():
        nearest = griddata(points, values, (lon_mesh, lat_mesh), method="nearest")
        field = np.where(np.isnan(field), nearest, field)
    return lon_grid, lat_grid, field


def plot_rollout_map_fields(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    baseline_fields: dict[str, np.ndarray],
    ml_fields: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    day_label: str,
    baseline_label: str = "Default",
    ml_label: str = "Full ML",
    comparison_title: str | None = None,
) -> None:
    """Save filled maps for baseline and ML final snapshots."""
    plt = _get_pyplot()
    ccrs = _get_cartopy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_specs = [
        ("T_low", "T_low", "coolwarm"),
        ("q_v_low", "q_v_low", "viridis"),
        ("precip", "Precip", "Blues"),
        ("wind", "Wind", "magma"),
    ]
    fig, axes = plt.subplots(
        2,
        len(field_specs),
        figsize=(16.0, 7.0),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for col, (name, label, cmap) in enumerate(field_specs):
        baseline = np.asarray(baseline_fields[name], dtype=float)
        ml = np.asarray(ml_fields[name], dtype=float)
        vmin = float(min(np.nanmin(baseline), np.nanmin(ml)))
        vmax = float(max(np.nanmax(baseline), np.nanmax(ml)))
        lon_grid, lat_grid, baseline_ll = _interpolate_to_latlon(lon_deg, lat_deg, baseline)
        _, _, ml_ll = _interpolate_to_latlon(lon_deg, lat_deg, ml)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        for row, (title, field_ll) in enumerate((
            (baseline_label, baseline_ll),
            (ml_label, ml_ll),
        )):
            ax = axes[row, col]
            im = ax.pcolormesh(
                lon_mesh,
                lat_mesh,
                field_ll,
                shading="auto",
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.coastlines(linewidth=0.5, color="gray")
            ax.set_global()
            ax.set_title(name)
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    title,
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=11,
                )
            cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
            cbar.set_label(label)

    if comparison_title is None:
        comparison_title = f"Maps, Day {day_label}"
    fig.suptitle(comparison_title, fontsize=13)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_difference_maps(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    baseline_fields: dict[str, np.ndarray],
    ml_fields: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    day_label: str,
    baseline_label: str = "Default",
    ml_label: str = "Full ML",
) -> None:
    """Save ML-minus-baseline difference maps."""
    plt = _get_pyplot()
    ccrs = _get_cartopy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_specs = [
        ("T_low", "dT_low"),
        ("q_v_low", "dq_v_low"),
        ("precip", "dPrecip"),
        ("wind", "dWind"),
    ]
    fig, axes = plt.subplots(
        1,
        len(field_specs),
        figsize=(16.0, 4.2),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for ax, (name, label) in zip(np.atleast_1d(axes).ravel(), field_specs, strict=True):
        delta = np.asarray(ml_fields[name], dtype=float) - np.asarray(baseline_fields[name], dtype=float)
        lon_grid, lat_grid, delta_ll = _interpolate_to_latlon(lon_deg, lat_deg, delta)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        vmax = float(np.nanmax(np.abs(delta_ll)))
        im = ax.pcolormesh(
            lon_mesh,
            lat_mesh,
            delta_ll,
            shading="auto",
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.coastlines(linewidth=0.5, color="gray")
        ax.set_global()
        ax.set_title(name)
        cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
        cbar.set_label(label)

    fig.suptitle(
        f"Difference Maps, Day {day_label}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

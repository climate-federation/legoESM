"""Plotting utilities for WeatherBench2 evaluation results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp


def plot_scorecard(
    results: dict,
    variables: list[str],
    levels: list[int],
    lead_times: list[int],
    metric: str = "rmse",
    save_path: str | None = None,
):
    """Plot a heatmap scorecard of metrics by variable/level and lead time.

    Parameters
    ----------
    results : dict
        Scorecard dict from ``compute_scorecard``.
    variables : list of str
        Variables to include.
    levels : list of int
        Pressure levels.
    lead_times : list of int
        Lead times in hours.
    metric : str
        Metric to plot ("rmse", "acc", or "bias").
    save_path : str, optional
        Path to save the figure. If None, displays interactively.
    """
    import matplotlib.pyplot as plt

    # Build matrix: rows = (variable, level), cols = lead_time
    row_labels = []
    data = []
    for var in variables:
        for lev in levels:
            row_labels.append(f"{var} {lev}")
            row = []
            for lt in lead_times:
                entry = results.get(var, {}).get(lev, {}).get(lt, {})
                row.append(entry.get(metric, np.nan))
            data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(1, 1, figsize=(len(lead_times) * 1.5 + 2, len(row_labels) * 0.5 + 2))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r" if metric == "rmse" else "RdYlGn")
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" for lt in lead_times])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Lead time")
    ax.set_title(f"{metric.upper()} Scorecard")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_rmse_vs_leadtime(
    results: dict,
    variable: str,
    level: int,
    baseline_results: dict | None = None,
    save_path: str | None = None,
):
    """Plot RMSE vs lead time for a single variable/level.

    Parameters
    ----------
    results : dict
        Scorecard dict.
    variable : str
        Variable name.
    level : int
        Pressure level.
    baseline_results : dict, optional
        Baseline scorecard for comparison.
    save_path : str, optional
        Path to save figure.
    """
    import matplotlib.pyplot as plt

    var_data = results.get(variable, {}).get(level, {})
    lead_times = sorted(var_data.keys())
    rmse_values = [var_data[lt].get("rmse", np.nan) for lt in lead_times]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(lead_times, rmse_values, "o-", label="Model", linewidth=2)

    if baseline_results:
        base_data = baseline_results.get(variable, {}).get(level, {})
        base_rmse = [base_data.get(lt, {}).get("rmse", np.nan) for lt in lead_times]
        ax.plot(lead_times, base_rmse, "s--", label="Baseline", linewidth=2)

    ax.set_xlabel("Lead time [hours]")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE — {variable} at {level} hPa")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_acc_vs_leadtime(
    results: dict,
    variable: str,
    level: int,
    save_path: str | None = None,
):
    """Plot ACC vs lead time for a single variable/level.

    Parameters
    ----------
    results : dict
        Scorecard dict.
    variable : str
        Variable name.
    level : int
        Pressure level.
    save_path : str, optional
        Path to save figure.
    """
    import matplotlib.pyplot as plt

    var_data = results.get(variable, {}).get(level, {})
    lead_times = sorted(var_data.keys())
    acc_values = [var_data[lt].get("acc", np.nan) for lt in lead_times]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(lead_times, acc_values, "o-", linewidth=2)
    ax.axhline(y=0.6, color="r", linestyle="--", alpha=0.5, label="Useful skill (0.6)")
    ax.set_xlabel("Lead time [hours]")
    ax.set_ylabel("ACC")
    ax.set_title(f"ACC — {variable} at {level} hPa")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_global_map(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    variable: str,
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    save_path: str | None = None,
):
    """Side-by-side Mollweide projection of prediction and target.

    Parameters
    ----------
    pred : array, shape (n_lat, n_lon)
        Predicted field.
    target : array, shape (n_lat, n_lon)
        Target field.
    variable : str
        Variable name for title.
    lat : array, shape (n_lat,)
        Latitudes in radians.
    lon : array, shape (n_lon,)
        Longitudes in radians.
    save_path : str, optional
        Path to save figure.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    pred_np = np.array(pred)
    target_np = np.array(target)
    lat_deg = np.degrees(np.array(lat))
    lon_deg = np.degrees(np.array(lon))

    vmin = min(pred_np.min(), target_np.min())
    vmax = max(pred_np.max(), target_np.max())

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 5),
        subplot_kw={"projection": ccrs.Mollweide()},
    )

    for ax, data, title in [
        (ax1, pred_np, f"Prediction: {variable}"),
        (ax2, target_np, f"Target: {variable}"),
        (ax3, pred_np - target_np, f"Error: {variable}"),
    ]:
        if title.startswith("Error"):
            err_max = max(abs((pred_np - target_np).min()), abs((pred_np - target_np).max()))
            im = ax.pcolormesh(
                lon_deg, lat_deg, data,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r", vmin=-err_max, vmax=err_max,
            )
        else:
            im = ax.pcolormesh(
                lon_deg, lat_deg, data,
                transform=ccrs.PlateCarree(),
                cmap="viridis", vmin=vmin, vmax=vmax,
            )
        ax.set_title(title)
        ax.coastlines()
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

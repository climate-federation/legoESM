"""Run the shallow water model for 100 days — Williamson Test Case 5.

Williamson Test 5: Zonal flow over an isolated mountain at C48 resolution.
This generates Rossby wave trains that are a classic benchmark for
dynamical core validation.

Produces:
  - Height field snapshots at days 0, 15, 50, 100
  - Wind speed maps
  - Conservation diagnostics time series
  - Vorticity field
  - Error norms vs. day 0
"""

import time
import os

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water import (
    ShallowWaterModel, ShallowWaterConfig,
)
from tests.test_cases.williamson import (
    williamson_test5, compute_error_norms,
)
from legoesm.core.conservation import compute_conservation_diagnostics
from legoesm.core.field import Field
from legoesm.core.operators import curl_z

# =====================================================================
# Configuration
# =====================================================================
RESOLUTION = 48         # C48 ~ 200 km
DT = 600.0              # 10 min timestep [s]
DURATION_DAYS = 100
SAVE_DIAG_EVERY = 144   # Save conservation diagnostics every N steps (~1 day)
SNAPSHOT_DAYS = [0, 15, 50, 100]  # Days at which to save full state

OUTPUT_DIR = "results/100day_williamson5"

# =====================================================================
# Setup
# =====================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("legoESM — 100-Day Williamson Test Case 5")
print("=" * 70)

print(f"\nCreating C{RESOLUTION} cubed-sphere grid...")
t0 = time.time()
grid = create_cubed_sphere(RESOLUTION)
print(f"  Grid created in {time.time()-t0:.1f}s")
print(f"  Resolution: ~{grid.resolution_km:.0f} km")
print(f"  Total cells: {grid.n_cells:,}")

# Hyperdiffusion coefficient: scale with grid spacing
mean_dx = float(jnp.mean(grid.dx))
hyperdiff_coeff = 1e-4 * mean_dx**4 / DT
print(f"  Mean dx: {mean_dx/1000:.0f} km")
print(f"  Hyperdiffusion coeff: {hyperdiff_coeff:.2e}")

config = ShallowWaterConfig(
    hyperdiff_coeff=hyperdiff_coeff,
    use_conservation_fixer=True,
    fix_mass=True,
    fix_energy=True,
    use_upwind_advection=True,
)
model = ShallowWaterModel(grid, config)

print("\nInitializing Williamson Test Case 5 (mountain)...")
state = williamson_test5(grid)
state_init = state  # Keep for error computation

print(f"  Height range: [{float(jnp.min(state.h.data)):.0f}, "
      f"{float(jnp.max(state.h.data)):.0f}] m")
print(f"  Wind speed max: {float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2))):.1f} m/s")
print(f"  Mountain peak: {float(jnp.max(state.h_s.data)):.0f} m")

# =====================================================================
# Integration
# =====================================================================
n_steps = int(DURATION_DAYS * 86400 / DT)
snapshot_steps = [int(d * 86400 / DT) for d in SNAPSHOT_DAYS]

print(f"\nIntegrating for {DURATION_DAYS} days ({n_steps:,} steps, dt={DT:.0f}s)...")

# Storage
diagnostics = []
snapshots = {}
error_norms_history = []

# Save initial state
diag = compute_conservation_diagnostics(state, grid)
diagnostics.append(diag)
snapshots[0] = state

# Warm up JIT
print("  Warming up JIT...", end=" ", flush=True)
t0 = time.time()
_ = model.step(state, DT)
jax.block_until_ready(_)
print(f"done ({time.time()-t0:.1f}s)")

# Main loop
print(f"\n{'Step':>8s}  {'Day':>6s}  {'h_min':>8s}  {'h_max':>8s}  {'|u|_max':>8s}  "
      f"{'mass_Δ':>10s}  {'energy_Δ':>10s}  {'wall_s':>7s}")
print("-" * 85)

t_start = time.time()
t_lap = t_start

for step in range(1, n_steps + 1):
    state = model.step(state, DT)

    # Save diagnostics
    if step % SAVE_DIAG_EVERY == 0 or step in snapshot_steps:
        jax.block_until_ready(state.h.data)
        diag = compute_conservation_diagnostics(state, grid)
        diagnostics.append(diag)

        day = step * DT / 86400.0
        mass_rel = float((diag['total_mass'] - diagnostics[0]['total_mass'])
                         / diagnostics[0]['total_mass'])
        energy_rel = float((diag['total_energy'] - diagnostics[0]['total_energy'])
                           / diagnostics[0]['total_energy'])
        h_min = float(jnp.min(state.h.data))
        h_max = float(jnp.max(state.h.data))
        umax = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))

        elapsed = time.time() - t_lap
        t_lap = time.time()

        print(f"{step:8d}  {day:6.1f}  {h_min:8.0f}  {h_max:8.0f}  "
              f"{umax:8.1f}  {mass_rel:10.2e}  {energy_rel:10.2e}  {elapsed:7.1f}")

    # Save snapshot
    if step in snapshot_steps:
        snapshots[step] = state
        # Compute error norms vs initial state
        norms = compute_error_norms(state, state_init, grid)
        error_norms_history.append((step * DT / 86400.0, norms))

total_time = time.time() - t_start
steps_per_sec = n_steps / total_time
print("-" * 85)
print(f"Completed in {total_time:.1f}s ({steps_per_sec:.0f} steps/s)")

# =====================================================================
# Visualization
# =====================================================================
print("\n\nGenerating visualizations...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Style
plt.rcParams.update({
    'font.size': 11,
    'figure.facecolor': 'white',
})

lon_deg = np.asarray(grid.lon) * 180 / np.pi
lat_deg = np.asarray(grid.lat) * 180 / np.pi
point_size = max(1.0, 120 / RESOLUTION)


def scatter_field(ax, data, cmap, vmin, vmax, alpha=0.9):
    """Scatter plot a field on all 6 faces."""
    for face in range(6):
        sc = ax.scatter(
            lon_deg[face].ravel(), lat_deg[face].ravel(),
            c=np.asarray(data[face]).ravel(),
            s=point_size, cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), edgecolors="none", alpha=alpha,
        )
    return sc


# ------------------------------------------------------------------
# 1. Height field snapshots (4-panel: days 0, 15, 50, 100)
# ------------------------------------------------------------------
print("  [1/5] Height field snapshots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                          subplot_kw={"projection": ccrs.Robinson()})
axes = axes.ravel()

for idx, (step_num, snap) in enumerate(snapshots.items()):
    day = step_num * DT / 86400.0
    ax = axes[idx]
    h_data = np.asarray(snap.h.data)
    sc = scatter_field(ax, snap.h.data, "RdYlBu_r",
                       vmin=5000, vmax=6100)
    ax.coastlines(linewidth=0.5, color="gray")
    ax.gridlines(linewidth=0.3, alpha=0.4)
    ax.set_global()
    ax.set_title(f"Day {day:.0f}", fontsize=13, fontweight="bold")

fig.suptitle("Williamson Test 5 — Fluid Depth h [m]  (C48, 100 days)",
             fontsize=15, fontweight="bold")
cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.05, orientation="horizontal",
                     label="Fluid depth [m]")
plt.savefig(f"{OUTPUT_DIR}/height_snapshots.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 2. Height perturbation (h - h_init) at day 100
# ------------------------------------------------------------------
print("  [2/5] Height perturbation at day 100...")
fig, ax = plt.subplots(1, 1, figsize=(14, 7),
                        subplot_kw={"projection": ccrs.Robinson()})

last_step = max(snapshots.keys())
h_pert = np.asarray(snapshots[last_step].h.data - state_init.h.data)
vabs = max(abs(float(np.nanmin(h_pert))), abs(float(np.nanmax(h_pert))))
# Clamp for visualization
vabs = min(vabs, 500)

sc = scatter_field(ax, h_pert, "RdBu_r", -vabs, vabs)
ax.coastlines(linewidth=0.5, color="gray")
ax.gridlines(linewidth=0.3, alpha=0.4)
ax.set_global()
ax.set_title(f"Height Perturbation h(day 100) − h(day 0)  [m]",
             fontsize=14, fontweight="bold")
plt.colorbar(sc, ax=ax, shrink=0.6, orientation="horizontal",
             label="Δh [m]", pad=0.06)
plt.savefig(f"{OUTPUT_DIR}/height_perturbation_day100.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 3. Wind speed + vorticity at day 50
# ------------------------------------------------------------------
print("  [3/5] Wind speed and vorticity at day 50...")
day50_step = int(50 * 86400 / DT)
snap50 = snapshots.get(day50_step, snapshots[last_step])

fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                          subplot_kw={"projection": ccrs.Robinson()})

# Wind speed
u50 = np.asarray(snap50.u.data)
v50 = np.asarray(snap50.v.data)
speed = np.sqrt(u50**2 + v50**2)
sc1 = scatter_field(axes[0], speed, "magma", 0, float(np.nanmax(speed)))
axes[0].coastlines(linewidth=0.5, color="gray")
axes[0].gridlines(linewidth=0.3, alpha=0.4)
axes[0].set_global()
axes[0].set_title("Wind Speed [m/s]  —  Day 50", fontsize=13, fontweight="bold")
plt.colorbar(sc1, ax=axes[0], shrink=0.65, orientation="horizontal",
             label="Speed [m/s]", pad=0.06)

# Vorticity
vort = curl_z(snap50.u, snap50.v, grid)
vort_data = np.asarray(vort.data) * 1e5  # Scale to 10^-5 /s
vort_abs = min(float(np.nanmax(np.abs(vort_data))), 5.0)
sc2 = scatter_field(axes[1], vort_data, "RdBu_r", -vort_abs, vort_abs)
axes[1].coastlines(linewidth=0.5, color="gray")
axes[1].gridlines(linewidth=0.3, alpha=0.4)
axes[1].set_global()
axes[1].set_title("Relative Vorticity [×10⁻⁵ s⁻¹]  —  Day 50",
                  fontsize=13, fontweight="bold")
plt.colorbar(sc2, ax=axes[1], shrink=0.65, orientation="horizontal",
             label="Vorticity [×10⁻⁵ s⁻¹]", pad=0.06)

plt.suptitle("Williamson Test 5 — C48 Shallow Water", fontsize=15, fontweight="bold")
plt.savefig(f"{OUTPUT_DIR}/wind_vorticity_day50.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 4. Conservation diagnostics time series
# ------------------------------------------------------------------
print("  [4/5] Conservation time series...")
n_diag = len(diagnostics)
diag_dt = SAVE_DIAG_EVERY * DT
times_days = np.arange(n_diag) * diag_dt / 86400.0
# The first entry is at step 0, rest at every SAVE_DIAG_EVERY steps
# Adjust: step 0 is at time 0, step SAVE_DIAG_EVERY is at diag_dt, etc.

mass = np.array([float(d['total_mass']) for d in diagnostics])
energy = np.array([float(d['total_energy']) for d in diagnostics])

mass_rel = (mass - mass[0]) / mass[0]
energy_rel = (energy - energy[0]) / energy[0]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(times_days, mass_rel, 'b-', linewidth=1.5)
axes[0].set_ylabel("Relative mass change", fontsize=12)
axes[0].set_title("Mass Conservation", fontsize=13, fontweight="bold")
axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
axes[0].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
axes[0].grid(True, alpha=0.3)

axes[1].plot(times_days, energy_rel, 'r-', linewidth=1.5)
axes[1].set_ylabel("Relative energy change", fontsize=12)
axes[1].set_xlabel("Time [days]", fontsize=12)
axes[1].set_title("Energy Conservation", fontsize=13, fontweight="bold")
axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
axes[1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
axes[1].grid(True, alpha=0.3)

fig.suptitle(f"Conservation Diagnostics — Williamson Test 5 (C{RESOLUTION}, {DURATION_DAYS} days)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/conservation.png", dpi=150, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 5. Mountain topography + initial height
# ------------------------------------------------------------------
print("  [5/5] Mountain topography...")
fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                          subplot_kw={"projection": ccrs.Robinson()})

# Topography
h_s = np.asarray(state_init.h_s.data)
sc1 = scatter_field(axes[0], state_init.h_s.data, "terrain",
                    0, float(np.nanmax(h_s)))
axes[0].coastlines(linewidth=0.5, color="gray")
axes[0].gridlines(linewidth=0.3, alpha=0.4)
axes[0].set_global()
axes[0].set_title("Surface Topography h_s [m]", fontsize=13, fontweight="bold")
plt.colorbar(sc1, ax=axes[0], shrink=0.65, orientation="horizontal",
             label="h_s [m]", pad=0.06)

# Initial height
sc2 = scatter_field(axes[1], state_init.h.data, "RdYlBu_r", 5000, 6100)
axes[1].coastlines(linewidth=0.5, color="gray")
axes[1].gridlines(linewidth=0.3, alpha=0.4)
axes[1].set_global()
axes[1].set_title("Initial Fluid Depth h [m]", fontsize=13, fontweight="bold")
plt.colorbar(sc2, ax=axes[1], shrink=0.65, orientation="horizontal",
             label="h [m]", pad=0.06)

plt.suptitle("Williamson Test 5 — Setup", fontsize=15, fontweight="bold")
plt.savefig(f"{OUTPUT_DIR}/setup.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Resolution:        C{RESOLUTION} ({grid.n_cells:,} cells)")
print(f"  Duration:          {DURATION_DAYS} days ({n_steps:,} steps)")
print(f"  Wall time:         {total_time:.1f}s ({steps_per_sec:.0f} steps/s)")
print(f"  Final mass Δ:      {mass_rel[-1]:.2e}")
print(f"  Final energy Δ:    {energy_rel[-1]:.2e}")
print(f"  Final h range:     [{float(jnp.min(state.h.data)):.0f}, "
      f"{float(jnp.max(state.h.data)):.0f}] m")
print(f"  Final |u| max:     {float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2))):.1f} m/s")

if error_norms_history:
    for day, norms in error_norms_history:
        print(f"  Error norms day {day:.0f}: "
              f"L1={norms['l1']:.4e}  L2={norms['l2']:.4e}  Linf={norms['linf']:.4e}")

print(f"\n  Figures saved to: {OUTPUT_DIR}/")
print(f"    - height_snapshots.png")
print(f"    - height_perturbation_day100.png")
print(f"    - wind_vorticity_day50.png")
print(f"    - conservation.png")
print(f"    - setup.png")
print("=" * 70)

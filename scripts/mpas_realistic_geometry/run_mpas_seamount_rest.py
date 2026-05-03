"""MPAS seamount rest-state diagnostic — canonical PGF-over-topography test.

P6 of ``docs/ocean_experiments/realistic_geometry_mpas_plan.md``,
mirroring the lat-lon ``run_phase3a_seamount.py``.  An isolated
Gaussian seamount on an otherwise-flat-bottom ico mesh; stratified
T(z); no forcing; integrate for several hours.  In a true ocean the
horizontal PGF is zero (hydrostatic balance) and ``|u|max`` should
stay near machine zero.  The Adcroft-Campin face correction
(``pgf_scheme="adcroft"``) is the partial-cell stack's primary
mechanism for keeping the seamount-induced PGF residual bounded.

Comparison runs:
    pgf_scheme="centered"  (no AC correction)
    pgf_scheme="adcroft"   (AC face correction; default for partial cells)

Acceptance:
- Both schemes integrate for the full duration without NaN.
- ``|u|max`` for the AC run stays below ~1e-3 m/s for the chosen
  resolution, with the centered run typically larger.
  (At ico-2 the absolute numbers are O(1e-5); at ico-4 they should
  drop further as dx halves.)
- AC residual is concentrated near the seamount edge (visible in
  the saved ``seamount_uvel.png`` snapshot) and not basin-wide.

Usage:
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/run_mpas_seamount_rest.py
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/run_mpas_seamount_rest.py --subdivision 4 --hours 24
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import (
    compute_centroid_depth,
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


def make_seamount_bathymetry(
    mesh, H_max: float, seamount_height: float = 2000.0,
    seamount_radius_deg: float = 15.0, seamount_lat_deg: float = 0.0,
    seamount_lon_deg: float = 0.0,
):
    """Gaussian seamount: H = H_max - seamount_height * exp(-r^2 / sigma^2)."""
    lat = np.asarray(mesh.latCell)
    lon = np.asarray(mesh.lonCell)
    lat0 = np.deg2rad(seamount_lat_deg)
    lon0 = np.deg2rad(seamount_lon_deg)
    # Approximate great-circle distance (small-angle ok for a single bump).
    dlat = lat - lat0
    dlon = (lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    r2 = dlat ** 2 + (dlon * np.cos(lat0)) ** 2
    sigma2 = np.deg2rad(seamount_radius_deg) ** 2
    bump = seamount_height * np.exp(-r2 / sigma2)
    H = H_max - bump
    return jnp.asarray(H, dtype=jnp.float64)


def run(args):
    print(f"[mpas-seamount] mesh subdivision={args.subdivision}, "
          f"H_max={args.H_max} m, seamount={args.seamount_height} m")
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
    z_coord = create_ocean_z_star(n_levels=args.n_levels, H_max=args.H_max)
    H_bathy = make_seamount_bathymetry(
        mesh, H_max=args.H_max, seamount_height=args.seamount_height,
    )
    pc_coord = create_partial_cell_coordinate(z_coord, H_bathy)

    print(f"[mpas-seamount] nCells={mesh.nCells}, nEdges={mesh.nEdges}, "
          f"nlev={z_coord.n_levels}; H range = "
          f"[{float(jnp.min(H_bathy)):.0f}, {float(jnp.max(H_bathy)):.0f}] m")

    # Build a state with T initialized per actual centroid depth (the
    # physically meaningful rest state for partial cells over topography).
    centroid = compute_centroid_depth(jnp.zeros_like(H_bathy), H_bathy, pc_coord)
    T_per_cell = args.T_deep + (args.T_surf - args.T_deep) * jnp.exp(
        -centroid / _SCALE_DEPTH,
    )
    T_per_cell = jnp.where(pc_coord.is_active, T_per_cell, args.T_deep)

    state = rest_state_mpas_ocean(
        mesh, z_coord, H_max=args.H_max, land_lat_threshold=90.0,
    )
    state = state._replace(
        H_bathy=Field(
            data=H_bathy.astype(state.H_bathy.data.dtype),
            name="H_bathy", dims=("nCells",), units="m",
        ),
        T=Field(
            data=T_per_cell.astype(state.T.data.dtype),
            name="T", dims=("nCells", "nlev"), units="degC",
        ),
    )

    results = {}
    for scheme in args.schemes:
        cfg = MPASOceanConfig(
            barotropic_solver="implicit_cn",
            A_h=args.A_h, A_v=1.0e-3, K_v=1.0e-4,
            bottom_drag_r=1.1e-3,
            barotropic_implicit_pcg_tol=1.0e-10,
            barotropic_implicit_pcg_maxiter=300,
            min_water_column_m=1.0,
            pgf_scheme=scheme,
            pv_scheme="enstrophy",
        )
        model = MPASOceanModel(mesh, pc_coord, cfg)
        s = state
        n_steps = int(args.hours * 3600.0 / args.dt)
        max_u = 0.0
        t0 = time.time()
        for k in range(n_steps):
            s = model.step(s, dt=args.dt)
            mu = float(jnp.max(jnp.abs(s.u.data)))
            max_u = max(max_u, mu)
            if (k + 1) % max(1, n_steps // 10) == 0:
                print(f"  [{scheme:8s}] step {k+1}/{n_steps}: "
                      f"max|u|={mu:.3e}, max|eta|={float(jnp.max(jnp.abs(s.eta.data))):.3e}")
        elapsed = time.time() - t0
        print(f"[{scheme:8s}] done in {elapsed:.1f}s, "
              f"max|u| over run = {max_u:.3e} m/s")
        results[scheme] = (s, max_u)

    # Save velocity snapshot for the AC run.
    if "adcroft" in results:
        s_ac, _ = results["adcroft"]
        from legoesm.ocean.init_mpas import reconstruct_cell_velocity
        u_east, v_north = reconstruct_cell_velocity(s_ac.u.data, mesh)
        speed = np.asarray(jnp.sqrt(u_east[:, 0] ** 2 + v_north[:, 0] ** 2))
        lat_deg = np.asarray(jnp.degrees(mesh.latCell))
        lon_deg = np.asarray(jnp.degrees(mesh.lonCell))
        fig, ax = plt.subplots(figsize=(10, 5))
        sc = ax.scatter(lon_deg, lat_deg, c=speed, s=8, cmap="viridis")
        plt.colorbar(sc, ax=ax, label="surface speed [m/s]")
        ax.set_title(f"MPAS seamount rest, pgf=adcroft, t={args.hours}h")
        ax.set_xlabel("lon [deg]"); ax.set_ylabel("lat [deg]")
        out = Path(args.outdir) / "seamount_speed_adcroft.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"[mpas-seamount] snapshot saved to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subdivision", type=int, default=2,
                   help="Voronoi mesh subdivision level (2=162 cells, 4=2562, 5=10242)")
    p.add_argument("--n-levels", dest="n_levels", type=int, default=10)
    p.add_argument("--H-max", dest="H_max", type=float, default=4000.0)
    p.add_argument("--seamount-height", dest="seamount_height", type=float, default=2000.0)
    p.add_argument("--T-surf", dest="T_surf", type=float, default=20.0)
    p.add_argument("--T-deep", dest="T_deep", type=float, default=2.0)
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--dt", type=float, default=300.0)
    p.add_argument("--A-h", dest="A_h", type=float, default=1.0e4)
    # ``centered`` is the working partial-cell PGF on legoesm (its
    # ``dz_ref``-integrated p_prime already gives the small bare
    # gradient that AC is meant to recover).  ``adcroft`` adds a 260×-
    # too-large correction on the seamount and grows the rest-state
    # flow to ~0.6 m/s over 6 hours; kept available for diagnostic /
    # convention-comparison work but not used in production.  See
    # plan-doc status block "pgf_scheme=adcroft known-bad on legoesm".
    p.add_argument("--schemes", nargs="+", default=["centered"])
    p.add_argument("--outdir", default="outputs/mpas_seamount")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

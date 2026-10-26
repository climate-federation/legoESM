#!/usr/bin/env python
"""20-year ORCA1 tripolar run — complete MRC production config."""
import os, sys, time, argparse
from pathlib import Path
from functools import partial
import numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax; import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())
from legoesm.core.field import Field
from legoesm.grids.tripole import create_tripole_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
from legoesm.ocean.eos import scale_depth as _SD
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig, KPPConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig, GMRediConfig, VisbeckConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig, EnhancedDiffusionConfig

DT = 600.0
BLOCK = 200
OUTPUT = Path("results/ocean/tripole_20yr")

def snap_partial(H, z):
    azh = jnp.abs(z.z_half_ref); nl = z.n_levels; Hf = H.ravel()
    na = jnp.sum(azh[None,:] < Hf[:,None], axis=1)
    bl = jnp.clip(na-1, 0, nl-1); dzb = z.dz_ref[bl]; pt = Hf - azh[bl]
    fr = pt / jnp.maximum(dzb, 1e-10)
    zu = azh[bl]; zl = azh[jnp.minimum(bl+1, nl)]
    Hs = jnp.where(Hf-zu < zl-Hf, zu, zl)
    need = (fr < 0.3) & (fr > 0) & (Hf > 0)
    Hn = jnp.where(need, Hs, Hf)
    return jnp.where(Hn <= 0, 0.0, Hn).reshape(H.shape)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=float, default=20.0)
    p.add_argument("--etopo", default=os.environ.get("LEGOESM_ETOPO_PATH", "data/bathymetry/etopo_1deg.nc"))
    p.add_argument("--grid", default="data/grids/eORCA1.2_mesh_mask.nc")
    p.add_argument("--fold-convention", default="auto",
                   choices=["auto", "n_lon-1-i", "(n_lon-i)%n_lon"],
                   help="Tripole T-fold seam origin (default auto; auto raises "
                        "on a genuinely ambiguous near-constant fold row).")
    args = p.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    days = args.years * 365.0
    n_steps = int(days * 86400 / DT)
    n_blocks = n_steps // BLOCK

    geom = create_tripole_grid(args.grid, fold_convention=args.fold_convention)

    H_raw, mask = init_ocean_bathymetry(geom, BathymetryConfig(
        source="file", path=args.etopo, H_max=5500.0, H_min=200.0,
        smoothing_passes=2, r_factor_max=0.2, depth_is_negative=True,
        north_cap_lat=None, south_cap_lat=-75.0))
    H_raw = jnp.asarray(H_raw, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=jnp.float64)
    z_base = create_ocean_z_star(n_levels=20, H_max=5500.0, dz_surface=20.0, dz_deep=500.0)
    H_snap = snap_partial(H_raw, z_base)
    mask = jnp.where(H_snap > 0, mask, 0.0)
    z_coord = create_partial_cell_coordinate(z_base, H_snap)
    n_ocean = int(jnp.sum(mask > 0.5))

    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="combined",
            prescribed=PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.1,
                tropical_wind_scale=0.5, tropical_wind_lat_deg=15.0),
            restoring=RestoringConfig(tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0, S_star=35.0, T_profile="cosine")),
        vertical_mixing=VerticalMixingConfig(scheme="kpp", kpp=KPPConfig(K_conv=1.0)),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0)),
        shortwave_penetration=None)
    oc = LatLonCGridOceanConfig.from_flat(
        # A_h=None = derive from this mesh's narrowest wet cell, anchored so a
        # ~1 degree mesh keeps 1e5 (resolved below, where the mask exists).
        A_h=None, C_smag_lap=0.33, A_h_floor=1000.0, A_v=1e-4, K_v=1e-5,
        bottom_drag_r=1e-3, bottom_drag_bbl_thickness=100.0, bottom_drag_bg_velocity=0.1,
        barotropic_solver="implicit_cn", pgf_scheme="adcroft",
        ke_gradient_scheme="hollingsworth",
        implicit_vertical_mixing=True, tracer_advection="tvd",
        eos="wright", freshwater_closure="virtual_salt_flux", S_ref=35.0,
        gm_redi=GMRediConfig(kappa_GM=2400.0, kappa_Redi=2400.0, S_max=0.01,
            visbeck=VisbeckConfig(enabled=False), slope_scheme="centered"),
        physics=physics)
    if oc.lateral_viscosity.A_h is None:
        from legoesm.grids.tripole import DEFAULT_MIN_DX_M
        from legoesm.ocean.state import (
            resolution_scaled_lateral_viscosity, wet_min_spacing,
        )
        _lv = oc.lateral_viscosity
        _dx_min = wet_min_spacing(geom, mask, clamp_floor_m=DEFAULT_MIN_DX_M)
        oc = oc._replace(lateral_viscosity=_lv._replace(
            A_h=resolution_scaled_lateral_viscosity(_dx_min, _lv),
            A_h_dx_m=_dx_min))
        print(f"  lateral viscosity from the mesh: "
              f"A_h={oc.lateral_viscosity.A_h:.6g} m2/s "
              f"(narrowest wet cell {_dx_min:.1f} m); note A_h_floor="
              f"{_lv.A_h_floor:g} m2/s still applies to the scaled field")
    model = LatLonCGridOceanModel(geom, z_coord, oc)
    state = rest_state_latlon_cgrid_ocean(geom, z_base, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=5500.0, land_mask_override=mask, H_bathy_override=H_snap)
    zf = np.asarray(z_base.z_full_ref)
    Tp = 2.0 + 18.0 * np.exp(zf / _SD)
    Td = np.array(state.T.data)
    for k in range(20): Td[..., k] = Tp[k]
    Td = Td * np.asarray(mask)[..., np.newaxis]
    state = state._replace(T=state.T.replace(data=jnp.array(Td)))

    def scan_body(st, _):
        return model.step(st, DT), None
    @partial(jax.jit, static_argnames=("n",))
    def block_fn(st, n: int):
        st, _ = jax.lax.scan(scan_body, st, None, length=n)
        return st

    # --- Snapshot helper ---
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    snap_dir = OUTPUT / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    mask_np = np.asarray(mask) > 0.5
    ocean_lat = np.degrees(np.asarray(geom.lat_T))[mask_np]
    ocean_lon = np.degrees(np.asarray(geom.lon_T))[mask_np]
    ocean_lon = np.where(ocean_lon > 180, ocean_lon - 360, ocean_lon)
    tri = mtri.Triangulation(ocean_lon, ocean_lat)
    lon_tri = ocean_lon[tri.triangles]
    tri.set_mask(np.max(lon_tri, axis=1) - np.min(lon_tri, axis=1) > 180)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        _has_cartopy = True
    except ImportError:
        _has_cartopy = False

    def save_snapshot(state, day, yr):
        u_np = np.asarray(state.u.data); v_np = np.asarray(state.v.data)
        u_cell = 0.5 * (u_np[:, :-1, :] + u_np[:, 1:, :])
        v_cell = 0.5 * (v_np[:-1, :, :] + v_np[1:, :, :])
        speed = np.sqrt(u_cell[:,:,0]**2 + v_cell[:,:,0]**2)
        eta = np.asarray(state.eta.data)
        sst = np.asarray(state.T.data[:, :, 0])
        sss = np.asarray(state.S.data[:, :, 0])
        fields = [
            ("Surface speed", speed[mask_np], "magma", 0, max(0.01, np.nanpercentile(speed[mask_np], 99)), "m/s"),
            ("SSH", eta[mask_np], "RdBu_r", None, None, "m"),
            ("SST", sst[mask_np], "RdYlBu_r", None, None, "°C"),
            ("SSS", sss[mask_np], "YlGnBu", None, None, "PSU"),
        ]
        fig = plt.figure(figsize=(20, 14))
        for idx, (title, f, cmap, vmin, vmax, unit) in enumerate(fields):
            if vmin is None:
                vm = max(1e-6, np.nanpercentile(np.abs(f), 99))
                vmin, vmax = (-vm, vm) if "RdBu" in cmap else (np.nanpercentile(f, 1), np.nanpercentile(f, 99))
            if _has_cartopy:
                ax = fig.add_subplot(2, 2, idx+1, projection=ccrs.Robinson())
                ax.set_global()
                ax.add_feature(cfeature.LAND, color="lightgray", zorder=1)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=2)
                tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax,
                                  transform=ccrs.PlateCarree(), shading="flat", zorder=0)
            else:
                ax = fig.add_subplot(2, 2, idx+1)
                tc = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
                ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            plt.colorbar(tc, ax=ax, label=unit, shrink=0.7)
            ax.set_title(title, fontsize=13)
        fig.suptitle(f"ORCA1 Tripolar — Day {day:.0f} (Year {yr:.2f})", fontsize=15)
        fig.tight_layout()
        fname = snap_dir / f"snapshot_day{int(day):06d}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return fname

    # --- Integration ---
    print(f"=== ORCA1 tripolar {args.years:.0f}-year run ===", flush=True)
    print(f"{n_ocean} cells, dt={DT}s, {n_blocks} blocks x {BLOCK}", flush=True)
    print(f"KPP+GM/Redi+TVD+Smag+Hollingsworth+Wright+Adcroft", flush=True)
    print(f"Output every 3 months (restarts + snapshots)", flush=True)
    t0 = time.time()
    steps_done = 0
    # Output every ~91.25 days (3 months)
    output_interval_days = 91.25
    output_every = max(1, int(output_interval_days * 86400 / DT / BLOCK))
    last_output_day = -1.0

    for b in range(n_blocks):
        state = block_fn(state, BLOCK)
        steps_done += BLOCK
        day = steps_done * DT / 86400.0
        yr = day / 365.0

        do_output = (b+1) % output_every == 0 or (b+1) == n_blocks
        if do_output:
            jax.block_until_ready(state.eta.data)
            mu = float(jnp.max(jnp.abs(state.u.data)))
            me = float(jnp.max(jnp.abs(state.eta.data)))
            T = state.T.data
            elapsed = time.time() - t0
            rate = steps_done / elapsed
            print(f"  Year {yr:>5.1f}: max|u|={mu:.3e} max|eta|={me:.3e} "
                  f"T=[{float(jnp.min(T)):.1f},{float(jnp.max(T)):.1f}] "
                  f"({rate:.0f} steps/s, {elapsed/3600:.1f}h)", flush=True)
            if np.isnan(mu):
                print("BLOWUP", flush=True); break

            # Save restart
            payload = {"step": steps_done, "day": day, "year": yr}
            for f in state._fields:
                obj = getattr(state, f)
                if obj is not None and hasattr(obj, "data"):
                    payload[f] = np.asarray(obj.data)
            np.savez_compressed(OUTPUT / f"restart_day{int(day):06d}.npz", **payload)

            # Save snapshot
            fname = save_snapshot(state, day, yr)
            print(f"    Saved: {fname.name}", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== Done: {yr:.1f} years in {elapsed/3600:.1f} hours ===", flush=True)
    print(f"  All finite: {bool(jnp.all(jnp.isfinite(state.u.data)))}", flush=True)

if __name__ == "__main__":
    main()

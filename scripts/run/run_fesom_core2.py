"""Global FESOM2 (fesom-jax) CORE2 hindcast arm for the cross-grid comparison.

The user's standing ask is agreement between THREE grids — tripole (eORCA1),
MPAS Voronoi, and FESOM2 — against the NEMO ORCA1 oracle.  The tripole and
MPAS arms run through run_omip_core2.py; this driver runs the FESOM2 arm
through the fesom-jax CORE2 realistic setup (arXiv:2608.01546, Zenodo
21324319: CORE2 mesh, PHC3.0 cold start, JRA55-do 1958 forcing) — the
configuration whose hindcast the FESOM2-JAX paper validates, i.e. an
independently-validated third dynamical core on the third grid family.

DELIBERATE PROTOCOL DIFFERENCES vs the xgrid matched pair (state them with
every number): JRA55-do 1958 forcing (not CORE-II normal-year), PHC3.0 winter
IC (not annual WOA), fesom-jax physics defaults (not the ORCA1 zdftke card).
Cross-grid numbers from this arm are therefore a THREE-MODEL comparison, not
a one-variable pair; the shared reference stays the NEMO month climatology in
compare_three_way_nemo.py, which regrids node clouds exactly like MPAS cells.

Outputs snapshot_day*.npz with the same key conventions the comparator's
_load_legoesm expects (T3d/S3d as (n, nlev) node arrays plus lat/lon/mask),
and a run_manifest.json with the full provenance.

Usage (inside an sbatch with the FESOM_* env vars set — see
scripts/cluster/omip_nemo/_fesom2_core2_d30.sbatch):
    python scripts/run/run_fesom_core2.py --days 30 \
        --mesh-dir data/fesom2_core2/mesh_ic/core2_mesh_ic/mesh_core2 \
        --ic-dir   data/fesom2_core2/mesh_ic/core2_mesh_ic/ic_core2 \
        --output results/omip_nemo/fesom_core2_d30
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh-dir", required=True)
    p.add_argument("--ic-dir", required=True)
    p.add_argument("--days", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=1800.0,
                   help="Timestep [s]; 1800 is the published CORE2 setting.")
    p.add_argument("--year", type=int, default=1958,
                   help="JRA55-do forcing year (the Zenodo package ships 1958).")
    p.add_argument("--snapshot-every-days", type=float, default=30.0)
    p.add_argument("--output", required=True)
    return p


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]),
             "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort
        return "unknown"


def write_snapshot(out_dir: Path, tag: str, state, mesh) -> Path:
    """Snapshot in the comparator's node-cloud conventions.

    fesom-jax state: T/S are (nl-1, nod2D) level-first; the comparator's
    unstructured path (shared with MPAS) wants cell/node-last-level (n, nlev)
    plus 1-D lat/lon in DEGREES and a wet mask.  mesh.geo_coord_nod2D is
    (nod2D, 2) = (lon, lat) in RADIANS (measured; the array's layout is an
    API).  eta is (nod2D,).
    """
    T = np.asarray(state.T, dtype=np.float64)
    S = np.asarray(state.S, dtype=np.float64)
    if T.shape[0] == mesh.nod2D:          # tolerate either orientation
        T3, S3 = T, S
    else:
        T3, S3 = T.T, S.T
    # fesom-jax state tracers carry nl slots: nl-1 real levels plus a padded
    # bottom slot (repeated bottom value).  mesh.Z has the nl-1 real
    # midpoints; slice tracers to match or every downstream (T, z) pairing is
    # off by one (measured: MLD diagnostic broadcast (1,47) vs (n,48)).
    nreal = np.asarray(mesh.Z).size
    T3, S3 = T3[:, :nreal], S3[:, :nreal]
    lon = np.degrees(np.asarray(mesh.geo_coord_nod2D[:, 0], dtype=np.float64))
    lat = np.degrees(np.asarray(mesh.geo_coord_nod2D[:, 1], dtype=np.float64))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    wet = np.asarray(mesh.node_layer_mask[:, 0], dtype=np.float64)
    z_center = -np.asarray(mesh.Z, dtype=np.float64)        # positive-down
    H = -np.asarray(mesh.depth, dtype=np.float64)           # positive-down
    path = out_dir / f"snapshot_{tag}.npz"
    np.savez_compressed(
        path,
        T=T3, S=S3,
        eta=np.asarray(state.eta_n, dtype=np.float64),
        ice_concentration=np.asarray(state.a_ice, dtype=np.float64),
        ice_thickness=np.asarray(state.m_ice, dtype=np.float64),
        lat_T=lat, lon_T=lon, land_mask=wet,                # 1.0 == OCEAN
        H_bathy=H, z_center_ref=z_center,
    )
    return path


def main() -> int:
    args = build_arg_parser().parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from fesom_jax.mesh import load_mesh
    from fesom_jax.phc_ic import cold_start_state
    from fesom_jax import surface_forcing
    from fesom_jax.ssh import build_ssh_operator
    from fesom_jax.integrate import integrate
    from fesom_jax.kpp import KppConfig
    from fesom_jax.gm import GMConfig
    from fesom_jax.ice import IceConfig

    mesh = load_mesh(args.mesh_dir)
    print(f"[mesh] nodes {mesh.nod2D:,} | triangles {mesh.elem2D:,} | "
          f"levels {mesh.nl}", flush=True)
    state = cold_start_state(mesh, args.ic_dir)
    sst0 = jnp.asarray(state.T[0] if state.T.shape[0] != mesh.nod2D
                       else state.T[:, 0])

    t0 = time.time()
    forcing = surface_forcing.build_surface_forcing(mesh, args.year, sst_ic=sst0)
    print(f"[forcing] JRA55-do {args.year} ready in {time.time()-t0:.1f} s",
          flush=True)

    op = build_ssh_operator(mesh, dt=args.dt)
    stress = jnp.zeros((mesh.elem2D, 2))

    steps_per_day = int(round(86400.0 / args.dt))
    n_days = int(round(args.days))
    manifest = {
        "run": {"command_line": " ".join(sys.argv),
                "creation_time": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "reproducibility": {"git_sha": _git_sha(),
                            "fesom_jax": str(Path(
                                sys.modules["fesom_jax"].__file__).parent)},
        "config": {"dt_s": args.dt, "days": n_days, "year": args.year,
                   "physics": "fesom-jax defaults (published CORE2 hindcast)",
                   "protocol_note": ("JRA55-do year forcing + PHC3.0 winter "
                                     "cold start; NOT the xgrid matched-pair "
                                     "protocol -- three-model comparison")},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Day-chunked integration: stacking all steps' forcing at once is ~6 GB
    # at 30 days; one day (48 steps) at a time keeps it ~200 MB.
    t_start = time.time()
    for day in range(1, n_days + 1):
        dates = surface_forcing.dates_for_steps(args.year, args.dt,
                                                day * steps_per_day)
        day_dates = dates[(day - 1) * steps_per_day: day * steps_per_day]
        step_forcings = forcing.stack(day_dates)
        # The published CORE2 hindcast card: KPP + GM + the sea-ice model.
        # Omitting ice_cfg DISABLES the ice model entirely -- the first run
        # did, and ice-covered Arctic water relaxed to the JRA winter air
        # temperature (SST median -18.6 C, min -26.7 C, measured on
        # fesom_core2_d30 snapshot_day0030 before this fix).
        state = integrate(state, mesh, op, stress,
                          n_steps=steps_per_day, dt=args.dt,
                          step_forcings=step_forcings,
                          forcing_static=forcing.static,
                          kpp_cfg=KppConfig(), gm_cfg=GMConfig(),
                          ice_cfg=IceConfig())
        sst = np.asarray(state.T[0] if state.T.shape[0] != mesh.nod2D
                         else state.T[:, 0])
        if not np.isfinite(sst[np.asarray(mesh.node_layer_mask[:, 0]) > 0]).all():
            write_snapshot(out, f"day{day:04d}_NONFINITE", state, mesh)
            raise SystemExit(f"FATAL: non-finite SST at day {day}")
        rate = day * steps_per_day / (time.time() - t_start)
        print(f"[day {day:3d}/{n_days}] mean SST "
              f"{float(np.nanmean(np.where(np.asarray(mesh.node_layer_mask[:, 0]) > 0, sst, np.nan))):.3f} C"
              f"  ({rate:.2f} steps/s)", flush=True)
        if day % max(1, int(round(args.snapshot_every_days))) == 0 or day == n_days:
            p = write_snapshot(out, f"day{day:04d}", state, mesh)
            print(f"[snapshot] {p}", flush=True)

    write_snapshot(out, "final", state, mesh)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

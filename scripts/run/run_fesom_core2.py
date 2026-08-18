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
    p.add_argument("--forcing", default="jra55",
                   choices=("jra55", "core2_nyf"),
                   help="Atmospheric forcing: 'jra55' = JRA55-do --year (the "
                        "published hindcast card); 'core2_nyf' = the CORE-II "
                        "normal-year store the NEMO reference and the legoESM "
                        "tripole/MPAS arms use (needs --nyf-zarr) -- the "
                        "matched-protocol option for the three-grid "
                        "comparison. SSS restoring/runoff/chl and the PHC IC "
                        "stay fesom-side either way; state those residual "
                        "differences with every scored number.")
    p.add_argument("--nyf-zarr", default=None,
                   help="Path to nyf.zarr (built by legoESM's "
                        "scripts/data/build_core2_nyf_zarr.py). Required with "
                        "--forcing core2_nyf.")
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
    # fesom-jax State contract: tracers are NODE-first (nod2D, nl).  Require
    # it rather than guessing from shapes (a mesh with nod2D == nl would make
    # any heuristic silently transpose wrong -- codex FESOM-arm review).
    if T.shape[0] != mesh.nod2D:
        raise SystemExit(f"state.T is not node-first: {T.shape} vs "
                         f"nod2D={mesh.nod2D}")
    T3, S3 = T, S
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
    from fesom_jax.step import step as fesom_step
    from fesom_jax.gm import GMConfig
    from fesom_jax.ice import IceConfig
    from fesom_jax.ale import AleConfig
    from fesom_jax.tke import TkeConfig

    mesh = load_mesh(args.mesh_dir)
    print(f"[mesh] nodes {mesh.nod2D:,} | triangles {mesh.elem2D:,} | "
          f"levels {mesh.nl}", flush=True)
    # Cavity nodes (ulevels > 1) would be misclassified as land by the
    # snapshot writer's surface-layer mask; CORE2 has none -- assert it.
    ul = np.asarray(mesh.ulevels_nod2D)
    if int(ul.max(initial=1)) > 1:
        raise SystemExit("FATAL: mesh has ice-shelf cavity nodes; the "
                         "snapshot writer does not support them")
    state = cold_start_state(mesh, args.ic_dir)
    if np.asarray(state.T).shape[0] != mesh.nod2D:
        raise SystemExit("cold_start_state returned non-node-first tracers")
    sst0 = jnp.asarray(state.T[:, 0])

    t0 = time.time()
    if args.forcing == "core2_nyf":
        if not args.nyf_zarr:
            raise SystemExit("--forcing core2_nyf requires --nyf-zarr")
        forcing = surface_forcing.build_surface_forcing(
            mesh, args.year, sst_ic=sst0, nyf_zarr=args.nyf_zarr)
        print(f"[forcing] CORE-II NYF ({args.nyf_zarr}) ready in "
              f"{time.time()-t0:.1f} s", flush=True)
    elif args.forcing == "jra55":
        forcing = surface_forcing.build_surface_forcing(mesh, args.year,
                                                        sst_ic=sst0)
        print(f"[forcing] JRA55-do {args.year} ready in {time.time()-t0:.1f} s",
              flush=True)
    else:  # argparse choices guard this; keep the dispatch loud anyway
        raise SystemExit(f"unknown --forcing {args.forcing!r}")

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
                   "forcing": args.forcing, "nyf_zarr": args.nyf_zarr,
                   "physics": ("core2_full.yaml paper card: zstar ALE + "
                               "prognostic TKE + GM + mEVP ice (whichEVP=1); "
                               "AB2-continuous day chunks (bootstrap once)"),
                   "protocol_note": (
                       "CORE-II NYF atmosphere (NEMO-matched) + PHC3.0 winter "
                       "cold start; SSS/runoff/chl remain fesom-side"
                       if args.forcing == "core2_nyf" else
                       "JRA55-do year forcing + PHC3.0 winter "
                       "cold start; NOT the xgrid matched-pair "
                       "protocol -- three-model comparison")},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    # AB2-CONTINUOUS DAY CHUNKS.  integrate() bootstraps AB2 on its first
    # step by design, so calling it once per day re-bootstrapped the momentum
    # time-stepping every 48 steps (codex FESOM-arm RED-1); but the full-run
    # stacked forcing OOMs the GPU (measured: RESOURCE_EXHAUSTED at 30 days).
    # So: bootstrap ONCE with the first step, then scan each day's forcing
    # with is_first_step=False -- exactly integrate()'s own internal split
    # (integrate.py:153-165), just re-entered per chunk so only one day of
    # forcing is resident.
    cfgs = dict(ale_cfg=AleConfig(), tke_cfg=TkeConfig(), gm_cfg=GMConfig(),
                ice_cfg=IceConfig(whichEVP=1))

    def _scan_day(state_in, sf_day):
        def body(carry, sf):
            return fesom_step(carry, mesh, op, stress, None, dt=args.dt,
                              is_first_step=False, step_forcing=sf,
                              forcing_static=forcing.static, **cfgs), None
        out_state, _ = jax.lax.scan(body, state_in, sf_day)
        return out_state

    _scan_day_jit = jax.jit(_scan_day)
    n_steps = n_days * steps_per_day
    # core2_nyf is a perpetual 365-day climatology: generate the calendar in a
    # fixed NON-LEAP year so no Feb-29 is ever injected into the cycle (codex
    # 9431498 HIGH -- a 1960 leap day would shift the atmospheric season by a
    # day and desynchronise the SSS/chl month), and refuse runs longer than
    # one cycle until a true no-leap generator exists.
    if args.forcing == "core2_nyf":
        if n_days > 365:
            raise SystemExit("--forcing core2_nyf supports <= 365 days per "
                             "run (perpetual-year calendar); split the run.")
        _date_year = 1959                     # any non-leap year
    else:
        _date_year = args.year
    all_dates = surface_forcing.dates_for_steps(_date_year, args.dt, n_steps)
    t_start = time.time()
    wetmask = np.asarray(mesh.node_layer_mask[:, 0]) > 0
    for day in range(1, n_days + 1):
        day_dates = all_dates[(day - 1) * steps_per_day: day * steps_per_day]
        sf_day = forcing.stack(day_dates)
        if day == 1:
            sf0 = jax.tree.map(lambda x: x[0], sf_day)
            state = fesom_step(state, mesh, op, stress, None, dt=args.dt,
                               is_first_step=True, step_forcing=sf0,
                               forcing_static=forcing.static, **cfgs)
            sf_day = jax.tree.map(lambda x: x[1:], sf_day)
        state = _scan_day_jit(state, sf_day)
        sst = np.asarray(state.T[:, 0])
        if not np.isfinite(sst[wetmask]).all():
            write_snapshot(out, f"day{day:04d}_NONFINITE", state, mesh)
            raise SystemExit(f"FATAL: non-finite SST at day {day}")
        rate = day * steps_per_day / (time.time() - t_start)
        print(f"[day {day:3d}/{n_days}] unweighted wet-node mean SST "
              f"{float(np.nanmean(np.where(wetmask, sst, np.nan))):.3f} C "
              f"(liveness only, refinement-biased)  ({rate:.2f} steps/s)",
              flush=True)
        if day % max(1, int(round(args.snapshot_every_days))) == 0 or day == n_days:
            p_out = write_snapshot(out, f"day{day:04d}", state, mesh)
            print(f"[snapshot] {p_out}", flush=True)

    write_snapshot(out, "final", state, mesh)
    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

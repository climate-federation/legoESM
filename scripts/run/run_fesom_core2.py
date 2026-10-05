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
    p.add_argument("--tke-surface-bc", default="neumann",
                   choices=("neumann", "dirichlet"),
                   help="Surface TKE BC: 'neumann' = the C-parity FESOM2 "
                        "reference (default); 'dirichlet' = the ported NEMO "
                        "en(1)=max(rn_emin0, rn_ebb*|tau|/rho0) boundary "
                        "value (fesom-jax cb389c7) -- the first ORCA1-card "
                        "branch for the three-grid convergence.")
    p.add_argument("--tke-mxl0-anchor", default="off", choices=("on", "off"),
                   help="NEMO ln_mxl0 surface mixing-length anchor (ORCA1 "
                        "sets .true.; NEMO pairs it with the Dirichlet BC — "
                        "running dirichlet without it is a half-port).")
    p.add_argument("--allow-half-ported-tke", action="store_true",
                   help="Permit --tke-surface-bc dirichlet with the surface "
                        "mixing-length anchor off. The reference card sets "
                        "both together, so this runs a configuration neither "
                        "model uses; it exists for isolating which of the two "
                        "moves a result, and must be stated deliberately.")
    p.add_argument("--ice-ic", default="fesom", choices=("fesom", "nemo"),
                   help="Sea-ice cold start: 'fesom' = the C-faithful "
                        "a_ice=0.9-where-SST<0 seed (SH m_ice=2 m); 'nemo' = "
                        "NEMO's January Ice_initialization.nc (needs "
                        "--ice-init-file) -- the NEMO-matched choice for a "
                        "January start (the fesom seed loads the summer SH "
                        "with ~40%% ice whose melt freshens the Antarctic).")
    p.add_argument("--ice-init-file", default=None,
                   help="Path to NEMO Ice_initialization.nc (at_i/ht_i/ht_s "
                        "on eORCA1). Required with --ice-ic nemo.")
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
                        "scripts/data/build_core2_nyf_zarr.py). Defaults to "
                        "the SAME cache the tripole and MPAS arms resolve "
                        "(legoesm.ocean.forcing.core2_nyf_cache_dir), so a "
                        "three-grid comparison cannot silently end up with "
                        "one grid on the raw CORE-II winds and another on "
                        "the bias-corrected ones.")
    p.add_argument("--snapshot-every-days", type=float, default=30.0)
    p.add_argument("--output", required=True)
    return p


def _git_sha() -> str:
    from legoesm.io.git_provenance import git_provenance
    return git_provenance(__file__).commit or "unknown"


def write_snapshot(out_dir: Path, tag: str, state, mesh, extra=None) -> Path:
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
    # Geographic node velocity (east, north) per layer for the equatorial-
    # undercurrent probe: element velocity -> nodes (fesom's own
    # compute_vel_nodes), then the exact inverse of the mesh rotation.
    _u_geo = {}
    if getattr(state, "uv", None) is not None:
        from fesom_jax.pp import compute_vel_nodes
        from legoesm.ocean.dynamics.ocean_model_fesom import (
            rotated_to_geographic_node_vector,
        )
        _uvn = np.asarray(compute_vel_nodes(mesh, state.uv))       # (nod2D, nl, 2)
        _cols = [rotated_to_geographic_node_vector(mesh, _uvn[:, k, 0], _uvn[:, k, 1])
                 for k in range(nreal)]
        _u_geo = {"u_east": np.stack([np.asarray(c[0]) for c in _cols], axis=1),
                  "v_north": np.stack([np.asarray(c[1]) for c in _cols], axis=1)}
    path = out_dir / f"snapshot_{tag}.npz"
    np.savez_compressed(
        path,
        T=T3, S=S3, **_u_geo,
        eta=np.asarray(state.eta_n, dtype=np.float64),
        ice_concentration=np.asarray(state.a_ice, dtype=np.float64),
        ice_thickness=np.asarray(state.m_ice, dtype=np.float64),
        lat_T=lat, lon_T=lon, land_mask=wet,                # 1.0 == OCEAN
        H_bathy=H, z_center_ref=z_center,
        **(extra or {}),
    )
    return path


def validate_tke_pair(args) -> None:
    """The ported turbulence card sets two things; refuse to run half of it.

    The reference model's card turns the Dirichlet surface value and the
    surface mixing-length anchor on TOGETHER, and the flags' own help calls
    either one alone a half port.  BOTH splits are refused, not just the one
    that reads more naturally: running the anchor against the default Neumann
    surface value is exactly as unported as the reverse, and an earlier
    version of this guard admitted it.
    """
    dirichlet = args.tke_surface_bc == "dirichlet"
    anchored = args.tke_mxl0_anchor == "on"
    if dirichlet == anchored or args.allow_half_ported_tke:
        return
    raise SystemExit(
        f"--tke-surface-bc {args.tke_surface_bc} with --tke-mxl0-anchor "
        f"{args.tke_mxl0_anchor} is a half port: the reference card sets the "
        "Dirichlet surface value and the mixing-length anchor together, and "
        "the pair is what was validated. Select both or neither, or pass "
        "--allow-half-ported-tke to run the split deliberately (the run is "
        "then not the ported one).")

def main() -> int:
    args = build_arg_parser().parse_args()
    validate_tke_pair(args)
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
    if args.ice_ic == "nemo":
        # January ice climatology instead of the C-faithful a_ice=0.9-where-
        # SST<0 seed, which loads the SH with ~40% mid-summer ice whose melt
        # builds a fresh Antarctic cap (measured 2026-08-18). The static
        # forcing a_ice mask is unchanged (prognostic ice supersedes it).
        from fesom_jax.nemo_ice_ic import seed_ice_from_nemo
        state = seed_ice_from_nemo(state, mesh, args.ice_init_file)
        print(f"[ice-ic] NEMO January climatology from {args.ice_init_file}",
              flush=True)
    elif args.ice_ic != "fesom":
        raise SystemExit(f"unknown --ice-ic {args.ice_ic!r}")
    if np.asarray(state.T).shape[0] != mesh.nod2D:
        raise SystemExit("cold_start_state returned non-node-first tracers")
    sst0 = jnp.asarray(state.T[:, 0])

    t0 = time.time()
    if args.ice_ic == "nemo" and not args.ice_init_file:
        raise SystemExit("--ice-ic nemo requires --ice-init-file")
    if args.forcing == "core2_nyf":
        nyf_zarr = args.nyf_zarr
        if not nyf_zarr:
            # Same resolution the tripole and MPAS drivers use, so the three
            # grids cannot drift onto different CORE-II wind fields.
            from legoesm.ocean.forcing import core2_nyf_cache_dir
            nyf_zarr = str(core2_nyf_cache_dir() / "nyf.zarr")
        if not os.path.exists(nyf_zarr):
            raise SystemExit(
                f"--forcing core2_nyf: no CORE-II cache at {nyf_zarr}. Build "
                "it with scripts/data/build_core2_nyf_zarr.py --wind-variant "
                "mod, or pass --nyf-zarr explicitly.")
        forcing = surface_forcing.build_surface_forcing(
            mesh, args.year, sst_ic=sst0, nyf_zarr=nyf_zarr)
        print(f"[forcing] CORE-II NYF ({nyf_zarr}) ready in "
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
                   "ice_ic": args.ice_ic, "ice_init_file": args.ice_init_file,
                   "tke_surface_bc": args.tke_surface_bc,
                   "tke_mxl0_anchor": args.tke_mxl0_anchor,
                   "allow_half_ported_tke": bool(args.allow_half_ported_tke),
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
    _tke_cfg = TkeConfig(
        use_dirichlet=(args.tke_surface_bc == "dirichlet"),
        use_mxl0_anchor=(args.tke_mxl0_anchor == "on"))
    cfgs = dict(ale_cfg=AleConfig(), tke_cfg=_tke_cfg, gm_cfg=GMConfig(),
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

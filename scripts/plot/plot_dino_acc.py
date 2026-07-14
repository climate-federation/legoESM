#!/usr/bin/env python
"""DINO barotropic streamfunction + ACC (channel) transport spin-up series.

For a lat-lon (Mercator C-grid) DINO run — the paper's grid type — computes:
  * the barotropic streamfunction Psi(x,y) [Sv] for the last snapshot (gyres +
    re-entrant channel jet, cf. Kamm et al. 2025 Fig. 4), and
  * the Antarctic-Circumpolar-Current transport time series [Sv]: the Drake-band
    throughflow at every snapshot, i.e. the spin-up trajectory toward the
    paper's R1 equilibrium value (206 Sv at 3000 yr).

The streamfunction + ACC reduction REUSE the canonical, partial-cell-aware
diagnostics (``ocean.diagnostics_streamfunction.barotropic_streamfunction`` +
``ocean.diagnostics_climate.acc_transport``) rather than re-deriving the
transport integral; only the partial-cell layer thickness is reconstructed
here from the snapshot's bottom depth + the z* reference grid.

Usage::

    JAX_ENABLE_X64=1 python scripts/plot/plot_dino_acc.py results/dino_latlon
    ... --out results/dino_acc.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAPER_ACC_R1_SV = 206.0   # Kamm et al. 2025, R1 (1deg) equilibrium channel transport
_SV = 1.0e6


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="lat-lon DINO output dir (results/dino_latlon)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG (default: <run_dir>/../dino_acc.png).")
    return p.parse_args()


def _grid_and_z(run_dir: Path):
    """Rebuild the lat-lon grid object + z* dz_ref from the run metadata."""
    from legoesm.ocean.experiments.dino import (
        DINOConfig, dino_lat_lon_grid, create_dino_z_star)
    with open(run_dir / "run_metadata.json") as f:
        meta = json.load(f)
    if not meta["grid"]["kind"].startswith("latlon"):
        raise SystemExit("plot_dino_acc requires a lat-lon run (the paper's "
                         "Mercator C-grid); MPAS barotropic-Psi needs TRiSK "
                         "velocity reconstruction (not implemented here).")
    n_lon = int(meta["args"].get("n_lon", 50))
    cfg = DINOConfig()
    grid = dino_lat_lon_grid(cfg, n_lon=n_lon)
    # Level count from the first snapshot: 35 = the NEMO-exact
    # masked-zco ladder (r1_exact preset), 36 = legacy z*.
    import dataclasses

    from legoesm.ocean.experiments.dino import dino_lat_lon_vertical
    snaps = _snaps(run_dir)
    nlev = None
    if snaps:
        with np.load(snaps[0]) as f0:
            nlev = int(f0["T"].shape[-1])
    if nlev == cfg.n_levels - 1:
        cfg = dataclasses.replace(cfg, vertical_coordinate="masked_zco")
        z = dino_lat_lon_vertical(grid, cfg)
    elif nlev in (None, cfg.n_levels):
        z = create_dino_z_star(cfg)
    else:
        raise SystemExit(
            f"snapshots have {nlev} levels; expected {cfg.n_levels} "
            f"(legacy z*) or {cfg.n_levels - 1} (masked_zco)")
    return cfg, grid, np.asarray(z.dz_ref)


def _snaps(run_dir: Path):
    return sorted((run_dir / "snapshots").glob("snapshot_*.npz"))


def main():
    args = _parse_args()
    cfg, grid, dz_ref = _grid_and_z(args.run_dir)
    from legoesm.ocean.diagnostics_streamfunction import (
        barotropic_streamfunction, partial_cell_thickness,
    )
    from legoesm.ocean.diagnostics_climate import acc_transport
    lat_deg = np.degrees(np.asarray(grid.lat))
    lon_deg = np.degrees(np.asarray(grid.lon))

    snaps = _snaps(args.run_dir)
    if not snaps:
        raise SystemExit(f"no snapshots in {args.run_dir}")
    days, acc, last_psi, last_day = [], [], None, None
    for s in snaps:
        with np.load(s) as d:
            u = np.asarray(d["u"]); mask = np.asarray(d["land_mask"])
            H_bathy = np.asarray(d["H_bathy"]); day = float(d["time_days"])
        if not np.isfinite(u).all():       # require ALL finite (skip blown snapshots)
            continue
        h_partial = partial_cell_thickness(H_bathy, dz_ref)
        psi_Sv = np.asarray(barotropic_streamfunction(u, h_partial, mask, grid))
        a = acc_transport(psi_Sv * _SV, lat_deg,
                          drake_lat_south=cfg.channel_lat_south_deg,
                          drake_lat_north=cfg.channel_lat_north_deg).transport_Sv
        if not np.isfinite(a):
            continue
        days.append(day); acc.append(a); last_psi = psi_Sv; last_day = day
    if not days:
        raise SystemExit("no finite snapshots")
    print(f"ACC transport: day {days[0]:.0f} = {acc[0]:.1f} Sv -> "
          f"day {days[-1]:.0f} = {acc[-1]:.1f} Sv  (paper R1 equilibrium {PAPER_ACC_R1_SV:.0f} Sv)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6))
    a0.plot(days, acc, "o-", color="C0", lw=2)
    a0.axhline(PAPER_ACC_R1_SV, color="C3", ls="--",
               label=f"Kamm 2025 R1 equilibrium ({PAPER_ACC_R1_SV:.0f} Sv, 3000 yr)")
    a0.set_xlabel("model day"); a0.set_ylabel("ACC channel transport [Sv]")
    a0.set_title("DINO ACC transport — spin-up trajectory"); a0.legend(); a0.grid(alpha=0.3)

    ext = [lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()]
    pmax = np.nanpercentile(np.abs(last_psi), 99) or 1.0
    im = a1.imshow(last_psi, origin="lower", extent=ext, aspect="auto",
                   cmap="RdBu_r", vmin=-pmax, vmax=pmax)
    a1.axhline(cfg.channel_lat_south_deg, color="k", lw=0.7, ls=":")
    a1.axhline(cfg.channel_lat_north_deg, color="k", lw=0.7, ls=":")
    a1.set_title(f"Barotropic streamfunction Psi [Sv], day {last_day:.0f}\n"
                 "(gyres + re-entrant channel; cf. Kamm 2025 Fig. 4)")
    a1.set_xlabel("lon"); a1.set_ylabel("lat"); fig.colorbar(im, ax=a1, label="Sv")
    fig.tight_layout()
    out = args.out or (args.run_dir.parent / "dino_acc.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

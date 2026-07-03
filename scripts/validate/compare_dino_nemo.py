"""Compare a legoESM DINO run against the NEMO DINO_R1 monthly reference.

Level-1 exact-match harness plots (docs/ocean/fidelity/dino_l1_exactness_audit.md):
maps (SST/SSS + diffs), zonal-mean T/S sections (+diffs), ACC / basin-mean
timeseries, and basin-mean profile drift — legoESM snapshots (run_dino NPZ)
vs the NEMO ``DINO_1m_grid_{T,U}.nc`` output.

Both grids are the same 1° Mercator DINO domain; NEMO carries 2 cyclic-halo
longitude columns (stripped) and one extra latitude row (nearest-lat aligned).

Usage:
    python scripts/validate/compare_dino_nemo.py \
        --legoesm-dir results/omip_nemo/dino_r1_exact \
        --nemo-gridt .../DINO_1m_grid_T.nc --nemo-gridu .../DINO_1m_grid_U.nc \
        --month 24 --out-dir results/omip_nemo/dino_compare_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legoesm import constants

# Fixed categorical identities (never cycled): legoESM blue, NEMO orange.
C_LEGO = "#4269d0"
C_NEMO = "#e8871a"
CMAP_SEQ_T = "viridis"      # magnitude: one perceptual ramp
CMAP_SEQ_S = "cividis"
CMAP_DIV = "RdBu_r"         # polarity: two hues, neutral midpoint


def _parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--legoesm-dir", type=Path, required=True)
    p.add_argument("--nemo-gridt", type=Path, required=True)
    p.add_argument("--nemo-gridu", type=Path, required=True)
    p.add_argument("--month", type=int, default=24,
                   help="1-based month index to compare (default 24)")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _load_lego(run_dir: Path, idx: int):
    f = np.load(run_dir / "snapshots" / f"snapshot_{idx:05d}.npz")
    return {k: np.asarray(f[k]) for k in f.files}


def _load_nemo(gridt: Path, gridu: Path):
    import netCDF4 as nc

    t = nc.Dataset(gridt)
    u = nc.Dataset(gridu)
    out = {
        "lat2d": np.asarray(t["nav_lat"][:], dtype=np.float64)[:, 1:-1],
        "deptht": np.asarray(t["deptht"][:], dtype=np.float64),
        # (time, z, y, x) -> strip the 2 cyclic-halo lon columns
        "T": np.asarray(t["toce"][:], dtype=np.float64)[..., 1:-1],
        "S": np.asarray(t["soce"][:], dtype=np.float64)[..., 1:-1],
        "e3t": np.asarray(t["e3t"][:], dtype=np.float64)[..., 1:-1],
        "u": np.asarray(u["uoce"][:], dtype=np.float64)[..., 1:-1],
        "e3u": np.asarray(u["e3u"][:], dtype=np.float64)[..., 1:-1],
    }
    fillv = 0.0
    for k in ("T", "S", "u"):
        arr = out[k]
        arr[np.abs(arr) > 1.0e10] = np.nan
        out[k] = arr
    t.close(); u.close()
    return out


def _align_nemo_lat(nemo_field, nemo_lat1d, lego_lat1d):
    """Nearest-lat alignment of NEMO's 199 rows onto legoESM's 198 (same
    Mercator construction; offsets are sub-cell)."""
    idx = np.abs(nemo_lat1d[None, :] - lego_lat1d[:, None]).argmin(axis=1)
    return nemo_field[..., idx, :]


def _dino_geometry(nlev: int):
    """legoESM DINO grid geometry (lat, dz_ref) from the experiment module.

    ``nlev`` comes from the run's snapshots: 35 = the NEMO-exact
    masked-zco ladder (r1_exact preset — jpk convention, analytic
    t-depths), 36 = the legacy z* midpoint ladder.
    """
    from legoesm.ocean.experiments.dino import (
        DINOConfig, create_dino_z_star, dino_lat_lon_grid,
        dino_lat_lon_vertical,
    )
    import dataclasses

    cfg = DINOConfig()
    if nlev == cfg.n_levels - 1:
        cfg = dataclasses.replace(cfg, vertical_coordinate="masked_zco")
    elif nlev != cfg.n_levels:
        raise SystemExit(
            f"snapshots have {nlev} levels; expected {cfg.n_levels} "
            f"(legacy z*) or {cfg.n_levels - 1} (masked_zco)")
    grid = dino_lat_lon_grid(cfg, n_lon=50)
    z = (dino_lat_lon_vertical(grid, cfg)
         if cfg.vertical_coordinate == "masked_zco"
         else create_dino_z_star(cfg))
    lat = np.degrees(np.asarray(grid.lat))
    return cfg, grid, lat, z


def _lego_acc_sv(lg, dz, cfg, grid):
    """Drake-band ACC [Sv] via the canonical partial-cell-aware barotropic
    streamfunction + ``acc_transport`` reduction (max−min of ψ in the
    channel band) — the same method as ``scripts/plot/plot_dino_acc.py``.
    """
    from legoesm.ocean.diagnostics_climate import acc_transport
    from legoesm.ocean.diagnostics_streamfunction import (
        barotropic_streamfunction, partial_cell_thickness,
    )
    u = np.asarray(lg["u"])
    if not np.isfinite(u).all():
        return float("nan")
    h_partial = partial_cell_thickness(
        np.asarray(lg["H_bathy"]), np.asarray(dz))
    psi_Sv = np.asarray(barotropic_streamfunction(
        u, h_partial, np.asarray(lg["land_mask"]), grid))
    lat_deg = np.degrees(np.asarray(grid.lat))
    return acc_transport(psi_Sv * 1e6, lat_deg,
                         drake_lat_south=cfg.channel_lat_south_deg,
                         drake_lat_north=cfg.channel_lat_north_deg).transport_Sv


def _sigma0_seos(T, S):
    """S-EOS potential density anomaly at z=0 (the DINO oracle EOS)."""
    from legoesm.ocean.eos import NemoSEOSConfig

    c = NemoSEOSConfig()
    Ta = T - 10.0
    Sa = S - 35.0
    return (-c.a0 * (1.0 + 0.5 * c.lambda1 * Ta) * Ta + c.b0 * Sa)


def _mld_003(T, S, depth_c):
    """MLD [m]: first depth where sigma0 exceeds the 10 m value by 0.03."""
    sig = _sigma0_seos(T, S)
    k10 = int(np.argmin(np.abs(depth_c - 10.0)))
    dsig = sig - sig[..., k10:k10 + 1]
    exceed = dsig > 0.03
    exceed[..., :k10 + 1] = False
    idx = np.argmax(exceed, axis=-1)
    mld = depth_c[idx]
    mld[~exceed.any(axis=-1)] = depth_c[-1]
    return mld


def main():
    args = _parse()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _probe = _load_lego(args.legoesm_dir, 0)
    cfg, grid, lat, z = _dino_geometry(int(_probe["T"].shape[-1]))
    dz = np.asarray(z.dz_ref)
    nlev = int(dz.size)
    # cell-centre depths from the coordinate's own t-depths (ANALYTIC
    # mi96 ladder on the masked-zco grid — NOT interface midpoints;
    # codex r4 HIGH #2)
    depth_c = np.abs(np.asarray(z.z_full_ref))
    nemo = _load_nemo(args.nemo_gridt, args.nemo_gridu)
    # NEMO jpk counts a permanently-masked dummy bottom level: trim
    # every 3-D field + the depth axis to the legoESM wet level count
    # so sections/profiles align (z is the FIRST axis of the nc arrays).
    for k in ("T", "S", "e3t", "u", "e3u"):
        if k in nemo and nemo[k] is not None and nemo[k].ndim == 4:
            nemo[k] = nemo[k][:, :nlev]
    nemo["deptht"] = nemo["deptht"][:nlev]
    nemo_lat1d = nemo["lat2d"][:, 0]
    m = args.month

    lego = _load_lego(args.legoesm_dir, m)
    lego0 = _load_lego(args.legoesm_dir, 0)
    mask = lego["land_mask"] > 0.5
    w = np.cos(np.deg2rad(lat))[:, None] * mask
    wsum = w.sum()

    def _nemoT(k, rec):
        f = _align_nemo_lat(nemo[k][rec], nemo_lat1d, lat)   # (z, y, x)
        return np.moveaxis(f, 0, -1)                          # (y, x, z)

    nT = _nemoT("T", m - 1)
    nS = _nemoT("S", m - 1)
    nT0 = _nemoT("T", 0)
    nS0 = _nemoT("S", 0)

    # ---------------- F1: SST / SSS maps + diffs ----------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    lon = np.arange(lego["T"].shape[1]) * 1.0
    for row, (le, ne, name, cmap, dl) in enumerate([
            (lego["T"][..., 0], nT[..., 0], "SST [°C]", CMAP_SEQ_T, 2.0),
            (lego["S"][..., 0], nS[..., 0], "SSS [PSU]", CMAP_SEQ_S, 0.5)]):
        le = np.where(mask, le, np.nan)
        vmin = np.nanmin([np.nanmin(le), np.nanmin(ne)])
        vmax = np.nanmax([np.nanmax(le), np.nanmax(ne)])
        for col, (fld, ttl) in enumerate([(le, f"legoESM {name}"),
                                          (ne, f"NEMO {name}")]):
            pc = axes[row, col].pcolormesh(lon, lat, fld, cmap=cmap,
                                           vmin=vmin, vmax=vmax)
            axes[row, col].set_title(ttl)
            fig.colorbar(pc, ax=axes[row, col], shrink=0.85)
        d = le - ne
        pc = axes[row, 2].pcolormesh(lon, lat, d, cmap=CMAP_DIV,
                                     vmin=-dl, vmax=dl)
        axes[row, 2].set_title(f"legoESM − NEMO (rms {np.sqrt(np.nanmean(d**2)):.3f})")
        fig.colorbar(pc, ax=axes[row, 2], shrink=0.85)
    for ax in axes.ravel():
        ax.set_xlabel("lon [°]"); ax.set_ylabel("lat [°]")
    fig.suptitle(f"DINO month {m}: legoESM (r1_exact) vs NEMO DINO_R1")
    fig.savefig(args.out_dir / "dino_compare_maps.png", dpi=140)
    plt.close(fig)

    # ---------------- F2: zonal-mean sections + diffs ----------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, (le3, ne3, name, cmap, dl) in enumerate([
            (lego["T"], nT, "T [°C]", CMAP_SEQ_T, 1.5),
            (lego["S"], nS, "S [PSU]", CMAP_SEQ_S, 0.3)]):
        le3 = np.where(mask[..., None], le3, np.nan)
        lz = np.nanmean(le3, axis=1)
        nz = np.nanmean(ne3, axis=1)
        vmin = np.nanmin([np.nanmin(lz), np.nanmin(nz)])
        vmax = np.nanmax([np.nanmax(lz), np.nanmax(nz)])
        for col, (fld, ttl) in enumerate([(lz, f"legoESM ⟨{name}⟩"),
                                          (nz, f"NEMO ⟨{name}⟩")]):
            pc = axes[row, col].pcolormesh(lat, depth_c, fld.T, cmap=cmap,
                                           vmin=vmin, vmax=vmax)
            axes[row, col].set_title(ttl)
            fig.colorbar(pc, ax=axes[row, col], shrink=0.85)
        d = lz - nz
        pc = axes[row, 2].pcolormesh(lat, depth_c, d.T, cmap=CMAP_DIV,
                                     vmin=-dl, vmax=dl)
        axes[row, 2].set_title(f"diff (rms {np.sqrt(np.nanmean(d**2)):.3f})")
        fig.colorbar(pc, ax=axes[row, 2], shrink=0.85)
    for ax in axes.ravel():
        ax.invert_yaxis()
        ax.set_xlabel("lat [°]"); ax.set_ylabel("depth [m]")
    fig.suptitle(f"DINO month {m}: zonal means")
    fig.savefig(args.out_dir / "dino_compare_sections.png", dpi=140)
    plt.close(fig)

    # ---------------- F3: timeseries (ACC, basin-mean SST/SSS) --------
    months = list(range(1, m + 1))
    acc_l, acc_n, sst_l, sst_n, sss_l, sss_n = [], [], [], [], [], []
    for mm in months:
        lg = _load_lego(args.legoesm_dir, mm)
        acc_l.append(_lego_acc_sv(lg, dz, cfg, grid))
        sst_l.append(float((np.where(mask, lg["T"][..., 0], 0.0) * w).sum() / wsum))
        sss_l.append(float((np.where(mask, lg["S"][..., 0], 0.0) * w).sum() / wsum))
        nu = _align_nemo_lat(nemo["u"][mm - 1], nemo_lat1d, lat)   # (z, y, x)
        ne3u = _align_nemo_lat(nemo["e3u"][mm - 1], nemo_lat1d, lat)
        dphi = np.gradient(lat) * np.pi / 180.0
        dy = constants.R_earth * dphi
        sec = np.nan_to_num(nu[:, :, 0] * ne3u[:, :, 0])          # (z, y)
        acc_n.append(float((sec * dy[None, :]).sum() / 1e6))
        nTm = _nemoT("T", mm - 1)[..., 0]
        nSm = _nemoT("S", mm - 1)[..., 0]
        wn = np.cos(np.deg2rad(lat))[:, None] * np.isfinite(nTm)
        sst_n.append(float(np.nansum(nTm * wn) / wn.sum()))
        sss_n.append(float(np.nansum(nSm * wn) / wn.sum()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (yl, yn, name, unit) in zip(axes, [
            (acc_l, acc_n, "ACC transport", "Sv"),
            (sst_l, sst_n, "basin-mean SST", "°C"),
            (sss_l, sss_n, "basin-mean SSS", "PSU")]):
        ax.plot(months, yl, color=C_LEGO, lw=2, label="legoESM r1_exact")
        ax.plot(months, yn, color=C_NEMO, lw=2, label="NEMO DINO_R1")
        ax.set_title(f"{name} [{unit}]")
        ax.set_xlabel("month")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("DINO trajectory comparison")
    fig.savefig(args.out_dir / "dino_compare_timeseries.png", dpi=140)
    plt.close(fig)

    # ---------------- F4: basin-mean profiles + drift ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    def _pmean(f3, msk):
        ww = np.cos(np.deg2rad(lat))[:, None, None] * msk[..., None]
        return np.nansum(np.where(msk[..., None], f3, 0.0) * ww, axis=(0, 1)) / ww.sum(axis=(0, 1))

    for ax, (l_now, l_ic, n_now, n_ic, name) in zip(axes, [
            (lego["T"], lego0["T"], nT, nT0, "T [°C]"),
            (lego["S"], lego0["S"], nS, nS0, "S [PSU]")]):
        nmask = np.isfinite(nT[..., 0])
        ax.plot(_pmean(l_now, mask), depth_c, color=C_LEGO, lw=2,
                label="legoESM")
        ax.plot(_pmean(np.nan_to_num(n_now), nmask), depth_c, color=C_NEMO,
                lw=2, label="NEMO")
        ax.plot(_pmean(l_ic, mask), depth_c, color=C_LEGO, lw=1, ls="--",
                label="legoESM IC")
        ax.plot(_pmean(np.nan_to_num(n_ic), nmask), depth_c, color=C_NEMO,
                lw=1, ls="--", label="NEMO month 1")
        ax.invert_yaxis()
        ax.set_xlabel(name); ax.set_ylabel("depth [m]")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"DINO basin-mean profiles (month {m} vs initial)")
    fig.savefig(args.out_dir / "dino_compare_profiles.png", dpi=140)
    plt.close(fig)

    # ---------------- F5: MLD maps (same 0.03 σ0 criterion) ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    mld_l = _mld_003(np.where(mask[..., None], lego["T"], np.nan),
                     np.where(mask[..., None], lego["S"], np.nan), depth_c)
    mld_n = _mld_003(nT, nS, depth_c)
    vmax = np.nanpercentile(np.concatenate([mld_l.ravel(), mld_n.ravel()]), 98)
    for ax, (fld, ttl) in zip(axes, [(np.where(mask, mld_l, np.nan), "legoESM MLD [m]"),
                                     (mld_n, "NEMO MLD [m]")]):
        pc = ax.pcolormesh(lon, lat, fld, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(ttl)
        fig.colorbar(pc, ax=ax, shrink=0.85)
    d = np.where(mask, mld_l, np.nan) - mld_n
    dl = np.nanpercentile(np.abs(d), 95)
    pc = axes[2].pcolormesh(lon, lat, d, cmap=CMAP_DIV, vmin=-dl, vmax=dl)
    axes[2].set_title(f"diff (rms {np.sqrt(np.nanmean(d**2)):.0f} m)")
    fig.colorbar(pc, ax=axes[2], shrink=0.85)
    for ax in axes:
        ax.set_xlabel("lon [°]"); ax.set_ylabel("lat [°]")
    fig.suptitle(f"DINO MLD (σ0 +0.03 criterion), month {m}")
    fig.savefig(args.out_dir / "dino_compare_mld.png", dpi=140)
    plt.close(fig)

    print(f"[compare_dino_nemo] wrote 5 figures to {args.out_dir}")
    print(f"  ACC month {m}: legoESM {acc_l[-1]:.1f} Sv, NEMO {acc_n[-1]:.1f} Sv")
    print(f"  SST rms diff: {np.sqrt(np.nanmean((np.where(mask, lego['T'][...,0], np.nan) - nT[...,0])**2)):.3f} °C")


if __name__ == "__main__":
    main()

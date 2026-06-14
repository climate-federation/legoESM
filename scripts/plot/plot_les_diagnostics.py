"""Publication-quality diagnostics for the plane boundary-layer LES.

Reads the frames written by ``scripts/run/run_les_plane.py`` under
``<output>/snapshots/snap_NNN.npz`` (horizontal cross-sections at the surface +
four heights) and ``<output>/profiles/prof_NNN.npz`` (planar-mean profiles +
resolved turbulence statistics) and renders:

* ``<output>/snapshots/snap_NNN.png`` — one figure per frame: a 3-row
  (vertical velocity w, potential-temperature anomaly θ', horizontal wind
  speed |U| with a wind quiver) by N-column (surface + 4 heights) panel of
  horizontal cross-sections.
* ``<output>/les_<case>_snapshot_final.png`` — the final frame, the hero figure.
* ``<output>/les_<case>_profile_evolution.png`` — a 2×3 panel of the mean
  θ(z), |U|(z), TKE(z), σ_w(z), resolved momentum flux u'w'(z) and the (u,v)
  hodograph, each overlaid for every recorded time and coloured by hour.

Styling targets print: serif/STIX maths, 300 dpi, perceptually-uniform /
diverging scientific colormaps, physical-unit axes and shared per-row colorbars.

Usage
-----
.. code-block:: bash

   python scripts/plot/plot_les_diagnostics.py results/les_gabls1
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Publication style.                                                           #
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
})

_CASE_TITLE = {
    "gabls1": "GABLS1 stable boundary layer",
    "ekman": "Ekman neutral boundary layer",
    "wangara": "Wangara convective boundary layer",
    "neutral": "Neutral boundary layer",
}


def _sym_limit(arr, pct=99.0):
    """Symmetric robust colour limit for a signed field."""
    v = float(np.nanpercentile(np.abs(arr), pct))
    return v if v > 0 else 1.0


def _panel_label(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, fontweight="bold", color="k",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.7))


# --------------------------------------------------------------------------- #
# Snapshot figure (cross-sections at surface + 4 heights).                    #
# --------------------------------------------------------------------------- #
def plot_snapshot(npz_path: Path, out_png: Path | None = None) -> Path:
    d = np.load(npz_path)
    case = str(d["case"])
    t_hours = float(d["t_hours"])
    heights = np.asarray(d["heights"])
    Lx = float(d["Lx"]) / 1000.0
    Ly = float(d["Ly"]) / 1000.0
    extent = [0.0, Lx, 0.0, Ly]
    nh = heights.shape[0]

    w = np.asarray(d["w"])             # (nh, ny, nx)
    theta = np.asarray(d["theta"])
    u = np.asarray(d["u"])
    v = np.asarray(d["v"])
    thp = theta - theta.mean(axis=(1, 2), keepdims=True)   # θ anomaly per height
    spd = np.sqrt(u ** 2 + v ** 2)

    rows = [
        ("w",   w,   "RdBu_r", "w  [m s$^{-1}$]",          True),
        ("thp", thp, "coolwarm", r"$\theta'$  [K]",        True),
        ("spd", spd, "viridis", r"$|U|$  [m s$^{-1}$]",    False),
    ]
    if "qc" in d.files:                         # moist run: cloud-water row
        rows.append(("qc", np.asarray(d["qc"]) * 1.0e3, "Blues",
                     r"$q_c$  [g kg$^{-1}$]", False))
    w_lim = _sym_limit(w)
    th_lim = _sym_limit(thp)

    # constrained_layout sizes the per-row colorbars with a real gap from the
    # panels (tight_layout jams them against the last column).
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, nh, figsize=(2.7 * nh + 1.2, 2.6 * nrows),
                             squeeze=False, layout="constrained")
    for r, (key, fld, cmap, label, sym) in enumerate(rows):
        if sym:
            lim = w_lim if key == "w" else th_lim
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = 0.0, float(np.nanpercentile(fld, 99.5))
            if key == "qc":                 # mostly-zero cloud field: scale to
                vmax = max(float(fld.max()), 1e-3)   # the max, not a percentile
        im = None
        for c in range(nh):
            ax = axes[r][c]
            interpolation = "nearest" if key == "qc" else "bilinear"
            im = ax.imshow(fld[c], origin="lower", extent=extent, cmap=cmap,
                           vmin=vmin, vmax=vmax, aspect="equal",
                           interpolation=interpolation, rasterized=True)
            if r == 0:
                ax.set_title(f"z = {heights[c]:.0f} m"
                             + ("  (surface)" if c == 0 else ""), fontsize=9)
            if key == "spd":                      # wind quiver on the |U| row
                ny, nx = fld[c].shape
                step = max(1, nx // 14)
                xs = np.linspace(0, Lx, nx)[::step]
                ys = np.linspace(0, Ly, ny)[::step]
                ax.quiver(*np.meshgrid(xs, ys),
                          u[c][::step, ::step], v[c][::step, ::step],
                          color="white", scale=None, width=0.006,
                          alpha=0.8, pivot="mid")
            if c == 0:
                ax.set_ylabel("y  [km]")
            else:
                ax.set_yticklabels([])
            if r == nrows - 1:
                ax.set_xlabel("x  [km]")
            else:
                ax.set_xticklabels([])
            _panel_label(ax, f"({chr(97 + r * nh + c)})")
        cbar = fig.colorbar(im, ax=axes[r].tolist(), shrink=0.92, aspect=28,
                            pad=0.03)
        cbar.set_label(label)

    fig.suptitle(
        f"{_CASE_TITLE.get(case, case)} — horizontal cross-sections, "
        f"t = {t_hours:.2f} h\n"
        rf"$\Delta x=\Delta y={float(d['dx']):.0f}$ m, "
        rf"domain {Lx:.1f}$\times${Ly:.1f} km",
        fontsize=12)
    out_png = out_png or npz_path.with_suffix(".png")
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# 3D stacked-slice figure (the recorded height cross-sections as horizontal    #
# planes in a perspective box — a genuine 3D view of the turbulent field).     #
# --------------------------------------------------------------------------- #
def plot_snapshot_3d(npz_path: Path, out_png: Path | None = None,
                     field: str = "w") -> Path:
    """Render the surface + four height cross-sections as semi-transparent
    coloured planes stacked at their physical heights in a 3D axes. Uses only
    the data already in ``snap_NNN.npz`` (no extra recording)."""
    d = np.load(npz_path)
    case = str(d["case"])
    t_hours = float(d["t_hours"])
    heights = np.asarray(d["heights"])              # (nh,) ascending [m]
    Lx = float(d["Lx"]) / 1000.0
    Ly = float(d["Ly"]) / 1000.0
    nh = heights.shape[0]

    if field == "w":
        fld = np.asarray(d["w"]); cmap = plt.get_cmap("RdBu_r")
        lim = _sym_limit(fld); norm = plt.Normalize(-lim, lim)
        clabel = r"$w$  [m s$^{-1}$]"
    else:                                            # θ anomaly per height
        th = np.asarray(d["theta"])
        fld = th - th.mean(axis=(1, 2), keepdims=True)
        cmap = plt.get_cmap("coolwarm")
        lim = _sym_limit(fld); norm = plt.Normalize(-lim, lim)
        clabel = r"$\theta'$  [K]"

    ny, nx = fld[0].shape
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    for k in range(nh):
        Z = np.full_like(X, float(heights[k]))
        fc = cmap(norm(fld[k]))
        fc[..., 3] = 0.78 if k < nh - 1 else 0.92    # lower planes more opaque
        ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False,
                        rasterized=True)
        ax.text(Lx * 1.02, 0.0, float(heights[k]), f"{heights[k]:.0f} m",
                fontsize=7, color="0.25")

    ax.set_xlabel("x  [km]", labelpad=8)
    ax.set_ylabel("y  [km]", labelpad=8)
    ax.set_zlabel("z  [m]", labelpad=8)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_zlim(0, float(heights[-1]) * 1.05)
    ax.set_box_aspect((1.0, 1.0, 0.85))
    ax.view_init(elev=22, azim=-58)
    ax.xaxis.pane.set_alpha(0.04); ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=22, pad=0.10)
    cbar.set_label(clabel)
    ax.set_title(f"{_CASE_TITLE.get(case, case)} — 3D cross-sections, "
                 f"t = {t_hours:.2f} h", fontsize=12)
    out_png = out_png or npz_path.with_name(npz_path.stem + "_3d.png")
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# Profile-evolution figure.                                                   #
# --------------------------------------------------------------------------- #
def plot_profile_evolution(prof_files, out_png: Path) -> Path:
    series = [np.load(p) for p in prof_files]
    hours = np.array([float(s["t_hours"]) for s in series])
    case = str(series[0]["case"])
    z = np.asarray(series[0]["z"])
    order = np.argsort(z)                                  # ascending z for lines
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=hours.min(),
                         vmax=max(hours.max(), hours.min() + 1e-9))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    def _line(ax, key, transform=None):
        for h, s in zip(hours, series):
            x = np.asarray(s[key])[order]
            if transform is not None:
                x = transform(x)
            ax.plot(x, z[order], color=cmap(norm(h)), lw=1.3, alpha=0.9)

    _line(axes[0, 0], "theta")
    axes[0, 0].set_xlabel(r"$\overline{\theta}$  [K]")

    _line(axes[0, 1], "spd")
    axes[0, 1].set_xlabel(r"$|\overline{U}|$  [m s$^{-1}$]")

    _line(axes[0, 2], "tke")
    axes[0, 2].set_xlabel(r"resolved TKE  [m$^2$ s$^{-2}$]")

    _line(axes[1, 0], "ww", transform=lambda a: np.sqrt(np.maximum(a, 0.0)))
    axes[1, 0].set_xlabel(r"$\sigma_w=\sqrt{\overline{w'^2}}$  [m s$^{-1}$]")

    _line(axes[1, 1], "uw")
    axes[1, 1].axvline(0.0, color="0.6", lw=0.7, ls="--")
    axes[1, 1].set_xlabel(r"resolved $\overline{u'w'}$  [m$^2$ s$^{-2}$]")

    # Hodograph: mean wind vector (u,v) through the column, coloured by time.
    axh = axes[1, 2]
    for h, s in zip(hours, series):
        axh.plot(np.asarray(s["u"])[order], np.asarray(s["v"])[order],
                 color=cmap(norm(h)), lw=1.3, alpha=0.9)
    axh.axhline(0.0, color="0.6", lw=0.7, ls="--")
    axh.axvline(0.0, color="0.6", lw=0.7, ls="--")
    axh.set_xlabel(r"$\overline{u}$  [m s$^{-1}$]")
    axh.set_ylabel(r"$\overline{v}$  [m s$^{-1}$]")
    axh.set_title("wind hodograph", fontsize=10)
    axh.set_aspect("equal", adjustable="datalim")

    for k, ax in enumerate(axes.flat):
        if ax is not axh:
            ax.set_ylabel("z  [m]")
            ax.grid(alpha=0.3)
            ax.set_ylim(z.min(), z.max())
        _panel_label(ax, f"({chr(97 + k)})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, pad=0.05)
    cbar.set_label("time  [h]")
    fig.suptitle(
        f"{_CASE_TITLE.get(case, case)} — mean-profile & turbulence evolution "
        f"({len(hours)} times, t = {hours.min():.2f}…{hours.max():.2f} h)",
        fontsize=13)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: plot_les_diagnostics.py <output_dir> [<output_dir> ...]")
        return 2
    rc = 0
    for arg in sys.argv[1:]:
        out_dir = Path(arg)
        snap_files = sorted((out_dir / "snapshots").glob("snap_*.npz"))
        prof_files = sorted((out_dir / "profiles").glob("prof_*.npz"))
        if not snap_files and not prof_files:
            print(f"[skip] no frames under {out_dir}")
            rc = 1
            continue
        case = str(np.load(prof_files[0])["case"]) if prof_files else out_dir.name
        for p in snap_files:
            png = plot_snapshot(p)
            png3d = plot_snapshot_3d(p)
            print(f"  wrote {png}  {png3d}")
        if snap_files:
            hero = out_dir / f"les_{case}_snapshot_final.png"
            plot_snapshot(snap_files[-1], hero)
            hero3d = out_dir / f"les_{case}_3d_final.png"
            plot_snapshot_3d(snap_files[-1], hero3d)
            print(f"  wrote {hero}  {hero3d}")
        if prof_files:
            evo = out_dir / f"les_{case}_profile_evolution.png"
            plot_profile_evolution(prof_files, evo)
            print(f"  wrote {evo}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

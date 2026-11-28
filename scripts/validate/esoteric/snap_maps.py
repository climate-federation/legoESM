"""Map panels (rows = checkpoints, cols = fields) from MPAS atmosphere checkpoints.

Wind is the instantaneous lowest-level eastward wind (Perot reconstruction).
Usage: snap_maps.py out.png mesh_level ridge_lon_deg ck1 [ck2 ...]
"""
import sys

import numpy as np

from zonal_state import cell_u_east

FIELDS = (("lowest-level T [K]", "RdYlBu_r", (250, 300)),
          ("surface pressure [hPa]", "viridis", (970, 1030)),
          ("lowest-level eastward wind [m/s] (snapshot)", "RdBu_r", (-12, 12)),
          ("T at model level 20 [K]", "RdYlBu_r", None))


def panel_fields(ck, mesh):
    """The four mapped fields of one checkpoint, each (nCells,), all finite."""
    T = np.asarray(ck["T"]); u = np.asarray(ck["u"])
    fields = [T[:, -1], np.asarray(ck["p_s"]) / 100, cell_u_east(u[:, -1:], mesh)[:, 0], T[:, 20]]
    for f, (name, _, _) in zip(fields, FIELDS):
        if not np.all(np.isfinite(f)):
            raise ValueError(f"non-finite {name}")
    return fields


def main(argv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from legoesm.grids.voronoi import create_voronoi_mesh

    out, lev, rlon, cks = argv[0], int(argv[1]), float(argv[2]), argv[3:]
    if not cks:
        raise ValueError("no checkpoints given")
    mesh = create_voronoi_mesh(lev)
    lat = np.degrees(np.asarray(mesh.latCell)); lon = (np.degrees(np.asarray(mesh.lonCell)) + 180) % 360 - 180
    tri = mtri.Triangulation(lon, lat)
    x = lon[tri.triangles]
    tri.set_mask((x.max(1) - x.min(1)) > 180)   # drop dateline-wrapping triangles (seam left blank)
    fig, axes = plt.subplots(len(cks), 4, figsize=(20, 3.6 * len(cks)), squeeze=False)
    for r, p in enumerate(cks):
        ck = np.load(p, allow_pickle=True)
        day = float(ck["day"])
        for k, (f, (name, cmap, lim)) in enumerate(zip(panel_fields(ck, mesh), FIELDS)):
            ax = axes[r, k]
            vmin, vmax = lim if lim else (np.percentile(f, 1), np.percentile(f, 99))
            m = ax.tripcolor(tri, f, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
            ax.axvline((rlon + 180) % 360 - 180, color="k", lw=0.8, ls="--")
            ax.set_title(f"day {day:.0f}: {name}", fontsize=9); ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            fig.colorbar(m, ax=ax, shrink=0.8)
    fig.suptitle(f"E&M2009 coupled, MPAS level {lev} (dashed = ridge at {rlon:g}E)")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])

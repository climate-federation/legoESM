"""Zonal-mean T and eastward wind by latitude band and level from an MPAS atmosphere checkpoint.

Eastward wind is the model's own Perot edge->cell reconstruction, area-weighted on
the same cells as T. Usage: zonal_state.py checkpoint_day_NNNN.npz mesh_level [expected_day]
"""
import sys

import numpy as np

from _common import BANDS, band_means


def cell_u_east(u_edge, mesh):
    """Cell-centre eastward wind [m/s], shape (nCells, nlev), from edge-normal u (nEdges, nlev)."""
    import jax.numpy as jnp
    from legoesm.grids.voronoi import reconstruct_cell_velocity

    u_east, _ = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    return np.asarray(u_east)


def main(argv):
    from legoesm.grids.voronoi import create_voronoi_mesh

    ck = np.load(argv[0], allow_pickle=True)
    mesh = create_voronoi_mesh(int(argv[1]))
    if len(argv) > 2 and float(ck["day"]) != float(argv[2]):
        raise ValueError(f"checkpoint is stamped day {float(ck['day']):g}, expected {argv[2]}")
    T, u, ps = np.asarray(ck["T"]), np.asarray(ck["u"]), np.asarray(ck["p_s"])
    if T.shape[0] != mesh.nCells or u.shape[0] != mesh.nEdges:
        raise ValueError(f"layout mismatch: T {T.shape}, u {u.shape}, mesh {mesh.nCells}/{mesh.nEdges}")
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(u))):
        raise ValueError("non-finite state")
    lat = np.degrees(np.asarray(mesh.latCell)); area = np.asarray(mesh.areaCell)
    ue = cell_u_east(u, mesh)
    tb, ub = band_means(T, lat, area), band_means(ue, lat, area)
    nlev = T.shape[1]
    print(f"day {float(ck['day']):.1f}  p_s area mean {np.sum(ps * area) / area.sum():.1f} Pa")
    print("level k (0 = top) | area-weighted band mean T [K] | eastward wind [m/s]")
    print("bands:", BANDS)
    for k in list(range(0, nlev, 4)) + [nlev - 1]:
        print(f"k={k:2d} T " + " ".join(f"{x:6.1f}" for x in tb[:, k])
              + " | u " + " ".join(f"{x:+6.2f}" for x in ub[:, k]))


if __name__ == "__main__":
    main(sys.argv[1:])

"""Compare the eORCA1 tripole card's 3-D ocean geometry with the NEMO run's.

NEMO ORCA1 runs ``domain_cfg.nc``: ``bottom_level`` wet levels per column and
3-D ``e3t_0`` thicknesses (``domzgr.F90:141-142``).  For each runner geometry
(``--nemo-geometry mesh_mask``: mesh-file column depth + thin-cell snap;
``--nemo-geometry domain_cfg``: NEMO's own cells) this probe builds the
coordinate with the runner's own functions and counts, over NEMO's inner
domain, the columns whose wet-level count differs, the largest bottom-cell
thickness error, and checks that the cyclic halo columns copy their partners.
It also checks the T ladder the domain_cfg geometry uses (``e3_to_depth_1d``
of ``e3w_1d``, ``depth_e3.F90:125-130``) against the mesh file's ``gdept_1d``
(ORCA1's ``key_vco_1d3d`` keeps W thickness and depths 1-D, so no 3-D ladder
is compared), and that z* layers still sum to H + eta.

Exits 1 when ``--expect-domain-cfg-exact`` is given and the domain_cfg
geometry differs from NEMO anywhere.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _read(path, name):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        a = np.asarray(ds[name][:], dtype=np.float64)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return a


def compare(pc, j0, i0, kb, e3t_0):
    """Wet-level count and bottom-thickness mismatch over the inner frame."""
    nj, ni = kb.shape
    ours = np.asarray(pc.bottom_level)[j0:j0 + nj, i0:i0 + ni] + 1   # land -> 0
    h = np.asarray(pc.h_partial)[j0:j0 + nj, i0:i0 + ni, :]
    diff = ours != kb
    jj, ii = np.nonzero((~diff) & (kb > 0))
    kk = kb[jj, ii].astype(int) - 1
    dh = np.abs(h[jj, ii, kk] - e3t_0[kk, jj, ii])
    halo = (np.array_equal(np.asarray(pc.bottom_level)[:, 0],
                           np.asarray(pc.bottom_level)[:, -2])
            and np.array_equal(np.asarray(pc.bottom_level)[:, -1],
                               np.asarray(pc.bottom_level)[:, 1]))
    return {"columns_differ": int(diff.sum()),
            "shallower": int((diff & (ours < kb)).sum()),
            "deeper": int((diff & (ours > kb)).sum()),
            "max_abs_bottom_dz_m": float(dh.max()) if dh.size else 0.0,
            "halo_copies_partner": halo}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", required=True)
    p.add_argument("--domain-cfg", required=True)
    p.add_argument("--min-levels", type=int, default=2)
    p.add_argument("--expect-domain-cfg-exact", action="store_true")
    a = p.parse_args(argv)

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.fidelity.nemo_dino_mesh import e3_to_depth_1d
    from legoesm.ocean.init_tripole import (
        nemo_domain_offset, read_domain_cfg_geometry, read_mesh_mask_bathy,
    )
    from legoesm.ocean.vertical import (
        create_full_step_coordinate, create_z_star_from_thicknesses,
    )
    from scripts.run.run_omip_core2 import _ew_overlap_fill, make_partial_cell

    e3t_1d = _read(a.domain_cfg, "e3t_1d")
    e3w_1d = _read(a.domain_cfg, "e3w_1d")
    e3t_0 = _read(a.domain_cfg, "e3t_0")
    kb = _read(a.domain_cfg, "bottom_level").astype(int)
    lat, lon = _read(a.mesh, "gphit"), _read(a.mesh, "glamt")
    j0, i0 = nemo_domain_offset(lat, lon, _read(a.domain_cfg, "gphit"),
                                _read(a.domain_cfg, "glamt"), a.domain_cfg)
    print(f"[geom] domain_cfg inner frame at mesh offset (j0, i0) = ({j0}, {i0})")
    gdept, _ = e3_to_depth_1d(e3t_1d, e3w_1d)
    print(f"[geom] T ladder e3_to_depth_1d(domain_cfg) vs mesh gdept_1d: "
          f"max|diff| {np.abs(gdept - _read(a.mesh, 'gdept_1d')).max():.3e} m")
    thick = np.zeros(kb.shape, dtype=bool)
    jj, ii = np.nonzero(kb > 0)
    thick[jj, ii] = e3t_0[kb[jj, ii] - 1, jj, ii] > e3t_1d[kb[jj, ii] - 1]
    print(f"[geom] NEMO: {int((kb > 0).sum())} wet columns, min wet levels "
          f"{int(kb[kb > 0].min())}, {int(thick.sum())} bottom cells thicker "
          "than their reference level")

    zc = create_z_star_from_thicknesses(e3t_1d, gdept)
    land, H = read_mesh_mask_bathy(a.mesh, nemo_domain_cfg=a.domain_cfg)
    land, H = _ew_overlap_fill(land), _ew_overlap_fill(H)
    pc, _, _ = make_partial_cell(zc, H, land, min_levels=a.min_levels)
    print(f"[geom] mesh_mask (card today): {compare(pc, j0, i0, kb, e3t_0)}")

    n_wet, bot, _ = read_domain_cfg_geometry(a.domain_cfg, lat, lon)
    n_wet, bot = _ew_overlap_fill(n_wet), _ew_overlap_fill(bot)
    if not np.array_equal(n_wet > 0, land > 0.5):
        print("[geom] FAIL: domain_cfg wet columns != masked mesh wet columns")
        return 1
    pcd = create_full_step_coordinate(zc, n_wet - 1, bottom_thickness=bot)
    r = compare(pcd, j0, i0, kb, e3t_0)
    print(f"[geom] domain_cfg: {r}")
    from legoesm.ocean.vertical import compute_layer_thickness
    Hd = np.asarray(pcd.h_partial).sum(axis=-1)
    eta = np.where(Hd > 0, 0.5, 0.0)
    h = np.asarray(compute_layer_thickness(eta, Hd, pcd))
    wet = Hd > 0
    print(f"[geom] z* closure, eta=0.5 m: max|sum h - (H+eta)| "
          f"{np.abs(h.sum(-1) - (Hd + eta))[wet].max():.3e} m")
    if a.expect_domain_cfg_exact and (r["columns_differ"]
                                      or r["max_abs_bottom_dz_m"] > 0.0
                                      or not r["halo_copies_partner"]):
        print("[geom] FAIL: domain_cfg geometry does not reproduce NEMO")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

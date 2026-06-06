"""Frame recorder for the spectral ABL-LES drivers.

Writes the SAME per-frame npz layout that ``scripts/plot/plot_les_diagnostics.py``
consumes: ``<out>/snapshots/snap_NNN.npz`` (horizontal cross-sections of w, θ, u,
v at the surface + four heights) and ``<out>/profiles/prof_NNN.npz`` (planar-mean
profiles + resolved second moments). Shared by the neutral/CBL/SBL drivers so the
slice + turbulence-statistic maths live in exactly one place.

The spectral core uses a UNIFORM vertical grid (``z`` ascending, index 0 = lowest
cell centre); u, v are cell-centred ``(ny, nx, nz)`` and w is supplied already
interpolated to cell centres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Heights (fraction of the domain depth Lz) for the four cross-sections above the
# surface — biased low so they span the boundary layer rather than the free
# atmosphere / sponge aloft.
_HEIGHT_FRACS = (0.05, 0.15, 0.30, 0.60)


def select_heights(z, Lz):
    """Indices + heights[m] for the surface (lowest cell) + four BL heights,
    ascending in z, de-duplicated."""
    z = np.asarray(z)
    surf = int(np.argmin(z))
    idx = [surf] + [int(np.argmin(np.abs(z - f * float(Lz)))) for f in _HEIGHT_FRACS]
    seen, out = set(), []
    for k in sorted(idx, key=lambda kk: z[kk]):
        if k not in seen:
            seen.add(k); out.append(k)
    out = np.array(out, dtype=int)
    return out, z[out]


def _profiles(z, u3, v3, wc3, theta3, z0, case):
    """Planar-mean profiles + resolved second moments (matches the layout of
    ``run_les_plane._resolved_profiles``)."""
    um = u3.mean((0, 1)); vm = v3.mean((0, 1)); wm = wc3.mean((0, 1))
    up, vp, wp = u3 - um, v3 - vm, wc3 - wm
    uw = (up * wp).mean((0, 1)); vw = (vp * wp).mean((0, 1))
    uu = (up * up).mean((0, 1)); vv = (vp * vp).mean((0, 1)); ww = (wp * wp).mean((0, 1))
    tke = 0.5 * (uu + vv + ww)
    theta = theta3.mean((0, 1))
    spd = np.sqrt(um ** 2 + vm ** 2)
    u_star = float((uw[0] ** 2 + vw[0] ** 2) ** 0.25)
    return dict(z=z, theta=theta, u=um, v=vm, spd=spd, wvar=ww,
                uu=uu, vv=vv, ww=ww, tke=tke, uw=uw, vw=vw,
                u_star=u_star, z0=z0, case=case)


def record_frame(out_dir, frame, t_hours, case, z, u3, v3, wc3, theta3,
                 Lx, Ly, h_idx, h_z, z0):
    """Save one snapshot npz (height cross-sections) + one profile npz.

    ``u3, v3, wc3, theta3`` are host (numpy) arrays of shape (ny, nx, nz);
    ``wc3`` is the cell-centred vertical velocity.
    """
    out_dir = Path(out_dir)
    snap_dir = out_dir / "snapshots"; snap_dir.mkdir(parents=True, exist_ok=True)
    prof_dir = out_dir / "profiles"; prof_dir.mkdir(parents=True, exist_ok=True)
    nx = u3.shape[1]
    np.savez(
        snap_dir / f"snap_{frame:03d}.npz",
        t_hours=t_hours, case=case, heights=h_z, dx=Lx / nx, Lx=Lx, Ly=Ly,
        w=np.stack([wc3[:, :, k] for k in h_idx]),
        theta=np.stack([theta3[:, :, k] for k in h_idx]),
        u=np.stack([u3[:, :, k] for k in h_idx]),
        v=np.stack([v3[:, :, k] for k in h_idx]),
    )
    prof = _profiles(np.asarray(z), u3, v3, wc3, theta3, z0, case)
    np.savez(prof_dir / f"prof_{frame:03d}.npz", t_hours=t_hours, **prof)

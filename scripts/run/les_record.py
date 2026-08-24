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


def _profiles(z, u3, v3, wc3, theta3, z0, case, scalars=None, qt3=None):
    """Planar-mean profiles + resolved second moments (matches the layout of
    ``run_les_plane._resolved_profiles``).

    Flux sign convention: z is positive UP, so every ``⟨w'x'⟩`` here is positive
    for UPWARD transport of x (same convention as the ``uw``/``vw`` momentum
    fluxes below and as ``run_spectral_cbl.diagnose``). ``wth`` is the RESOLVED
    kinematic heat flux [K m/s]; the SGS part is not included (for a moist run the
    caller passes θ_l as ``theta3``, so ``wth`` is then ⟨w'θ_l'⟩).
    """
    um = u3.mean((0, 1)); vm = v3.mean((0, 1)); wm = wc3.mean((0, 1))
    up, vp, wp = u3 - um, v3 - vm, wc3 - wm
    uw = (up * wp).mean((0, 1)); vw = (vp * wp).mean((0, 1))
    uu = (up * up).mean((0, 1)); vv = (vp * vp).mean((0, 1)); ww = (wp * wp).mean((0, 1))
    tke = 0.5 * (uu + vv + ww)
    theta = theta3.mean((0, 1))
    wth = (wp * (theta3 - theta)).mean((0, 1))   # resolved ⟨w'θ'⟩ [K m/s]
    spd = np.sqrt(um ** 2 + vm ** 2)
    u_star = float((uw[0] ** 2 + vw[0] ** 2) ** 0.25)
    out = dict(z=z, theta=theta, u=um, v=vm, spd=spd, wvar=ww,
               uu=uu, vv=vv, ww=ww, tke=tke, uw=uw, vw=vw, wth=wth,
               u_star=u_star, z0=z0, case=case)
    # The reference-artifact reader asks for the heat flux by its long name.
    # It is the same array as ``wth``; both are written so neither consumer
    # has to know the other's spelling.
    out["wtheta"] = wth
    # TOTAL water and its resolved flux, when the caller has it. A moist
    # artifact is scored on total water, not on vapour alone, and summing the
    # species after the fact cannot recover the flux: the covariance of a sum
    # is only the sum of covariances because the planar mean is linear, so it
    # is computed here, from the fields, in the one place the perturbation
    # maths lives.
    if qt3 is not None:
        qt_mean = qt3.mean((0, 1))
        out["qt"] = qt_mean
        out["wqt"] = (wp * (qt3 - qt_mean)).mean((0, 1))
    # Optional extra scalars (moist runs): planar mean + resolved kinematic
    # flux ⟨w'x'⟩, same positive-up convention as wth. Kept here so the
    # perturbation maths lives in exactly one place.
    for name, arr in (scalars or {}).items():
        mean = arr.mean((0, 1))
        out[name] = mean
        out[f"w{name}"] = (wp * (arr - mean)).mean((0, 1))
    return out


def record_frame(out_dir, frame, t_hours, case, z, u3, v3, wc3, theta3,
                 Lx, Ly, h_idx, h_z, z0, qc3=None, rho_z=None, qr3=None,
                 surface_precip=None, qv3=None):
    """Save one snapshot npz (height cross-sections) + one profile npz.

    ``u3, v3, wc3, theta3`` are host (numpy) arrays of shape (ny, nx, nz);
    ``wc3`` is the cell-centred vertical velocity. MOIST runs pass ``qc3``
    (cloud-water mixing ratio, same shape) and ``rho_z`` ((nz,) reference
    density): the snapshot then also stores the q_c cross-sections + the
    liquid-water-path map [g/m²], and the profile gains q_c/cloud-fraction.
    Lagrangian SDM runs may also pass ``qr3`` and cumulative
    ``surface_precip`` [kg/m²]. Moist runs SHOULD also pass ``qv3`` (water
    vapour mixing ratio): the profile then gains ``qv`` and the resolved
    moisture flux ``wqv`` [kg/kg m/s], which is what an SCM turbulence closure
    is tuned against.
    """
    out_dir = Path(out_dir)
    snap_dir = out_dir / "snapshots"; snap_dir.mkdir(parents=True, exist_ok=True)
    prof_dir = out_dir / "profiles"; prof_dir.mkdir(parents=True, exist_ok=True)
    nx = u3.shape[1]
    extra, prof_extra, scalars = {}, {}, {}
    if qv3 is not None:
        scalars["qv"] = qv3
    if qc3 is not None:
        extra["qc"] = np.stack([qc3[:, :, k] for k in h_idx])
        if rho_z is not None:
            dz = float(abs(z[1] - z[0]))     # spectral grid: uniform dz
            extra["lwp"] = (qc3 * rho_z[None, None, :]).sum(-1) * dz * 1e3
        scalars["qc"] = qc3
        prof_extra["cloud_frac"] = (qc3 > 1.0e-5).mean(axis=(0, 1))
    if qr3 is not None:
        extra["qr"] = np.stack([qr3[:, :, k] for k in h_idx])
        scalars["qr"] = qr3
    if surface_precip is not None:
        extra["surface_precip"] = np.asarray(surface_precip)
        prof_extra["surface_precip_mean"] = float(np.asarray(surface_precip).mean())
    np.savez(
        snap_dir / f"snap_{frame:03d}.npz",
        t_hours=t_hours, case=case, heights=h_z, dx=Lx / nx, Lx=Lx, Ly=Ly,
        w=np.stack([wc3[:, :, k] for k in h_idx]),
        theta=np.stack([theta3[:, :, k] for k in h_idx]),
        u=np.stack([u3[:, :, k] for k in h_idx]),
        v=np.stack([v3[:, :, k] for k in h_idx]),
        **extra,
    )
    # Total water is the sum over the species this run carries: vapour plus
    # whatever condensate is present. Built here, from the 3-D fields, so its
    # resolved flux is a real covariance rather than a sum of stored fluxes.
    qt3 = None
    for _species in (qv3, qc3, qr3):
        if _species is not None:
            qt3 = _species if qt3 is None else qt3 + _species
    prof = _profiles(np.asarray(z), u3, v3, wc3, theta3, z0, case,
                     scalars=scalars, qt3=qt3)
    np.savez(prof_dir / f"prof_{frame:03d}.npz", t_hours=t_hours,
             **prof, **prof_extra)

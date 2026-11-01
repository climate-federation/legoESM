"""Does a rebuilt subgrid (SGS) flux close CloudBench's stored scalar tendency?

CloudBench's stored *_diffusive_flux_z are never written (zeros), so the SGS heat
and moisture flux must be rebuilt offline (PI 2026-10-31) with swirl-lm's own
Smagorinsky-Lilly closure as configured for CloudBench (c_s 0.18, Pr_t 0.3,
molecular 0.1 m2/s, buoyancy factor, Delta = (dx dy dz)^(1/3), centred node
gradients, face flux 0.5(rhoK[k]+rhoK[k-1])(phi[k]-phi[k-1])/dz; swirl-lm commit
98f96f80, physics/turbulence/sgs_model.py, numerics/diffusion.py,
physics/thermodynamics/water.py).

The only stored check is ``<X>_les_tendency``, whose definition is not in the public
code. For one case and two output times this prints, per height band, the
horizontal-mean stored tendency and residuals against the resolved vertical flux
divergence alone and with the rebuilt SGS flux, each under two readings of the
stored tendency (per unit mass, or multiplied by rho). A third pair uses the LES's own vertical advection flux (QUICK, upwinded) in
place of the node-centred resolved divergence. Numbers only; no verdict.

Streams the needed fields from the public bucket into memory; nothing is written.

usage: cloudbench_sgs_budget_probe.py SITE MONTH EXP TIME_INDEX [TIME_INDEX ...]
"""
from __future__ import annotations

import sys

import numpy as np
import xarray as xr

URL = "https://storage.googleapis.com/cloudbench-simulation-output/{}/{}/{}/data.zarr"
# swirl-lm CloudBench configuration (values reproduced from that code, not ours)
C_S, PR_T, D_MOL = 0.18, 0.3, 0.1
G_SWIRL, R_D_SWIRL, R_V_SWIRL = 9.81, 286.69, 461.89
P00 = 1.0e5
CP_D_SWIRL, CP_V_SWIRL = 728.31 + 286.69, 1859.0
L_DOMAIN = 6000.0
VARS = ("u", "v", "w", "theta_li", "q_t", "rho", "T", "q_c", "p_ref",
        "theta_li_les_tendency", "q_t_les_tendency")


def ddx(f, d, axis):
    return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2.0 * d)


def ddz(f, dz):
    out = np.full_like(f, np.nan)
    out[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2.0 * dz)
    return out


def quick_face_flux(m, phi):
    """swirl-lm's vertical scalar flux at face k-1/2 (index k), as configured for
    CloudBench: QUICK face values from the upwind side, upwinded by the face mass
    flux 0.5(m[k-1]+m[k]) (Rhie-Chow off; numerics/convection.py flux_upwinding,
    face_interp_fn_quick; utility/get_kernel_fn.py kf2). NaN where the stencil
    leaves the grid (k < 2 or k > nz-2)."""
    nz = phi.shape[-1]
    f = np.full(phi.shape, np.nan)
    k = np.arange(2, nz - 1)
    m_f = 0.5 * (m[..., k - 1] + m[..., k])
    s_pos = -0.125 * phi[..., k - 2] + 0.75 * phi[..., k - 1] + 0.375 * phi[..., k]
    s_neg = 0.375 * phi[..., k - 1] + 0.75 * phi[..., k] - 0.125 * phi[..., k + 1]
    f[..., k] = np.maximum(m_f, 0.0) * s_pos + np.minimum(m_f, 0.0) * s_neg
    return f


def rebuild_rhok(s, dx, dy, dz):
    u, v, w = s["u"], s["v"], s["w"]
    g = [[ddx(c, dx, 0), ddx(c, dy, 1), ddz(c, dz)] for c in (u, v, w)]
    div = g[0][0] + g[1][1] + g[2][2]
    s2 = 0.0
    for i in range(3):
        for j in range(3):
            sij = 0.5 * (g[i][j] + g[j][i]) - (div / 3.0 if i == j else 0.0)
            s2 = s2 + sij ** 2
    smag = np.sqrt(2.0 * s2)
    eps = R_V_SWIRL / R_D_SWIRL
    q_t, q_c = s["q_t"], s["q_c"]
    r_m = R_D_SWIRL * (1.0 + (eps - 1.0) * q_t - eps * q_c)
    cp_m = (1.0 - q_t) * CP_D_SWIRL + (q_t - q_c) * CP_V_SWIRL
    theta = s["T"] * (s["p_ref"] / P00) ** (-r_m / cp_m)
    theta_v = (r_m / R_D_SWIRL) * theta
    n2 = G_SWIRL / theta_v * ddz(theta_v, dz)
    ri = np.where(smag > 0.0, n2 / np.where(smag > 0.0, smag, 1.0) ** 2, 0.0)
    fb = np.where(ri <= 0.0, 1.0, np.maximum(0.0, 1.0 - ri / PR_T) ** 0.25)
    delta = (dx * dy * dz) ** (1.0 / 3.0)
    k = (C_S ** 2 / PR_T) * (delta * fb) ** 2 * smag + D_MOL
    return s["rho"] * k


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    site, month, exp, tidx = a[0], a[1], a[2], [int(x) for x in a[3:]]
    ds = xr.open_zarr(URL.format(site, month, exp), consolidated=True)
    nx, ny, nz = ds.sizes["x"], ds.sizes["y"], ds.sizes["z"]
    dx, dy, dz = L_DOMAIN / (nx - 1), L_DOMAIN / (ny - 1), L_DOMAIN / (nz - 1)
    z = np.asarray(ds["z"].values, dtype=np.float64)
    print(f"case {site}/{month}/{exp}: grid {nx}x{ny}x{nz} dx {dx:.3f} dz {dz:.3f} "
          f"z[0..2] {z[:3]} t hours {ds['t'].values[tidx].astype('timedelta64[s]').astype(float) / 3600}")
    bands = [(2, 80), (80, 160), (160, 240), (240, 400)]   # node index bands
    for ti in tidx:
        s = {n: np.asarray(ds[n].isel(t=ti).values, dtype=np.float64) for n in VARS}
        bad = [n for n, f in s.items() if not np.all(np.isfinite(f))]
        if bad:
            raise SystemExit(f"t index {ti}: non-finite {bad}")
        rhok = rebuild_rhok(s, dx, dy, dz)
        rho_m = s["rho"].mean(axis=(0, 1))
        for name in ("theta_li", "q_t"):
            phi = s[name]
            tend = s[name + "_les_tendency"].mean(axis=(0, 1))
            res_flux = (s["rho"] * s["w"] * phi).mean(axis=(0, 1))       # nodes
            d_res = ddz(res_flux, dz)                                   # d<rho w phi>/dz
            f_face = 0.5 * (rhok[..., 1:] + rhok[..., :-1]) * np.diff(phi, axis=-1) / dz
            f_m = f_face.mean(axis=(0, 1))                              # faces k-1/2, k=1..
            d_sgs = np.full(nz, np.nan)
            d_sgs[1:-1] = (f_m[1:] - f_m[:-1]) / dz                     # +dF/dz at nodes
            q_m = quick_face_flux(s["rho"] * s["w"], phi).mean(axis=(0, 1))  # face k-1/2
            d_q = np.full(nz, np.nan)
            d_q[:-1] = (q_m[1:] - q_m[:-1]) / dz                        # node k
            cands = {
                "res, per mass": tend - (-d_res) / rho_m,
                "res+sgs, per mass": tend - (-d_res + d_sgs) / rho_m,
                "res, rho-weighted": tend - (-d_res),
                "res+sgs, rho-weighted": tend - (-d_res + d_sgs),
                "QUICK res, rho-weighted": tend - (-d_q),
                "QUICK res+sgs, rho-weighted": tend - (-d_q + d_sgs),
            }
            sgs_kin = -f_m / (0.5 * (rho_m[1:] + rho_m[:-1]))           # upward flux
            print(f"\n t index {ti} {name}: rebuilt SGS upward flux at faces "
                  f"z={0.5 * (z[2] + z[3]):.1f},{0.5 * (z[10] + z[11]):.1f},"
                  f"{0.5 * (z[100] + z[101]):.1f} m: {sgs_kin[2]:.3e} {sgs_kin[10]:.3e} "
                  f"{sgs_kin[100]:.3e}; resolved <w phi'> at those nodes "
                  f"{((s['w'] * (phi - phi.mean(axis=(0, 1)))).mean(axis=(0, 1)))[[2, 10, 100]]}")
            for lo, hi in bands:
                rms_t = np.sqrt(np.mean(tend[lo:hi] ** 2))
                line = f"  z {z[lo]:6.0f}-{z[hi]:6.0f} m  rms<tend> {rms_t:.3e}"
                for lab, r in cands.items():
                    line += f" | {lab}: {np.sqrt(np.mean(r[lo:hi] ** 2)):.3e}"
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

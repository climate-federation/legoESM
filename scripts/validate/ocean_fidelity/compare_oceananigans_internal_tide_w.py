"""Single-step w / continuity match over the internal_tide partial-cell staircase
(issue #576). Prescribes the SAME state (u=U2, v=V0*exp(-x^2/2W^2), b=N2 z) on the
same bump and compares legoESM's diagnosed w vs Oceananigans' w (continuity) — to
localize the grid-scale (2dx) w-noise at the staircase. Faithful flux_form.

    LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref JAX_ENABLE_X64=1 \
      .venv/bin/python scripts/validate/ocean_fidelity/compare_oceananigans_internal_tide_w.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MOM_ADV", "flux_form")
import numpy as np
import jax.numpy as jnp
from netCDF4 import Dataset
from legoesm.ocean.vertical import compute_centroid_depth
sys.path.insert(0, os.path.dirname(__file__))
import compare_oceananigans_internal_tide as C

V0 = 0.1


def set_prescribed(grid, z, state):
    z_full = np.asarray(z.z_full_ref)
    mask = np.asarray(state.land_mask.data)
    n_lat, n_lon = mask.shape
    nlev = len(z_full)
    xc = -C.LX / 2 + (np.arange(n_lon) + 0.5) * C.DX_M
    # b = N^2 * z at the PHYSICAL cell-center depth (matches Oceananigans, which
    # sets b=N^2 z at its partial-cell centers) — NOT z_full_ref, which would put
    # an inconsistent b at the partial bottom cells and fake a ∂b/∂x at the steps.
    H_bathy = jnp.asarray(np.asarray(state.H_bathy.data))
    z_phys = -np.asarray(compute_centroid_depth(jnp.zeros_like(H_bathy), H_bathy, z))
    T = (C.T_REF_C + (C.N2 * z_phys) / (C.G * C.ALPHA_T)) * mask[:, :, None]
    uface = np.minimum(mask, np.roll(mask, 1, axis=1))
    uface = np.concatenate([uface, uface[:, :1]], axis=1)
    u = np.full((n_lat, n_lon + 1, nlev), C.U2) * uface[:, :, None]
    v_x = V0 * np.exp(-xc ** 2 / (2.0 * C.WIDTH ** 2))
    v = np.zeros((n_lat + 1, n_lon, nlev))
    v[1:-1, :, :] = v_x[None, :, None]
    return state._replace(
        T=state.T.replace(data=jnp.asarray(T)),
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)))


def main():
    dt = 150.0
    grid, wall, z, state, model = C.build_setup()
    state = set_prescribed(grid, z, state)
    T0 = np.asarray(state.T.data)
    s1 = model.step(state, dt, surface_forcing=None)      # populates state.w + advances T
    jrow = C.NY // 2
    w_lego = np.asarray(s1.w.data)[jrow]                  # (n_lon, nlev_w)
    # buoyancy tendency db/dt = G*alpha*(T_after - T_before)/dt (advection; K_v small)
    dbdt_lego = C.G * C.ALPHA_T * (np.asarray(s1.T.data) - T0)[jrow] / dt   # (n_lon, nlev)
    z_full = np.asarray(z.z_full_ref)
    Hb = np.asarray(state.H_bathy.data)[jrow]

    ds = Dataset(os.path.join(os.environ["LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"],
                              "internal_tide", "internal_tide_w.nc"))
    o_x = np.asarray(ds.variables["x"][:])
    o_zf = np.asarray(ds.variables["zf"][:])              # (Nz+1,) z-faces
    o_w = np.asarray(ds.variables["w"][:]).T              # (zf,x) -> (x, Nz+1) col-major reversal
    nx = len(o_x)
    print(f"[it-w] lego w shape {w_lego.shape}  oracle w shape {o_w.shape}  "
          f"o_zf[{len(o_zf)}] z_full[{len(z_full)}]", flush=True)
    print(f"  max|w| lego={np.abs(w_lego).max():.3e}  oracle={np.abs(o_w).max():.3e}", flush=True)

    # 2dx roughness in x at each code, restricted to the bump region.
    bump = np.abs(o_x) < 1.5e5
    def roughness(w2d):
        wb = w2d[bump]
        d2 = np.abs(wb[2:] - 2 * wb[1:-1] + wb[:-2])
        return d2.max(), np.abs(wb).max()
    rl, ml = roughness(w_lego[:nx])
    ro, mo = roughness(o_w[:nx])
    print(f"  2dx-roughness(bump): lego max={rl:.3e} (max|w|={ml:.3e})  "
          f"oracle max={ro:.3e} (max|w|={mo:.3e})", flush=True)
    print(f"  -> lego rough/|w| = {rl/ml:.3f}   oracle rough/|w| = {ro/mo:.3f}", flush=True)
    # direct w PATTERN match (oracle face-w averaged to cell centers vs lego state.w)
    o_w_cc = 0.5 * (o_w[:nx, :-1] + o_w[:nx, 1:])         # (nx, Nz) cell-centers
    wl = w_lego[:nx]
    nzc = min(o_w_cc.shape[1], wl.shape[1])
    wetw = (z_full[None, :nzc] > -Hb[:nx, None])
    a = wl[:, :nzc][wetw]; b_ = o_w_cc[:, :nzc][wetw]
    ca = a - a.mean(); cb = b_ - b_.mean()
    wc = float(np.sum(ca * cb) / (np.sqrt(np.sum(ca**2)*np.sum(cb**2)) + 1e-30))
    print(f"  w pattern_corr(wet, cell-center) = {wc:+.4f}", flush=True)

    # --- buoyancy-tendency (tracer advection) comparison ---
    o_Gb = np.asarray(ds.variables["Gb"][:]).T            # (x, z) col-major reversal
    o_zc = np.asarray(ds.variables["z"][:])
    # interp oracle Gb onto legoESM z_full per column
    o_Gb_on = np.array([np.interp(-z_full, -o_zc, o_Gb[ix, :]) for ix in range(nx)])
    lego_Gb = dbdt_lego[:nx, :]
    wet = (z_full[None, :] > -Hb[:nx, None])
    m = wet
    print(f"\n  db/dt (tracer advection): max|lego|={np.abs(lego_Gb[m]).max():.3e}  "
          f"max|oracle|={np.abs(o_Gb_on[m]).max():.3e}", flush=True)
    ca = lego_Gb[m] - lego_Gb[m].mean(); cb = o_Gb_on[m] - o_Gb_on[m].mean()
    corr = float(np.sum(ca * cb) / (np.sqrt(np.sum(ca ** 2) * np.sum(cb ** 2)) + 1e-30))
    print(f"  db/dt pattern_corr(wet) = {corr:+.4f}", flush=True)
    d = np.where(wet, np.abs(lego_Gb - o_Gb_on), 0.0)
    xi, ki = np.unravel_index(np.argmax(d), d.shape)
    print(f"  worst db/dt diff at x={o_x[xi]/1e3:.0f}km z={z_full[ki]:.0f}m "
          f"(lego={lego_Gb[xi,ki]:.3e} oracle={o_Gb_on[xi,ki]:.3e}) Hb={Hb[xi]:.0f}m", flush=True)
    # 2dx roughness of db/dt over the bump
    def rough2(f2):
        fb = f2[bump]; d2 = np.abs(fb[2:] - 2 * fb[1:-1] + fb[:-2]); return d2.max(), np.abs(fb).max()
    rgl, mgl = rough2(lego_Gb); rgo, mgo = rough2(o_Gb_on)
    print(f"  db/dt 2dx-rough/|.|: lego={rgl/ (mgl+1e-30):.3f}  oracle={rgo/(mgo+1e-30):.3f}", flush=True)


if __name__ == "__main__":
    main()

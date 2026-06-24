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
sys.path.insert(0, os.path.dirname(__file__))
import compare_oceananigans_internal_tide as C

V0 = 0.1


def set_prescribed(grid, z, state):
    z_full = np.asarray(z.z_full_ref)
    mask = np.asarray(state.land_mask.data)
    n_lat, n_lon = mask.shape
    nlev = len(z_full)
    xc = -C.LX / 2 + (np.arange(n_lon) + 0.5) * C.DX_M
    T = (C.T_REF_C + (C.N2 * z_full[None, None, :]) / (C.G * C.ALPHA_T)) * mask[:, :, None]
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
    grid, wall, z, state, model = C.build_setup()
    state = set_prescribed(grid, z, state)
    s1 = model.step(state, 150.0, surface_forcing=None)   # populates state.w from the diagnosis
    jrow = C.NY // 2
    w_lego = np.asarray(s1.w.data)[jrow]                  # (n_lon, nlev_w)
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


if __name__ == "__main__":
    main()

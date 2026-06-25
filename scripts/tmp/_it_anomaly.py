"""Internal-tide comparison on the TIME-VARYING (tidal) anomaly b'(t)-b'(0),
which removes the STATIC partial-cell coordinate offset that dominates the raw
b' field in BOTH codes (oracle static b'(0)=0.198 at the bottom; legoESM has its
own static offset at z_full_ref). The internal tide IS the dynamically generated
perturbation from the rest representation — subtracting b'(0) isolates it.

  LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref JAX_ENABLE_X64=1 \
    .venv/bin/python scripts/tmp/_it_anomaly.py [stop_days]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MOM_ADV", "flux_form")
import numpy as np
import jax
import jax.numpy as jnp
from netCDF4 import Dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validate", "ocean_fidelity"))
import compare_oceananigans_internal_tide as C


def main():
    stop_days = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    dt = float(os.environ.get("DT", "300.0"))
    grid, wall, z, state, model = C.build_setup()
    state = C.set_ic(grid, z, state)
    jrow = C.NY // 2
    z_full = np.asarray(z.z_full_ref)
    Hb = np.asarray(state.H_bathy.data)[jrow]
    wet = (z_full[None, :] > -Hb[:, None])

    ds = Dataset(os.path.join(os.environ["LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"],
                              "internal_tide", "internal_tide_zlevel.nc"))
    o_t = np.asarray(ds.variables["times_s"][:]); o_days = o_t / 86400.0
    o_bp = np.asarray(ds.variables["bprime"][:])   # (z,x,t)
    zc = np.asarray(ds.variables["z"][:])
    o_bp0 = o_bp[:, :, 0].T                        # (x,z) static rest field

    umask = jnp.asarray((np.asarray(state.u.data) != 0.0).astype(np.asarray(state.u.data).dtype))

    @jax.jit
    def step_and_force(s, tt):
        s = model.step(s, dt, surface_forcing=None)
        f = C.A2 * jnp.sin(C.OMEGA2 * tt)
        return s._replace(u=s.u.replace(data=(s.u.data + f * dt) * umask))

    def lego_bp(st):
        T = np.asarray(st.T.data)[jrow]
        return C.G * C.ALPHA_T * (T - C.T_REF_C) - C.N2 * z_full[None, :]

    bp0_lego = lego_bp(state)                       # legoESM static rest field
    nsteps = int(round(stop_days * 86400.0 / dt))
    t = 0.0
    print(" day | lego max|Δb'| | oracle max|Δb'| | corr(Δb', wet) | raw-corr", flush=True)
    for it in range(1, nsteps + 1):
        state = step_and_force(state, t); t += dt
        td = t / 86400.0
        if any(abs(td - od) < dt / 86400.0 / 2 for od in o_days):
            oi = int(np.argmin([abs(td - od) for od in o_days]))
            bp = lego_bp(state)
            dlego = bp - bp0_lego                    # legoESM tidal anomaly
            o_bi = o_bp[:, :, oi].T                  # (x,z)
            o_bi0 = o_bp0
            nxc = min(bp.shape[0], o_bi.shape[0])
            # interp oracle onto z_full per column
            def on(field):
                return np.array([np.interp(-z_full, -zc, field[ix, :]) for ix in range(nxc)])
            d_or = on(o_bi) - on(o_bi0)
            m = wet[:nxc]
            corr = C.amp2dx(dlego[:nxc][m], d_or[m])
            raw = C.amp2dx(bp[:nxc][m], on(o_bi)[m])
            print(f" {td:4.1f} | {np.abs(dlego[wet]).max():.3e}   | "
                  f"{np.abs(d_or[m]).max():.3e}    | {corr:+.4f}        | {raw:+.4f}",
                  flush=True)


if __name__ == "__main__":
    main()

"""2Δx grid-mode DECAY-RATE gate: legoESM vs Oceananigans (the §5 residual, localized).

The §5 blow-up is a 2Δx-in-lon, v-dominant ROTATIONAL grid mode that legoESM
UNDER-dissipates vs Oceananigans. The single-step per-node operators already match
(compare_oceananigans_tendency.py: vorticity flux corr 0.997-0.9999), so the residual is
the INTEGRATED dissipation of a DEVELOPED grid mode — invisible to a single-step gate.

This gate makes it LOCALIZABLE + FAST + DETERMINISTIC: both codes seed the bickley jet
with the IDENTICAL small 2Δx-in-lon v perturbation (a continuous sin-of-Nyquist that is
(-1)^i at the shared lon centres) and integrate a SHORT fixed-dt window; we compare the
DECAY RATE of that mode (amp(t)/amp(0)). legoESM must match the oracle's decay; a slower
decay (ratio_lego/ratio_oracle > 1) is the under-dissipation, in a single fast number the
loop can drive to 1.

Reference: scripts/data/generate_oceananigans_gridmode_decay_reference.jl.

Run::

    LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref \
      JAX_ENABLE_X64=1 .venv/bin/python \
      scripts/validate/ocean_fidelity/compare_oceananigans_gridmode_decay.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from netCDF4 import Dataset

_spec = importlib.util.spec_from_file_location(
    "bk", os.path.join(os.path.dirname(__file__), "compare_oceananigans_bickley_jet.py"))
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)

AMP = float(os.environ.get("SEED_AMP", "0.01"))
M_NYQ = bk.NH // 2


def seed_gridmode(grid, state):
    """Add the 2Δx-in-lon v seed at the v-points (lon-centre, lat-face). Injected in
    INDEX space as A·(-1)^j·env(φ) — the same pure 2Δx-in-lon checkerboard the oracle's
    continuous sin(Nyquist) produces at its lon centres. Robust to legoESM's lon-grid
    convention; the lon index aligns with the oracle (bridge lon-roll 0/-1), and the
    decay RATE of a 2Δx mode is phase-invariant, so a half-cell offset doesn't matter."""
    lat_c = np.degrees(np.asarray(grid.lat))
    dlat = lat_c[1] - lat_c[0]
    lat_f = lat_c - dlat / 2.0
    v = np.asarray(state.v.data)                              # (n_lat+1, n_lon, 1)
    nlatp1, nlon = v.shape[0], v.shape[1]
    lat_f_ext = np.r_[lat_f, lat_f[-1:] + dlat][:nlatp1]
    checker = (-1.0) ** np.arange(nlon)                       # (-1)^j 2Δx-in-lon
    env = np.exp(-(lat_f_ext / 20.0) ** 2)                    # jet-core envelope in φ
    seed = AMP * env[:, None] * checker[None, :]              # (n_lat+1, n_lon)
    v_seed = v[:, :, 0] + seed
    return state._replace(v=state.v.replace(data=jnp.asarray(v_seed[:, :, None])))


def amp2dx(v2d):
    """2Δx-in-lon amplitude: RMS over lat of |DFT_lon(v)[Nyquist]|/nlon. Matches the
    oracle deck's amp2dx (normalization cancels in the decay RATIO we compare)."""
    v2d = np.asarray(v2d, dtype=float)
    nlon = v2d.shape[1]
    comp = np.abs(np.fft.fft(v2d, axis=1)[:, M_NYQ]) / nlon   # per-lat Nyquist magnitude
    return float(np.sqrt(np.mean(comp ** 2)))


def main():
    ref_root = os.environ["LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"]
    # GMD_REFDIR selects the amplitude-matched reference subdir (the oracle deck is
    # weakly amplitude-dependent, so the gate must compare like-for-like).
    refdir = os.environ.get("GMD_REFDIR", "gridmode_decay")
    ds = Dataset(os.path.join(ref_root, refdir, "gridmode_decay.nc"))
    o_amp = np.asarray(ds.variables["amp2dx"][:])
    dt = float(ds.getncattr("dt"))
    nsteps = len(o_amp) - 1
    o_ratio = o_amp / o_amp[0]
    print(f"[oracle] Nsteps={nsteps} dt={dt} amp0={o_amp[0]:.4e} "
          f"ampN={o_amp[-1]:.4e} decay_ratio={o_ratio[-1]:.4f}", flush=True)

    grid, wall, z, state, model = bk.build_bickley()
    state = bk.set_bickley_ic(grid, state)
    state = seed_gridmode(grid, state)

    # Interior v rows (drop the wall rows) so the lat extent best matches the oracle;
    # the decay RATIO is normalization-invariant regardless.
    def lego_amp(s):
        v = np.asarray(s.v.data)[:, :, 0]
        return amp2dx(v[1:-1] if v.shape[0] > grid.lat.shape[0] else v)

    step = jax.jit(lambda s: model.step(s, dt, surface_forcing=None))
    l_amp = [lego_amp(state)]
    for _ in range(nsteps):
        state = step(state)
        l_amp.append(lego_amp(state))
    l_amp = np.array(l_amp)
    l_ratio = l_amp / l_amp[0]
    finite = bool(np.all(np.isfinite(l_amp)))
    print(f"[lego]   amp0={l_amp[0]:.4e} ampN={l_amp[-1]:.4e} "
          f"decay_ratio={l_ratio[-1]:.4f} finite={finite}", flush=True)

    print("\n  step | lego amp/amp0 | oracle amp/amp0")
    for n in range(0, nsteps + 1, max(1, nsteps // 8)):
        print(f"  {n:4d} | {l_ratio[n]:.4f}        | {o_ratio[n]:.4f}")

    gate = l_ratio[-1] / o_ratio[-1] if o_ratio[-1] > 1e-30 else float("nan")
    print(f"\n  GATE (decay_ratio_lego / decay_ratio_oracle) = {gate:.3f}")
    print("  ==1.0 ⇒ legoESM dissipates the 2Δx mode at the oracle's rate (TARGET).")
    print("  >1.0  ⇒ legoESM UNDER-dissipates (mode decays slower / grows) = §5 residual.")
    print("  <1.0  ⇒ legoESM OVER-dissipates.")
    return gate, finite


if __name__ == "__main__":
    main()

"""The MITgcm-faithful gyre stepper reproduces the laminar 0.031 equilibrium.

This is the integrator the gyre oracle uses to match MITgcm's
``tutorial_barotropic_gyre`` (the canonical split model runs the marginally-
resolved Munk gyre turbulent; see mitgcm_gyre_faithful.py docstring).
"""

from __future__ import annotations

import numpy as np
from legoesm.ocean.fidelity.mitgcm_gyre_faithful import GyreFaithfulModel


def _spin_up(model, nsteps):
    s = model.rest_state()
    for _ in range(nsteps):
        s = model.step(s)
    return s


def test_spins_up_a_dipole_at_the_right_scale():
    """10 steps from rest: a free-surface dipole at the O(1e-2 m/s) gyre scale."""
    m = GyreFaithfulModel(n=62, dx_m=20.0e3, dt_s=1200.0)
    s = _spin_up(m, 10)
    assert np.all(np.isfinite(s.u)) and np.all(np.isfinite(s.eta))
    assert s.eta.min() < 0.0 < s.eta.max()           # wind-curl tilts a dipole
    assert 1e-5 < np.abs(s.u).max() < 1e-1


def test_reproduces_mitgcm_laminar_equilibrium():
    """Spin-up stays LAMINAR and equilibrates at MITgcm's |u|max≈0.031 / |v|max≈0.084
    (measured on-box), NOT the canonical split model's turbulent 0.15–0.37."""
    m = GyreFaithfulModel(n=62, dx_m=20.0e3, dt_s=1200.0)
    s = _spin_up(m, 15000)                            # ~0.57 yr (MITgcm plateaus by 0.4 yr)
    umax = float(np.abs(s.u).max()); vmax = float(np.abs(s.v).max())
    # within 20% of MITgcm's laminar equilibrium, and well below the turbulent band
    assert 0.025 < umax < 0.040, f"|u|max={umax:.4f} not at MITgcm's laminar 0.031"
    assert 0.065 < vmax < 0.105, f"|v|max={vmax:.4f} not at MITgcm's laminar 0.084"


def test_steady_not_growing():
    """Laminar = bounded/steady: |u|max barely changes between 0.5 and 0.95 yr
    (a turbulent attractor would still be growing/oscillating an order larger)."""
    m = GyreFaithfulModel(n=62, dx_m=20.0e3, dt_s=1200.0)
    s = m.rest_state()
    for _ in range(13000):
        s = m.step(s)
    u_mid = float(np.abs(s.u).max())
    for _ in range(12000):
        s = m.step(s)
    u_late = float(np.abs(s.u).max())
    assert abs(u_late - u_mid) < 0.01, f"not steady: {u_mid:.4f} -> {u_late:.4f}"
    assert u_late < 0.05

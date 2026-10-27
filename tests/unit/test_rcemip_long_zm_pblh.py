"""run_rcemip_long hands Zhang-McFarlane a zero PBL height (follow-up row 23)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "run"))


def test_zm_gets_zero_pblh_per_column(monkeypatch):
    mod = importlib.import_module("run_rcemip_long")
    import legoesm.atmosphere.physics.convection.integration as conv_int
    import legoesm.atmosphere.physics.microphysics.integration as micro_int
    import legoesm.atmosphere.physics.radiation.integration as rad_int

    seen = {}
    zero = lambda *a, **k: [jnp.zeros(())]  # noqa: E731

    def conv_spy(*a, phys_state=None, **k):
        seen["phys_state"] = phys_state
        return [jnp.zeros(())]

    monkeypatch.setattr(micro_int, "make_microphysics_physics", lambda *a, **k: zero)
    monkeypatch.setattr(rad_int, "make_radiation_physics", lambda *a, **k: zero)
    monkeypatch.setattr(conv_int, "make_convection_physics", lambda *a, **k: conv_spy)
    fn = mod._compose_nh_moist_physics("nonhydrostatic", 60.0,
                                       convection_scheme="zhang_mcfarlane")
    state = SimpleNamespace(theta_prime=SimpleNamespace(data=jnp.ones((6, 3, 3, 5))))
    fn(state)
    ps = seen["phys_state"]
    np.testing.assert_array_equal(np.asarray(ps.pblh), np.zeros(6 * 3 * 3))
    assert ps.cloud_fraction is None

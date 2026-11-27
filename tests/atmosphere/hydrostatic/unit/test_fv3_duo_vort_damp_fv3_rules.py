"""User decision 2026-11-27 ("FV3 rules"): the duo deck's vorticity /
delp / pt damping follows FV3 instead of the reference deck's verbatim
``vtdm4 = 0.12, nord_v = 2``:

* ``nord_v = min(2, nord)``  (dyn_core.F90:757);
* ``vtdm4 = d4_bg / 3`` unless given (FV3 vtdm4 namelist doc: below
  d4_bg, first guess ~1/3 of it).

Each test goes red on revert of the corresponding rule."""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nord, nord_v", [(0, 0), (1, 1), (2, 2), (3, 2)])
def test_nord_v_is_min_2_nord(nord, nord_v):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import duo_sw_deck
    assert duo_sw_deck(nord=nord, d4_bg=0.05).nord_v == nord_v


def test_vtdm4_defaults_to_a_third_of_d4_bg_and_overrides():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, ORACLE_DAMPING, duo_sw_deck)
    cfg = FV3DuoConfig()
    assert cfg.vtdm4 is None and (cfg.nord, cfg.d4_bg) == (1, 0.05)
    prod = duo_sw_deck(nord=cfg.nord, d4_bg=cfg.d4_bg, vtdm4=cfg.vtdm4)
    assert prod.nord_v == 1
    assert prod.damp_v == pytest.approx(0.05 / 3.0, rel=1e-15)
    # the rule tracks a d4_bg sweep
    assert duo_sw_deck(nord=1, d4_bg=0.09).damp_v == pytest.approx(0.03)
    # explicit override path: wins over the rule (oracle deck pins 0.12)
    assert duo_sw_deck(nord=1, d4_bg=0.05, vtdm4=0.07).damp_v == 0.07
    assert duo_sw_deck(nord=1, d4_bg=0.05, vtdm4=0.0).damp_v == 0.0
    o = duo_sw_deck(**{k: ORACLE_DAMPING[k] for k in ("nord", "d4_bg", "vtdm4")})
    assert (o.nord_v, o.damp_v) == (2, 0.12)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="vtdm4"):
            duo_sw_deck(nord=1, d4_bg=0.05, vtdm4=bad)


def test_model_forwards_vtdm4_to_the_step(monkeypatch):
    """FV3DuoConfig.vtdm4 (None and explicit) reaches the deck the model
    compiles."""
    import legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics as m
    from legoesm.grids.factory import create_fv3_duo_grid
    seen = []
    real = m.duo_sw_deck

    def spy(**kw):
        seen.append(real(**kw))
        return seen[-1]
    monkeypatch.setattr(m, "duo_sw_deck", spy)
    grid = create_fv3_duo_grid(12, 3)
    m.FV3DuoDynamicsModel(grid, m.FV3DuoConfig(km=5, n_split=2))
    m.FV3DuoDynamicsModel(grid, m.FV3DuoConfig(km=5, n_split=2, vtdm4=0.07))
    assert [(d.nord_v, d.damp_v) for d in seen] == [(1, 0.05 / 3.0), (1, 0.07)]


@pytest.mark.slow
def test_rest_state_topo_spurious_wind_bound():
    """DCMIP 2012 rest state over a 2 km mountain, C36 L32 production deck,
    1 day (45 x 1920 s).  Predeclared bound 3.0 m/s: the pre-decision deck
    (vtdm4=0.12, nord_v=2) reached 6.80 m/s, the FV3-rules deck ~2.1."""
    from scripts.matrix._cube_duo_adapters import build_duo_hydro
    duo = build_duo_hydro(36, shared_case="rest_state_topo",
                          ic_kwargs={"h_0": 2000.0})
    s = duo.state
    for _ in range(45):
        s = duo.step(s, duo.dt)
    u, v = duo.winds(s)
    wmax = float(np.sqrt(np.asarray(u) ** 2 + np.asarray(v) ** 2).max())
    assert np.isfinite(wmax) and wmax < 3.0, wmax

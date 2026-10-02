"""Decision B2 plumbing: the duo's d_sw5 divergence-damping knobs
(``nord``, ``d4_bg``) reach the jitted step through ONE SWConfig deck that
is, at the oracle-deck values, exactly the config the phases build for
``cfg=None`` -- so the certified step is unchanged until the deck moves."""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.factory import create_fv3_duo_grid  # noqa: E402

N, NG, KM = 12, 3, 5


@pytest.fixture(scope="module")
def grid():
    return create_fv3_duo_grid(N, NG)


def test_deck_at_oracle_values_is_the_phases_own_deck():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import duo_sw_deck
    from legoesm.core.fv3_dsw_phase_3d import DSW_DUO_DECK
    from legoesm.core.fv3_dsw_tail_3d import _TAIL_DECK
    deck = duo_sw_deck(nord=2, d4_bg=0.12)

    def diff(a, b):
        return {f for f in a._fields if getattr(a, f) != getattr(b, f)}
    # The merged deck differs from each phase's own deck ONLY in a knob
    # that phase never reads: the transport deck leaves the d_sw5 damping
    # trio at SWConfig defaults (dddmp 0.2; the tail pins 0.0), the tail
    # deck leaves the transport order hord_tr at its default 8 (transport
    # pins 6).  The step-level test below proves the step is unchanged.
    assert diff(deck, DSW_DUO_DECK) == {"dddmp"} and deck.dddmp == 0.0
    assert diff(deck, _TAIL_DECK) == {"hord_tr"} and deck.hord_tr == 6
    assert (deck.nord, deck.d4_bg) == (2, 0.12)
    assert duo_sw_deck(nord=1, d4_bg=0.07).nord == 1
    with pytest.raises(ValueError, match="nord"):
        duo_sw_deck(nord=4, d4_bg=0.1)
    with pytest.raises(ValueError, match="d4_bg"):
        duo_sw_deck(nord=1, d4_bg=-0.1)


def _bundles_equal(a, b):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def test_oracle_pinned_step_is_bitwise_the_cfg_none_step(grid, monkeypatch):
    """The model now always passes a deck; at the ORACLE values (pinned
    explicitly, no longer the default) that deck must reproduce the
    ``cfg=None`` step bit for bit (the certified identities rest on it)."""
    import legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics as m
    dyn = m.FV3DuoDynamicsModel(
        grid, m.FV3DuoConfig(km=KM, n_split=2, **m.ORACLE_DAMPING))
    ic = dyn.dcmip16_initial_state(n_tracers=1)
    out = dyn.step(ic, 120.0)
    monkeypatch.setattr(m, "duo_sw_deck", lambda **kw: None)
    dyn_none = m.FV3DuoDynamicsModel(grid, m.FV3DuoConfig(km=KM, n_split=2))
    assert _bundles_equal(dyn_none.step(ic, 120.0), out)


def test_default_is_the_production_deck_not_the_oracle(grid):
    """User decision 1b (2026-10-02): the DEFAULT is the MPAS-matched
    production damping (nord=1, d4_bg=0.05); the oracle deck is opt-in
    and differs from it on a real step."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel, ORACLE_DAMPING)
    assert (FV3DuoConfig().nord, FV3DuoConfig().d4_bg) == (1, 0.05)
    assert (FV3DuoConfig().sponge_del2_top_layers,
            FV3DuoConfig().sponge_del2_top_factor) == (2, 8.0)
    assert FV3DuoConfig().sponge_d2_top > 0.0
    assert ORACLE_DAMPING == {"nord": 2, "d4_bg": 0.12, "sponge_del2_top_layers": 0}
    prod = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=KM, n_split=2))
    ic = prod.dcmip16_initial_state(n_tracers=1)
    oracle = FV3DuoDynamicsModel(
        grid, FV3DuoConfig(km=KM, n_split=2, **ORACLE_DAMPING))
    assert not _bundles_equal(prod.step(ic, 120.0), oracle.step(ic, 120.0))


def test_nord_and_d4_bg_bind_on_the_step(grid):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import FV3DuoConfig, FV3DuoDynamicsModel
    base = FV3DuoDynamicsModel(
        grid, FV3DuoConfig(km=KM, n_split=2, nord=2, d4_bg=0.12))
    ic = base.dcmip16_initial_state(n_tracers=1)
    ref = base.step(ic, 120.0)
    del4 = FV3DuoDynamicsModel(
        grid, FV3DuoConfig(km=KM, n_split=2, nord=1, d4_bg=0.08)).step(ic, 120.0)
    off = FV3DuoDynamicsModel(
        grid, FV3DuoConfig(km=KM, n_split=2, nord=1, d4_bg=0.0)).step(ic, 120.0)
    for a, b in ((del4, ref), (off, del4)):
        assert not _bundles_equal(a, b)
    for b in (del4, off):
        assert all(np.isfinite(np.asarray(x)).all()
                   for x in jax.tree_util.tree_leaves(b))


def test_factory_forwards_the_knobs_to_the_duo_config():
    """The AMIP driver's DycoreConfig fields reach FV3DuoConfig (source
    gate on the one call site, plus the field names themselves)."""
    import inspect

    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import FV3DuoConfig
    from legoesm.driver import component_factory as cf
    from legoesm.driver.config import DycoreConfig
    assert FV3DuoConfig()._asdict()["nord"] == DycoreConfig().fv3_duo_nord == 1
    assert FV3DuoConfig()._asdict()["d4_bg"] == DycoreConfig().fv3_duo_d4_bg == 0.05
    src = inspect.getsource(cf._create_fv3_duo_column_model)
    assert src.count("nord=config.dycore.fv3_duo_nord") == 1
    assert src.count("d4_bg=config.dycore.fv3_duo_d4_bg") == 1
    src_hs = inspect.getsource(cf.create_atmosphere_dycore)
    assert src_hs.count("nord=config.dycore.fv3_duo_nord") >= 1
    for name in ("layers", "factor", "d2_top"):
        assert src.count(f"config.dycore.fv3_duo_sponge_{name}") == 1, name
        assert src_hs.count(f"config.dycore.fv3_duo_sponge_{name}") >= 1, name


def test_merged_deck_knobs_are_read_only_by_their_own_phase():
    """GLM 2026-10-02: the merged deck hands the transport phase a dddmp it
    did not carry before (0.0, not the SWConfig default 0.2) and the tail
    a hord_tr (6, not 8).  Those values are consumed ONLY by the other
    phase: the transport source never reads ``dddmp`` and the tail source
    never reads ``hord_tr`` (the attribute reads below are the ones that
    run; the bitwise step test above is the runtime witness)."""
    import inspect
    import re

    from legoesm.core import fv3_dsw_phase_3d, fv3_dsw_tail_3d
    transport = inspect.getsource(fv3_dsw_phase_3d)
    tail = inspect.getsource(fv3_dsw_tail_3d)
    assert re.search(r"\.hord_tr\b", transport)          # transport reads it
    assert re.search(r"\.dddmp\b", tail)                 # tail reads it
    assert not re.search(r"\.dddmp\b", transport)
    assert not re.search(r"\.hord_tr\b", tail)


def test_default_sponge_binds_on_the_step_and_off_is_bitwise_the_bare_deck(grid):
    """Decision B1: the default deck carries the top del-2 sponge (the
    step differs from layers=0), a zero coefficient under layers=2 is
    bitwise the layers=0 step (the static gate, no dormant term), and the
    sponge needs dddmp == 0 on the deck."""
    import pytest
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel, duo_sw_deck)
    on = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=KM, n_split=2))
    off = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=KM, n_split=2, sponge_del2_top_layers=0))
    zero = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=KM, n_split=2, sponge_d2_top=0.0))
    ic = on.dcmip16_initial_state(n_tracers=1)
    o_on, o_off, o_zero = on.step(ic, 120.0), off.step(ic, 120.0), zero.step(ic, 120.0)
    assert not _bundles_equal(o_on, o_off)
    assert _bundles_equal(o_zero, o_off)
    with pytest.raises(ValueError, match="layers"):
        FV3DuoDynamicsModel(grid, FV3DuoConfig(km=KM, n_split=2, sponge_del2_top_layers=KM + 1))
    assert duo_sw_deck(nord=1, d4_bg=0.05).sponge_del2_top_layers == 0
    deck = duo_sw_deck(nord=1, d4_bg=0.05, sponge_del2_top_layers=2, sponge_d2_top=0.01)
    assert deck.dddmp == 0.0 and deck.sponge_d2_top == 0.01

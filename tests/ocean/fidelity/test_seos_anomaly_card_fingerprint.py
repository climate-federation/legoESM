"""WHICH CARDS the exact S-EOS anomaly reaches -- a gate, not a claim.

``iterate_eos_and_pressure_anomaly`` now returns NEMO's own density anomaly
(``eosbn2.F90:365-367``) instead of ``rho - rho_0`` whenever the EOS publishes
S-EOS coefficients that agree with the caller's ``rho_0`` and the depth is
geometric.  That is a change to a SHARED helper, so every card that reaches it
changes, and Rule 12 says the set must be enumerated and MEASURED rather than
asserted.

This file is that enumeration, machine-checked: if a future card starts
selecting ``eos="nemo_seos"`` with ``eos_depth="geometric"``, or an existing
one stops, the fingerprint below goes red and somebody has to look.  The
numbers themselves are in
``scripts/validate/ocean_fidelity/dino_1226/seos_restart_density_gate.py``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import make_eos_fn, nemo_seos_anomaly

jax.config.update("jax_enable_x64", True)


#: The exact-anomaly branch fires for exactly these (eos, eos_depth) pairs.
#: ``eos_depth="insitu"`` never reaches it (the iterated-pressure branch is
#: untouched), and an EOS that does not publish coefficients cannot be asked
#: for its anomaly.
FIRES = {("nemo_seos", "geometric")}


def _fires(eos, eos_depth) -> bool:
    return (eos, eos_depth) in FIRES


def test_named_ocean_recipes_are_unchanged():
    """None of the catalogued recipes selects the S-EOS at geometric depth."""
    from legoesm.ocean.recipes import get_recipe, list_recipes

    def dig(o, key):
        if isinstance(o, dict):
            if key in o:
                return o[key]
            for v in o.values():
                got = dig(v, key)
                if got is not None:
                    return got
        return None

    names = list_recipes()
    assert names, "the recipe catalogue is empty; this gate would be vacuous"
    changed = []
    for name in names:
        bundle = get_recipe(name)
        eos = dig(bundle, "eos")
        depth = dig(bundle, "eos_depth") or "insitu"
        if _fires(eos, depth):
            changed.append((name, eos, depth))
    assert changed == [], (
        f"named recipes now reach the exact-anomaly branch: {changed}. "
        "Their density changed; measure it before this list is widened")


def test_nemo_testcase_cards_are_unchanged_and_say_why():
    """The NEMO test cases run TEOS-10, a separate polynomial and a separate
    path: ``make_eos_fn`` publishes S-EOS coefficients only for the S-EOS, so
    those cards keep ``rho - rho_0`` byte for byte."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    cases = ["LOCK_EXCHANGE-zco", "OVERFLOW-zps"]
    seen = []
    for case in cases:
        card = build_nemo_testcase_card(case)
        mc = card.recipe.model_config
        eos = getattr(mc, "eos", None)
        depth = getattr(mc, "eos_depth", "insitu")
        seen.append((case, eos, depth))
        assert not _fires(eos, depth), (
            f"{case} now reaches the exact-anomaly branch ({eos}, {depth}); "
            "its density changed and must be re-measured")
        assert getattr(make_eos_fn(eos), "nemo_seos_cfg", None) is None
    assert len(seen) == len(cases)


@pytest.mark.parametrize(
    "recipe,expect_fires",
    [("nemo_dino_kamm_mlf", True),
     ("nemo_paper", True),
     ("legoesm_default", False),
     ("veros", False),
     ("mitgcm", False),
     ("oceananigans", False)],
)
def test_dino_recipe_fingerprint(recipe, expect_fires):
    """The DINO recipes, one row each -- the only cards that change."""
    from legoesm.ocean.experiments import dino as dm

    cfg = dm.dino_config_for_recipe(recipe)
    eos = cfg.eos
    depth = getattr(cfg, "eos_depth", "insitu")
    assert _fires(eos, depth) is expect_fires, (
        f"recipe {recipe!r} resolves to (eos={eos!r}, eos_depth={depth!r}), "
        f"which {'does' if _fires(eos, depth) else 'does not'} reach the "
        f"exact-anomaly branch; the fingerprint expected "
        f"{'it to' if expect_fires else 'it not to'}")


def test_the_change_is_measurable_and_is_the_rho0_roundtrip():
    """On a card that DOES fire, name the size of the change and prove it is
    exactly the round trip -- not some other statement moving at the same
    time.  Without this row the fingerprint would only say WHERE, never WHAT.
    """
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import NemoSEOSConfig

    rng = np.random.default_rng(11)
    ny, nx, nz = 6, 5, 12
    T = jnp.asarray(rng.uniform(-1.5, 27.0, (ny, nx, nz)))
    S = jnp.asarray(rng.uniform(33.5, 37.0, (ny, nx, nz)))
    dz = jnp.full((nz,), 300.0)
    depth = jnp.cumsum(dz) - 0.5 * dz
    rho0 = 1026.0
    cfg = NemoSEOSConfig(rho0=rho0)
    eos_fn = make_eos_fn("nemo_seos", eos_nemo_seos=cfg)
    _rho, new, _p = iterate_eos_and_pressure_anomaly(
        T, S, jnp.ones((ny, nx), dtype=bool), lambda f: f, eos_fn, dz,
        rho0, constants.g, n_iter=2, eos_depth="geometric",
        eos_geometric_depth_1d=depth)
    old = np.asarray(eos_fn(T, S, (rho0 * constants.g) * depth)) - rho0
    exact = np.asarray(nemo_seos_anomaly(T, S, depth, cfg))
    assert np.array_equal(np.asarray(new), exact)
    moved = int((old != exact).sum())
    assert moved > 0, (
        "old and new agree everywhere, so this card's density did not change "
        "and the fingerprint's 'True' rows are wrong")
    rel = np.abs(old - exact).max() / np.sqrt(np.mean(exact ** 2))
    assert 1e-16 < rel < 1e-11, (
        f"the change is {rel:.3e} of the anomaly's rms -- outside the band a "
        "pure rho0 round trip can produce, so something else moved too")

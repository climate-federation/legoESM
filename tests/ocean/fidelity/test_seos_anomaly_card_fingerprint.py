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
    """(eos, eos_depth) is NECESSARY, not sufficient.

    The runtime ALSO requires the ``rho_prime`` else-branch: a caller passing
    ``rho_ref_z_static`` or ``use_depth_dependent_ref`` subtracts a reference
    PROFILE and never reaches the new arm.  This predicate does not model
    that, so it OVER-counts -- which is the safe direction for a fingerprint
    (it can name a card that did not change; it cannot miss one that did).
    ``test_dino_cards_take_the_plain_reference_branch`` below closes the gap
    for the cards that fire.
    """
    return (eos, eos_depth) in FIRES


def test_dino_cards_take_the_plain_reference_branch():
    """The firing cards must subtract rho_0, not a reference profile."""
    from legoesm.ocean.experiments import dino as dm

    for recipe in sorted(DINO_RECIPES_THAT_FIRE):
        cfg = dm.dino_config_for_recipe(recipe)
        for field in ("rho_ref_z_static", "use_depth_dependent_ref"):
            val = getattr(cfg, field, None)
            assert not val, (
                f"{recipe} sets {field}={val!r}, so its density does NOT "
                "reach the exact-anomaly branch and the fingerprint's "
                "'fires' verdict is wrong for it")


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


#: The DINO recipes that reach the exact-anomaly branch.  MEASURED, and the
#: test below iterates the REGISTRY rather than this list, so a new card that
#: fires turns it red instead of passing unnoticed.  An earlier version
#: parametrized six literal names and omitted ``nemo_dino_kamm`` -- a live
#: card whose density changed and which the commit message did not list.  An
#: independent diff review found it.
DINO_RECIPES_THAT_FIRE = {
    "nemo_paper", "nemo_dino_kamm", "nemo_dino_kamm_mlf",
}


def test_dino_recipe_fingerprint_covers_every_registered_card():
    """Every DINO recipe, one row each -- from the registry, not a literal."""
    from legoesm.ocean.experiments import dino as dm

    names = sorted(dm.DINO_RECIPES)
    assert len(names) >= 6, (
        f"only {len(names)} DINO recipes registered; this gate would be "
        "nearly vacuous")
    fires, quiet = set(), set()
    for recipe in names:
        cfg = dm.dino_config_for_recipe(recipe)
        depth = getattr(cfg, "eos_depth", "insitu")
        (fires if _fires(cfg.eos, depth) else quiet).add(recipe)
    assert fires == DINO_RECIPES_THAT_FIRE, (
        f"the set of DINO cards whose density changed is {sorted(fires)}, "
        f"not {sorted(DINO_RECIPES_THAT_FIRE)}. Measure the new card before "
        "widening this set -- every name in it is a card whose trajectory "
        "this change can move")
    assert quiet, "no DINO card is unaffected; that is implausible, re-read"


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

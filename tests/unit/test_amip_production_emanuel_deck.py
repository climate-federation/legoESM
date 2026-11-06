"""config/amip/amip_production_emanuel.yaml: the production deck with Emanuel
convection instead of Zhang-McFarlane, for the #2266 momentum-transport A/B.

Pins that the deck differs from production only in the convection rows, that
it resolves on the production (MPAS) lane without new validation errors, and
that the A/B's two arms (deck alone vs deck + --emanuel-enable-cmt) differ in
exactly one resolved field and request the MPAS wind reconstruction
(``mpas_cmt``) only when on.  That the bridge then puts Emanuel's tendency on
the MPAS edge winds (and exact zeros when off) is pinned by
tests/unit/test_mpas_cmt_wiring.py.
"""
from __future__ import annotations

import pathlib

from legoesm.driver.config import ZM_SCALAR_FIELDS, ZM_WIDTH_FIELDS
from legoesm.driver.physics_pipeline import convection_config_for
from legoesm.driver.run_config_yaml import load_yaml_config, read_yaml_with_includes

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"
_PROD = _AMIP / "amip_production.yaml"
_DECK = _AMIP / "amip_production_emanuel.yaml"


def _cfg(deck, *argv):
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(deck, p))
    return run_amip.build_config_from_args(p.parse_args(list(argv)))


def _errors(cfg):
    # The decks alone lack the launcher's forcing paths, so validate_strict
    # reports those; compare the error sets instead of requiring none.
    try:
        cfg.validate_strict()
    except ValueError as exc:
        return set(str(exc).splitlines())
    return set()


def test_deck_differs_from_production_only_in_convection_rows():
    prod, deck = read_yaml_with_includes(_PROD), read_yaml_with_includes(_DECK)
    diff = {k for k in prod.keys() | deck.keys() if prod.get(k) != deck.get(k)}
    assert diff == {"convection", "emanuel_enable_cmt",
                    *ZM_SCALAR_FIELDS, *ZM_WIDTH_FIELDS}
    assert prod["convection"] == "zhang_mcfarlane"
    assert deck["convection"] == "emanuel"
    assert deck["emanuel_enable_cmt"] is False
    assert all(deck[f] is None for f in (*ZM_SCALAR_FIELDS, *ZM_WIDTH_FIELDS))


def test_deck_adds_no_validation_error_on_the_production_lane():
    prod, off = _cfg(_PROD), _cfg(_DECK)
    assert off.grid.grid_type == prod.grid.grid_type
    assert off.dycore.discretization == prod.dycore.discretization == "mpas"
    assert _errors(off) <= _errors(prod)
    assert _errors(_cfg(_DECK, "--emanuel-enable-cmt")) <= _errors(prod)


def test_arms_differ_in_exactly_one_resolved_field():
    off, on = _cfg(_DECK), _cfg(_DECK, "--emanuel-enable-cmt")
    a, b = off._asdict(), on._asdict()
    assert {k for k in a if a[k] != b[k]} == {"emanuel_enable_cmt"}
    c_off, c_on = convection_config_for(off), convection_config_for(on)
    assert c_off.scheme == c_on.scheme == "emanuel"
    assert c_off.emanuel.enable_cmt is False and c_off.mpas_cmt is False
    assert c_on.emanuel.enable_cmt is True and c_on.mpas_cmt is True
    assert c_off._replace(mpas_cmt=True, emanuel=c_on.emanuel) == c_on

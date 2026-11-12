"""The two_region flux A/B instrument covers every deck that selects it."""
import importlib.util
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _script():
    path = ROOT / "scripts" / "validate" / "oracle" / "ecrad_ifs" / "two_region_flux_ab.py"
    spec = importlib.util.spec_from_file_location("two_region_flux_ab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _two_region_decks():
    pat = re.compile(r"^\s*cloud_optics_inhomogeneity:\s*\"?two_region", re.M)
    return {str(p.relative_to(ROOT)) for p in (ROOT / "config").rglob("*.yaml")
            if pat.search(p.read_text())}


def test_deck_table_is_exactly_the_two_region_decks():
    assert set(_script().DECKS) == _two_region_decks()


def _find(d, keys):
    """All values of any of ``keys`` anywhere in a nested deck."""
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += [v] if k in keys else _find(v, keys)
    return out


def test_each_deck_row_matches_its_cloud_scheme_and_subcolumns():
    mod = _script()
    for deck, setup in mod.DECKS.items():
        y = yaml.safe_load((ROOT / deck).read_text())
        (scheme,) = set(_find(y, ("cloud", "clouds")))
        (n_sub,) = set(_find(y, ("cloud_n_subcolumns",)))
        want_scheme, want_n, _ = mod.SETUPS[setup]
        assert (scheme, n_sub) == (want_scheme, want_n), deck

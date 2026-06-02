"""Keep .zenodo.json and CITATION.cff in sync (Stage D — DOI readiness).

Zenodo reads ``.zenodo.json`` to build the deposition (and mint the DOI) on a
GitHub release; ``CITATION.cff`` is what GitHub renders as "Cite this repository".
The two carry the same facts, so they must not drift — this guards title, license,
the author/creator name, and that the keyword sets agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _load():
    zenodo = json.loads((_ROOT / ".zenodo.json").read_text())
    cff = yaml.safe_load((_ROOT / "CITATION.cff").read_text())
    return zenodo, cff


def test_title_matches() -> None:
    zenodo, cff = _load()
    assert zenodo["title"] == cff["title"]


def test_license_matches() -> None:
    zenodo, cff = _load()
    assert zenodo["license"] == cff["license"]


def test_creator_matches_first_author() -> None:
    zenodo, cff = _load()
    author = cff["authors"][0]
    expected = f"{author['family-names']}, {author['given-names']}"
    creators = [c["name"] for c in zenodo["creators"]]
    assert expected in creators
    assert zenodo["creators"][0]["affiliation"] == author["affiliation"]


def test_keywords_are_a_superset_of_the_cff_keywords() -> None:
    zenodo, cff = _load()
    assert set(cff["keywords"]) <= set(zenodo["keywords"])


def test_zenodo_declares_open_software() -> None:
    zenodo, _cff = _load()
    assert zenodo["upload_type"] == "software"
    assert zenodo["access_right"] == "open"

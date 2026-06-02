"""Every page the mkdocs nav references must exist (Stage D — docs).

Guards against a nav entry pointing at a moved/renamed doc, which would break the
``mkdocs build`` (and the published site) silently until someone runs it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _nav_pages(nav):
    """Flatten the mkdocs nav tree to the list of referenced doc paths."""
    pages = []
    if isinstance(nav, str):
        pages.append(nav)
    elif isinstance(nav, list):
        for item in nav:
            pages.extend(_nav_pages(item))
    elif isinstance(nav, dict):
        for value in nav.values():
            pages.extend(_nav_pages(value))
    return pages


def test_every_nav_page_exists() -> None:
    cfg = yaml.safe_load((_ROOT / "mkdocs.yml").read_text())
    docs_dir = _ROOT / cfg.get("docs_dir", "docs")
    missing = [p for p in _nav_pages(cfg["nav"]) if not (docs_dir / p).is_file()]
    assert not missing, f"mkdocs nav references missing docs: {missing}"


def test_index_page_present() -> None:
    cfg = yaml.safe_load((_ROOT / "mkdocs.yml").read_text())
    docs_dir = _ROOT / cfg.get("docs_dir", "docs")
    assert (docs_dir / "index.md").is_file()

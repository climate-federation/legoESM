"""Every page in the Sphinx index toctree must exist (Stage D — docs).

Guards against a toctree entry pointing at a moved/renamed doc, which would break
``sphinx-build`` (and the published site) silently until someone runs it.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"


def _toctree_entries(index_md: str) -> list[str]:
    """Extract the document names listed in the index.md MyST {toctree} block."""
    entries: list[str] = []
    in_block = False
    for line in index_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("```{toctree}"):
            in_block = True
            continue
        if in_block:
            if stripped.startswith("```"):
                in_block = False
                continue
            if not stripped or stripped.startswith(":"):
                continue  # toctree option (:maxdepth:, :caption:) or blank
            # MyST entries may be "Title <target>" — resolve to the target doc.
            m = re.search(r"<([^>]+)>", stripped)
            entries.append(m.group(1) if m else stripped)
    return entries


def test_sphinx_conf_present() -> None:
    assert (_DOCS / "conf.py").is_file()
    assert (_DOCS / "index.md").is_file()


def test_every_toctree_entry_resolves_to_a_doc() -> None:
    entries = _toctree_entries((_DOCS / "index.md").read_text())
    assert entries, "index.md has no toctree entries"
    # ``self`` is the Sphinx special target for the index page itself; external
    # URL entries are not local files. Neither resolves to a docs/*.md.
    missing = [e for e in entries
               if e != "self"
               and not e.startswith(("http://", "https://"))
               and not (_DOCS / f"{e}.md").is_file()
               and not (_DOCS / f"{e}.rst").is_file()]
    assert not missing, f"toctree references missing docs: {missing}"


def test_conf_is_markdown_first() -> None:
    conf = (_DOCS / "conf.py").read_text()
    assert "myst_parser" in conf
    assert re.search(r'"\.md"', conf)  # .md is a recognised source suffix

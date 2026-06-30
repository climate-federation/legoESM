"""Sphinx configuration for the legoESM documentation site (Stage D).

Markdown-first (MyST) so the existing ``docs/*.md`` render directly.  Build with::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

The project version is single-sourced from the installed package metadata (the
same source ``CITATION.cff`` and ``pyproject.toml`` agree on — see
``tests/test_version_single_source.py``).
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

project = "legoESM"
author = "Pierre Gentine"
copyright = "2026, Pierre Gentine"  # noqa: A001 - Sphinx config name

try:
    release = _pkg_version("legoesm")
except Exception:  # pragma: no cover - docs may build without an install
    release = "0.0.0"
version = release

extensions = ["myst_parser"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
root_doc = "index"

# Only the curated pages are part of the rendered tree; the many internal logs /
# plans / audits under docs/ (now collected in ``md_files/``) stay browsable in
# the repo but out of the published nav.
exclude_patterns = [
    "_build",
    "archive/**",
    "planning/**",
    "research/**",
    "issues/**",
    "md_files/**",  # internal dev notes / logs / audits
    "specs/**",  # full LaTeX-style spec, linked explicitly from pages
    "forcing/**",
    "scaling/**",
    "ocean_experiments/**",
    "ocean_fidelity/**",
    "ocean_long_runs/**",
]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

# MyST: don't choke on the many bare cross-page anchors in curated pages.
myst_heading_anchors = 3
suppress_warnings = ["myst.header", "myst.xref_missing", "toc.not_readable"]

html_theme = "furo"
html_title = "legoESM"
html_theme_options = {
    "source_repository": "https://github.com/climate-federation/legoESM",
    "source_branch": "main",
    "source_directory": "docs/",
}

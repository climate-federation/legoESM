"""FV3_3D iter 368: doc-structure regression for iter-365
compaction.

The iter-365 compaction reduced FV3_3D.md from 3691 to 3327
lines by merging 40 iter prose entries (320-359) into a single
ToC block.  iter-368 pins the compaction structure so a future
edit doesn't accidentally re-expand the block.

Tests
-----

1. ``test_iter365_compaction_present`` — the compacted block
   marker is present in FV3_3D.md.
2. ``test_doc_size_below_3600_lines`` — doc stays compact
   post-iter-365.
3. ``test_iter365_compaction_covers_all_topics`` — compacted
   block lists every iter 320-359 group.
"""
from __future__ import annotations

from pathlib import Path

import pytest


DOC = Path(__file__).resolve().parents[1] / "FV3_3D.md"


@pytest.fixture(scope="module")
def doc_text():
    return DOC.read_text()


def test_iter365_compaction_present(doc_text):
    assert "**Iters 320-359 (compacted iter 365)**" in doc_text, (
        "iter-365 compaction block missing — doc may have been "
        "re-expanded."
    )


def test_doc_size_below_3600_lines(doc_text):
    n_lines = len(doc_text.splitlines())
    assert n_lines < 3600, (
        f"FV3_3D.md has {n_lines} lines (post-iter-365 compaction "
        f"target was < 3600).  Check for re-expansion or missing "
        f"compaction."
    )


def test_iter390_compaction_present(doc_text):
    """iter-390 compacted iter 375-384 into a single block."""
    assert "**Iters 375-384 (compacted iter 390)**" in doc_text, (
        "iter-390 compaction block missing — doc re-expanded."
    )


def test_iter390_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 375-384 (compacted iter 390)**"
    )
    section_end = doc_text.find(
        "- **Iters 360-374", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 375", "iter 376/377", "iter 378/379", "iter 380",
        "iter 381", "iter 382", "iter 383", "iter 384",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-390 compacted block missing topic ``{marker}``."
        )


def test_iter380_compaction_present(doc_text):
    """iter-380 compacted iter 360-374 into a single block."""
    assert "**Iters 360-374 (compacted iter 380)**" in doc_text, (
        "iter-380 compaction block missing — doc re-expanded."
    )


def test_iter380_compaction_covers_topics(doc_text):
    """iter-380 block lists every iter 360-374 topic."""
    section_start = doc_text.find(
        "**Iters 360-374 (compacted iter 380)**"
    )
    section_end = doc_text.find("**TL;DR**", section_start)
    assert section_start >= 0 and section_end > section_start, (
        "Could not locate iter-380 compacted section."
    )
    section = doc_text[section_start:section_end]
    required = [
        "iter 360", "iter 361/362", "iter 363", "iter 364",
        "iter 365", "iter 366", "iter 367", "iter 368", "iter 369",
        "iter 370", "iter 371", "iter 372", "iter 373", "iter 374",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-380 compacted block missing topic ``{marker}``."
        )


def test_iter365_compaction_covers_all_topics(doc_text):
    """Compacted block lists every iter 320-359 topic group."""
    required_markers = [
        "iter 320",
        "iter 321-323",
        "iter 324",
        "iter 325/326/327",
        "iter 328/329",
        "iter 330",
        "iter 331/343",
        "iter 332",
        "iter 333/334/335",
        "iter 336/337",
        "iter 338/339/344/347/348/349/350/351/352",
        "iter 340/342/353/361/362",
        "iter 341/346/355",
        "iter 345",
        "iter 354",
        "iter 356",
        "iter 357/358/359/360",
    ]
    # Find compacted section
    section_start = doc_text.find(
        "**Iters 320-359 (compacted iter 365)**"
    )
    # iter-380 compacted iter 364 into a block; section_end now
    # bounds at the next compaction block start.
    section_end = doc_text.find(
        "- **Iters 360-374", section_start,
    )
    if section_end < 0:
        section_end = len(doc_text)
    assert section_start >= 0 and section_end > section_start, (
        "Could not locate compacted section bounds."
    )
    section = doc_text[section_start:section_end]
    for marker in required_markers:
        assert marker in section, (
            f"Compacted block missing topic marker ``{marker}``."
        )

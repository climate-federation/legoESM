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


def test_doc_size_below_3650_lines(doc_text):
    n_lines = len(doc_text.splitlines())
    assert n_lines < 3650, (
        f"FV3_3D.md has {n_lines} lines (target < 3650 after iter-"
        f"454 production-usage expansion).  Check for "
        f"re-expansion or missing compaction."
    )


def test_iter423_summary_table_accurate(doc_text):
    """iter-423 summary table mentions all current flags."""
    section_start = doc_text.find("## FV3-fidelity stack")
    section_end = doc_text.find("## Production usage", section_start)
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    # All 5 NH flags
    for flag in [
        "use_fv3_d_con_cv",
        "use_fv3_vector_halo_uv",
        "use_fv3_dynamic_exner",
        "use_fv3_metric_aware_d_con",
        "use_fv3_cross_face_du_proj",
    ]:
        assert flag in section, f"NH flag {flag} missing from table"
    # PE-specific flag
    assert "use_fv3_a2b_zeta_corner" in section
    # Factory mention
    assert "make_fv3_faithful_pe_config" in section
    assert "make_fv3_faithful_nh_config" in section


def test_iter454_doc_example_lists_new_flags(doc_text):
    """iter-454: production usage doc example mentions the
    new factory-default flags from iter-431..453."""
    section_start = doc_text.find("## Production usage")
    section_end = doc_text.find("## Final state", section_start)
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    expected_markers = [
        "d_con_top_zero_levels = 2",
        "delt_max = 1.0",
        "nord_v = 1",
        "corner_div_damp_nord = 1",
        "corner_div_damp_d4_bg = 0.16",
        "use_fv3_sponge_damp_w = True",
        "use_fv3_sponge_damp_v = True",
        "rf_tau_days",
    ]
    for marker in expected_markers:
        assert marker in section, (
            f"iter-454 doc-update missing ``{marker}`` in "
            f"production-usage section."
        )


def test_iter417_production_section_present(doc_text):
    """iter-417 added a production-usage section (header
    updated by iter-454 to include update marker)."""
    assert "## Production usage (iter 417" in doc_text, (
        "iter-417 production-usage section missing — users lose "
        "the factory-based recommended setup recipe."
    )
    section_start = doc_text.find("## Production usage")
    section_end = doc_text.find("## Final state", section_start)
    section = doc_text[section_start:section_end]
    assert "make_fv3_faithful_pe_config" in section
    assert "make_fv3_faithful_nh_config" in section
    assert "use_duogrid=True" in section


def test_iter420_compaction_present(doc_text):
    """iter-420 compacted iter 405-414."""
    assert "**Iters 405-414 (compacted iter 420)**" in doc_text, (
        "iter-420 compaction block missing — doc re-expanded."
    )


def test_iter420_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 405-414 (compacted iter 420)**"
    )
    section_end = doc_text.find(
        "- **Iters 395-404", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 405", "iter 406", "iter 407", "iter 408",
        "iter 409", "iter 412", "iter 413", "iter 414",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-420 compacted block missing topic ``{marker}``."
        )


def test_iter400_compaction_present(doc_text):
    """iter-400 compacted iter 385-394 into a single block."""
    assert "**Iters 385-394 (compacted iter 400)**" in doc_text, (
        "iter-400 compaction block missing — doc re-expanded."
    )


def test_iter400_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 385-394 (compacted iter 400)**"
    )
    section_end = doc_text.find(
        "- **Iters 375-384", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 385", "iter 386/387", "iter 388", "iter 389",
        "iter 390", "iter 391", "iter 392", "iter 393", "iter 394",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-400 compacted block missing topic ``{marker}``."
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


def test_iter460_compaction_present(doc_text):
    """iter-460 compacted iter 445-454 into a single block."""
    assert "**Iters 445-454 (compacted iter 460)**" in doc_text, (
        "iter-460 compaction block missing — doc re-expanded."
    )


def test_iter460_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 445-454 (compacted iter 460)**"
    )
    section_end = doc_text.find(
        "**Iters 435-444", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 445", "iter 446", "iter 447", "iter 448",
        "iter 449", "iter 450", "iter 451", "iter 452",
        "iter 453", "iter 454",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-460 compacted block missing topic ``{marker}``."
        )


def test_iter450_compaction_present(doc_text):
    """iter-450 compacted iter 435-444 into a single block."""
    assert "**Iters 435-444 (compacted iter 450)**" in doc_text, (
        "iter-450 compaction block missing — doc re-expanded."
    )


def test_iter450_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 435-444 (compacted iter 450)**"
    )
    section_end = doc_text.find(
        "**Iters 425-434", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 435", "iter 436", "iter 437", "iter 438",
        "iter 439", "iter 440", "iter 441", "iter 442",
        "iter 443", "iter 444",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-450 compacted block missing topic ``{marker}``."
        )


def test_iter440_compaction_present(doc_text):
    """iter-440 compacted iter 425-434 into a single block."""
    assert "**Iters 425-434 (compacted iter 440)**" in doc_text, (
        "iter-440 compaction block missing — doc re-expanded."
    )


def test_iter440_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 425-434 (compacted iter 440)**"
    )
    section_end = doc_text.find(
        "**Iters 415-424", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 425", "iter 426", "iter 427", "iter 428",
        "iter 429", "iter 430", "iter 431", "iter 432",
        "iter 433", "iter 434",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-440 compacted block missing topic ``{marker}``."
        )


def test_iter430_compaction_present(doc_text):
    """iter-430 compacted iter 415-424 into a single block."""
    assert "**Iters 415-424 (compacted iter 430)**" in doc_text, (
        "iter-430 compaction block missing — doc re-expanded."
    )


def test_iter430_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 415-424 (compacted iter 430)**"
    )
    section_end = doc_text.find(
        "**Iters 405-414", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 415", "iter 416", "iter 418", "iter 419",
        "iter 421", "iter 422", "iter 423", "iter 424",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-430 compacted block missing topic ``{marker}``."
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

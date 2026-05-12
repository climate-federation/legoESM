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


def test_doc_size_below_5180_lines(doc_text):
    n_lines = len(doc_text.splitlines())
    assert n_lines < 5180, (
        f"FV3_3D.md has {n_lines} lines (target < 5180).  "
        f"Next compaction at iter 780."
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


def test_iter488_duogrid_investigation_summary(doc_text):
    """iter-488: synthesis section ``Duogrid investigation
    summary`` exists + mentions key iter numbers."""
    section_start = doc_text.find(
        "## Duogrid investigation summary"
    )
    section_end = doc_text.find("## Final state", section_start)
    assert section_start >= 0 and section_end > section_start, (
        "iter-488 Duogrid investigation summary section "
        "missing."
    )
    section = doc_text[section_start:section_end]
    # Table uses bare iter numbers in first column.
    for it_num in [
        "461", "466", "471", "473", "480", "482", "487",
        "489", "490", "493",   # iter-494 additions
    ]:
        assert f"| {it_num} |" in section, (
            f"iter-488/494 summary table missing row for iter "
            f"{it_num}."
        )


def test_iter486_edge_min_factories_doc_section(doc_text):
    """iter-486: production usage doc section for the 4
    legoESM-min-edge factories (iter-467/468/483/484) exists +
    mentions each factory by name."""
    section_start = doc_text.find(
        "### Edge-artifact-minimized factories"
    )
    section_end = doc_text.find("## Final state", section_start)
    assert section_start >= 0 and section_end > section_start, (
        "iter-486 min-edge factories doc section missing."
    )
    section = doc_text[section_start:section_end]
    expected_names = [
        "make_legoesm_nh_min_edge_config",
        "make_legoesm_nh_min_edge_aggressive_config",
        "make_legoesm_pe_min_edge_config",
        "make_legoesm_pe_min_edge_aggressive_config",
    ]
    for name in expected_names:
        assert name in section, (
            f"iter-486 min-edge doc section missing "
            f"factory name ``{name}``."
        )


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


def test_iter600_compaction_present(doc_text):
    """iter-600 compacted iter 591-599 into a single block."""
    assert "**Iters 591-599 (compacted iter 600)**" in doc_text, (
        "iter-600 compaction block missing — doc re-expanded."
    )


def test_iter600_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 591-599 (compacted iter 600)**"
    )
    section_end = doc_text.find(
        "**Iters 581-589", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 591", "iter 592", "iter 593", "iter 594",
        "iter 595", "iter 596", "iter 597", "iter 598",
        "iter 599",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-600 compacted block missing topic ``{marker}``."
        )


def test_iter610_compaction_present(doc_text):
    """iter-610 compacted iter 601-609 into a single block."""
    assert "**Iters 601-609 (compacted iter 610)**" in doc_text, (
        "iter-610 compaction block missing — doc re-expanded."
    )


def test_iter610_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 601-609 (compacted iter 610)**"
    )
    section_end = doc_text.find(
        "**Iters 591-599", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 601", "iter 602", "iter 603", "iter 604",
        "iter 605", "iter 606", "iter 607", "iter 608",
        "iter 609",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-610 compacted block missing topic ``{marker}``."
        )


def test_iter620_compaction_present(doc_text):
    """iter-620 compacted iter 611-619 into a single block."""
    assert "**Iters 611-619 (compacted iter 620)**" in doc_text, (
        "iter-620 compaction block missing — doc re-expanded."
    )


def test_iter620_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 611-619 (compacted iter 620)**"
    )
    section_end = doc_text.find(
        "**Iters 601-609", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "611", "612", "613", "614",
        "615", "616", "617", "618", "619",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-620 compacted block missing topic ``{marker}``."
        )


def test_iter630_compaction_present(doc_text):
    """iter-630 compacted iter 621-629 into a single block."""
    assert "**Iters 621-629 (compacted iter 630)**" in doc_text, (
        "iter-630 compaction block missing — doc re-expanded."
    )


def test_iter630_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 621-629 (compacted iter 630)**"
    )
    section_end = doc_text.find(
        "**Iters 611-619", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "621", "622", "623", "624",
        "625", "626", "627", "628", "629",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-630 compacted block missing topic ``{marker}``."
        )


def test_iter640_compaction_present(doc_text):
    """iter-640 compacted iter 631-639 into a single block."""
    assert "**Iters 631-639 (compacted iter 640)**" in doc_text, (
        "iter-640 compaction block missing — doc re-expanded."
    )


def test_iter640_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 631-639 (compacted iter 640)**"
    )
    section_end = doc_text.find(
        "**Iters 621-629", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "631", "632", "633", "634",
        "635", "636", "637", "638", "639",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-640 compacted block missing topic ``{marker}``."
        )


def test_iter650_compaction_present(doc_text):
    """iter-650 compacted iter 641-649 into a single block."""
    assert "**Iters 641-649 (compacted iter 650)**" in doc_text, (
        "iter-650 compaction block missing — doc re-expanded."
    )


def test_iter650_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 641-649 (compacted iter 650)**"
    )
    section_end = doc_text.find(
        "**Iters 631-639", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "641", "642", "643", "644",
        "645", "646", "647", "648", "649",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-650 compacted block missing topic ``{marker}``."
        )


def test_iter660_compaction_present(doc_text):
    """iter-660 compacted iter 651-659 into a single block."""
    assert "**Iters 651-659 (compacted iter 660)**" in doc_text, (
        "iter-660 compaction block missing — doc re-expanded."
    )


def test_iter660_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 651-659 (compacted iter 660)**"
    )
    section_end = doc_text.find(
        "**Iters 641-649", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "651", "652", "653", "654",
        "655", "656", "657", "658", "659",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-660 compacted block missing topic ``{marker}``."
        )


def test_iter670_compaction_present(doc_text):
    """iter-670 compacted iter 661-669 into a single block."""
    assert "**Iters 661-669 (compacted iter 670)**" in doc_text, (
        "iter-670 compaction block missing — doc re-expanded."
    )


def test_iter670_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 661-669 (compacted iter 670)**"
    )
    section_end = doc_text.find(
        "**Iters 651-659", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "661", "662", "663", "664",
        "665", "666", "667", "668", "669",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-670 compacted block missing topic ``{marker}``."
        )


def test_iter770_compaction_present(doc_text):
    """iter-770 compacted iter 760-769 into single block."""
    assert "**Iters 760-769 (compacted iter 770)**" in doc_text, (
        "iter-770 compaction block missing — doc re-expanded."
    )


def test_iter770_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 760-769 (compacted iter 770)**"
    )
    section_end = doc_text.find(
        "**Iters 751-759", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "760", "761", "762", "763", "764",
        "765", "766", "767", "768", "769",
        "lcl_temperature", "lcl_pressure", "lcl_height",
        "vapor_pressure_from_q", "dew_point", "relative_humidity",
        "lcl_state", "MSE", "Bolton",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-770 compacted block missing topic ``{marker}``."
        )


def test_iter760_compaction_present(doc_text):
    """iter-760 compacted iter 751-759 into single block."""
    assert "**Iters 751-759 (compacted iter 760)**" in doc_text, (
        "iter-760 compaction block missing — doc re-expanded."
    )


def test_iter760_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 751-759 (compacted iter 760)**"
    )
    section_end = doc_text.find(
        "**Iters 741-749", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "751", "752", "753", "754", "755",
        "756", "757", "758", "759",
        "moist_static_energy", "precipitable_water",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-760 compacted block missing topic ``{marker}``."
        )


def test_iter750_compaction_present(doc_text):
    """iter-750 compacted iter 741-749 into single block."""
    assert "**Iters 741-749 (compacted iter 750)**" in doc_text, (
        "iter-750 compaction block missing — doc re-expanded."
    )


def test_iter750_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 741-749 (compacted iter 750)**"
    )
    section_end = doc_text.find(
        "**Iters 731-739", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "741", "742", "743", "744", "745",
        "746", "747", "748", "749",
        "column_integral", "quartet",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-750 compacted block missing topic ``{marker}``."
        )


def test_iter740_compaction_present(doc_text):
    """iter-740 compacted iter 731-739 into single block."""
    assert "**Iters 731-739 (compacted iter 740)**" in doc_text, (
        "iter-740 compaction block missing — doc re-expanded."
    )


def test_iter740_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 731-739 (compacted iter 740)**"
    )
    section_end = doc_text.find(
        "**Iters 721-729", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "731", "732", "733", "734", "735",
        "736", "737", "738", "739",
        "theta_dry", "exner", "air_density",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-740 compacted block missing topic ``{marker}``."
        )


def test_iter730_compaction_present(doc_text):
    """iter-730 compacted iter 721-729 into single block."""
    assert "**Iters 721-729 (compacted iter 730)**" in doc_text, (
        "iter-730 compaction block missing — doc re-expanded."
    )


def test_iter730_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 721-729 (compacted iter 730)**"
    )
    section_end = doc_text.find(
        "**Iters 711-719", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "721", "722", "723", "724", "725",
        "726", "727", "728", "729",
        "virtual_temp", "compute_pkz", "cappa_moist",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-730 compacted block missing topic ``{marker}``."
        )


def test_iter720_compaction_present(doc_text):
    """iter-720 compacted iter 711-719 into a single block."""
    assert "**Iters 711-719 (compacted iter 720)**" in doc_text, (
        "iter-720 compaction block missing — doc re-expanded."
    )


def test_iter720_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 711-719 (compacted iter 720)**"
    )
    section_end = doc_text.find(
        "**Iters 701-709", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "711", "712", "713", "714", "715",
        "716", "717", "718", "719",
        "ppme", "moist_cv", "moist_cp", "Bolton",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-720 compacted block missing topic ``{marker}``."
        )


def test_iter710_compaction_present(doc_text):
    """iter-710 compacted iter 701-709 into a single block."""
    assert "**Iters 701-709 (compacted iter 710)**" in doc_text, (
        "iter-710 compaction block missing — doc re-expanded."
    )


def test_iter710_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 701-709 (compacted iter 710)**"
    )
    section_end = doc_text.find(
        "**Iters 691-699", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "701", "702", "703", "704", "705",
        "706", "707", "708", "709",
        "cs_prof", "cs_interpolator", "−23.2",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-710 compacted block missing topic ``{marker}``."
        )


def test_iter700_compaction_present(doc_text):
    """iter-700 compacted iter 691-699 into a single block."""
    assert "**Iters 691-699 (compacted iter 700)**" in doc_text, (
        "iter-700 compaction block missing — doc re-expanded."
    )


def test_iter700_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 691-699 (compacted iter 700)**"
    )
    section_end = doc_text.find(
        "**Iters 681-689", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "691", "692", "693", "694", "695",
        "696", "697", "698", "699",
        "a2b_ord4", "−25.8", "−21.9",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-700 compacted block missing topic ``{marker}``."
        )


def test_iter690_compaction_present(doc_text):
    """iter-690 compacted iter 681-689 into a single block."""
    assert "**Iters 681-689 (compacted iter 690)**" in doc_text, (
        "iter-690 compaction block missing — doc re-expanded."
    )


def test_iter690_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 681-689 (compacted iter 690)**"
    )
    section_end = doc_text.find(
        "**Iters 671-679", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "681", "682", "683", "684",
        "685", "686", "687", "688", "689",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-690 compacted block missing topic ``{marker}``."
        )


def test_iter680_compaction_present(doc_text):
    """iter-680 compacted iter 671-679 into a single block."""
    assert "**Iters 671-679 (compacted iter 680)**" in doc_text, (
        "iter-680 compaction block missing — doc re-expanded."
    )


def test_iter680_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 671-679 (compacted iter 680)**"
    )
    section_end = doc_text.find(
        "**Iters 661-669", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "671", "672", "673", "674",
        "675", "676", "677", "678", "679",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-680 compacted block missing topic ``{marker}``."
        )


def test_iter590_compaction_present(doc_text):
    """iter-590 compacted iter 581-589 into a single block."""
    assert "**Iters 581-589 (compacted iter 590)**" in doc_text, (
        "iter-590 compaction block missing — doc re-expanded."
    )


def test_iter590_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 581-589 (compacted iter 590)**"
    )
    section_end = doc_text.find(
        "**Iters 571-579", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 581", "iter 583", "iter 584", "iter 585",
        "iter 586", "iter 587", "iter 588", "iter 589",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-590 compacted block missing topic ``{marker}``."
        )


def test_iter580_compaction_present(doc_text):
    """iter-580 compacted iter 571-579 into a single block."""
    assert "**Iters 571-579 (compacted iter 580)**" in doc_text, (
        "iter-580 compaction block missing — doc re-expanded."
    )


def test_iter580_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 571-579 (compacted iter 580)**"
    )
    section_end = doc_text.find(
        "**Iters 561-569", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 571", "iter 572", "iter 573",
        "iter 576", "iter 578", "iter 579",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-580 compacted block missing topic ``{marker}``."
        )


def test_iter570_compaction_present(doc_text):
    """iter-570 compacted iter 561-569 into a single block."""
    assert "**Iters 561-569 (compacted iter 570)**" in doc_text, (
        "iter-570 compaction block missing — doc re-expanded."
    )


def test_iter570_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 561-569 (compacted iter 570)**"
    )
    section_end = doc_text.find(
        "**Iters 551-559", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 561", "iter 562", "iter 563", "iter 564",
        "iter 565", "iter 566", "iter 567", "iter 568",
        "iter 569",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-570 compacted block missing topic ``{marker}``."
        )


def test_iter560_compaction_present(doc_text):
    """iter-560 compacted iter 551-559 into a single block."""
    assert "**Iters 551-559 (compacted iter 560)**" in doc_text, (
        "iter-560 compaction block missing — doc re-expanded."
    )


def test_iter560_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 551-559 (compacted iter 560)**"
    )
    section_end = doc_text.find(
        "**Iters 541-549", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 551", "iter 552", "iter 553",
        "iter 555", "iter 556", "iter 557",
        "iter 558", "iter 559",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-560 compacted block missing topic ``{marker}``."
        )


def test_iter550_compaction_present(doc_text):
    """iter-550 compacted iter 541-549 into a single block."""
    assert "**Iters 541-549 (compacted iter 550)**" in doc_text, (
        "iter-550 compaction block missing — doc re-expanded."
    )


def test_iter550_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 541-549 (compacted iter 550)**"
    )
    section_end = doc_text.find(
        "**Iters 531-539", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 541", "iter 543", "iter 544",
        "iter 546", "iter 547", "iter 548", "iter 549",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-550 compacted block missing topic ``{marker}``."
        )


def test_iter540_compaction_present(doc_text):
    """iter-540 compacted iter 531-539 into a single block."""
    assert "**Iters 531-539 (compacted iter 540)**" in doc_text, (
        "iter-540 compaction block missing — doc re-expanded."
    )


def test_iter540_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 531-539 (compacted iter 540)**"
    )
    section_end = doc_text.find(
        "**Iters 521-529", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 531", "iter 532", "iter 533", "iter 534",
        "iter 535", "iter 536", "iter 537", "iter 538",
        "iter 539",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-540 compacted block missing topic ``{marker}``."
        )


def test_iter530_compaction_present(doc_text):
    """iter-530 compacted iter 521-529 into a single block."""
    assert "**Iters 521-529 (compacted iter 530)**" in doc_text, (
        "iter-530 compaction block missing — doc re-expanded."
    )


def test_iter530_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 521-529 (compacted iter 530)**"
    )
    section_end = doc_text.find(
        "**Iters 511-519", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 521", "iter 522", "iter 523", "iter 524",
        "iter 525", "iter 526", "iter 527", "iter 528",
        "iter 529",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-530 compacted block missing topic ``{marker}``."
        )


def test_iter520_compaction_present(doc_text):
    """iter-520 compacted iter 511-519 into a single block."""
    assert "**Iters 511-519 (compacted iter 520)**" in doc_text, (
        "iter-520 compaction block missing — doc re-expanded."
    )


def test_iter520_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 511-519 (compacted iter 520)**"
    )
    section_end = doc_text.find(
        "**Iters 501-509", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 511", "iter 512", "iter 513", "iter 514",
        "iter 515", "iter 516", "iter 517", "iter 518",
        "iter 519",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-520 compacted block missing topic ``{marker}``."
        )


def test_iter510_compaction_present(doc_text):
    """iter-510 compacted iter 501-509 into a single block."""
    assert "**Iters 501-509 (compacted iter 510)**" in doc_text, (
        "iter-510 compaction block missing — doc re-expanded."
    )


def test_iter510_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 501-509 (compacted iter 510)**"
    )
    section_end = doc_text.find(
        "**Iters 485-494", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 501", "iter 502", "iter 503", "iter 504",
        "iter 505", "iter 506", "iter 507", "iter 508",
        "iter 509",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-510 compacted block missing topic ``{marker}``."
        )


def test_iter500_compaction_present(doc_text):
    """iter-500 compacted iter 485-494 into a single block."""
    assert "**Iters 485-494 (compacted iter 500)**" in doc_text, (
        "iter-500 compaction block missing — doc re-expanded."
    )


def test_iter500_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 485-494 (compacted iter 500)**"
    )
    section_end = doc_text.find(
        "**Iters 475-484", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 485", "iter 486", "iter 487", "iter 488",
        "iter 489", "iter 490", "iter 491", "iter 492",
        "iter 493", "iter 494",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-500 compacted block missing topic ``{marker}``."
        )


def test_iter490_compaction_present(doc_text):
    """iter-490 compacted iter 475-484 into a single block."""
    assert "**Iters 475-484 (compacted iter 490)**" in doc_text, (
        "iter-490 compaction block missing — doc re-expanded."
    )


def test_iter490_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 475-484 (compacted iter 490)**"
    )
    section_end = doc_text.find(
        "**Iters 465-474", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 475", "iter 476", "iter 477", "iter 478",
        "iter 479", "iter 480", "iter 481", "iter 482",
        "iter 483", "iter 484",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-490 compacted block missing topic ``{marker}``."
        )


def test_iter480_compaction_present(doc_text):
    """iter-480 compacted iter 465-474 into a single block."""
    assert "**Iters 465-474 (compacted iter 480)**" in doc_text, (
        "iter-480 compaction block missing — doc re-expanded."
    )


def test_iter480_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 465-474 (compacted iter 480)**"
    )
    section_end = doc_text.find(
        "**Iters 455-464", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 465", "iter 466", "iter 467", "iter 468",
        "iter 469", "iter 470", "iter 471", "iter 472",
        "iter 473", "iter 474",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-480 compacted block missing topic ``{marker}``."
        )


def test_iter470_compaction_present(doc_text):
    """iter-470 compacted iter 455-464 into a single block."""
    assert "**Iters 455-464 (compacted iter 470)**" in doc_text, (
        "iter-470 compaction block missing — doc re-expanded."
    )


def test_iter470_compaction_covers_topics(doc_text):
    section_start = doc_text.find(
        "**Iters 455-464 (compacted iter 470)**"
    )
    section_end = doc_text.find(
        "**Iters 445-454", section_start,
    )
    assert section_start >= 0 and section_end > section_start
    section = doc_text[section_start:section_end]
    required = [
        "iter 455", "iter 456", "iter 457", "iter 458",
        "iter 459", "iter 460", "iter 461", "iter 462",
        "iter 463", "iter 464",
    ]
    for marker in required:
        assert marker in section, (
            f"iter-470 compacted block missing topic ``{marker}``."
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

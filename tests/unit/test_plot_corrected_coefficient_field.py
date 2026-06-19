"""Unit test for ``scripts/plot/plot_corrected_coefficient_field.py`` — the per-column
corrected-coefficient field map (single + multi outputs, 2-D + non-2-D grids).
"""

from __future__ import annotations

import json


def test_extract_coefficient_fields_single_and_multi():
    from scripts.plot.plot_corrected_coefficient_field import extract_coefficient_fields

    assert extract_coefficient_fields({"C_K": [0.4, 0.5]}) == {"C_K": [0.4, 0.5]}
    multi = {"fields": {"clubb_lite_C_K": [0.4], "clubb_lite_Pr_t": [1.0]}}
    assert extract_coefficient_fields(multi) == {"clubb_lite_C_K": [0.4], "clubb_lite_Pr_t": [1.0]}
    assert extract_coefficient_fields({"grid": {}}) == {}          # none present


def test_plot_corrected_coefficient_field_2d_single(tmp_path):
    from scripts.plot.plot_corrected_coefficient_field import plot_corrected_coefficient_field

    out = {"C_K": [0.4 + 0.001 * i for i in range(128)], "grid": {"shape_2d": [8, 16]}}
    png = tmp_path / "c.png"
    plot_corrected_coefficient_field(out, str(png))
    assert png.exists() and png.stat().st_size > 0


def test_plot_corrected_coefficient_field_multi_and_non2d(tmp_path):
    """A multi-coefficient output on a NON-2-D (cubed-sphere (6,4,4)) grid renders one
    per-column-line panel per coefficient — the imshow fallback, no crash."""
    from scripts.plot.plot_corrected_coefficient_field import plot_corrected_coefficient_field

    multi = {"fields": {"clubb_lite_C_K": [0.4] * 96, "clubb_lite_Pr_t": [1.0] * 96},
             "grid": {"shape_2d": [6, 4, 4]}}
    png = tmp_path / "m.png"
    plot_corrected_coefficient_field(multi, str(png))
    assert png.exists() and png.stat().st_size > 0


def test_plot_corrected_coefficient_field_no_coeff(tmp_path):
    from scripts.plot.plot_corrected_coefficient_field import plot_corrected_coefficient_field

    png = tmp_path / "none.png"
    plot_corrected_coefficient_field({"grid": {"shape_2d": [8, 16]}}, str(png))
    assert png.exists()                                            # the 'no field' note panel


def test_plot_corrected_coefficient_main(tmp_path):
    from scripts.plot.plot_corrected_coefficient_field import main

    j = tmp_path / "out.json"
    j.write_text(json.dumps({"C_K": [0.4] * 128, "grid": {"shape_2d": [8, 16]}}))
    png = tmp_path / "p.png"
    assert main([str(j), "--png", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 0

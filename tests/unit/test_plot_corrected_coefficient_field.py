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


def test_grid_caption_names_the_grid_or_empty():
    """The field-map panel names WHICH grid the per-column coefficients index (iter 272):
    a per-column field is meaningless without it. Absent grid → '' (a mock still renders)."""
    from scripts.plot.plot_corrected_coefficient_field import _grid_caption

    assert _grid_caption({}) == ""
    assert _grid_caption({"grid_type": "latlon", "shape_2d": [8, 16]}) == "  —  latlon (8, 16)"
    assert _grid_caption({"grid_type": "mpas", "ncol": 162}) == "  —  mpas 162 cols"


def test_plot_corrected_coefficient_field_2d_single(tmp_path):
    from scripts.plot.plot_corrected_coefficient_field import plot_corrected_coefficient_field

    out = {"C_K": [0.4 + 0.001 * i for i in range(128)],
           "grid": {"shape_2d": [8, 16], "grid_type": "latlon"}}
    png = tmp_path / "c.png"
    plot_corrected_coefficient_field(out, str(png))
    assert png.exists() and png.stat().st_size > 0


def test_plot_corrected_coefficient_field_imshow_orientation_is_row_major(tmp_path, monkeypatch):
    """The per-column field → grid reshape MUST be row-major (C-order), matching the
    manifest's ``flat_index = row*nlon + col``, so the spatial map shows each
    correction at its TRUE cell.  A transpose / F-order reshape would garble the map
    — the operator would read corrections at the wrong columns — and the PNG-exists
    test could NOT catch it.  Spy on ``Axes.imshow`` to lock the array a single
    distinctive value lands in: flat index 19 on an 8×16 grid ⇒ cell (1, 3)."""
    import matplotlib.axes
    import numpy as np

    from scripts.plot.plot_corrected_coefficient_field import (
        plot_corrected_coefficient_field,
    )

    captured = {}
    real_imshow = matplotlib.axes.Axes.imshow

    def spy_imshow(self, data, *a, **k):
        captured["arr"] = np.asarray(data, dtype=float)
        return real_imshow(self, data, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", spy_imshow)
    field = [0.0] * 128
    field[19] = 9.9                                      # flat 19 ⇒ (row 1, col 3) on 8×16
    out = {"C_K": field, "grid": {"shape_2d": [8, 16], "grid_type": "latlon"}}
    plot_corrected_coefficient_field(out, str(tmp_path / "c.png"))

    arr2d = captured["arr"]
    assert arr2d.shape == (8, 16)
    assert abs(float(arr2d[1, 3]) - 9.9) < 1e-9          # C-order: flat 19 = 1*16 + 3
    others = arr2d.copy()
    others[1, 3] = 0.0
    assert np.allclose(others, 0.0)                      # the value is at (1,3) and NOWHERE else


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

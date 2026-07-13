"""Unit tests for :mod:`legoesm.land.evaluation.fluxio`."""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.land.evaluation import fluxio


def _write_flux_out(path, n_rows=5):
    """Write a synthetic flux.out with the right number of columns."""
    ncols = len(fluxio.FLUX_VARS)
    rows = []
    for i in range(n_rows):
        rows.append(" ".join(f"{float(i + j):.3f}" for j in range(ncols)))
    path.write_text("\n".join(rows) + "\n")


def test_load_out_basic(tmp_path):
    p = tmp_path / "x.out"
    p.write_text("1 2 3\n4 5 6\n\n# junk line\n7 8 9\n")
    arr = fluxio.load_out(p)
    assert arr.shape == (3, 3)
    assert arr[2, 2] == 9.0


def test_load_out_modal_width(tmp_path):
    # A trailing truncated row (killed run) is dropped in favour of the mode.
    p = tmp_path / "x.out"
    p.write_text("1 2 3\n4 5 6\n7 8\n")
    arr = fluxio.load_out(p)
    assert arr.shape == (2, 3)


def test_load_out_empty(tmp_path):
    p = tmp_path / "empty.out"
    p.write_text("\n\n")
    arr = fluxio.load_out(p)
    assert arr.size == 0


def test_load_out_named_maps_schema(tmp_path):
    p = tmp_path / "CHATS7_2007-05_flux.out"
    _write_flux_out(p, n_rows=4)
    cols = fluxio.load_out_named(p, "flux")
    assert set(cols) == {s[0] for s in fluxio.FLUX_VARS}
    # First column is 'time', values are i + 0 = i.
    assert np.allclose(cols["time"], [0, 1, 2, 3])
    # 'shflx' is column index 3 -> value i + 3.
    idx = [s[0] for s in fluxio.FLUX_VARS].index("shflx")
    assert np.allclose(cols["shflx"], [idx + i for i in range(4)])


def test_load_out_named_unknown_tag(tmp_path):
    p = tmp_path / "x.out"
    p.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="Unknown .out tag"):
        fluxio.load_out_named(p, "nope")


def test_load_out_named_too_few_columns(tmp_path):
    p = tmp_path / "flux.out"
    p.write_text("1 2 3\n4 5 6\n")  # only 3 cols, flux needs 18
    with pytest.raises(ValueError, match="expected >="):
        fluxio.load_out_named(p, "flux")


def test_load_out_named_empty_is_empty_dict(tmp_path):
    p = tmp_path / "flux.out"
    p.write_text("")
    assert fluxio.load_out_named(p, "flux") == {}


def test_schemas_have_unique_keys_per_tag():
    for tag, specs in fluxio.SCHEMAS.items():
        keys = [s[0] for s in specs]
        assert len(keys) == len(set(keys)), f"dup key in schema {tag}"

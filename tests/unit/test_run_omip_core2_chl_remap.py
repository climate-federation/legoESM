"""Chlorophyll remapping: the oracle uses bilinear weights, we used IDW.

ORCA1 selects weights_reg05_bilinear.nc for sn_chl (namelist_cfg:170). That
file is a 4-triple SCRIP bilinear map whose weights sum to 1 to machine
precision and whose destination is the same `ncks -F -d lon,2,361 -d lat,1,331`
interior as the CORE-II weights. Its source is (y=361, x=721) -- a half-degree
grid with DUPLICATE endpoint columns, hence 721 and not 720. All verified.

Chlorophyll sets the shortwave penetration depth, so this feeds upper-ocean
heating; it is not a cosmetic difference.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

RUNNER = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "run" / "run_omip_core2.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_omip_runner_chl", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- the tripole halo fill --------------------------------------------------

def test_the_dropped_entries_are_filled_not_left_zero():
    """A silent zero in a forcing field is the failure this repo keeps hitting.
    The weights cover only the interior; the rest is the grid's own redundancy
    and must be reconstructed."""
    m = _mod()
    interior = np.arange(331 * 360, dtype=float).reshape(331, 360) + 1.0
    full = m._scrip_to_full_tripole(interior, 332, 362)
    assert full.shape == (332, 362)
    assert np.all(full != 0.0), "some destination entry was left as zero"


def test_the_cyclic_overlap_columns_mirror_the_interior():
    m = _mod()
    interior = np.random.default_rng(0).normal(size=(331, 360)) + 10.0
    full = m._scrip_to_full_tripole(interior, 332, 362)
    # column 0 is the overlap of column 360; column 361 of column 1
    assert np.array_equal(full[0:331, 0], full[0:331, 360])
    assert np.array_equal(full[0:331, 361], full[0:331, 1])
    # and the interior itself is placed unchanged
    assert np.array_equal(full[0:331, 1:361], interior)


def test_the_fold_row_is_filled_from_the_row_below():
    m = _mod()
    interior = np.random.default_rng(1).normal(size=(331, 360)) + 10.0
    full = m._scrip_to_full_tripole(interior, 332, 362)
    assert np.array_equal(full[331], full[330])


# --- the selector -----------------------------------------------------------

def test_unknown_remap_raises():
    m = _mod()
    with pytest.raises(SystemExit, match="unknown chl_remap"):
        m.load_nemo_chl_monthly(None, "tripole", np.zeros((332, 362)),
                                np.zeros((332, 362)), chl_remap="nearest")


def test_scrip_refuses_a_grid_it_was_not_built_for():
    """The weights encode ONE destination grid. Remapping a different grid
    through them would produce a plausible, wrong field rather than an error,
    so the shape check must refuse."""
    m = _mod()
    with pytest.raises(SystemExit, match="eORCA1 tripole"):
        m.load_nemo_chl_monthly(None, "fesom", np.zeros((100, 200)),
                                np.zeros((100, 200)), chl_remap="nemo_scrip")


def test_default_is_still_idw_so_nothing_moved():
    """The signature default must remain the pre-existing behaviour."""
    import inspect
    sig = inspect.signature(_mod().load_nemo_chl_monthly)
    assert sig.parameters["chl_remap"].default == "idw"


def test_the_weights_path_points_at_a_readable_file():
    """codex has twice caught paths that were broken symlinks in RUN_GATEWAY;
    the INPUTS copies are the real ones."""
    p = pathlib.Path(_mod()._CHL_SCRIP_WEIGHTS)
    if not p.parent.exists():
        pytest.skip("oracle tree not present")
    assert p.is_file() and p.stat().st_size > 0

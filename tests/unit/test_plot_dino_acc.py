"""Unit tests for the DINO barotropic-Psi / ACC-transport diagnostic.

The transport integral itself is the shared, partial-cell-aware
``ocean.diagnostics_streamfunction.barotropic_streamfunction`` +
``ocean.diagnostics_climate.acc_transport`` (each separately tested); the only
new local logic is the partial-cell thickness reconstruction from the
snapshot's bottom depth + z* reference grid.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from legoesm.ocean.diagnostics_streamfunction import partial_cell_thickness

_SPEC = importlib.util.spec_from_file_location(
    "plot_dino_acc",
    Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_dino_acc.py")
pda = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pda)


def test_partial_cell_full_column():
    """A column deeper than the deepest interface keeps full thicknesses."""
    dz = np.array([10.0, 20.0, 30.0, 40.0])    # interfaces at 10,30,60,100
    h = partial_cell_thickness(np.full((2, 3), 1000.0), dz)
    assert h.shape == (2, 3, 4)
    np.testing.assert_allclose(h, np.broadcast_to(dz, (2, 3, 4)))


def test_partial_cell_clipped_bottom():
    """A floor mid-cell clips that level and zeros everything below."""
    dz = np.array([10.0, 20.0, 30.0, 40.0])    # z_bot = 10,30,60,100
    h = partial_cell_thickness(np.full((1, 1), 45.0), dz)[0, 0]
    np.testing.assert_allclose(h, [10.0, 20.0, 15.0, 0.0])   # 45-30 = 15 in level 2


def test_partial_cell_land_column_zero():
    dz = np.array([10.0, 20.0, 30.0])
    h = partial_cell_thickness(np.zeros((1, 1)), dz)[0, 0]
    np.testing.assert_allclose(h, [0.0, 0.0, 0.0])


def test_partial_cell_conserves_depth():
    """Summed active thickness equals the (capped) water-column depth."""
    dz = np.array([10.0, 20.0, 30.0, 40.0])
    for H in (5.0, 25.0, 70.0, 100.0, 250.0):
        col = partial_cell_thickness(np.full((1, 1), H), dz)[0, 0]
        np.testing.assert_allclose(col.sum(), min(H, dz.sum()))

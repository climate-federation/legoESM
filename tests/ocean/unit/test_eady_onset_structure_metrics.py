"""The onset-structure indices must give the KNOWN answer on known fields.

These two numbers are meant to separate a 2*dx null-space mode from a
resolved baroclinic mode in the eady_uniform/mpas_channel blow-up. A metric
whose name sounds right and whose behaviour is unverified is exactly how this
repo has manufactured confident wrong physics before, so both are pinned here
against fields whose answer is known by construction.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = (_REPO / "scripts" / "validate" / "ocean_fidelity"
           / "eady_channel_onset_structure.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_onset", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_onset"] = m
    spec.loader.exec_module(m)
    return m


def _ring(n):
    """A 1-D periodic ring of n cells: edge j joins cell j and cell j+1.

    Simplest connectivity with the same (2, nEdges) layout as the real mesh,
    and one where a chequerboard is unambiguous for even n.
    """
    j = np.arange(n)
    return np.stack([j, (j + 1) % n])


def test_a_perfect_chequerboard_scores_plus_one(mod):
    n = 40
    eta = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    idx, used = mod.chequerboard_index(eta, _ring(n))
    assert idx == pytest.approx(1.0)
    assert used == n


def test_a_smooth_single_wave_scores_negative(mod):
    """One long wave around the ring: neighbours almost always agree."""
    n = 40
    eta = np.cos(2 * np.pi * np.arange(n) / n)
    idx, _ = mod.chequerboard_index(eta, _ring(n))
    assert idx < -0.8, idx


def test_a_constant_sign_field_scores_minus_one(mod):
    n = 40
    idx, _ = mod.chequerboard_index(np.ones(n), _ring(n))
    assert idx == pytest.approx(-1.0)


def test_dry_neighbours_are_dropped(mod):
    """A land cell holds a constant and would bias the index toward it."""
    n = 40
    eta = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    wet = np.ones(n, dtype=bool)
    wet[: n // 2] = False
    idx, used = mod.chequerboard_index(eta, _ring(n), wet)
    assert idx == pytest.approx(1.0)
    assert used < n, "edges touching dry cells must be excluded"


def test_concentration_separates_a_hot_cell_from_a_broad_mode(mod):
    n = 100
    spike = np.zeros(n); spike[7] = 1.0
    broad = np.cos(2 * np.pi * np.arange(n) / n)
    c_spike, _ = mod.concentration(spike)
    c_broad, _ = mod.concentration(broad)
    assert c_spike == pytest.approx(0.01), "one cell in a hundred"
    assert c_broad > 0.5, c_broad
    assert c_spike < c_broad


def test_an_all_zero_field_is_nan_not_zero(mod):
    """A dead field must not read as 'perfectly concentrated'."""
    c, _ = mod.concentration(np.zeros(20))
    assert np.isnan(c)

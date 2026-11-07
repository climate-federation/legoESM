"""MPAS SPMD restarts record their edge layout; a restart from another layout
is refused on load (its velocity would be silently scrambled)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.run import run_omip


def _npz(tmp_path, **extra):
    p = tmp_path / "r.npz"
    np.savez(p, time_days=1.0, step=10, grid_type="mpas", **extra)
    return str(p)


def test_legacy_spmd_restart_without_edge_layout_is_refused(tmp_path):
    path = _npz(tmp_path, mpas_spmd_n_devices=4)
    with pytest.raises(ValueError, match="edges in the 'owner' layout"):
        run_omip._load_restart(path, None, grid_type="mpas", mpas_spmd_n_devices=4)


def test_other_edge_layout_is_refused(tmp_path):
    path = _npz(tmp_path, mpas_spmd_n_devices=4, mpas_spmd_edge_order="hilbert")
    with pytest.raises(ValueError, match="'hilbert' layout"):
        run_omip._load_restart(path, None, grid_type="mpas", mpas_spmd_n_devices=4)


def test_matching_layout_and_serial_restarts_pass_the_guard(tmp_path):
    """Past the guard the template is dereferenced (None here), so reaching
    that AttributeError/TypeError proves the guard let the file through."""
    ok = _npz(tmp_path, mpas_spmd_n_devices=4,
              mpas_spmd_edge_order=run_omip.MPAS_SPMD_EDGE_ORDER)
    serial = str(tmp_path / "s.npz")
    np.savez(serial, time_days=1.0, step=10, grid_type="mpas", mpas_spmd_n_devices=0)
    for path, nd in ((ok, 4), (serial, 0)):
        with pytest.raises((AttributeError, TypeError)):
            run_omip._load_restart(path, None, grid_type="mpas", mpas_spmd_n_devices=nd)

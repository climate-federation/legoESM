"""#1442: a snapshot must carry the tracer-advecting mass flux when captured.

`state.u` is the velocity BEFORE the barotropic transport correction
`delta_U = (Hu_avg - Hu_3d)/H_u_old`. That correction reaches the tracer flux
and never reaches `state.u`, so an offline diagnostic rebuilding `h * u` from a
snapshot is missing a depth-uniform mode — measured at 0.35-1.28 Sv on eORCA1,
~100% of the apparent net transport at 66N.

#1440 made the corrected flux available in state (`store_mass_flux`); these
tests pin that the snapshot writer actually persists it, and that a run WITHOUT
the capture still writes a loadable snapshot.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "_run_omip_core2_snap", REPO / "scripts" / "run" / "run_omip_core2.py")


def _load():
    mod = importlib.util.module_from_spec(_SPEC)
    sys.modules["_run_omip_core2_snap"] = mod
    _SPEC.loader.exec_module(mod)
    return mod


def _field(a):
    return SimpleNamespace(data=np.asarray(a))


def _state(n_lat=4, n_lon=6, nlev=2, with_flux=True):
    st = SimpleNamespace(
        T=_field(np.zeros((n_lat, n_lon, nlev))),
        S=_field(np.zeros((n_lat, n_lon, nlev))),
        u=_field(np.ones((n_lat, n_lon + 1, nlev))),
        v=_field(np.ones((n_lat + 1, n_lon, nlev))),
        eta=_field(np.zeros((n_lat, n_lon))),
        land_mask=_field(np.ones((n_lat, n_lon))),
        mass_flux_u=None, mass_flux_v=None, mass_flux_w=None,
    )
    if with_flux:
        # DIFFERENT from h*u on purpose: the correction is what makes them differ.
        st.mass_flux_u = _field(np.full((n_lat, n_lon + 1, nlev), 7.0))
        st.mass_flux_v = _field(np.full((n_lat + 1, n_lon, nlev), 5.0))
        st.mass_flux_w = _field(np.full((n_lat, n_lon, nlev + 1), 3.0))
    return st


def test_snapshot_carries_the_captured_mass_flux(tmp_path):
    mod = _load()
    st = _state(with_flux=True)
    lat2d = np.zeros((4, 6)); lon2d = np.zeros((4, 6))
    mod._save_snapshot(tmp_path, "d10", st, lat2d, lon2d)
    npz = np.load(next(tmp_path.glob("*.npz")))
    for k, want in (("mass_flux_u", 7.0), ("mass_flux_v", 5.0),
                    ("mass_flux_w", 3.0)):
        assert k in npz.files, (
            f"{k} missing — an offline transport diagnostic is back to "
            f"rebuilding h*u from the UNCORRECTED state.u (#1442)")
        assert float(npz[k].max()) == want
    # And it is genuinely different from what h*u would give.
    assert not np.allclose(npz["mass_flux_u"], npz["u"])


def test_snapshot_without_capture_is_still_written_and_loadable(tmp_path):
    """store_mass_flux off: no extra keys, old readers unaffected."""
    mod = _load()
    st = _state(with_flux=False)
    lat2d = np.zeros((4, 6)); lon2d = np.zeros((4, 6))
    mod._save_snapshot(tmp_path, "d10", st, lat2d, lon2d)
    npz = np.load(next(tmp_path.glob("*.npz")))
    assert "mass_flux_u" not in npz.files
    for k in ("T", "S", "u", "v", "land_mask", "lat_T", "lon_T"):
        assert k in npz.files


class _GridWithMetrics:
    def __init__(self, n_lat=4, n_lon=6):
        self.dy_u = np.full((n_lat, n_lon + 1), 1.1e5)
        self.dx_v = np.full((n_lat + 1, n_lon), 2.2e5)
        self.dx_u = np.full((n_lat, n_lon + 1), 3.3e5)
        self.dy_v = np.full((n_lat + 1, n_lon), 4.4e5)


def test_face_metrics_are_saved_with_the_flux(tmp_path):
    """The flux is m^2/s; without dy_u/dx_v a reader cannot reach m^3/s."""
    mod = _load()
    st = _state(with_flux=True)
    lat2d = np.zeros((4, 6)); lon2d = np.zeros((4, 6))
    mod._save_snapshot(tmp_path, "d10", st, lat2d, lon2d,
                       grid=_GridWithMetrics())
    npz = np.load(next(tmp_path.glob("*.npz")))
    for k, want in (("dy_u", 1.1e5), ("dx_v", 2.2e5)):
        assert k in npz.files, (
            f"{k} missing — mass_flux_* is thickness-weighted velocity, so a "
            f"reader cannot integrate it to m^3/s without the face widths")
        assert float(npz[k].max()) == want


def test_face_metrics_are_not_saved_without_a_flux(tmp_path):
    """No capture -> no extra keys at all, even with a grid passed."""
    mod = _load()
    st = _state(with_flux=False)
    lat2d = np.zeros((4, 6)); lon2d = np.zeros((4, 6))
    mod._save_snapshot(tmp_path, "d10", st, lat2d, lon2d,
                       grid=_GridWithMetrics())
    npz = np.load(next(tmp_path.glob("*.npz")))
    assert "dy_u" not in npz.files and "mass_flux_u" not in npz.files


def test_missing_grid_metrics_do_not_break_the_snapshot(tmp_path):
    """A regular lat-lon grid may not expose dy_u/dx_v — must not raise."""
    mod = _load()
    st = _state(with_flux=True)
    lat2d = np.zeros((4, 6)); lon2d = np.zeros((4, 6))
    mod._save_snapshot(tmp_path, "d10", st, lat2d, lon2d,
                       grid=SimpleNamespace())
    npz = np.load(next(tmp_path.glob("*.npz")))
    assert "mass_flux_u" in npz.files
    assert "dy_u" not in npz.files

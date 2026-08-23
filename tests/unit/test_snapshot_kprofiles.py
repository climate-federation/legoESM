"""A snapshot must be able to carry the closure's own K_M / K_H.

NEMO publishes ``avm`` and ``avt`` in its five-day output, so the oracle's
turbulent Prandtl number at the equator is a file read. Ours were built inside
the step and never persisted, which left every statement about our equatorial
mixing an inference from TKE and stratification rather than a measurement of
the quantity NEMO publishes.

``--kprofile-snapshots`` asks the model for one tendency evaluation at the
snapshot's own state and stores the ``K_v``/``A_v`` it carries. These tests pin
that the flag exists and is off by default, that the arrays reach the file, and
-- the part that matters for trusting a null result -- that a configuration
producing no diffusivities SAYS SO rather than writing a silently empty
snapshot.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "_run_omip_core2_kprof", REPO / "scripts" / "run" / "run_omip_core2.py")


def _load():
    mod = importlib.util.module_from_spec(_SPEC)
    sys.modules["_run_omip_core2_kprof"] = mod
    _SPEC.loader.exec_module(mod)
    return mod


def _field(a):
    return SimpleNamespace(data=np.asarray(a))


def _state(n_lat=4, n_lon=6, nlev=2):
    return SimpleNamespace(
        T=_field(np.zeros((n_lat, n_lon, nlev))),
        S=_field(np.zeros((n_lat, n_lon, nlev))),
        u=_field(np.ones((n_lat, n_lon + 1, nlev))),
        v=_field(np.ones((n_lat + 1, n_lon, nlev))),
        eta=_field(np.zeros((n_lat, n_lon))),
        land_mask=_field(np.ones((n_lat, n_lon))),
        mass_flux_u=None, mass_flux_v=None, mass_flux_w=None,
    )


class _ZCoord:
    """A z-coordinate stub carrying interface depths (z_half_ref, <=0)."""
    def __init__(self, nlev):
        self.z_half_ref = -np.linspace(0.0, 300.0, nlev + 1)


class _Model:
    """Stands in for the ocean model's diagnose_vertical_K entry point.

    The production TKE closure returns K_v=None on the TENDENCY -- the
    diffusivities are built inside the implicit solve -- so a probe that read
    the tendency would measure nothing for the arm under test. The real path
    is diagnose_vertical_K, and this stub exercises exactly that; a stub with
    a `tendencies` method but no diagnose_vertical_K must be reported skipped,
    which is what the real TKE model would do if the method were removed.
    """
    def __init__(self, K_H, K_M, raises=False):
        self._K_H, self._K_M, self._raises = K_H, K_M, raises
        self.calls = []

    def diagnose_vertical_K(self, state, dt, surface_forcing=None):
        if self._raises:
            raise RuntimeError("solve setup failed on this configuration")
        self.calls.append((dt, surface_forcing))
        return (None if self._K_H is None else _field(self._K_H),
                None if self._K_M is None else _field(self._K_M))


def test_the_flag_is_off_by_default_and_parses():
    mod = _load()
    p = mod._build_arg_parser()
    base = ["--grid", "tripole"]
    assert p.parse_args(base).kprofile_snapshots is False
    assert p.parse_args(base + ["--kprofile-snapshots"]).kprofile_snapshots


def test_kprofiles_reads_the_solves_own_diffusivities_and_true_depths():
    """The probe must call diagnose_vertical_K -- the K the implicit solve
    consumes, not the tendency's None -- at the snapshot's own state and dt,
    and must store the interior interface depths, not a cell-centre guess."""
    mod = _load()
    nlev = 3
    K_H = np.full((4, 6, nlev - 1), 1.5e-3)
    K_M = np.full((4, 6, nlev - 1), 3.0e-3)
    m = _Model(K_H, K_M)
    out = mod._kprofiles(m, _state(nlev=nlev), "SF", 150.0, _ZCoord(nlev))
    assert set(out) == {"K_H_diag", "K_M_diag", "z_interface_ref"}
    assert np.allclose(out["K_H_diag"], K_H)
    assert np.allclose(out["K_M_diag"], K_M)
    # interior interfaces only (drop surface and seafloor), positive-down
    assert out["z_interface_ref"].size == nlev - 1
    assert np.all(out["z_interface_ref"] > 0)
    assert m.calls == [(150.0, "SF")]


def test_a_model_without_the_diagnostic_method_is_skipped(capsys):
    """A grid whose model has no diagnose_vertical_K (e.g. MPAS) must be
    reported skipped rather than crashing or silently writing nothing."""
    mod = _load()
    m = SimpleNamespace(tendencies=lambda *a, **k: None)   # no diagnose method
    assert mod._kprofiles(m, _state(), None, 150.0, _ZCoord(2)) == {}
    assert "SKIPPED" in capsys.readouterr().out


def test_a_failed_evaluation_does_not_kill_the_run(capsys):
    """A diagnostic must never be able to end an integration -- a freshwater
    band print that read a field one grid did not have killed every MPAS run
    for three days."""
    mod = _load()
    m = _Model(None, None, raises=True)
    assert mod._kprofiles(m, _state(), None, 150.0, _ZCoord(2)) == {}
    assert "SKIPPED" in capsys.readouterr().out


def test_the_arrays_reach_the_file_and_absence_leaves_it_loadable(tmp_path):
    mod = _load()
    st = _state()
    lat2d = np.zeros((4, 6)); lon2d = np.zeros((4, 6))
    K_H = np.full((4, 6, 1), 1.5e-3)
    K_M = np.full((4, 6, 1), 3.0e-3)

    mod._save_snapshot(tmp_path, "with", st, lat2d, lon2d,
                       extra={"K_H_diag": K_H, "K_M_diag": K_M})
    z = np.load(tmp_path / "snapshot_with.npz")
    assert np.allclose(z["K_H_diag"], K_H) and np.allclose(z["K_M_diag"], K_M)

    # Same writer, no extras: the keys must simply be absent, not zero-filled,
    # so a reader can tell "not measured" from "measured as zero".
    mod._save_snapshot(tmp_path, "without", st, lat2d, lon2d, extra=None)
    z2 = np.load(tmp_path / "snapshot_without.npz")
    assert "K_H_diag" not in z2 and "K_M_diag" not in z2
    assert np.allclose(z2["T"], np.asarray(st.T.data))

    # A None-valued extra is dropped rather than written as an object array,
    # which is what `np.savez` would otherwise do to it.
    mod._save_snapshot(tmp_path, "partial", st, lat2d, lon2d,
                       extra={"K_H_diag": K_H, "K_M_diag": None})
    z3 = np.load(tmp_path / "snapshot_partial.npz")
    assert "K_H_diag" in z3 and "K_M_diag" not in z3

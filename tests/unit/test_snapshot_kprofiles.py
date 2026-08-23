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


class _Model:
    """Stands in for the ocean model's public ``tendencies`` entry point."""

    def __init__(self, K_v, A_v, raises=False):
        self._K_v, self._A_v, self._raises = K_v, A_v, raises
        self.calls = []

    def tendencies(self, state, surface_forcing=None, dt=None):
        if self._raises:
            raise RuntimeError("no physics on this configuration")
        self.calls.append((surface_forcing, dt))
        return SimpleNamespace(
            K_v=None if self._K_v is None else _field(self._K_v),
            A_v=None if self._A_v is None else _field(self._A_v))


def test_the_flag_is_off_by_default_and_parses():
    mod = _load()
    p = mod._build_arg_parser()
    base = ["--grid", "tripole"]
    assert p.parse_args(base).kprofile_snapshots is False
    assert p.parse_args(base + ["--kprofile-snapshots"]).kprofile_snapshots


def test_kprofiles_returns_the_models_own_diffusivities():
    """The probe must read the SAME tendency the step consumes, and must pass
    the snapshot's own state and timestep to it -- a diagnostic evaluated at a
    different state or dt is not the run's mixing."""
    mod = _load()
    K_H = np.full((4, 6, 1), 1.5e-3)
    K_M = np.full((4, 6, 1), 3.0e-3)
    m = _Model(K_H, K_M)
    st = _state()
    out = mod._kprofiles(m, st, "SF", 150.0)
    assert set(out) == {"K_H_diag", "K_M_diag"}
    assert np.allclose(out["K_H_diag"], K_H)
    assert np.allclose(out["K_M_diag"], K_M)
    assert m.calls == [("SF", 150.0)]


def test_a_configuration_with_no_diffusivities_says_so(capsys):
    mod = _load()
    assert mod._kprofiles(_Model(None, None), _state(), None, 150.0) == {}
    assert "SKIPPED" in capsys.readouterr().out


def test_a_failed_tendency_evaluation_does_not_kill_the_run(capsys):
    """A diagnostic must never be able to end an integration -- a freshwater
    band print that read a field one grid did not have killed every MPAS run
    for three days."""
    mod = _load()
    assert mod._kprofiles(_Model(None, None, raises=True), _state(), None,
                          150.0) == {}
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

"""CORE-II OMIP KPP boundary-layer-depth override flags (vmix MLD lever).

Covers the two pure helpers behind ``--kpp-ri-crit`` / ``--kpp-cv``:
the override builder (which must leave the production KPPConfig byte-identical
when no knob is set, reject out-of-range knobs, and thread through
``build_mpas_ocean -> _create_setup``) and the mpas-only dispatch guard (fail
loud, never silently ignore a flag).
"""

from __future__ import annotations

import pytest

from scripts.run.run_omip_core2 import (
    _kpp_vmix_override, _validate_kpp_grid,
    _KPP_RI_CRIT_RANGE, _KPP_CV_RANGE,
)
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig


def test_override_none_when_no_knob_set():
    """No knob -> None so the caller uses _create_setup's default KPP config
    (byte-identical to the pre-flag runs)."""
    assert _kpp_vmix_override(None, None) is None


def test_override_sets_ri_crit_only_and_preserves_all_other_fields():
    vm = _kpp_vmix_override(kpp_ri_crit=0.5, kpp_cv=None)
    assert vm is not None and vm.scheme == "kpp"
    # FULL equality: only Ri_crit moved, every other KPP field is the default
    # (guards against a silent change to any other field).
    assert vm.kpp == KPPConfig()._replace(Ri_crit=0.5)


def test_override_sets_cv_only_and_preserves_all_other_fields():
    vm = _kpp_vmix_override(kpp_ri_crit=None, kpp_cv=2.5)
    assert vm.kpp == KPPConfig()._replace(Cv=2.5)


def test_override_sets_both():
    vm = _kpp_vmix_override(kpp_ri_crit=0.45, kpp_cv=2.2)
    assert vm.kpp == KPPConfig()._replace(Ri_crit=0.45, Cv=2.2)


@pytest.mark.parametrize("bad", [0.0, -0.3, float("nan"), float("inf"),
                                 _KPP_RI_CRIT_RANGE[1] + 0.1,   # above hi
                                 _KPP_RI_CRIT_RANGE[0] - 0.01])  # below lo
def test_override_rejects_ri_crit_out_of_range(bad):
    with pytest.raises(ValueError):
        _kpp_vmix_override(kpp_ri_crit=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"),
                                 _KPP_CV_RANGE[1] + 1.0, _KPP_CV_RANGE[0] - 0.1])
def test_override_rejects_cv_out_of_range(bad):
    with pytest.raises(ValueError):
        _kpp_vmix_override(kpp_cv=bad)


def test_override_accepts_range_endpoints():
    lo, hi = _KPP_RI_CRIT_RANGE
    assert _kpp_vmix_override(kpp_ri_crit=lo).kpp.Ri_crit == lo
    assert _kpp_vmix_override(kpp_ri_crit=hi).kpp.Ri_crit == hi


def test_guard_raises_on_nonmpas_grid_when_flag_set():
    for grid in ("tripole", "latlon_bathy", "cubed_sphere"):
        with pytest.raises(SystemExit):
            _validate_kpp_grid(grid, kpp_ri_crit=0.5, kpp_cv=None)
        with pytest.raises(SystemExit):
            _validate_kpp_grid(grid, kpp_ri_crit=None, kpp_cv=2.5)


def test_guard_noop_on_mpas_or_no_flags():
    _validate_kpp_grid("mpas", kpp_ri_crit=0.5, kpp_cv=2.5)   # mpas+flags: ok
    for grid in ("tripole", "latlon_bathy", "cubed_sphere", "mpas"):
        _validate_kpp_grid(grid, None, None)                  # no flags: ok


def test_build_mpas_ocean_threads_override_to_create_setup(monkeypatch):
    """Integration: build_mpas_ocean MUST forward ``vertical_mixing`` to
    _create_setup (a future dropped kwarg would otherwise pass the pure-helper
    tests).  Short-circuit the heavy mesh/regrid machinery with a sentinel."""
    import scripts.run.run_omip_core2 as core2
    from scripts.run import run_omip

    captured = {}

    class _Stop(Exception):
        pass

    def _fake_create_setup(*a, **k):
        captured["vm"] = k.get("vertical_mixing")
        raise _Stop

    monkeypatch.setattr(run_omip, "_create_setup", _fake_create_setup)
    vm = _kpp_vmix_override(0.5, 2.5)
    with pytest.raises(_Stop):
        core2.build_mpas_ocean(75, 6000.0, "dummy_mesh.nc", level=7,
                               vertical_mixing=vm)
    assert captured["vm"] is vm
    assert captured["vm"].kpp.Ri_crit == 0.5
    assert captured["vm"].kpp.Cv == 2.5

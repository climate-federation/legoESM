"""CORE-II OMIP KPP boundary-layer-depth override flags (vmix MLD lever).

Covers the two pure helpers behind ``--kpp-ri-crit`` / ``--kpp-cv``:
the override builder (which must leave the production KPPConfig byte-identical
when no knob is set, reject out-of-range knobs, and thread through
``build_mpas_ocean``/``build_latlon_bathy -> _create_setup``) and the dispatch
guard, which allows the KPP-running grids (mpas, latlon_bathy) and fails loud on
grids that would silently ignore the flag (tripole = no KPP, cubed_sphere).
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


def test_guard_raises_on_non_kpp_grid_when_flag_set():
    # tripole ships physics=None (dynamics-core implicit vmix, no KPP boundary
    # layer) and cubed_sphere is not wired -> the override would silently do
    # nothing, so the guard must still fail loud on those.
    for grid in ("tripole", "cubed_sphere"):
        with pytest.raises(SystemExit):
            _validate_kpp_grid(grid, kpp_ri_crit=0.5, kpp_cv=None)
        with pytest.raises(SystemExit):
            _validate_kpp_grid(grid, kpp_ri_crit=None, kpp_cv=2.5)
    # The rejection must point tripole users at the actionable alternative
    # (--tripole-vmix selects that lane's opt-in closure).
    with pytest.raises(SystemExit, match="--tripole-vmix"):
        _validate_kpp_grid("tripole", kpp_ri_crit=0.5, kpp_cv=None)


def test_guard_noop_on_kpp_grids_or_no_flags():
    # mpas AND latlon_bathy run the KPP boundary layer -> override is live -> ok.
    _validate_kpp_grid("mpas", kpp_ri_crit=0.5, kpp_cv=2.5)
    _validate_kpp_grid("latlon_bathy", kpp_ri_crit=0.5, kpp_cv=2.5)
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


def test_build_latlon_bathy_threads_override_to_create_setup(monkeypatch):
    """Integration: build_latlon_bathy MUST forward ``vertical_mixing`` to
    _create_setup so the KPP Ri_crit/Cv override reaches the live KPP scheme on
    the lat-lon bathy path (the NH-midlat winter-MLD lever).  A dropped kwarg
    would pass the pure-helper tests but silently ignore the flag."""
    import scripts.run.run_omip_core2 as core2
    from scripts.run import run_omip

    captured = {}

    class _Stop(Exception):
        pass

    def _fake_create_setup(*a, **k):
        captured["vm"] = k.get("vertical_mixing")
        raise _Stop

    monkeypatch.setattr(run_omip, "_create_setup", _fake_create_setup)
    vm = _kpp_vmix_override(0.2, None)          # shoal the too-deep winter ML
    with pytest.raises(_Stop):
        core2.build_latlon_bathy(75, 6000.0, "dummy_mesh.nc",
                                 vertical_mixing=vm)
    assert captured["vm"] is vm
    assert captured["vm"].kpp.Ri_crit == 0.2


@pytest.mark.parametrize("knob", [
    dict(kpp_ri_crit=0.5), dict(kpp_cv=2.5), dict(kpp_eice=3),
])
def test_guard_rejects_each_kpp_knob_under_mpas_tke(knob):
    """--mpas-vmix tke deselects KPP, so EVERY --kpp-* knob must fail loud on
    MPAS under it (parametrized per knob: a guard that checks only one of the
    three would pass a single combined test; codex MED 2026-07-27)."""
    kwargs = dict(kpp_ri_crit=None, kpp_cv=None, kpp_eice=None)
    kwargs.update(knob)
    with pytest.raises(SystemExit, match="mpas-vmix"):
        _validate_kpp_grid("mpas", mpas_vmix="tke", **kwargs)
    if "kpp_eice" in knob:
        # --kpp-eice is rejected on MPAS even under KPP (the MPAS KPP bridge
        # receives no ice concentration), just with the eice-specific message
        # rather than the closure-mismatch one.
        with pytest.raises(SystemExit, match="kpp-eice"):
            _validate_kpp_grid("mpas", mpas_vmix="kpp", **kwargs)
    else:
        # ri_crit/cv stay accepted under the default KPP closure.
        _validate_kpp_grid("mpas", mpas_vmix="kpp", **kwargs)


def test_mpas_tke_callsite_pins_diagnostic_card():
    """The main() MPAS call-site must pin tke_prognostic=False: post-#1326
    the ORCA1 card defaults to the prognostic carry, which MPAS rejects (no
    MPASOceanState.tke yet), so silently dropping the pin would crash every
    --mpas-vmix tke run at build. Source tripwire + behavior of the pinned
    expression (codex MED 2026-07-27)."""
    import inspect
    import re

    import scripts.run.run_omip_core2 as core2

    src = inspect.getsource(core2)
    m = re.search(
        r'build_tripole_vmix_config\("tke",\s*iwm=None,\s*'
        r'tke_prognostic=False\)\s*\n\s*if args\.mpas_vmix == "tke"', src)
    assert m, "--mpas-vmix tke call-site no longer pins tke_prognostic=False"
    # The pinned expression yields the diagnostic ORCA1 card with the rest of
    # the #1326 defaults intact (Mode-B + nn_mxl=3 + Dirichlet BC + nn_eice=3).
    vm = core2.build_tripole_vmix_config("tke", iwm=None,
                                         tke_prognostic=False)
    assert vm.scheme == "tke"
    assert vm.tke.prognostic is False
    assert vm.tke.tke_mxl_choice == 3
    assert vm.tke.surface_bc == "nemo_dirichlet"
    assert vm.tke.eice == 3

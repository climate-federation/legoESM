"""zdf_sh2's sea level under NEMO RK3: the whole-step entry level (V1-selected).

stprk3.f90:185 calls zdf_phy(kstp, Nbb, Nbb, Nrhs), so zdfsh2.f90:98-108's
e3w_1d*(1+r3u(Kmm)) uses the step-entry ssh, not the stage Kmm ssh that
dyn_zdf/tra_zdf read.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _nemo_sh2_eta
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

STAGE, ENTRY = jnp.asarray([0.3]), jnp.asarray([0.1])


def test_default_keeps_the_stage_eta_on_every_other_card():
    assert _nemo_sh2_eta(TKEConfig(), prognostic=True, eta_now=STAGE,
                         eta_step_entry=ENTRY) is STAGE


def test_selected_flag_reads_the_whole_step_entry_eta():
    cfg = TKEConfig(nemo_sh2_step_entry_ssh=True)
    assert _nemo_sh2_eta(cfg, prognostic=True, eta_now=STAGE,
                         eta_step_entry=ENTRY) is ENTRY
    # no stage eta at all: the caller's own state.eta is the entry level
    assert _nemo_sh2_eta(cfg, prognostic=True, eta_now=None,
                         eta_step_entry=None) is None


def test_selected_flag_refuses_paths_that_would_ignore_it():
    cfg = TKEConfig(nemo_sh2_step_entry_ssh=True)
    with pytest.raises(ValueError, match="needs the whole-step entry eta"):
        _nemo_sh2_eta(cfg, prognostic=True, eta_now=STAGE, eta_step_entry=None)
    with pytest.raises(ValueError, match="requires prognostic TKE"):
        _nemo_sh2_eta(cfg, prognostic=False, eta_now=STAGE, eta_step_entry=ENTRY)


class _ShearSeen(Exception):
    pass


def _shear_eta_at_call_site(flag):
    """Run the production vertical-mixing method up to its shear call and
    return the eta it hands ``_tke_step_entry_p_sh2`` (review gap: the call
    site could bypass the helper and every unit test above stayed green)."""
    from types import MethodType, SimpleNamespace

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    tke = TKEConfig(prognostic=True, tke_shear_production="nemo_face_native",
                    nemo_sh2_step_entry_ssh=flag)
    physics = SimpleNamespace(
        vertical_mixing=SimpleNamespace(scheme="tke", tke=tke),
        convection=SimpleNamespace(scheme="none"))
    cfg = SimpleNamespace(physics=physics, eos="linear", eos_linear=None,
                          eos_nemo_seos=None, implicit_ice_ocean_top_drag=False)

    def spy(self, state, *, eta_now, **kwargs):
        raise _ShearSeen(eta_now)

    model = SimpleNamespace(
        config=cfg, z_coord=None, _iwm_forcing=None,
        grid=SimpleNamespace(lat_T=jnp.zeros((1, 1))),
        _nemo_ws_test_hooks=SimpleNamespace(zdf_momentum_observer=None),
        _tke_post_mixing_active=lambda: False,
        _lat_dependent_constant_vmix=lambda: False)
    model._tke_step_entry_p_sh2 = MethodType(spy, model)
    with pytest.raises(_ShearSeen) as seen:
        LatLonCGridOceanModel._apply_implicit_vertical_mixing(
            model, SimpleNamespace(), 1.0, None,
            eta_now=STAGE, eta_step_entry=ENTRY)
    return seen.value.args[0]


def test_production_shear_call_site_routes_through_the_selector():
    assert _shear_eta_at_call_site(True) is ENTRY
    assert _shear_eta_at_call_site(False) is STAGE


class _LatSeen(Exception):
    pass


def _htau_lat_at_call_site(evaluation, monkeypatch):
    """Run the production vertical-mixing method up to its closure call and
    return the latitude it hands it for htau (zdftke.f90:864 reads gphit;
    round-7 review gap: the V1 arm could fall back to degrees(lat_T))."""
    from types import SimpleNamespace

    import legoesm.ocean.physics.vertical_mixing as vm
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    def spy(*args, lat_deg=None, **kwargs):
        raise _LatSeen(lat_deg)

    monkeypatch.setattr(vm, "compute_vertical_K_profiles", spy)
    tke = TKEConfig(tke_htau_evaluation=evaluation,
                    tke_shear_production="nemo_face_native")
    physics = SimpleNamespace(
        vertical_mixing=SimpleNamespace(scheme="tke", tke=tke),
        convection=SimpleNamespace(scheme="none"))
    cfg = SimpleNamespace(physics=physics, eos="linear", eos_linear=None,
                          eos_nemo_seos=None, implicit_ice_ocean_top_drag=False,
                          A_v=1e-4, K_v=1e-5)
    model = SimpleNamespace(
        config=cfg, z_coord=None, _iwm_forcing=None,
        grid=SimpleNamespace(lat_T=jnp.asarray([[0.5]]),
                             native_lat_T_deg=GPHIT),
        _nemo_ws_test_hooks=SimpleNamespace(zdf_momentum_observer=None),
        _tke_post_mixing_active=lambda: False,
        _lat_dependent_constant_vmix=lambda: False)
    with pytest.raises(_LatSeen) as seen:
        LatLonCGridOceanModel._apply_implicit_vertical_mixing(
            model, SimpleNamespace(), 1.0, None)
    return seen.value.args[0]


GPHIT = jnp.asarray([[28.0]])
x64 = pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")


@x64
@pytest.mark.parametrize("evaluation", ["nemo_literal", "nemo_scalar_libm"])
def test_production_htau_arms_read_native_gphit(evaluation, monkeypatch):
    assert _htau_lat_at_call_site(evaluation, monkeypatch) is GPHIT


@x64
def test_default_htau_keeps_degrees_of_lat_t(monkeypatch):
    lat = _htau_lat_at_call_site("jax_expression", monkeypatch)
    assert lat is not GPHIT and float(lat[0, 0]) == float(jnp.degrees(0.5))

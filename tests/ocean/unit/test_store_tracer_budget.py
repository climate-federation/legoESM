"""store_tracer_budget: the per-step temperature record on the lat-lon C-grid lane.

Off: no record and the step is unchanged.  On: the explicit tendency split by
stage sums to the total explicit tendency, the record's final T is the state's,
and lanes whose applied flux is not a single-stage pair are refused.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_shortwave_single_owner as box  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from legoesm.ocean.state import OceanSurfaceForcing  # noqa: E402


def _step(**cfg_kw):
    grid, zc_pc, st, lm = box._box()
    cfg = box._cfg(True)._replace(**cfg_kw)
    sf = OceanSurfaceForcing(tau_x=jnp.full_like(st.eta.data, 0.05) * lm,
                             tau_y=jnp.zeros_like(st.eta.data),
                             q_net=jnp.asarray(box.Q * lm))
    return st, LatLonCGridOceanModel(grid, zc_pc, cfg).step(st, box.DT, surface_forcing=sf)


def test_off_by_default_keeps_no_record():
    _, new = _step()
    assert new.tbudget is None


def test_record_stages_sum_to_total_and_final_T_matches():
    st, new = _step(store_tracer_budget=True)
    tb = new.tbudget
    stages = sum(np.asarray(tb["R"][k]) for k in ("pre", "physics", "surface", "sponge"))
    np.testing.assert_allclose(stages, np.asarray(tb["R_total"]), rtol=1e-12, atol=1e-18)
    assert np.abs(np.asarray(tb["R"]["surface"])).max() > 0.0
    np.testing.assert_array_equal(np.asarray(tb["T1"]), np.asarray(new.T.data))
    np.testing.assert_array_equal(np.asarray(tb["T0"]), np.asarray(st.T.data))
    _, ref = _step()
    np.testing.assert_array_equal(np.asarray(new.T.data), np.asarray(ref.T.data))


def test_record_closes_the_explicit_and_advective_heat_content_change():
    # Wind-driven flow over a stratified box: nonzero transports, so the recorded divergences carry the
    # advective change h1*T_adv - h0*T_mid exactly, and the explicit stages carry T_mid - T0.
    st, new = _step(store_tracer_budget=True)
    tb = {k: (np.asarray(v) if not isinstance(v, dict) else v) for k, v in new.tbudget.items()}
    act = tb["active"] > 0.5
    div = tb["div_h"] + tb["div_z"]
    assert np.abs(tb["div_z"][act]).max() > 0.0 and np.abs(tb["div_h"][act]).max() > 0.0
    np.testing.assert_allclose(tb["div_x"] + tb["div_y"], tb["div_h"], rtol=1e-10, atol=1e-18)
    lhs = tb["h1"] * tb["T_adv"]
    rhs = tb["h0"] * tb["T_mid"] - box.DT * div
    np.testing.assert_allclose(lhs[act], rhs[act], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose((tb["T_mid"] - tb["T0"])[act], box.DT * tb["R_total"][act], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kw", [dict(tracer_time_integrator="ab2"), dict(tracer_advection="fct2")])
def test_refused_where_the_applied_flux_is_not_one_stage(kw):
    with pytest.raises(ValueError, match="store_tracer_budget"):
        _step(store_tracer_budget=True, **kw)

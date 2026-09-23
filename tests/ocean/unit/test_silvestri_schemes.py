"""Silvestri scheme presets: each of the 5 baroclinic-jet schemes maps to a
valid, constructible LatLonCGridOceanConfig and selects the intended blocks."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
from legoesm.ocean.experiments.silvestri_schemes import (
    SILVESTRI_JET_SCHEMES, SILVESTRI_JET_MAIN, apply_silvestri_scheme, scheme_label,
)


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _recipe(**kw):
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    return build_eady_uniform_setup(n_lat=12, n_lon=12, **kw)


def test_all_five_main_schemes_present():
    assert set(SILVESTRI_JET_MAIN) == {"UP3", "W9V", "W9D", "SM2", "QG2"}
    for s in SILVESTRI_JET_MAIN:
        assert s in SILVESTRI_JET_SCHEMES


@pytest.mark.parametrize("scheme", SILVESTRI_JET_MAIN)
def test_scheme_builds_and_validates(scheme):
    """Each preset applied to a base config constructs the model (passes all the
    fail-loud validations: momentum_advection, weno_smoothness, flux scheme,
    lateral_friction_scheme, double-friction guard)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    r = _recipe()
    cfg = apply_silvestri_scheme(r.model_config, scheme)
    # No explicit viscosity left on (the schemes are self-dissipating / use a
    # dedicated closure) — so the double-friction guard cannot trip.
    assert cfg.lateral_viscosity.A_h == 0.0 and cfg.lateral_viscosity.B_h == 0.0 and cfg.lateral_viscosity.C_smag == 0.0 and cfg.lateral_viscosity.C_leith == 0.0
    LatLonCGridOceanModel(r.grid, r.z_coord, cfg)   # constructs + validates


def test_scheme_specifics():
    base = _recipe().model_config
    w9v = apply_silvestri_scheme(base, "W9V")
    assert w9v.momentum_advection == "weno9" and w9v.weno_smoothness == "split"
    w9d = apply_silvestri_scheme(base, "W9D")
    assert w9d.momentum_advection == "weno9" and w9d.weno_smoothness == "standard"
    up3 = apply_silvestri_scheme(base, "UP3")
    assert (up3.momentum_advection == "flux_form"
            and up3.momentum_flux_scheme == "oceananigans_up3")
    sm2 = apply_silvestri_scheme(base, "SM2")
    assert sm2.momentum_advection == "vector_invariant"
    assert sm2.lateral_friction_scheme == "om4p25"
    qg2 = apply_silvestri_scheme(base, "QG2")
    assert qg2.lateral_friction_scheme == "qg_leith" and qg2.qg_leith_coeff == 2.0


def test_unknown_scheme_raises():
    base = _recipe().model_config
    with pytest.raises(ValueError, match="unknown Silvestri jet scheme"):
        apply_silvestri_scheme(base, "BOGUS")


def test_qg2_uses_full_stretching():
    """QG2 is now the FULL QG-Leith with the baroclinic stretching term (B5b)."""
    base = _recipe().model_config
    qg2 = apply_silvestri_scheme(base, "QG2")
    assert qg2.qg_leith_stretching is True
    assert scheme_label("QG2") == "QG2"      # faithful — no longer "barotropic"
    assert scheme_label("W9V") == "W9V"


@pytest.mark.parametrize("scheme", SILVESTRI_JET_MAIN)
def test_full_step_finite(scheme):
    """One forward step under each scheme from the Eady IC stays finite."""
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    r = _recipe()
    cfg = apply_silvestri_scheme(r.model_config, scheme)
    model = LatLonCGridOceanModel(r.grid, r.z_coord, cfg)
    s = r.initial_state

    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S),
                   u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    nxt = model.step(s, 300.0)
    assert bool(jnp.all(jnp.isfinite(nxt.u.data))), scheme
    assert bool(jnp.all(jnp.isfinite(nxt.T.data))), scheme

"""Two-leaf canopy closure with the P-hydro instantaneous optimum.

``stomatal_model="phydro"`` appends the soil-to-leaf draw-down of each leaf
class to the Newton state; its residual is the stationarity of the
instantaneous profit ``A - gamma*dpsi^2`` (land/phydro.py).  Checks: the
8-state solve converges on wet, dry and night columns; the draw-down is the
hydraulic state (positive by day, zero at night, smaller gs on dry soil);
the implicit backward pass differentiates the new unknowns (central finite
differences); the 6-state models are untouched.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.solver import (
    CanopyForcingBundle, canopy_forward, solve_canopy_closure, state_size)

needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"), reason="needs JAX_ENABLE_X64=1")
_a = jnp.asarray


def _bundle(APAR_Sun=800.0, APAR_Sh=200.0, psi_s=-0.5, gplant=4e-3, gamma=1.0,
            ASW_Sun=400.0, ASW_Sh=100.0, ASW_Soil=50.0):
    """One midday column; the P-hydro supply fields filled."""
    return CanopyForcingBundle(
        LAI=_a(3.0), SZA=_a(30.0), La=_a(350.0), epsf=_a(0.97), epss=_a(0.96),
        fSun=_a(0.6), APAR_Sun=_a(APAR_Sun), APAR_Sh=_a(APAR_Sh),
        Vcmax25_Sun=_a(60.0), Vcmax25_Sh=_a(30.0),
        Vcmax25_C4Sun=_a(0.0), Vcmax25_C4Sh=_a(0.0),
        ASW_Sun=_a(ASW_Sun), ASW_Sh=_a(ASW_Sh), ASW_Soil=_a(ASW_Soil),
        Ts_bc=_a(300.0), Ca=_a(400.0), Ps=_a(101325.0), Ta=_a(298.0),
        lam=_a(2.5e6), Cp=_a(1005.0), rhoa=_a(1.2), Tv_atm=_a(298.5), q_atm=_a(0.012),
        m=_a(4.0), b0=_a(0.01), alf=_a(0.3), TgC=_a(20.0), fC4=_a(0.0),
        fStress_soil=_a(0.8), ur=_a(3.0), CI=_a(0.75), z0m=_a(0.5), displa=_a(3.0),
        z0=_a(10.0), cv=_a(0.0135), d_leaf=_a(0.025), r_soil_surface=_a(0.0),
        psi_s_mpa=_a(psi_s), rsoil_leaf=_a(0.0), gplant_mol=_a(gplant),
        psi50_xylem_mpa=_a(-2.0), ck_xylem=_a(1.5), gamma_cost=_a(gamma),
    )


_CFG = CanopyConfig(stomatal_model="phydro", tol=1e-6, max_iters=200)
_X0 = jnp.array([298.0, 298.0, 280.0, 280.0, 298.0, 0.013, 0.5, 0.5])


def _solve(bundle, cfg=_CFG, x0=_X0):
    x, n, conv = solve_canopy_closure(x0, bundle, cfg)
    f = canopy_forward(x, bundle, cfg.LE_module, cfg.stomatal_model,
                       cfg.le_cap_mode, cfg.use_ta_for_photosynthesis)
    return x, bool(conv), f


def test_state_size_per_model():
    assert state_size("phydro") == 8
    for m in ("ball_berry", "medlyn", "leuning"):
        assert state_size(m) == 6


def test_converges_and_dpsi_is_the_hydraulic_state():
    x_wet, conv_wet, f_wet = _solve(_bundle())
    x_dry, conv_dry, f_dry = _solve(_bundle(psi_s=-1.5))
    assert conv_wet and conv_dry
    assert np.all(np.isfinite(np.asarray(x_wet))) and np.all(np.isfinite(np.asarray(x_dry)))
    # By day both classes draw water (dpsi > 0), the sunlit one more.
    assert float(x_wet[6]) > float(x_wet[7]) > 0.0
    # Drier soil: less conductance and less assimilation at the optimum.
    assert float(f_dry["gs_Sun"]) < float(f_wet["gs_Sun"])
    assert float(f_dry["An_Sun"]) < float(f_wet["An_Sun"])
    assert f_wet["dpsi_Sun"] is not None and float(f_wet["dpsi_Sun"]) == float(x_wet[6])


def test_night_closes_the_draw_down():
    x, conv, f = _solve(_bundle(APAR_Sun=0.0, APAR_Sh=0.0, ASW_Sun=0.0,
                                ASW_Sh=0.0, ASW_Soil=0.0))
    assert conv
    # The draw-down rows are KKT (Fischer-Burmeister) residuals, flat near the
    # dpsi = 0 corner, so the converged draw-down is "closed" to the solver's
    # residual tolerance: a few 1e-3 MPa, i.e. a transpiration of ~1e-5
    # mol/m2/s (< 0.5 W/m2).
    assert abs(float(x[6])) < 5e-3 and abs(float(x[7])) < 5e-3
    # A dark leaf respires: net An = -Rd < 0 (not floored at zero), and the
    # diffusion balance puts its Ci above Ca.
    assert float(f["An_Sun"]) < 0.0 and float(f["An_Sh"]) < 0.0
    assert float(x[2]) > 400.0 and float(x[3]) > 400.0


def test_six_state_models_unchanged():
    """The appended unknowns are phydro-only: a Medlyn solve still takes and
    returns a 6-vector, and its forward reports no draw-down."""
    cfg = CanopyConfig(stomatal_model="medlyn", tol=1e-6, max_iters=200)
    x, conv, f = _solve(_bundle(), cfg=cfg, x0=_X0[:6])
    assert conv and x.shape == (6,)
    assert f["dpsi_Sun"] is None and f["dpsi_Sh"] is None


def test_phydro_needs_the_supply_in_the_bundle():
    b = _bundle()._replace(psi_s_mpa=None)
    with pytest.raises(ValueError, match="psi_s_mpa=None"):
        solve_canopy_closure(_X0, b, _CFG)


@needs_x64
@pytest.mark.parametrize("name", ["gplant", "psi_s", "gamma"])
def test_ift_gradient_of_dpsi_matches_finite_differences(name):
    """d dpsi_sun / d(hydraulic input) through the implicit backward pass
    equals central finite differences."""
    base = dict(gplant=4e-3, psi_s=-0.5, gamma=1.0)

    def dpsi_sun(v):
        return solve_canopy_closure(_X0, _bundle(**{**base, name: v}), _CFG)[0][6]

    v0 = base[name]
    g = float(jax.grad(dpsi_sun)(_a(v0)))
    h = 1e-4 * abs(v0)
    fd = (float(dpsi_sun(_a(v0 + h))) - float(dpsi_sun(_a(v0 - h)))) / (2.0 * h)
    assert np.isfinite(g) and g != 0.0
    np.testing.assert_allclose(g, fd, rtol=2e-3, atol=1e-8 * max(1.0, abs(fd)))

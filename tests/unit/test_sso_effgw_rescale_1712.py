"""Direct test of ``scripts/validate/sso_effgw_rescale_1712.py`` (#1712):
the S statistic, the ratio and its S_new == 0 guard, the hand launch-stress
recomputation and the doubling property, on the 14 CAM6 oracle columns."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.gravity_wave_drag.config import McFarlaneConfig
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd

_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = _ROOT / "tests" / "atmosphere" / "oracle_fixtures" / "gw_oro_cam6.npz"


def _probe():
    path = _ROOT / "scripts" / "validate" / "sso_effgw_rescale_1712.py"
    spec = importlib.util.spec_from_file_location("sso1712_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _probe()


def test_area_mean_abs_weights_by_area():
    assert P.area_mean_abs([1.0, -3.0], [3.0, 1.0]) == pytest.approx(1.5)


def test_rescale_ratio_and_zero_guard():
    assert P.rescale_effgw(0.125, 2.0, 4.0) == pytest.approx(0.0625)
    with pytest.raises(ValueError, match="S_new == 0"):
        P.rescale_effgw(0.125, 2.0, 0.0)


@pytest.fixture(scope="module")
def fx():
    if not jax.config.jax_enable_x64:
        pytest.skip("fp64 check (JAX_ENABLE_X64=1)")
    return dict(np.load(FIXTURE))


def _call(f, **over):
    cfg = McFarlaneConfig()._replace(**over)
    j = {k: jnp.asarray(f[k]) for k in ("u", "v", "T", "p_full", "p_half",
                                        "z_full", "z_half", "rho", "lat")}
    args = (j["u"], j["v"], j["T"], j["p_full"], j["p_half"], j["z_full"],
            j["z_half"], j["rho"], j["lat"], float(f["dt"]), cfg)
    kw = dict(h_topo_col=jnp.asarray(f["sgh"]), land_frac_col=jnp.asarray(f["lf"]),
              p_ref_half=jnp.asarray(f["p_ref_half"]))
    _, tau = mcfarlane_gwd(*args, **kw, return_stress=True)
    return {"args": args, "kw": kw, "tau": np.asarray(tau)}


_BIG = float(np.finfo(np.float64).max)
_OFF = {"umcfac": _BIG, "tndmax_per_day": _BIG}


def test_doubling_without_limiters_is_exact(fx):
    t1 = _call(fx, effgw_oro=0.125, **_OFF)["tau"][:, -1]
    t2 = _call(fx, effgw_oro=0.25, **_OFF)["tau"][:, -1]
    nz = t1 > 0
    assert nz.sum() >= 3
    np.testing.assert_allclose(t2[nz], 2.0 * t1[nz], rtol=1e-12)


def test_limiters_bind_aloft_bottom_stress_stays_linear(fx):
    """In the fixture's umcfac/tndmax columns the limiters change the stress
    PROFILE (non-vacuous: they bind), but CAM's redetermination carries the
    limited divergence downward, so the bottom-interface stress -- the S
    statistic -- is unchanged and doubles exactly with effgw_oro.  Only a
    limiter binding in the bottom layer itself would break linearity."""
    t = _call(fx)["tau"]
    f = _call(fx, **_OFF)["tau"]
    assert not np.allclose(t, f, rtol=1e-12, atol=0)       # limiters bind
    np.testing.assert_array_equal(t[:, -1], f[:, -1])
    t2 = _call(fx, effgw_oro=0.25)["tau"][:, -1]
    nz = t[:, -1] > 0
    np.testing.assert_allclose(t2[nz], 2.0 * t[nz, -1], rtol=1e-12)


def test_hand_launch_stress_matches_model(fx):
    rec = _call(fx, **_OFF)
    t = rec["tau"][:, -1]
    cols = np.flatnonzero(t > 0)
    hand = P.hand_bottom_stress(rec, cols)
    np.testing.assert_allclose(hand, t[cols], rtol=1e-10)

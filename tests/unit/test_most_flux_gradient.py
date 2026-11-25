"""Known-answer checks for the flux-gradient instrument."""
import importlib.util
import pathlib

import numpy as np
import pytest

from legoesm import constants
from legoesm.core.bulk_flux import psi_h, psi_m

_PATH = pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit/flux_gradient.py"
_SPEC = importlib.util.spec_from_file_location("flux_gradient", _PATH)
fg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fg)

KAPPA = constants.kappa_von_karman


def test_log_gradient_is_exact_for_a_log_profile():
    z1, z2, ustar, z0 = 1.0, 4.0, 0.3, 0.01
    u = lambda z: ustar / KAPPA * np.log(z / z0)
    z_g, du = fg.log_gradient(u(z1), u(z2), z1, z2)
    assert z_g == pytest.approx(2.0)
    assert fg.phi_observed(du, ustar, z_g) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("L", [50.0, -50.0])
def test_recovers_model_phi_from_a_model_profile(L):
    z1, z2, ustar, z0 = 1.8, 2.2, 0.3, 0.01
    u = lambda z: ustar / KAPPA * (np.log(z / z0) - float(psi_m(z / L)))
    theta = lambda z: 0.1 / KAPPA * (np.log(z / z0) - float(psi_h(z / L)))
    z_g, du = fg.log_gradient(u(z1), u(z2), z1, z2)
    _, dth = fg.log_gradient(theta(z1), theta(z2), z1, z2)
    zeta = np.array([z_g / L])
    assert fg.phi_observed(du, ustar, z_g) == pytest.approx(fg.phi_model(zeta, "dyer1974", "m")[0], rel=2e-3)
    assert fg.phi_observed(dth, 0.1, z_g) == pytest.approx(fg.phi_model(zeta, "dyer1974", "h")[0], rel=2e-3)


def test_obukhov_sign_follows_the_heat_flux():
    assert fg.obukhov_length(0.3, 0.1, 300.0) < 0
    assert fg.obukhov_length(0.3, -0.01, 300.0) > 0


def test_half_hour_mean_drops_incomplete_and_flagged_samples():
    t = np.array([600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0])
    x = np.arange(6.0)
    t30, m = fg.to_half_hour(t, x, np.array([1, 1, 1, 1, 4, 1]))
    assert t30.tolist() == [1800, 3600]
    assert m[0] == pytest.approx(1.0) and np.isnan(m[1])


def test_log_interp_recovers_a_log_profile():
    z1, z2, z = 2.0, 4.0, 2.3
    assert fg.log_interp(np.log(z1), np.log(z2), z1, z2, z) == pytest.approx(np.log(z))


def test_half_hour_mean_refuses_duplicate_stamps():
    with pytest.raises(ValueError, match="duplicate"):
        fg.to_half_hour(np.array([600.0, 600.0, 1800.0]), np.ones(3), np.ones(3))

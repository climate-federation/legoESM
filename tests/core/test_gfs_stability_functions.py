"""NCEP GFS surface-layer stability functions (sfc_diff.f, subroutine stability).

Oracle expressions are transcribed from UFS ccpp-physics
physics/SFC_Layer/UFS/sfc_diff.f (identical in MYNN-SFC psi*_full_gfs).
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.core import bulk_flux as bf

jax.config.update("jax_enable_x64", True)

ALPHA4 = 20.0
RAT_M = (-3.975, 12.32, -7.755, 6.041)
RAT_H = (-7.941, 24.75, -8.705, 7.899)


def _fortran_pm_stable(hl1, hl0):
    aa, aa0 = math.sqrt(1 + ALPHA4 * hl1), math.sqrt(1 + ALPHA4 * hl0)
    return aa0 - aa + math.log((1 + aa) / (1 + aa0))


def _fortran_unstable(hl, kind):
    a0, a1, b1, b2 = RAT_M if kind == "m" else RAT_H
    if hl >= -0.5:
        return (a0 + a1 * hl) * hl / (1 + (b1 + b2 * hl) * hl)
    h = -hl
    tem1 = 1 / math.sqrt(h)
    if kind == "m":
        return math.log(h) + 2.0 * math.sqrt(tem1) - 0.8776
    return math.log(h) + 0.5 * tem1 + 1.386


STABLE_Z = [1e-3, 0.05, 0.25, 1.0, 3.0, 9.5]
UNSTABLE_Z = [-1e-3, -0.1, -0.3, -0.5, -0.5000001, -0.7, -2.0, -6.0]


@pytest.mark.parametrize("z", STABLE_Z)
def test_stable_matches_gfs_with_and_without_z0_term(z):
    for fn in (bf.psi_m, bf.psi_h):
        got = float(fn(jnp.array([z]), "gfs", zeta_cap_width=0.0)[0])
        assert got == pytest.approx(_fortran_pm_stable(z, 0.0), rel=1e-12, abs=1e-14)
        z0 = z * 1e-3  # z0/L term: the psi(0) shift cancels in the difference
        diff = got - float(fn(jnp.array([z0]), "gfs", zeta_cap_width=0.0)[0])
        assert diff == pytest.approx(_fortran_pm_stable(z, z0), rel=1e-10, abs=1e-14)


@pytest.mark.parametrize("z", UNSTABLE_Z)
def test_unstable_hard_limit_matches_gfs(z):
    for kind, fn in (("m", bf._gfs_unstable_psi_m), ("h", bf._gfs_unstable_psi_h)):
        got = float(fn(jnp.array([z]), sharpness=math.inf)[0])
        assert got == pytest.approx(_fortran_unstable(z, kind), rel=1e-12, abs=1e-14)


# Measured smooth-default minus GFS-step gap per fixture point (RULE 5),
# sharpness 20; regenerate with the GAP_PRINT block below if the form changes.
RECORDED_GAP = {
    "m": {-0.001: 6.176652958284762e-08, -0.1: 1.1416828936461343e-05,
          -0.3: 0.00019827660738624964, -0.5: -1.730277463152241e-05,
          -0.5000001: -0.014369020757334927, -0.7: -0.014344541927014065,
          -2.0: -0.014370862548647612, -6.0: -0.014370862548640506},
    "h": {-0.001: 4.705153058837119e-08, -0.1: 1.2088530422116328e-05,
          -0.3: 0.0002382597381629381, -0.5: -3.1593079101055466e-05,
          -0.5000001: -0.013659348889895284, -0.7: -0.013665742255028368,
          -2.0: -0.013662285036268429, -6.0: -0.01366228503625777},
}


@pytest.mark.parametrize("z", UNSTABLE_Z)
def test_unstable_default_gap_recorded(z):
    for kind, fn in (("m", bf._gfs_unstable_psi_m), ("h", bf._gfs_unstable_psi_h)):
        gap = float(fn(jnp.array([z]))[0]) - _fortran_unstable(z, kind)
        assert gap == pytest.approx(RECORDED_GAP[kind][z], rel=1e-6, abs=1e-12)


def test_far_field_gap_is_the_branch_mismatch():
    # Hand value: rational(-0.5) - free(0.5) from the sfc_diff.f forms.
    for kind, fn, d in (("m", bf._gfs_unstable_psi_m, -0.0143517193),
                        ("h", bf._gfs_unstable_psi_h, -0.0136277572)):
        gap = float(fn(jnp.array([-6.0]))[0]) - _fortran_unstable(-6.0, kind)
        assert abs(gap - d) < 5e-5


def test_neutral_limits():
    z = jnp.array([0.0])
    for fn in (bf._gfs_unstable_psi_m, bf._gfs_unstable_psi_h):
        assert abs(float(fn(z)[0])) < 1e-12
        assert float(bf.phi_from_psi(fn, z)[0]) == pytest.approx(1.0, abs=1e-12)
    assert float(bf._gfs_stable_psi(z)[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(bf.phi_from_psi(bf._gfs_stable_psi, z)[0]) == 1.0


def test_stable_phi_analytic_and_sublinear():
    z = jnp.array(STABLE_Z)
    a = jnp.sqrt(1.0 + ALPHA4 * z)
    np.testing.assert_allclose(bf.phi_from_psi(bf._gfs_stable_psi, z),
                               1.0 + 10.0 * z / (1.0 + a), rtol=1e-12)


def _phi_one_sided(fn, z0, eps=1e-5):
    zl, zr = jnp.array([z0 - eps]), jnp.array([z0 + eps])
    return float(bf.phi_from_psi(fn, zl)[0]), float(bf.phi_from_psi(fn, zr)[0])


@pytest.mark.parametrize("fn", [bf._gfs_unstable_psi_m, bf._gfs_unstable_psi_h])
def test_default_is_c1_at_switch_and_hard_step_is_not(fn):
    left, right = _phi_one_sided(fn, -0.5)
    assert abs(left - right) < 1e-3  # phi continuous across the blend
    hl, hr = _phi_one_sided(lambda z: fn(z, sharpness=math.inf), -0.5)
    assert abs(hl - hr) > 1e-2  # the GFS step itself jumps (~0.013)


def test_phi_continuous_at_zero_slope_may_jump():
    for fn in (bf._gfs_unstable_psi_m, bf._gfs_unstable_psi_h):
        assert float(bf.phi_from_psi(fn, jnp.array([-1e-7]))[0]) == pytest.approx(1.0, abs=1e-5)
    assert float(bf.phi_from_psi(bf._gfs_stable_psi, jnp.array([1e-7]))[0]) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("z", [-0.05, -0.45, -0.5, -0.55, -3.0])
@pytest.mark.parametrize("fn", [bf._gfs_unstable_psi_m, bf._gfs_unstable_psi_h])
def test_unstable_gradients_at_default(fn, z):
    g = jax.grad(lambda x: fn(x))(jnp.float64(z))
    assert np.isfinite(float(g)) and float(g) != 0.0
    check_grads(fn, (jnp.array([z]),), order=2, modes=["fwd", "rev"])


@pytest.mark.parametrize("z", [-0.3, 1e-2, 0.3, 5.0])
def test_stable_gradients(z):
    check_grads(lambda x: bf.psi_m(x, "gfs"), (jnp.array([z]),), order=2, modes=["fwd", "rev"])


def test_unstable_registry_and_stable_selector_keep_businger_dyer():
    assert bf.UNSTABLE_PSI_FUNCTIONS["gfs"] == (bf._gfs_unstable_psi_m, bf._gfs_unstable_psi_h)
    z = jnp.array([-2.0, -0.3, -0.01])
    np.testing.assert_array_equal(bf.psi_m(z, "gfs"), bf.psi_m(z, "dyer1974"))
    np.testing.assert_array_equal(bf.psi_h(z, "gfs"), bf.psi_h(z, "dyer1974"))


def test_traced_sharpness_rejected():
    with pytest.raises(TypeError):
        bf._gfs_unstable_psi_m(jnp.array([-2.0]), sharpness=jnp.float64(math.inf))


def test_driver_accepts_gfs():
    from legoesm.driver.config import ExperimentConfig
    ExperimentConfig(turbulence="louis", surface_stability_scheme="gfs",
                     surface_bulk_scheme="coare3").validate_strict()
    with pytest.raises(ValueError, match="surface_stability_scheme"):
        ExperimentConfig(turbulence="louis", surface_stability_scheme="gfs_typo",
                         surface_bulk_scheme="coare3").validate_strict()


def test_run_scripts_accept_gfs():
    from scripts.run.run_amip import build_arg_parser
    from scripts.run.run_coupled import build_parser
    from scripts.run.run_omip import parse_args
    assert build_arg_parser().parse_args(
        ["--surface-stability-scheme", "gfs"]).surface_stability_scheme == "gfs"
    assert parse_args(["--grid", "latlon", "--surface-stability-scheme", "gfs"]
                      ).surface_stability_scheme == "gfs"
    assert build_parser().parse_args(
        ["--ice-stability-scheme", "gfs"]).ice_stability_scheme == "gfs"


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        bf.psi_m(jnp.array([0.1]), "GFS")


if __name__ == "__main__":  # GAP_PRINT: regenerate RECORDED_GAP
    for kind, fn in (("m", bf._gfs_unstable_psi_m), ("h", bf._gfs_unstable_psi_h)):
        print(kind, {z: float(fn(jnp.array([z]))[0]) - _fortran_unstable(z, kind) for z in UNSTABLE_Z})

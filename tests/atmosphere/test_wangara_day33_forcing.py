"""The Wangara forcing is shared by the LES and the SCM, so it is pinned here.

An SCM scored against an LES that was given a different surface flux or a
different geostrophic profile measures the difference between the two
forcings, not between the turbulence closures. One module defines both; these
tests pin its values and, above all, its TIME CONVENTION.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.forcing import wangara_day33 as w  # noqa: E402


def test_clock_is_absolute_not_run_relative():
    """The easy thing to get wrong, and it flips the sign of the forcing.

    The run starts at 09:00. Evaluating the diurnal cosine on run-relative
    time would put hour 0 into a curve that peaks at 13:00 and troughs near
    midnight, so a CONVECTIVE case would begin with a negative surface heat
    flux.
    """
    at_start = float(w.surface_theta_flux(w.T_START_S))
    run_relative = float(w.surface_theta_flux(0.0))
    assert at_start > 0.0, "09:00 must be a warming surface"
    assert run_relative < 0.0, (
        "hour 0 is the middle of the night; this is the value a run-relative "
        "clock would have used")
    assert at_start == pytest.approx(0.0897, abs=5.0e-4)


def test_flux_peaks_at_1300_local():
    hours = np.arange(0.0, 24.0, 0.25)
    flux = np.asarray([float(w.surface_theta_flux(h * 3600.0)) for h in hours])
    assert hours[int(np.argmax(flux))] == pytest.approx(13.0)
    assert flux.max() == pytest.approx(0.216, rel=1e-9)


def test_moisture_flux_shares_the_diurnal_shape():
    """Same cosine, different amplitude: the ratio must be height/time-flat."""
    for hour in (9.0, 11.0, 13.0, 16.0):
        t = hour * 3600.0
        ratio = float(w.surface_qv_flux(t)) / float(w.surface_theta_flux(t))
        assert ratio == pytest.approx(2.29e-5 / 0.216, rel=1e-9)


def test_coriolis_is_southern_hemisphere():
    f = w.coriolis_parameter()
    assert f < 0.0, "Wangara is at 34.5 S"
    expect = 2.0 * constants.Omega * np.sin(np.deg2rad(-34.5))
    assert f == pytest.approx(expect, rel=1e-12)


def test_geostrophic_wind_is_easterly_and_kinked_at_1km():
    z = np.array([0.0, 500.0, 999.9, 1000.1, 1500.0])
    ug = np.asarray(w.geostrophic_u(z))
    assert ug[0] == pytest.approx(-5.5)
    assert np.all(ug < 0.0), "easterly throughout the profile"
    # weakens (towards zero) with height
    assert np.all(np.diff(ug) > 0.0)
    # continuous across the kink to within the two fits' own disagreement
    assert abs(ug[2] - ug[3]) < 0.1


def test_scm_matrix_case_uses_the_shared_module(tmp_path):
    """The SCM side must not carry its own copy of these formulas."""
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts" / "matrix"))
    spec = importlib.util.spec_from_file_location(
        "_wangara_scm", root / "scripts" / "matrix" / "scm" / "wangara.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"SCM matrix case not importable here: {exc}")
    assert mod._w_th_s is w.surface_theta_flux
    assert mod._w_qv_s is w.surface_qv_flux
    assert mod._F_C == pytest.approx(w.coriolis_parameter())
    assert mod._THETA_INIT == pytest.approx(w.THETA_INIT_K)
    z = np.array([0.0, 500.0, 1500.0])
    assert np.allclose(np.asarray(mod._u_geo_profile(3, z)),
                       np.asarray(w.geostrophic_u(z)))


def test_wangara_is_registered_as_its_own_scm_case():
    from legoesm.atmosphere.forcing.scm.analytic_scm_case import (
        ANALYTIC_SCM_CASES,
    )
    assert "wangara" in ANALYTIC_SCM_CASES
    spec = ANALYTIC_SCM_CASES["wangara"]
    assert spec.theta0_K == pytest.approx(w.THETA_INIT_K)
    assert spec.f_c < 0.0
    assert spec.scored == ("theta", "u", "v")
    # and it must NOT be confusable with the Nieuwstadt case
    cbl = ANALYTIC_SCM_CASES["cbl"]
    assert cbl.theta0_K != spec.theta0_K
    assert cbl.f_c == 0.0


# --- the SCM arm must be driven by this module, not by a frozen snapshot ----

def test_the_scm_case_uses_the_shared_diurnal_flux():
    """The LES driver reads wangara_day33; the SCM case must read the SAME
    functions, or the two sides are driven differently while claiming not to
    be. The loader used to build `w_th_s = lambda _t: <the 09:00 value>`, which
    held that value through the 13:00 maximum and the whole afternoon decay."""
    from legoesm.atmosphere.forcing.scm.analytic_scm_case import (
        ANALYTIC_SCM_CASES,
    )
    spec = ANALYTIC_SCM_CASES["wangara"]
    assert spec.surface_theta_flux_fn is w.surface_theta_flux
    assert spec.geostrophic_u_fn is w.geostrophic_u
    assert spec.t_start_s == pytest.approx(w.T_START_S)
    # The steady cases must NOT have acquired one.
    for name in ("cbl", "gabls1", "ekman"):
        assert ANALYTIC_SCM_CASES[name].surface_theta_flux_fn is None
        assert ANALYTIC_SCM_CASES[name].geostrophic_u_fn is None


def test_the_built_case_actually_varies_in_time_and_height():
    """NON-VACUOUS: assert the FORCING the SCM will call, not the spec field.

    A spec entry that the loader ignores looks identical from the table.
    """
    import numpy as np
    from legoesm.atmosphere.forcing.scm.analytic_scm_case import (
        load_analytic_scm_case,
    )
    case = load_analytic_scm_case("wangara", nlev=24, dt=10.0)

    # TIME: the flux must trace out the cycle, and peak near 13:00 local, i.e.
    # 4 h into a 09:00 start.
    hours = np.arange(0.0, 8.01, 0.5)
    flux = np.array([float(case.forcing.w_th_s(h * 3600.0)) for h in hours])
    assert flux.std() > 0.02, (
        f"the surface flux is effectively constant ({flux.std():.4g} K m/s "
        "spread) over the 8 h run")
    assert hours[int(np.argmax(flux))] == pytest.approx(4.0, abs=0.5)
    # ...and the first value is the one the old code froze, so the defect is
    # exactly "the whole run at flux[0]".
    assert abs(flux[-1] - flux[0]) > 0.02

    # HEIGHT: the geostrophic wind must be sheared, not a uniform -5.5 m/s.
    ug = np.asarray(case.forcing.u_geo(0.0))
    assert ug.shape == (24,)
    assert ug.max() - ug.min() > 1.0, (
        f"geostrophic u spans only {ug.max() - ug.min():.3g} m/s; the LES uses "
        "a kinked easterly jet")


def test_the_steady_cases_are_untouched_by_the_new_path():
    """cbl/gabls1/ekman must still get exactly their scalar flux."""
    import numpy as np
    from legoesm.atmosphere.forcing.scm.analytic_scm_case import (
        ANALYTIC_SCM_CASES,
        load_analytic_scm_case,
    )
    for name in ("cbl", "gabls1", "ekman"):
        case = load_analytic_scm_case(name, nlev=16, dt=10.0)
        want = ANALYTIC_SCM_CASES[name].sfc_theta_flux_K_m_s
        for t in (0.0, 1234.0, 28800.0):
            assert float(case.forcing.w_th_s(t)) == pytest.approx(want), name
        if case.forcing.u_geo is not None:
            ug = np.asarray(case.forcing.u_geo(0.0))
            assert ug.min() == ug.max() == pytest.approx(
                ANALYTIC_SCM_CASES[name].u_geo_m_s), name


def test_the_flux_is_traceable():
    """It is called inside the differentiated rollout; a NumPy-materialising
    forcing would break the tuner rather than merely be slow."""
    import jax
    import jax.numpy as jnp
    from legoesm.atmosphere.forcing.scm.analytic_scm_case import (
        load_analytic_scm_case,
    )
    case = load_analytic_scm_case("wangara", nlev=8, dt=10.0)
    got = jax.jit(lambda t: case.forcing.w_th_s(t))(jnp.asarray(3600.0))
    assert jnp.isfinite(got)

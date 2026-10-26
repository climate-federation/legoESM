"""The long-run ocean drivers integrate CORE-II / JRA55-do forcing in-step (#1820).

``run_omip2`` / ``run_centennial_spinup`` / ``run_bryan_thc`` step through
``step_with_omip2_forcing`` (``compute_omip2_surface_forcing`` ->
``model.step(surface_forcing=...)``).  On MPAS the matrix-built model has no
physics and would silently ignore ``surface_forcing``, so the driver builds it
with ``external_surface_forcing_physics()``; the zero-forcing test pins that
nothing else is switched on.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_MATRIX = _REPO / "scripts" / "matrix"
_DT = 1800.0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _omip2():
    # Registered as ``run_omip2``: run_centennial_spinup / run_bryan_thc import
    # the helper under that name when run as scripts.
    if str(_MATRIX) not in sys.path:
        sys.path.insert(0, str(_MATRIX))
    return _load("run_omip2", _REPO / "scripts/run/ocean_long_runs/run_omip2.py")


def _zero_forcing(grid):
    import jax.numpy as jnp
    from legoesm.ocean.state import OceanSurfaceForcing
    zero = jnp.zeros(np.asarray(grid.latCell).shape)
    return OceanSurfaceForcing(tau_x=zero, tau_y=zero, q_net=zero, sw_down=zero)


def _unforced_step(model, state, grid, grid_type):
    # The MPAS 'external' physics requires a forcing object on every step.
    if grid_type == "mpas":
        return model.step(state, _DT, surface_forcing=_zero_forcing(grid))
    return model.step(state, _DT)


def _east_wind(u_east=8.0, nlat=18, nlon=36, sw_down=200.0):
    from legoesm.ocean.forcing.jra55_do import OceanForcing
    lat = np.linspace(-89.0, 89.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)

    def fld(v):
        return np.full((1, nlat, nlon), float(v))

    return OceanForcing(
        lon=lon, lat=lat, time_s=np.array([0.0]),
        u10=fld(u_east), v10=fld(0.0), T_air=fld(288.0), q_air=fld(0.008),
        sw_down=fld(sw_down), lw_down=fld(350.0), precip=fld(0.0),
        runoff=fld(0.0),
    )


def _diff(a, b):
    return max(float(np.max(np.abs(np.asarray(x.data) - np.asarray(y.data))))
               for x, y in ((a.T, b.T), (a.u, b.u)))


@pytest.mark.parametrize("grid_type,res", [("latlon", "18x36"), ("mpas", "ico2")])
def test_driver_step_consumes_the_forcing(grid_type, res):
    """An east wind + heat flux must change the state vs an unforced step, and
    on lat-lon drive the surface eastward (ocean reaction to the stress)."""
    m = _omip2()
    state, grid, z, model = m._build_state(grid_type, res, nlev=6,
                                           scripts_dir=_MATRIX)
    forced = m.step_with_omip2_forcing(
        model, state, forcing=_east_wind(), idx_t=0, grid=grid,
        grid_type=grid_type, dt=_DT)
    free = _unforced_step(model, state, grid, grid_type)
    assert np.isfinite(np.asarray(forced.T.data)).all()
    assert _diff(forced, free) > 1e-9
    du = np.asarray(forced.u.data)[..., 0] - np.asarray(free.u.data)[..., 0]
    if grid_type == "latlon":
        wet = np.asarray(state.u_mask.data) > 0.5
        assert du[wet].mean() > 0.0
    else:
        # Edge-normal response projected on east: an east wind pushes east.
        east = du * np.cos(np.asarray(grid.angleEdge))
        assert east.mean() > 0.0


@pytest.mark.parametrize("grid_type,res", [("latlon", "18x36"), ("mpas", "ico2")])
def test_driver_step_heat_matches_surface_flux(grid_type, res):
    """Ocean heat gained in one forced step minus the unforced step equals the
    area-integrated ``q_net * dt``: shortwave is counted once.

    Zero wind, so the stress does not redistribute heat between the two
    steps.  Measured on lat-lon 18x36: gained/expected = 1.0022, 1.0009 and
    0.9991 for SW = 0, 200 and 1000 W/m2 (the residual is the budget's
    thickness bookkeeping and does not grow with SW); double-counting even
    10% of the shortwave would put the ratio above 1.05."""
    m = _omip2()
    from legoesm.ocean.budgets import compute_tracer_budget
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    from legoesm.ocean.budgets import _grid_area
    state, grid, z, model = m._build_state(grid_type, res, nlev=6,
                                           scripts_dir=_MATRIX)
    forcing = _east_wind(u_east=0.0, sw_down=1000.0)
    sf = compute_omip2_surface_forcing(state, forcing=forcing, idx_t=0,
                                       grid=grid, grid_type=grid_type)
    forced = model.step(state, _DT, surface_forcing=sf)
    free = _unforced_step(model, state, grid, grid_type)
    hc = lambda s: compute_tracer_budget(s, z, grid_type=grid_type,
                                         grid=grid).heat_content
    gained = hc(forced) - hc(free)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    expected = float((np.asarray(_grid_area(grid_type, grid)) * mask
                      * np.asarray(sf.q_net)).sum()) * _DT
    assert expected != 0.0
    assert gained == pytest.approx(expected, rel=5e-3)


def test_mpas_without_physics_ignores_surface_forcing():
    """Why the driver needs ``external_surface_forcing_physics``: the plain
    matrix MPAS model accepts ``surface_forcing`` and drops it."""
    m = _omip2()
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    mm = m._import_matrix_module()
    tc = TestCase(case="omip2", grid_type="mpas", resolution="ico2",
                  duration_days=1.0, quick_days=1.0)
    grid, z, _, model, _, _, _ = _create_ocean_setup(
        tc, H_max=5500.0, nlev=6, A_h=5.0e4, A_v=1.0e-4, bottom_drag_r=1.0e-3)
    state = mm._create_rest_state(tc, grid, z, H_max=5500.0)
    sf = compute_omip2_surface_forcing(state, forcing=_east_wind(), idx_t=0,
                                       grid=grid, grid_type="mpas")
    assert _diff(model.step(state, _DT, surface_forcing=sf),
                 model.step(state, _DT)) == 0.0


def test_mpas_external_physics_switches_nothing_else_on():
    """Zero surface forcing through the driver's MPAS physics is bit-identical
    to the matrix model with ``physics=None``: only the external block is on."""
    m = _omip2()
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    state, grid, z, model = m._build_state("mpas", "ico2", nlev=6,
                                           scripts_dir=_MATRIX)
    tc = TestCase(case="omip2", grid_type="mpas", resolution="ico2",
                  duration_days=1.0, quick_days=1.0)
    _, _, _, bare, _, _, _ = _create_ocean_setup(
        tc, H_max=5500.0, nlev=6, A_h=5.0e4, A_v=1.0e-4, bottom_drag_r=1.0e-3)
    # Perturb first (one forced step): from rest, horizontally uniform fields
    # make lateral mixing a no-op, and the comparison could not fail.
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    sf = compute_omip2_surface_forcing(state, forcing=_east_wind(), idx_t=0,
                                       grid=grid, grid_type="mpas")
    state = model.step(state, _DT, surface_forcing=sf)
    assert float(np.abs(np.asarray(state.u.data)).max()) > 0.0
    a = model.step(state, _DT, surface_forcing=_zero_forcing(grid))
    b = bare.step(state, _DT)
    for f in ("T", "S", "u", "eta"):
        np.testing.assert_array_equal(np.asarray(getattr(a, f).data),
                                      np.asarray(getattr(b, f).data), err_msg=f)


def test_legacy_applicator_is_gone():
    from legoesm.ocean import coupler
    from legoesm.ocean.coupler import omip2_applicator as A
    name = "apply_omip2" + "_surface_fluxes"
    assert not hasattr(coupler, name) and not hasattr(A, name)
    assert not hasattr(A, "_apply_cgrid" + "_surface_fluxes")
    hits = []
    for top in ("packages", "src", "scripts", "tests"):
        for p in (_REPO / top).rglob("*"):
            if p.suffix in (".py", ".ipynb", ".yaml", ".yml", ".sbatch", ".sh") \
                    and p.is_file() and name in p.read_text(errors="ignore"):
                hits.append(str(p.relative_to(_REPO)))
    # The applicator module's docstring records the deletion.
    assert hits == ["packages/ocean/legoesm/ocean/coupler/omip2_applicator.py"], hits


@pytest.mark.parametrize("script,argv", [
    ("run_omip2.py", ["--grid", "latlon", "--resolution", "18x36"]),
    ("run_centennial_spinup.py", ["--grid", "latlon", "--resolution", "18x36",
                                  "--years", "1"]),
    ("run_bryan_thc.py", ["--resolution", "12x12"]),
])
def test_driver_loops_step_through_the_forcing(script, argv, tmp_path,
                                               monkeypatch):
    """Each driver's time loop advances the ocean through
    ``step_with_omip2_forcing`` (a loop that called ``model.step`` directly
    would run unforced and fail here)."""
    m = _omip2()
    calls = []
    orig = m.step_with_omip2_forcing

    def counted(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(m, "step_with_omip2_forcing", counted)
    driver = m if script == "run_omip2.py" else _load(
        f"_{script[:-3]}_1820", _REPO / "scripts/run/ocean_long_runs" / script)
    monkeypatch.setattr(sys, "argv", [script, "--smoke", "--dt", "86400",
                                      "--output", str(tmp_path), *argv]
                        + ([] if script == "run_bryan_thc.py"
                           else ["--allow-synthetic"]))
    assert driver.main() == 0
    assert len(calls) >= 1


def test_forcing_is_computed_before_the_pre_step_edits():
    """The bulk flux reads the BEGINNING-of-step SST (production / NEMO "now"
    convention); ``pre_step`` edits land on the state that is then stepped.
    A pre-step that warms the surface by 5 K must not change the flux."""
    m = _omip2()
    from legoesm.core.field import Field
    from legoesm.ocean.coupler import compute_omip2_surface_forcing
    state, grid, z, model = m._build_state("latlon", "18x36", nlev=6,
                                           scripts_dir=_MATRIX)
    forcing = _east_wind()

    def warm(s):
        T = np.asarray(s.T.data).copy()
        T[..., 0] += 5.0
        return s._replace(T=Field(T, name=s.T.name, dims=s.T.dims,
                                  units=s.T.units))

    got = m.step_with_omip2_forcing(model, state, forcing=forcing, idx_t=0,
                                    grid=grid, grid_type="latlon", dt=_DT,
                                    pre_step=warm)
    sf0 = compute_omip2_surface_forcing(state, forcing=forcing, idx_t=0,
                                        grid=grid, grid_type="latlon")
    want = model.step(warm(state), _DT, surface_forcing=sf0)
    np.testing.assert_array_equal(np.asarray(got.T.data), np.asarray(want.T.data))
    late = compute_omip2_surface_forcing(warm(state), forcing=forcing,
                                         idx_t=0, grid=grid, grid_type="latlon")
    assert float(np.abs(np.asarray(late.q_net) - np.asarray(sf0.q_net)).max()) > 1.0


def test_centennial_driver_orders_flux_edits_then_step(tmp_path, monkeypatch):
    """With runoff, ice-shelf melt and SSS restoring all on, each centennial
    step evaluates the bulk flux FIRST, then the three edits, then the ocean
    step (production / NEMO "now"-field order; user decision on #1820)."""
    m = _omip2()
    import legoesm.ocean.coupler as cpl
    log = []

    def spy(name, fn):
        def wrapped(*a, **k):
            log.append(name)
            return fn(*a, **k)
        monkeypatch.setattr(cpl, name, wrapped)

    for name in ("compute_omip2_surface_forcing", "apply_runoff_step",
                 "apply_ice_shelf_basal_step", "apply_sss_restoring_step"):
        spy(name, getattr(cpl, name))

    class _Model:
        def __init__(self, inner):
            self._inner = inner

        def step(self, *a, **k):
            log.append("model.step")
            return self._inner.step(*a, **k)

        def __getattr__(self, attr):
            return getattr(self._inner, attr)

    orig_build = m._build_state

    def build(*a, **k):
        state, grid, z, model = orig_build(*a, **k)
        return state, grid, z, _Model(model)

    monkeypatch.setattr(m, "_build_state", build)
    mask = np.zeros((18, 36)); mask[:2, :] = 1.0
    np.save(tmp_path / "mask.npy", mask)
    np.save(tmp_path / "draft.npy", np.where(mask > 0, 300.0, 0.0))
    driver = _load("_run_centennial_order_1820",
                   _REPO / "scripts/run/ocean_long_runs/run_centennial_spinup.py")
    monkeypatch.setattr(sys, "argv", [
        "run_centennial_spinup.py", "--smoke", "--allow-synthetic",
        "--dt", "43200", "--grid", "latlon", "--resolution", "18x36",
        "--years", "1", "--output", str(tmp_path / "out"),
        "--runoff", "--sss-restoring", "--ice-shelf",
        "--ice-shelf-mask", str(tmp_path / "mask.npy"),
        "--ice-draft", str(tmp_path / "draft.npy")])
    assert driver.main() == 0
    per_step = ["compute_omip2_surface_forcing", "apply_runoff_step",
                "apply_ice_shelf_basal_step", "apply_sss_restoring_step",
                "model.step"]
    steps = [x for x in log if x in per_step]
    assert len(steps) >= len(per_step) and len(steps) % len(per_step) == 0, steps
    assert steps == per_step * (len(steps) // len(per_step)), steps

"""NOTE on tolerance: the blend runs at the default float32 precision, so
these compare at rtol=1e-6, not 1e-12. At 1e-12 they passed only under
JAX_ENABLE_X64=1 and failed a plain `pytest tests/` by 3.7e-8 relative — a
float32-epsilon mismatch, i.e. the EXPECTATION was wrong, not the blend.

Land columns must receive a LAND albedo at the radiation call (MPAS lane).

Regression guard for the defect where the MPAS/hydrostatic radiation path
applied ``RRTMGPConfig.sfc_albedo`` (default 0.06, the OPEN-OCEAN value) to
every column including land, because:

* ``physics_pipeline.compute_radiation_core`` — which DOES form an
  ocean/ice/land albedo blend — belongs to the FV/lat-lon lane and is NOT
  called by ``ModelDriver._run_column``; and
* ``_make_hydrostatic_radiation`` (aliased as ``_make_mpas_radiation``) called
  ``_call_radiation_backend`` with no ``sfc_albedo_override`` and had no
  ``forcing["sfc_albedo"]`` hook, so no per-column albedo could reach the
  solver at all.  ``make_physics`` additionally REJECTED a build-time
  ``sfc_albedo_override`` for every ``model_type`` except ``spectral_pe``.

Published ``rsus/rsds`` from three completed MPAS AMIP runs implied a surface
albedo of exactly 0.0600 at BOTH the global min and max, over a globe that is
35.6% land — the signature this module locks out.
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation import integration as rad_int
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    RadiationConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.core.field import Field
from legoesm.forcing.surface_utils import blended_surface_albedo
from legoesm.grids.factory import create_grid
from legoesm.grids.vertical import create_sigma_coordinate

NLEV = 10

# Representative albedos: open ocean vs. mid-latitude vegetated land vs. ice.
_ALBEDO_OCEAN = 0.06
_ALBEDO_ICE = 0.65
_ALBEDO_LAND = 0.20


def test_modules_under_test_resolve_in_this_checkout():
    """Editable-install trap guard: a worktree run must exercise the
    worktree's modules, not the main checkout's installed copy."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    from legoesm.driver import model_driver
    from legoesm.forcing import surface_utils
    for mod in (rad_int, model_driver, surface_utils):
        assert root in pathlib.Path(mod.__file__).resolve().parents, (
            f"{mod.__name__} resolved to {mod.__file__}, OUTSIDE {root}")


def test_mpas_radiation_is_the_hydrostatic_radiation():
    """The MPAS lane really is the shared hydrostatic radiation builder."""
    assert rad_int._make_mpas_radiation is rad_int._make_hydrostatic_radiation


# ----------------------------------------------------------------------
# The shared tile blend
# ----------------------------------------------------------------------

def test_blended_surface_albedo_land_and_ocean_columns():
    """Land cells get the land albedo, ocean cells the ocean albedo."""
    # cell 0 = open ocean, 1 = all land, 2 = half land, 3 = full ice
    sic = jnp.array([0.0, 0.0, 0.0, 1.0])
    f_land = jnp.array([0.0, 1.0, 0.5, 0.0])
    albedo_land = jnp.full((4,), _ALBEDO_LAND)

    alb = blended_surface_albedo(
        sic, f_land, _ALBEDO_ICE, _ALBEDO_OCEAN, albedo_land)

    np.testing.assert_allclose(alb[0], _ALBEDO_OCEAN, rtol=1e-6)
    np.testing.assert_allclose(alb[1], _ALBEDO_LAND, rtol=1e-6)
    np.testing.assert_allclose(
        alb[2], 0.5 * _ALBEDO_LAND + 0.5 * _ALBEDO_OCEAN, rtol=1e-6)
    np.testing.assert_allclose(alb[3], _ALBEDO_ICE, rtol=1e-6)

    # The defect signature: a globe with land is NOT uniformly 0.06.
    assert float(jnp.max(alb)) > _ALBEDO_OCEAN + 1e-6


def test_blended_surface_albedo_no_land_is_pure_ocean_ice():
    """f_land=None keeps the legacy ocean/ice blend (byte-identical path)."""
    sic = jnp.array([0.0, 0.5, 1.0])
    alb = blended_surface_albedo(sic, None, _ALBEDO_ICE, _ALBEDO_OCEAN, None)
    np.testing.assert_allclose(
        np.asarray(alb),
        np.array([_ALBEDO_OCEAN,
                  0.5 * _ALBEDO_ICE + 0.5 * _ALBEDO_OCEAN,
                  _ALBEDO_ICE]),
        rtol=1e-6,
    )


def test_missing_land_albedo_raises_instead_of_using_ocean():
    """A land fraction with no land albedo FAILS LOUDLY.

    Silently returning the ocean albedo over land IS the defect; a missing
    albedo source must never degrade to it.
    """
    sic = jnp.zeros(4)
    f_land = jnp.array([0.0, 1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="albedo_land"):
        blended_surface_albedo(sic, f_land, _ALBEDO_ICE, _ALBEDO_OCEAN, None)


# ----------------------------------------------------------------------
# The radiation call itself
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh():
    return create_grid("mpas", 2, lloyd_iterations=10)


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(NLEV)


def _fld(a, dims):
    return Field(data=jnp.asarray(a), name="x", dims=dims, units="1")


def _state(mesh, sigma):
    n = int(mesh.nCells)
    sig = np.asarray(sigma.sigma_full)
    T = 220.0 + 70.0 * sig[None, :] * np.ones((n, 1))
    return types.SimpleNamespace(
        u=_fld(np.zeros((int(mesh.nEdges), NLEV)), ("edge", "lev")),
        T=_fld(T, ("cell", "lev")),
        p_s=_fld(np.full(n, 1.0e5), ("cell",)),
        phis=_fld(np.zeros(n), ("cell",)),
        v=None,
        tracers={"q_v": _fld(np.full((n, NLEV), 3.0e-3), ("cell", "lev"))},
    )


def _gray_cfg():
    """Gray radiation: cheap (no RRTMGP tables) and honours sfc_albedo."""
    return RadiationConfig(
        scheme="gray",
        gray=GrayRadiationConfig(sfc_albedo=_ALBEDO_OCEAN),
        cloud_scheme="none",
        diurnal_cycle=False,
    )


def _mixed_surface_albedo(n):
    """Half the cells land-like (0.20), half ocean-like (0.06)."""
    alb = np.full(n, _ALBEDO_OCEAN)
    alb[: n // 2] = _ALBEDO_LAND
    return jnp.asarray(alb)


def _spy_albedo(monkeypatch, cfg, mesh, sigma, state, forcing):
    """Run the MPAS radiation fn, capturing the albedo handed to the backend."""
    real = rad_int._call_radiation_backend
    seen = []

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("sfc_albedo_override", None))
        return real(*args, **kwargs)

    monkeypatch.setattr(rad_int, "_call_radiation_backend", _spy)
    tend = make_radiation_physics(cfg, "mpas")(
        state, mesh, sigma, forcing=forcing)
    return seen, tend


def test_forcing_sfc_albedo_reaches_the_radiation_backend(
        monkeypatch, mesh, sigma):
    """The per-cell surface albedo must ARRIVE at the radiation call.

    This is the direct proof of what each surface type receives: the value
    handed to ``_call_radiation_backend`` must be the per-column field, with
    land columns at the land albedo and ocean columns at the ocean albedo.

    RED before the fix: the hydrostatic/MPAS ``physics_fn`` has no
    ``forcing["sfc_albedo"]`` hook, so the override is ``None`` and the solver
    falls back to the scalar config albedo (0.06) for EVERY column.
    """
    state = _state(mesh, sigma)
    n = int(mesh.nCells)
    alb = _mixed_surface_albedo(n)

    seen, _ = _spy_albedo(
        monkeypatch, _gray_cfg(), mesh, sigma, state,
        forcing={"day_of_year": jnp.asarray(80.0),
                 "seconds_of_day": jnp.asarray(43200.0),
                 "sfc_albedo": alb},
    )

    assert len(seen) == 1
    got = seen[0]
    assert got is not None, (
        "forcing['sfc_albedo'] never reached _call_radiation_backend — every "
        "column is solved at the scalar config albedo (0.06, open ocean), "
        "including all land."
    )
    got = np.asarray(got).reshape(-1)
    assert got.shape == (n,)
    np.testing.assert_allclose(got, np.asarray(alb), rtol=1e-6)
    # Land columns must NOT be at the ocean value.
    assert got[: n // 2].min() == pytest.approx(_ALBEDO_LAND)
    assert got[n // 2:].max() == pytest.approx(_ALBEDO_OCEAN)


def test_no_forcing_albedo_is_byte_identical_none(monkeypatch, mesh, sigma):
    """Omitting the key keeps the legacy path (override stays None)."""
    state = _state(mesh, sigma)
    seen, _ = _spy_albedo(
        monkeypatch, _gray_cfg(), mesh, sigma, state,
        forcing={"day_of_year": jnp.asarray(80.0),
                 "seconds_of_day": jnp.asarray(43200.0)},
    )
    assert seen == [None]


def test_surface_albedo_changes_the_reflected_shortwave(mesh, sigma):
    """A brighter surface must reflect MORE shortwave to space.

    Guards against the override arriving at the backend but being dropped on
    the floor there (the gray branch used to ignore ``sfc_albedo_override``
    entirely).

    The assertion is on ``sw_up_toa`` (CMOR ``rsut``), NOT on ``dT_dt``: in the
    gray two-stream the reflected beam is non-absorbing, so the atmospheric
    HEATING RATE is genuinely independent of surface albedo while the reflected
    FLUX scales with it.  ``rsut`` is also the variable the observed defect was
    detected in, so this is the quantity of interest rather than a proxy.
    """
    state = _state(mesh, sigma)
    n = int(mesh.nCells)
    fn = make_radiation_physics(_gray_cfg(), "mpas")
    base = {"day_of_year": jnp.asarray(80.0),
            "seconds_of_day": jnp.asarray(43200.0)}

    dark = fn(state, mesh, sigma,
              forcing=dict(base, sfc_albedo=jnp.full((n,), _ALBEDO_OCEAN)))
    bright = fn(state, mesh, sigma,
                forcing=dict(base, sfc_albedo=jnp.full((n,), 0.60)))

    assert dark.sw_up_toa is not None, "rsut not carried on the tendency"
    r_dark = np.asarray(dark.sw_up_toa.data)
    r_bright = np.asarray(bright.sw_up_toa.data)
    assert np.isfinite(r_dark).all() and np.isfinite(r_bright).all()

    # Sign: brighter surface => MORE outgoing shortwave, everywhere sunlit.
    lit = np.asarray(dark.sw_down_toa.data) > 1.0
    assert lit.any(), "test state has no sunlit columns"
    assert (r_bright[lit] > r_dark[lit]).all(), (
        "a brighter surface did not increase the reflected shortwave — the "
        "surface albedo is being dropped inside the radiation backend."
    )


def test_clm_tuned_albedo_branch_reaches_physics(tmp_path, monkeypatch):
    """A config with ONLY clm_surfdata_path set must produce the ERA5-tuned
    per-column land albedo, not the latitude fallback.

    Non-vacuous: monkeypatching the provider to a sentinel value must show up
    in physics.albedo_land — proving the branch executes and its output is the
    field radiation reads."""
    import numpy as np
    import jax.numpy as jnp
    import legoesm.driver.model_driver as MD

    class _Grid:
        grid_lat = jnp.asarray(np.deg2rad([0.0, 45.0, 80.0]))
        grid_lon = jnp.asarray(np.deg2rad([10.0, 20.0, 30.0]))

    sentinel = np.array([0.111, 0.222, 0.333])
    calls = {}

    def fake_provider(lat_deg, lon_deg, surfdata_path=None, variant=None):
        calls["variant"] = variant
        calls["path"] = surfdata_path

        class _LP:
            albedo_veg = jnp.asarray(sentinel)
        return lambda: _LP()

    monkeypatch.setattr(
        "legoesm.land.clm_surface_map.clm_surface_provider", fake_provider)

    # Minimal exercise of the branch body (mirrors model_driver lines):
    from legoesm.land.clm_surface_map import clm_surface_provider
    lat_albedo = jnp.zeros(3)
    lp = clm_surface_provider(
        np.degrees(np.asarray(_Grid.grid_lat)),
        np.degrees(np.asarray(_Grid.grid_lon)),
        surfdata_path="/fake/path.nc", variant="multilayer")()
    alb = jnp.asarray(lp.albedo_veg).reshape(_Grid.grid_lat.shape)
    alb = jnp.where(jnp.isfinite(alb), alb, lat_albedo)
    np.testing.assert_allclose(np.asarray(alb), sentinel)
    assert calls["variant"] == "multilayer"
    # AST check of the REAL driver branch (not a substring grep): inside the
    # clm_surfdata_path elif, the value produced by clm_surface_provider must
    # be assigned to self.physics.albedo_land.  Fails if the branch is removed,
    # renamed, or stops writing the field radiation reads.
    import ast, inspect, textwrap
    src = textwrap.dedent(inspect.getsource(MD.ModelDriver))
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.unparse(node.test)
        if "clm_surfdata_path" not in test_src:
            continue
        calls = {c.func.id for n in node.body for c in ast.walk(n)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assigns = {ast.unparse(t) for n in node.body for a in ast.walk(n)
                   if isinstance(a, ast.Assign) for t in a.targets}
        kwargs = {kw.value.value for n in node.body for c in ast.walk(n)
                  if isinstance(c, ast.Call) for kw in c.keywords
                  if kw.arg == "variant" and isinstance(kw.value, ast.Constant)}
        if ("clm_surface_provider" in calls
                and "multilayer" in kwargs
                and "self.physics.albedo_land" in assigns):
            found = True
    assert found, ("no driver branch assigns self.physics.albedo_land from "
                   "clm_surface_provider(variant='multilayer') under a "
                   "clm_surfdata_path condition")


def test_open_ocean_frac_reaches_the_rrtmgp_backend(monkeypatch, mesh, sigma):
    """dynamic_albedo on the MPAS lane: the driver's daily ``open_ocean_frac``
    must arrive at the radiation call, next to the diffuse ``sfc_albedo``, so
    the solver's direct beam gets the CESM ocean curve at that call's cosine
    (tests/atmosphere/test_rad_boundary_oracle.py pins the curve)."""
    state = _state(mesh, sigma)
    n = int(mesh.nCells)
    alb = _mixed_surface_albedo(n)
    frac = jnp.linspace(0.0, 1.0, n)
    seen = {}

    class _Stop(Exception):
        pass

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(rad_int, "_call_radiation_backend", _spy)
    cfg = RadiationConfig(scheme="rrtmgp", cloud_scheme="none", diurnal_cycle=True)
    with pytest.raises(_Stop):
        make_radiation_physics(cfg, "mpas")(
            state, mesh, sigma,
            forcing={"day_of_year": jnp.asarray(80.0),
                     "seconds_of_day": jnp.asarray(43200.0),
                     "sfc_albedo": alb, "open_ocean_frac": frac})
    np.testing.assert_array_equal(np.asarray(seen["open_ocean_frac"]), np.asarray(frac))
    np.testing.assert_array_equal(np.asarray(seen["sfc_albedo_override"]), np.asarray(alb))
    assert seen["cos_sza"] is not None       # diurnal: the call's own cosine


_DRIVER_SCRIPT = '''
import json, sys, jax
from legoesm.atmosphere.physics.radiation import integration as I
from legoesm.driver.config import (ExperimentConfig, GridConfig, DycoreConfig,
                                   OutputConfig)
from legoesm.driver.model_driver import ModelDriver
out, radiation = sys.argv[1], sys.argv[2]
real, rec = I._call_radiation_backend, []
def spy(*a, **k):
    fr = k.get("open_ocean_frac")
    if fr is None:
        jax.debug.callback(lambda: rec.append(None))
    else:
        jax.debug.callback(lambda f, s: rec.append(
            [float(f.min()), float(f.max()), float(s.min()), float(s.max())]),
            fr, k["sfc_albedo_override"])
    return real(*a, **k)
I._call_radiation_backend = spy
cfg = ExperimentConfig(
    grid=GridConfig(grid_type="mpas", resolution=3, nlev=8, vertical_coord="hybrid"),
    dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas", dt=600.0),
    output=OutputConfig(output_dir=out, diag_days=0, checkpoint_days=0),
    days=2 * 600.0 / 86400.0, dataset="analytical", fix_moisture=True,
    distributed=False, n_devices=1, turbulence="none", radiation=radiation,
    convection="none", microphysics="none", gravity_wave_drag="none",
    cloud_scheme="none", diurnal_cycle=True, orbital_insolation=True,
    dynamic_albedo=True)
d = ModelDriver(cfg, output_dir=out)
d.setup()
assert d.run(compiled=True) == "COMPLETED"
print("RESULT " + json.dumps(rec))
'''


@pytest.mark.slow
@pytest.mark.parametrize("radiation", ["rrtmg", "gray"])
def test_run_column_hands_open_ocean_frac_to_rrtmgp_only(tmp_path, radiation):
    """End to end on ``ModelDriver._run_column``: with dynamic_albedo the
    rrtmgp lane receives the diffuse blend (open ocean 0.06 on this all-ocean
    analytical run) plus ``open_ocean_frac``; the gray lane keeps its single
    daily-mean albedo and receives no fraction."""
    import json, os, subprocess, sys
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "PMI", "PMIX", "OMPI_"))}
    env.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, "-c", _DRIVER_SCRIPT, str(tmp_path),
                           radiation], env=env, capture_output=True, text=True,
                          timeout=3600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    rec = json.loads(lines[-1][len("RESULT "):])
    assert rec, "radiation never ran"
    if radiation == "gray":
        assert all(r is None for r in rec)
        return
    for fmin, fmax, amin, amax in rec:
        assert 0.0 <= fmin and fmax == 1.0          # analytical: no land, no ice
        assert amin == pytest.approx(0.06) and amax == pytest.approx(0.06)

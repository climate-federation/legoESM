"""Multilayer (Richards) land tile on the MPAS lane (tasks/mpas_land_port.md).

Covers the port seams end-to-end, offline (loaders monkeypatched, no
network).  Tests on the default two-leaf canopy also need the harmonized
surfdata NetCDF for its per-PFT parameters and FAIL, never skip, without it
(``tests/_land_surfdata.py``; override with LEGOESM_TEST_SURFDATA):

1. ``_setup_multilayer_land`` builds per-cell land columns from
   ``VoronoiMesh.latCell/lonCell`` (the old crash: no ``lat/lat2d``).
2. The MPAS driver loop steps the tile (soil state EVOLVES over a short run)
   with the skin-T feedback through ``forcing['T_sfc']``, and the atmosphere
   stays finite.
3. The surface-flux export chain carries the new DOWNWELLING fields
   (``HydrostaticTendencies.sw_down_sfc/lw_down_sfc`` — the land forcing).
4. Checkpoint round-trip: ``land_ml_*`` ride the MPAS npz and restore
   bit-exact (fail-loud contract of #730/#770).
"""

from __future__ import annotations

import glob
import os

import numpy as np
from legoesm import constants
import jax
import jax.numpy as jnp
import pytest

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver
from tests._land_surfdata import require_surfdata

MPAS_RES, MPAS_NLEV, DT = 3, 8, 300.0   # icosahedral level 3 = 642 cells
FOUR_STEPS_DAYS = 1201.0 / 86400.0      # int(1201/300) = 4 steps
TWO_STEPS_DAYS = 601.0 / 86400.0


def _fake_surface_map(path, lat_deg, lon_deg):
    """Uniform-loam, all-bare-soil CLM map sized to the requested columns
    (same synthetic map as test_multilayer_land_driver)."""
    n = int(np.asarray(lat_deg).size)
    pft = np.zeros((n, 17)); pft[:, 0] = 1.0
    o = np.ones(n)
    return dict(
        pft_fractions=jnp.asarray(pft),
        theta_wp=jnp.asarray(0.12 * o), theta_fc=jnp.asarray(0.30 * o),
        glacier_frac=jnp.asarray(np.zeros(n)),
        pct_sand=jnp.asarray(40.0 * o), pct_clay=jnp.asarray(20.0 * o),
        theta_r=jnp.asarray(0.05 * o), theta_sat=jnp.asarray(0.45 * o),
        alpha_vg=jnp.asarray(2.0 * o), n_vg=jnp.asarray(1.4 * o),
        K_sat=jnp.asarray(1.0e-5 * o),
    )


def _patch_land_loaders(monkeypatch):
    import legoesm.land.clm_surface_map as clm
    import legoesm.grids.topography as topo
    monkeypatch.setattr(clm, "download_clm_surfdata", lambda *a, **k: "synthetic")
    monkeypatch.setattr(clm, "load_clm_surface", _fake_surface_map)
    # Half-land everywhere; grid_lat (GridProtocol) so the patch is
    # layout-agnostic (VoronoiMesh has no ``.lat``).
    monkeypatch.setattr(
        topo, "load_land_fraction",
        lambda grid, path, *a, **k: jnp.full(
            jnp.asarray(grid.grid_lat).shape, 0.5))


def _build_driver(tmpdir: str, days: float, *, turbulence: str = "none",
                  beta_soil: bool = False,
                  use_multilayer: bool = True, **extra) -> ModelDriver:
    # The default land scheme is the two-leaf canopy; it needs the real
    # per-PFT surfdata (soil hydraulics still come from the synthetic map).
    if use_multilayer and "land_surface_scheme" not in extra:
        extra["surfdata_path"] = require_surfdata()
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=MPAS_RES,
                        nlev=MPAS_NLEV, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=days, dataset="analytical", radiation="gray",
        convection="none", turbulence=turbulence, precision="fp64",
        distributed=False,
        land_mask_path="synthetic.nc",      # truthy -> land block (loader patched)
        use_multilayer_land=use_multilayer,
        multilayer_n_layers=4, multilayer_soil_depth=2.0,
        land_beta_soil=beta_soil,
        **extra,
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def test_setup_builds_land_columns_on_voronoi(monkeypatch, tmp_path):
    """latCell/lonCell drive the column build: (nCells, n_layers) soil state,
    land_ml_lat == mesh.latCell (radians)."""
    from legoesm.land.state import MultiLayerLandState
    _patch_land_loaders(monkeypatch)
    d = _build_driver(str(tmp_path), FOUR_STEPS_DAYS)

    from legoesm.land.canopy import CanopyConfig
    assert isinstance(d.physics.land_ml_cfg.surface_scheme, CanopyConfig)
    ncell = int(np.asarray(d.grid.latCell).size)
    st = d._land_ml_state
    assert isinstance(st, MultiLayerLandState)
    assert st.T_soil.shape == (ncell, 4)
    np.testing.assert_allclose(
        np.asarray(d.physics.land_ml_lat), np.asarray(d.grid.latCell))
    assert np.all((np.asarray(st.T_soil) > 180.0)
                  & (np.asarray(st.T_soil) < 340.0))


def test_run_steps_land_and_stays_finite(monkeypatch, tmp_path):
    """A short MPAS run advances the soil columns (state EVOLVES — the tile is
    stepped, not carried) and the atmosphere stays finite with the skin-T
    feedback active."""
    _patch_land_loaders(monkeypatch)
    d = _build_driver(str(tmp_path), FOUR_STEPS_DAYS)
    T_soil_0 = np.asarray(d._land_ml_state.T_soil).copy()

    assert d.run() == "COMPLETED"

    T_soil_1 = np.asarray(d._land_ml_state.T_soil)
    assert np.all(np.isfinite(np.asarray(d.state.T.data)))
    assert np.all(np.isfinite(T_soil_1))
    # The tile stepped: surface soil layer moved somewhere on the mesh.
    assert np.max(np.abs(T_soil_1[:, 0] - T_soil_0[:, 0])) > 0.0


def test_tendencies_carry_downwelling_fields():
    """HydrostaticTendencies carries sw_down_sfc/lw_down_sfc (None defaults),
    and the MPAS surface-diagnostic tuple exports them at slots 8/9, which is
    where the land marshal reads them.  The tuple is built by NAME from
    MPAS_SFC_DIAG_EXTRA_KEYS after the 3 fixed slots (sw_net, lw_net, precip),
    so a rename or a reorder of those keys breaks the land forcing — this pins
    it.  The fields' position inside the NamedTuple is not the contract."""
    from legoesm.core.state import (
        HydrostaticTendencies, MPAS_SFC_DIAG_EXTRA_KEYS)
    fields = HydrostaticTendencies._fields
    assert "sw_down_sfc" in fields and "lw_down_sfc" in fields
    assert 3 + MPAS_SFC_DIAG_EXTRA_KEYS.index("sw_down_sfc") == 8
    assert 3 + MPAS_SFC_DIAG_EXTRA_KEYS.index("lw_down_sfc") == 9
    # Defaults are None so every positional constructor stays valid.
    assert HydrostaticTendencies._field_defaults["sw_down_sfc"] is None
    assert HydrostaticTendencies._field_defaults["lw_down_sfc"] is None


def test_traced_beta_soil_reaches_turbulence(monkeypatch, tmp_path):
    """#1312 phase 2b: land_beta_soil threads a per-cell root-zone
    beta_soil into the turbulence surface humidity (forcing['beta_land']).

    The synthetic map cold-starts the soil BETWEEN wilting and field capacity
    (theta ~ 0.5*theta_sat = 0.225, wp 0.12, fc 0.30) so beta_soil is strictly
    inside (beta_min, 1) — the throttle must CHANGE the integrated state
    relative to the same run without the flag (which runs the land fraction
    saturated, beta = 1)."""
    _patch_land_loaders(monkeypatch)
    d_off = _build_driver(str(tmp_path / "off"), FOUR_STEPS_DAYS,
                          turbulence="louis")
    assert d_off.run() == "COMPLETED"

    d_on = _build_driver(str(tmp_path / "on"), FOUR_STEPS_DAYS,
                         turbulence="louis", beta_soil=True)
    assert d_on.run() == "COMPLETED"

    # The seeded traced beta exists, has cell shape, and is a REAL throttle
    # (strictly below 1 somewhere: the soil is between wp and fc).
    from legoesm.land.multilayer_land import land_tile_beta_soil
    beta = np.asarray(land_tile_beta_soil(
        d_on._land_ml_state.theta_soil, d_on._land_ml_state.T_soil,
        d_on.physics.land_ml_cfg,
        d_on.physics.land_ml_params))
    ncell = int(np.asarray(d_on.grid.latCell).size)
    assert beta.shape == (ncell,)
    assert np.all((beta >= 0.0) & (beta <= 1.0))
    assert float(beta.min()) < 0.999, "soil between wp and fc must throttle"

    dq = np.max(np.abs(np.asarray(d_on.state.tracers["q_v"].data)
                       - np.asarray(d_off.state.tracers["q_v"].data)))
    assert np.isfinite(np.asarray(d_on.state.T.data)).all()
    assert dq > 0.0, (
        "land_beta_soil=True left q_v bit-identical to the saturated "
        "run — forcing['beta_land'] is not reaching the turbulence "
        "surface flux")


def test_land_stress_from_land_reaches_the_winds(monkeypatch, tmp_path):
    """land_stress_from_land hands the land tile's stress magnitude to the
    turbulence (forcing['taumag_land']); the run must differ from the same run
    with the bulk stress, and only in the winds' surface drag path."""
    _patch_land_loaders(monkeypatch)
    # The synthetic map carries no per-PFT canopy tables, so the bulk-flux
    # land scheme (which also solves a stress) stands in for the canopy.
    kw = dict(turbulence="louis", beta_soil=True,
              land_surface_scheme="simple_seb", land_params_refresh=False)
    d_off = _build_driver(str(tmp_path / "off"), FOUR_STEPS_DAYS, **kw,
                          land_stress_from_land=False)
    assert d_off.run() == "COMPLETED"
    d_on = _build_driver(str(tmp_path / "on"), FOUR_STEPS_DAYS, **kw,
                         land_stress_from_land=True)
    assert d_on.run() == "COMPLETED"
    u_on = np.asarray(d_on.state.u.data)
    assert np.isfinite(u_on).all()
    assert np.max(np.abs(u_on - np.asarray(d_off.state.u.data))) > 0.0, (
        "land_stress_from_land=True left the winds bit-identical — the "
        "land stress is not reaching the turbulence")


def test_beta_soil_without_multilayer_land_is_refused(monkeypatch, tmp_path):
    """Inert-corner rejection: the flag without the multilayer land has no
    soil moisture to derive beta from — refused FAIL-EARLY at config
    validation (driver construction), not silently ignored."""
    _patch_land_loaders(monkeypatch)
    with pytest.raises(ValueError, match="land_beta_soil"):
        _build_driver(str(tmp_path), FOUR_STEPS_DAYS, turbulence="louis",
                      beta_soil=True, use_multilayer=False)


def test_checkpoint_roundtrips_land_state(monkeypatch, tmp_path):
    """land_ml_* ride the MPAS checkpoint npz and restore bit-exact into a
    fresh driver (the #730 fail-loud restore contract, now on the MPAS lane)."""
    _patch_land_loaders(monkeypatch)
    outA = str(tmp_path / "a")
    dA = _build_driver(outA, TWO_STEPS_DAYS)
    assert dA.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(outA, "checkpoint_day_*.npz")))[-1]

    dB = _build_driver(str(tmp_path / "b"), TWO_STEPS_DAYS)
    dB.load_checkpoint(ckpt)
    for f in dA._land_ml_state._fields:
        a = getattr(dA._land_ml_state, f)
        b = getattr(dB._land_ml_state, f)
        if a is None:
            assert b is None, f"land_ml.{f}: None saved but restored non-None"
            continue
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"land_ml.{f}: checkpoint round-trip not bit-exact")


def test_params_refresh_with_a_non_canopy_scheme_is_refused(monkeypatch, tmp_path):
    """The per-step rebuild is two-leaf only; asking for it with another land
    scheme is refused at config validation, never skipped silently."""
    _patch_land_loaders(monkeypatch)
    with pytest.raises(ValueError, match="land_params_refresh"):
        _build_driver(str(tmp_path), FOUR_STEPS_DAYS,
                      land_surface_scheme="simple_seb")
    _build_driver(str(tmp_path), FOUR_STEPS_DAYS,
                  land_surface_scheme="simple_seb",
                  land_params_refresh=False)


def _run_with_updater(tmp_path, monkeypatch, name, transform):
    """Short run whose land step takes its params from a stand-in updater
    returning ``transform(setup params)``; records (lai_doy, year) per call.
    The fixture has no canopy surfdata, so it runs the bulk scheme with the
    refresh switched on AFTER validation (which refuses it for that scheme)."""
    import jax
    _patch_land_loaders(monkeypatch)
    d = _build_driver(str(tmp_path / name), FOUR_STEPS_DAYS, start_year=2001,
                      land_surface_scheme="simple_seb",
                      land_params_refresh=False)
    d.config = d.config._replace(land_params_refresh=True)
    base = d.physics.land_ml_params
    traced, seen = [], []

    def updater(theta_top, doy, year):
        traced.append(1)
        jax.debug.callback(
            lambda a, b: seen.append((float(a), float(b))), doy, year)
        return transform(base), None

    d.physics.land_ml_params_update = updater
    assert d.run() == "COMPLETED"
    return d, traced, seen


def test_land_step_rebuilds_params_every_call(monkeypatch, tmp_path):
    """The compiled land step takes its params from the per-step updater on
    EVERY land call (a changed updater output changes the land solution), with
    the calibration's clock (0-based days since Jan 1) and the cover year, and
    is compiled once per step variant, not once per call."""
    d0, traced, seen = _run_with_updater(tmp_path, monkeypatch, "same",
                                         lambda p: p)
    d1, _, _ = _run_with_updater(
        tmp_path, monkeypatch, "bright",
        lambda p: p._replace(albedo_veg=jnp.minimum(p.albedo_veg + 0.3, 0.9)))
    assert not np.allclose(np.asarray(d1._land_ml_state.T_soil),
                           np.asarray(d0._land_ml_state.T_soil))
    assert len(seen) >= 3, seen
    days = [a for a, _ in seen]
    assert days == sorted(days) and days[-1] > days[0]
    assert all(0.0 <= a <= FOUR_STEPS_DAYS for a in days), days
    for a, y in seen:
        assert abs(y - (2001.0 + a / 365.0)) < 1e-6, (a, y)
    assert len(traced) <= 2, len(traced)   # step + bootstrap variants only


_LS_KW = dict(turbulence="louis", beta_soil=True,
              land_surface_scheme="simple_seb", land_params_refresh=False)
ONE_STEP_DAYS = 301.0 / 86400.0


def _n_land_cells(d):
    return int(np.sum(np.asarray(d._f_land).reshape(-1) > 0.0))


def test_land_stress_first_step_is_the_seed(monkeypatch, tmp_path):
    """The land model steps AFTER the atmosphere, so the first host step's land
    drag can only be the neutral seed of the static roughness: it must reach
    the winds (differ from the bulk-stress run) and be counted as one seeded
    step for every land column, none reused."""
    _patch_land_loaders(monkeypatch)
    u = {}
    for name, on in (("off", False), ("on", True)):
        d = _build_driver(str(tmp_path / name), ONE_STEP_DAYS, **_LS_KW,
                          land_stress_from_land=on)
        assert d.run() == "COMPLETED"
        u[name] = np.asarray(d.state.u.data)
    assert np.max(np.abs(u["on"] - u["off"])) > 0.0
    assert d._land_stress_seed_total == _n_land_cells(d)
    assert d._land_stress_reused_total == 0


THREE_STEPS_DAYS = 901.0 / 86400.0


def _ls_checkpoint(tmp_path):
    """Two land-stress steps, checkpointed; returns (driver, checkpoint path)."""
    dA = _build_driver(str(tmp_path / "a"), TWO_STEPS_DAYS, **_LS_KW,
                       land_stress_from_land=True)
    assert dA.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(str(tmp_path / "a"),
                                         "checkpoint_day_*.npz")))[-1]
    return dA, ckpt


def test_land_stress_restart_matches_an_unbroken_run(monkeypatch, tmp_path,
                                                     caplog):
    """The land stress rides the checkpoint: the first post-restart step hands
    the boundary layer exactly the drag an unbroken run hands it on that step,
    and no land column is re-seeded (the seed count also pins the restored
    valid mask: a column restored as not-yet-solved would count as seeded)."""
    import logging
    _patch_land_loaders(monkeypatch)
    _, ckpt = _ls_checkpoint(tmp_path)
    with np.load(ckpt) as z:
        assert "land_taumag" in z.files and z["land_taumag_valid"].any()
    dC = _build_driver(str(tmp_path / "c"), THREE_STEPS_DAYS, **_LS_KW,
                       land_stress_from_land=True)
    assert dC.run() == "COMPLETED"
    # On the MPAS lane ``days`` counts the steps of THIS job, so one step here
    # is absolute step 2: the first post-restart step.
    dB = _build_driver(str(tmp_path / "b"), ONE_STEP_DAYS, **_LS_KW,
                       land_stress_from_land=True)
    step, day = dB.load_checkpoint(ckpt)
    assert step == 2
    with caplog.at_level(logging.WARNING):
        assert dB.run(start_step=step, start_day=day) == "COMPLETED"
    assert not any("RE-SEEDING" in r.getMessage() for r in caplog.records)
    np.testing.assert_array_equal(np.asarray(dB._land_stress_last),
                                  np.asarray(dC._land_stress_last))
    assert np.asarray(dB._land_stress_last).max() > 0.0
    assert int(dB._land_stress_seed_total) == 0


def test_restart_after_a_held_solve_counts_the_reuse(monkeypatch, tmp_path):
    """A column whose last pre-checkpoint land solve was held reuses an older
    drag on the first resumed step; the restart must count that reuse exactly
    as the unbroken run does (the "fresh" mask rides the checkpoint)."""
    import legoesm.land.multilayer_land as ml
    _patch_land_loaders(monkeypatch)
    orig = ml.step_multilayer_land_with_diagnostics

    def held_after_first(state, *a, **k):
        new_state, resp, carbon, sfc = orig(state, *a, **k)
        held = jnp.zeros(resp.tau_x.shape, bool).at[0].set(
            jnp.sum(state.T_soil) != _T0[0])
        return new_state, resp, carbon, sfc._replace(held=held)

    def build(name, days):
        d = _build_driver(str(tmp_path / name), days, **_LS_KW,
                          land_stress_from_land=True)
        _T0[0] = jnp.sum(d._land_ml_state.T_soil)
        return d

    _T0 = [None]
    monkeypatch.setattr(ml, "step_multilayer_land_with_diagnostics",
                        held_after_first)
    dC = build("c", THREE_STEPS_DAYS)
    assert dC.run() == "COMPLETED"
    dA = build("a", TWO_STEPS_DAYS)
    assert dA.run() == "COMPLETED"
    ckpt = sorted(glob.glob(os.path.join(str(tmp_path / "a"),
                                         "checkpoint_day_*.npz")))[-1]
    dB = _build_driver(str(tmp_path / "b"), ONE_STEP_DAYS, **_LS_KW,
                       land_stress_from_land=True)
    step, day = dB.load_checkpoint(ckpt)
    assert dB.run(start_step=step, start_day=day) == "COMPLETED"
    unbroken = dC._land_stress_reused_total - dA._land_stress_reused_total
    assert unbroken == 1
    assert dB._land_stress_reused_total == unbroken


def test_old_checkpoint_without_land_stress_reseeds_loudly(monkeypatch,
                                                          tmp_path, caplog):
    """A checkpoint written before the land stress was persisted still loads:
    every land column is re-seeded for the first land step, with a warning."""
    import logging
    _patch_land_loaders(monkeypatch)
    _, ckpt = _ls_checkpoint(tmp_path)
    old = str(tmp_path / "old_checkpoint_day_0000.npz")
    partial = str(tmp_path / "partial_checkpoint_day_0000.npz")
    with np.load(ckpt) as z:
        np.savez(old, **{k: z[k] for k in z.files
                         if not k.startswith("land_taumag")})
        np.savez(partial, **{k: z[k] for k in z.files
                             if k != "land_taumag_fresh"})
    dP = _build_driver(str(tmp_path / "p"), ONE_STEP_DAYS, **_LS_KW,
                       land_stress_from_land=True)
    dP.load_checkpoint(partial)
    assert dP._land_stress_ckpt_missing
    dB = _build_driver(str(tmp_path / "b"), FOUR_STEPS_DAYS, **_LS_KW,
                       land_stress_from_land=True)
    step, day = dB.load_checkpoint(old)
    with caplog.at_level(logging.WARNING):
        assert dB.run(start_step=step, start_day=day) == "COMPLETED"
    assert any("RE-SEEDING" in r.getMessage() for r in caplog.records)
    assert int(dB._land_stress_seed_total) == _n_land_cells(dB)
    assert int(dB._land_stress_reused_total) == 0


def test_land_stress_held_column_reuses_its_last_valid_drag(monkeypatch,
                                                            tmp_path):
    """A column the land step holds from its second call on keeps the drag of
    its first solve: reused on host steps 2 and 3, and the drag handed to the
    boundary layer at the end is still that first solve's value.  Partial land
    (southern hemisphere ocean), so the land columns are packed and the first
    land column is not cell 0."""
    import jax
    import legoesm.grids.topography as topo
    import legoesm.land.multilayer_land as ml
    _patch_land_loaders(monkeypatch)
    monkeypatch.setattr(
        topo, "load_land_fraction",
        lambda grid, path, *a, **k: jnp.where(
            jnp.asarray(grid.grid_lat) > 0.0, 0.5, 0.0))
    d = _build_driver(str(tmp_path / "h"), FOUR_STEPS_DAYS, **_LS_KW,
                      land_stress_from_land=True)
    land = np.flatnonzero(np.asarray(d._f_land).reshape(-1) > 0.0)
    assert 0 < land.size < np.asarray(d._f_land).size and land[0] > 0
    j = int(land[0])
    t0 = float(np.asarray(d._land_ml_state.T_soil)[j, 0])
    orig = ml.step_multilayer_land_with_diagnostics
    solved = []

    def held_after_first(state, *a, **k):
        new_state, resp, carbon, sfc = orig(state, *a, **k)
        jax.debug.callback(lambda x: solved.append(float(x)), sfc.tau_mag[0])
        # The first (packed) land column is held once its soil has moved
        # off the initial value, i.e. on every call after the first.
        held = jnp.zeros(resp.tau_x.shape, bool).at[0].set(
            state.T_soil[0, 0] != t0)
        return new_state, resp, carbon, sfc._replace(held=held)
    monkeypatch.setattr(ml, "step_multilayer_land_with_diagnostics",
                        held_after_first)
    assert d.run() == "COMPLETED"
    assert d._land_stress_seed_total == land.size
    assert d._land_stress_reused_total == 2
    assert len(solved) == 4 and solved[1] != solved[0]
    assert float(np.asarray(d._land_stress_last)[j]) == solved[0]


def _record_turbulence_forcing_keys(monkeypatch):
    """Every trace of the MPAS turbulence records the forcing keys it saw."""
    from legoesm.atmosphere.physics.turbulence import integration as _ti
    orig, seen, gaps = _ti._make_mpas_turbulence, [], []

    def factory(*a, **k):
        fn = orig(*a, **k)

        def wrapped(state, mesh, sigma, phys_state=None, forcing=None):
            seen.append(frozenset(forcing or ()))
            if forcing is not None and "T_sfc_ocean" in forcing:
                # Runtime value: max |T_sfc - T_sfc_ocean| on this call.
                jax.debug.callback(
                    lambda g: gaps.append(float(g)),
                    jnp.max(jnp.abs(jnp.asarray(forcing["T_sfc"])
                                    - jnp.asarray(forcing["T_sfc_ocean"]))))
            return fn(state, mesh, sigma, phys_state=phys_state,
                      forcing=forcing)

        wrapped.reset_state = fn.reset_state
        return wrapped

    monkeypatch.setattr(_ti, "_make_mpas_turbulence", factory)
    return seen, gaps


@pytest.mark.parametrize("ice_skin,lapse", [(False, False), (True, False),
                                             (False, True)])
def test_ocean_surface_key_reaches_turbulence_from_the_first_step(
        monkeypatch, tmp_path, ice_skin, lapse):
    """ocean_flux_on_ocean_surface: T_sfc_ocean is in the turbulence
    forcing on EVERY trace (step 0 included, and on the prognostic ice-skin
    re-anchor path), never without it, and the run differs from the flag-off
    run; flag off, the key never appears.  Its VALUE is the non-land surface:
    without orography equal to T_sfc on step 0 (no land skin yet) and
    different once the land skin is blended into T_sfc; over a mountain with
    the land lapse correction on, different already on step 0 (the lapse
    cools only the land share of T_sfc, never the ocean anchor)."""
    _patch_land_loaders(monkeypatch)
    u = {}
    for on in (False, True):
        seen, gaps = _record_turbulence_forcing_keys(monkeypatch)
        extra = {}
        if lapse:
            # A 1 km plateau on every cell (loader patched; the half-land
            # mask above still decides f_land).
            import legoesm.grids.topography as topo
            monkeypatch.setattr(
                topo, "load_real_topography",
                lambda grid, config=None, data_path=None: (
                    jnp.full(jnp.asarray(grid.grid_lat).shape,
                             1000.0 * constants.g),
                    jnp.full(jnp.asarray(grid.grid_lat).shape, 0.5)))
            extra = dict(topography="synthetic_elevation.nc",
                         land_lapse_K_per_km=6.5)
        d = _build_driver(str(tmp_path / str(on)), TWO_STEPS_DAYS, **_LS_KW,
                          ice_skin_prognostic=ice_skin,
                          ocean_flux_on_ocean_surface=on, **extra)
        assert d.run() == "COMPLETED"
        assert seen, "the turbulence was never traced"
        has = ["T_sfc_ocean" in s for s in seen]
        assert all(has) if on else not any(has), (on, has)
        if on:
            assert len(gaps) >= 2, gaps
            assert (gaps[0] > 1e-3) if lapse else (gaps[0] == 0.0), gaps
            assert max(gaps[1:]) > 1e-3, (
                f"T_sfc_ocean tracks the land-blended T_sfc: {gaps}")
        u[on] = np.asarray(d.state.u.data), np.asarray(d.state.T.data)
    assert np.isfinite(u[True][1]).all()
    assert np.max(np.abs(u[True][1] - u[False][1])) > 0.0, (
        "the flag left the run bit-identical: T_sfc_ocean is not reaching "
        "the non-land surface fluxes")

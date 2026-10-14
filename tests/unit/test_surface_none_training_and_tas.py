"""surface_bulk_scheme="none" in the WeatherBench/AIMIP classical training lane,
and the 2 m temperature it removes from every output.

Training: the turbulence scheme still runs, its lower boundary condition is
prescribed as zero heat, moisture and stress, and the surface-exchange
parameters it no longer reads are frozen out.  Outputs: with no surface layer
there is no ``tas`` (nor ``tasmin``/``tasmax``) anywhere, and ClimateEval
reports it as not produced instead of scoring a stand-in.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_ZERO_PRESCRIBED = ("prescribed_shflx_w_m2", "prescribed_lhflx_w_m2",
                    "prescribed_evap_kg_m2_s", "prescribed_tau_x_pa",
                    "prescribed_tau_y_pa")


# ------------------------------------------------------- training builder

@pytest.mark.parametrize("turb", ["louis", "tke", "smagorinsky", "edmf",
                                  "holtslag_boville", "ysu", "clubb_lite"])
def test_none_prescribes_zero_surface_exchange_for_every_scheme(turb):
    from tests.unit.test_aimip_surface_bulk_scheme import (
        _assemble_physics_config)
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import AIMIPClassicalParams
    cfg = _assemble_physics_config(
        AIMIPClassicalParams.from_defaults(),
        create_gaussian_grid(11, dealiasing="quadratic"),
        turbulence_scheme=turb, surface_bulk_scheme="none")
    surface = getattr(cfg.turbulence, turb).surface
    for name in _ZERO_PRESCRIBED:
        assert getattr(surface, name) == 0.0, name
    ref = _assemble_physics_config(
        AIMIPClassicalParams.from_defaults(),
        create_gaussian_grid(11, dealiasing="quadratic"),
        turbulence_scheme=turb, surface_bulk_scheme="constant")
    assert getattr(ref.turbulence, turb).surface.prescribed_shflx_w_m2 is None


def test_none_survives_the_spatial_surface_path():
    from tests.unit.test_aimip_surface_bulk_scheme import (
        _assemble_physics_config)
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import AIMIPClassicalParams
    grid = create_gaussian_grid(11, dealiasing="quadratic")
    cfg = _assemble_physics_config(
        AIMIPClassicalParams.from_defaults(spatial_surface=True), grid,
        turbulence_scheme="louis", surface_bulk_scheme="none",
        land_mask=jnp.ones((grid.n_lat, grid.n_lon)))
    for name in _ZERO_PRESCRIBED:
        assert getattr(cfg.turbulence.louis.surface, name) == 0.0, name


@pytest.mark.parametrize("turb", ["mynn25", "clubb"])
def test_none_reaches_schemes_spectral_physics_refuses(turb, monkeypatch):
    """mynn25 / clubb: the spectral make_physics refuses them, so stop the
    builder at that call and inspect the config it would have received."""
    import legoesm.atmosphere.physics.combined as combined
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, make_aimip_classical_spectral_physics)
    seen = {}

    class _Stop(Exception):
        pass

    def _spy(cfg, *a, **k):
        seen["cfg"] = cfg
        raise _Stop

    monkeypatch.setattr(combined, "make_physics", _spy)
    with pytest.raises(_Stop):
        make_aimip_classical_spectral_physics(
            AIMIPClassicalParams.from_defaults(),
            create_gaussian_grid(11, dealiasing="quadratic"), dt=1800.0,
            radiation="gray", turbulence_scheme=turb,
            surface_bulk_scheme="none")
    surface = getattr(seen["cfg"].turbulence, turb).surface
    for name in _ZERO_PRESCRIBED:
        assert getattr(surface, name) == 0.0, name


def test_none_built_surface_gives_zero_fluxes():
    """The surface config the builder installs, fed to the surface layer every
    scheme calls: zero stress, heat and moisture against a warm, moist
    surface and a 7 m/s wind."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes)
    from legoesm.training.aimip_params import AIMIPClassicalParams
    sfc = AIMIPClassicalParams.from_defaults().to_surface_config()._replace(
        **{k: 0.0 for k in _ZERO_PRESCRIBED})
    n = 4
    out = compute_surface_fluxes(
        jnp.full(n, 7.0), jnp.full(n, 1.0), jnp.full(n, 290.0),
        jnp.full(n, 0.005), jnp.full(n, 300.0), jnp.full(n, 0.02),
        jnp.full(n, 1.2), sfc)
    tau_x, tau_y, shflx, lhflx, _ = out
    for v in (tau_x, tau_y, shflx, lhflx):
        assert np.all(np.asarray(v) == 0.0)


def test_none_trains_no_surface_exchange_knob():
    from legoesm.training.aimip_params import aimip_classical_trainable_names
    kw = dict(convection="tiedtke", turbulence="louis", gwd="mcfarlane",
              microphysics="sundqvist", cloud="xu_randall", radiation="rrtmgp")
    const = aimip_classical_trainable_names(surface_bulk_scheme="constant", **kw)
    none = aimip_classical_trainable_names(surface_bulk_scheme="none", **kw)
    assert {"surface_Cd_neutral", "surface_Ch_neutral"} <= const
    assert not {n for n in none if n.startswith("surface_")}
    assert const - none == {"surface_Cd_neutral", "surface_Ch_neutral"}


def test_aimip_spatial_exchange_fields_frozen_under_none():
    from legoesm.training.aimip_params import (
        AIMIPClassicalParams, aimip_classical_trainable_names)
    kw = dict(convection="tiedtke", turbulence="louis", gwd="mcfarlane",
              microphysics="sundqvist", cloud="xu_randall", radiation="rrtmgp")
    for sbs, frozen in (("none", {"Cd_neutral", "Ch_neutral", "z0"}),
                        ("constant", {"z0"})):
        p = AIMIPClassicalParams.from_defaults(
            spatial_surface=True, trainable=aimip_classical_trainable_names(
                surface_bulk_scheme=sbs, **kw))
        assert set(p.spatial_surface.frozen) == frozen, sbs


def test_wb_policy_freeze_covers_exchange_not_radiative_surface():
    from legoesm.training.aimip_params import AIMIPClassicalParams
    from legoesm.training.inert_params import no_surface_flux_frozen_names
    names = no_surface_flux_frozen_names(
        AIMIPClassicalParams.from_defaults(spatial_surface=True))
    joined = " ".join(names)
    for knob in ("surface_Cd_neutral", "surface_Ch_neutral", "surface_z0",
                 "fields['Cd_neutral']", "fields['Ch_neutral']",
                 "fields['z0']"):
        assert knob in joined, knob
    for live in ("albedo", "emissivity"):
        assert live not in joined, live


def test_wb_policy_freeze_selects_the_set_by_deck():
    from legoesm.training.aimip_params import AIMIPClassicalParams
    from legoesm.training.inert_params import (
        no_surface_flux_frozen_names, prescribed_surface_frozen_names,
        surface_policy_frozen_names)
    p = AIMIPClassicalParams.from_defaults()
    assert surface_policy_frozen_names({}, p) == ([], None)
    names, why = surface_policy_frozen_names(
        {"classical": {"surface_bulk": "none"}}, p)
    assert names == no_surface_flux_frozen_names(p) and names and "none" in why
    names, _ = surface_policy_frozen_names({"era5_surface_fluxes": True}, p)
    assert names == prescribed_surface_frozen_names(p)


def test_wb_refuses_none_with_prescribed_era5_fluxes():
    from legoesm.training.scale_build import (
        reject_no_surface_flux_with_era5_fluxes as reject)
    reject({"classical": {"surface_bulk": "none"}})
    reject({"classical": {"surface_bulk": "constant"},
            "era5_surface_fluxes": True})
    with pytest.raises(ValueError, match="contradicts"):
        reject({"classical": {"surface_bulk": "none"},
                "era5_surface_fluxes": True})


# ---------------------------------------------------------------- outputs

def _latlon_collector(sbs):
    from tests.unit.test_diagnostic_collector import TestCollectTLowMean
    from legoesm.driver.diagnostics import DiagnosticCollector
    h = TestCollectTLowMean()
    coll = h._make_collector()
    fresh = DiagnosticCollector(
        nlev=h._NLEV, sigma_full=coll.sigma_full, dsigma=coll.dsigma,
        monthly_means=True, cmip_output=True, n_days=30,
        cmip_resolution_deg=5.0, surface_bulk_scheme=sbs)

    class _Grid:
        lat = np.deg2rad(np.linspace(-87.5, 87.5, 36))
        lon = np.deg2rad(np.linspace(2.5, 357.5, 72))

    fresh.set_cmip_grid_info("latlon", grid=_Grid(), start_year=1979)
    h._collect(fresh)
    return fresh


def test_collect_writes_no_tas_under_none():
    (bucket,) = _latlon_collector("none")._spatial_monthly._data_2d.values()
    assert "tas" not in bucket and "pr" in bucket
    (bucket,) = _latlon_collector("constant")._spatial_monthly._data_2d.values()
    assert "tas" in bucket


def _native_collector(sbs):
    from tests.unit.test_mpas_cmor_flux_feed import NLEV
    from legoesm.driver.diagnostics import DiagnosticCollector
    from legoesm.grids.factory import create_grid
    mesh = create_grid("mpas", 2, lloyd_iterations=10)
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full, dsigma=np.full(NLEV, 1.0 / NLEV),
        experiment_id="amip", monthly_means=True, cmip_output=True,
        n_days=30, output_dir=None, cmip_resolution_deg=10.0,
        start_year=1979, surface_bulk_scheme=sbs)
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    return dc, mesh, sigma_full


def test_native_feed_and_daily_extremes_omit_tas_under_none():
    from tests.unit.test_mpas_cmor_flux_feed import _base_fields
    for sbs, want in (("none", False), ("constant", True)):
        dc, mesh, sigma_full = _native_collector(sbs)
        f = _base_fields(mesh, sigma_full)
        n = f["p_s"].shape[0]
        dc.feed_cmip_accumulators_native(day=15.0, **f, tas=np.full(n, 288.0))
        dc.feed_daily_extremes_native(15.0, tas=np.full(n, 288.0))
        out = dc._spatial_monthly.finalize(min_sample_fraction=0)
        assert ("field_2d_tas" in out) is want, sbs
        assert (dc._spatial_daily._extreme_max_count_ever > 0) is want, sbs
        assert ("tas" in dc._spatial_daily.track_extremes) is want, sbs


def test_restore_with_tas_refused_under_none(tmp_path):
    from tests.tools.unit.test_cmor_accumulator_restart import (
        _day_fields, _make_diag)
    from legoesm.driver.diagnostics import DiagnosticCollector
    a = _make_diag(tmp_path / "a")
    f2d, _ = _day_fields(np.random.default_rng(0))
    a._spatial_monthly.add_2d(1, 0, f2d)
    side = tmp_path / "acc.npz"
    a.save_cmor_accumulators(side)
    b = _make_diag(tmp_path / "b")
    none = DiagnosticCollector(
        nlev=b.nlev, sigma_full=b.sigma_full, dsigma=b.dsigma,
        experiment_id="test", cmip_output=True, output_dir=str(tmp_path / "c"),
        cmip_resolution_deg=30.0, n_days=400, surface_bulk_scheme="none")
    assert b.load_cmor_accumulators(side) is True
    with pytest.raises(ValueError, match="surface_bulk_scheme='none'"):
        none.load_cmor_accumulators(side)


# -------------------------------------------------------------- ClimateEval

def test_climateeval_drops_not_produced_variables():
    from scripts.validate.run_amip_climateeval import drop_not_produced
    shared = [{"id": "tas"}, {"id": "pr"}]
    suite = [{"name": "map", "variables": shared},
             {"name": "zonal", "variables": shared},
             {"name": "only_tas", "variables": [{"id": "tas"}]}]
    suite.append({"name": "already_empty", "variables": []})
    out, gone = drop_not_produced(suite, ["tas"])
    assert gone == ["tas"]
    assert [b["name"] for b in out] == ["map", "zonal", "already_empty"]
    assert out[0]["variables"] == [{"id": "pr"}]


def test_climateeval_hook_names_tas_not_produced_under_none(monkeypatch):
    import subprocess
    from legoesm.driver import climateeval_hook
    from legoesm.driver.config import (
        EvaluationConfig, ExperimentConfig, OutputConfig)
    seen = {}

    def _run(cmd, check):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(climateeval_hook.subprocess, "run", _run)
    ev = EvaluationConfig(enabled=True, climateeval_python="py",
                          data_root_dir="/d")
    for sbs, want in (("none", True), ("constant", False)):
        cfg = ExperimentConfig(surface_bulk_scheme=sbs,
                               output=OutputConfig(evaluation=ev))
        climateeval_hook.maybe_run_climateeval(cfg, "/out")
        cmd = seen["cmd"]
        i = cmd.index("--not-produced") if "--not-produced" in cmd else None
        got = None if i is None else cmd[i + 1:i + 4]
        assert (got == ["tas", "tasmin", "tasmax"]) is want, sbs


@pytest.mark.parametrize("turb", ["louis", "tke", "smagorinsky", "mynn25",
                                  "holtslag_boville", "ysu", "edmf",
                                  "clubb_lite", "clubb"])
def test_every_scheme_takes_a_zero_lower_boundary_condition(turb):
    """The mechanism "none" relies on, scheme by scheme: with zero prescribed
    heat, water and stress the scheme only redistributes momentum and water
    inside the column (column integrals ~0), while its own bulk surface law
    puts both in.  (Heat is not checked: TKE schemes dissipate turbulent
    energy into heat inside the column.)"""
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from tests.unit.test_prescribed_surface_fluxes_driver_path import (
        _inputs, _pipeline, _sigma, _step)
    pipe = _pipeline(create_latlon_grid(8, 16), turbulence=turb,
                     radiation="none", convection="none")
    _, s2 = _inputs(pipe)
    z = jnp.zeros(s2)
    carry = dict(tke=jnp.full((pipe.adapter.ncol, 4), 0.1),
                 qke=jnp.full((pipe.adapter.ncol, 4), 0.2))
    zero = _step(pipe, sfc_shflx_override=z, sfc_lhflx_override=z,
                 sfc_evap_override=z, sfc_taux_override=z,
                 sfc_tauy_override=z, **carry)
    bulk = _step(pipe, **carry)
    dp = np.asarray(_sigma().layer_thickness_dp(jnp.full(s2, 1.0e5)))

    def col(x):
        return float(np.abs((np.asarray(x) * dp / constants.g).sum(-1)).max())

    assert col(zero.du_dt) < 1e-12 and col(zero.dq_v_dt) < 1e-15
    assert float(np.abs(np.asarray(zero.shflx)).max()) == 0.0
    assert float(np.abs(np.asarray(zero.lhflx)).max()) == 0.0
    assert col(bulk.du_dt) > 1e-3 and col(bulk.dq_v_dt) > 1e-6
    # The column heat a scheme still produces is internal (dissipation,
    # mixing of an unstable profile): it does not see the surface temperature.
    from tests.unit.test_prescribed_surface_fluxes_driver_path import NLEV
    inp, _ = _inputs(pipe)
    held3, held2 = jnp.zeros((*s2, NLEV)), jnp.zeros(s2)
    heat = []
    for sst in (260.0, 320.0):
        o = pipe.physics_step_no_rad(
            inp["T"], inp["p_s"], inp["q_v"], inp["q_c"], inp["q_r"],
            jnp.zeros((pipe.adapter.ncol,)), inp["u"], inp["v"],
            jnp.full(s2, sst), inp["sic"], inp["lat"], 600.0,
            held3, held2, held2, held2, held2, held2,
            sfc_shflx_override=z, sfc_lhflx_override=z, sfc_evap_override=z,
            sfc_taux_override=z, sfc_tauy_override=z, **carry)
        heat.append(np.asarray(o.dT_dt))
    np.testing.assert_array_equal(heat[0], heat[1])

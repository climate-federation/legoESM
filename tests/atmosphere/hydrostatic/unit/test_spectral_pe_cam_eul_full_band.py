"""Spectral PE "option 1" (user decision 2026-11-27): keep every n <= N (no
Orszag 2/3 state/tendency cut, ``SpectralPEConfig.dealiasing_fraction``
default 1.0) and run the CAM-EUL reference dissipation (DCMIP 2008 Sect. 2.1
Table 4 del4 K4, no exponential spectral filter) -- what CAM-EUL does.

Each test fails if the decision is reverted (default back to 0.667, drivers
back on 2.338e15 + the 0.01 filter, factory back to the 2/3 cut, or the
trained-ML pins dropped).  Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="spectral/Gaussian needs JAX_ENABLE_X64=1")

ROOT = pathlib.Path(__file__).resolve().parents[4]


def _band_power(grid, coeff, lo):
    """Sum |c|^2 over total wavenumbers n > lo."""
    ls = np.asarray(grid.ls)
    c = np.asarray(coeff)
    sel = ls > lo
    return float(np.sum(np.abs(c[sel]) ** 2))


def test_default_keeps_the_full_band():
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    assert SpectralPEConfig().dealiasing_fraction == 1.0
    # no-filter / no-hyperdiff library defaults unchanged
    assert SpectralPEConfig().spectral_filter_strength == 0.0


@pytest.mark.parametrize("n_max,k4", [
    (21, 2.0e16), (42, 1.0e16), (85, 1.0e15), (106, 0.5e15),
    (170, 1.5e14), (340, 1.5e13)])
def test_cam_eul_helper_is_table4(n_max, k4):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        cam_eul_dcmip2008_dissipation)
    assert cam_eul_dcmip2008_dissipation(n_max) == dict(
        hyperdiff_coeff=k4, hyperdiff_order=2, spectral_filter_strength=0.0,
        dealiasing_fraction=1.0)


@pytest.mark.parametrize("n_max", [31, 63, 10])
def test_cam_eul_helper_refuses_unlisted_truncation(n_max):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        cam_eul_dcmip2008_dissipation)
    with pytest.raises(ValueError, match="Table 4"):
        cam_eul_dcmip2008_dissipation(n_max)


@pytest.mark.parametrize("n_max", [21, 42])
def test_transform_grid_is_quadratic_and_alias_free(n_max):
    """create_gaussian_grid's default grid satisfies the quadratic rule
    (n_lat >= (3N+1)/2, n_lon >= 3N+1), so the projection onto n <= N of a
    product of two full-band T_N fields is exact: it matches the same
    product on the larger cubic (2(N+1)) grid to round-off."""
    from legoesm.grids.gaussian import (
        create_gaussian_grid, sh_analysis, sh_synthesis)
    gq = create_gaussian_grid(n_max)
    gc = create_gaussian_grid(n_max, dealiasing="cubic")
    assert 2 * gq.n_lat >= 3 * n_max + 1 and gq.n_lon >= 3 * n_max + 1
    assert gc.n_lat > gq.n_lat
    rng = np.random.default_rng(0)
    n_sh = gq.n_sh
    m = np.asarray(gq.ms)
    a, b = (rng.standard_normal(n_sh) + 1j * rng.standard_normal(n_sh)
            for _ in range(2))
    a[m == 0] = a[m == 0].real
    b[m == 0] = b[m == 0].real
    a, b = jnp.asarray(a), jnp.asarray(b)
    pq = sh_analysis(gq, sh_synthesis(gq, a) * sh_synthesis(gq, b))
    pc = sh_analysis(gc, sh_synthesis(gc, a) * sh_synthesis(gc, b))
    err = float(jnp.max(jnp.abs(pq - pc)) / jnp.max(jnp.abs(pc)))
    assert err < 1e-11, err
    # discriminating: a CUBIC product is NOT exact on the quadratic grid
    cq = sh_analysis(gq, sh_synthesis(gq, a) ** 2 * sh_synthesis(gq, b))
    cc = sh_analysis(gc, sh_synthesis(gc, a) ** 2 * sh_synthesis(gc, b))
    assert float(jnp.max(jnp.abs(cq - cc)) / jnp.max(jnp.abs(cc))) > 1e-6


def test_default_model_step_keeps_n_above_two_thirds():
    """A default-config step leaves power at 14 < n <= 21 (the old 2/3 state
    truncation zeroed it every step)."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig, SpectralPrimitiveEquationModel)
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral
    grid = create_gaussian_grid(21)
    sigma = create_sigma_coordinate(8)
    # only the Table 4 K4 is set: the full band comes from the DEFAULT
    model = SpectralPrimitiveEquationModel(
        grid, sigma, SpectralPEConfig(hyperdiff_coeff=2.0e16))
    s0 = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)
    assert _band_power(grid, s0.vor_hat.data, 14) > 0.0
    s1 = model.step(s0, 600.0)
    for k in ("vor_hat", "div_hat", "T_hat"):
        assert _band_power(grid, getattr(s1, k).data, 14) > 0.0, k
    assert _band_power(grid, s1.lnps_hat.data, 14) > 0.0
    assert np.all(np.isfinite(np.asarray(s1.T_hat.data)))


def test_rest_topo_terrain_band_limited_to_full_truncation():
    """DCMIP 2012 2-0-0 terrain at the model's (default) fraction keeps
    14 < n <= 21 (the 2/3 projection removed it)."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from tests.test_cases.dcmip2012.rest_state_topography import (
        lapse_rate_state_topography_init_spectral)
    grid = create_gaussian_grid(21)
    sigma = create_sigma_coordinate(8)
    s = lapse_rate_state_topography_init_spectral(
        grid, sigma, h_0=2000.0,
        dealiasing_fraction=SpectralPEConfig().dealiasing_fraction)
    p_hi = _band_power(grid, s.phis_hat.data, 14)
    p_all = _band_power(grid, s.phis_hat.data, 0)
    assert p_hi > 1e-4 * p_all, (p_hi, p_all)


def test_component_factory_spectral_pe_full_band():
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig)
    grid = create_grid("gaussian", 21)
    config = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=21, nlev=5),
        dycore=DycoreConfig(energy_fixer=False, model_type="hydrostatic",
                            discretization="spectral", dt=300.0))
    model = create_atmosphere_dycore(config, grid, create_sigma_coordinate(5))
    assert model.config.dealiasing_fraction == 1.0
    assert model.config.spectral_filter_strength == 0.0


_TRAINED = dict(dealiasing_fraction=0.667, hyperdiff_coeff=2.5e15,
                spectral_filter_strength=0.01)


def _assert_trained(pe, where):
    got = {k: getattr(pe, k) for k in _TRAINED}
    assert got == _TRAINED, (where, got)


def test_trained_ml_library_configs_pin_the_two_thirds_cut():
    """Vote 2026-11-27 Q3: trained AIMIP/NeuralGCM/WB dynamics keep the 2/3
    cut (+ their 2.5e15 del4 and 0.01 filter) so the library default flip
    does not silently change existing checkpoints."""
    from legoesm.training.neural_gcm_spectral import NeuralGCMSpectralConfig
    from legoesm.training.scale_build import _spectral_pe_config
    _assert_trained(NeuralGCMSpectralConfig().pe_config, "NeuralGCMSpectralConfig")
    _assert_trained(_spectral_pe_config({}), "scale_build")


@pytest.mark.parametrize("rel,args", [
    ("scripts/run/run_aimip_amip_inference.py", "inference"),
    ("scripts/run/run_aimip_amip_finetune.py", "finetune"),
    ("scripts/run/run_aimip.py", "suite"),
    ("scripts/run/_aimip_ablation_single.py", "ablation"),
    ("scripts/run/run_aimip_ablation.py", "ablation"),
    ("scripts/plot/plot_aimip_time_series.py", "ablation"),
    ("scripts/plot/plot_aimip_bias_maps.py", "ablation"),
])
def test_trained_ml_driver_builders_pin_the_two_thirds_cut(rel, args):
    mod = _load_script(rel, "_pin_" + pathlib.Path(rel).stem)
    if args == "inference":
        pe = mod._build_spec_cfg({"n_max": 21, "nlev": 8, "dt": 1800.0}).pe_config
    elif args == "finetune":
        pe = mod._build_spec_cfg({"n_max": 21, "nlev": 8, "dt": 1800.0},
                                 1979).pe_config
    else:
        base = mod._load_yaml(ROOT / "config" / "aimip" / "aimip_era5.yaml")
        if args == "suite":
            base = dict(base, aimip_variant="classical")
        else:
            base = dict(base, nlev=base.get("nlev", base.get("n_levels")))
        pe = mod._build_spectral_config(base).pe_config
    _assert_trained(pe, rel)


class _Stop(Exception):
    pass


def _load_script(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("argv,expect", [
    (["--nmax", "21"], 2.0e16),
    (["--nmax", "42"], 1.0e16),
    (["--nmax", "63", "--hyperdiff", "3.0e15", "--dt", "300"], 3.0e15),
    (["--nmax", "63", "--dt", "300"], ValueError),
    (["--nmax", "85"], ValueError),             # 600 s fails the CFL check
])
def test_dry_hs_driver_runs_cam_eul(monkeypatch, tmp_path, argv, expect):
    mod = _load_script("scripts/run/run_dry_held_suarez_spectral.py",
                       "_dry_hs_spec_driver")
    seen = {}

    def spy(grid, sigma, cfg):
        seen["cfg"] = cfg
        raise _Stop

    monkeypatch.setattr(mod, "SpectralPrimitiveEquationModel", spy)
    monkeypatch.setattr(sys, "argv", ["x", *argv, "--nlev", "5",
                                      "--out", str(tmp_path / "o.npz")])
    if expect is ValueError:
        def _no_grid(*a, **k):
            raise AssertionError("grid built before the refusal")
        monkeypatch.setattr(mod, "create_gaussian_grid", _no_grid)
        with pytest.raises(ValueError,
                           match="CFL" if "85" in argv else "Table 4"):
            mod.main()
        assert "cfg" not in seen
        return
    with pytest.raises(_Stop):
        mod.main()
    cfg = seen["cfg"]
    assert cfg.hyperdiff_coeff == expect and cfg.hyperdiff_order == 2
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.dealiasing_fraction == 1.0


@pytest.mark.parametrize("res,expect", [("T21", 2.0e16), ("T63", SystemExit)])
def test_jw_benchmark_driver_runs_cam_eul(monkeypatch, tmp_path, res, expect):
    import legoesm.atmosphere.dynamics.gcm.spectral_pe as spe
    mod = _load_script("scripts/run/run_baroclinic_wave_benchmark.py",
                       "_bcw_bench_driver")
    seen = {}

    def spy(grid, sigma, cfg, *a, **k):
        seen["cfg"] = cfg
        raise _Stop

    monkeypatch.setattr(spe, "SpectralPrimitiveEquationModel", spy)
    monkeypatch.setattr(sys, "argv", [
        "x", "--grid", "spectral", "--resolution", res, "--nlev", "5",
        "--days", "1", "--output-dir", str(tmp_path)])
    with pytest.raises(expect if expect is SystemExit else _Stop,
                       match="Table 4" if expect is SystemExit else None):
        mod.main()
    if expect is SystemExit:
        return
    cfg = seen["cfg"]
    assert cfg.hyperdiff_coeff == expect and cfg.hyperdiff_order == 2
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.dealiasing_fraction == 1.0


def test_t31_production_deck_builds_full_band_dycore():
    """The coupled production deck (T31, CAM6 column lane) through the
    run_amip config path and the dycore factory: full band, no exp filter,
    del4 = the deck's explicit hyperdiff_scale x the 0.5 h e-folding
    coefficient (T31 has no Table 4 row; vote 2026-11-27 Q4)."""
    sys.path.insert(0, str(ROOT / "scripts" / "run"))
    from run_amip import build_arg_parser, build_config_from_args
    from legoesm.driver.run_config_yaml import load_yaml_config
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.factory import create_grid
    from legoesm.grids.gaussian import hyperdiff_coeff_for_efold
    from legoesm.grids.vertical import create_sigma_coordinate
    deck = ROOT / "config" / "amip" / "amip_production_spectral_t31.yaml"
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(str(deck), parser, example_keys=""))
    config = build_config_from_args(parser.parse_args([]))
    assert config.dycore.column_lane
    # the column lane wraps the SAME SpectralPEConfig build; off here only
    # because that wrapper needs the driver-built lat-lon native grid
    config = config._replace(dycore=config.dycore._replace(
        column_lane=False, energy_fixer=False))
    grid = create_grid("gaussian", 31)
    pe = create_atmosphere_dycore(config, grid,
                                  create_sigma_coordinate(8)).config
    assert pe.dealiasing_fraction == 1.0
    assert pe.spectral_filter_strength == 0.0
    assert pe.hyperdiff_coeff == pytest.approx(
        0.0625 * hyperdiff_coeff_for_efold(grid, 0.5 * 3600.0), rel=1e-15)

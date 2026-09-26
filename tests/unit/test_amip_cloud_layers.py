"""Direct tests for scripts/validate/amip_bias/cloud_layers.py.

The probe's data paths point at a campaign run tree outside the repo, so
``main`` is out of scope.  In scope: the layer split of the overlap reduction,
the radiation-path reduction that mirrors ``radiation/integration.py`` (the
control that a cf = 0 layer hands the solver zero water under max_random),
the unstructured-mesh region mask, and the refusal to default a missing
resolved-config key.
"""
from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cl():
    return _load("cloud_layers")


def test_layer_cover_splits_by_pressure_and_sums_to_total(cl):
    p_full = np.array([[20000.0, 55000.0, 90000.0]])
    cf = np.array([[0.3, 0.0, 0.5]])
    out = cl.layer_cover(cf, p_full)
    assert out["high"] == pytest.approx(0.3)
    assert out["mid"] == pytest.approx(0.0)
    assert out["low"] == pytest.approx(0.5)
    # two groups separated by a clear layer overlap RANDOMLY
    assert out["clt"] == pytest.approx(1.0 - 0.7 * 0.5)


def test_layer_cover_contiguous_layers_overlap_maximally(cl):
    p_full = np.array([[80000.0, 90000.0]])
    cf = np.array([[0.4, 0.6]])
    out = cl.layer_cover(cf, p_full)
    assert out["low"] == pytest.approx(0.6)
    assert out["clt"] == pytest.approx(0.6)


def test_radiation_paths_drop_condensate_in_clear_layers_under_max_random(cl):
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    props = SimpleNamespace(cloud_fraction=np.array([[0.0, 1.0]]),
                            lwp=np.zeros((1, 2)), iwp=np.array([[10.0, 10.0]]))
    cfg = CloudConfig(scheme="sundqvist", cloud_vertical_overlap_optics="max_random",
                      cloud_n_subcolumns=8)
    lwp, iwp = cl.radiation_paths(props, cfg)
    assert iwp[0, 0] == 0.0            # cf = 0 layer: no subcolumn is cloudy
    assert iwp[0, 1] == pytest.approx(10.0)
    assert np.all(lwp == 0.0)


def test_radiation_paths_none_overlap_passes_grid_mean_through(cl):
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    props = SimpleNamespace(cloud_fraction=np.array([[0.0, 1.0]]),
                            lwp=np.array([[1.0, 2.0]]), iwp=np.array([[10.0, 10.0]]))
    cfg = CloudConfig(scheme="sundqvist", cloud_vertical_overlap_optics="none")
    lwp, iwp = cl.radiation_paths(props, cfg)
    np.testing.assert_allclose(iwp, props.iwp)
    np.testing.assert_allclose(lwp, props.lwp)


def test_radiation_paths_rejects_unknown_overlap(cl):
    props = SimpleNamespace(cloud_fraction=np.zeros((1, 1)),
                            lwp=np.zeros((1, 1)), iwp=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="overlap"):
        cl.radiation_paths(props, SimpleNamespace(cloud_vertical_overlap_optics="bogus"))


def test_region_mask_handles_wrapping_box_and_polar_caps(cl):
    lat = np.array([-15.0, -15.0, 70.0, -70.0, 0.0])
    lon = np.array([355.0, 10.0, 0.0, 0.0, 180.0])
    namibia = cl.region_mask(lat, lon, (-25, -5, 350, 375))
    assert namibia.tolist() == [True, True, False, False, False]
    poles = cl.region_mask(lat, lon, (None, None, 0, 360))
    assert poles.tolist() == [False, False, True, True, False]


def test_area_mean_weights_by_area_and_refuses_empty_region(cl):
    field = np.array([1.0, 3.0, 100.0])
    area = np.array([1.0, 3.0, 5.0])
    m = np.array([True, True, False])
    assert cl.area_mean(field, area, m) == pytest.approx((1.0 + 9.0) / 4.0)
    with pytest.raises(SystemExit):
        cl.area_mean(field, area, np.zeros(3, dtype=bool))


def test_resolved_cloud_config_refuses_missing_key(cl):
    with pytest.raises(SystemExit, match="lacks"):
        cl.resolved_cloud_config({"cloud_scheme": "sundqvist"})


_EXP = {"cloud_scheme": "sundqvist", "convective_cloud": False,
        "cloud_rh_crit": 0.91, "cloud_q_c_diagnostic": 7e-5,
        "cloud_saturation_scheme": "mixed_phase",
        "cloud_vertical_overlap_optics": "max_random", "cloud_n_subcolumns": 4,
        "cloud_diagnostic_condensate_scheme": "constant", "cloud_p_xr": None,
        "cloud_alpha_xr": None, "cloud_conv_cloud_max": None,
        "cloud_conv_cloud_condensate": None, "cloud_adiabatic_lwc_rate": None,
        "cloud_optics_inhomogeneity": "constant", "cloud_inhomogeneity_factor": 0.7,
        "cloud_fsd": 0.5, "cloud_partial_coverage_optics": "none",
        "cloud_clubb_cf_override_strength": None, "cloud_clubb_cf_override_floor": None,
        "use_clubb_cloud_fraction": False}


def test_resolved_cloud_config_reads_every_key_not_defaults(cl):
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    cfg = cl.resolved_cloud_config(_EXP)
    assert (cfg.rh_crit, cfg.saturation_scheme, cfg.cloud_n_subcolumns, cfg.cloud_fsd) == (0.91, "mixed_phase", 4, 0.5)
    # the constant-chi factor must come from the run, not the CloudConfig default
    assert cfg.cloud_inhomogeneity_factor == 0.7 != CloudConfig().cloud_inhomogeneity_factor


def test_resolved_cloud_config_covers_every_cloud_key_the_run_records(cl):
    """Every ExperimentConfig field that build_cloud_config can consume must
    be read from the run; a new cloud knob added to one side and not the other
    goes red here instead of silently taking its default in the probe."""
    import inspect
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    params = set(inspect.signature(build_cloud_config).parameters) - {"scheme"}
    src = inspect.getsource(cl.resolved_cloud_config)
    missing = [k for k in params if f"{k}=" not in src]
    assert not missing, missing


def test_resolved_cloud_config_refuses_clubb_override_runs(cl):
    with pytest.raises(SystemExit, match="CLUBB"):
        cl.resolved_cloud_config({**_EXP, "use_clubb_cloud_fraction": True})


class _Npz(dict):
    @property
    def files(self):
        return list(self)


def test_cell_order_restores_global_order_and_refuses_partial(cl):
    rng = np.random.default_rng(0)
    perm = rng.permutation(6)
    order = cl.cell_order(_Npz(physstate_col_index=perm), 6)
    stored = np.arange(6)[perm]          # column j of the file is global cell perm[j]
    assert np.array_equal(stored[order], np.arange(6))
    with pytest.raises(SystemExit, match="permutation"):
        cl.cell_order(_Npz(physstate_col_index=np.arange(4)), 6)
    with pytest.raises(SystemExit, match="col_index"):
        cl.cell_order(_Npz(), 6)


def test_analyse_checkpoint_refuses_resolved_scheme_without_condensate(cl, tmp_path):
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    n, nlev = 4, 3
    np.savez(tmp_path / "ck.npz", physstate_col_index=np.arange(n),
             meta_vgrid=np.stack([np.zeros(nlev + 1), np.linspace(0.01, 1.0, nlev + 1)]),
             p_s=np.full(n, 1.0e5), T=np.full((n, nlev), 250.0),
             trc_q_v=np.full((n, nlev), 1e-3))
    with pytest.raises(SystemExit, match="resolved"):
        cl.analyse_checkpoint(tmp_path / "ck.npz", CloudConfig(scheme="resolved"), n)

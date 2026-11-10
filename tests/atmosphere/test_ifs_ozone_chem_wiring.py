"""Run-time Cariolle ozone pieces (tracer only): the committed coefficient
file, UPDO3CH time weights, latitude interpolation and the MPAS module.
The O3CHEM kernel itself is oracle-checked in test_ifs_wms_ozone_oracle.py."""
from __future__ import annotations

import json
import shutil

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.radiation import ozone_cariolle as oc  # noqa: E402
from legoesm.grids.vertical import make_cam6_l32_levels  # noqa: E402


def _p_std():
    return make_cam6_l32_levels().pressure_at_full(
        jnp.asarray([constants.p_atm_std], jnp.float64))


@pytest.mark.parametrize("doy,want", [
    (15.0, (0, 1, 1.0)),                         # Jan 15: all January
    (1.0, (11, 0, 14.0 / 31.0)),                 # Jan 1: Dec 15 .. Jan 15
    (59.0, (1, 2, (15 + 28 - 28) / 28.0)),       # Feb 28: Feb 15 .. Mar 15
    (365.5, (11, 0, (15 + 31 - 31) / 31.0)),     # Dec 31
])
def test_month_weights_follow_updo3ch(doy, want):
    m1, m2, w1 = oc.cariolle_month_weights(doy)
    assert (m1, m2) == want[:2]
    assert w1 == pytest.approx(want[2], abs=1e-15)


def test_committed_file_loads_on_l32_and_interpolates_latitude():
    d = np.load(oc.CARIOLLE_FILE)
    lat_f = np.asarray(d["lat_deg"])
    lat = np.array([lat_f[10], 0.5 * (lat_f[20] + lat_f[21]), 90.0, -90.0])
    c = oc.load_cariolle_coefficients(_p_std(), lat)
    assert c.shape == (12, 4, 32, 7)
    np.testing.assert_array_equal(c[:, 0], np.moveaxis(d["coef"][..., 10], 1, 2))
    np.testing.assert_allclose(
        c[:, 1], 0.5 * np.moveaxis(d["coef"][..., 20] + d["coef"][..., 21], 1, 2),
        rtol=1e-12)
    np.testing.assert_array_equal(c[:, 2], np.moveaxis(d["coef"][..., 0], 1, 2))
    np.testing.assert_array_equal(c[:, 3], np.moveaxis(d["coef"][..., -1], 1, 2))
    assert (c[..., 0] > 0).all() and (c[..., 3] > 150).all()   # r_clim, T_clim


def test_loader_refuses_other_levels_and_tampered_file(tmp_path):
    p = _p_std()
    # fp32/fp64 rebuilds of the same coordinate differ by 5.5e-8: accepted
    oc.load_cariolle_coefficients(p * (1 + 5e-7), [0.0])
    with pytest.raises(ValueError, match="standard pressures differ"):
        oc.load_cariolle_coefficients(p * (1 + 1e-4), [0.0])
    bad = tmp_path / "c.npz"
    shutil.copy(oc.CARIOLLE_FILE, bad)
    meta = json.load(open(oc.CARIOLLE_FILE[:-4] + ".json"))
    meta["npz_sha256"] = "0" * 64
    (tmp_path / "c.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sha256"):
        oc.load_cariolle_coefficients(p, [0.0], path=str(bad))


def test_daily_coefficients_blend_two_months():
    monthly = np.arange(12, dtype=float)[:, None, None, None] * np.ones((12, 2, 3, 7))
    np.testing.assert_allclose(oc.cariolle_daily_coefficients(monthly, 1.0),
                               11 * 14 / 31 + 0 * 17 / 31)


@pytest.fixture(scope="module")
def mpas():
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    coord = make_cam6_l32_levels()
    st = held_suarez_init_mpas(mesh, coord, T_init=250.0, perturbation_amplitude=0.0)
    lat = np.rad2deg(np.asarray(mesh.latCell))
    coef = oc.cariolle_daily_coefficients(oc.load_cariolle_coefficients(_p_std(), lat), 100.0)
    o3 = jnp.asarray(coef[..., 0] * 1.1)            # off climatology: nonzero tendency
    st = st._replace(tracers={oc.CARIOLLE_TRACER: st.T.replace(
        data=o3, name=oc.CARIOLLE_TRACER, units="kg/kg")})
    return mesh, coord, st, coef


def test_mpas_module_is_the_kernel_on_the_state(mpas):
    mesh, coord, st, coef = mpas
    fn = oc.make_cariolle_physics(oc.OzoneCariolleConfig(enabled=True), 600.0)
    got = fn(st, mesh, coord, forcing={"cariolle_coef": coef})
    do3 = got.tracer_tendencies[oc.CARIOLLE_TRACER].data
    p_s = st.p_s.data
    want = oc.cariolle_tendency(          # the module computes in the state dtype
        st.tracers[oc.CARIOLLE_TRACER].data, st.T.data, coord.pressure_at_full(p_s),
        coord.pressure_at_half(p_s), jnp.asarray(coef, st.T.data.dtype), 600.0,
        oc.OzoneCariolleConfig())
    np.testing.assert_array_equal(np.asarray(do3), np.asarray(want))
    assert float(jnp.abs(do3).max()) > 0.0
    assert float(jnp.abs(got.dT_dt.data).max()) == 0.0         # tracer only
    with pytest.raises(ValueError, match="cariolle_coef"):
        fn(st, mesh, coord, forcing={})


def test_explicit_column_source_matches_analytic(mpas):
    """O4a: with the implicit terms off (k3 = k7 = 0) and the floor and
    temperature clamp inactive, the column tendency is the analytic source
    sum((k2 + k5 (T - k4)) dp) / g."""
    mesh, coord, st, coef = mpas
    c = np.array(coef)
    c[..., 2] = 0.0
    c[..., 6] = 0.0
    T = np.asarray(c[..., 3]) + 3.0                 # inside the 15 K clamp
    st2 = st._replace(T=st.T.replace(data=jnp.asarray(T)))
    fn = oc.make_cariolle_physics(oc.OzoneCariolleConfig(enabled=True), 600.0)
    do3 = np.asarray(fn(st2, mesh, coord, forcing={"cariolle_coef": c})
                     .tracer_tendencies[oc.CARIOLLE_TRACER].data)
    dp = np.asarray(coord.layer_thickness_dp(st.p_s.data))
    col = (do3 * dp).sum(1) / constants.g
    want = ((c[..., 1] + c[..., 4] * (T - c[..., 3])) * dp).sum(1) / constants.g
    np.testing.assert_allclose(col, want, rtol=1e-6, atol=1e-30)


def test_make_physics_adds_ozone_only_when_enabled(mpas):
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    mesh, coord, st, coef = mpas
    forcing = {"T_sfc": jnp.full((st.T.data.shape[0],), 290.0), "cariolle_coef": coef}
    cfg = PhysicsConfig(radiation=RadiationConfig(scheme="none"),
                        cariolle=oc.OzoneCariolleConfig(enabled=True))
    out = make_physics(cfg, model_type="mpas", dt=600.0)(st, mesh, coord, forcing=forcing)
    comb = out[0] if isinstance(out, tuple) else out
    want = oc.make_cariolle_physics(cfg.cariolle, 600.0)(st, mesh, coord, forcing=forcing)
    np.testing.assert_allclose(
        np.asarray(comb.tracer_tendencies[oc.CARIOLLE_TRACER].data),
        np.asarray(want.tracer_tendencies[oc.CARIOLLE_TRACER].data), rtol=1e-14)
    off = make_physics(cfg._replace(cariolle=oc.OzoneCariolleConfig()), model_type="mpas",
                       dt=600.0)(st, mesh, coord, forcing=forcing)
    off = off[0] if isinstance(off, tuple) else off
    assert oc.CARIOLLE_TRACER not in (off.tracer_tendencies or {})
    with pytest.raises(ValueError, match="MPAS column lane"):
        make_physics(cfg, model_type="hydrostatic", dt=600.0)


def test_validate_strict_refuses_off_the_mpas_dycore():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    with pytest.raises(ValueError, match="ifs_ozone_chem runs only"):
        ExperimentConfig(ifs_ozone_chem=True).validate_strict()
    duo = ExperimentConfig(ifs_ozone_chem=True, grid=GridConfig(grid_type="mpas"),
                           dycore=DycoreConfig(discretization="fv3_duo", column_lane=True))
    with pytest.raises(ValueError, match="ifs_ozone_chem runs only"):
        duo.validate_strict()


def test_monthly_writer_emits_ozone_tracer_and_column(tmp_path):
    """The CMIP writer silently skips names it cannot look up, so o3_ifs and
    toz_ifs must be registered (non-CMIP6) or they vanish from output."""
    netCDF4 = pytest.importorskip("netCDF4")
    from legoesm.io import cmor_table_loader as tables
    from legoesm.io.cmor_output import CFWriter
    lat = np.linspace(-87.5, 87.5, 8)
    lon = np.linspace(0.0, 337.5, 16)
    plev = np.array([float(p) for p in
                     tables.coordinate_axis("plev19")["requested"]])
    w = CFWriter(output_dir=str(tmp_path), experiment_id="amip",
                 model_id="legoESM-1-0", freq="mon", calendar="noleap",
                 ref_date="1979-01-01")
    p3 = w.write_field("o3_ifs", np.full((plev.size, 8, 16), 1e-6),
                       time=15.0, time_bounds=(0.0, 31.0), lat=lat, lon=lon,
                       plev=plev, table="Amon")
    p2 = w.write_field("toz_ifs", np.full((8, 16), 300.0), time=15.0,
                       time_bounds=(0.0, 31.0), lat=lat, lon=lon, table="Amon")
    w.close()
    for path, var, units, val in ((p3, "o3_ifs", "kg kg-1", 1e-6),
                                  (p2, "toz_ifs", "DU", 300.0)):
        with netCDF4.Dataset(path) as ds:
            assert ds[var].units == units
            np.testing.assert_allclose(np.asarray(ds[var][:]), val, rtol=1e-6)


def test_transport_undershoot_keeps_global_ozone_mass():
    """The post-transport positivity stage borrows for o3_ifs instead of
    clamping, so an undershoot creates no ozone (codex review 5, vote V11a)."""
    from legoesm.core.conservation import apply_water_positivity
    o3 = jnp.asarray([[2e-6, -5e-7, 1e-6], [3e-6, 1e-6, -2e-7]])
    dp = jnp.asarray([[1e3, 2e3, 3e3], [1e3, 2e3, 3e3]])
    area = jnp.asarray([1.0, 2.0])
    out, _ = apply_water_positivity({"o3_ifs": o3}, None, dp, conservative=True,
                                    energy_consistent=False, area=area)
    mass = lambda q: float(((q * dp).sum(-1) * area).sum())  # noqa: E731
    assert float(out["o3_ifs"].min()) >= 0.0
    assert mass(out["o3_ifs"]) == pytest.approx(mass(o3), rel=1e-12)


def test_packaged_top_seed_is_positive_everywhere():
    """Vote Y1a measured claim: on the shipped coefficients the model-top
    overhead column k6 - k1*(p_full - p_half)/g is positive in every month and
    latitude (4.0e-5 .. 1.2e-4 kg/m2)."""
    sig = make_cam6_l32_levels()
    ps = jnp.asarray([constants.p_atm_std], jnp.float64)
    pf = np.asarray(sig.pressure_at_full(ps)).reshape(-1)
    ph = np.asarray(sig.pressure_at_half(ps)).reshape(-1)
    with np.load(oc.CARIOLLE_FILE) as d:
        k = np.asarray(d["coef"], np.float64)              # (12, 7, nlev, nlat)
    seed = k[:, 5, 0] - k[:, 0, 0] * (pf[0] - ph[0]) / constants.g
    assert seed.min() > 0.0


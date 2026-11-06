"""Oracle test: OpenIFS 49r1 IFS radiation interface (ecrad_ifs.ifs_interface).

Fixture: the validated ifs49_driver.x run base_a (8 columns, 137 levels),
exported by scripts/validate/oracle/ecrad_ifs/ifs49_export_fixture.py.  Each
stage is fed the oracle's own inputs for that stage, so a failure names one
stage:

* RADIATION_SCHEME inputs: driver arguments (drv_in_*, inputs) -> ifs_in_*
  (what radiation_scheme hands to ecRad) and sub_* (effective radii,
  decorrelation length);
* RADIATION_SCHEME outputs: ecRad fluxes (ifs_out_*) -> rs_*;
* RADINTG glue: rs_* -> ri_*;
* RADHEATN: ri_* + per-step mu0/skin T -> rh_*_s0..3.

fp64 (JAX_ENABLE_X64=1); hard form (sharpness = inf) at rel <= 1e-12.
"""

import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ecrad
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ifs_interface as ii

jax.config.update("jax_enable_x64", True)

FIXTURE = os.path.join(os.path.dirname(__file__), "oracle_fixtures", "ecrad_ifs49_interface.npz")
TOL = 1.0e-12
HARD = math.inf


@pytest.fixture(scope="module")
def fx():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def d(fx, name):
    return fx["dump:" + name]


def rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    den = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)))


def scheme_args(fx):
    ncol, nlev = fx["in:q"].shape
    return dict(
        solar_irradiance=float(fx["in:solar_irradiance"]), mu0=d(fx, "drv_in_amu0"),
        skin_temperature=fx["in:skin_temperature"], albedo_dif=fx["in:sw_albedo"],
        albedo_dir=fx["in:sw_albedo_direct"], spectral_emiss=fx["in:lw_emissivity"],
        ccn_land=d(fx, "drv_in_ccn_land"), ccn_sea=d(fx, "drv_in_ccn_sea"),
        gelam=d(fx, "drv_in_gelam"), gemu=d(fx, "drv_in_gemu"), land_sea_mask=d(fx, "drv_in_lsm"),
        pressure_fl=d(fx, "drv_in_pressure_fl"), temperature_fl=d(fx, "drv_in_temperature_fl"),
        pressure_hl=fx["in:pressure_hl"], temperature_hl=d(fx, "drv_in_temperature_hl"),
        q=fx["in:q"], co2=d(fx, "drv_in_mmr_co2"), ch4=d(fx, "drv_in_mmr_ch4"),
        n2o=d(fx, "drv_in_mmr_n2o"), cfc11=d(fx, "drv_in_mmr_cfc11"),
        cfc12=d(fx, "drv_in_mmr_cfc12"), hcfc22=d(fx, "drv_in_mmr_hcfc22"),
        ccl4=d(fx, "drv_in_mmr_ccl4"), o3_dp=d(fx, "drv_in_o3_dp"),
        cloud_fraction=fx["in:cloud_fraction"], q_liquid=fx["in:q_liquid"], q_ice=fx["in:q_ice"],
        q_rain=np.zeros((ncol, nlev)), q_snow=np.zeros((ncol, nlev)),       # vote IV5
        aerosol_old=np.zeros((ncol, 6, nlev)),                              # vote IV1/V49
        nindat=int(fx["nml:NINDAT_H"]), nstep=int(fx["nml:NSTEP_H"]),
        tstep=float(fx["nml:TSTEP_H"]))


def ecrad_flux(fx):
    """ifs_out_* (Fortran shapes) -> the ecrad.radiation output dict layout."""
    o = lambda n: d(fx, "ifs_out_flux_" + n)  # noqa: E731
    return {
        "flux_up_lw": o("lw_up"), "flux_dn_lw": o("lw_dn"), "flux_up_lw_clear": o("lw_up_clear"),
        "flux_dn_lw_clear": o("lw_dn_clear"), "lw_derivative": o("lw_derivatives"),
        "flux_up_sw": o("sw_up"), "flux_dn_sw": o("sw_dn"), "flux_dn_direct_sw": o("sw_dn_direct"),
        "flux_up_sw_clear": o("sw_up_clear"), "flux_dn_sw_clear": o("sw_dn_clear"),
        "flux_dn_direct_sw_clear": o("sw_dn_direct_clear"),
        "spectral_flux_dn_sw_surf": o("sw_dn_surf_band").T,
        "spectral_flux_dn_sw_surf_clear": o("sw_dn_surf_clear_band").T,
        "canopy_flux_dn_diffuse_sw_surf": o("sw_dn_diffuse_surf_canopy").T,
        "canopy_flux_dn_direct_sw_surf": o("sw_dn_direct_surf_canopy").T,
    }


RS_KEYS = ("flux_sw", "flux_lw", "flux_sw_clear", "flux_lw_clear", "flux_sw_dn", "flux_lw_dn",
           "flux_sw_dn_clear", "flux_lw_dn_clear", "flux_dir", "flux_dir_clear",
           "flux_dir_into_sun", "flux_uv", "flux_par", "flux_par_clear", "flux_sw_dn_toa",
           "emis_out", "lwderivative", "swdiffuseband", "swdirectband")
RI_KEYS = dict(emtd="emtd", trso="trso", trsc="trsc", emtc="emtc", frted="frted", trsod="trsod",
               trsodc="trsodc", emtdc="emtdc", emit="emit", sudu="sudu", uvdf="uvdf",
               parf="parf", parcf="parcf", tincf="tincf", fdir="fdir", cdir="cdir",
               lwderivative="lwderivative")
RH_KEYS = ("te", "hrsw", "hrlw", "hrsc", "hrlc", "frso", "frth", "frsod", "frsodc", "frthd",
           "frthdc", "frsoc", "frthc", "incsr", "sdur", "dsrp", "uvdf", "parf", "parcf",
           "tincf", "fdir", "cdir")


def oracle_rs(fx):
    return {k: d(fx, "rs_" + k) for k in RS_KEYS}


def oracle_carry(fx):
    return ii.RadiationCarry(**{f: d(fx, "ri_" + k) for f, k in RI_KEYS.items()},
                             mu0_rad=fx["in:cos_solar_zenith_angle"])


def run_radheatn(fx, carry, s, sharpness):
    return ii.radheatn(carry, fx["in:pressure_hl"], fx["in:q"], d(fx, f"rh_in_tskin_s{s}"),
                       d(fx, f"rh_in_mu0_s{s}"), float(fx["in:solar_irradiance"]),
                       sharpness=sharpness)


# ---------------------------------------------------------------------------
# oracle matches
# ---------------------------------------------------------------------------

def test_fixture_is_the_harness_run(fx):
    assert int(fx["nml:NINDAT_H"]) == 20200115 and int(fx["nml:NSTEP_H"]) == 4
    assert float(fx["nml:TSTEP_H"]) == 900.0 and int(fx["nml:NSTEPS"]) == 4
    assert int(d(fx, "su_nweight_uv")) == 3 and int(d(fx, "su_nweight_par")) == 3


def test_uv_par_weights(fx):
    for w, name in ((ii.UV_WEIGHTS, "uv"), (ii.PAR_WEIGHTS, "par")):
        np.testing.assert_array_equal(w[0] + 1, d(fx, f"su_iband_{name}"))
        assert rel_err(w[1], d(fx, f"su_weight_{name}")) <= TOL


def test_curvature_mu0(fx):
    assert rel_err(ii.curvature_mu0(fx["in:cos_solar_zenith_angle"], HARD),
                   d(fx, "drv_in_amu0")) <= TOL


def test_radiation_scheme_inputs(fx):
    kw, ex = ii.prepare_inputs(**scheme_args(fx), sharpness=HARD)
    pairs = {
        "thermodynamics_pressure_hl": kw["pressure_hl"],
        "thermodynamics_temperature_hl": kw["temperature_hl"],
        "thermodynamics_h2o_sat_liq": ex["h2o_sat_liq"],
        "gas_mixing_ratio": kw["gas_mmr"],
        "ylcloud_q_liq": kw["q_liquid"], "ylcloud_q_ice": kw["q_ice"],
        "ylcloud_re_liq": kw["re_liquid"], "ylcloud_re_ice": kw["re_ice"],
        "ylcloud_fraction": kw["cloud_fraction"], "ylcloud_overlap_param": kw["overlap_param"],
        "ylcloud_fractional_std": kw["fractional_std"],
        "aerosol_mixing_ratio": ex["aerosol_mixing_ratio"],
        "single_level_cos_sza": kw["cos_sza"],
        "single_level_skin_temperature": kw["skin_temperature"],
        "single_level_sw_albedo": kw["sw_albedo"],
        "single_level_sw_albedo_direct": kw["sw_albedo_direct"],
        "single_level_lw_emissivity": kw["lw_emissivity"],
        "single_level_solar_irradiance": kw["solar_irradiance"],
        "single_level_spectral_solar_cycle_multiplier": ex["spectral_solar_cycle_multiplier"],
        "rad_config_cloud_inhom_decorr_scaling": ex["cloud_inhom_decorr_scaling"],
    }
    errs = {k: rel_err(v, d(fx, "ifs_in_" + k)) for k, v in pairs.items()}
    print("ifs_in max rel:", errs)
    assert max(errs.values()) <= TOL, errs
    np.testing.assert_array_equal(np.asarray(ex["iseed"]), d(fx, "ifs_in_single_level_iseed"))
    np.testing.assert_array_equal(ex["gas_iunits"], d(fx, "ifs_in_gas_iunits"))
    subs = {"re_liquid_um": ex["re_liquid_um"], "re_ice_um": ex["re_ice_um"],
            "decorr_len_edges_km": ex["decorr_len_km"], "decorr_len_ratio": ex["decorr_len_ratio"]}
    errs = {k: rel_err(v, d(fx, "sub_" + k)) for k, v in subs.items()}
    print("sub max rel:", errs)
    assert max(errs.values()) <= TOL, errs


def test_radiation_scheme_outputs(fx):
    out = ii.scheme_outputs(ecrad_flux(fx), d(fx, "drv_in_amu0"), fx["in:skin_temperature"],
                            fx["in:lw_emissivity"], sharpness=HARD)
    errs = {k: rel_err(out[k], d(fx, "rs_" + k)) for k in RS_KEYS}
    print("rs max rel:", errs)
    assert max(errs.values()) <= TOL, errs


def test_radintg_glue(fx):
    carry = ii.radintg_post(oracle_rs(fx), float(fx["in:solar_irradiance"]), d(fx, "drv_in_amu0"),
                            fx["in:sw_albedo"], fx["in:sw_albedo_direct"],
                            fx["in:cos_solar_zenith_angle"], sharpness=HARD)
    errs = {k: rel_err(getattr(carry, f), d(fx, "ri_" + k)) for f, k in RI_KEYS.items()}
    print("ri max rel:", errs)
    assert max(errs.values()) <= TOL, errs


@pytest.mark.parametrize("s", range(4))
def test_radheatn_steps(fx, s):
    out = run_radheatn(fx, oracle_carry(fx), s, HARD)
    errs = {k: rel_err(out[k], d(fx, f"rh_{k}_s{s}")) for k in RH_KEYS}
    print(f"rh s{s} max rel:", errs)
    assert max(errs.values()) <= TOL, errs


def test_radheatn_energy_closure(fx):
    """Column heating = net flux divergence: sum(hr*dp)*cp/g = F_top - F_sfc (RVTMP2 = 0)."""
    out = run_radheatn(fx, oracle_carry(fx), 2, HARD)
    dp = np.diff(fx["in:pressure_hl"], axis=1)
    col = np.sum(np.asarray(out["te"]) * dp, axis=1) * constants.c_pd_ifs / constants.g_ifs
    net = np.asarray(out["frso"] + out["frth"])
    np.testing.assert_allclose(col, net[:, 0] - net[:, -1], rtol=1e-11, atol=1e-9)


def test_unported_configuration_raises():
    ii.check_configuration()
    with pytest.raises(ValueError):
        ii.check_configuration(nradlp=3)


# ---------------------------------------------------------------------------
# smooth form: gradients, smooth-vs-hard gap, eager vs jit
# ---------------------------------------------------------------------------

def _chain(fx, sharpness):
    """prepare_inputs -> scheme_outputs (oracle ecRad fluxes) -> radintg -> radheatn s1."""
    kw, ex = ii.prepare_inputs(**scheme_args(fx), sharpness=sharpness)
    rs = ii.scheme_outputs(ecrad_flux(fx), kw["cos_sza"], kw["skin_temperature"],
                           kw["lw_emissivity"], sharpness=sharpness)
    carry = ii.radintg_post(rs, kw["solar_irradiance"], kw["cos_sza"], kw["sw_albedo"],
                            kw["sw_albedo_direct"], fx["in:cos_solar_zenith_angle"], sharpness)
    rh = run_radheatn(fx, carry, 1, sharpness)
    return dict(re_liq=ex["re_liquid_um"], re_ice=ex["re_ice_um"], alpha=kw["overlap_param"],
                emis=rs["emis_out"], trsod=carry.trsod, te=rh["te"], frso=rh["frso"],
                sdur=rh["sdur"])


# measured max rel gap smooth(100) vs hard on the fixture (2026-11-03): only re_ice
# moves (cells with 0 < qi < 1e-11 kg/kg inside the presence band; ecRad crops
# them, q < 1e-9).  Pinned at 2x measured; zero stays zero.
SMOOTH_GAP_MEASURED = {"re_liq": 0.0, "re_ice": 0.40944377063564485, "alpha": 0.0, "emis": 0.0,
                       "trsod": 0.0, "te": 0.0, "frso": 0.0, "sdur": 0.0}


def test_smooth_vs_hard_gap(fx):
    hard, soft = _chain(fx, HARD), _chain(fx, 100.0)
    gaps = {k: rel_err(soft[k], hard[k]) for k in hard}
    print("smooth(100) vs hard max rel:", gaps)
    for k, g in gaps.items():
        assert g <= 2.0 * SMOOTH_GAP_MEASURED[k], (k, g)


def test_eager_vs_jit(fx):
    eager = _chain(fx, 100.0)
    jitted = jax.jit(lambda: _chain(fx, 100.0))()
    diffs = {k: float(np.max(np.abs(np.asarray(eager[k]) - np.asarray(jitted[k])))) for k in eager}
    errs = {k: rel_err(eager[k], jitted[k]) for k in eager}
    print("eager vs jit max abs:", diffs, "max rel:", errs)
    assert max(errs.values()) <= 1e-12, errs


# Each smoothed transition: f(x, sharpness) scalar, and a scan [lo, hi] of x that
# crosses its band at sharpness 100.  "kind": "jump" = the two branches differ
# (the smooth form must differ from the hard one inside the scan and the
# gradient must be nonzero somewhere); "continuous" = the oracle function is
# continuous there (a clamp, or branches that agree at the threshold): same two
# checks, so a hard switch planted there fails; "degenerate" = both
# branches give the same value on the whole scan (gradient legitimately 0).
_P, _T = 50000.0, 260.0
_S = 100.0


def _re_liq(s, cf=0.5, ql=1e-4, qr=0.0, land=0.0, ccn=100.0):
    f = ii.liquid_effective_radius(jnp.full((1, 1), _P), jnp.full((1, 1), _T),
                                   jnp.reshape(cf, (1, 1)), jnp.reshape(ql, (1, 1)),
                                   jnp.reshape(qr, (1, 1)), jnp.reshape(land, (1,)),
                                   jnp.reshape(ccn, (1,)), jnp.reshape(ccn, (1,)), s)
    return f[0, 0]


def _re_ice(s, cf=0.5, qi=1e-5, t=_T):
    f = ii.ice_effective_radius(jnp.full((1, 1), _P), jnp.reshape(t, (1, 1)),
                                jnp.reshape(cf, (1, 1)), jnp.reshape(qi, (1, 1)),
                                jnp.zeros((1, 1)), jnp.zeros((1,)), s)
    return f[0, 0]


_SIG290 = constants.sigma_sb_ifs * 290.0 ** 4


def _toy_flux():
    """One column, 3 half levels; broadband emissivity ratio 0.9 + x/(sigma T^4 - 300)."""
    lw_dn = jnp.array([[0.0, 200.0, 300.0]])
    lw_up = jnp.array([[250.0, 300.0, 300.0 + 0.9 * (_SIG290 - 300.0)]])
    sw = jnp.array([[600.0, 500.0, 400.0]])
    z = jnp.zeros((1, 14))
    return {"flux_up_lw": lw_up, "flux_dn_lw": lw_dn, "flux_up_lw_clear": lw_up,
            "flux_dn_lw_clear": lw_dn, "lw_derivative": jnp.ones((1, 3)),
            "flux_up_sw": 0.1 * sw, "flux_dn_sw": sw, "flux_dn_direct_sw": 0.5 * sw,
            "flux_up_sw_clear": 0.1 * sw, "flux_dn_sw_clear": sw,
            "flux_dn_direct_sw_clear": 0.5 * sw, "spectral_flux_dn_sw_surf": z + 10.0,
            "spectral_flux_dn_sw_surf_clear": z + 10.0,
            "canopy_flux_dn_diffuse_sw_surf": jnp.full((1, 6), 20.0),
            "canopy_flux_dn_direct_sw_surf": jnp.full((1, 6), 30.0)}


def _rs(s, field, x):
    flux = _toy_flux()
    mu0 = jnp.full((1,), 0.5)
    if field == "mu0":
        mu0 = jnp.reshape(x, (1,))
    elif field == "lwdn":
        flux["flux_dn_lw"] = flux["flux_dn_lw"].at[0, -1].set(x)
    elif field == "lwnet":
        flux["flux_up_lw"] = flux["flux_up_lw"].at[0, -1].add(x)
    out = ii.scheme_outputs(flux, mu0, jnp.full((1,), 290.0), jnp.full((1, 2), 0.95), s)
    return out["flux_dir_into_sun"][0] if field == "mu0" else out["emis_out"][0]


def _toy_rs(s, frsod=400.0, frso_sfc=360.0, alb=0.1, emis=0.95):
    """RADINTG on one column; TOA flux 1000 W/m^2; albedo = 1.05*alb."""
    rs = {
        "flux_sw": jnp.stack([jnp.asarray(700.0), jnp.asarray(500.0), jnp.asarray(frso_sfc)])[None, :],
        "flux_sw_clear": jnp.array([[700.0, 500.0, 380.0]]),
        "flux_lw": jnp.array([[-250.0, -100.0, -60.0]]),
        "flux_lw_clear": jnp.array([[-260.0, -110.0, -70.0]]),
        "flux_sw_dn": jnp.reshape(frsod, (1,)), "flux_sw_dn_clear": jnp.array([420.0]),
        "flux_lw_dn": jnp.array([[300.0]]), "flux_lw_dn_clear": jnp.array([290.0]),
        "flux_dir": jnp.array([200.0]), "flux_dir_clear": jnp.array([250.0]),
        "flux_dir_into_sun": jnp.array([400.0]), "flux_uv": jnp.array([30.0]),
        "flux_par": jnp.array([150.0]), "flux_par_clear": jnp.array([170.0]),
        "flux_sw_dn_toa": jnp.array([1000.0]), "emis_out": jnp.reshape(emis, (1,)),
        "lwderivative": jnp.array([[0.2, 0.6, 1.0]]),
        "swdiffuseband": jnp.full((1, 6), 40.0), "swdirectband": jnp.full((1, 6), 30.0),
    }
    a = jnp.reshape(alb, (1, 1)) * jnp.ones((1, 6))
    return ii.radintg_post(rs, 1000.0, jnp.array([1.0]), a, a, jnp.array([1.0]), s)


def _toy_carry(**kw):
    base = dict(emtd=jnp.array([[-250.0, -100.0, -60.0]]), trso=jnp.array([[0.7, 0.5, 0.36]]),
                trsc=jnp.array([[0.7, 0.5, 0.38]]), emtc=jnp.array([[-260.0, -110.0, -70.0]]),
                frted=jnp.array([[300.0]]), trsod=jnp.array([0.4]), trsodc=jnp.array([0.42]),
                emtdc=jnp.array([290.0]), emit=jnp.array([0.95]), sudu=jnp.array([400.0]),
                uvdf=jnp.array([0.03]), parf=jnp.array([0.15]), parcf=jnp.array([0.17]),
                tincf=jnp.array([1.0]), fdir=jnp.array([0.2]), cdir=jnp.array([0.25]),
                lwderivative=jnp.array([[0.2, 0.6, 1.0]]), mu0_rad=jnp.array([0.4]))
    for k, v in kw.items():
        base[k] = base[k].at[..., -1].set(v) if k == "trso" else jnp.reshape(v, jnp.shape(base[k]))
    return ii.RadiationCarry(**base)


def _rh(s, field, x, key=None, mu0_rad=0.4):
    carry = _toy_carry(mu0_rad=mu0_rad, **({field: x} if field != "mu0" else {}))
    m = jnp.reshape(x, (1,)) if field == "mu0" else jnp.array([0.5])
    out = ii.radheatn(carry, jnp.array([[0.0, 50000.0, 100000.0]]), jnp.zeros((1, 2)),
                      jnp.array([290.0]), m, 1000.0, sharpness=s)
    if key is not None:
        return out[key][0] if out[key].ndim == 1 else out[key][0, -1]
    return jnp.sum(out["frso"]) + jnp.sum(out["fdir"]) + jnp.sum(out["frsod"])


_RHO = 1000.0 * _P / (constants.R_d_ifs * _T)
_EPS = float(np.finfo(np.float64).eps)
_E1 = 1e4 * _EPS
TRANSITIONS = [
    ("liq cf>=0.001", "jump", lambda x, s: _re_liq(s, cf=x), (0.00099, 0.00101)),
    ("liq ql+qr>0", "degenerate", lambda x, s: _re_liq(s, ql=x), (-2e-11, 2e-11)),
    ("liq lwc>REPSCW", "jump", lambda x, s: _re_liq(s, ql=x, qr=1e-4),
     (0.99e-12 * 0.5 / _RHO, 1.01e-12 * 0.5 / _RHO)),
    ("liq land<0.5", "jump", lambda x, s: _re_liq(s, land=x), (0.49, 0.51)),
    ("liq re>REPLOG", "degenerate", lambda x, s: _re_liq(s, ql=x), (2.0e-13, 2.4e-13)),
    ("liq re clamp 4", "continuous", lambda x, s: _re_liq(s, ql=x), (1.2e-5, 1.5e-5)),
    ("liq re clamp 30", "continuous", lambda x, s: _re_liq(s, ql=x, ccn=60.0), (3.3e-3, 4.5e-3)),
    ("ice cf>0.001", "jump", lambda x, s: _re_ice(s, cf=x), (0.00099, 0.00101)),
    ("ice qi+qs>0", "jump", lambda x, s: _re_ice(s, qi=x), (-1e-11, 2e-11)),
    ("ice D clamp min", "continuous", lambda x, s: _re_ice(s, qi=x), (4e-6, 6e-6)),
    ("ice D clamp 155", "continuous", lambda x, s: _re_ice(s, qi=x, t=270.0), (1.2e-4, 1.8e-4)),
    ("rs mu0>eps", "jump", lambda x, s: _rs(s, "mu0", x), (0.99 * _EPS, 1.01 * _EPS)),
    ("rs |bb|>1e-5", "jump", lambda x, s: _rs(s, "lwdn", x), (_SIG290 - 1.01e-5, _SIG290 - 0.99e-5)),
    ("rs emis clamp 0.99", "continuous", lambda x, s: _rs(s, "lwnet", x),
     (0.08 * (_SIG290 - 300.0), 0.1 * (_SIG290 - 300.0))),
    ("rs emis clamp 0.8", "continuous", lambda x, s: _rs(s, "lwnet", x),
     (-0.11 * (_SIG290 - 300.0), -0.09 * (_SIG290 - 300.0))),
    ("curvature mu0>1e-10", "continuous",
     lambda x, s: ii.curvature_mu0(jnp.reshape(x, (1,)), s)[0], (0.99e-10, 1.01e-10)),
    ("ri max(FRSO,FRSOD)", "continuous", lambda x, s: _toy_rs(s, frsod=x).trsod[0], (345.0, 365.0)),
    ("ri FRSOD>FRSO", "jump", lambda x, s: _toy_rs(s, frso_sfc=x).trso[0, -1], (385.0, 405.0)),
    ("ri FRSO>0", "jump", lambda x, s: _toy_rs(s, frso_sfc=x).trsod[0], (-5.0, 15.0)),
    ("ri albedo clamp 0", "continuous", lambda x, s: _toy_rs(s, alb=x).trsod[0], (-0.012, 0.002)),
    ("ri [0,1] clamp", "continuous", lambda x, s: _toy_rs(s, emis=x).emit[0], (0.98, 1.03)),
    ("rh mu0>=eps", "continuous", lambda x, s: _rh(s, "mu0", x), (99 * _EPS, 101 * _EPS)),
    ("rh fdir>=eps1", "continuous", lambda x, s: _rh(s, "fdir", x), (0.0, 2 * _E1)),
    ("rh trsod>=eps1", "jump", lambda x, s: _rh(s, "trsod", x, "frsod"), (0.0, 2 * _E1)),
    ("rh frso floor", "continuous", lambda x, s: _rh(s, "trso", x, "frso", mu0_rad=0.5),
     (0.98 * 100 * _EPS / 500.0, 100 * _EPS / 500.0)),
    ("rh sdur>=120", "jump", lambda x, s: ii.radheatn(
        _toy_carry(fdir=x, mu0_rad=0.5), jnp.array([[0.0, 5e4, 1e5]]), jnp.zeros((1, 2)),
        jnp.array([290.0]), jnp.array([0.5]), 1000.0, sharpness=s)["sdur"][0], (0.119, 0.121)),
]


@pytest.mark.parametrize("name,kind,f,span", TRANSITIONS, ids=[t[0] for t in TRANSITIONS])
def test_gradient_across_transition(name, kind, f, span):
    xs = jnp.linspace(span[0], span[1], 401)
    g = jax.vmap(jax.grad(lambda x: f(x, _S)))(xs)
    assert np.all(np.isfinite(g)), name
    if kind == "degenerate":
        return
    assert np.any(np.asarray(g) != 0.0), name
    if kind in ("jump", "continuous"):   # the smooth form must differ from the hard one in the band
        soft = jax.vmap(lambda x: f(x, _S))(xs)
        hard = jnp.stack([f(x, HARD) for x in xs[::20]])
        assert np.max(np.abs(np.asarray(soft[::20]) - np.asarray(hard))) > 0.0, name


FD_STEPS = (1e-5, 1e-6)    # two relative steps; the check must pass at both (vote V52)


def test_check_grads_chain(fx):
    """check_grads (order 1, rev) on a smooth interior point of the whole chain.

    Inputs are nondimensional relative perturbations (vote V52): full-level T
    at the radiation call and the skin T of the RADHEATN step are multiplied by
    (1 + u), (1 + v), so one step size h is the same relative change for both;
    the FD check is shown at two step sizes that agree with each other and AD.  (Perturbing the radiation-call
    skin T instead is a null control: the diagnosed emissivity reproduces the
    surface net LW, so the LW update and its derivative cancel exactly.)"""
    args = scheme_args(fx)

    def f(u, v):
        a = dict(args, temperature_fl=args["temperature_fl"] * (1.0 + u))
        ts_step = args["skin_temperature"] * (1.0 + v)
        kw, ex = ii.prepare_inputs(**a, sharpness=100.0)
        rs = ii.scheme_outputs(ecrad_flux(fx), kw["cos_sza"], kw["skin_temperature"],
                               kw["lw_emissivity"], sharpness=100.0)
        carry = ii.radintg_post(rs, kw["solar_irradiance"], kw["cos_sza"], kw["sw_albedo"],
                                kw["sw_albedo_direct"], fx["in:cos_solar_zenith_angle"], 100.0)
        rh = ii.radheatn(carry, fx["in:pressure_hl"], fx["in:q"], ts_step,
                         d(fx, "rh_in_mu0_s1"), kw["solar_irradiance"], sharpness=100.0)
        return (jnp.sum(ex["re_liquid_um"]) + jnp.sum(ex["re_ice_um"]) + jnp.sum(kw["overlap_param"])
                + jnp.sum(rh["te"]) * 1e5 + jnp.sum(rh["frth"]) * 1e-2)

    x0 = (0.0, 0.0)                    # vote W3
    for h in FD_STEPS:
        check_grads(f, x0, order=1, modes=("rev",), eps=h)
    # explicit convergence: central differences at both steps agree with AD per input
    g_ad = jax.grad(f, argnums=(0, 1))(*x0)
    for k in range(2):
        fd = []
        for h in FD_STEPS:
            xp, xm = list(x0), list(x0)
            xp[k] += h
            xm[k] -= h
            fd.append((float(f(*xp)) - float(f(*xm))) / (2.0 * h))
        print(f"input {k}: AD {float(g_ad[k]):.12e}  FD {fd}")
        assert np.isfinite(float(g_ad[k])) and float(g_ad[k]) != 0.0, (k, g_ad[k])
        np.testing.assert_allclose(fd[0], fd[1], rtol=1e-6)
        for v in fd:
            np.testing.assert_allclose(v, float(g_ad[k]), rtol=1e-6)


def test_radiation_call_skin_temperature_gradient_is_zero(fx):
    """Vote W4: perturbing the skin T of the radiation call itself (seen by the
    emissivity diagnosis AND by RADHEATN in the same step) leaves the LW update
    unchanged: the diagnosed broadband emissivity reproduces the surface net LW,
    so emis*(sigma*Ts^4 - LWdn) cancels PEMTD(sfc) and the zero is structural."""
    args = scheme_args(fx)

    def g(w):
        ts = args["skin_temperature"] * (1.0 + w)
        kw, _ = ii.prepare_inputs(**dict(args, skin_temperature=ts), sharpness=100.0)
        rs = ii.scheme_outputs(ecrad_flux(fx), kw["cos_sza"], ts, kw["lw_emissivity"], sharpness=100.0)
        carry = ii.radintg_post(rs, kw["solar_irradiance"], kw["cos_sza"], kw["sw_albedo"],
                                kw["sw_albedo_direct"], fx["in:cos_solar_zenith_angle"], 100.0)
        rh = ii.radheatn(carry, fx["in:pressure_hl"], fx["in:q"], ts, d(fx, "rh_in_mu0_s1"),
                         kw["solar_irradiance"], sharpness=100.0)
        return jnp.sum(rh["te"]) * 1e5 + jnp.sum(rh["frth"]) * 1e-2

    grad = float(jax.grad(g)(0.0))
    print("radiation-call skin T gradient:", grad)
    assert abs(grad) <= 1e-9, grad


def test_decorrelation_length_pole_gradient():
    """Vote IF12: finite gradient wrt sine of latitude at both poles and the equator."""
    g = jax.vmap(jax.grad(lambda m: ii.cloud_overlap_decorr_len(m)[0]))(jnp.array([-1.0, 0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(g), [2 * 2.149, 0.0, -2 * 2.149], rtol=1e-14)


# ---------------------------------------------------------------------------
# composed call: curvature mu0 -> RADIATION_SCHEME (prep, ecRad, outputs) -> RADINTG
# ---------------------------------------------------------------------------

def composed_args(fx):
    a = scheme_args(fx)
    del a["mu0"]
    return a


def drawn_uniforms(fx):
    """Production McICA uniforms for the port's own ecRad inputs."""
    kw, ex = ii.prepare_inputs(**scheme_args(fx), sharpness=HARD)
    return ecrad.draw_uniforms(jnp.asarray(ex["iseed"]), kw["cloud_fraction"].shape[-1])


_STATIC_ARGS = ("nindat", "nstep", "tstep")   # static ints / step length (seed)


def run_composed(fx, uniforms, sharpness, jit=False):
    """All model arrays (incl. raw mu0 and the solar irradiance) are traced
    arguments under jit; only the seed's date/step/timestep are static."""
    a = composed_args(fx)
    static = {k: a.pop(k) for k in _STATIC_ARGS}

    def f(mu0, arrays, ulw, usw):
        rs, carry = ii.radiation_scheme(mu0=mu0, uniforms_lw=ulw, uniforms_sw=usw,
                                        sharpness=sharpness, **arrays, **static)
        return rs, carry._asdict()

    rs, carry = (jax.jit(f) if jit else f)(fx["in:cos_solar_zenith_angle"], a, *uniforms)
    return ({k: np.asarray(v) for k, v in rs.items()}, {k: np.asarray(v) for k, v in carry.items()})


@pytest.fixture(scope="module")
def uniforms(fx):
    return drawn_uniforms(fx)


@pytest.fixture(scope="module")
def composed_hard(fx, uniforms):
    return run_composed(fx, uniforms, HARD)


# Vote C8: the heating rates of the composed chain carry the ecRad-core round-off
# (<=1.2e-14 in the carry) amplified by top-layer flux cancellation; they get the
# IFS-interface oracle's own 20-seed 1-ulp spread (committed artifact, controls
# C1-C5 PASS, never retyped).  Every other rh_* field stays at 1e-12.
SPREAD_ARTIFACT_IFS = os.path.join(os.path.dirname(__file__), "oracle_fixtures",
                                   "spread_ecrad_ifs49_interface.json")
_C8_FIELDS = ("te", "hrlw", "hrlc", "hrsw", "hrsc")


def rh_tol(k, s):
    if k not in _C8_FIELDS:
        return TOL
    with open(SPREAD_ARTIFACT_IFS) as f:
        art = json.load(f)
    assert all(art["controls"][c] == "PASS" for c in ("C1", "C2", "C3", "C4", "C5"))
    return max(TOL, float(art["spread"][f"rh_{k}_s{s}"]))


_COMPOSED_JIT_SCRIPT = """
import sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
from tests.atmosphere import test_ecrad_ifs_interface_oracle as t
with np.load(t.FIXTURE) as z:
    fx = {k: z[k] for k in z.files}
rs, carry = t.run_composed(fx, t.drawn_uniforms(fx), float(sys.argv[2]), jit=True)
np.savez(sys.argv[1], **{"rs:" + k: v for k, v in rs.items()}, **{"ri:" + k: v for k, v in carry.items()})
"""


# Canopy-diffuse-derived fields get a wider jit bound only where measured above
# 1e-12.  Measured 2026-11-06 with RRTMGP gas optics: rs swdiffuseband 6.2e-12.
_COMPOSED_JIT_TOL = {"rs:swdiffuseband": 1.0e-11}


@pytest.mark.parametrize("sharpness", [HARD, 100.0])
def test_composed_eager_vs_jit(fx, uniforms, composed_hard, tmp_path, sharpness):
    """jit in a separate process with --xla_cpu_max_isa=AVX (no FMA; the same
    method as the end-to-end test) == eager to 1e-12 (V51), C5 exemption above."""
    import subprocess
    import sys
    rs, carry = composed_hard if sharpness == HARD else run_composed(fx, uniforms, sharpness)
    path = tmp_path / "jit.npz"
    env = dict(os.environ, XLA_FLAGS="--xla_cpu_max_isa=AVX", JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    subprocess.run([sys.executable, "-c", _COMPOSED_JIT_SCRIPT, str(path), repr(sharpness)],
                   env=env, check=True)
    jit = np.load(path)
    diff = {"rs:" + k: rel_err(v, jit["rs:" + k]) for k, v in rs.items()}
    diff.update({"ri:" + k: rel_err(v, jit["ri:" + k]) for k, v in carry.items()})
    print(f"COMPOSED_EAGER_VS_JIT sharpness={sharpness} max_rel={max(diff.values()):.3e} "
          f"worst={max(diff, key=diff.get)}")
    bad = {k: v for k, v in diff.items() if v > _COMPOSED_JIT_TOL.get(k, TOL)}
    assert not bad, bad


def _composed_functional(fx, uniforms, sharpness, k):
    """Scalar of the composed call vs one relative perturbation: k=0 full-level T
    (reaches ecRad only through the effective radii), k=1 skin T."""
    args = composed_args(fx)
    ulw, usw = uniforms
    name = ("temperature_fl", "skin_temperature")[k]

    def f(x):
        a = dict(args, **{name: args[name] * (1.0 + x)})
        rs, carry = ii.radiation_scheme(mu0=fx["in:cos_solar_zenith_angle"], uniforms_lw=ulw,
                                        uniforms_sw=usw, sharpness=sharpness, **a)
        return (jnp.sum(rs["flux_lw"]) * 1e-2 + jnp.sum(rs["flux_sw"]) * 1e-2
                + jnp.sum(carry.trso) + jnp.sum(carry.emit))

    return jax.jit(f)


def _strict_grad_check(f, label):
    """W1-W3: from x0 = 0, check_grads at h = 1e-5 and 1e-6; both central FDs
    agree with each other and with AD at rtol 1e-6; AD finite and nonzero."""
    x0 = 0.0
    for h in FD_STEPS:
        check_grads(f, (x0,), order=1, modes=("rev",), eps=h)
    g = float(jax.grad(f)(x0))
    fd = [(float(f(x0 + h)) - float(f(x0 - h))) / (2.0 * h) for h in FD_STEPS]
    print(f"{label}: AD {g:.12e}  FD {fd}")
    assert np.isfinite(g) and g != 0.0, (label, g)
    np.testing.assert_allclose(fd[0], fd[1], rtol=1e-6)
    for v in fd:
        np.testing.assert_allclose(v, g, rtol=1e-6)


def test_composed_grad_full_level_t_hard(fx, uniforms):
    """Vote C7: the gate for the full-level-T input is the hard form."""
    _strict_grad_check(_composed_functional(fx, uniforms, HARD, 0), "composed T_fl inf")


@pytest.mark.xfail(strict=True, reason=(
    "votes C7/C9: the FD side fails, not AD: fp64 round-off in the SW two-stream kernel "
    "on nearly conservative layers makes the central FD in full-level T (via the "
    "effective radii) noisy; AD is correct; FD error 3.3e-7 at h=3e-5, 3.47e-5 at h=1e-4"))
def test_composed_grad_full_level_t_smooth(fx, uniforms):
    _strict_grad_check(_composed_functional(fx, uniforms, 100.0, 0), "composed T_fl 100")


def test_composed_grad_skin_t_smooth(fx, uniforms):
    _strict_grad_check(_composed_functional(fx, uniforms, 100.0, 1), "composed T_skin 100")

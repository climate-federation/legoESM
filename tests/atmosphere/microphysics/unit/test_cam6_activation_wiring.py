"""CAM6 prescribed-aerosol droplet activation wired into the MPAS lane
(user decisions 2026-10-24): monthly records held two at a time and reloaded
when the bracket moves, linear time interpolation every step, activation
(microp_aero_run) immediately before the microphysics, CLUBB's w'^2 / Kh and
CAM's CLDO carried in the physics state.  x64.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import cam6_aerosol_climatology as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import cam6_dropmixnuc as D  # noqa: E402
from legoesm.atmosphere.physics.microphysics.cam6_modal_aerosol import MAM4_SPECIES  # noqa: E402


def test_trop_cloud_top_lev_is_press_lim_idx_top():
    pref = np.array([50.0, 100.0, 150.0, 300.0])
    assert C.trop_cloud_top_lev(pref, 100.0) == 2        # strict >, ref_pres.F90:155
    assert C.trop_cloud_top_lev(pref, 10.0) == 0
    assert C.trop_cloud_top_lev(pref, 1000.0) == 4       # none: pver (k_lim = pver+1, 1-based)


def _fake_records(nlevf, ncol, scale=1.0):
    fields = ("PS",) + C.PPE_SPECIES_FIELDS + C.PPE_NUMBER_FIELDS
    out = {}
    for i, v in enumerate(fields):
        if v == "PS":
            out[v] = np.full((2, ncol), 1.0e5)
        else:
            base = (i + 1) * scale * np.linspace(1.0, 2.0, nlevf)[None, :, None]
            out[v] = np.concatenate([base, 3.0 * base]) * np.ones((2, nlevf, ncol))
    return out


def test_bracket_reloads_only_when_the_months_change(monkeypatch):
    calls = []

    def load(data_dir, months, lon_w, lat_w, fields=None):
        calls.append(months)
        return _fake_records(4, 3)
    monkeypatch.setattr(C, "load_bracket_records", load)
    t0 = C.CLIM_RECORD_TIMES[0]                          # Jan 16 00:00
    held, f1 = C.cam6_aerosol_forcing("d", t0 - 2.0, None, None)
    held2, f2 = C.cam6_aerosol_forcing("d", t0 - 1.0, None, None, held=held)
    assert calls == [(11, 0)] and held2 is held          # Dec->Jan, no reload
    held3, f3 = C.cam6_aerosol_forcing("d", t0 + 1.0, None, None, held=held2)
    assert calls == [(11, 0), (0, 1)]
    for t, f, (n, np1) in ((t0 - 2.0, f1, (11, 0)), (t0 + 1.0, f3, (0, 1))):
        want = C.time_fact1(C.CLIM_RECORD_TIMES[n], C.CLIM_RECORD_TIMES[np1], t, C.NOLEAP_ONE_YR)
        assert float(f["fact1"]) == float(want)
    assert 0.0 < float(f2["fact1"]) < float(f1["fact1"]) < 1.0   # moves toward January


def test_aerosol_at_model_levels_is_interpolate_to_model_per_field():
    ncol, nlev, nlevf = 3, 5, 4
    recs = {k: jnp.asarray(v) for k, v in _fake_records(nlevf, ncol).items()}
    aer = {"rec": {k: v for k, v in recs.items() if k != "PS"}, "PS": recs["PS"],
           "fact1": jnp.asarray(0.25)}
    hyam = np.zeros(nlevf)
    hybm = np.linspace(0.1, 0.9, nlevf)
    setup = C.Cam6AerosolActivation(None, hyam, hybm, 1.0e5, 0, 1)
    pmid = jnp.linspace(5.0e3, 9.5e4, nlev)[None, :] * jnp.ones((ncol, 1))
    num, mass = C.aerosol_at_model_levels(aer, setup, pmid)
    assert num.shape == (4, ncol, nlev) and [m.shape[0] for m in mass] == [len(s) for s in MAM4_SPECIES]
    r = recs["dst_a3"]
    want = C.interpolate_to_model(r[0].T, r[1].T, recs["PS"][0], recs["PS"][1], 0.25,
                                  hyam, hybm, 1.0e5, pmid)
    np.testing.assert_array_equal(np.asarray(mass[2][0]), np.asarray(want))   # coarse: dst first
    r = recs["num_a2"]
    want = C.interpolate_to_model(r[0].T, r[1].T, recs["PS"][0], recs["PS"][1], 0.25,
                                  hyam, hybm, 1.0e5, pmid)
    np.testing.assert_array_equal(np.asarray(num[1]), np.asarray(want))


def test_turbulence_publishes_the_activation_inputs():
    from legoesm.atmosphere.physics.turbulence import integration as ti
    from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
    z = jnp.zeros((2, 3))
    out = TurbulenceOutput(*([z] * 4), Km=z, Kh=z, shflx=z[:, 0], lhflx=z[:, 0],
                           ustar=z[:, 0], h_pbl=z[:, 0], wp2_iface=jnp.ones((2, 4)),
                           Kh_iface=2.0 * jnp.ones((2, 4)))
    u = ti._carry_update_with_cloud_fraction("tke", None, out)
    assert float(u["clubb_wp2_iface"][0, 0]) == 1.0 and float(u["clubb_kh_iface"][0, 0]) == 2.0
    u = ti._carry_update_with_cloud_fraction("tke", None, out._replace(wp2_iface=None, Kh_iface=None))
    assert "clubb_wp2_iface" not in u and "clubb_kh_iface" not in u


def test_experiment_config_cli_and_threading():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    from scripts.run.run_amip import _postprocess_args, build_arg_parser, build_config_from_args
    d = ExperimentConfig._field_defaults
    assert d["morrison_prognostic_droplet_number"] is False and d["cam6_aerosol_dir"] is None
    assert MorrisonConfig().prognostic_droplet_number is False
    cfg = ExperimentConfig(microphysics="morrison", morrison_prognostic_droplet_number=True)
    assert thread_morrison_scalars(cfg, "morrison", MorrisonConfig()).prognostic_droplet_number
    with pytest.raises(ValueError, match="needs cam6_aerosol_dir"):
        cfg.validate_strict()
    parser = build_arg_parser()
    c = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--morrison-prognostic-droplet-number", "--cam6-aerosol-dir", "/x"]), parser))
    assert c.morrison_prognostic_droplet_number is True and c.cam6_aerosol_dir == "/x"
    c = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--dataset", "analytical"]), parser))
    assert c.morrison_prognostic_droplet_number is False and c.cam6_aerosol_dir is None


def test_production_deck_selects_droplet_numbers():
    from scripts.run.run_amip import build_arg_parser
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config("config/amip/amip_production.yaml", build_arg_parser())
    assert keys["morrison_prognostic_droplet_number"] is True
    assert keys["cam6_aerosol_dir"] == "/work/bb1596/b309178/diffESM/legoesm_ap/data/cam6_optics"
    assert keys["aerosol_ccn"] is False


def test_factory_refuses_without_the_activation_setup():
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    kw = dict(cloud_config=CloudConfig(scheme="cam6_clubb"), clubb_config=CLUBBConfig())
    on = MicrophysicsConfig(scheme="morrison", morrison=MorrisonConfig(
        warm_rain_incloud=True, prognostic_droplet_number=True))
    with pytest.raises(ValueError, match="cam6_activation"):
        make_microphysics_physics(on, "mpas", 60.0, **kw)
    no_incloud = MicrophysicsConfig(scheme="morrison", morrison=MorrisonConfig(
        prognostic_droplet_number=True))
    with pytest.raises(ValueError, match="needs warm_rain_incloud=True"):
        make_microphysics_physics(no_incloud, "mpas", 60.0, cam6_activation=object(), **kw)
    off = MicrophysicsConfig(scheme="morrison", morrison=MorrisonConfig(warm_rain_incloud=True))
    with pytest.raises(ValueError, match="read only under"):
        make_microphysics_physics(off, "mpas", 60.0, cam6_activation=object(), **kw)


def test_mg_preset_never_overwrites_an_explicit_ice_to_snow_choice():
    """User decision 2026-10-24, 1a."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import resolve_morrison_flavor
    assert MorrisonConfig().ice_to_snow_scheme is None
    r = lambda **k: resolve_morrison_flavor(MorrisonConfig(**k)).ice_to_snow_scheme
    assert r(morrison_flavor="mg") == "mg_ferrier"
    assert r(morrison_flavor="mg", ice_to_snow_scheme="mg_ferrier") == "mg_ferrier"
    assert r(morrison_flavor="sam") == "m2005_autoconv"
    assert r(morrison_flavor="sam", ice_to_snow_scheme="heuristic") == "heuristic"
    for bad in ("m2005_autoconv", "heuristic"):
        with pytest.raises(ValueError, match="morrison_flavor='mg'"):
            r(morrison_flavor="mg", ice_to_snow_scheme=bad)


def _lane(n=2):
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from tests.atmosphere.hydrostatic.unit.test_cam6_dropmixnuc_oracle import _parse
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none", cloud_config=CloudConfig(scheme="cam6_clubb")),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="clubb", liquid_partition=True,
                                    clubb=CLUBBConfig(prognostic=True)),
        microphysics=MicrophysicsConfig(scheme="morrison", morrison=MorrisonConfig(
            warm_rain_scheme="kk2000_cam6", warm_rain_incloud=True,
            liquid_from_closure=True, prognostic_droplet_number=True)))
    nlevf = 4
    setup = C.Cam6AerosolActivation(_parse()[2], np.zeros(nlevf), np.linspace(0.05, 1.0, nlevf),
                                    1.0e5, 0, 40)
    return make_physics(cfg, model_type="mpas", dt=600.0, cld_macmic_num_steps=n,
                        cam6_activation=setup), cfg, nlevf


def _moist_mg2_state():
    sys.path.insert(0, "tests/unit")
    from test_physics_macmic import _moist_setup
    mesh, sigma, state = _moist_setup()
    tr = dict(state.tracers)
    for k, v in (("N_s", 0.0), ("N_c", 5.0e7), ("N_i", 0.0), ("N_r", 0.0)):
        tr[k] = state.T.replace(data=jnp.full_like(state.T.data, v), name=k)
    # ice-supersaturated cloud ice on the top half: CAM6 aist > 0 there, so the
    # cloud (ast) and with it CLDO are non-zero
    ncol, nlev = state.T.data.shape
    top = (jnp.arange(nlev) < nlev // 2)[None, :] * jnp.ones((ncol, 1))
    tr["q_i"] = state.T.replace(data=1.0e-5 * top, name="q_i")
    tr["q_v"] = tr["q_v"].replace(data=jnp.maximum(tr["q_v"].data, 2.0e-3 * top))
    return mesh, sigma, state._replace(tracers=tr)


def _forcing(ncol, nlevf, scale=1.0):
    recs = {k: jnp.asarray(v) for k, v in _fake_records(nlevf, ncol).items()}
    rec = {k: 1.0e-10 * scale * v if not k.startswith("num") else 1.0e8 * scale * v
           for k, v in recs.items() if k != "PS"}
    return {"cam6_aerosol": {"rec": rec, "PS": recs["PS"], "fact1": jnp.asarray(0.5)}}


def test_activation_runs_before_microphysics_with_carried_inputs(monkeypatch):
    from legoesm.atmosphere.physics.microphysics import integration as mi
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import integration as ti
    acts, micro_kw, published = [], [], []
    real_act, real_micro = D.cam6_microp_aero_npccn, mi.morrison_microphysics
    real_upd = ti._carry_update_with_cloud_fraction

    def act(*a, **k):
        out = real_act(*a, **k)
        acts.append(dict(wp2=np.asarray(a[8]), kvh=np.asarray(a[9]), ast=np.asarray(a[10]),
                         cldo=np.asarray(a[11]), dt=a[12], npccn=np.asarray(out)))
        return out

    def micro(*a, **k):
        micro_kw.append({kk: np.asarray(v) for kk, v in k.items() if kk in ("npccn", "cloud_fraction")})
        return real_micro(*a, **k)

    def upd(f, v, out):
        published.append((np.asarray(out.wp2_iface), np.asarray(out.Kh_iface)))
        return real_upd(f, v, out)
    monkeypatch.setattr(D, "cam6_microp_aero_npccn", act)
    monkeypatch.setattr(mi, "morrison_microphysics", micro)
    monkeypatch.setattr(ti, "_carry_update_with_cloud_fraction", upd)
    fn, cfg, nlevf = _lane()
    mesh, sigma, state = _moist_mg2_state()
    ncol, nlev = state.T.data.shape
    ps = init_physics_state(ncol, nlev, cfg)
    _, ps_out = fn(state, mesh, sigma, phys_state=ps, forcing=_forcing(ncol, nlevf))
    assert len(acts) == len(micro_kw) == len(published) == 2
    assert np.all(acts[0]["cldo"] == 0.0)                              # CAM cold start
    assert np.any(acts[1]["cldo"] > 0.0)                               # a cloud to carry
    np.testing.assert_array_equal(acts[1]["cldo"], micro_kw[0]["cloud_fraction"])  # CLDO = ast
    np.testing.assert_array_equal(np.asarray(ps_out.cloud_fraction_old), micro_kw[1]["cloud_fraction"])
    for a, m, (wp2, kh) in zip(acts, micro_kw, published):
        np.testing.assert_array_equal(a["ast"], m["cloud_fraction"])
        np.testing.assert_array_equal(a["wp2"], wp2)                    # same macmic sub-step
        np.testing.assert_array_equal(a["kvh"], kh)
        assert a["dt"] == 300.0
        assert np.all(np.isfinite(a["npccn"])) and np.any(a["npccn"] != 0.0)
        # per kg -> per m^3 by a positive density: same zeros and signs
        np.testing.assert_array_equal(np.sign(m["npccn"]), np.sign(a["npccn"]))


def test_gradient_through_the_aerosol_is_finite_and_nonzero():
    """Growing cloud (cldn > cldo): activation, not decay, sets npccn, so the
    droplet source carries the aerosol's gradient."""
    from legoesm.atmosphere.physics.microphysics.integration import cam6_activation_npccn
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from tests.atmosphere.hydrostatic.unit.test_cam6_dropmixnuc_oracle import _parse
    fn, cfg, nlevf = _lane()
    _, sigma, state = _moist_mg2_state()
    ncol, nlev = state.T.data.shape
    setup = C.Cam6AerosolActivation(_parse()[2], np.zeros(nlevf), np.linspace(0.05, 1.0, nlevf),
                                    1.0e5, 0, 40)
    ps = init_physics_state(ncol, nlev, cfg)._replace(
        clubb_wp2_iface=jnp.full((ncol, nlev + 1), 0.5), clubb_kh_iface=jnp.full((ncol, nlev + 1), 10.0))
    p_s = state.p_s.data
    T = state.T.data
    q = {k: state.tracers[k].data for k in ("q_v", "q_c", "q_i", "N_c")}
    pf, ph = sigma.pressure_at_full(p_s), sigma.pressure_at_half(p_s)

    def f(s):
        n = cam6_activation_npccn(T, q["q_v"], pf, ph, q["N_c"], q["q_c"], q["q_i"],
                                  jnp.full_like(T, 0.5), ps, _forcing(ncol, nlevf, s), setup, 300.0)
        return jnp.sum(n)

    g = float(jax.grad(f)(1.0))
    assert np.isfinite(g) and g > 0.0          # more aerosol activates more droplets

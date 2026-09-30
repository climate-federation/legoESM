"""CAM6 MG2 in-cloud warm rain on the Morrison lane (``warm_rain_incloud``).

Oracle: a line-by-line numpy transcription of CAM6 MG2
(docs/references/cam6/src/physics/cam/micro_mg2_0.F90:878 lcldm,
:1224-1236 qcic/ncic, :1256-1266 precip_frac 'in_cloud', :1312-1322 qric,
micro_mg_utils.F90 kk2000_liq_autoconversion :689-736 and
accrete_cloud_water_rain :1292-1337, grid tendency = in-cloud rate x lcldm
at :1666/:1890) compared with the APPLIED autoconversion / accretion of
``morrison_microphysics``.  Every test that exercises cf<1 fails with the
in-cloud mapping reverted to the grid-mean state.
"""
from __future__ import annotations

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

MINCLD, QSMALL, ICSMALL = 1.0e-4, 1.0e-18, 1.0e-8
DROPLET_MASS_25UM = 4.0 / 3.0 * math.pi * 1000.0 * (25.0e-6) ** 3


def _var_coef(relvar, a):
    r = min(max(relvar, 0.001), 10.0)          # clubb_intr.F90:2425 clip
    return math.gamma(r + a) / math.gamma(r) / r ** a


def mg2_oracle(qc, qr, qi, nc_vol, cf, rho, *, law, relvar=10.0,
               accre_enhan=1.0, predict_nc=False):
    """Grid-mean prc, pra [kg/kg/s] and nprc1 [#/m3/s], MG2 line by line.

    ``law="mg2"`` is kk2000_liq_autoconversion; ``law="sam"`` swaps in the
    SAM M2005 KK2000 mass law (1350 qc^2.47 Nc^-1.79, no icsmall gate) inside
    the SAME MG2 in-cloud wrapper.  ``pra`` omits MG2's var_coef(relvar,1.15)
    (dropped by our accretion; applied by the caller when comparing)."""
    ncol, nlev = qc.shape
    prc = np.zeros_like(qc)
    pra = np.zeros_like(qc)
    nprc1 = np.zeros_like(qc)
    for i in range(ncol):
        lcldm = [max(cf[i, k], MINCLD) for k in range(nlev)]
        precip_frac = list(lcldm)                 # precip_frac = cldm
        for k in range(nlev):
            if k != 0 and qc[i, k] < QSMALL and qi[i, k] < QSMALL:
                precip_frac[k] = precip_frac[k - 1]
            if qc[i, k] >= QSMALL:
                qcic = min(qc[i, k] / lcldm[k], 5.0e-3)
                nc_kg = nc_vol[i, k] / rho[i, k]
                ncic = max(nc_kg / lcldm[k], 0.0) if predict_nc else nc_kg
            else:
                qcic, ncic = 0.0, 0.0
            qric = min(qr[i, k] / precip_frac[k], 0.01)
            if qric < QSMALL:
                qric = 0.0
            if law == "mg2":
                p = (_var_coef(relvar, 2.47) * 0.01 * 1350.0 * qcic ** 2.47
                     * (ncic * 1.0e-6 * rho[i, k]) ** (-1.1)
                     if qcic >= ICSMALL else 0.0)
            else:
                p = (1350.0 * qcic ** 2.47 * (ncic * 1.0e-6 * rho[i, k]) ** (-1.79)
                     if qcic > 0.0 else 0.0)
            a = (accre_enhan * 67.0 * (qcic * qric) ** 1.15
                 if (qric >= QSMALL and qcic >= QSMALL) else 0.0)
            prc[i, k] = p * lcldm[k]
            pra[i, k] = a * lcldm[k]
            nprc1[i, k] = (p * ncic / qcic * lcldm[k] * rho[i, k]
                           if qcic > 0.0 else 0.0)
    return prc, pra, nprc1


def _columns():
    """Two columns, level 0 = top.  Covers: lcldm floor (cf 1e-6), in-cloud
    cap (qc/cf > 5e-3), qric cap, sub-qsmall and sub-icsmall water, cloud-free
    gaps that inherit the precip fraction from above, and a gap with ice that
    breaks the inheritance."""
    cf = np.array([
        [0.3, 1e-6, 0.05, 0.6, 0.0, 0.0, 0.25, 0.0, 1.0, 0.0],
        [0.8, 0.8, 0.02, 0.0, 0.0, 0.5, 0.9, 0.0, 0.0, 0.4]])
    qc = np.array([
        [2e-4, 3e-7, 1e-3, 4e-4, 0.0, 1e-19, 3e-4, 0.0, 6e-4, 5e-9],
        [6e-4, 3e-4, 2e-5, 0.0, 0.0, 2e-4, 1e-3, 0.0, 0.0, 1e-4]])
    qi = np.zeros_like(qc)
    qi[0, 7] = 1e-6                                   # breaks inheritance
    qr = np.array([
        [1e-6, 2e-6, 4e-5, 1e-4, 2e-4, 1e-4, 3e-4, 2e-4, 5e-4, 4e-4],
        [0.0, 1e-5, 3e-4, 2e-4, 1e-4, 2e-4, 3e-4, 4e-4, 1e-20, 3e-4]])
    rho = np.linspace(0.5, 1.2, 10)[None, :].repeat(2, 0)
    return cf, qc, qr, qi, rho


def _run(mcfg, cf, qc, qr, qi, rho, dt=1.0e-3, N_c=None, jit=False):
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    shape = qc.shape
    z = jnp.zeros(shape)
    f = dict(q_c=jnp.asarray(qc), q_r=jnp.asarray(qr), q_i=jnp.asarray(qi),
             N_r=jnp.full(shape, 1e5), N_i=jnp.full(shape, 1e4))
    if N_c is not None:
        f["N_c"] = jnp.asarray(N_c)
    hyd = HydrometeorState(**{k: f.get(k, z) for k in HydrometeorState._fields})
    # q_v supersaturated (q_sat ~ 1.1e-2 at 285 K, 800 hPa): no cloud
    # evaporation sink, so at dt = 1e-3 s the q_c donor clamp cannot bind and
    # the published terms ARE the kernel rates.
    args = (jnp.full(shape, 285.0), jnp.full(shape, 1.5e-2), hyd,
            jnp.full(shape, 8.0e4), jnp.full((shape[0], shape[1] + 1), 8.0e4),
            jnp.asarray(rho), jnp.full(shape, 200.0), dt)
    kw = {} if cf is None else {"cloud_fraction": jnp.asarray(cf)}
    fn = morrison_microphysics
    if jit:
        fn = jax.jit(morrison_microphysics, static_argnums=(7, 8))
    return fn(*args, mcfg._replace(publish_qc_budget=True), **kw)


def _cfg(**kw):
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    return MorrisonConfig(**kw)


@pytest.mark.parametrize("scheme,law", [("kk2000_cam6", "mg2"), ("kk2000", "sam")])
@pytest.mark.parametrize("relvar", [10.0, 1.0])
def test_rates_match_the_mg2_transcription(scheme, law, relvar):
    cf, qc, qr, qi, rho = _columns()
    nc = 7.0e7
    cfg = _cfg(warm_rain_scheme=scheme, warm_rain_incloud=True, Nc_0=nc,
               kk2000_cam6_relvar=relvar, accre_enhan_fact=1.3)
    b = _run(cfg, cf, qc, qr, qi, rho).qc_budget
    prc, pra, _ = mg2_oracle(qc, qr, qi, np.full(qc.shape, nc), cf, rho,
                             law=law, relvar=relvar, accre_enhan=1.3)
    assert prc.max() > 0 and pra.max() > 0
    np.testing.assert_allclose(-np.asarray(b["autoconversion"]), prc,
                               rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(-np.asarray(b["accretion"]), pra,
                               rtol=1e-12, atol=1e-300)
    # the oracle is not the grid-mean law in disguise: at cf<1 they differ
    g_prc, g_pra, _ = mg2_oracle(qc, qr, qi, np.full(qc.shape, nc),
                                 np.ones_like(cf), rho, law=law, relvar=relvar,
                                 accre_enhan=1.3)
    assert np.abs(g_prc - prc).max() > 1e-3 * prc.max()
    assert np.abs(g_pra - pra).max() > 1e-3 * pra.max()


def test_mg2_accretion_varcoef_is_the_one_dropped_factor():
    """Documented departure: MG2 multiplies pra by var_coef(relvar, 1.15)
    (~1.007 at relvar 10); ours does not.  Pin its size so it is visible."""
    assert abs(_var_coef(10.0, 1.15) - 1.0) < 0.01
    assert _var_coef(1.0, 1.15) > 1.05          # a real bias at small relvar


def test_prognostic_droplet_number_is_rescaled_in_cloud():
    """predict_Nc: ncic = nc/lcldm and the cloud-number sink is MG2
    nprc1 x lcldm (warm column, no ice, so riming adds nothing)."""
    cf, qc, qr, _, rho = _columns()
    qi = np.zeros_like(qc)
    nc = np.full(qc.shape, 9.0e7)
    cfg = _cfg(warm_rain_scheme="kk2000", warm_rain_incloud=True,
               predict_Nc=True)
    out = _run(cfg, cf, qc, qr, qi, rho, N_c=nc)
    prc, _, nprc1 = mg2_oracle(qc, qr, qi, nc, cf, rho, law="sam",
                               predict_nc=True)
    np.testing.assert_allclose(-np.asarray(out.qc_budget["autoconversion"]),
                               prc, rtol=1e-12, atol=1e-300)
    # Pre-existing floor: the sink is safe_divide(., x_c, eps=1e-15), zero
    # below a 1e-15 kg mean droplet (~0.6 um); at the lcldm floor the
    # in-cloud number is 1e4x the grid mean and hits it.  MG2 has no floor.
    x_c = np.where(qc >= QSMALL, np.minimum(qc / np.maximum(cf, MINCLD), 5e-3)
                   * rho / (nc / np.maximum(cf, MINCLD)), 0.0)
    live = x_c > 1e-15
    assert live.sum() >= 10 and (~live & (nprc1 > 0)).sum() >= 1
    np.testing.assert_allclose(-np.asarray(out.dN_c_dt)[live], nprc1[live],
                               rtol=1e-10, atol=1e-300)
    assert (np.asarray(out.dN_c_dt)[~live] == 0).all()


@pytest.mark.parametrize("scheme", ["kk2000", "kk2000_cam6", "seifert_beheng",
                                    "seifert_beheng_sb2001"])
@pytest.mark.parametrize("predict_nc", [False, True])
def test_cf_one_is_bitwise_the_grid_mean_path(scheme, predict_nc):
    """cf = 1 everywhere reproduces the switch-off result bit for bit, on the
    domain q_c in {0} U [1e-18, 5e-3], q_r in {0} U [1e-18, 0.01)."""
    _, qc, qr, qi, rho = _columns()
    qr = np.where(qr < QSMALL, 0.0, qr)
    qc = np.where(qc < QSMALL, 0.0, np.minimum(qc, 5e-3))
    nc = np.full(qc.shape, 8.0e7) if predict_nc else None
    kw = dict(warm_rain_scheme=scheme, predict_Nc=predict_nc)
    off = _run(_cfg(**kw), None, qc, qr, qi, rho, dt=60.0, N_c=nc)
    on = _run(_cfg(warm_rain_incloud=True, **kw), np.ones_like(qc), qc, qr,
              qi, rho, dt=60.0, N_c=nc)
    la, lb = jax.tree_util.tree_leaves(off), jax.tree_util.tree_leaves(on)
    assert len(la) == len(lb)
    for a, b in zip(la, lb):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)


def test_cf_to_zero_limit():
    """cf -> 0 hits the lcldm floor: rates are those at cf = 1e-4, finite,
    and the applied water is bounded by the donor clamp."""
    _, qc, qr, qi, rho = _columns()
    cfg = _cfg(warm_rain_scheme="kk2000_cam6", warm_rain_incloud=True)
    a = _run(cfg, np.full(qc.shape, 1e-9), qc, qr, qi, rho).qc_budget
    b = _run(cfg, np.full(qc.shape, 1e-4), qc, qr, qi, rho).qc_budget
    for k in ("autoconversion", "accretion"):
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))
        assert np.isfinite(np.asarray(a[k])).all()
    prc, _, _ = mg2_oracle(qc, qr, qi, np.full(qc.shape, 1e8),
                           np.full(qc.shape, 1e-9), rho, law="mg2")
    np.testing.assert_allclose(-np.asarray(a["autoconversion"]), prc,
                               rtol=1e-12, atol=1e-300)
    out = _run(cfg, np.full(qc.shape, 1e-9), qc, qr, qi, rho, dt=1800.0)
    removed = -(out.qc_budget["autoconversion"] + out.qc_budget["accretion"]) * 1800.0
    assert bool(jnp.all(removed <= jnp.asarray(qc) * (1 + 1e-9)))


def _column_water_residual(out, rho, dz=200.0):
    tend = (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt + out.dq_i_dt
            + out.dq_s_dt + out.dq_g_dt)
    return np.asarray(jnp.sum(tend * rho * dz, axis=1) + out.precipitation)


def test_column_water_is_conserved_like_the_baseline():
    cf, qc, qr, qi, rho = _columns()
    kw = dict(warm_rain_scheme="kk2000_cam6")
    base = _run(_cfg(**kw), None, qc, qr, qi, rho, dt=300.0)
    inc = _run(_cfg(warm_rain_incloud=True, **kw), cf, qc, qr, qi, rho, dt=300.0)
    scale = float(jnp.sum(jnp.abs(inc.dq_c_dt) * rho * 200.0))
    assert scale > 0
    r_base = np.abs(_column_water_residual(base, jnp.asarray(rho))).max()
    r_inc = np.abs(_column_water_residual(inc, jnp.asarray(rho))).max()
    assert r_inc <= max(10.0 * r_base, 1e-13 * scale), (r_inc, r_base, scale)
    # the switch moved the warm-rain transfer (so this is not a no-op check)
    assert not np.allclose(np.asarray(inc.dq_r_dt), np.asarray(base.dq_r_dt))


def test_jit_parity_and_finite_gradients():
    cf, qc, qr, qi, rho = _columns()
    cfg = _cfg(warm_rain_scheme="kk2000_cam6", warm_rain_incloud=True)
    e = _run(cfg, cf, qc, qr, qi, rho, dt=60.0)
    j = _run(cfg, cf, qc, qr, qi, rho, dt=60.0, jit=True)
    np.testing.assert_allclose(np.asarray(j.dq_r_dt), np.asarray(e.dq_r_dt),
                               rtol=1e-12, atol=1e-300)

    def loss(cf_, qc_):
        return jnp.sum(_run(cfg, cf_, qc_, qr, qi, rho, dt=60.0).dq_r_dt)
    g_cf, g_qc = jax.grad(loss, argnums=(0, 1))(jnp.asarray(cf), jnp.asarray(qc))
    assert np.isfinite(np.asarray(g_cf)).all() and np.isfinite(np.asarray(g_qc)).all()
    assert float(jnp.max(jnp.abs(g_cf))) > 0.0      # cf carries a gradient


def test_raises_without_cloud_fraction_or_with_two_sources():
    _, qc, qr, qi, rho = _columns()
    with pytest.raises(ValueError, match="cloud_fraction"):
        _run(_cfg(warm_rain_incloud=True), None, qc, qr, qi, rho)
    with pytest.raises(ValueError, match="subgrid_autoconversion"):
        _run(_cfg(warm_rain_incloud=True, subgrid_autoconversion=True),
             np.ones_like(qc), qc, qr, qi, rho)


# --- the MPAS lane: which cloud fraction reaches the call --------------------

def _mpas_physics(incloud, turb="clubb", n=2):
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics import MicrophysicsConfig
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.radiation import RadiationConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme=turb),
        microphysics=MicrophysicsConfig(
            scheme="morrison",
            morrison=MorrisonConfig(warm_rain_scheme="kk2000_cam6",
                                    warm_rain_incloud=incloud)))
    return make_physics(cfg, model_type="mpas", dt=600.0,
                        cld_macmic_num_steps=n), cfg


def test_lane_guard_needs_clubb_and_the_subcycle():
    with pytest.raises(ValueError, match="cld_macmic_num_steps>=2"):
        _mpas_physics(True, turb="tke", n=2)
    with pytest.raises(ValueError, match="cld_macmic_num_steps>=2"):
        _mpas_physics(True, turb="clubb", n=1)
    assert callable(_mpas_physics(False, turb="tke", n=1)[0])


def test_microphysics_reads_clubbs_cloud_fraction_from_the_same_substep(monkeypatch):
    """Spy on both ends: the cloud fraction handed to Morrison in sub-step i
    is the array CLUBB returned in sub-step i (CAM: clubb_tend_cam, then MG2
    on ast), not the previous step's carry."""
    sys.path.insert(0, "tests/unit")
    from legoesm.atmosphere.physics.microphysics import integration as mi
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence import integration as ti
    from test_physics_macmic import _moist_setup
    written, read = [], []
    real_upd = ti._carry_update_with_cloud_fraction
    real_micro = mi.morrison_microphysics

    def upd(carry_field, carry_val, turb_out):
        written.append(np.asarray(turb_out.cloud_fraction))
        return real_upd(carry_field, carry_val, turb_out)

    def micro(*a, cloud_fraction=None, **k):
        read.append(None if cloud_fraction is None else np.asarray(cloud_fraction))
        return real_micro(*a, cloud_fraction=cloud_fraction, **k)
    monkeypatch.setattr(ti, "_carry_update_with_cloud_fraction", upd)
    monkeypatch.setattr(mi, "morrison_microphysics", micro)
    mesh, sigma, state = _moist_setup()
    fn, cfg = _mpas_physics(True, n=2)
    ps = init_physics_state(*state.T.data.shape, cfg)
    ps = ps._replace(cloud_fraction=jnp.full(state.T.data.shape, 0.123))
    fn(state, mesh, sigma, phys_state=ps)
    assert len(written) == len(read) == 2
    for w, r in zip(written, read):
        np.testing.assert_array_equal(w.reshape(r.shape), r)
    assert not np.all(read[0] == 0.123)


def test_experiment_config_cli_and_threading():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig, GridConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars

    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    assert ExperimentConfig._field_defaults["morrison_warm_rain_incloud"] is False
    assert MorrisonConfig().warm_rain_incloud is False
    good = dict(grid=GridConfig(grid_type="mpas", resolution=2, nlev=8),
                microphysics="morrison", turbulence="clubb",
                cld_macmic_num_steps=3, morrison_warm_rain_incloud=True)
    cfg = ExperimentConfig(**good)
    cfg.validate_strict()
    base = MorrisonConfig()
    assert thread_morrison_scalars(cfg, "morrison", base).warm_rain_incloud is True
    assert thread_morrison_scalars(
        cfg._replace(morrison_warm_rain_incloud=False), "morrison", base) is base
    for bad in (dict(turbulence="tke"), dict(cld_macmic_num_steps=1),
                dict(subgrid_autoconversion=True)):
        with pytest.raises(ValueError, match="morrison_warm_rain_incloud"):
            ExperimentConfig(**{**good, **bad}).validate_strict()
    with pytest.raises((ValueError, TypeError)):
        ExperimentConfig(**{**good, "morrison_warm_rain_incloud": 1}).validate_strict()
    parser = build_arg_parser()
    for flag, want in (("--morrison-warm-rain-incloud", True),
                       ("--no-morrison-warm-rain-incloud", False)):
        c = build_config_from_args(_postprocess_args(parser.parse_args([
            "--dataset", "analytical", "--microphysics", "morrison", flag]),
            parser))
        assert c.morrison_warm_rain_incloud is want
    c = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical"]), parser))
    assert c.morrison_warm_rain_incloud is False


def test_real_state_probe_runs_and_passes_on_a_synthetic_checkpoint(tmp_path, capsys):
    import importlib.util
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate"
           / "amip_bias" / "mg2_incloud_oracle.py")
    spec = importlib.util.spec_from_file_location("mg2_incloud_oracle", src)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    cf, qc, qr, qi, _ = _columns()
    # raw columns (hold sub-qsmall water, outside the bitwise domain) plus the
    # same columns cleaned into it, so both probe branches run
    qcc = np.where(qc < QSMALL, 0.0, np.minimum(qc, 5e-3))
    qrc = np.where(qr < QSMALL, 0.0, qr)
    cf, qi = np.concatenate([cf, cf, cf[::-1]]), np.concatenate([qi, qi, qi[::-1]])
    qc = np.concatenate([qc, qcc, qcc[::-1]])
    qr = np.concatenate([qr, qrc, qrc[::-1]])
    nlev = qc.shape[1]
    sig = np.linspace(0.0, 1.0, nlev + 1)
    path = tmp_path / "ckpt.npz"
    np.savez(path, T=np.full(qc.shape, 280.0), p_s=np.full(qc.shape[0], 1.0e5),
             meta_vgrid=np.stack([0.02 * (1 - sig), sig]),
             trc_q_v=np.full(qc.shape, 5e-3), trc_q_c=qc, trc_q_r=qr,
             trc_q_i=qi, physstate_cloud_fraction=cf)
    probe.main([str(path), "--ncol", "6"])
    out = capsys.readouterr().out
    import re
    assert "ORACLE PASS" in out and re.search(r"BITWISE on [1-9]\d* of", out)

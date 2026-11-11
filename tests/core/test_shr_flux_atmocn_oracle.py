"""Oracle fixture: ``compute_sam_oceflx_fluxes(variant="cesm")`` reproduces the
COMPILED CESM2.1 ocean surface-flux law.

Oracle: CIME tag cime5.6.47 (SHA d8e9b894d7eb96fe96f8f8a6943b421147f94606),
``src/share/util/shr_flux_mod.F90`` (sha1 40295186e9fcd7d5b099c9451534ad42dfcc8a58)
subroutine ``shr_flux_atmOcn``, compiled unmodified and run with legoESM's
constants and the CESM2.1 f09 driver settings (flux_max_iteration = 5,
flux_convergence = 0, coldair_outbreak_mod = .true., minwind 0.5, gust_fac 0).
The fixture ``oracle_fixtures/shr_flux_atmocn_cesm21.npz`` holds the oracle's
inputs and outputs (1054 states: ERA5 open ocean incl. 42 cold-air-outbreak
points, six named regimes, eight synthetic edges incl. one where the
flux_convergence = 0 early exit binds); its ``meta`` string records
the provenance.  Regenerate with::

    bash scripts/validate/oracle/shr_flux_atmocn/build.sh <cime5.6.47> <build>
    python scripts/validate/oracle/shr_flux_atmocn/make_fixture.py \\
        --drv <build>/drv --out tests/core/oracle_fixtures/shr_flux_atmocn_cesm21.npz

The oracle's own sea-surface humidity ``ssq`` is fed in as ``q_sfc`` (CESM's
private qsat fit is not the model's curve).  CESM fluxes are positive DOWN and
its stress is the stress on the ocean along the wind, so the model's fluxes and
stress are compared with a sign flip.  Pass: max relative error <= 1e-10
(fp64), and exact agreement where the oracle output is exactly zero.
"""
from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax.test_util import check_grads  # noqa: E402
from legoesm.core.bulk_flux import MOST_ZETA_CAP_WIDTH, compute_sam_oceflx_fluxes  # noqa: E402

_FIXTURE = Path(__file__).parent / "oracle_fixtures" / "shr_flux_atmocn_cesm21.npz"
_RTOL = 1e-10
# The smooth |hol| cap alone (default width minus the source's clamp), per state.
_CAP_GAP_FILE = Path(__file__).parent / "oracle_fixtures" / "shr_flux_atmocn_hol_cap_gap.npz"


def _fixture():
    return np.load(_FIXTURE)


def _ours(d, zeta_cap_width=0.0):
    """Default: the source's hard |hol| <= 10 clamp (RULE 5 hard limit)."""
    j = {k: jnp.asarray(d[k]) for k in d.files if k.startswith(("in_", "oracle_"))}
    tx, ty, sh, lh, us, tref = compute_sam_oceflx_fluxes(
        j["in_ubot"], j["in_vbot"], j["in_thbot"], j["in_qbot"], j["in_ts"],
        j["oracle_ssq"], j["in_rbot"], z_bot=j["in_zbot"], t_bot=j["in_tbot"],
        variant="cesm", return_2m=True, zeta_cap_width=zeta_cap_width)
    return {"taux": -tx, "tauy": -ty, "sen": -sh, "lat": -lh, "ustar": us, "tref": tref}


def test_fixture_covers_the_cold_air_outbreak_and_both_stabilities():
    d = _fixture()
    assert "cime5.6.47" in str(d["meta"]) and "coldair_outbreak_mod=.true." in str(d["meta"])
    assert int(np.sum(d["in_tbot"] - d["in_ts"] < -10.0)) >= 20
    delt = d["in_thbot"] - d["in_ts"]
    assert (delt > 0).sum() >= 50 and (delt < 0).sum() >= 50
    # a state where the Fortran exact-ustar early exit changes the answer
    assert "edge_exact_ustar_repeat_early_exit" in set(d["labels"])


def test_cesm_variant_matches_compiled_shr_flux_atmocn():
    d = _fixture()
    ours = _ours(d)
    for k, a in ours.items():
        a = np.asarray(a)
        b = d[f"oracle_{k}"]
        zero = b == 0.0
        np.testing.assert_array_equal(a[zero], 0.0, err_msg=k)
        rel = np.abs(a[~zero] - b[~zero]) / np.abs(b[~zero])
        assert rel.max() <= _RTOL, (k, float(rel.max()))


def test_cesm_variant_ustar_matches_at_1e12():
    """Campaign bar (rel 1e-12, fp64).  An additive 1e-12 m^2/s^2 in the wind
    speed's square root (the calm-wind AD guard) moved ustar by 1.5e-12."""
    d = _fixture()
    a, b = np.asarray(_ours(d)["ustar"]), d["oracle_ustar"]
    nz = b != 0.0
    assert (np.abs(a[nz] - b[nz]) / np.abs(b[nz])).max() <= 1e-12


def test_calm_wind_value_is_the_floor_and_gradient_finite():
    z = jnp.zeros(1)
    args = (z, z, jnp.full(1, 300.0), jnp.full(1, 0.015), jnp.full(1, 301.0),
            jnp.full(1, 0.02), jnp.full(1, 1.2))
    kw = dict(z_bot=jnp.full(1, 10.0), t_bot=jnp.full(1, 299.0), variant="cesm")
    g = jax.grad(lambda u, v: compute_sam_oceflx_fluxes(u, v, *args[2:], **kw)[4].sum(),
                 argnums=(0, 1))(*args[:2])
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g)
    u = 1e-3 * jnp.ones(1)
    calm = compute_sam_oceflx_fluxes(*args, **kw)[4]
    breeze = compute_sam_oceflx_fluxes(u, *args[1:], **kw)[4]
    assert float(calm[0]) == float(breeze[0])  # both at the minwind floor
    nan = compute_sam_oceflx_fluxes(jnp.full(1, jnp.nan), *args[1:], **kw)[4]
    assert bool(jnp.isnan(nan[0]))  # a NaN wind is not floored away


def test_cesm_variant_gradient_finite_on_fixture():
    d = _fixture()
    J = lambda k: jnp.asarray(d[k])  # noqa: E731

    def loss(u, th, q, ts, tb):
        r = compute_sam_oceflx_fluxes(
            u, J("in_vbot"), th, q, ts, J("oracle_ssq"), J("in_rbot"),
            z_bot=J("in_zbot"), t_bot=tb, variant="cesm")
        return jnp.sum(r[0] + r[2] + r[3] + r[4])

    g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
        J("in_ubot"), J("in_thbot"), J("in_qbot"), J("in_ts"), J("in_tbot"))
    for x in g:
        assert np.all(np.isfinite(np.asarray(x)))
    # t_bot carries gradient exactly where the cold-air-outbreak block is active
    coa = np.asarray(d["in_tbot"] - d["in_ts"] < -10.0)
    assert np.all(np.asarray(g[4])[coa] != 0.0) and np.all(np.asarray(g[4])[~coa] == 0.0)


def test_cesm_variant_gradient_finite_at_the_cold_air_outbreak_threshold():
    """Exactly at ``tbot - ts == td0 = -10 K`` the oracle's block is off
    (``tdiff .lt. td0`` false, shr_flux_mod.F90:329), so the forward value
    is the no-outbreak one; the unused branch's sqrt(0) must not turn the
    gradient into NaN (oracle audit row 57, safe-where)."""
    d = _fixture()
    J = lambda k: jnp.asarray(d[k])  # noqa: E731
    # Dyadic SST so tbot - ts is exactly -10.0 in fp64.
    ts = jnp.round(J("in_ts") * 4.0) / 4.0
    tb = ts - 10.0
    assert np.all(np.asarray(tb - ts) == -10.0)

    def flux(u, th, q, ts_, tb_):
        return compute_sam_oceflx_fluxes(
            u, J("in_vbot"), th, q, ts_, J("oracle_ssq"), J("in_rbot"),
            z_bot=J("in_zbot"), t_bot=tb_, variant="cesm", return_2m=True)

    args = (J("in_ubot"), J("in_thbot"), J("in_qbot"), ts)
    at_td0 = flux(*args, tb)
    no_coa = flux(*args, ts)        # tdiff = 0: block off
    for a, b in zip(at_td0, no_coa):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def loss(u, th, q, ts_, tb_):
        r = flux(u, th, q, ts_, tb_)
        return jnp.sum(r[0] + r[2] + r[3] + r[4])

    g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(*args, tb)
    for x in g:
        assert np.all(np.isfinite(np.asarray(x)))
    assert np.all(np.asarray(g[4]) == 0.0)


def test_generator_driver_constants_are_legoesm_constants():
    """The oracle driver hard-codes legoESM's constants (driver.F90); the
    generator's mirror of them must equal both the Fortran and
    ``legoesm.constants``, or a regenerated fixture tests other constants."""
    import re

    from legoesm import constants
    from scripts.validate.oracle.shr_flux_atmocn import make_fixture as mf

    for k, v in mf.DRIVER_LEGO.items():
        assert getattr(constants, k) == v, k
    src = (Path(mf.__file__).parent / "driver.F90").read_text()
    fort = {m[0]: float(m[1]) for m in re.findall(r"L_(\w+)=([0-9.e]+)_R8", src)}
    name = {"RD": "R_d", "RV": "R_v", "CPD": "c_pd", "CPV": "c_pv", "G": "g",
            "KARMAN": "kappa_von_karman", "LV": "L_v"}
    assert set(fort) == set(name)
    for f, k in name.items():
        assert fort[f] == mf.DRIVER_LEGO[k], f


# --- smooth |hol| cap (RULE 5): the source's [-10, 10] clamp ------------------
def _cap_gaps():
    d = _fixture()
    soft, hard = _ours(d, MOST_ZETA_CAP_WIDTH), _ours(d, 0.0)
    return {k: np.asarray(soft[k]) - np.asarray(hard[k]) for k in soft}


def test_hol_cap_gap_per_state():
    """KNOWN DIFFERENCE of the smooth cap vs the source clamp, pinned per state."""
    stored = np.load(_CAP_GAP_FILE)
    gaps = _cap_gaps()
    assert sorted(gaps) == sorted(stored.files)
    for k, g in gaps.items():
        np.testing.assert_allclose(g, stored[k], rtol=1e-6, atol=1e-12, err_msg=k)
    assert max(np.abs(stored["sen"]).max(), np.abs(stored["lat"]).max()) > 1e-6


def test_check_grads_hol_cap_at_default():
    """At the fixture state the cap moves most: smooth (C2) in the sea temperature."""
    d = _fixture()
    i = int(np.argmax(np.abs(_cap_gaps()["sen"])))
    x = {k: jnp.asarray(d[k][i:i + 1]) for k in d.files if k.startswith(("in_", "oracle_"))}

    def f(ts, w=MOST_ZETA_CAP_WIDTH):
        out = compute_sam_oceflx_fluxes(
            x["in_ubot"], x["in_vbot"], x["in_thbot"], x["in_qbot"], jnp.atleast_1d(ts),
            x["oracle_ssq"], x["in_rbot"], z_bot=x["in_zbot"], t_bot=x["in_tbot"],
            variant="cesm", zeta_cap_width=w)
        return jnp.sum(out[0] + out[2] + out[3] + out[4])
    t0 = x["in_ts"][0]
    assert float(f(t0)) != float(f(t0, 0.0))
    check_grads(f, (t0,), order=2, modes=("fwd", "rev"), eps=1e-6)
    assert float(jax.grad(f)(t0)) != 0.0


def test_sam_lane_keeps_the_clamp_and_production_resolves_to_the_smooth_cap():
    d = _fixture()
    j = {k: jnp.asarray(d[k]) for k in d.files if k.startswith(("in_", "oracle_"))}
    args = (j["in_ubot"], j["in_vbot"], j["in_thbot"], j["in_qbot"], j["in_ts"],
            j["oracle_ssq"], j["in_rbot"])
    for a, b in zip(compute_sam_oceflx_fluxes(*args, z_bot=j["in_zbot"]),
                    compute_sam_oceflx_fluxes(*args, z_bot=j["in_zbot"], zeta_cap_width=0.0)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    from legoesm.coupler.config import CouplerConfig
    from legoesm.driver.config import ExperimentConfig
    for c in (SurfaceLayerConfig(), CouplerConfig()):
        assert c.most_zeta_cap_width == MOST_ZETA_CAP_WIDTH
    assert ExperimentConfig().surface_most_zeta_cap_width == MOST_ZETA_CAP_WIDTH
    # The production deck (CESM bulk on the CLUBB surface) resolves to it.
    import importlib.util
    from legoesm.driver.physics_pipeline import turbulence_config_for
    from legoesm.driver.run_config_yaml import load_yaml_config
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("_ra_cap", root / "scripts/run/run_amip.py")
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)
    p = ra.build_arg_parser()
    p.set_defaults(**load_yaml_config(str(root / "config/amip/amip_production.yaml"), p))
    dummy = ["--ic-path", "/d/e.zarr", "--forcing-path", "/d/s.nc", "--sic-path", "/d/i.nc",
             "--solar-file", "/d/so.nc", "--ozone-file", "/d/o.nc", "--ghg-file", "/d/g.nc",
             "--aerosol-file", "/d/a.nc", "--volcanic-aerosol-file", "/d/v.nc",
             "--topography", "/d/t.nc", "--days", "10", "--output", "/d/out"]
    cfg = ra.build_config_from_args(ra._postprocess_args(p.parse_args(dummy), p))
    surf = getattr(turbulence_config_for(cfg), cfg.turbulence).surface
    assert (cfg.surface_bulk_scheme, surf.bulk_scheme) == ("large_yeager_cesm",) * 2
    assert surf.most_zeta_cap_width == MOST_ZETA_CAP_WIDTH == 2.0


if __name__ == "__main__":
    np.savez_compressed(_CAP_GAP_FILE, **_cap_gaps())
    print("wrote", _CAP_GAP_FILE)

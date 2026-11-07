"""Independent unstable-regime Monin-Obukhov form (``unstable_scheme``).

``stability_scheme`` selects the stable branch (zeta > 0); ``unstable_scheme``
selects the unstable branch (zeta < 0) independently:

- ``"native"`` (default): each bulk path's own form -- Businger-Dyer in
  ``psi_m``/``psi_h`` (most / large_yeager), the COARE 3.0 Fairall blend in
  ``psi_m_coare``/``psi_h_coare`` (coare3).  Must stay byte-identical.
- ``"businger_dyer"`` / ``"coare3"``: force that form on every path.

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.core.bulk_flux import (
    VALID_UNSTABLE_SCHEMES,
    compute_most_fluxes,
    psi_h,
    psi_h_coare,
    psi_m,
    psi_m_coare,
    validate_unstable_scheme,
)

REPO = Path(__file__).resolve().parents[2]
STABLE = ("dyer1974", "beljaars_holtslag1991", "grachev2007_sheba", "gryanik2020")
ZETA = jnp.linspace(-8.0, 8.0, 801)
# Outside the COARE neutral blend half-width (1e-3): pure unstable branch.
ZNEG = jnp.linspace(-8.0, -0.01, 400)


def _phi(psi_fn, z):
    """phi = 1 - zeta dpsi/dzeta (pointwise derivative of a scalar psi)."""
    d = jax.vmap(jax.grad(lambda x: psi_fn(x)))(z)
    return 1.0 - z * d


# --- defaults: native is the default, businger_dyer == native on the BD path,
#     coare3 == native on the COARE path (bit-for-bit) --------------------------
@pytest.mark.parametrize("stable", STABLE)
def test_native_aliases_are_bit_identical(stable):
    for f in (psi_m, psi_h):
        base = np.asarray(f(ZETA, stable))
        np.testing.assert_array_equal(base, np.asarray(f(ZETA, stable, unstable_scheme="native")))
        np.testing.assert_array_equal(
            base, np.asarray(f(ZETA, stable, unstable_scheme="businger_dyer")))
    for f in (psi_m_coare, psi_h_coare):
        base = np.asarray(f(ZETA, stable))
        np.testing.assert_array_equal(base, np.asarray(f(ZETA, stable, unstable_scheme="native")))
        if not (f is psi_h_coare and stable == "gryanik2020"):
            np.testing.assert_array_equal(
                base, np.asarray(f(ZETA, stable, unstable_scheme="coare3")))


# --- default == frozen PRE-CHANGE outputs (base branch, scripts/tmp/
#     gen_unstable_golden.py): values, derivatives and full flux solves --------
_GOLDEN = REPO / "tests" / "core" / "oracle_fixtures" / "unstable_scheme_native_golden.npz"


def test_default_matches_frozen_pre_change_outputs():
    g = np.load(_GOLDEN)
    zeta = jnp.asarray(g["zeta"])
    fns = dict(psi_m=psi_m, psi_h=psi_h, psi_m_coare=psi_m_coare, psi_h_coare=psi_h_coare)
    for s in STABLE:
        for name, fn in fns.items():
            np.testing.assert_array_equal(fn(zeta, s), g[f"{name}/{s}"])
            d = jax.vmap(jax.grad(lambda z: fn(z, s)))(zeta)
            np.testing.assert_array_equal(d, g[f"d{name}/{s}"])
    st = {k: g[f"in/{k}"] for k in ("u", "v", "Ta", "dT", "qa", "dq")}
    n = len(st["u"])
    for scheme in ("most", "coare3", "large_yeager"):
        for s in STABLE:
            r = compute_most_fluxes(st["u"], st["v"], st["Ta"], st["qa"], st["Ta"] + st["dT"],
                                    np.maximum(st["qa"] + st["dq"], 1e-5), np.full(n, 1.2),
                                    scheme=scheme, stability_scheme=s, return_2m=True,
                                    z0_init=1e-3)
            for i, x in enumerate(r):
                np.testing.assert_array_equal(x, g[f"flux/{scheme}/{s}/{i}"],
                                              err_msg=f"{scheme}/{s}/{i}")


# --- the swap: unstable side changes to the other family, stable side intact ---
@pytest.mark.parametrize("stable", ("dyer1974", "grachev2007_sheba"))
def test_swap_uses_the_other_family_on_the_unstable_side_only(stable):
    np.testing.assert_allclose(psi_m(ZNEG, stable, unstable_scheme="coare3"),
                               psi_m_coare(ZNEG, stable), rtol=0, atol=1e-12)
    np.testing.assert_allclose(psi_h(ZNEG, stable, unstable_scheme="coare3"),
                               psi_h_coare(ZNEG, stable), rtol=0, atol=1e-12)
    np.testing.assert_allclose(psi_m_coare(ZNEG, stable, unstable_scheme="businger_dyer"),
                               psi_m(ZNEG, stable), rtol=0, atol=1e-12)
    np.testing.assert_allclose(psi_h_coare(ZNEG, stable, unstable_scheme="businger_dyer"),
                               psi_h(ZNEG, stable), rtol=0, atol=1e-12)
    # the families really differ there (non-vacuity of the two checks above)
    assert np.max(np.abs(np.asarray(psi_m(ZNEG, stable) - psi_m_coare(ZNEG, stable)))) > 1e-2
    zpos = jnp.linspace(0.01, 8.0, 200)
    for f in (psi_m, psi_h, psi_m_coare, psi_h_coare):
        for u in VALID_UNSTABLE_SCHEMES:
            np.testing.assert_array_equal(np.asarray(f(zpos, stable, unstable_scheme=u)),
                                          np.asarray(f(zpos, stable)))


# --- smoothness rule: phi continuous at 0 for every stable x unstable pair ----
@pytest.mark.parametrize("stable", STABLE)
@pytest.mark.parametrize("unstable", ("businger_dyer", "coare3"))
def test_phi_continuous_at_zero(stable, unstable):
    eps = jnp.asarray([-1e-7, 1e-7])
    for f in (psi_m, psi_h):
        lo, hi = np.asarray(_phi(lambda x: f(x, stable, unstable_scheme=unstable), eps))
        assert abs(lo - hi) < 1e-5, (f.__name__, stable, unstable, lo, hi)
    for f in (psi_m_coare, psi_h_coare):   # neutral_width=0: the pure branches
        lo, hi = np.asarray(_phi(
            lambda x: f(x, stable, 0.0, unstable_scheme=unstable), eps))
        assert abs(lo - hi) < 1e-5, (f.__name__, stable, unstable, lo, hi)


def test_gryanik_pr0_scales_the_explicit_coare3_unstable_heat_branch():
    z = jnp.asarray(-0.5)
    np.testing.assert_allclose(
        psi_h(z, "gryanik2020", unstable_scheme="coare3"),
        0.98 * psi_h(z, "dyer1974", unstable_scheme="coare3"), rtol=1e-14)
    np.testing.assert_allclose(
        psi_h_coare(z, "gryanik2020", unstable_scheme="coare3"),
        0.98 * psi_h_coare(z, "gryanik2020"), rtol=1e-14)


# --- gradients finite through the new branches -------------------------------
def test_gradients_through_new_branches():
    check_grads(lambda z: psi_m(z, "dyer1974", unstable_scheme="coare3"),
                (jnp.asarray(-0.7),), order=2, modes=["rev"])
    check_grads(lambda z: psi_h_coare(z, "grachev2007_sheba", unstable_scheme="businger_dyer"),
                (jnp.asarray(-0.7),), order=2, modes=["rev"])


# --- dispatch hardening ------------------------------------------------------
def test_unknown_unstable_scheme_raises():
    with pytest.raises(ValueError, match="unstable_scheme"):
        validate_unstable_scheme("businger")
    for f in (psi_m, psi_h, psi_m_coare, psi_h_coare):
        with pytest.raises(ValueError, match="unstable_scheme"):
            f(ZETA, "dyer1974", unstable_scheme="coare30")
    with pytest.raises(ValueError, match="unstable_scheme"):
        compute_most_fluxes(3.0, 0.0, 290.0, 0.008, 295.0, 0.012, 1.2,
                            unstable_scheme="bd")


def test_nemo_exact_and_coare_bd_width_guards():
    z = jnp.asarray([-0.3, 0.2])
    np.testing.assert_array_equal(psi_m(z, nemo_source_exact=True, unstable_scheme="businger_dyer"),
                                  psi_m(z, nemo_source_exact=True))
    with pytest.raises(ValueError, match="unstable_scheme"):
        psi_h(z, nemo_source_exact=True, unstable_scheme="coare3")
    with pytest.raises(ValueError, match="neutral_width"):
        psi_m_coare(z, "dyer1974", 0.1, unstable_scheme="businger_dyer")
    assert np.all(np.isfinite(psi_h_coare(z, "dyer1974", 0.05, unstable_scheme="businger_dyer")))


# --- compute_most_fluxes: unstable column moves, stable column does not ------
def _fluxes(scheme, unstable, t_sfc):
    out = compute_most_fluxes(jnp.full(3, 4.0), jnp.zeros(3), jnp.full(3, 290.0),
                              jnp.full(3, 0.008), jnp.full(3, t_sfc),
                              jnp.full(3, 0.011), jnp.full(3, 1.2),
                              scheme=scheme, z0_init=1e-3, unstable_scheme=unstable)
    return np.asarray(out[2])


@pytest.mark.parametrize("scheme,other", [("coare3", "businger_dyer"),
                                          ("large_yeager", "coare3")])
def test_flux_solver_threads_unstable_scheme(scheme, other):
    warm = 296.0   # surface warmer than air: unstable
    assert not np.allclose(_fluxes(scheme, "native", warm), _fluxes(scheme, other, warm))
    cold = 284.0   # stable: unstable branch never selected after convergence
    np.testing.assert_allclose(_fluxes(scheme, "native", cold),
                               _fluxes(scheme, other, cold), rtol=1e-12)


def test_surface_layer_config_threads_unstable_scheme():
    def sh(u):
        cfg = SurfaceLayerConfig(bulk_scheme="coare3", unstable_scheme=u)
        args = [jnp.full(3, v) for v in (4.0, 0.0, 290.0, 0.008, 296.0, 0.011, 1.2)]
        return np.asarray(compute_surface_fluxes(*args, cfg)[2])
    assert SurfaceLayerConfig().unstable_scheme == "native"
    assert not np.allclose(sh("native"), sh("businger_dyer"))


def test_coupler_and_sea_ice_thread_unstable_scheme():
    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ice.config import SeaIceConfig
    from legoesm.ice.sea_ice import _bulk_flux_dispatch

    shape = (2, 3)

    def full(v):
        return jnp.full(shape, v)

    forcing = AtmToSurface(
        sw_down=full(200.0), lw_down=full(300.0),
        precip_total=jnp.zeros(shape), precip_snow=jnp.zeros(shape),
        T_lowest=full(270.0), q_lowest=full(0.002),
        u_lowest=full(3.0), v_lowest=full(0.5),
        p_lowest=full(95000.0), p_surface=full(101325.0),
        rho_lowest=full(1.2), cos_zenith=full(0.5),
        co2_ppmv=jnp.asarray(400.0),
        has_radiation=jnp.asarray(1.0), has_precipitation=jnp.asarray(1.0),
    )
    zeros = jnp.zeros(shape)

    def ocean(u):
        cfg = CouplerConfig(bulk_scheme="coare3", unstable_scheme=u)
        return np.asarray(ocean_tile_response(forcing, full(285.0), zeros, zeros, cfg).shflx)

    def ice(u):
        cfg = SeaIceConfig(bulk_scheme="most", unstable_scheme=u)
        return np.asarray(_bulk_flux_dispatch(full(272.0), forcing, cfg, U_min=0.5)[2])

    assert CouplerConfig().unstable_scheme == "native"
    assert SeaIceConfig().unstable_scheme == "native"
    assert not np.allclose(ocean("native"), ocean("businger_dyer"))
    assert not np.allclose(ice("native"), ice("coare3"))


# --- experiment config, injection, CLIs --------------------------------------
def test_experiment_config_validates_and_injects():
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import turbulence_config_for

    assert ExperimentConfig().surface_unstable_scheme == "native"
    with pytest.raises(ValueError, match="surface_unstable_scheme"):
        ExperimentConfig(turbulence="louis", surface_bulk_scheme="coare3",
                         surface_unstable_scheme="bd").validate_strict()
    for bulk in ("constant", "large_yeager_cesm"):
        with pytest.raises(ValueError, match="surface_unstable_scheme"):
            ExperimentConfig(turbulence="louis", surface_bulk_scheme=bulk,
                             surface_unstable_scheme="coare3").validate_strict()
    ExperimentConfig(turbulence="louis", surface_bulk_scheme="large_yeager",
                     surface_unstable_scheme="coare3").validate_strict()
    cfg = ExperimentConfig(turbulence="louis", surface_bulk_scheme="coare3",
                           surface_unstable_scheme="businger_dyer")
    cfg.validate_strict()
    assert turbulence_config_for(cfg).louis.surface.unstable_scheme == "businger_dyer"
    # default experiment: injection left alone (fast path keeps identity)
    tc = turbulence_config_for(ExperimentConfig(turbulence="louis"))
    assert tc.louis.surface.unstable_scheme == "native"


def test_air_sea_consistency_rejects_an_unstable_split():
    from legoesm.coupler.config import CouplerConfig
    from legoesm.driver.air_sea_consistency import validate_air_sea_consistency
    from legoesm.driver.config import ExperimentConfig

    atm = ExperimentConfig(turbulence="louis", surface_bulk_scheme="coare3",
                           surface_unstable_scheme="businger_dyer")
    validate_air_sea_consistency(
        atm, CouplerConfig(bulk_scheme="coare3", unstable_scheme="businger_dyer"))
    with pytest.raises(ValueError, match="unstable_scheme"):
        validate_air_sea_consistency(atm, CouplerConfig(bulk_scheme="coare3"))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}_mod", REPO / "scripts" / "run" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_flags_round_trip():
    amip = _load("run_amip")
    p = amip.build_arg_parser()
    assert p.parse_args([]).surface_unstable_scheme == "native"
    assert p.parse_args(["--surface-unstable-scheme", "coare3"]).surface_unstable_scheme == "coare3"
    with pytest.raises(SystemExit):
        p.parse_args(["--surface-unstable-scheme", "bd"])
    cfg = amip.build_config_from_args(p.parse_args(
        ["--surface-bulk-scheme", "coare3", "--surface-unstable-scheme", "coare3"]))
    assert cfg.surface_unstable_scheme == "coare3"

    from scripts.run.run_omip import parse_args
    assert parse_args(["--grid", "latlon"]).surface_unstable_scheme == "native"
    assert parse_args(["--grid", "latlon", "--surface-unstable-scheme",
                       "businger_dyer"]).surface_unstable_scheme == "businger_dyer"

    coupled = _load("run_coupled")
    args = coupled.build_parser().parse_args(
        ["--surface-bulk-scheme", "coare3", "--surface-unstable-scheme", "businger_dyer"])
    assert coupled.build_coupler_config(args).unstable_scheme == "businger_dyer"
    args = coupled.build_parser().parse_args([])
    assert coupled.build_coupler_config(args) is None
    args = coupled.build_parser().parse_args(
        ["--ice-bulk-scheme", "most", "--ice-unstable-scheme", "coare3"])
    assert coupled.build_sea_ice_config(args).unstable_scheme == "coare3"
    args = coupled.build_parser().parse_args(["--ice-unstable-scheme", "coare3"])
    with pytest.raises(SystemExit):
        coupled.build_sea_ice_config(args)

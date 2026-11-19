"""Reverse-mode derivative of prognostic CLUBB on real production columns.

Measured 2026-10-29 on the production deck run at res 3 (ERA5 1979-01-05, first
physics call, x64, one CLUBB sub-step): reverse AD of the turbulence module
with respect to temperature was NaN in ~45 % of the entries under ``jit`` (and
under the calibration branch's loop remat), finite when the same gradient was
run op by op.  Cause: the monotonic flux limiter's windowed ``jnp.min`` /
``jnp.max``.  JAX differentiates a min/max reduction by locating the chosen
element through EQUALITY with the reduced value and dividing by the number of
matches; XLA recomputed the operand in a second fusion that rounded 1 ulp
differently (-300.36085384784803 vs -300.360853847848), nothing matched, and
the gradient was 0/0.  The limiter now picks the element by index.

The fixture is four columns of that call (``clubb_fixtures/
clubb_ad_real_columns.npz``: the column inputs the MPAS lane hands
``clubb_turbulence_prognostic``, its land-injected surface flux and per-column
prescribed evaporation; the moments are the cold-start carry, which is what
that call carried).  The CLUBB configuration is the production deck's, built
through run_amip's own argument path.  Heights enter as an independent input:
on the lane they are hydrostatic functions of T, and differentiating through
them is what exposes the defect (with T alone the compiled graph happened not
to trip it).
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jax.test_util import check_grads  # noqa: E402

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    clubb_turbulence_prognostic,
    init_clubb_moments,
    pack_clubb_moments,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FIX = pathlib.Path(__file__).resolve().parent / "clubb_fixtures" / "clubb_ad_real_columns.npz"


def _production_clubb_config():
    """The deck's CLUBBConfig, as the MPAS lane builds it (run_amip args ->
    ExperimentConfig -> turbulence_config_for)."""
    from legoesm.atmosphere.physics.turbulence.integration import materialize_sub_config
    from legoesm.driver.physics_pipeline import turbulence_config_for
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    parser = build_arg_parser()
    keys = load_yaml_config(str(_ROOT / "config" / "amip" / "amip_production.yaml"), parser)
    parser.set_defaults(**keys, _config_keys=frozenset(keys))
    # placeholder input paths (never opened): the parser requires them
    argv = ["--ic-path", "/dummy/era5.zarr", "--forcing-path", "/dummy/sst.nc",
            "--sic-path", "/dummy/sic.nc", "--solar-file", "/dummy/solar.nc",
            "--ozone-file", "/dummy/o3.nc", "--ghg-file", "/dummy/ghg.nc",
            "--aerosol-file", "/dummy/aero.nc",
            "--volcanic-aerosol-file", "/dummy/volc.nc",
            "--topography", "/dummy/etopo.nc", "--days", "1",
            "--output", "/dummy/out"]
    args = _postprocess_args(parser.parse_args(argv), parser, argv)
    cfg = build_config_from_args(args)
    assert cfg.turbulence == "clubb" and cfg.clubb_prognostic and cfg.clubb_liquid_partition
    return materialize_sub_config(turbulence_config_for(cfg)).clubb


def _case():
    d = dict(np.load(_FIX))
    a = {k: jnp.asarray(v) for k, v in d.items()}
    cfg = _production_clubb_config()
    # the lane folds the land model's evaporation into the surface sub-config
    cfg = cfg._replace(surface=cfg.surface._replace(
        prescribed_evap_kg_m2_s=a["prescribed_evap_kg_m2_s"]))
    ncol, nlev = a["T"].shape
    fixed = dict(
        u=a["u"], v=a["v"], p_full=a["p_full"], p_half=a["p_half"],
        z_half=a["z_half"], T_sfc=a["T_sfc"], q_sfc=a["q_sfc"], rho=a["rho"],
        q_c=a["q_c"], lat=a["lat"], dt=float(d["dt"]),
        clubb_moments=pack_clubb_moments(init_clubb_moments(ncol, nlev, cfg)),
        surface_flux=tuple(a["sfc_" + s] for s in
                           ("tau_x", "tau_y", "shflx", "lhflx", "ustar")))
    x0 = (a["T"], a["q_v"], a["z_full"])
    return cfg, fixed, x0


@pytest.fixture(scope="module")
def case():
    return _case()


def _fn(cfg, fixed):
    def f(T, q_v, z_full):
        out, moments = clubb_turbulence_prognostic(
            T=T, q_v=q_v, z_full=z_full, config=cfg, **fixed)
        return out, moments
    return f


# 600 s = the production CLUBB call (1800 s / 3 macmic): clubb_dt 600 is one
# sub-step (the failing case), 300 the deck's two sub-steps.
@pytest.mark.parametrize("clubb_dt", [600.0, 300.0])
def test_clubb_reverse_ad_is_finite_under_jit(case, clubb_dt):
    cfg, fixed, x0 = case
    f = _fn(cfg._replace(clubb_dt=clubb_dt), fixed)
    rng = np.random.default_rng(0)
    w = jax.tree_util.tree_map(lambda y: jnp.asarray(rng.standard_normal(y.shape)),
                               jax.eval_shape(f, *x0))

    def loss(T, q_v, z_full):
        return sum(jnp.vdot(a, b) for a, b in zip(jax.tree_util.tree_leaves(w),
                                                  jax.tree_util.tree_leaves(f(T, q_v, z_full))))
    g = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(*x0)
    for name, gi in zip(("T", "q_v", "z_full"), g):
        bad = int(np.sum(~np.isfinite(np.asarray(gi))))
        assert bad == 0, f"d/d{name}: {bad} non-finite of {gi.size}"


@pytest.mark.parametrize("clubb_dt", [600.0, 300.0])
def test_clubb_vjp_matches_finite_differences(case, clubb_dt):
    """check_grads (reverse mode, order 1, jax's fp64 tolerance and step) on the
    jitted step.  The inputs are rescaled to the AD map's perturbation sizes
    (0.1 K, 1e-3 relative q_v, 1 m) so the unit random direction stays in the
    linear range.  Forward mode is not checked: the shared Thomas solver is a
    custom_vjp (no JVP rule)."""
    cfg, fixed, (T0, q0, z0) = case
    f0 = _fn(cfg._replace(clubb_dt=clubb_dt), fixed)

    def f(a, b, c):
        return f0(T0 + 0.1 * a, q0 * (1.0 + 1.0e-3 * b), z0 + 1.0 * c)
    zeros = (jnp.zeros_like(T0), jnp.zeros_like(q0), jnp.zeros_like(z0))
    check_grads(jax.jit(f), zeros, order=1, modes=("rev",))

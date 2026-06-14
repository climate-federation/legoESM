"""TrainableOceanParams: transforms, round-trips, and per-parameter
LIVENESS through the real GEOMETRIC consuming formulas.

The liveness pins are the ocean instance of the AIMIP dead-DOF lesson
(commit d018a795 / test_physics_params_audit.py): every trainable knob
must produce a FINITE, NONZERO derivative through the code path that the
model actually executes — at states designed to keep the formula off its
clips. A knob whose plumbing dies (renamed field, discarded branch) must
fail here before any calibration compute is spent.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.lateral_mixing.eke import (
    EKEConfig,
    GeometricConfig,
    eke_apply_local_source,
    geometric_dissipation_length,
    geometric_kappa_gm,
    geometric_rossby_radius,
)
from legoesm.training.trainable_ocean_params import (
    GEOMETRIC_TRAINABLE,
    TrainableOceanParams,
    constrain,
    ensemble_from_priors,
    names,
    unconstrain,
)


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def test_default_roundtrip_and_interior():
    p = TrainableOceanParams.from_defaults()
    vals = p.as_dict()
    for spec in GEOMETRIC_TRAINABLE:
        c = spec.constraint
        got = float(vals[c.name])
        assert np.isclose(got, spec.default, rtol=1e-10), (
            f"{c.name}: round-trip {got} != default {spec.default}")
        # saturation lesson: defaults must sit interior with real margin
        lo, hi = c.min_val, c.max_val
        margin = 0.02
        assert lo * (1 + margin) < got < hi * (1 - margin), (
            f"{c.name} default {got} too close to bounds [{lo}, {hi}]")


def test_bounds_respected_at_extremes():
    for spec in GEOMETRIC_TRAINABLE:
        lo = float(constrain(jnp.asarray(-50.0), spec))
        hi = float(constrain(jnp.asarray(50.0), spec))
        assert lo >= spec.constraint.min_val * (1 - 1e-9)
        assert hi <= spec.constraint.max_val * (1 + 1e-9)


def test_unconstrain_constrain_inverse():
    for spec in GEOMETRIC_TRAINABLE:
        for frac in (0.2, 0.5, 0.8):
            c = spec.constraint
            if spec.log_space:
                v = 10.0 ** (np.log10(c.min_val)
                             + frac * (np.log10(c.max_val) - np.log10(c.min_val)))
            else:
                v = c.min_val + frac * (c.max_val - c.min_val)
            back = float(constrain(jnp.asarray(unconstrain(v, spec)), spec))
            assert np.isclose(back, v, rtol=1e-9), (spec.constraint.name, v, back)


def test_to_geometric_config_and_traced_projection():
    base = GeometricConfig()
    p = TrainableOceanParams.from_defaults()
    cfg = p.to_geometric_config(base)
    assert isinstance(cfg, GeometricConfig)
    # non-trainable fields untouched
    assert cfg.mn_floor == base.mn_floor
    # projection is differentiable end-to-end (traced config fields)
    def f(raw):
        cfg_t = TrainableOceanParams(raw, p.specs).to_geometric_config(base)
        return cfg_t.alpha * 1.0 + cfg_t.kappa_u * 1e-4
    g = jax.grad(f)(p.raw)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g[0])) > 0 and float(jnp.abs(g[2])) > 0


def test_ensemble_from_priors_shape_and_spread():
    ens = ensemble_from_priors(jax.random.PRNGKey(0), 64)
    assert ens.shape == (64, len(GEOMETRIC_TRAINABLE))
    base = TrainableOceanParams.from_defaults().raw
    # mean near defaults; nonzero spread on every parameter
    assert float(jnp.max(jnp.abs(jnp.mean(ens, 0) - base))) < 0.5
    assert float(jnp.min(jnp.std(ens, 0))) > 0.1
    # constrained members stay strictly inside bounds (sigmoid guarantees)
    for i, spec in enumerate(GEOMETRIC_TRAINABLE):
        vals = np.asarray(constrain(ens[:, i], spec))
        assert vals.min() > spec.constraint.min_val
        assert vals.max() < spec.constraint.max_val


# ---------------------------------------------------------------------------
# per-parameter liveness through the REAL consuming formulas
# ---------------------------------------------------------------------------

def _grad_wrt_param(fn, value):
    g = jax.grad(fn)(jnp.asarray(value, jnp.float64))
    assert bool(jnp.isfinite(g)), "non-finite parameter gradient"
    assert float(jnp.abs(g)) > 0.0, "DEAD parameter: zero gradient"
    return float(g)


def test_liveness_alpha():
    """alpha -> geometric_kappa_gm (Eq. 6), away from the kappa clips."""
    geom = GeometricConfig()
    int_E = jnp.asarray(2.5)            # m^3/s^2, typical spun value
    int_m2n = jnp.asarray(5e-4)         # m/s — kappa ~ 0.04*2.5/5e-4 = 200

    def f(alpha):
        return geometric_kappa_gm(int_E, int_m2n, geom._replace(alpha=alpha))

    g = _grad_wrt_param(f, geom.alpha)
    assert np.isclose(g, float(int_E / jnp.maximum(int_m2n, geom.mn_floor)),
                      rtol=1e-12)  # analytic d kappa/d alpha


def test_liveness_rossby_factor():
    """rossby_factor -> geometric_rossby_radius, interior to the R_d clip."""
    geom = GeometricConfig()
    int_N_dz = jnp.asarray(4.0)         # m/s; R_d = 0.4*4/1e-4 = 16 km,
    f_cor = jnp.asarray(1.0e-4)         # interior to the [2, 40] km clip

    def f(rf):
        return geometric_rossby_radius(int_N_dz, f_cor,
                                       geom._replace(rossby_factor=rf))

    g = _grad_wrt_param(f, geom.rossby_factor)
    assert np.isclose(g, 4.0 / 1.0e-4, rtol=1e-12)


def test_liveness_c_eps_geometric():
    """c_eps_geometric -> the shared semi-implicit dissipation fold, via the
    EXACT threading the model uses (ocean_model_latlon_cgrid.py:
    ``cfg._replace(c_eps=geom.c_eps_geometric, k_iso=geom.kappa_e)``) with
    ``L = geometric_dissipation_length`` — d E_next / d c_eps < 0."""
    geom = GeometricConfig()
    r_d = jnp.asarray(16e3)
    H = jnp.asarray(2000.0)
    L_eff = geometric_dissipation_length(r_d, H)
    E = jnp.asarray(2.5e-2)             # m^2/s^2-scale fold input
    sigma = jnp.asarray(1e-6)
    cfg = EKEConfig()

    def f(c_eps):
        out = eke_apply_local_source(
            E, sigma, L_eff, cfg._replace(c_eps=c_eps), dt=4800.0)
        return out[0] if isinstance(out, tuple) else out

    g = _grad_wrt_param(f, geom.c_eps_geometric)
    assert g < 0.0, "more dissipation must reduce next-step EKE"


def test_liveness_kappa_u():
    """kappa_u -> geometric_barotropic_production B_T (Eq. 3): the depth
    integral is LINEAR in kappa_u, so dB_T/dkappa_u = B_T/kappa_u > 0 on a
    sheared flow."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        geometric_barotropic_production,
    )

    grid = create_latlon_grid(8, 12)
    key = jax.random.PRNGKey(0)
    nlev = 4
    u = 0.1 * jax.random.normal(key, (8, 13, nlev), dtype=jnp.float64)
    v = 0.1 * jax.random.normal(
        jax.random.fold_in(key, 1), (9, 12, nlev), dtype=jnp.float64)
    dz = jnp.full((8, 12, nlev), 250.0)
    mask = jnp.ones((8, 12))
    u_mask = jnp.ones((8, 13))
    v_mask = jnp.ones((9, 12))

    def f(kappa_u):
        bt = geometric_barotropic_production(
            u, v, grid, kappa_u, dz, mask, u_mask, v_mask)
        return jnp.sum(bt)

    geom = GeometricConfig()
    g = _grad_wrt_param(f, geom.kappa_u)
    # linearity: g == B_T(1500)/1500 == B_T(1.0)
    assert np.isclose(g, float(f(jnp.asarray(1.0))), rtol=1e-10)
    assert g > 0.0


def test_liveness_kappa_e_threading():
    """kappa_e is threaded verbatim into the shared EKE machinery as
    ``k_iso`` (ocean_model_latlon_cgrid.py:1931). Pin the threading is
    differentiable; full step-level liveness is covered by the Stage-0b
    twin campaign (the k_iso consumer is the shared, separately tested
    isopycnal E-diffusion)."""
    geom = GeometricConfig()
    cfg = EKEConfig()

    def f(kappa_e):
        threaded = cfg._replace(c_eps=geom.c_eps_geometric, k_iso=kappa_e)
        return threaded.k_iso * 2.0

    _grad_wrt_param(f, geom.kappa_e)


def test_names_order_stable():
    assert names() == ("alpha", "c_eps_geometric", "kappa_u", "kappa_e",
                       "rossby_factor")


def test_package_reexports():
    from legoesm.training import GEOMETRIC_TRAINABLE as g, TrainableOceanParams as t
    assert g is GEOMETRIC_TRAINABLE and t is TrainableOceanParams


def test_module_is_jit_and_grad_safe():
    """codex-review F1 pin: the params object must pass through jit/grad
    WHOLE (specs static, raw the only leaf) — a plain NamedTuple flattened
    spec strings into leaves and broke here."""
    import equinox as eqx

    p = TrainableOceanParams.from_defaults()
    val = jax.jit(lambda q: jnp.sum(q.raw))(p)
    assert bool(jnp.isfinite(val))
    g = eqx.filter_grad(lambda q: jnp.sum(q.as_dict()["alpha"] ** 2))(p)
    assert g.raw.shape == p.raw.shape
    assert bool(jnp.all(jnp.isfinite(g.raw)))
    assert float(jnp.abs(g.raw[0])) > 0.0
    leaves = jax.tree_util.tree_leaves(p)
    assert len(leaves) == 1, f"expected raw as the only leaf, got {len(leaves)}"

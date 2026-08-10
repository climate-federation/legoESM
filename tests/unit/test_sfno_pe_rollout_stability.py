"""Post-step damping + moisture clamp in the SFNO PE ``state_update`` rollout.

WHY THESE EXIST (measured, 2026-07-28, ``scripts/tmp/diag_sfno_full_rollout.py``
on a trained T63 ``sfno_full`` checkpoint): ``step()`` used to apply only a
dry-air-mass correction. Because ``unpack_pe_output`` rebuilds vorticity and
divergence from the network's grid-space ``u, v`` by spectral differentiation
(multiplying each coefficient by ~n), the small-scale end of the field grew
unchecked — |u850|max went 33 -> 158 m/s by macro step 3, 969 m/s at step 4,
1.3e4 at step 7, and the rollout NaN'd at step 12 (72 h). T, q and z500 left
physical range only at steps 6-7 and the global-mean p_s was pinned exactly by
the mass corrector until step 9, i.e. the wind cascade was the source.

``spectral_filter_strength`` (the dycore's own exponential filter, applied once
per macro step) and ``clip_q`` (ACE2 corrector step 1, arXiv:2411.11268 §4.3)
are the two knobs added for that. Each test below fails if its knob is removed
or silently defaulted back.

Same tiny-T8 fixtures as ``test_sfno_pe.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    isothermal_rest_state_spectral,
)
from legoesm.atmosphere.dynamics.neural.sfno_pe import (
    SFNOPrimitiveEquationConfig,
    SFNOPrimitiveEquationModel,
)
from legoesm.core.field import Field
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.ml.channel_packing import PE3DChannelSpec
from legoesm.ml.sfno import SFNOConfig

jax.config.update("jax_enable_x64", True)

_DT_SFNO = 21600.0


@pytest.fixture(scope="module")
def grid_t8():
    return create_gaussian_grid(n_max=8)


@pytest.fixture(scope="module")
def sigma_coord():
    return create_sigma_coordinate(n_levels=3)


@pytest.fixture(scope="module")
def sfno_config(sigma_coord):
    n_channels = PE3DChannelSpec(nlev=sigma_coord.n_levels).n_channels
    return SFNOConfig(in_channels=n_channels, out_channels=n_channels,
                      embed_dim=12, n_blocks=2, mlp_expansion=2)


def _state(grid, sigma, q_value=5.0e-3):
    nlev = sigma.n_levels
    qv = jnp.full((grid.n_lat, grid.n_lon, nlev), q_value, dtype=jnp.float64)
    tracers = {"q_v": Field(data=qv, name="q_v",
                            dims=("lat", "lon", "level"), units="kg/kg")}
    return isothermal_rest_state_spectral(grid, sigma, tracers=tracers)


def _model(grid, sigma, sfno_config, **cfg_kwargs):
    cfg = SFNOPrimitiveEquationConfig(
        sfno_config=sfno_config, mode="state_update", dt_sfno=_DT_SFNO,
        correct_mass=False, correct_moisture_budget=False, **cfg_kwargs)
    return SFNOPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma, config=cfg,
        key=jax.random.PRNGKey(0))


# ------------------------------------------------------------ filter config ---

def test_filter_is_off_by_default():
    """Legacy default preserved: a caller that says nothing gets no damping."""
    cfg = SFNOPrimitiveEquationConfig()
    assert cfg.spectral_filter_strength == 0.0
    assert cfg.clip_q is False


def test_disabled_filter_builds_no_array(grid_t8, sigma_coord, sfno_config):
    m = _model(grid_t8, sigma_coord, sfno_config)
    assert m._spectral_filter is None


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.5])
def test_out_of_range_strength_raises(grid_t8, sigma_coord, sfno_config, bad):
    """It is a filter VALUE at n_max, not a coefficient: [0, 1) or raise."""
    with pytest.raises(ValueError, match="spectral_filter_strength"):
        _model(grid_t8, sigma_coord, sfno_config,
               spectral_filter_strength=bad)


def test_filter_damps_small_scales_and_leaves_the_mean_alone(
        grid_t8, sigma_coord, sfno_config):
    """n=0 must pass through at 1.0, n=n_max must be damped to the strength.

    The n=0 coefficient carries the global mean, which the dry-air-mass
    corrector sets — a filter that touched it would undo the correction.
    """
    strength = 0.01
    m = _model(grid_t8, sigma_coord, sfno_config,
               spectral_filter_strength=strength)
    f = np.asarray(m._spectral_filter)
    ls = np.asarray(grid_t8.ls)
    assert f[ls == 0] == pytest.approx(1.0)
    assert f[ls == grid_t8.n_max] == pytest.approx(strength, rel=1e-6)
    # Monotone non-increasing in total wavenumber.
    order = np.argsort(ls, kind="stable")
    assert np.all(np.diff(f[order]) <= 1e-12)


def test_step_actually_applies_the_filter(grid_t8, sigma_coord, sfno_config):
    """Not just built — the SAME symbol that runs (``step``) must use it.

    Filtered vs unfiltered steps from an identical state and identical
    weights must differ, and the filtered one must carry strictly less
    power at the truncation limit.
    """
    state = _state(grid_t8, sigma_coord)
    plain = _model(grid_t8, sigma_coord, sfno_config)
    damped = _model(grid_t8, sigma_coord, sfno_config,
                    spectral_filter_strength=0.01)
    # Same weights on both (constructed from the same PRNGKey).
    a = plain.step(state, _DT_SFNO)
    b = damped.step(state, _DT_SFNO)
    ls = np.asarray(grid_t8.ls)
    tail = ls >= grid_t8.n_max - 1
    pow_a = float(np.sum(np.abs(np.asarray(a.vor_hat.data)[tail]) ** 2))
    pow_b = float(np.sum(np.abs(np.asarray(b.vor_hat.data)[tail]) ** 2))
    assert pow_b < pow_a
    assert not np.allclose(np.asarray(a.T_hat.data),
                           np.asarray(b.T_hat.data))


def test_filter_must_run_before_the_mass_correction(grid_t8, sigma_coord,
                                                    sfno_config):
    """Ordering is load-bearing: filter THEN correct, never correct THEN filter.

    The filter leaves the n=0 coefficient at 1.0, so it preserves the global
    mean of ln(p_s) — but the conserved quantity is the global mean of
    p_s = exp(lnps), and by Jensen those differ. Filtering after the correction
    therefore re-breaks the dry-air mass it just fixed.

    Deterministic: no network. A smooth p_s perturbation is imposed directly so
    p_s stays positive and ``_apply_conservation``'s ``max(p_s, 1.0)`` floor
    (which is itself lossy once p_s goes negative) never fires — isolating the
    ordering effect from that clamp.
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        apply_spectral_filter_to_state,
    )
    from legoesm.grids.gaussian import sh_analysis, sh_synthesis

    state = _state(grid_t8, sigma_coord)
    w = np.asarray(grid_t8.weights)[:, None]

    def _mean_ps(s):
        ps = np.asarray(jnp.exp(sh_synthesis(grid_t8, s.lnps_hat.data)))
        assert ps.min() > 1.0, "fixture drove p_s below the floor"
        return float(np.sum(ps * w) / (np.sum(w) * grid_t8.n_lon))

    # A drifted, small-scale-rich p_s: mean offset + a wave-2 pattern the
    # filter will bite into.
    lam = np.asarray(grid_t8.lon)[None, :]
    phi = np.asarray(grid_t8.lat)[:, None]
    ps_pred = 1.0e5 * (1.02 + 0.03 * np.cos(2 * lam) * np.cos(phi) ** 4)
    drifted = state._replace(
        lnps_hat=state.lnps_hat.replace(
            data=sh_analysis(grid_t8, jnp.log(jnp.asarray(ps_pred)))))

    cfg = SFNOPrimitiveEquationConfig(
        sfno_config=sfno_config, mode="state_update", dt_sfno=_DT_SFNO,
        correct_mass=True, spectral_filter_strength=0.01)
    m = SFNOPrimitiveEquationModel(grid=grid_t8, sigma_coord=sigma_coord,
                                   config=cfg, key=jax.random.PRNGKey(0))
    target = _mean_ps(state)

    # Shipped order: filter, then correct. filter_lnps=True matches the SFNO
    # step, the one path that still filters lnps (the ordering contract under
    # test only exists when the filter touches lnps at all).
    good = m._apply_conservation(
        apply_spectral_filter_to_state(
            drifted, m._spectral_filter, filter_lnps=True), state)
    # Rejected order: correct, then filter.
    bad = apply_spectral_filter_to_state(
        m._apply_conservation(drifted, state), m._spectral_filter,
        filter_lnps=True)

    assert _mean_ps(good) == pytest.approx(target, rel=1e-10)
    assert _mean_ps(bad) != pytest.approx(target, rel=1e-10)


def test_step_conserves_mass_when_the_ps_floor_does_not_fire(
        grid_t8, sigma_coord, sfno_config):
    """End-to-end on ``step``: with correct_mass + filter both on, global-mean
    p_s is unchanged — provided the run has not already gone unphysical.

    The ``max(p_s, 1.0)`` floor inside ``_apply_conservation`` is lossy once the
    prediction goes negative, which a randomly-initialised net does every time;
    that regime is what the filter exists to avoid, and is not asserted here.
    """
    from legoesm.grids.gaussian import sh_synthesis

    state = _state(grid_t8, sigma_coord)
    cfg = SFNOPrimitiveEquationConfig(
        sfno_config=sfno_config, mode="state_update", dt_sfno=_DT_SFNO,
        correct_mass=True, clip_q=True, spectral_filter_strength=0.01)
    m = SFNOPrimitiveEquationModel(grid=grid_t8, sigma_coord=sigma_coord,
                                   config=cfg, key=jax.random.PRNGKey(0))
    out = m.step(state, _DT_SFNO)
    ps_out = np.asarray(jnp.exp(sh_synthesis(grid_t8, out.lnps_hat.data)))
    if ps_out.min() <= 1.0:
        pytest.skip("random net drove p_s under the floor; see the "
                    "ordering test for the deterministic assertion")
    w = np.asarray(grid_t8.weights)[:, None]
    ps_in = np.asarray(jnp.exp(sh_synthesis(grid_t8, state.lnps_hat.data)))
    denom = np.sum(w) * grid_t8.n_lon
    assert (float(np.sum(ps_out * w) / denom)
            == pytest.approx(float(np.sum(ps_in * w) / denom), rel=1e-10))


# --------------------------------------------------------------- clip_q ------

def test_clip_q_removes_negative_moisture(grid_t8, sigma_coord, sfno_config):
    """A randomly-initialised net readily emits q<0; clip_q must remove it."""
    state = _state(grid_t8, sigma_coord)
    off = _model(grid_t8, sigma_coord, sfno_config, clip_q=False)
    on = _model(grid_t8, sigma_coord, sfno_config, clip_q=True)
    q_off = np.asarray(off.step(state, _DT_SFNO).tracers["q_v"].data)
    q_on = np.asarray(on.step(state, _DT_SFNO).tracers["q_v"].data)
    assert q_off.min() < 0.0, (
        "fixture no longer produces negative q — the clamp test would be "
        "vacuous; perturb the network or the input state")
    assert q_on.min() >= 0.0
    # Only the negatives moved.
    np.testing.assert_allclose(q_on, np.maximum(q_off, 0.0))


def test_clip_q_is_a_noop_without_tracers(grid_t8, sigma_coord, sfno_config):
    """A dry state must not crash the clamp path."""
    dry = isothermal_rest_state_spectral(grid_t8, sigma_coord)
    m = _model(grid_t8, sigma_coord, sfno_config, clip_q=True)
    out = m.step(dry, _DT_SFNO)
    assert np.all(np.isfinite(np.asarray(out.T_hat.data)))


# ------------------------------------------------------- differentiability ---

def test_damped_step_stays_differentiable(grid_t8, sigma_coord, sfno_config):
    """Both knobs are pointwise ops; grads must still reach the SFNO weights."""
    import equinox as eqx

    from legoesm.ml.sfno import SFNO

    state = _state(grid_t8, sigma_coord)
    net = SFNO(sfno_config, grid_t8, key=jax.random.PRNGKey(0))
    cfg = SFNOPrimitiveEquationConfig(
        sfno_config=sfno_config, mode="state_update", dt_sfno=_DT_SFNO,
        correct_mass=True, clip_q=True, spectral_filter_strength=0.01)

    def loss(m):
        model = SFNOPrimitiveEquationModel(
            grid=grid_t8, sigma_coord=sigma_coord, config=cfg, sfno_model=m)
        out = model.step(state, _DT_SFNO)
        return jnp.sum(jnp.abs(out.T_hat.data) ** 2).real

    g = eqx.filter_grad(loss)(net)
    leaves = [x for x in jax.tree.leaves(eqx.filter(g, eqx.is_array))]
    assert leaves, "no differentiable leaves"
    assert any(float(jnp.sum(jnp.abs(x))) > 0.0 for x in leaves)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)

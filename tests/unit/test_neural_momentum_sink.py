"""#1464: the learned column arm must be able to carry a momentum sink.

`make_column_physics_fn` returned ZERO vor/div tendencies, i.e. a model with no
surface drag of any kind, while the classical arm it was scored against
inherits `TurbulenceConfig.scheme = "smagorinsky"` whenever a `PhysicsConfig`
is built naming only radiation and convection. The two arms therefore differed
by the presence of a momentum sink, and the T106 campaign's degradation ranked
exactly as that predicts (wind_speed_10m 3.83x, u10 3.02x at 240 h, monotone in
height, z500 the only field that improved).

These tests pin: the confound is real and reachable in code; the new
`momentum_physics_fn` hook actually changes vor/div; the default stays
bit-identical; and the network still owns the thermodynamics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.neural_physics import (  # noqa: E402
    build_column_physics,
    make_column_physics_fn,
)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.training.neural_gcm_spectral import (  # noqa: E402
    carry_to_spectral_state,
    make_turbulence_only_spectral_physics,
)
from tests.unit.test_learned_column import _mini_spectral_state  # noqa: E402

NLEV = 4


def _setup(seed=0):
    grid = create_gaussian_grid(n_max=10)
    model = build_column_physics(nlev=NLEV, hidden_dim=8, n_layers=2,
                                 residual_scale=0.1,
                                 key=jax.random.PRNGKey(seed))
    carry, sigma = _mini_spectral_state(grid, NLEV)
    return grid, model, carry_to_spectral_state(carry, grid), sigma


def test_the_confound_is_real_default_arm_has_no_momentum_tendency():
    """Documents the defect: with no momentum source, vor/div are EXACTLY 0."""
    grid, model, state, sigma = _setup()
    out = make_column_physics_fn(model, grid)(state, grid, sigma)
    assert float(jnp.max(jnp.abs(out.vor_hat.data))) == 0.0
    assert float(jnp.max(jnp.abs(out.div_hat.data))) == 0.0


def test_the_classical_arm_inherits_a_momentum_sink():
    """The other half of the confound: naming only rad+conv keeps smagorinsky."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    p = PhysicsConfig()
    assert p.turbulence.scheme == "smagorinsky"
    assert PhysicsConfig(radiation=p.radiation,
                         convection=p.convection).turbulence.scheme == \
        "smagorinsky"


def test_momentum_physics_fn_actually_reaches_vor_div():
    grid, model, state, sigma = _setup()
    marker_vor = 3.5e-6
    marker_div = -1.25e-6

    def _mom(st, g, sc):
        return st._replace(
            vor_hat=st.vor_hat.replace(
                data=jnp.full_like(st.vor_hat.data, marker_vor)),
            div_hat=st.div_hat.replace(
                data=jnp.full_like(st.div_hat.data, marker_div)),
            # A source that ALSO emits lnps: must be ignored (see below).
            lnps_hat=st.lnps_hat.replace(
                data=jnp.full_like(st.lnps_hat.data, 7.0e-9)),
        )

    out = make_column_physics_fn(model, grid, momentum_physics_fn=_mom)(
        state, grid, sigma)
    np.testing.assert_allclose(np.asarray(out.vor_hat.data), marker_vor)
    np.testing.assert_allclose(np.asarray(out.div_hat.data), marker_div)
    assert float(np.max(np.abs(np.asarray(out.lnps_hat.data)))) == 0.0, (
        "lnps was taken from the momentum source. Surface stress has no "
        "surface-pressure tendency, and the PE ADDS this to its own "
        "continuity-derived dlnps/dt, so a mean value here breaks dry-mass "
        "conservation with anchoring off (codex, #1464)")


def test_thermodynamics_still_come_from_the_network():
    """The hook must not let the momentum source touch T or the tracers."""
    grid, model, state, sigma = _setup(seed=3)

    def _mom(st, g, sc):
        return st._replace(
            vor_hat=st.vor_hat.replace(
                data=jnp.full_like(st.vor_hat.data, 1e-5)),
            div_hat=st.div_hat.replace(
                data=jnp.full_like(st.div_hat.data, 1e-5)),
            T_hat=st.T_hat.replace(
                data=jnp.full_like(st.T_hat.data, 999.0)),   # must be IGNORED
        )

    base = make_column_physics_fn(model, grid)(state, grid, sigma)
    with_mom = make_column_physics_fn(model, grid, momentum_physics_fn=_mom)(
        state, grid, sigma)
    np.testing.assert_allclose(np.asarray(with_mom.T_hat.data),
                               np.asarray(base.T_hat.data), rtol=0, atol=0)


def test_default_is_bit_identical_to_the_historical_behaviour():
    grid, model, state, sigma = _setup(seed=5)
    a = make_column_physics_fn(model, grid)(state, grid, sigma)
    b = make_column_physics_fn(model, grid, momentum_physics_fn=None)(
        state, grid, sigma)
    for f in ("vor_hat", "div_hat", "T_hat", "lnps_hat"):
        np.testing.assert_array_equal(np.asarray(getattr(a, f).data),
                                      np.asarray(getattr(b, f).data))


def _sheared(state, seed=0):
    """A state with real vor/div — smagorinsky is a SHEAR closure, so on the
    quiescent mini-state it returns exactly zero and proves nothing."""
    rng = np.random.default_rng(seed)
    return state._replace(
        vor_hat=state.vor_hat.replace(data=jnp.asarray(
            rng.normal(scale=1e-5, size=state.vor_hat.data.shape))),
        div_hat=state.div_hat.replace(data=jnp.asarray(
            rng.normal(scale=1e-6, size=state.div_hat.data.shape))))


def test_turbulence_only_builder_produces_a_real_momentum_tendency():
    grid, model, state, sigma = _setup(seed=7)
    out = make_turbulence_only_spectral_physics(dt=1200.0)(
        _sheared(state), grid, sigma)
    assert float(jnp.max(jnp.abs(out.vor_hat.data))) > 0.0, (
        "no vorticity tendency — the paired builder supplies no sink, so the "
        "learned arm would still have no drag")
    assert float(jnp.max(jnp.abs(out.div_hat.data))) > 0.0


def test_quiescent_state_gives_zero_so_the_shear_test_is_not_vacuous():
    """Guards the test above: on an unsheared state the same call is 0."""
    grid, model, state, sigma = _setup(seed=7)
    out = make_turbulence_only_spectral_physics(dt=1200.0)(state, grid, sigma)
    assert float(jnp.max(jnp.abs(out.vor_hat.data))) == 0.0


def test_the_sources_own_heat_diffusion_is_discarded_by_the_adapter():
    """smagorinsky ALSO diffuses temperature (measured 1.64e-5 on a sheared
    state). The adapter must take only vor/div/lnps, so the network keeps sole
    ownership of the thermodynamics."""
    grid, model, state, sigma = _setup(seed=9)
    sheared = _sheared(state, seed=1)
    src = make_turbulence_only_spectral_physics(dt=1200.0)
    src_out = src(sheared, grid, sigma)
    assert float(jnp.max(jnp.abs(src_out.T_hat.data))) > 0.0, (
        "the source has no heat tendency here — this test cannot show it is "
        "discarded")
    base = make_column_physics_fn(model, grid)(sheared, grid, sigma)
    with_mom = make_column_physics_fn(
        model, grid, momentum_physics_fn=src)(sheared, grid, sigma)
    np.testing.assert_array_equal(np.asarray(with_mom.T_hat.data),
                                  np.asarray(base.T_hat.data))
    assert float(jnp.max(jnp.abs(with_mom.vor_hat.data))) > 0.0


def test_turbulence_only_builder_rejects_an_unknown_scheme():
    with pytest.raises(Exception):
        make_turbulence_only_spectral_physics(dt=1200.0,
                                              turbulence_scheme="not_a_scheme")


def test_stateful_turbulence_schemes_are_refused():
    """The builder threads no PhysicsState, so a prognostic scheme would
    re-seed its carry from the floor every step — refuse, don't run it wrong."""
    for scheme in ("tke", "mynn25", "clubb"):
        with pytest.raises(ValueError, match="PROGNOSTIC"):
            make_turbulence_only_spectral_physics(dt=1200.0,
                                                  turbulence_scheme=scheme)


def test_campaign_builder_wires_the_drag_when_asked():
    """END-TO-END on the route the T106 campaign actually takes.

    The first version of this fix was DEAD CODE: scale_build built the neural
    arm without the hook, so nothing changed for the benchmark (codex
    Critical). This test fails if that regresses.
    """
    from legoesm.training import scale_build as sb
    import inspect
    src = inspect.getsource(sb.build_scale_case
                            if hasattr(sb, "build_scale_case") else sb)
    assert "momentum_physics_fn=" in src, (
        "scale_build no longer passes momentum_physics_fn — the neural arm "
        "is back to running with no surface drag (#1464)")
    assert "surface_drag" in src, (
        "the opt-in flag is gone; the campaign cannot request the drag")


def _schemes_the_drag_builder_refuses():
    """Derived by asking the builder, not copied from it.

    A hand-kept copy of its refusal list goes stale the moment a prognostic
    scheme is added, and the campaign then dies at its first step instead of
    in review.
    """
    from legoesm.training.neural_gcm_spectral import (
        make_turbulence_only_spectral_physics,
    )
    refused = set()
    for name in ("smagorinsky", "louis", "holtslag_boville", "ysu",
                 "tke", "mynn25", "clubb", "clubb_lite", "edmf"):
        try:
            make_turbulence_only_spectral_physics(600.0, name)
        except ValueError:
            refused.add(name)
    assert refused, "the builder refuses nothing — the guard below is vacuous"
    return refused


def test_the_scheme_name_is_not_cosmetic():
    """Before pinning WHICH scheme, prove the choice changes the answer.

    If louis and smagorinsky gave the same momentum tendency, matching them
    would be bookkeeping and the config guard below would pin nothing.
    """
    grid, _model, state, sigma = _setup(seed=11)
    sheared = _sheared(state, seed=2)
    out_l = make_turbulence_only_spectral_physics(1200.0, "louis")(
        sheared, grid, sigma)
    out_s = make_turbulence_only_spectral_physics(1200.0, "smagorinsky")(
        sheared, grid, sigma)
    d_l = np.asarray(out_l.vor_hat.data)
    d_s = np.asarray(out_s.vor_hat.data)
    peak = float(np.max(np.abs(d_l)))
    assert peak > 0.0, "louis produced no vorticity tendency"
    # Scale-aware on purpose: these tendencies are O(1e-10), so the default
    # absolute tolerance of allclose would call any two of them identical.
    gap = float(np.max(np.abs(d_l - d_s)))
    assert gap > 0.1 * peak, (
        f"louis and smagorinsky differ by only {gap:.3g} against a louis peak "
        f"of {peak:.3g}, so matching the scheme names would pin nothing")


def test_the_campaign_builder_reads_the_scheme_and_not_just_the_flag():
    """The seam that was dead once already.

    `scale_build` must pass the campaign's chosen scheme through to the
    builder. If it reads only the on/off flag and keeps its own default, the
    learned arm silently gets a different scheme from the classical arm and
    every config guard below stays green while the confound survives.
    """
    import inspect

    from legoesm.training import scale_build as sb
    src = inspect.getsource(sb)
    assert "surface_drag_scheme" in src, (
        "scale_build no longer reads surface_drag_scheme — the campaign's "
        "choice of momentum scheme cannot reach the learned arm")
    assert "momentum_physics_fn=" in src


def test_the_learned_arm_gets_the_SAME_momentum_scheme_as_the_classical_arm():
    """Every campaign that scores the two arms must equalise their momentum.

    Three rules, and all three are needed:

    1. A campaign whose classical arm runs a scheme the drag builder accepts
       MUST enable the drag. This is the original defect: the learned column
       has no momentum head, so with the drag off it runs with no physical
       surface stress at all while the classical arm has one.
    2. It must NAME the scheme, and that name must equal the classical arm's.
       Handing the learned arm a different scheme is the same confound wearing
       a new label; leaving it unnamed silently takes the builder's default.
    3. A campaign whose classical arm runs a scheme the builder REFUSES — the
       prognostic family, whose carry this wrapper cannot thread — cannot be
       equalised at all. It must leave the drag off AND say so with
       ``surface_drag_confounded: true``, so the confound is enumerable rather
       than a comment somebody has to notice.
    """
    import pathlib
    import yaml

    refused = _schemes_the_drag_builder_refuses()
    root = pathlib.Path(__file__).resolve().parents[2]
    problems, checked = [], 0
    for path in sorted((root / "config").rglob("*.y*ml")):
        try:
            doc = yaml.safe_load(path.read_text(errors="replace"))
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        neural, classical = doc.get("neural_gcm"), doc.get("classical")
        if not isinstance(neural, dict) or not isinstance(classical, dict):
            continue
        checked += 1
        rel = str(path.relative_to(root))
        classical_scheme = classical.get("turbulence")
        drag_on = neural.get("surface_drag") is True
        drag_scheme = neural.get("surface_drag_scheme")

        # An explicit, machine-readable declaration that this campaign's two
        # arms cannot be equalised. Allowed, but it must be DECLARED — the
        # point is that every confounded campaign can be listed by grepping
        # one key, instead of hiding behind a comment.
        if neural.get("surface_drag_confounded") is True:
            if drag_on:
                problems.append(
                    f"{rel}: declares itself confounded and also enables the "
                    "drag; pick one")
            continue

        if classical_scheme in refused:
            if drag_on:
                problems.append(
                    f"{rel}: classical runs {classical_scheme!r}, which the "
                    "drag builder cannot reproduce, so no choice of "
                    f"surface_drag_scheme equalises the arms (named "
                    f"{drag_scheme!r})")
            elif neural.get("surface_drag_confounded") is not True:
                problems.append(
                    f"{rel}: classical runs {classical_scheme!r} and the drag "
                    "is off, so the learned arm has no surface stress; mark "
                    "it `surface_drag_confounded: true` or stop scoring the "
                    "two arms against each other")
            continue

        if not drag_on:
            problems.append(
                f"{rel}: classical runs {classical_scheme!r} but the learned "
                "arm has no surface stress at all — the original confound")
        elif drag_scheme is None:
            problems.append(
                f"{rel}: enables the drag without naming a scheme, so it "
                f"takes the builder's default instead of {classical_scheme!r}")
        elif drag_scheme != classical_scheme:
            problems.append(
                f"{rel}: learned arm gets {drag_scheme!r}, classical arm runs "
                f"{classical_scheme!r}")

    assert checked, "no config pairs a neural_gcm block with a classical block"
    assert not problems, (
        "these campaigns would score two arms whose momentum differs:\n  "
        + "\n  ".join(problems))

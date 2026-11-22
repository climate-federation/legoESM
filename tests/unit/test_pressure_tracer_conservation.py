"""T24a native-route budgets, sigma equivalence and fp32 derivatives.

Reuses the preregistered live-callback census rather than a second instrument.
Only the vertical instantaneous budget is claimed by these tests.
"""
import functools
import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids import vertical as v

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "_t24a_census", ROOT / "scripts/validate/diag_hybrid_tracer_smoothing.py")
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)

PATHS = [f"{lane}/{coord}/{scheme}"
         for lane in ("MPAS", "latlon", "spectral PE")
         for coord in ("sigma", "hybrid") for scheme in ("sb", "van_leer")]
PATHS += [f"MPAS/sigma/thermal_van_leer/{scheme}" for scheme in ("sb", "van_leer")]
PATHS += [f"standalone {lane}/sigma/{scheme}"
          for lane in ("MPAS", "latlon", "spectral", "cube")
          for scheme in ("sb", "van_leer")]


@pytest.fixture(scope="module")
def routes(tmp_path_factory):
    assert os.environ.get("PBS_JOBID"), "T24 measurements require PBS"
    previous = get_policy()
    try:
        return census.all_vertical_paths(tmp_path_factory.mktemp("t24a"), pressure_only=True)
    finally:
        set_policy(previous)


@pytest.mark.parametrize("path", PATHS)
def test_native_route_paired_budget(routes, path):
    rows = [r for r in routes if r["path"] == path]
    assert rows, f"unobserved native route {path}"
    assert all(r["max_relative"] <= 1e-12 for r in rows), rows
    scheme = path.rsplit("/", 1)[1]
    assert all(r["scheme"] in (scheme, "vertical_advection_hybrid_" + scheme) for r in rows)


def test_census_controls_can_fail(routes):
    controls = {r["scheme"]: r["verdict"] for r in routes if r["path"] == "control"}
    assert controls == {"vertical_advection_hybrid_sb": "CONSERVES",
                        "vertical_advection_hybrid_van_leer": "CONSERVES",
                        "vertical_advection_hybrid": "LEAKS", "planted": "LEAKS"}


@pytest.mark.parametrize("scheme", ["sb", "van_leer"])
@pytest.mark.parametrize("width,flux_width", [(1e-10, 1e-6), (0., 0.), (0., 1e-6), (1e-10, 0.)])
def test_sigma_is_zero_a_hybrid(scheme, width, flux_width):
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        sigma = v.create_sigma_coordinate(32)
        hybrid = v.hybrid_from_sigma(sigma)
        rng = np.random.default_rng(0)
        ps = jnp.asarray(1e5 + 3e3 * rng.standard_normal(6))
        q = jnp.asarray(1e-2 * rng.random((6, 32)))
        sd = jnp.asarray(1e-5 * rng.standard_normal((6, 33))).at[:, (0, -1)].set(0.)
        kw = dict(scheme=scheme, smoothing_width=width, flux_smoothing_width=flux_width)
        sigma_result = jax.jit(lambda x: v.vertical_advection_tracer(x, sd, sigma, ps, **kw))(q)
        if scheme == "sb":
            hybrid_result = v.vertical_advection_hybrid_sb(q, sd * ps[:, None], ps, hybrid)
        else:
            hybrid_result = v.vertical_advection_hybrid_van_leer(
                q, sd * ps[:, None], ps, hybrid, smoothing_width=width,
                flux_smoothing_width=flux_width)
        scale = np.max(np.abs(hybrid_result))
        assert scale > 0
        assert np.max(np.abs(sigma_result - hybrid_result)) / scale <= 1e-12
    finally:
        set_policy(previous)


def test_unknown_dispatch_raises():
    with pytest.raises(ValueError, match="unknown tracer vertical scheme"):
        v.vertical_advection_tracer(None, None, None, scheme="typo")


def test_native_vertical_terms_have_fp32_gradients(tmp_path, monkeypatch):
    """Differentiate each live native vertical term wrt q and its face flux.

    The surrounding census provides the actual caller's state/geometry and
    checks that all 22 routes execute; gradients are of its vertical term,
    not a claim about the complete model's precision or climate loss.
    """
    assert os.environ.get("PBS_JOBID"), "T24 measurements require PBS"
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as mp
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_latlon_cgrid as ll
    from legoesm.atmosphere.dynamics.gcm import spectral_pe as sp
    from legoesm.atmosphere.dynamics.gcm import tracer_transport_mpas as tm
    from legoesm.atmosphere.dynamics.gcm import tracer_transport_latlon as tl
    from legoesm.atmosphere.dynamics.gcm import tracer_transport_spectral as ts
    from legoesm.atmosphere.dynamics.shared import tracer_transport as tc
    checked = []

    def wrap(real, name):
        @functools.wraps(real)
        def call(*a, **kw):
            args = jax.tree.map(lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") else x, a)
            def loss(q, flux):
                out = real(q, flux, *args[2:], **kw)
                weights = jnp.arange(out.shape[-1], dtype=jnp.float32) + 1
                return jnp.sum(weights * out)
            gq, gf = jax.grad(loss, argnums=(0, 1))(args[0], args[1])
            def check(x, y):
                assert x.dtype == y.dtype == np.float32
                assert np.isfinite(x).all() and np.isfinite(y).all()
                assert np.any(x != 0) and np.any(y != 0)
                checked.append(name)
            jax.debug.callback(check, gq, gf, ordered=True)
            return real(*a, **kw)
        return call

    for module in (mp, ll, sp, tm, tl, ts, tc):
        for name in ("vertical_advection_tracer", "vertical_advection_hybrid_sb", "vertical_advection_hybrid_van_leer"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, wrap(getattr(module, name), module.__name__ + name))
    rows = census.all_vertical_paths(tmp_path, pressure_only=True)
    jax.effects_barrier()
    assert set(PATHS) <= {r["path"] for r in rows}
    assert len(checked) >= len(PATHS)

@pytest.mark.parametrize("module_name,fn_name,config_name", [
    ("legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral", "tracer_tendencies_spectral", "TracerTransportSpectralConfig"),
    ("legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon", "tracer_tendencies_latlon", "TracerTransportLatLonConfig"),
    ("legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas", "tracer_advection_tendency", "TracerTransportMPASConfig"),
    ("legoesm.atmosphere.dynamics.shared.tracer_transport", "tracer_tendencies", "TracerTransportConfig"),
])
def test_standalone_unknown_scheme_raises_before_empty_or_horizontal_state(module_name, fn_name, config_name):
    import importlib
    module = importlib.import_module(module_name)
    config = getattr(module, config_name)(tracer_vertical_scheme="typo")
    # Fail before touching the state, including spectral winds without sigma_dot.
    args = [None] * (5 if fn_name == "tracer_advection_tendency" else 4)
    with pytest.raises(ValueError, match="unknown tracer vertical scheme"):
        getattr(module, fn_name)(*args, config=config)


@pytest.mark.parametrize("lane", ["mpas", "latlon"])
def test_production_fixture_step_bitwise(lane):
    base, candidate = os.environ.get("T24_SNAPSHOT_BASE"), os.environ.get("T24_SNAPSHOT_CANDIDATE")
    if not base or not candidate:
        pytest.skip("run with the separately retained ed2c8ead3a/current PBS snapshots")
    import json
    bp = json.loads((Path(base) / "provenance.json").read_text())
    cp = json.loads((Path(candidate) / "provenance.json").read_text())
    assert bp["commit"].startswith("ed2c8ead3a")
    assert bp["complete"] and cp["complete"]
    assert bp["precision"] == cp["precision"] == "fp64"
    b, c = np.load(Path(base) / "step.npz"), np.load(Path(candidate) / "step.npz")
    assert b.files and set(b.files) == set(c.files)
    keys = [key for key in b.files if key.startswith(lane + "_")]
    assert keys
    for key in keys:
        assert b[key].shape == c[key].shape and b[key].dtype == c[key].dtype
        np.testing.assert_array_equal(b[key], c[key], err_msg=key)
        assert b[key].tobytes() == c[key].tobytes(), key
    assert np.any(c[lane + "_initial_q_v"] != c[lane + "_final_q_v"])


@pytest.mark.parametrize("solver", ["fv3_duo_primitive_equations", "spectral_compressible_euler", "mpas_compressible_euler", "plane_compressible_euler"])
@pytest.mark.parametrize("scheme", ["sb", "van_leer"])
def test_unsupported_lane_refuses_explicit_selection(solver, scheme):
    from legoesm.driver.config import resolve_tracer_vertical_scheme
    assert resolve_tracer_vertical_scheme(None, solver) is None
    with pytest.raises(ValueError, match="silently inert"):
        resolve_tracer_vertical_scheme(scheme, solver)

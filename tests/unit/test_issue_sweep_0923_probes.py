"""Direct tests for the two probes added by the 2026-09-23 issue sweep.

Both exist to settle a question on an open issue, and both are the kind of
instrument this repo requires to be tested before its number is quoted:

* the ``LEGOESM_TOPO_SEED`` override in the atmosphere matrix runner (#1029),
  which must reach the initial-condition routine and must be inert when unset;
* ``scripts/validate/cg_helmholtz_mixed_precision_1675.py`` (#1675 site 1),
  whose whole job is to compare a precision fix against its own control.
"""
from __future__ import annotations

import ast
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parents[2]
MATRIX = REPO / "scripts" / "matrix" / "run_atmosphere_test_matrix.py"
PROBE = REPO / "scripts" / "validate" / "cg_helmholtz_mixed_precision_1675.py"


# ----------------------------------------------------------------------
# #1029 -- the ensemble-seed override
# ----------------------------------------------------------------------

def _topo_init_calls() -> list[ast.Call]:
    """Every ``held_suarez_topo_init*`` call in the matrix runner."""
    tree = ast.parse(MATRIX.read_text())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id.startswith("held_suarez_topo_init")):
            out.append(node)
    return out


def test_every_topo_init_call_routes_through_the_kwargs_helper():
    """All four grid lanes must take the seed, not only the one an arm used.

    A per-lane override is how the height knob would have silently skipped a
    grid; the helper is the single point that cannot.
    """
    calls = _topo_init_calls()
    assert len(calls) == 4, f"expected 4 topo init call sites, found {len(calls)}"
    for c in calls:
        starstar = [k for k in c.keywords if k.arg is None]
        assert len(starstar) == 1, ast.dump(c)
        assert isinstance(starstar[0].value, ast.Call)
        assert starstar[0].value.func.id == "_topo_init_kwargs"
        # and the height must NOT also be passed positionally/by name, which
        # would silently win over the helper's entry
        assert not any(k.arg == "h_0" for k in c.keywords)


def test_seed_override_is_inert_when_unset(monkeypatch):
    """Unset -> the init routine's own default, i.e. no committed run moves."""
    monkeypatch.delenv("LEGOESM_TOPO_SEED", raising=False)
    src = MATRIX.read_text()
    assert '_topo_seed = None' in src
    assert 'if _topo and "LEGOESM_TOPO_SEED" in os.environ:' in src
    # the helper adds the key only when the override was seen
    helper = src.split("def _topo_init_kwargs():", 1)[1].split("\n\n", 1)[0]
    assert 'if _topo_seed is not None:' in helper
    assert '"seed"' in helper


def test_seed_actually_changes_the_initial_state():
    """Non-vacuity: a different seed must give a different initial condition.

    If the seed did not reach the perturbation the ensemble would measure zero
    spread and the reading rule would be satisfied for the wrong reason -- the
    exact failure mode of a control that perturbs nothing.
    """
    from legoesm.atmosphere.idealized.held_suarez_topo import (
        held_suarez_topo_init_latlon)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(16, 32)
    sigma = create_sigma_coordinate(8)
    a = held_suarez_topo_init_latlon(grid, sigma, h_0=2000.0, seed=42)
    b = held_suarez_topo_init_latlon(grid, sigma, h_0=2000.0, seed=1234)
    da = np.asarray(a.T.data if hasattr(a.T, "data") else a.T)
    db = np.asarray(b.T.data if hasattr(b.T, "data") else b.T)
    assert da.shape == db.shape
    assert not np.array_equal(da, db), "seed does not reach the perturbation"
    # and the perturbation stays small -- it is an IC seed, not a new case
    assert np.max(np.abs(da - db)) < 10.0


# ----------------------------------------------------------------------
# #1029 -- the all-NaN argmax trap in the blow-up localisation
# ----------------------------------------------------------------------

def _load_matrix_module():
    """Import the matrix runner as a module so its helpers can be called.

    It has to be registered in ``sys.modules`` BEFORE ``exec_module``: the
    runner defines NamedTuples at module scope, and typing resolves their
    annotations through ``sys.modules[cls.__module__]``, which is ``None``
    for a module that is being executed but has not been registered.
    """
    import importlib.util
    name = "_matrix_runner_1029"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, MATRIX)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def test_argmax_location_refuses_to_localise_a_wholly_nonfinite_field():
    """The bug this guard exists for, stated as a test.

    Measured: ``jnp.argmax`` returns index 0 on a wholly non-finite field.  A
    blow-up diagnostic samples the state after it has gone non-finite, so the
    unguarded version localised twelve held_suarez_topo arms -- every mountain
    height, both failure timescales -- to that one coordinate, which on a
    lat-lon grid unravels to the southernmost row at the model top and reads
    exactly like one of the physical suspects.
    """
    mod = _load_matrix_module()
    f = jnp.full((4, 5, 6), jnp.nan)
    assert mod._argmax_location(f, f.shape) is None, (
        "a wholly non-finite field must not be localised; the unguarded "
        "argmax returns (0, 0, 0), a plausible-looking corner of the domain")


def test_argmax_location_still_finds_a_real_maximum():
    """Non-vacuity: the guard must not blind the diagnostic on healthy fields."""
    mod = _load_matrix_module()
    f = np.zeros((3, 4, 5))
    f[2, 1, 3] = 7.0
    assert mod._argmax_location(jnp.asarray(f), f.shape) == (2, 1, 3)


def test_a_partly_nan_field_is_refused_because_its_argmax_is_meaningless():
    """The conservative branch, kept because a measurement said to keep it.

    A review argued the opposite -- that NaN never wins a comparison reduction,
    so a mixed field localises to its largest finite element and refusing it
    throws away the informative case.  Measured on jax 0.9.1, a NaN DOES win,
    exactly as in NumPy: a field with one NaN away from index 0 and a clear
    finite maximum elsewhere returns the NaN's index.  So the argmax of a
    partially non-finite field points at the corruption, not the physics, and
    is not a localisation.  See the argmax_nan_semantics probe for the table.
    """
    mod = _load_matrix_module()
    f = np.zeros((3, 4))
    f[1, 1] = 5.0
    f[2, 3] = np.nan          # neither index 0 nor the finite maximum
    raw = tuple(int(x) for x in
                jnp.unravel_index(jnp.argmax(jnp.asarray(f)), f.shape))
    assert raw == (2, 3), (
        f"raw argmax returned {raw}; measured behaviour on jax 0.9.1 is that "
        f"the NaN at (2, 3) wins. If this is now (1, 1) the backend has "
        f"changed and the guard can be narrowed to all-non-finite")
    assert mod._argmax_location(jnp.asarray(f), f.shape) is None


def test_first_nonfinite_location_finds_the_earliest_bad_cell():
    """The quantity a postmortem actually wants, and it is not an argmax.

    An Inf usually appears before the NaNs in a divergence, so the first
    non-finite cell is the earliest spatial signal a tripped state still
    carries.
    """
    mod = _load_matrix_module()
    f = np.zeros((3, 4))
    f[2, 1] = np.inf
    f[2, 3] = np.nan
    assert mod._first_nonfinite_location(jnp.asarray(f), f.shape) == (2, 1)


def test_first_nonfinite_location_is_none_on_a_clean_field():
    """Non-vacuity: it must stay silent when there is nothing to report."""
    mod = _load_matrix_module()
    f = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert mod._first_nonfinite_location(jnp.asarray(f), f.shape) is None


def test_blowup_check_interval_override_is_parsed_and_validated():
    """The override must reject a non-positive stride rather than hang.

    ``step % 0`` raises ZeroDivisionError from inside the time loop, thousands
    of steps after the typo; ``step % -1`` is always 0 and would trip the
    blow-up check on the very first step. Both are caught at parse time.
    """
    src = MATRIX.read_text()
    assert 'os.environ.get("LEGOESM_BLOWUP_CHECK_INTERVAL")' in src
    assert "must be a positive integer" in src
    # the guard has to run BEFORE the loop that consumes the value
    i_guard = src.index('LEGOESM_BLOWUP_CHECK_INTERVAL')
    i_use = src.index("if step % blowup_check_interval == 0:")
    assert i_guard < i_use, (
        "the override is parsed after the loop that reads it")


# ----------------------------------------------------------------------
# #1675 site 1 -- the CG mixed-precision probe
# ----------------------------------------------------------------------

def test_cg_probe_reports_both_arms_and_they_agree(capsys):
    """The probe's own finding, at a size cheap enough for CI.

    Promoting the Laplacian metric arrays is a no-op because the CG iterate is
    already float64 and JAX promotes f64 x f32 -> f64, so the two arms must
    agree closely.  If a future change makes them diverge, that is a real
    signal and this test should be re-read, not deleted.
    """
    sys.path.insert(0, str(PROBE.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_cg_probe_1675", PROBE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.main(["--n", "8", "--kappa", "1e4", "--maxiter", "200"])
    finally:
        sys.path.pop(0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "metrics PROMOTED (the fix)" in out
    assert "metrics f32 (pre-fix control)" in out

    def _grab(label, key):
        block = out.split(label, 1)[1]
        line = [ln for ln in block.splitlines() if key in ln][0]
        return float(line.split("=")[1].split()[0])

    fix = _grab("metrics PROMOTED (the fix)", "reported rel_res")
    ctl = _grab("metrics f32 (pre-fix control)", "reported rel_res")
    assert fix > 0.0 and ctl > 0.0
    # same number to within a few percent: the promotion changes nothing
    assert abs(fix - ctl) / ctl < 0.05, (
        f"promoted {fix:.3e} vs control {ctl:.3e} -- the arms have diverged; "
        f"re-read #1675 site 1 before trusting either")
    # and the answer is accurate even though the residual is not tiny
    fwd = _grab("metrics PROMOTED (the fix)", "forward error vs fp64")
    assert fwd < 1.0e-6, f"forward error {fwd:.3e} is not float32-limited"

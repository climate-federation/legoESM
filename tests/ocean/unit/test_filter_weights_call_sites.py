"""Every caller of ``compute_filter_weights`` must unpack all four values.

#1609 widened the barotropic averaging window to 2n-1 substeps so its first
moment lands on t+dt, and added the loop length ``n_loop`` as a fourth return
value. Two call sites were updated; a THIRD -- the lat-lon C-grid solver's
non-NEMO branch -- was missed and shipped to main unpacking three, which

  * raised ``ValueError: too many values to unpack`` the moment that branch
    ran (it took the ocean benchmark suite down on the lat-lon rest-state
    cases), and
  * had ``n_loop = n_substeps`` hardcoded underneath, so "fixing" the unpack
    by dropping the fourth value would silently restore the HALF window #1609
    removed -- the one that ran gravity waves at about half speed.

Two adversarial reviews of #1609 both checked "the production callers" and
both found only two. This test is the mechanical version of that check, so
the next change to the signature cannot rely on anyone enumerating callers
correctly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_ROOTS = (_REPO / "packages", _REPO / "scripts", _REPO / "src")
_FN = "compute_filter_weights"


def _call_sites():
    """(path, lineno, n_targets) for every ``... = compute_filter_weights(...)``."""
    out = []
    for root in _ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                call = node.value
                if not isinstance(call, ast.Call):
                    continue
                fn = call.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else None)
                if name != _FN:
                    continue
                tgt = node.targets[0]
                n = len(tgt.elts) if isinstance(tgt, (ast.Tuple, ast.List)) else 1
                out.append((path.relative_to(_REPO), node.lineno, n))
    return out


def test_every_call_site_unpacks_four():
    """The check that would have caught the shipped defect."""
    sites = _call_sites()
    assert sites, "found no call sites -- this test has stopped checking anything"
    bad = [(p, ln, n) for p, ln, n in sites if n != 4]
    assert not bad, (
        "these unpack the wrong number of values from compute_filter_weights "
        f"(it returns 4: w_filter, w_total, w_transport, n_loop): {bad}")


def test_the_function_really_returns_four():
    """Non-vacuity: pin the arity the test above is asserting against.

    If the signature ever changes deliberately, this goes red first and says
    so, rather than the caller test failing mysteriously."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_common import compute_filter_weights
    got = compute_filter_weights(8, jnp.float64, use_cosine=True)
    assert len(got) == 4


def test_no_caller_hardcodes_the_loop_length():
    """``n_loop = n_substeps`` right after a call is the half window #1609 removed.

    The missed call site had exactly that line underneath it, so a careless fix
    to the unpack -- drop the fourth value, keep the old assignment -- would
    have turned the error green while restoring the defect.

    Scoped to the few lines FOLLOWING a call, not the whole file: the same
    module has a legitimate ``n_loop = n_substeps`` in its MOM6/AM4 branch,
    which builds its own weights of length n_substeps and never calls this
    function. Flagging that one would be a false positive, and a guard that
    cries wolf gets deleted."""
    for path, lineno, _n in _call_sites():
        lines = (_REPO / path).read_text().splitlines()
        window = lines[lineno - 1:lineno + 6]
        offenders = [w.strip() for w in window
                     if w.strip() == "n_loop = n_substeps"]
        assert not offenders, (
            f"{path}:{lineno} hardcodes n_loop = n_substeps just after calling "
            f"compute_filter_weights; that is the HALF window. Take the value "
            f"the function returns.")


def test_the_legitimate_hardcode_is_still_there():
    """Non-vacuity for the scoping above.

    If the MOM6/AM4 branch ever stops setting its own loop length, the test
    above has quietly narrowed to nothing and should be re-examined rather
    than trusted."""
    src = (_REPO / "packages/ocean/legoesm/ocean/dynamics"
           / "barotropic_latlon_cgrid.py").read_text()
    assert "n_loop = n_substeps" in src, (
        "the branch this test was scoped around has gone; re-check that the "
        "scoping in test_no_caller_hardcodes_the_loop_length still makes sense")


@pytest.mark.parametrize("time_filter", ["cosine", "box"])
def test_default_filter_path_steps_the_model(time_filter):
    """End-to-end: the DEFAULT barotropic filter branch can take a real step.

    The gates above are STATIC (they read source).  This one RUNS the stock
    lat-lon C-grid model -- ``barotropic_solver="explicit_substep"`` plus the
    default ``barotropic_time_filter="cosine"``, i.e. the ``else`` branch of
    ``_compute_weights``.  Pre-fix it raises ``ValueError: too many values to
    unpack (expected 3)`` inside the first step.

    NOT a rest state: a rest basin evolves to exactly zero, so "the fields are
    finite" would be satisfied by all-zeros and could only ever catch a hard
    raise.  The initial free surface carries a bump, and the test asserts the
    step actually moved the ocean, so a solver that silently returned its input
    would fail too.

    The ``n_loop`` assertion is the second half of the guard: re-hardcoding
    ``n_loop = n_substeps`` (the careless "fix" for the unpack error) restores
    the half window #1609 removed while leaving the model perfectly able to
    step, so the stepping assertions alone would not catch it.

    ``"box"`` shares the same branch and is parametrized so the coverage
    follows the branch rather than one config value.
    """
    import numpy as np
    import jax.numpy as jnp

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _compute_weights
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    n_lat, n_lon = 16, 32
    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    ocean_mask = np.ones((n_lat, n_lon))
    ocean_mask[:2] = 0.0          # 1 = ocean, 0 = land
    ocean_mask[-2:] = 0.0
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        land_mask_override=jnp.asarray(ocean_mask),
        H_bathy_override=jnp.full((n_lat, n_lon), 4000.0))

    # A free-surface bump so the barotropic mode has something to propagate.
    lat_i = np.arange(n_lat)[:, None]
    lon_j = np.arange(n_lon)[None, :]
    bump = 0.5 * np.exp(-(((lat_i - n_lat / 2) / 2.0) ** 2
                          + ((lon_j - n_lon / 2) / 2.0) ** 2))
    eta0 = jnp.asarray(bump * ocean_mask)
    state = state._replace(eta=state.eta.replace(data=eta0))

    cfg = LatLonCGridOceanConfig.from_flat(barotropic_time_filter=time_filter)
    # Non-vacuity: this really is the branch under test.  If a future default
    # moves the stock config off explicit_substep, the assert says so instead
    # of the test quietly covering a different solver.
    assert cfg.barotropic.barotropic_solver == "explicit_substep"
    assert LatLonCGridOceanConfig().barotropic.barotropic_time_filter == "cosine"

    # The centred window runs 2n-1 substeps; n_loop must come from the callee.
    n_sub = int(cfg.barotropic.n_barotropic_substeps)
    w_filter, _w_total, w_transport, n_loop = _compute_weights(
        cfg, n_sub, jnp.float64)
    assert n_loop == 2 * n_sub - 1, "half window restored"
    assert w_filter.shape[0] == n_loop
    assert w_transport.shape[0] == n_loop

    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    for _ in range(2):
        state = model.step(state, 600.0)

    eta = np.asarray(state.eta.data)
    assert np.all(np.isfinite(eta)), "eta not finite"
    assert np.all(np.isfinite(np.asarray(state.u.data))), "u not finite"
    assert np.all(np.isfinite(np.asarray(state.v.data))), "v not finite"
    # The step did something: the bump spread and spun up a flow.
    assert not np.allclose(eta, np.asarray(eta0)), "eta did not evolve"
    assert np.max(np.abs(np.asarray(state.u.data))) > 0.0, "no flow generated"

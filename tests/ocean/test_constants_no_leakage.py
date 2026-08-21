"""A run that pins its constants must not be able to read the library's (#1627).

Four review rounds on this issue produced four lists of call sites, each found
by reading and each incomplete, because "did this call pass the argument" is
one site at a time. This asks the question the other way round, once:

    pin the run's constants, then MOVE the library's defaults, and require
    every output to be unchanged.

Anything still reading a module-level constant sees the moved value and the
output shifts. It does not matter how the leak is spelled -- a default
argument, an aliased import, a module constant read inside a closure, a
helper three levels down -- and it does not need a list.

WHY THE PARTIAL CASE IS THE DANGEROUS ONE, and why this is worth more than the
syntactic rule beside it. A model consistently on the wrong gravity is a
slightly different planet and its budgets still close. A model where SOME
terms moved and others did not is two planets coupled by a spurious source,
with a fixed sign, integrating in time. The error concentrates where two
quantities built from different constants are differenced -- a buoyancy
frequency against a density, a boundary-layer criterion against the profile it
is applied to -- which is exactly where a near-cancellation turns a five-parts-
in-a-hundred-thousand constant error into a decision that flips.

Scope, stated rather than implied: this covers the ocean paths a small
rest-state column can drive. It is not a whole-model gate, and a path it does
not reach is not certified by it.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as _constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.eos import compute_ocean_rho, compute_ocean_rho_and_pressure
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.vertical import create_ocean_z_star

#: How far to move the library defaults. Large enough that any leak is
#: unmistakable against round-off, small enough to stay physical.
_SHIFT = 1.10


@pytest.fixture()
def column():
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    # A NON-ZERO FREE SURFACE, and it is load-bearing. The reference density
    # enters the in-situ pressure only through the surface term
    # ``rho_ref*g*eta``, so on a rest state -- where the free surface is
    # exactly zero -- moving that constant changes nothing and every
    # assertion below passes for free. Measured: with a flat surface the
    # density is bit-identical whether the reference density is passed or
    # left to a value ten per cent away.
    eta = 0.5 * jnp.sin(jnp.linspace(0.0, 6.0, state.eta.data.size)).reshape(
        state.eta.data.shape)
    state = state._replace(eta=state.eta.replace(data=eta))
    jac = jnp.ones_like(jnp.asarray(state.eta.data))
    return state, z_coord, jac


#: Modules that re-bind a library constant at import time. Patching the
#: library alone does NOT reach these -- the name was copied when the module
#: loaded -- and a first version of this gate missed an injected leak for
#: exactly that reason. Each entry is (module, attribute).
_REBOUND = (
    ("legoesm.ocean.eos", "rho_0"),
    ("legoesm.ocean.eos", "c_sw"),
)


@pytest.fixture()
def moved_library(monkeypatch):
    """Move the library's constants out from under the code.

    Both binding styles are moved: the attribute the library exposes, which
    catches every ``constants.g`` read at call time, and the copies modules
    made at import time, which catch every ``from ... import rho_0``.
    """
    import importlib

    monkeypatch.setattr(_constants, "g", _constants.g * _SHIFT, raising=True)
    monkeypatch.setattr(_constants, "rho_ocean",
                        _constants.rho_ocean * _SHIFT, raising=True)
    for mod_name, attr in _REBOUND:
        mod = importlib.import_module(mod_name)
        monkeypatch.setattr(mod, attr, getattr(mod, attr) * _SHIFT,
                            raising=True)
    return _SHIFT


def test_the_rebound_list_is_not_stale():
    """Every module named above must still re-bind the attribute claimed.

    A stale entry silently narrows this gate: the fixture would patch a name
    nothing reads and report clean.
    """
    import importlib

    for mod_name, attr in _REBOUND:
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, attr), f"{mod_name} no longer defines {attr!r}"


def _pinned():
    """A pinned pair equal to today's library values, so a leak is the ONLY
    thing that can change an answer when the library moves."""
    return ConstantsConfig()


@pytest.mark.parametrize("which", ["g", "rho_0"])
def test_the_probe_itself_is_not_inert(column, which):
    """A no-leakage test passes for free if the quantity ignores the constant.

    Before asserting that a pinned run does not move, show that EACH constant
    moves it. Checked separately per constant, because they do not enter the
    same way: gravity multiplies the whole hydrostatic integral, while the
    reference density enters only through the free-surface term -- so a probe
    with a flat surface is live for one and dead for the other, which is
    exactly the trap the first version of this fixture fell into.
    """
    state, z_coord, jac = column
    cc = _pinned()
    base = np.asarray(compute_ocean_rho(state, z_coord, jac,
                                        g=cc.g, rho0=cc.rho_0))
    bumped = cc._replace(**{which: getattr(cc, which) * _SHIFT})
    moved = np.asarray(compute_ocean_rho(state, z_coord, jac,
                                         g=bumped.g, rho0=bumped.rho_0))
    assert not np.array_equal(base, moved), (
        f"the density does not respond to {which!r} on this state, so a "
        f"no-leakage assertion about it would pass for free")


def _answer(state, z_coord, jac, fn, cc):
    """EVERY output, not just the first.

    One of these helpers returns a density and a pressure, and gravity enters
    the pressure much more directly than it enters the density. A probe that
    kept only the first element was blind to a gravity leak in the second --
    an injected one went undetected until this was fixed.
    """
    out = fn(state, z_coord, jac, g=cc.g, rho0=cc.rho_0)
    parts = out if isinstance(out, tuple) else (out,)
    return np.concatenate([np.asarray(p).ravel() for p in parts])


@pytest.mark.parametrize("fn", [compute_ocean_rho, compute_ocean_rho_and_pressure],
                         ids=lambda f: f.__name__)
def test_a_pinned_call_does_not_read_the_library(column, fn, request):
    """The pinned pair must fully determine the answer.

    Both constants are passed, so the module-level values must not appear in
    the result. If either is still read somewhere inside, moving the library
    moves the output.
    """
    state, z_coord, jac = column
    cc = _pinned()

    # Answer FIRST with the library where it is, then again with it moved.
    # The pinned pair is identical in both, so any difference is a leak.
    baseline = _answer(state, z_coord, jac, fn, cc)
    request.getfixturevalue("moved_library")
    with_moved = _answer(state, z_coord, jac, fn, cc)

    np.testing.assert_array_equal(
        with_moved, baseline,
        err_msg=(f"{fn.__name__} returns a different answer when the LIBRARY "
                 f"constants move, although the run pinned its own. Something "
                 f"inside it is still reading a module-level constant."))

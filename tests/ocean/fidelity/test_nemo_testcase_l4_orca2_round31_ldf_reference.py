"""Round 31: dyn_ldf's consumer-local frozen F thickness.

NEMO's lateral diffusion multiplies the live stretch onto the MESH reference
thickness ``e3f_3d`` (``dynldf_lev.f90:123``) while its vorticity operator
multiplies the same stretch onto its own masked four-cell ``e3f_0vor``
(``dynvor.f90:734-738`` over ``:914-937``).  These tests pin that the two
references are separable, that the separation is not vacuous, and that the
zero substitution is the statement that distinguishes the two builders.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.vertical import (
    nemo_dynvor_e3f_0vor,
    nemo_ldf_reference_e3f,
    nemo_qco_live_vorticity_e3f_cgrid,
)


class _NoMesh:
    pass


class _Raw:
    def __init__(self, e3f_0):
        self.e3f_0 = e3f_0


class _WithMesh:
    def __init__(self, e3f_0):
        self.nemo_een_barotropic = _Raw(e3f_0)


def _grid():
    from legoesm.grids.latlon import create_latlon_grid

    return create_latlon_grid(n_lat=4, n_lon=4)


def test_the_mesh_reference_is_fail_closed_when_the_card_carries_none():
    with pytest.raises(ValueError, match="dynldf_lev.f90:123"):
        nemo_ldf_reference_e3f(_NoMesh())
    with pytest.raises(ValueError, match="dynldf_lev.f90:123"):
        nemo_ldf_reference_e3f(_WithMesh(None))


def test_the_mesh_reference_is_returned_when_the_card_carries_it():
    field = np.arange(8.0).reshape(2, 2, 2)
    assert nemo_ldf_reference_e3f(_WithMesh(field)) is field


def test_the_zero_substitution_is_the_statement_that_separates_the_builders():
    grid = _grid()
    tmask = jnp.zeros((4, 4, 2), dtype=jnp.float64).at[:2, :2].set(1.0)
    # The card's reference thickness is already zero below the seabed, which
    # is why legoESM's unmasked four-cell fill is zero exactly where the
    # masked average is and the substitution never fires.
    e3t_0 = 10.0 * tmask
    mesh = jnp.ones((4, 4, 2), dtype=jnp.float64) * 77.0

    default = np.asarray(nemo_dynvor_e3f_0vor(
        e3t_0, tmask, grid=grid, dtype=jnp.float64))
    repaired = np.asarray(nemo_dynvor_e3f_0vor(
        e3t_0, tmask, grid=grid, dtype=jnp.float64, substitute_e3f=mesh))

    dry = default == 0.0
    # Non-vacuity: the fixture must actually contain fully dry vertices, or
    # the two builders would agree for the wrong reason.
    assert dry.any()
    assert np.array_equal(repaired[dry], np.full(int(dry.sum()), 77.0))
    assert np.array_equal(default[~dry], repaired[~dry])
    assert not np.array_equal(default, repaired)


def test_the_live_vorticity_builder_uses_the_cards_mesh_zero_substitution():
    grid = _grid()
    tmask = jnp.zeros((4, 4, 2), dtype=jnp.float64).at[:2, :2].set(1.0)
    e3t_0 = 10.0 * tmask
    mesh = jnp.ones((4, 4, 2), dtype=jnp.float64) * 77.0
    eta = jnp.zeros((4, 4), dtype=jnp.float64)

    without_mesh = np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
        eta, _NoMesh(), jnp.float64, grid=grid, e3t_0=e3t_0, tmask=tmask))
    with_mesh = np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
        eta, _WithMesh(mesh), jnp.float64, grid=grid,
        e3t_0=e3t_0, tmask=tmask))

    changed = without_mesh != with_mesh
    assert changed.any()
    assert np.array_equal(with_mesh[changed], np.full(changed.sum(), 77.0))


def test_the_two_orders_agree_on_the_masked_average_statement():
    grid = _grid()
    tmask = jnp.zeros((4, 4, 2), dtype=jnp.float64).at[:2, :2].set(1.0)
    e3t_0 = 10.0 * tmask
    mesh = jnp.ones((4, 4, 2), dtype=jnp.float64) * 77.0
    _, default_s1, _ = nemo_dynvor_e3f_0vor(
        e3t_0, tmask, grid=grid, dtype=jnp.float64, return_stages=True)
    _, repaired_s1, _ = nemo_dynvor_e3f_0vor(
        e3t_0, tmask, grid=grid, dtype=jnp.float64, substitute_e3f=mesh,
        return_stages=True)
    assert np.array_equal(np.asarray(default_s1), np.asarray(repaired_s1))


def test_an_unknown_e3f_type_is_refused():
    grid = _grid()
    from legoesm.ocean.vertical import nemo_qco_live_vorticity_e3f_cgrid

    with pytest.raises(ValueError, match="nn_e3f_typ"):
        nemo_qco_live_vorticity_e3f_cgrid(
            jnp.zeros((4, 4)), None, jnp.float64, 2, grid=grid,
            e3t_0=jnp.ones((4, 4, 2)), tmask=jnp.ones((4, 4, 2)))


def test_a_reference_of_the_wrong_shape_is_refused():
    grid = _grid()
    from legoesm.ocean.vertical import nemo_qco_live_vorticity_e3f_cgrid

    with pytest.raises(ValueError, match="native A2D F-point thickness"):
        nemo_qco_live_vorticity_e3f_cgrid(
            jnp.zeros((4, 4)), None, jnp.float64, grid=grid,
            e3t_0=jnp.ones((4, 4, 2)), tmask=jnp.ones((4, 4, 2)),
            reference_e3f=jnp.ones((3, 3, 2)))


# --- the production wiring itself, which the unit tests above do not bind ---

def _function_source(module_path, name):
    """Return the AST of the named function that RUNS, not a wrapper."""
    import ast

    tree = ast.parse(Path(module_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {module_path}")


def _reference_calls(node):
    """Calls to the shared builder inside ``node``, keyed by their reference."""
    import ast

    found = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        target = call.func
        if not (isinstance(target, ast.Name)
                and target.id == "nemo_qco_live_vorticity_e3f_cgrid"):
            continue
        reference = None
        for keyword in call.keywords:
            if keyword.arg == "reference_e3f":
                reference = keyword.value
        if reference is None:
            found.append("vorticity")
        elif (isinstance(reference, ast.Call)
              and isinstance(reference.func, ast.Name)
              and reference.func.id == "nemo_ldf_reference_e3f"):
            found.append("mesh")
        else:
            found.append("other")
    return found


MODEL = (Path(__file__).resolve().parents[3]
         / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py")
TENDENCIES = (Path(__file__).resolve().parents[3]
              / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py")


@pytest.mark.parametrize("caller", ["_step_impl", "_mom_pert_ws"])
def test_each_lateral_diffusion_call_site_asks_for_the_mesh_reference(caller):
    # These two names are the functions the runtime census recorded as the
    # lateral-diffusion callers, not wrappers around them.  ``_mom_pert_ws``
    # is nested inside ``_step_impl``, so the outer walk legitimately sees
    # both; what binds the fix is that EVERY builder call reachable inside a
    # lateral-diffusion caller takes the mesh reference and none takes the
    # vorticity one.
    calls = _reference_calls(_function_source(MODEL, caller))
    assert calls and set(calls) == {"mesh"}, calls


def test_the_vorticity_caller_keeps_its_own_frozen_array():
    node = _function_source(
        TENDENCIES, "latlon_cgrid_ocean_baroclinic_tendencies")
    assert "mesh" not in _reference_calls(node)

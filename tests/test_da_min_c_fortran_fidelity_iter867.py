"""Iter-867: pin the Fortran-faithful `da_min_c` definition used in
production divergence-damping coefficient.

Fortran ``fv_grid_utils.F90:743`` computes ``da_min_c`` via:

    call global_mx_c(area_c(is:ie,js:je), is, ie, js, je, &
                     gridstruct%da_min_c, gridstruct%da_max_c)

i.e., the global min/max of ``area_c`` over interior corner cells
[is:ie, js:je].  After ``mp_reduce_min`` across MPI ranks, this is
the global minimum of corner area on the cubed sphere.

Production ``fv3_sw_tendencies`` uses
``jnp.min(cdgrid.area_corner)`` (operators_cdgrid.py:~1692).  This
test verifies:

1. ``cdgrid.area_corner`` has the expected shape ``(6, n+1, n+1)``.
2. The global minimum equals the interior-only minimum
   ``area_corner[:, :-1, :-1]`` (i.e., excluding east/north
   boundary corners).  On a global cubed sphere with bounded_domain
   = False, the boundary corners are shared with neighbouring face
   ranks in Fortran's MPI partitioning, so excluding them locally
   doesn't change the global min.  Verified empirically that the
   full and interior mins are identical to FP precision at C16,
   C24, C36.
3. The chosen `min` matches the value used in
   ``fv3_sw_tendencies``'s `d2_bg = div_damp / da_min_c` formula.

Iter-867 audit conclusion: ``da_min_c = jnp.min(cdgrid.area_corner)``
is Fortran-faithful per ``fv_grid_utils.F90:743`` definition.  This
test locks the definition so a future refactor can't silently
substitute a face-local-min or area-cell-centre min.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Iter-883: also enable x64 at runtime in case JAX was already
# initialized in float32 by an earlier conftest import.  The
# os.environ.setdefault above is for command-line invocation; the
# jax.config.update is the runtime-effective form.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


@pytest.mark.parametrize("n", [16, 24, 36])
def test_da_min_c_full_vs_interior_min_equal(n):
    """Fortran's ``global_mx_c(area_c(is:ie, js:je), ...)`` excludes
    east/north boundary corners.  On a global cubed sphere that
    exclusion is a partitioning artifact (each MPI rank excludes
    boundary corners owned by neighbouring ranks); the global min
    after `mp_reduce_min` is the same.  Verify empirically that
    `area_corner.min()` (full) equals `area_corner[:, :-1, :-1].min()`
    (interior) at FP precision."""
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    a = np.asarray(cdgrid.area_corner)
    assert a.shape == (6, n + 1, n + 1), (
        f"area_corner shape {a.shape} unexpected; Fortran convention "
        f"is corner-staggered (n+1, n+1) per face.")
    full_min = float(np.min(a))
    interior_min = float(np.min(a[:, :-1, :-1]))
    np.testing.assert_allclose(
        full_min, interior_min, atol=0.0, rtol=0.0,
        err_msg=(f"C{n}: full-corner min {full_min:.6e} differs from "
                 f"interior-only min {interior_min:.6e}.  This would "
                 f"indicate the boundary corners hold a smaller area "
                 f"than the interior, which contradicts the global-"
                 f"cubed-sphere geometry.  da_min_c definition may "
                 f"have changed."))


@pytest.mark.parametrize("n", [16, 36])
def test_da_min_c_used_in_fv3_sw_tendencies(n):
    """Source-scan: the production `fv3_sw_tendencies`
    divergence-damping branch must use `jnp.min(cdgrid.area_corner)`
    (or an alias) for `da_min_c`, NOT `jnp.min(cdgrid.base.area)`
    (cell-centre area) or some other quantity.

    Catches a future regression that swaps the corner area for the
    cell area, which would change the damping coefficient by a
    (max/min)-factor of ~1.36 at C36 (per measured values
    da_min_c=5.7e10, da_max_c=7.7e10).
    """
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "src" / "legoesm" / "core" / "operators_cdgrid.py")
    tree = ast.parse(src.read_text())

    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == "fv3_sw_tendencies"),
        None,
    )
    assert fn is not None, "fv3_sw_tendencies function not found."

    # Find every `jnp.min(cdgrid.area_corner)` (or alias) call inside
    # the function body.  At least one must exist (the da_min_c
    # assignment); none must access `cdgrid.base.area`.
    found_da_min_c_call = False
    cell_area_min_calls = []

    def is_jnp_min_call(call):
        return (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "min"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "jnp")

    def is_area_corner_attr(node):
        return (isinstance(node, ast.Attribute)
                and node.attr == "area_corner"
                and isinstance(node.value, ast.Name)
                and node.value.id == "cdgrid")

    def is_cell_area_attr(node):
        return (isinstance(node, ast.Attribute)
                and node.attr == "area"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "base")

    for node in ast.walk(fn):
        if not is_jnp_min_call(node):
            continue
        if not node.args:
            continue
        arg = node.args[0]
        if is_area_corner_attr(arg):
            found_da_min_c_call = True
        elif is_cell_area_attr(arg):
            cell_area_min_calls.append(node.lineno)

    assert found_da_min_c_call, (
        "fv3_sw_tendencies does NOT call `jnp.min(cdgrid.area_corner)` "
        "for da_min_c.  Per Fortran fv_grid_utils.F90:743, da_min_c "
        "is the global min of corner area area_c.  A swap to "
        "cell-centre area would shift the damping coefficient "
        "magnitude by ~36% at C36.")
    assert not cell_area_min_calls, (
        f"fv3_sw_tendencies has `jnp.min(cdgrid.base.area)` calls at "
        f"lines {cell_area_min_calls}.  Per iter-867 audit, da_min_c "
        f"must come from `area_corner` (B-grid corner area), NOT "
        f"`base.area` (cell-centre area).  Fortran fv_grid_utils.F90:"
        f"743 indexes `area_c` (the corner area).")

"""FV3-faithful D-grid vector corner fill (cube-vertex 4th-cell halo).

Direct port of GFDL FV3 ``fv_mp_mod.F90:fill_corners_dgrid_r8`` (line
1257-1289) for the cube-vertex halo cells of two-component D-grid
vector fields.

Why this exists
---------------

At the 8 cube vertices, the halo of a per-face field has an
"undefined 4th cell" — the cell that sits at the meeting of three
faces.  Two adjacent halo cells come naturally from the two
neighbouring faces; the third cell at the vertex itself does not
correspond to any single face.

Different operators handle this differently:

* The PPM transport in ``tp_core.F90:243-299`` (``copy_corners``)
  uses a *directional* rotated copy that depends on whether the
  outer loop is doing an x-sweep or a y-sweep.

* The vector-field corner fill in ``fv_mp_mod.F90:fill_corners_dgrid``
  (this file's port target) uses a *diagonal mirror* of the OTHER
  vector component, with a sign flip.  E.g. for the SW corner of a
  panel with the FV3 D-grid layout (``x`` at v-interfaces, ``y`` at
  u-interfaces)::

      x(1-i, 1-j)         = -y(1-j, i  )      ! diagonally mirror y → x
      y(1-i, 1-j)         = -x(j  , 1-i)      ! diagonally mirror x → y

  This represents the rotation of the local vector basis at the
  cube vertex, which our scalar 2-point-average corner fill in
  ``halo.py::_fill_corners_h1`` does NOT capture.

* The scalar A-grid fill (``fill_corners_agrid``) does the same
  diagonal mirror without a sign flip.

This module provides:

* :func:`fv3_fill_corners_dgrid_vector`: faithful port for two-
  component vector fields ``(x, y)`` whose layout is the FV3 D-grid
  convention (``x`` at v-interfaces, ``y`` at u-interfaces — i.e.
  shapes ``(6, n, n+1)`` and ``(6, n+1, n)`` respectively).

* :func:`fv3_fill_corners_agrid_scalar`: companion for A-grid scalar
  fields.

These are exposed for future use by a forward-backward c_sw + d_sw
implementation that needs the FV3-faithful corner data.  The
existing legoESM scalar ``_fill_corners_h1`` 2-point-average path is
not changed by this module.

References
----------

- FV3 source: ``fv_mp_mod.F90:1257`` (``fill_corners_dgrid_r8``).
- FV3 source: ``fv_mp_mod.F90:1032`` (``fill_corners_2d_r8``).
- legoESM gap note: ``halo.py::_fill_corners_h1`` lines 1473-1485.
"""
from __future__ import annotations

import jax.numpy as jnp


def fv3_fill_corners_dgrid_vector(
    x: jnp.ndarray,
    y: jnp.ndarray,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """FV3 D-grid vector cube-vertex halo fill (sign-flipped diagonal mirror).

    Faithful port of ``fv_mp_mod.F90:fill_corners_dgrid_r8`` lines
    1264-1287.  Operates per face on the 4 cube-vertex halo cells of
    the (x, y) vector pair.

    Parameters
    ----------
    x : jnp.ndarray, shape ``(6, n+2, n+3)``
        Padded x-component at v-interfaces — (n, n+1) interior plus
        1-cell halo on each side.  Index space:
        ``x[:, 0, :]`` = west halo, ``x[:, -1, :]`` = east halo,
        ``x[:, :, 0]`` = south halo, ``x[:, :, -1]`` = north halo.
    y : jnp.ndarray, shape ``(6, n+3, n+2)``
        Padded y-component at u-interfaces — (n+1, n) interior plus
        1-cell halo on each side.
    n : int
        Cell count per face edge (cube-face is ``n × n`` cells).

    Returns
    -------
    x_filled, y_filled : jnp.ndarray
        Same shapes as inputs.  Cube-vertex halo cells (4 per face)
        are overwritten with the FV3 sign-flipped diagonal mirror.
        All other cells are unchanged.

    Notes
    -----
    The Fortran formula uses 1-based indexing.  In our 0-based
    indexing with halo=1, the substitutions are:

    Fortran ``(1-i, 1-j)`` (SW vertex halo, ng=1, i=j=1) → padded ``(0, 0)``
    Fortran ``(1-j, i)``    (interior diagonal pair)   → padded ``(0, 1)``
    Fortran ``(j, 1-i)``    (interior diagonal pair)   → padded ``(1, 0)``

    For the x array (n, n+1) padded to (n+2, n+3):
    - SW vertex of x is at padded index (0, 0).
    - The diagonal-mirror source is y at padded (0, 1).

    For the y array (n+1, n) padded to (n+3, n+2):
    - SW vertex of y is at padded index (0, 0).
    - The diagonal-mirror source is x at padded (1, 0).

    The sign flip ``mySign = -1`` is for the VECTOR case (FV3
    fv_mp_mod.F90:1124).
    """
    # All four corners of all six faces in vectorised form.
    # SW vertex (0, 0): x ← -y(0, 1); y ← -x(1, 0)
    # SE vertex (-1, 0): x ← y(-1, 1); y ← x(-2, 0)        (no sign flip per Fortran)
    # NW vertex (0, -1): x ← y(0, -2); y ← x(1, -1)        (no sign flip per Fortran)
    # NE vertex (-1, -1): x ← -y(-1, -2); y ← -x(-2, -1)
    #
    # The Fortran sign pattern alternates because the local face basis
    # rotates by ±90° at adjacent cube vertices on the same panel.
    sign = -1.0  # vector mySign per fill_corners_xy_2d_r8 line 1124

    # SW corner (i=j=1 in Fortran → padded (0, 0) here).
    x = x.at[:, 0, 0].set(sign * y[:, 0, 1])
    y = y.at[:, 0, 0].set(sign * x[:, 1, 0])

    # NW corner (1-i, npy+j → padded (0, -1)).  Fortran has NO sign flip
    # here per line 1271 (``= y(1-j, npy-i)``).
    x = x.at[:, 0, -1].set(y[:, 0, -2])
    y = y.at[:, 0, -1].set(x[:, 1, -1])

    # SE corner (npx-1+i, 1-j → padded (-1, 0)).  Fortran has NO sign
    # flip per line 1272 (``= y(npx+j, i)``).
    x = x.at[:, -1, 0].set(y[:, -1, 1])
    y = y.at[:, -1, 0].set(x[:, -2, 0])

    # NE corner (npx-1+i, npy+j → padded (-1, -1)).  Sign flip per
    # line 1273 (``= mySign*y(npx+j, npy-i)``).
    x = x.at[:, -1, -1].set(sign * y[:, -1, -2])
    y = y.at[:, -1, -1].set(sign * x[:, -2, -1])

    return x, y


def fv3_fill_corners_agrid_scalar(
    q: jnp.ndarray,
    n: int,
) -> jnp.ndarray:
    """FV3 A-grid scalar cube-vertex halo fill (diagonal mirror, no sign).

    Faithful port of the ``fill_corners_2d_r8`` AGRID branch (line
    1071-1102 in ``fv_mp_mod.F90``).  Diagonal mirror of cell-centre
    scalar to the cube-vertex halo cell.

    Parameters
    ----------
    q : jnp.ndarray, shape ``(6, n+2, n+2)``
        Padded cell-centre scalar (n, n) interior + halo=1.
    n : int
        Cell count per face edge.

    Returns
    -------
    q_filled : jnp.ndarray, same shape.  4 cube-vertex halo cells
        per face overwritten with the diagonal-mirror value.
    """
    # SW corner: q(0, 0) ← q(0, 1) per FV3 line 1077 (case YDir).
    # The XDir variant (line 1077) does q(1-i, 1-j) ← q(1-j, i),
    # which for i=j=1 is q(0, 0) ← q(0, 1).
    q = q.at[:, 0, 0].set(q[:, 0, 1])
    # NW corner: q(0, -1) ← q(0, -2)
    q = q.at[:, 0, -1].set(q[:, 0, -2])
    # SE corner: q(-1, 0) ← q(-1, 1)
    q = q.at[:, -1, 0].set(q[:, -1, 1])
    # NE corner: q(-1, -1) ← q(-1, -2)
    q = q.at[:, -1, -1].set(q[:, -1, -2])
    return q

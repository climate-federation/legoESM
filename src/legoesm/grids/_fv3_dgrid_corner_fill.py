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


def fv3_fill_corners_cdgrid_vector(
    u: jnp.ndarray,
    v: jnp.ndarray,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """FV3-style cube-vertex halo fill adapted for the legoESM C-D grid.

    Our C-D grid puts BOTH wind components at cell corners, shape
    ``(6, n+1, n+1, ...)``.  After halo padding the shape is
    ``(6, n+3, n+3, ...)``.  This function overwrites the 4 cube-
    vertex halo cells per face with a sign-flipped diagonal mirror
    of the OTHER component, mirroring FV3's
    ``fill_corners_dgrid_r8`` philosophy adapted for the C-D layout.

    The cube vertex of face ``F`` lives at the corner shared by
    three faces (the "3-faces-meet" point).  In our padded
    representation:

    * SW cube vertex halo: padded ``(0, 0)``
    * NW cube vertex halo: padded ``(0, -1)``
    * SE cube vertex halo: padded ``(-1, 0)``
    * NE cube vertex halo: padded ``(-1, -1)``

    The interior diagonal-mirror source is the corner one cell INSIDE
    diagonally — in our 0-based padding with halo=1 those are the
    interior corners adjacent to the cube vertex (i.e. padded ``(1, 1)``,
    ``(1, -2)``, ``(-2, 1)``, ``(-2, -2)``).

    The sign convention follows the Fortran fv_mp_mod.F90:1270-1273
    pattern: SW and NE corners flip sign, NW and SE do not.  This
    captures the local-basis 90°-rotation between adjacent panels.

    Trailing axes (e.g., a vertical level dimension) are passed through
    unchanged — the function vectorises automatically over them.

    Parameters
    ----------
    u : jnp.ndarray, shape ``(6, n+3, n+3, ...)``
        Padded x-component at corners.  Must be at corner positions
        (NOT cell centres) — the cube-vertex halo cells live at the
        outer ring of the padded array.
    v : jnp.ndarray, same shape as u.
    n : int
        Cell count per face edge.  Should match ``u.shape[1] - 3``
        and ``u.shape[2] - 3`` (interior n+1 corners + halo=1).

    Returns
    -------
    u_filled, v_filled : jnp.ndarray
        Same shape as inputs, with cube-vertex halo cells replaced
        by the sign-flipped diagonal mirror.

    Notes
    -----
    The interior diagonal-mirror is one cell INSIDE the corner —
    e.g., for the SW cube vertex, the mirror is at padded (1, 1)
    rather than (0, 1).  This differs from FV3's ``fill_corners_dgrid``
    indexing because our layout has both components at the SAME
    grid positions (corners), so the mirror must be at the diagonal
    interior point (not the off-component-adjacent point that FV3
    uses).
    """
    sign = -1.0  # Vector sign flip per FV3 fv_mp_mod.F90:1124.

    # SW cube vertex (padded (0, 0)): sign-flipped mirror from interior (1, 1).
    # Use the value of u at the diagonal mirror to fill v at the corner,
    # and vice versa, so that the rotation between u and v is captured.
    u_sw_src = v[:, 1, 1]   # interior diagonal mirror of cube vertex
    v_sw_src = u[:, 1, 1]
    u = u.at[:, 0, 0].set(sign * u_sw_src)
    v = v.at[:, 0, 0].set(sign * v_sw_src)

    # NW cube vertex (padded (0, -1)): no sign flip per FV3 line 1271.
    u_nw_src = v[:, 1, -2]
    v_nw_src = u[:, 1, -2]
    u = u.at[:, 0, -1].set(u_nw_src)
    v = v.at[:, 0, -1].set(v_nw_src)

    # SE cube vertex (padded (-1, 0)): no sign flip per FV3 line 1272.
    u_se_src = v[:, -2, 1]
    v_se_src = u[:, -2, 1]
    u = u.at[:, -1, 0].set(u_se_src)
    v = v.at[:, -1, 0].set(v_se_src)

    # NE cube vertex (padded (-1, -1)): sign flip per FV3 line 1273.
    u_ne_src = v[:, -2, -2]
    v_ne_src = u[:, -2, -2]
    u = u.at[:, -1, -1].set(sign * u_ne_src)
    v = v.at[:, -1, -1].set(sign * v_ne_src)

    return u, v


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

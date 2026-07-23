"""
JAX/Python translation of the CLM column data type module.

Provides the :class:`column_type` NamedTuple, its factory
:func:`init_column`, and the module-level singleton :data:`col`
that is populated during model initialisation.

Original Fortran module: ColumnType
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_main import clm_varpar  # Import module, not values
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import ispval
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval as nan

# ---------------------------------------------------------------------------
# column_type
# ---------------------------------------------------------------------------


class column_type(NamedTuple):
    """
    Column data type.

    Mirrors Fortran derived type ``column_type`` (lines 20-27).

    All arrays use 1-based column indexing (index 0 allocated but
    unused).  Soil/snow layer dimensions follow the Fortran conventions:

    .. code-block:: none

        dz, z  : Fortran (-nlevsno+1 : nlevgrnd)
                 Python  shape (endc+1, nlevsno+nlevgrnd+1)
                 Python index j  ↔  Fortran index j - nlevsno
                 → Fortran layer 1 = Python index nlevsno+1

        zi     : Fortran (-nlevsno+0 : nlevgrnd)
                 Python  shape (endc+1, nlevsno+nlevgrnd+1)
                 same offset as dz/z (one extra interface at the bottom
                 is accommodated within the same allocation size)

    Attributes:
        snl:      Number of snow layers (negative in CLM convention),
                  shape ``(endc+1,)``, initialised to ``ispval``.
        dz:       Soil/snow layer thickness (m),
                  shape ``(endc+1, nlevsno+nlevgrnd+1)``.
        z:        Soil/snow layer mid-point depth (m),
                  shape ``(endc+1, nlevsno+nlevgrnd+1)``.
        zi:       Soil/snow layer interface depth (m).
                  Fortran ``(-nlevsno : nlevgrnd)`` has one more
                  element than ``dz``/``z``; allocated with the same
                  Python shape ``(endc+1, nlevsno+nlevgrnd+1)`` so
                  that Fortran index ``-nlevsno`` maps to Python
                  index 0 and Fortran index ``nlevgrnd`` maps to
                  Python index ``nlevsno+nlevgrnd``.
        nbedrock: Variable depth-to-bedrock layer index,
                  shape ``(endc+1,)``, initialised to ``ispval``.
    """

    snl: jnp.ndarray  # (endc+1,)                    int
    dz: jnp.ndarray  # (endc+1, nlevsno+nlevgrnd+1) float
    z: jnp.ndarray  # (endc+1, nlevsno+nlevgrnd+1) float
    zi: jnp.ndarray  # (endc+1, nlevsno+nlevgrnd+1) float
    nbedrock: jnp.ndarray  # (endc+1,)                    int


# ---------------------------------------------------------------------------
# init_column  (replaces Fortran Init)
# ---------------------------------------------------------------------------


def init_column(begc: int, endc: int) -> column_type:
    """
    Allocate and initialise all column arrays.

    Mirrors Fortran subroutine ``Init`` (lines 38-49).

    **Shapes and fill values**:

    .. code-block:: none

        snl      : (endc+1,)                    → ispval
        dz, z    : (endc+1, nlevsno+nlevgrnd+1) → spval (nan)
        zi       : (endc+1, nlevsno+nlevgrnd+1) → spval (nan)
        nbedrock : (endc+1,)                    → ispval

    The second dimension ``nlevsno+nlevgrnd+1`` covers the Fortran
    range ``-nlevsno+1 : nlevgrnd`` for ``dz``/``z`` (which has
    ``nlevsno+nlevgrnd`` elements) and ``-nlevsno : nlevgrnd`` for
    ``zi`` (which has ``nlevsno+nlevgrnd+1`` elements).  Both are
    stored in the same Python shape to avoid separate indexing
    conventions; the extra element at index 0 is unused for
    ``dz``/``z``.

    Args:
        begc: Beginning column index (unused; present for API
              compatibility — in standalone mode always 1).
        endc: Ending column index.

    Returns:
        Initialised :class:`column_type`.
    """
    nc = endc + 1  # column dimension
    nth = clm_varpar.nlevsno + clm_varpar.nlevgrnd + 1  # layer dimension for dz, z, zi

    return column_type(
        snl=jnp.full((nc,), ispval, dtype=jnp.int32),
        dz=jnp.full((nc, nth), nan, dtype=jnp.float64),
        z=jnp.full((nc, nth), nan, dtype=jnp.float64),
        zi=jnp.full((nc, nth), nan, dtype=jnp.float64),
        nbedrock=jnp.full((nc,), ispval, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Module-level singleton — Fortran: type(column_type), public, target :: col
# ---------------------------------------------------------------------------

# Initialize clm_varpar before creating col to ensure nlevgrnd/nlevsno are set
clm_varpar.clm_varpar_init()

# Populated by calling col = init_column(begc, endc) during model init.
col: column_type = init_column(1, 1)

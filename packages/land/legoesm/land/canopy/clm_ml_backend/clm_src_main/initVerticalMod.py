"""
JAX translation of initVerticalMod Fortran module.

Initialize vertical components of the column data type. Computes
soil layer depths, thicknesses, and interface depths for either the
CLM4.5 or CLM5.0 vertical discretization, and sets the bedrock index
for each column based on tower site depth-to-bedrock data.

Original Fortran module: initVerticalMod
Fortran lines 1-115
"""

import jax.numpy as jnp

import legoesm.land.canopy.clm_ml_backend.offline_driver.TowerDataMod as TowerDataMod
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import zmin_bedrock  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, nlevsoi  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.clmSoilOptionMod import clm_phys  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level constant (Fortran line 38)
# ---------------------------------------------------------------------------

scalez: float = 0.025  # Soil layer thickness discretization (m)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def initVertical(bounds: bounds_type) -> None:
    """
    Initialize vertical column structure: layer depths, thicknesses,
    interface depths, and bedrock index.

    Mirrors Fortran subroutine ``initVertical`` (lines 22-110).

    Two vertical discretizations are supported, selected by
    ``clm_phys`` from ``clmSoilOptionMod``:

    **CLM4_5** (Fortran lines 52-72):
        Layer depths are defined by an exponential stretching formula:

        .. code-block:: none

            z(c,j) = scalez * (exp(0.5*(j - 0.5)) - 1)

        Layer thicknesses and interface depths are derived from the
        layer depths.

    **CLM5_0** (Fortran lines 74-96):
        Layer thicknesses are defined piecewise for soil layers
        (j = 1–4, 5–13, 14–nlevsoi) and bedrock layers
        (j = nlevsoi+1–nlevgrnd). Interface depths are cumulative
        sums of thicknesses; layer depths are midpoints of interfaces.

    After the grid is defined, the bedrock layer index ``nbedrock(c)``
    is determined for each column from the tower site depth-to-bedrock
    ``tower_zbed``, bounded below by the layer index at ``zmin_bedrock``
    (Fortran lines 100-112).

    Args:
        bounds: Decomposition bounds supplying ``begc`` and ``endc``
            for the local MPI task.
    """
    import legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType as column_module

    # Get global col
    global_col = column_module.col

    # Unpack mutable column arrays (Fortran associate block, lines 44-49)
    dz = global_col.dz  # Soil layer thickness (m)
    z = global_col.z  # Soil layer depth (m)
    zi = global_col.zi  # Soil layer depth at layer interface (m)
    nbedrock = global_col.nbedrock  # Depth to bedrock index

    begc = bounds.begc
    endc = bounds.endc  # Fortran line 51

    # ------------------------------------------------------------------
    # Define CLM layer structure for soil — Fortran lines 53-98
    # ------------------------------------------------------------------
    for c in range(begc, endc + 1):  # Fortran: do c = begc, endc

        if clm_phys == "CLM4_5":
            # --------------------------------------------------------
            # CLM4.5 exponential grid — Fortran lines 55-72
            # --------------------------------------------------------

            # Layer depths — Fortran lines 57-59
            for j in range(1, nlevgrnd + 1):
                z = z.at[c, j].set(scalez * (jnp.exp(0.5 * (j - 0.5)) - 1.0))

            # Layer thickness — Fortran lines 61-65
            dz = dz.at[c, 1].set(0.5 * (z[c, 1] + z[c, 2]))
            for j in range(2, nlevgrnd):  # j = 2, nlevgrnd-1
                dz = dz.at[c, j].set(0.5 * (z[c, j + 1] - z[c, j - 1]))
            dz = dz.at[c, nlevgrnd].set(z[c, nlevgrnd] - z[c, nlevgrnd - 1])

            # Interface depths — Fortran lines 67-71
            zi = zi.at[c, 0].set(0.0)
            for j in range(1, nlevgrnd):  # j = 1, nlevgrnd-1
                zi = zi.at[c, j].set(0.5 * (z[c, j] + z[c, j + 1]))
            zi = zi.at[c, nlevgrnd].set(z[c, nlevgrnd] + 0.5 * dz[c, nlevgrnd])

        elif clm_phys == "CLM5_0":
            # --------------------------------------------------------
            # CLM5.0 piecewise grid — Fortran lines 74-96
            # --------------------------------------------------------

            # Layer thickness: soil layers j = 1..4 — Fortran lines 76-78
            for j in range(1, 5):
                dz = dz.at[c, j].set(j * 0.02)

            # Layer thickness: soil layers j = 5..13 — Fortran lines 80-82
            for j in range(5, 14):
                dz = dz.at[c, j].set(dz[c, 4] + (j - 4) * 0.04)

            # Layer thickness: soil layers j = 14..nlevsoi — Fortran lines 84-86
            for j in range(14, nlevsoi + 1):
                dz = dz.at[c, j].set(dz[c, 13] + (j - 13) * 0.10)

            # Layer thickness: bedrock layers — Fortran lines 88-90
            for j in range(nlevsoi + 1, nlevgrnd + 1):
                dz = dz.at[c, j].set(dz[c, nlevsoi] + (((j - nlevsoi) * 25.0) ** 1.5) / 100.0)

            # Interface depths: cumulative sum of thicknesses — Fortran lines 92-94
            zi = zi.at[c, 0].set(0.0)
            for j in range(1, nlevgrnd + 1):
                zi = zi.at[c, j].set(float(jnp.sum(dz[c, 1 : j + 1])))

            # Layer depths: midpoint of interfaces — Fortran lines 96-98
            for j in range(1, nlevgrnd + 1):
                z = z.at[c, j].set(0.5 * (zi[c, j - 1] + zi[c, j]))

        else:
            endrun(msg=" ERROR: initVertical: clm_phys not valid")

    # ------------------------------------------------------------------
    # Set column bedrock index — Fortran lines 100-112
    # ------------------------------------------------------------------
    for c in range(begc, endc + 1):  # Fortran: do c = begc, endc

        # Depth to bedrock for the tower site — Fortran line 103
        zbedrock = float(TowerDataMod.tower_zbed[TowerDataMod.tower_num])

        # Minimum index for minimum soil depth (jmin_bedrock) — Fortran lines 105-109
        jmin_bedrock = 3
        for j in range(3, nlevsoi + 1):  # Fortran: do j = 3, nlevsoi
            if float(zi[c, j - 1]) < zmin_bedrock <= float(zi[c, j]):
                jmin_bedrock = j

        # Bedrock layer index — Fortran lines 111-115
        nbedrock = nbedrock.at[c].set(nlevsoi)
        for j in range(jmin_bedrock, nlevsoi + 1):  # Fortran: do j = jmin_bedrock, nlevsoi
            if float(zi[c, j - 1]) < zbedrock <= float(zi[c, j]):
                nbedrock = nbedrock.at[c].set(j)

    # ------------------------------------------------------------------
    # Write updated arrays back into the immutable column container.
    # Also initialise snl to 0 (no snow layers) for offline no-snow runs.
    # col.snl is set to ispval by init_column; the first real value is 0.
    # ------------------------------------------------------------------
    import jax.numpy as _jnp

    snl_zeros = _jnp.zeros_like(global_col.snl)
    updated_col = global_col._replace(
        snl=snl_zeros,
        dz=dz,
        z=z,
        zi=zi,
        nbedrock=nbedrock,
    )

    # Update the global col in ColumnType module
    column_module.col = updated_col

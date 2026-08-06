"""Shared checkerboard-mode projector for the DINO #1226 east-wall investigation.

Extracted from the mode-attribution instruments used during the #1226
east-wall v-checkerboard hunt (``mode_attrib_zdf*.py``, ``substage_bisect*.py``,
``twin_true_bundle.py`` in the session scratchpad — those scripts build the
projector inline and are NOT refactored to import this; this module is for
*future* instruments that need the same projector).

The projector isolates the grid-scale checkerboard mode that appears in the
east-wall v-field of a reference twin run: a 2*dx high-pass (raw minus a
5-point running mean along the longitude axis) restricted to the east-wall
column block (cols 45-51) and the top 8 levels (k 0-8), L2-normalized so
``project(v_field_with_unit_checkerboard, proj) ~= 1``.

NEMO row alignment: legoESM v-faces are stored on face row j+1 for NEMO's
row j (one extra south-wall row), so a projector built from a legoESM
v-field must drop row 0 (``proj[1:, :, :]``) before contracting with a NEMO
``vn``/``vtrd_*`` array of NEMO's native row count.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

# East-wall column block and upper-level band the checkerboard signal lives in
# (see #1226 mode-attribution session notes, 2026-07-23/24).
_COL_LO, _COL_HI = 45, 51
_LEV_LO, _LEV_HI = 0, 8
_SMOOTH_SIZE = 5  # running-mean window (grid points) defining the high-pass


def build_checkerboard_projector(v_field: np.ndarray) -> np.ndarray:
    """Build the normalized checkerboard-mode projector ``proj`` from a reference v.

    Parameters
    ----------
    v_field : (n_lat+1, n_lon, n_lev) array
        A legoESM v-face field (e.g. a reference twin's east-wall v snapshot)
        containing the checkerboard signature to isolate.

    Returns
    -------
    proj : same shape as ``v_field``, L2-normalized (``sum(proj**2) == 1``),
        zero outside the east-wall column block / upper-level band.
    """
    v_field = np.asarray(v_field)
    smooth = uniform_filter1d(v_field, size=_SMOOTH_SIZE, axis=1, mode="nearest")
    hp = v_field - smooth
    proj = np.zeros_like(v_field)
    proj[:, _COL_LO:_COL_HI, _LEV_LO:_LEV_HI] = hp[:, _COL_LO:_COL_HI, _LEV_LO:_LEV_HI]
    norm = np.sqrt(np.sum(proj**2))
    if norm > 0:
        proj /= norm
    return proj


def project(field: np.ndarray, proj: np.ndarray) -> float:
    """Contract ``field`` against projector ``proj`` (elementwise sum of products).

    ``field`` and ``proj`` must already share row alignment: use
    ``proj[1:, :, :]`` when ``field`` is a NEMO-native array (NEMO has one
    fewer v-row than legoESM's face storage — see module docstring).
    """
    return float(np.sum(np.asarray(field) * np.asarray(proj)))

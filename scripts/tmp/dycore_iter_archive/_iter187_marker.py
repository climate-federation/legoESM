"""FV3_3D iter 202: shared iter-187 marker substrings used by both
the PE iter-188 (``test_fv3_pe_toolkit_iter188.py``) and the NH
iter-172 (``test_fv3_nh_toolkit_iter172.py``) AST regression
guards.

Why this module exists
----------------------

iter-187 added the FV3 smag_vort cap to BOTH PE and NH 3D paths
with the same wiring pattern.  Both AST guards check for the
same marker substrings inside their respective source files.

iter-190 then changed the iter-187 site assignment from a local
``jax.vmap(...)`` call to ``= _zeta_a2b_ord4`` (precomputed at the
iter-170 site for dedup).  This required updating BOTH AST guard
substrings.

iter-190 updated ONLY the NH iter-172 substring; the PE iter-188
substring drifted out-of-sync silently.  iter-201 caught the
broken PE guard manually, but the bug had been silent for several
iterations because no recent test run included the PE guard.

This module provides ONE source-of-truth for the iter-187 marker
substrings.  Both PE and NH guards import from here, so a future
substring change is made in ONE place and automatically
propagates to both guards.
"""
from __future__ import annotations


# iter-187 smag_vort cap site marker.
#
# The site is::
#
#     _zeta_smag_corner = _zeta_a2b_ord4   # (post-iter-190 form)
#     _smag_arg = _delpc_initial ** 2 + _zeta_smag_corner ** 2
#     ...
#
# in both ``primitive_eq_cdgrid.py`` and ``compressible_euler_cdgrid.py``.
# The unique marker is the assignment ``_zeta_smag_corner =``;
# ``interp_center_to_corner_a2b_ord4`` is the helper that must
# remain present at the iter-170 site (which iter-190 dedup'd into).
ITER187_GATE = "_zeta_smag_corner ="
ITER187_HELPER = "interp_center_to_corner_a2b_ord4"

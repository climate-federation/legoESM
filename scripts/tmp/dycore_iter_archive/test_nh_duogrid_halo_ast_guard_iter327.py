"""FV3_3D iter 327: AST regression guard for iter-325 duogrid
threading on the NH 3D path.

iter-325 wired ``duogrid=grid.duogrid`` (or ``_nh_dg``) through the
3 NH halo sites that previously bypassed the duogrid Lagrange-
extended halo.  Without an AST guard a future refactor could
silently drop one of these kwargs, restoring the iter-325 cube-
edge artifact bug without any direct test failure (since the
no-duogrid path is bit-for-bit identical and the duogrid impact
tests use ``use_duogrid=True`` with the full toolkit).

iter-327 parses the NH source AST and asserts the iter-325 wiring
markers are present.

Tests
-----

1. ``test_nh_dg_helper_assigned`` — ``_nh_dg = grid.duogrid``
   appears in ``cdgrid_compressible_euler_slow_tendencies``
   (iter-325 helper).
2. ``test_packed_pad_halo_4d_passes_duogrid`` — the SPMD packed
   halo for K + π_prime passes ``duogrid=_nh_dg``.
3. ``test_packed_pad_halo_mpi_4d_passes_duogrid`` — the MPI packed
   halo for K + π_prime passes ``duogrid=_nh_dg``.
4. ``test_ke_correction_pad_passes_duogrid`` — the corner-div
   damping ke_correction halo passes ``duogrid=_nh_dg``.
"""
from __future__ import annotations

import re

import pytest

from tests.legoesm_paths import legoesm_source_path


SRC_PATH = legoesm_source_path(
    "atmosphere/dynamics/gcm/compressible_euler_cdgrid.py"
)


@pytest.fixture(scope="module")
def nh_source():
    return SRC_PATH.read_text()


def test_nh_dg_helper_assigned(nh_source):
    """``_nh_dg = grid.duogrid`` helper appears in the slow_tendencies
    function (iter-325 marker)."""
    assert re.search(r"_nh_dg\s*=\s*grid\.duogrid", nh_source), (
        "iter-325 marker '_nh_dg = grid.duogrid' missing from "
        f"{SRC_PATH.name}.  A refactor likely dropped the duogrid "
        f"helper assignment, restoring the cube-edge halo bypass "
        f"on the NH 3D path."
    )


def test_packed_pad_halo_4d_passes_duogrid(nh_source):
    """``packed_pad_halo_4d(K, pi_prime, mesh=..., duogrid=_nh_dg)``."""
    pat = (
        r"packed_pad_halo_4d\([^)]*K[^)]*pi_prime[^)]*"
        r"duogrid\s*=\s*_nh_dg"
    )
    assert re.search(pat, nh_source, re.DOTALL), (
        "iter-325 SPMD packed halo for (K, pi_prime) does not pass "
        "duogrid=_nh_dg.  A refactor likely dropped the kwarg, "
        "silently restoring the cube-edge halo bypass."
    )


def test_packed_pad_halo_mpi_4d_passes_duogrid(nh_source):
    """``packed_pad_halo_mpi_4d(K, pi_prime, topology=..., duogrid=_nh_dg)``."""
    pat = (
        r"packed_pad_halo_mpi_4d\([^)]*K[^)]*pi_prime[^)]*"
        r"duogrid\s*=\s*_nh_dg"
    )
    assert re.search(pat, nh_source, re.DOTALL), (
        "iter-325 MPI packed halo for (K, pi_prime) does not pass "
        "duogrid=_nh_dg.  A refactor likely dropped the kwarg."
    )


def test_ke_correction_pad_passes_duogrid(nh_source):
    """``_pad_halo_4d_module(_ke_correction, duogrid=_nh_dg)``."""
    pat = (
        r"_pad_halo_4d_module\(\s*_ke_correction\s*,\s*"
        r"duogrid\s*=\s*_nh_dg"
    )
    assert re.search(pat, nh_source, re.DOTALL), (
        "iter-325 ke_correction halo (corner-divergence damping "
        "site) does not pass duogrid=_nh_dg.  A refactor likely "
        "dropped the kwarg, restoring O(dx²) cube-edge bias at "
        "the FV3 corner-div damp gradient."
    )

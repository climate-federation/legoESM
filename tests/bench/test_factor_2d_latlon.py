"""Direct tests for ``run_cpu_mpi_scaling._factor_2d_latlon`` — the
rank-count -> (proc_lat, proc_lon) factorisation for the lat-lon 2-D pencil
benchmark path (``--latlon-2d``).

The 2-D STEP itself is validated by tests/distributed/test_latlon_2d_mpi_step.py
(mass<1e-12 + 2x2==1x4); this pins only the harness factorisation logic
(min-perimeter, per-block constraints, loud failure) — pure Python, no JAX.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# Import the harness module directly (scripts/ is not a package).  Register in
# sys.modules BEFORE exec_module — the module defines @dataclass classes, and
# dataclasses resolves ``cls.__module__`` via sys.modules during the decorator
# (an unregistered module -> None.__dict__ AttributeError at import).
_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "bench" / "run_cpu_mpi_scaling.py")
_spec = importlib.util.spec_from_file_location("_run_cpu_mpi_scaling", _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_run_cpu_mpi_scaling"] = _mod
_spec.loader.exec_module(_mod)
_factor = _mod._factor_2d_latlon


def _perim(n_lat, n_lon, pl, pc):
    return n_lat / pl + n_lon / pc


@pytest.mark.parametrize("n_ranks,n_lat,n_lon", [
    (4, 8, 16), (8, 8, 16), (16, 8, 16), (8, 16, 32), (6, 12, 24), (12, 24, 48),
])
def test_factor_valid_and_min_perimeter(n_ranks, n_lat, n_lon):
    pl, pc = _factor(n_ranks, n_lat, n_lon)
    # exact factor pair
    assert pl * pc == n_ranks
    # per-block constraints (>=2 lat rows for halo=2; >=2 lon cols)
    assert n_lat // pl >= 2 and n_lon // pc >= 2
    # minimal halo perimeter among ALL valid factor pairs
    valid = [
        (q, n_ranks // q) for q in range(1, n_ranks + 1)
        if n_ranks % q == 0
        and n_lat // q >= 2 and n_lon // (n_ranks // q) >= 2
    ]
    best = min(_perim(n_lat, n_lon, q, r) for q, r in valid)
    assert _perim(n_lat, n_lon, pl, pc) == pytest.approx(best)


def test_factor_balanced_block_for_square_friendly_counts():
    # n_ranks=8, n_lat=8, n_lon=16 -> (2,4) gives 4x4 blocks (perimeter 8,
    # strictly less than band (8,1)=invalid or (1,8)=10).
    assert _factor(8, 8, 16) == (2, 4)


def test_factor_raises_when_too_many_ranks():
    # 64 ranks can't keep >=2 rows AND >=2 cols on an 8x16 grid.
    with pytest.raises(ValueError, match="no 2-D factorisation"):
        _factor(64, 8, 16)


def test_factor_single_rank_is_full_domain():
    # 1 rank -> (1, 1): the whole domain (block == global), trivially valid.
    assert _factor(1, 8, 16) == (1, 1)

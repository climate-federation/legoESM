"""Node detection behind the LES scaling bench's ``--decomp auto`` rule.

Run (one node)::

    mpirun -np 4 python -m pytest tests/distributed/test_les_bench_node_count_mpi.py
"""
import sys
from pathlib import Path

from mpi4py import MPI

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "bench"))
sys.path.insert(0, str(_REPO / "scripts" / "run"))

import bench_spectral_les_dd_scaling as les_b  # noqa: E402


def test_count_nodes_on_one_node_selects_planes():
    comm = MPI.COMM_WORLD
    n_nodes = les_b.count_nodes(comm)
    host = MPI.Get_processor_name()
    hosts = set(comm.allgather(host))
    assert n_nodes == len(hosts)
    assert les_b.choose_decomp("auto", n_nodes) == (
        "planes" if n_nodes == 1 else "pencil")

"""A production AMIP arm runs on at least FOUR GPUs (user directive 2026-09-23).

The CAM6 run-1 arm was launched on a single device and managed about four
simulated days per hour, so a 60-day arm cost roughly fifteen hours of wall
clock the sharded mesh did not require.  This gate pins the floor in the batch
requests, which is where the count is actually chosen: the chain scripts take
whatever the request gives them.

It also pins the two consistency guards, because they are the part that
protects the science rather than the throughput.  Both arms of a pair must run
at the SAME device count: the ocean solver converges to a tolerance at one
device and runs a fixed iteration count when sharded, and the halo path has had
device-count dependent defects, so a mismatch is a confound and not a speed
difference.
"""
from __future__ import annotations

import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_LEVANTE = _REPO / "scripts" / "cluster" / "levante" / "amip_mpas_gpu_chain.sbatch"
_GINSBURG = _REPO / "scripts" / "cluster" / "amip" / "amip_ginsburg_production_chain.sbatch"
_FLOOR = 4


def _sbatch_ints(text: str, directive: str) -> list[int]:
    return [int(m) for m in re.findall(rf"^#SBATCH --{directive}=(?:gpu:)?(\d+)\s*$",
                                       text, re.M)]


@pytest.mark.parametrize("script", [_LEVANTE, _GINSBURG], ids=lambda p: p.name)
def test_batch_request_asks_for_at_least_four_gpus(script) -> None:
    text = script.read_text()
    counts = _sbatch_ints(text, "gpus-per-node") + _sbatch_ints(text, "gres")
    assert counts, f"{script.name} names no GPU count in its batch request"
    for n in counts:
        assert n >= _FLOOR, (
            f"{script.name} requests {n} GPU(s); a production AMIP arm runs on "
            f"at least {_FLOOR} (user directive 2026-09-23)")


def test_levante_chain_refuses_a_count_below_the_floor() -> None:
    """The floor is enforced at run time too, not only in the request.

    The request can be overridden on the sbatch command line, so the script
    itself has to refuse rather than trust it.
    """
    text = _LEVANTE.read_text()
    assert ': "${AMIP_N_GPUS:=4}"' in text, "no explicit device count"
    assert re.search(r"if \(\( AMIP_N_GPUS < 4 \)\); then", text), "no floor guard"
    assert "exit 64" in text, "the floor guard does not fail the job"


def test_levante_chain_pins_the_device_count_for_a_pair() -> None:
    """A run cannot change device count mid-chain, and a pair cannot differ."""
    text = _LEVANTE.read_text()
    # written on the first link, compared on every later one
    assert '_DEVCOUNT_FILE="${OUTDIR}/.device_count"' in text
    assert "exit 65" in text, "no guard against resuming at a different count"
    # and the partner arm is checked when one is named
    assert "AMIP_PAIR_DIR" in text
    assert "exit 66" in text, "no guard against a pair running at different counts"


def test_levante_chain_carries_the_count_into_the_next_link() -> None:
    """The self-resubmit must request the same count, or the chain drifts."""
    text = _LEVANTE.read_text()
    resubmit = text[text.index("NEXT=$(sbatch"):]
    assert '--gpus-per-node="${AMIP_N_GPUS}"' in resubmit
    assert "AMIP_N_GPUS=${AMIP_N_GPUS}" in resubmit, "count not exported forward"
    assert "AMIP_PAIR_DIR=${AMIP_PAIR_DIR:-}" in resubmit


def test_levante_chain_launches_one_rank_per_gpu() -> None:
    """MPAS federates with the mpi4jax halo backend, so multi-GPU means
    multi-RANK: distributed_mode 'spmd' is cubed-sphere only.  Without srun and
    --distributed the extra GPUs would simply sit idle."""
    text = _LEVANTE.read_text()
    assert 'srun --ntasks="${AMIP_N_GPUS}"' in text
    assert "CMD+=( --distributed )" in text

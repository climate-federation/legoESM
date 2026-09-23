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


# ---------------------------------------------------------------------------
# BEHAVIOURAL gates.  Codex round 1 proved the string assertions above are not
# enough: replacing both consistency `exit`s with `echo` left all six passing.
# These run the guards for real, in a shell, against a temporary run directory,
# and assert the EXIT CODE -- so a guard that stops refusing goes red.
# ---------------------------------------------------------------------------
import subprocess
import textwrap


def _guard_block() -> str:
    """The device-count guards, lifted from the chain between their markers.

    Lifted rather than duplicated: if the script's guards are edited, these
    tests run the edited ones.  A copy in the test would drift and bless
    whatever it remembered.
    """
    text = _LEVANTE.read_text()
    start = text.index('# DEVICE COUNT.  Standing user directive')
    end = text.index('echo "[chain] device count:')
    return text[start:end]


def _run_guards(tmp_path, *, outdir, n_gpus, pair_dir=None, ntasks="4",
                latest_ckpt=""):
    script = textwrap.dedent(f"""
        set -uo pipefail
        OUTDIR={outdir}
        LATEST_CKPT="{latest_ckpt}"
        AMIP_N_GPUS={n_gpus}
        SLURM_NTASKS={ntasks}
        {"AMIP_PAIR_DIR=" + str(pair_dir) if pair_dir else ""}
    """) + _guard_block() + "\nexit 0\n"
    p = tmp_path / "guards.sh"
    p.write_text(script)
    return subprocess.run(["bash", str(p)], capture_output=True, text=True).returncode


def test_guard_refuses_below_the_floor(tmp_path):
    d = tmp_path / "arm"; d.mkdir()
    assert _run_guards(tmp_path, outdir=d, n_gpus=1) == 64
    assert _run_guards(tmp_path, outdir=d, n_gpus=4) == 0


def test_guard_refuses_more_gpus_than_the_allocation(tmp_path):
    d = tmp_path / "arm"; d.mkdir()
    assert _run_guards(tmp_path, outdir=d, n_gpus=8, ntasks="4") == 67


def test_guard_refuses_a_chain_resumed_at_a_different_count(tmp_path):
    d = tmp_path / "arm"; d.mkdir()
    assert _run_guards(tmp_path, outdir=d, n_gpus=4) == 0       # stamps 4
    assert (d / ".device_count").read_text().strip() == "4"
    assert _run_guards(tmp_path, outdir=d, n_gpus=8, ntasks="8") == 65


def test_guard_refuses_checkpoints_with_no_provenance(tmp_path):
    """A one-GPU checkpoint continued here must not be stamped as four."""
    d = tmp_path / "arm"; d.mkdir()
    (d / "checkpoint_day_0005.npz").write_text("")
    rc = _run_guards(tmp_path, outdir=d, n_gpus=4,
                     latest_ckpt=str(d / "checkpoint_day_0005.npz"))
    assert rc == 69
    assert not (d / ".device_count").exists(), "invented provenance anyway"


def test_guard_refuses_a_pair_at_different_counts(tmp_path):
    a = tmp_path / "armA"; a.mkdir()
    b = tmp_path / "armB"; b.mkdir()
    assert _run_guards(tmp_path, outdir=a, n_gpus=4) == 0
    assert _run_guards(tmp_path, outdir=b, n_gpus=8, ntasks="8",
                       pair_dir=a) == 66


def test_guard_refuses_a_partner_that_has_not_started(tmp_path):
    """A missing partner stamp cannot be told apart from agreement."""
    a = tmp_path / "armA"; a.mkdir()
    b = tmp_path / "armB"; b.mkdir()
    assert _run_guards(tmp_path, outdir=b, n_gpus=4, pair_dir=a) == 68


def test_guard_refuses_a_pair_that_does_not_point_both_ways(tmp_path):
    """Otherwise 'the pair' can be one arm agreeing with itself."""
    a = tmp_path / "armA"; a.mkdir()
    b = tmp_path / "armB"; b.mkdir()
    c = tmp_path / "armC"; c.mkdir()
    assert _run_guards(tmp_path, outdir=a, n_gpus=4) == 0
    (a / ".pair_dir").write_text(str(c) + "\n")      # A thinks its partner is C
    assert _run_guards(tmp_path, outdir=b, n_gpus=4, pair_dir=a) == 70


def test_a_matching_pair_is_accepted(tmp_path):
    """The guards must not refuse a pair that genuinely agrees."""
    a = tmp_path / "armA"; a.mkdir()
    b = tmp_path / "armB"; b.mkdir()
    assert _run_guards(tmp_path, outdir=a, n_gpus=4) == 0
    (a / ".pair_dir").write_text(str(b) + "\n")
    assert _run_guards(tmp_path, outdir=b, n_gpus=4, pair_dir=a) == 0

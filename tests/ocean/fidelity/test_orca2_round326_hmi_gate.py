"""Round 326: the hm_i acquisition's reader matches its writer, and its plants fire.

Synthetic records are written in the writer's exact byte layout (names and
order taken from the committed Fortran source), admitted, and each plant the
launcher runs is shown to refuse.  The surface cross-check runs on the real
round-5 surface frames: records carrying those frames' fr_i/taum report 0
unequal points, a perturbed record reports the perturbation.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    nemo_testcase_l4_orca2_round326_hmi_gate as gate,
)

REPO = Path(__file__).resolve().parents[3]
ACQ = REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round326_hmi_acquisition"
WRITER = (ACQ / "zdftke_round326_writer.F90").read_text()
JPI, JPJ = ladder.NX, ladder.NY


def test_writer_source_matches_reader():
    names = re.findall(r"R326_2D\('([^']*)'", WRITER)
    assert all(len(n) == 16 for n in names)
    assert tuple(n.rstrip() for n in names) == gate.EXPECTED_ORDER
    assert "clmagic = '%s'" % gate.MAGIC in WRITER
    assert len(gate.MAGIC) <= 16
    assert int(re.search(r"jp_r326_last = (\d+)", WRITER).group(1)) == gate.N_STEPS
    header = re.search(r"WRITE\(iunit\) 1, kt,(.*?)nn_fsbc", WRITER, re.S).group(0)
    header = re.sub(r"\([^()]*\)", "", header.replace("&", ""))
    assert header.count(",") + 1 == len(gate.HEADER_NAMES)
    assert gate.HALO == ladder.HALO
    patch = (ACQ / "zdftke_round326.patch").read_text()
    assert [l for l in patch.splitlines() if l.startswith("-") and not l.startswith("---")] == []
    assert "+      CALL r326_write( kt, nn_mxlice, ln_mxl0 )" in patch


def _embed(owned: np.ndarray) -> np.ndarray:
    full = np.zeros((JPI, JPJ))
    full[gate.HALO:-gate.HALO, gate.HALO:-gate.HALO] = owned.T
    return full


def _write(path: Path, kt: int, rank: int, fields: dict[str, np.ndarray]) -> None:
    with path.open("wb") as out:
        out.write(gate.MAGIC.ljust(16).encode("ascii"))
        out.write(struct.pack("=12i", 1, kt, rank, 1, 1, JPI, JPJ, 64, 4, 2, 1, 2))
        for name in gate.EXPECTED_ORDER:
            out.write(name.ljust(16).encode("ascii"))
            out.write(struct.pack("=7i", 2, JPI, JPJ, 1, 1, 1, 1))
            out.write(np.asfortranarray(fields[name], dtype="=f8").tobytes(order="F"))


def _root(tmp_path: Path, source=None) -> Path:
    rng = np.random.default_rng(326)
    owned_shape = (JPJ - 2 * gate.HALO, JPI - 2 * gate.HALO)
    hm = {}
    for rank in gate.RANKS:
        for kt in range(1, gate.N_STEPS + 1):
            if kt % 2:
                hm[(rank, kt)] = _embed(rng.uniform(0.0, 3.0, owned_shape))
            else:
                hm[(rank, kt)] = hm[(rank, kt - 1)]
            if source is None:
                fr_i = taum = _embed(rng.uniform(0.0, 1.0, owned_shape))
            else:
                frame = source(kt, rank)
                fr_i, taum = _embed(frame["fr_i"]), _embed(frame["taum"])
            _write(tmp_path / gate.record_name(kt, rank), kt, rank, {
                "hm_i": hm[(rank, kt)], "fr_i": fr_i, "taum": taum,
                "tmask_1": _embed(np.ones(owned_shape))})
    (tmp_path / "producer_commit.txt").write_text("a" * 40 + "\n")
    return tmp_path


def test_synthetic_record_admitted_and_even_steps_reported(tmp_path):
    result = gate.run_gate(_root(tmp_path), expect_commit="a" * 40)
    assert result["status"] == "PASS_R326_HMI_ADMISSION"
    assert result["even_step_identity"]["holds"] is True
    assert len(result["records"]) == 20


@pytest.mark.parametrize("plant", gate.PLANTS)
def test_every_plant_refuses(tmp_path, plant):
    root = _root(tmp_path)
    with pytest.raises(gate.GateError):
        gate.run_gate(root, expect_commit="a" * 40, plant=plant)


def test_missing_frame_refuses(tmp_path):
    root = _root(tmp_path)
    (root / gate.record_name(10, 1)).unlink()
    with pytest.raises(gate.GateError, match="missing record"):
        gate.run_gate(root)


@pytest.mark.skipif(not gate.ROUND5_RUN.is_dir(), reason="round-5 record absent")
def test_surface_cross_check_on_round5_frames(tmp_path):
    def source(kt, rank):
        name = (f"oracle_ocean_surface_input_kt{kt:08d}.bin" if rank == 0 else
                f"oracle_ocean_surface_input_rank{rank:04d}_kt{kt:08d}.bin")
        return ladder.read_surface_fields(gate.ROUND5_RUN / name, kt, rank)

    root = _root(tmp_path, source)
    check = gate.run_gate(root, surface_root=gate.ROUND5_RUN)["surface_cross_check"]
    assert check["fr_i_taum_bit_equal"] is True
    assert sum(r["wet_ice_points"] for r in check["rows"]) > 0
    path = root / gate.record_name(3, 1)
    record = gate.read_record(path)
    fields = dict(record["fields"])
    fields["fr_i"] = fields["fr_i"].copy()
    fields["fr_i"][5, 7] += 1.0e-3
    _write(path, 3, 1, fields)
    check = gate.run_gate(root, surface_root=gate.ROUND5_RUN)["surface_cross_check"]
    assert check["fr_i_taum_bit_equal"] is False
    assert [r["fr_i_unequal"] for r in check["rows"] if (r["rank"], r["kt"]) == (1, 3)] == [1]

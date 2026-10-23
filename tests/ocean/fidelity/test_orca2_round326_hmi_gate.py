"""Round 326: the hm_i acquisition's reader matches its writer, and its plants fire.

Synthetic records are written in the writer's exact byte layout (names and
order taken from the committed Fortran source), admitted, and each plant the
launcher runs is shown to refuse for its own reason.  Admission (A2) runs on
the real round-5 surface frames.  The same-program check is exercised on
synthetic build trees, each planted difference refused.  The committed
ocean.output excerpt is pinned to the real round-5 output.
"""

from __future__ import annotations

import hashlib
import os
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


ROUND5_REQUIRED = os.environ.get("R326_REQUIRE_ROUND5") == "1"
needs_round5 = pytest.mark.skipif(not gate.ROUND5_RUN.is_dir() and not ROUND5_REQUIRED,
                                  reason="round-5 record absent")


def _round5_source(kt, rank):
    name = (f"oracle_ocean_surface_input_kt{kt:08d}.bin" if rank == 0 else
            f"oracle_ocean_surface_input_rank{rank:04d}_kt{kt:08d}.bin")
    return ladder.read_surface_fields(gate.ROUND5_RUN / name, kt, rank)


def test_synthetic_record_structure_only_without_surface(tmp_path):
    result = gate.run_gate(_root(tmp_path), expect_commit="a" * 40)
    assert result["status"] == "PASS_STRUCTURE_ONLY"
    assert result["even_step_identity"]["holds"] is True
    assert result["thickness"]["ice_covered_wet_points_with_hm_i"] > 0
    assert len(result["records"]) == 20


PLANT_REASONS = {
    "header": "nn_mxlice=2", "field-order": "array order", "truncation": "truncated",
    "stamp": "producer commit", "content": "negative hm_i", "hm-zero": "zero on every",
    "hm-thick": ">= 20.0 m", "even-step": "even step", "surface": "round-5 surface",
}


def test_plant_reasons_cover_every_plant():
    assert set(PLANT_REASONS) == set(gate.PLANTS)


@pytest.mark.parametrize("plant", sorted(set(gate.PLANTS) - {"surface"}))
def test_every_plant_refuses(tmp_path, plant):
    root = _root(tmp_path)
    with pytest.raises(gate.GateError, match=re.escape(PLANT_REASONS[plant])):
        gate.run_gate(root, expect_commit="a" * 40, plant=plant)


def test_missing_frame_refuses(tmp_path):
    root = _root(tmp_path)
    (root / gate.record_name(10, 1)).unlink()
    with pytest.raises(gate.GateError, match="missing record"):
        gate.run_gate(root)


@needs_round5
def test_admission_on_round5_frames_and_surface_plants(tmp_path):
    root = _root(tmp_path, _round5_source)
    result = gate.run_gate(root, surface_root=gate.ROUND5_RUN)
    assert result["status"] == "PASS_R326_HMI_ADMISSION"
    assert result["surface_cross_check"]["fr_i_taum_bit_equal"] is True
    with pytest.raises(gate.GateError, match=PLANT_REASONS["surface"]):
        gate.run_gate(root, surface_root=gate.ROUND5_RUN, plant="surface")
    path = root / gate.record_name(3, 1)
    fields = dict(gate.read_record(path)["fields"])
    fields["taum"] = fields["taum"].copy()
    fields["taum"][5, 7] += 1.0e-3
    _write(path, 3, 1, fields)
    with pytest.raises(gate.GateError, match=PLANT_REASONS["surface"]):
        gate.run_gate(root, surface_root=gate.ROUND5_RUN)


PATCH = ACQ / "zdftke_round326.patch"
INSERTED = [l[1:] for l in PATCH.read_text().splitlines() if l.startswith("+") and not l.startswith("+++")]


def _bld(root: Path, cfg: str, observer: list[str], writer: bool) -> Path:
    (root / "ppsrc/nemo").mkdir(parents=True)
    (root / "obj").mkdir()
    for name in gate.BUILD_CONFIG_FILES:
        (root / name).write_text(f"dir::root /cfgs/{cfg}/BLD\n%FCFLAGS -O3\n")
    (root / "ppsrc/nemo/icevar.f90").write_text("hm_i = vt_i * z1_at_i\n")
    (root / "ppsrc/nemo/zdftke.f90").write_text("\n".join(observer) + "\n")
    (root / "obj/icevar.o").write_bytes(b"\x7fELF icevar")
    (root / "obj/zdftke.o").write_bytes(b"\x7fELF zdftke" + bytes(writer))
    if writer:
        (root / "ppsrc/nemo/zdftke_round326_writer.f90").write_text("MODULE w\n")
        (root / "obj/zdftke_round326_writer.o").write_bytes(b"\x7fELF w")
    return root


REF_TKE = ["MODULE zdftke", "   USE zdfmxl", "      CALL tke_tke( a )", "      CALL tke_avn( a )"]
NEW_TKE = REF_TKE[:2] + [INSERTED[0]] + REF_TKE[2:3] + [INSERTED[1]] + REF_TKE[3:]


def _pair(tmp_path):
    ref = _bld(tmp_path / "ref", "R5", REF_TKE, False)
    new = _bld(tmp_path / "new", "R326", NEW_TKE, True)
    return ref, new


def _same(new, ref):
    return gate.same_program(new, ref, PATCH, new_cfg="R326", ref_cfg="R5")


def test_same_program_passes_observer_only(tmp_path):
    ref, new = _pair(tmp_path)
    assert len(INSERTED) == 2
    assert _same(new, ref)["status"] == "PASS_SAME_PROGRAM"


@pytest.mark.parametrize("plant, reason", [
    ("other-source", "ppsrc/nemo/icevar.f90 differs"),
    ("other-object", "obj/icevar.o differs"),
    ("flags", "build config differs: arch_nemo.fcm"),
    ("observer-edit", "zdftke.f90 replaces"),
    ("observer-extra", "zdftke.f90 inserts"),
    ("extra-unit", "new build adds"),
    ("missing-unit", "new build lacks"),
])
def test_same_program_refuses_planted_difference(tmp_path, plant, reason):
    ref, new = _pair(tmp_path)
    if plant == "other-source":
        (new / "ppsrc/nemo/icevar.f90").write_text("hm_i = 2 * vt_i * z1_at_i\n")
    elif plant == "other-object":
        (new / "obj/icevar.o").write_bytes(b"\x7fELF icevar2")
    elif plant == "flags":
        (new / "arch_nemo.fcm").write_text("dir::root /cfgs/R326/BLD\n%FCFLAGS -O2\n")
    elif plant == "observer-edit":
        (new / "ppsrc/nemo/zdftke.f90").write_text("\n".join(NEW_TKE).replace("tke_tke( a )", "tke_tke( b )") + "\n")
    elif plant == "observer-extra":
        (new / "ppsrc/nemo/zdftke.f90").write_text("\n".join(NEW_TKE + ["      x = 1"]) + "\n")
    elif plant == "extra-unit":
        (new / "ppsrc/nemo/extra.f90").write_text("x\n")
    elif plant == "missing-unit":
        (new / "obj/icevar.o").unlink()
    with pytest.raises(gate.GateError, match=re.escape(reason)):
        _same(new, ref)


EXCERPT = ACQ / "round5_ocean.output.excerpt"


@needs_round5
def test_committed_excerpt_is_verbatim_round5_output():
    text = EXCERPT.read_text().splitlines()
    source = gate.ROUND5_RUN / "ocean.output"
    digest = re.search(r"SHA-256: ([0-9a-f]{64})", "\n".join(text)).group(1)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    numbers = [int(n) for n in re.findall(r"\d+", next(l for l in text if l.startswith("# Source lines")))]
    lines = source.read_text().splitlines()
    assert [l for l in text if not l.startswith("#")] == [lines[n - 1] for n in numbers]

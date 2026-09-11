"""Fail-closed guards for the round-41 dyn_adv split record and replay."""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

TESTCASES = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round41_dynadv_split",
    TESTCASES / "nemo_testcase_l2_gyre_round41_dynadv_split.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

NX, NY, NZ = gate.DIMS
HEADER = (1, 1, 3, 1, 2, 3, 3, 0, NX, NY, NZ, 30, 3, 34, 3, 24, 64)


def _arrays() -> dict:
    rng = np.random.default_rng(41)

    def xyz(fill=0.0):
        return np.full((NY, NX, NZ), fill, dtype=np.float64)

    def xy(fill=0.0):
        return np.full((NY, NX), fill, dtype=np.float64)

    a = {name: xyz() for name in gate.EXPECTED if name not in gate.TWO_D | gate.SCALARS}
    a.update({name: xy(1.0) for name in gate.TWO_D})
    a["ln_vortex_force"] = 0.0
    for name in ("e3t_Kmm", "e3u_Kmm", "e3v_Kmm", "e3w_Kmm",
                 "e3t_0", "e3u_0", "e3v_0", "e3w_0"):
        a[name].fill(2.0)
    for name in ("tmask", "umask", "vmask", "wmask"):
        a[name].fill(1.0)
    a["uu_Kmm"][2:-1, 1:-1, :30] = rng.normal(scale=1e-3, size=(23, 34, 30))
    a["vv_Kmm"][1:-1, 2:-1, :30] = rng.normal(scale=1e-3, size=(24, 33, 30))
    a["ww"][1:-1, 1:-1, 1:30] = rng.normal(scale=1e-8, size=(24, 34, 29))
    a["before_keg_u"] = rng.normal(scale=1e-8, size=(NY, NX, NZ))
    a["before_keg_v"] = rng.normal(scale=1e-8, size=(NY, NX, NZ))
    a["after_keg_u"], a["after_keg_v"] = gate._keg_replay(a)
    a["after_zad_u"], a["after_zad_v"] = gate._zad_replay(a)
    return a


def _payload(value: np.ndarray, rank: int) -> bytes:
    if rank == 2:
        return np.asarray(value).T.ravel(order="F").tobytes()
    return np.asarray(value).transpose(1, 0, 2).ravel(order="F").tobytes()


def _write(directory: Path, *, nonzero_wsd=False, truncate=False) -> Path:
    a = _arrays()
    if nonzero_wsd:
        a["wsd_effective"][0, 0, 0] = 1.0
    path = directory / gate.RECORD
    with path.open("wb") as handle:
        handle.write(gate.MAGIC.ljust(16).encode("ascii"))
        handle.write(struct.pack(f"={len(HEADER)}i", *HEADER))
        for index, name in enumerate(gate.EXPECTED):
            handle.write(name.ljust(16).encode("ascii"))
            if name in gate.SCALARS:
                handle.write(struct.pack("=4i", 0, 1, 1, 1))
                raw = np.asarray([a[name]], dtype=np.float64).tobytes()
            elif name in gate.TWO_D:
                handle.write(struct.pack("=4i", 2, NX, NY, 1))
                raw = _payload(a[name], 2)
            else:
                handle.write(struct.pack("=4i", 3, NX, NY, NZ))
                raw = _payload(a[name], 3)
            handle.write(raw[:-8] if truncate and index == 0 else raw)
            if truncate and index == 0:
                break
    return path


def test_reader_and_both_source_replays_are_bit_exact(tmp_path):
    a = gate.read_split(_write(tmp_path))["arrays"]
    keg_u, keg_v = gate._keg_replay(a)
    zad_u, zad_v = gate._zad_replay(a)
    np.testing.assert_array_equal(keg_u, a["after_keg_u"])
    np.testing.assert_array_equal(keg_v, a["after_keg_v"])
    np.testing.assert_array_equal(zad_u, a["after_zad_u"])
    np.testing.assert_array_equal(zad_v, a["after_zad_v"])
    # The calibration plant moves the exact statement row.
    keg_u[2, 2, 0] = np.nextafter(keg_u[2, 2, 0], np.inf)
    assert np.count_nonzero(keg_u != a["after_keg_u"]) == 1


@pytest.mark.parametrize("kwargs, message", [
    ({"truncate": True}, "short payload"),
    ({"nonzero_wsd": True}, "effective wsd is not zero"),
])
def test_reader_refuses_invalid_payloads(tmp_path, kwargs, message):
    with pytest.raises(Exception, match=message):
        gate.read_split(_write(tmp_path, **kwargs))


def test_header_plant_exits_through_the_header_guard(tmp_path):
    with pytest.raises(Exception, match="wrong stage/branch header"):
        gate.read_split(_write(tmp_path), plant_header=True)

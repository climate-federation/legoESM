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
HEADER = (1, 1, 3, 2, 2, 3, 3, 0, NX, NY, NZ, 30, 3, 34, 3, 24, 64)


def _independent_nemo_answers(a: dict) -> tuple[np.ndarray, ...]:
    """Second transcription of compiled dynkeg/dynzad, independent of gate."""
    u, v = a["uu_Kmm"], a["vv_Kmm"]
    keg_u = np.array(a["before_keg_u"], copy=True)
    keg_v = np.array(a["before_keg_v"], copy=True)
    zh = np.zeros((NY, NX, 30), dtype=np.float64)
    zu = u[2:25, 1:34, :30] * u[2:25, 1:34, :30]
    zu = zu + u[2:25, 2:35, :30] * u[2:25, 2:35, :30]
    zv = v[1:24, 2:35, :30] * v[1:24, 2:35, :30]
    zv = zv + v[2:25, 2:35, :30] * v[2:25, 2:35, :30]
    zh[2:25, 2:35, :] = np.float64(0.25) * (zv + zu)
    keg_u[2:24, 2:34, :30] -= (
        (zh[2:24, 3:35, :] - zh[2:24, 2:34, :])
        * a["r1_e1u"][2:24, 2:34, None]
    )
    keg_v[2:24, 2:34, :30] -= (
        (zh[3:25, 2:34, :] - zh[2:24, 2:34, :])
        * a["r1_e2v"][2:24, 2:34, None]
    )

    zad_u, zad_v = np.array(keg_u, copy=True), np.array(keg_v, copy=True)
    carry_u = np.zeros((22, 32), dtype=np.float64)
    carry_v = np.zeros((22, 32), dtype=np.float64)
    for k in range(29):
        center = a["e1e2t"][2:24, 2:34] * a["ww"][2:24, 2:34, k + 1]
        east = a["e1e2t"][2:24, 3:35] * a["ww"][2:24, 3:35, k + 1]
        north = a["e1e2t"][3:25, 2:34] * a["ww"][3:25, 2:34, k + 1]
        next_u = (east + center) * (
            u[2:24, 2:34, k] - u[2:24, 2:34, k + 1])
        next_v = (north + center) * (
            v[2:24, 2:34, k] - v[2:24, 2:34, k + 1])
        scale_u = (np.float64(0.25) * a["r1_e1e2u"][2:24, 2:34]
                   / a["e3u_Kmm"][2:24, 2:34, k])
        scale_v = (np.float64(0.25) * a["r1_e1e2v"][2:24, 2:34]
                   / a["e3v_Kmm"][2:24, 2:34, k])
        zad_u[2:24, 2:34, k] -= scale_u * (carry_u + next_u)
        zad_v[2:24, 2:34, k] -= scale_v * (carry_v + next_v)
        carry_u, carry_v = next_u, next_v
    zad_u[2:24, 2:34, 29] -= (
        np.float64(0.25) * a["r1_e1e2u"][2:24, 2:34]
        / a["e3u_Kmm"][2:24, 2:34, 29] * carry_u
    )
    zad_v[2:24, 2:34, 29] -= (
        np.float64(0.25) * a["r1_e1e2v"][2:24, 2:34]
        / a["e3v_Kmm"][2:24, 2:34, 29] * carry_v
    )
    return keg_u, keg_v, zad_u, zad_v


def _arrays() -> dict:
    rng = np.random.default_rng(41)

    def xyz(fill=0.0):
        return np.full((NY, NX, NZ), fill, dtype=np.float64)

    def xy(fill=0.0):
        return np.full((NY, NX), fill, dtype=np.float64)

    a = {name: xyz() for name in gate.EXPECTED if name not in gate.TWO_D | gate.SCALARS}
    a.update({name: xy(1.0) for name in gate.TWO_D})
    a["ln_vortex_force"] = 0.0
    for name in ("e3t_Kmm", "e3w_Kmm", "e3t_0", "e3u_0", "e3v_0", "e3w_0"):
        a[name].fill(2.0)
    a["e3u_Kmm"][:] = rng.uniform(1.25, 1.75, size=(NY, NX, NZ))
    a["e3v_Kmm"][:] = rng.uniform(1.25, 1.75, size=(NY, NX, NZ))
    for name in ("tmask", "umask", "vmask", "wmask"):
        a[name].fill(1.0)
    a["uu_Kmm"][2:-1, 1:-1, :30] = rng.normal(scale=1e-3, size=(23, 34, 30))
    a["vv_Kmm"][1:-1, 2:-1, :30] = rng.normal(scale=1e-3, size=(24, 33, 30))
    a["ww"][1:-1, 1:-1, 1:30] = rng.normal(scale=1e-8, size=(24, 34, 29))
    a["before_keg_u"] = rng.normal(scale=1e-8, size=(NY, NX, NZ))
    a["before_keg_v"] = rng.normal(scale=1e-8, size=(NY, NX, NZ))
    (a["after_keg_u"], a["after_keg_v"], a["after_zad_u"],
     a["after_zad_v"]) = _independent_nemo_answers(a)
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


def test_flat_four_square_keg_plant_goes_red():
    a = _arrays()
    wrong = np.array(a["before_keg_u"], copy=True)
    u, v = a["uu_Kmm"], a["vv_Kmm"]
    zh = np.zeros((NY, NX, 30), dtype=np.float64)
    zh[2:25, 2:35, :] = np.float64(0.25) * (
        u[2:25, 1:34, :30] ** 2 + u[2:25, 2:35, :30] ** 2
        + v[1:24, 2:35, :30] ** 2 + v[2:25, 2:35, :30] ** 2)
    wrong[2:24, 2:34, :30] -= (
        (zh[2:24, 3:35, :] - zh[2:24, 2:34, :])
        * a["r1_e1u"][2:24, 2:34, None])
    assert np.count_nonzero(wrong != a["after_keg_u"]) > 0


def test_reference_thickness_zad_plant_goes_red():
    a = _arrays()
    planted = dict(a)
    planted["e3u_Kmm"] = a["e3u_0"]
    wrong_u, _ = gate._zad_replay(planted)
    assert np.count_nonzero(wrong_u != a["after_zad_u"]) > 0


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

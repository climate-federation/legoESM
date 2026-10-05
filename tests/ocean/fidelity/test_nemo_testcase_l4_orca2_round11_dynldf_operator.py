"""Direct tests for round 11's ORCA2 lateral-viscosity operator gate.

The gate quotes a disagreement between legoESM's production operator and
NEMO's compiled ``dynldf_lev_lap`` loops, so the transcription itself is
untrusted code until it passes its own controls.  These are those controls:
the index shifts do what their names say, the reciprocal is exact where the
metric is positive, the transcription returns EXACTLY zero on a field whose
divergence and vorticity both vanish (so a gross index error cannot hide), and
each of the two ablations the gate reports actually MOVES the answer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_GATE = (REPO_ROOT / "scripts/validate/ocean_fidelity/orca2_l4"
         / "nemo_testcase_l4_orca2_round11_dynldf_operator_gate.py")


@pytest.fixture(scope="module")
def gate():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "orca2_round11_dynldf_gate", _GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _uniform_mesh(ny=6, nx=8, nz=3, dx=4.0e4, dz=10.0):
    ones2 = np.ones((ny, nx))
    ones3 = np.ones((ny, nx, nz))
    mesh = {name: dx * ones2 for name in
            ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f")}
    for name in ("e3t_0", "e3u_0", "e3v_0", "e3f_0"):
        mesh[name] = dz * ones3
    for name in ("tmask", "umask", "vmask", "fmask"):
        mesh[name] = ones3.copy()
    return mesh


def test_index_shifts_and_reciprocal(gate):
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(gate._xp(a)[:, 0], a[:, 1])
    assert np.array_equal(gate._xp(a)[:, -1], a[:, 0])      # periodic in x
    assert np.array_equal(gate._xm(a)[:, 1], a[:, 0])
    assert np.array_equal(gate._yp(a)[0], a[1])
    assert np.all(np.isnan(gate._yp(a)[-1]))                # no wrap in y
    assert np.array_equal(gate._ym(a)[1], a[0])
    assert np.all(np.isnan(gate._ym(a)[0]))

    metric = np.array([0.0, 2.0, 0.5])
    got = gate._safe_reciprocal(metric)
    assert got[0] == 0.0
    assert np.array_equal(got[1:].view(np.uint64),
                          (1.0 / metric[1:]).view(np.uint64))


def test_transcription_is_exactly_zero_on_an_irrotational_nondivergent_field(
        gate):
    """A spatially CONSTANT velocity on a uniform mesh has zero divergence and
    zero vorticity, so NEMO's operator must return exactly zero -- a gross
    index or sign error in the transcription would not."""
    mesh = _uniform_mesh()
    ny, nx, nz = mesh["tmask"].shape
    u = np.full((ny, nx, nz), 0.37)
    v = np.full((ny, nx, nz), -0.11)
    ssh = np.zeros((ny, nx))
    ahm = np.full((ny, nx, nz), 1.0e4)
    du, dv = gate.nemo_dynldf_lev_lap_rot(mesh, ahm, ahm, u, v, ssh)
    interior = slice(2, ny - 2)
    assert np.all(du[interior] == 0.0)
    assert np.all(dv[interior] == 0.0)


def test_both_ablations_move_the_answer(gate):
    """An ablation that changes nothing proves nothing; each of the gate's two
    must be shown to bite before its magnitude is quoted."""
    mesh = _uniform_mesh()
    ny, nx, nz = mesh["tmask"].shape
    rng = np.random.default_rng(11)
    u = rng.normal(size=(ny, nx, nz))
    v = rng.normal(size=(ny, nx, nz))
    ssh = 0.05 * rng.normal(size=(ny, nx))
    ahm = np.full((ny, nx, nz), 1.0e4)

    base_u, base_v = gate.nemo_dynldf_lev_lap_rot(mesh, ahm, ahm, u, v, ssh)
    # The transcription leaves the rows whose jj neighbours fall outside the
    # owned block as NaN BY DESIGN, so compare only where it is defined.
    def moved(a, b):
        finite = np.isfinite(a) & np.isfinite(b)
        assert finite.any()
        return float(np.max(np.abs(a[finite] - b[finite])))

    extra = np.ones((ny, nx, nz))
    extra[2:4, 3:5] = 0.0
    mask_u, mask_v = gate.nemo_dynldf_lev_lap_rot(
        mesh, ahm, ahm, u, v, ssh, ahmf_extra_mask=extra)
    assert moved(mask_u, base_u) > 0.0
    assert moved(mask_v, base_v) > 0.0

    thick = (mesh["e3u_0"] * 1.5, mesh["e3v_0"] * 1.5, mesh["e3f_0"] * 1.5)
    th_u, th_v = gate.nemo_dynldf_lev_lap_rot(
        mesh, ahm, ahm, u, v, ssh, thickness_override=thick)
    assert moved(th_u, base_u) > 0.0
    assert moved(th_v, base_v) > 0.0


def test_score_refuses_a_one_representable_value_move(gate):
    a = np.linspace(1.0, 2.0, 24).reshape(2, 3, 4)
    weight = np.ones_like(a)
    assert gate.score(a, a, weight)["bit_identical"]
    planted = gate._plant_one_value(a)
    row = gate.score(planted, a, weight)
    assert not row["bit_identical"]
    assert row["unequal"] == 1

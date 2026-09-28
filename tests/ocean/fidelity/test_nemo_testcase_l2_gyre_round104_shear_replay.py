"""Guards for the round-104 shear-production replay.

Everything here runs without a GYRE model step and without the oracle roots:
that the replay reproduces NEMO's compiled association on a synthetic case
whose answer is known by hand, that every operand actually reaches the output
(so no row can pass vacuously through an ignored argument), that an override
of the wrong shape is REFUSED rather than broadcast, and that the operand
mirror's own free-surface ratio is the one NEMO writes.

Scoring the model against NEMO's record needs the record and a full GYRE
step, so that lives in the consolidated stage gate; its numbers are in the
round-104 receipt.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTCASES = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round104_shear_replay",
    TESTCASES / "nemo_testcase_l2_gyre_round104_shear_replay.py",
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

NLAT, NLON, NK = 3, 4, 5          # levels; the shear windows carry NK - 1


def _operands(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.uniform(0.5, 1.5, size=shape)

    return {
        "u_now": r(NLAT, NLON + 1, NK),
        "u_before": r(NLAT, NLON + 1, NK),
        "v_now": r(NLAT + 1, NLON, NK),
        "v_before": r(NLAT + 1, NLON, NK),
        "avm_face_u": r(NLAT, NLON + 1, NK - 1),
        "avm_face_v": r(NLAT + 1, NLON, NK - 1),
        "divisor_u": r(NLAT, NLON + 1, NK - 1),
        "divisor_v": r(NLAT + 1, NLON, NK - 1),
        "wumask": np.ones((NLAT, NLON + 1, NK - 1)),
        "wvmask": np.ones((NLAT + 1, NLON, NK - 1)),
        "coast_u": r(NLAT, NLON, NK - 1),
        "coast_v": r(NLAT, NLON, NK - 1),
    }


def test_replay_is_the_compiled_association_cell_by_cell():
    """Hand-evaluate zdfsh2.f90:99-113 at one cell and demand the same bits."""
    op = _operands()
    out = mod.rebuild_sh2(op)
    assert out.shape == (NLAT, NLON, NK - 1)
    j, i, k = 1, 2, 3

    def zsh2(kind, ii):
        pre, div, msk = (
            ("u", op["divisor_u"], op["wumask"]) if kind == "u"
            else ("v", op["divisor_v"], op["wvmask"]))
        n, b = op[f"{pre}_now"], op[f"{pre}_before"]
        idx = (j, ii, k) if kind == "u" else (ii, i, k)
        vel = (j, ii) if kind == "u" else (ii, i)
        return (op[f"avm_face_{pre}"][idx]
                * (n[vel][k] - n[vel][k + 1])
                * (b[vel][k] - b[vel][k + 1])
                / div[idx] * msk[idx])

    expected = 0.25 * ((zsh2("u", i) + zsh2("u", i + 1)) * op["coast_u"][j, i, k]
                       + (zsh2("v", j) + zsh2("v", j + 1))
                       * op["coast_v"][j, i, k])
    assert out[j, i, k].view(np.uint64) == np.float64(expected).view(np.uint64)


@pytest.mark.parametrize("name", sorted(_operands()))
def test_every_operand_reaches_the_output(name):
    """No row may pass because an argument is silently ignored."""
    op = _operands()
    clean = mod.rebuild_sh2(op)
    bumped = np.asarray(op[name], dtype=np.float64).copy()
    bumped += 0.25
    moved = mod.rebuild_sh2(op, **{name: bumped})
    assert mod._unequal(moved, clean)[0] > 0, f"{name} never reaches p_sh2"


def test_a_wrong_shape_override_is_refused_not_broadcast():
    op = _operands()
    with pytest.raises(mod.ReplayError, match="expected"):
        mod.rebuild_sh2(op, divisor_u=np.ones((NLAT, 1, NK - 1)))


def test_an_unknown_operand_is_refused():
    with pytest.raises(mod.ReplayError, match="unknown"):
        mod.rebuild_sh2(_operands(), not_an_operand=np.zeros((1,)))


def test_mirror_free_surface_ratio_is_nemos_statement():
    """domqco.f90:214-215 on a hand-built mesh, against the mirror."""

    class Z:
        nemo_hu_0 = np.full((NLAT, NLON), 4000.0)
        nemo_hv_0 = np.full((NLAT, NLON), 4000.0)
        nemo_e1e2t = np.full((NLAT, NLON), 1.0e9)
        nemo_e1e2u = np.full((NLAT, NLON), 1.1e9)
        nemo_e1e2v = np.full((NLAT, NLON), 1.2e9)
        nemo_e3w_0 = np.broadcast_to(
            np.arange(1.0, NK), (NLAT, NLON, NK - 1)).copy()

    rng = np.random.default_rng(7)
    eta = rng.uniform(-0.3, 0.3, size=(NLAT, NLON))
    ops = mod.model_operands(
        Z, np.zeros((NLAT, NLON + 1, NK - 1)),
        np.zeros((NLAT + 1, NLON, NK - 1)), eta,
        np.zeros((NLAT, NLON, NK - 2)),
        np.ones((NLAT, NLON + 1, NK - 1)), np.ones((NLAT + 1, NLON, NK - 1)))
    num = 0.5 * (Z.nemo_e1e2t * eta
                 + np.roll(Z.nemo_e1e2t * eta, -1, axis=1))
    r3u = num * (1.0 / Z.nemo_hu_0) / Z.nemo_e1e2u
    ref = Z.nemo_e3w_0[..., 1:]
    ref_u = np.concatenate([ref[:, -1:, :], ref], axis=1)
    r3u_full = np.concatenate([r3u[:, -1:], r3u], axis=1)
    want = (ref_u * (1.0 + r3u_full[..., None])) ** 2
    assert np.array_equal(
        np.asarray(ops["divisor_u"]).view(np.uint64), want.view(np.uint64))


def test_mirror_intermediates_close_to_the_registered_endpoint():
    """The optional statement rows reconstruct the ordinary mirror exactly."""

    class Z:
        nemo_hu_0 = np.full((NLAT, NLON), 4000.0)
        nemo_hv_0 = np.full((NLAT, NLON), 4000.0)
        nemo_e1e2t = np.full((NLAT, NLON), 1.0e9)
        nemo_e1e2u = np.full((NLAT, NLON), 1.1e9)
        nemo_e1e2v = np.full((NLAT, NLON), 1.2e9)
        nemo_e3w_0 = np.broadcast_to(
            np.arange(1.0, NK), (NLAT, NLON, NK - 1)).copy()

    rng = np.random.default_rng(11)
    u = rng.normal(size=(NLAT, NLON + 1, NK - 1))
    v = rng.normal(size=(NLAT + 1, NLON, NK - 1))
    eta = rng.normal(scale=0.1, size=(NLAT, NLON))
    avm = rng.uniform(1.0e-5, 1.0e-3, size=(NLAT, NLON, NK - 2))
    umask = np.ones_like(u)
    vmask = np.ones_like(v)
    operands, rows = mod.model_operands(
        Z, u, v, eta, avm, umask, vmask,
        return_intermediates=True)
    expected = 0.25 * (
        (rows["zsh2u"][:, :-1] + rows["zsh2u"][:, 1:])
        * operands["coast_u"]
        + (rows["zsh2v"][:-1] + rows["zsh2v"][1:])
        * operands["coast_v"])
    assert np.array_equal(
        mod.rebuild_sh2(operands).view(np.uint64),
        expected.view(np.uint64))

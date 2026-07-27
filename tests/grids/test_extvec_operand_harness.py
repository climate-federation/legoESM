"""Self-tests for the ext_vector operand-diff harness (codex fix-advice
rank-2): the comparator's mechanics must be provably non-vacuous
(seeded-violation detection) and the stage-dump hook must not perturb
the production exchange (8-retracted-claims lesson: validate the
instrument before trusting its numbers)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "fv3_native"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cmpmod():
    return _load("compare_extvec_stages")


def test_region_mask_classifies(cmpmod):
    # n=4 face, window Fortran -1..6 (2 halo rings each side)
    reg, ring = cmpmod.region_mask((8, 8), -1, -1, 4)
    assert reg[2, 2] == 0 and ring[2, 2] == 0          # (1,1) interior
    assert reg[0, 3] == 1 and ring[0, 3] == 2          # (-1,2) side halo
    assert reg[0, 0] == 2 and ring[0, 0] == 2          # (-1,-1) corner
    assert reg[7, 7] == 2 and ring[7, 7] == 2          # (6,6) corner
    # staggered axis: j extent 1..n+1
    reg2, _ = cmpmod.region_mask((8, 8), -1, -1, 4, stag_j=1)
    assert reg2[3, 6] == 0                             # (2,5)=n+1 interior


def test_compare_detects_seeded_violation(cmpmod):
    rng = np.random.default_rng(0)
    fort = rng.standard_normal((10, 10)) + 3.0
    ours = fort.copy()
    mx, arg, cov, ncmp = cmpmod.compare("x", ours, fort, -2, -2, 6, 1.0)
    assert mx == 0.0 and cov == 0 and ncmp == 100
    ours2 = fort.copy()
    ours2[1, 5] += 0.5                                 # Fortran (-1, 3)
    mx, arg, cov, _ = cmpmod.compare("x", ours2, fort, -2, -2, 6, 1.0)
    assert mx > 0.01 and arg == (-1, 3)
    # sign handling: ours == -fort must be exact under sign=-1
    mx, _, _, _ = cmpmod.compare("x", -fort, fort, -2, -2, 6, -1.0)
    assert mx == 0.0


def test_compare_coverage_and_masks(cmpmod):
    fort = np.full((10, 10), 2.0)
    ours = np.full((10, 10), 2.0)
    fort[0, 0] = cmpmod.SENT                           # corner (-2,-2)
    ours[0, 0] = np.nan                                # both unwritten: ok
    fort[0, 4] = cmpmod.SENT                           # side halo (-2,2)
    mx, _, cov, _ = cmpmod.compare("x", ours, fort, -2, -2, 6, 1.0)
    assert cov == 1                                    # ours wrote, fort didn't
    # corner exclusion must swallow a corner-only difference
    ours3 = np.full((10, 10), 2.0)
    fort3 = np.full((10, 10), 2.0)
    fort3[9, 9] = 5.0                                  # corner (7,7)
    mx, _, _, _ = cmpmod.compare("x", ours3, fort3, -2, -2, 6, 1.0,
                                 exclude="corners")
    assert mx == 0.0
    mx, _, _, _ = cmpmod.compare("x", ours3, fort3, -2, -2, 6, 1.0)
    assert mx > 0.1


def test_read_fort_roundtrip(cmpmod, tmp_path):
    p = tmp_path / "extvec_S2_ull_t1.dat"
    a = np.arange(6.0).reshape(2, 3) + 0.25
    with open(p, "w") as f:
        f.write(f"{-1:8d}{0:8d}{2:8d}{4:8d}\n")
        for j in range(3):
            for i in range(2):
                f.write(f"{i - 1:8d}{j + 2:8d}{a[i, j]:26.17E}\n")
    b, ilo, jlo = cmpmod.read_fort(p)
    assert (ilo, jlo) == (-1, 2)
    np.testing.assert_allclose(b, a, rtol=0, atol=0)


def test_stage_dump_hook_is_inert_and_complete():
    """Hook ON vs OFF: byte-identical exchanged winds; hook captures
    the full stage inventory (S1-S6, 6 tiles)."""
    from legoesm.core.fv3_native_duo_stepper import (
        analytic_six_face_state,
        build_six_face_duo_context,
    )
    from legoesm.grids.fv3_native_ext_vector import (
        ext_vector_dgrid_sixface,
    )

    ctx = build_six_face_duo_context(12, 3, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)
    states = analytic_six_face_state(ctx)

    def winds():
        return ([np.array(s["u"], copy=True) for s in states],
                [np.array(s["v"], copy=True) for s in states])

    u_a, v_a = winds()
    ext_vector_dgrid_sixface(u_a, v_a, ctx["ectx"])

    got = {}
    ctx["ectx"]["stage_dump"] = (
        lambda st, t, nm, a: got.setdefault(f"{st}_t{t + 1}_{nm}",
                                            np.array(a, copy=True)))
    u_b, v_b = winds()
    ext_vector_dgrid_sixface(u_b, v_b, ctx["ectx"])
    ctx["ectx"].pop("stage_dump")

    for t in range(6):
        np.testing.assert_array_equal(u_a[t], u_b[t])
        np.testing.assert_array_equal(v_a[t], v_b[t])
    for st, nm in (("S1", "uin"), ("S1", "vin"), ("S2", "ull"),
                   ("S2", "vll"), ("S3", "ullp1"), ("S3", "vllp1"),
                   ("S4", "ullp1"), ("S4", "vllp1"), ("S5", "ullp1"),
                   ("S5", "vllp1"), ("S5", "up1"), ("S5", "vp1"),
                   ("S6", "uin"), ("S6", "vin")):
        for t in range(1, 7):
            assert f"{st}_t{t}_{nm}" in got, (st, t, nm)
    # S6 capture must equal the actual returned halos (non-vacuous)
    for t in range(6):
        np.testing.assert_array_equal(got[f"S6_t{t + 1}_uin"], u_b[t])

"""Self-tests for the ext_vector operand-diff harness (codex fix-advice
rank-2): the comparator's mechanics must be provably non-vacuous
(seeded-violation detection, FAIL-LOUD on missing/empty inputs) and the
stage-dump hook must not perturb the production exchange
(8-retracted-claims lesson: validate the instrument before trusting its
numbers)."""

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
    mabs, mrel, arg, cov, ncmp = cmpmod.compare(ours, fort, -2, -2, 6,
                                                1.0)
    assert mabs == 0.0 and mrel == 0.0 and cov == 0 and ncmp == 100
    ours2 = fort.copy()
    ours2[1, 5] += 0.5                                 # Fortran (-1, 3)
    mabs, mrel, arg, cov, _ = cmpmod.compare(ours2, fort, -2, -2, 6,
                                             1.0)
    assert mabs == pytest.approx(0.5) and arg == (-1, 3)
    # sign handling: ours == -fort must be exact under sign=-1
    mabs, _, _, _, _ = cmpmod.compare(-fort, fort, -2, -2, 6, -1.0)
    assert mabs == 0.0


def test_compare_coverage_masks_and_empty(cmpmod):
    fort = np.full((10, 10), 2.0)
    ours = np.full((10, 10), 2.0)
    fort[0, 0] = cmpmod.SENTINELS[0]                   # corner (-2,-2)
    ours[0, 0] = np.nan                                # both unwritten: ok
    fort[0, 4] = cmpmod.SENTINELS[1]                   # side halo (-2,2)
    _, _, _, cov, _ = cmpmod.compare(ours, fort, -2, -2, 6, 1.0)
    assert cov == 1                                    # ours wrote, fort didn't
    # corner exclusion must swallow a corner-only difference
    ours3 = np.full((10, 10), 2.0)
    fort3 = np.full((10, 10), 2.0)
    fort3[9, 9] = 5.0                                  # corner (7,7)
    mabs, _, _, _, _ = cmpmod.compare(ours3, fort3, -2, -2, 6, 1.0,
                                      exclude="corners")
    assert mabs == 0.0
    mabs, _, _, _, _ = cmpmod.compare(ours3, fort3, -2, -2, 6, 1.0)
    assert mabs > 1.0
    # all-unwritten overlap must be an explicit EMPTY marker, never 0.0
    r = cmpmod.compare(np.full((4, 4), np.nan),
                       np.full((4, 4), cmpmod.SENTINELS[0]),
                       1, 1, 6, 1.0)
    assert r == ("EMPTY",)


def test_overlap_crop_windows(cmpmod):
    ours = np.arange(54.0 * 54).reshape(54, 54)        # lo (-2,-2)
    fort = np.zeros((50, 50))                          # window 0..49
    cr = cmpmod.overlap_crop(ours, -2, -2, fort, 0, 0)
    oc, fc, i0, j0 = cr
    assert oc.shape == (50, 50) and (i0, j0) == (0, 0)
    assert oc[0, 0] == ours[2, 2]                      # Fortran (0,0)
    assert cmpmod.overlap_crop(ours, -2, -2, fort, 200, 0) is None


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


def test_main_fatal_on_missing_dumps(cmpmod, tmp_path):
    """FAIL-LOUD gate (codex r1 P0-2): an empty fort dir or a missing
    stage must exit nonzero, never read as clean."""
    np.savez(tmp_path / "ours.npz", n=np.array(4), ng=np.array(3),
             git_sha=np.array("test"))
    rc = cmpmod.main(["--fort-dir", str(tmp_path),
                      "--ours", str(tmp_path / "ours.npz")])
    assert rc == 1


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
    # S6 capture must equal the actual returned halos, BOTH components
    # (non-vacuous; codex r1 flagged uin-only)
    for t in range(6):
        np.testing.assert_array_equal(got[f"S6_t{t + 1}_uin"], u_b[t])
        np.testing.assert_array_equal(got[f"S6_t{t + 1}_vin"], v_b[t])


@pytest.fixture(scope="module")
def twinmod():
    return _load("compare_state_twin")


def test_twin_corner_band_and_crop(twinmod):
    band = twinmod.corner_band_mask(48, 0, 0, 6)
    assert band[0, 0] and band[5, 5] and not band[23, 23]
    assert band[47, 0] and band[42, 5]
    a = np.arange(54.0 * 55).reshape(54, 55)           # u lattice lo (-2,-2)
    c = twinmod.compute_crop(a, -2, -2, 48, 0, 1)
    assert c.shape == (48, 49) and c[0, 0] == a[3, 3]  # Fortran (1,1)


def test_twin_sign_fit_and_fatal(twinmod, tmp_path):
    a = np.random.default_rng(1).standard_normal((8, 8))
    s, res = twinmod.fit_sign(a, -a)
    assert s == -1.0 and res == 0.0
    # end-to-end FATAL on empty inputs (fail-loud, never silent-clean)
    np.savez(tmp_path / "o.npz", n=np.array(4), ng=np.array(3))
    rc = twinmod.main(["--fort-dir", str(tmp_path),
                       "--ours", str(tmp_path / "o.npz")])
    assert rc == 1


def test_twin_detects_seeded_corner_violation(twinmod, tmp_path):
    """Synthetic violation: oracle and ours identical except one
    corner-band cell at block 1 — table must report it in the corner
    column with the right argmax (non-vacuous instrument)."""
    n, ng = 12, 3
    rng = np.random.default_rng(2)
    ours = {"n": np.array(n), "ng": np.array(ng)}
    for b in (0, 1):
        for k, (si, sj) in twinmod.STAG.items():
            for t in range(1, 7):
                a = rng.standard_normal((n + 2 * ng + si,
                                         n + 2 * ng + sj)) + 2.0
                if k == "pt":
                    a = np.ones_like(a)
                ours[f"b{b}_{k}_t{t}"] = a
                # oracle side: identical compute domain, sign +1
                fa = a.copy()
                if b == 1 and k == "u" and t == 3:
                    fa[ng + 1, ng + 1] += 7.0           # Fortran (2,2)
                p = tmp_path / f"dyncore_b{b}_{k}_t{t}.dat"
                with open(p, "w") as f:
                    ilo = jlo = 1 - ng
                    f.write(f"{ilo:8d}{ilo + fa.shape[0] - 1:8d}"
                            f"{jlo:8d}{jlo + fa.shape[1] - 1:8d}\n")
                    for j in range(fa.shape[1]):
                        for i in range(fa.shape[0]):
                            f.write(f"{ilo + i:8d}{jlo + j:8d}"
                                    f"{fa[i, j]:26.17E}\n")
    np.savez(tmp_path / "ours.npz", **ours)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = twinmod.main(["--fort-dir", str(tmp_path),
                           "--ours", str(tmp_path / "ours.npz")])
    out = buf.getvalue()
    assert rc == 0
    # the seeded 7.0 corner-band hit at block 1 / tile 3 / (2,2) must
    # appear in the u row with the corner column carrying it
    row = [ln for ln in out.splitlines()
           if ln.strip().startswith("1") and " u " in f" {ln} "][0]
    assert "7.0000e+00" in row and "(3, 2, 2)" in row
    # block 0 must be clean (identical IC)
    row0 = [ln for ln in out.splitlines()
            if ln.strip().startswith("0") and " u " in f" {ln} "][0]
    assert "0.0000e+00" in row0

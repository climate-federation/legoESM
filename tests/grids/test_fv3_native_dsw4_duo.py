"""Phase-4c — duo d_sw4 per-stage oracle (chained after certified d_sw1).

Certifies ``d_sw4_duo`` (the symmetryclean sw_core.F90:1390-1472
4-corner KE fix) bit-exact (uint64) against the verbatim Fortran
d_sw1 -> d_sw4 chain on the COMMITTED inputs.  The guard
``.not.bounded .or. .not.duogrid`` is always true on the global cube,
so the corner formulas fire on the duo lane, reading u/v and ONLY
duo-written ut/vt cells (no sentinel dependence).  ke is INTENT(INOUT)
(no shim needed), pre-initialised to 1e30 on both sides; the FULL-domain
comparison proves exactly the four corner B-nodes were written.

Fixture ``dsw4_duo_oracle_c12.npz`` from ``scripts/cluster/fv3_native/
dsw4_duo_oracle.sbatch``.
"""

import hashlib
import importlib.util
import os

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
REPO = os.path.join(os.path.dirname(__file__), "..", "..")

SENTINEL = 1.0e30


def _load_gen(name):
    path = os.path.join(REPO, "scripts", "validate", "fv3_native", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def inputs():
    return np.load(os.path.join(FIX, "dswcore_input.npz"),
                   allow_pickle=True)


@pytest.fixture(scope="module")
def oracle():
    return np.load(os.path.join(FIX, "dsw4_duo_oracle_c12.npz"),
                   allow_pickle=True)


def test_input_hash_enforced(oracle):
    gen = _load_gen("gen_dsw1_duo_oracle.py")
    blob, _res, _ng, _dt = gen.serialize_inputs()
    assert hashlib.sha256(blob).hexdigest() == str(oracle["input_sha256"])


_EXTRACT_SHA = {
    "fv3_dsw4_duo_extract.F90": None,  # pinned via the fixture record
    "fv3_dsw1_duo_extract.F90":
        "b91d91462c24deea842cc2cd9bd7ad5fa90e14d760356824ae27d60a7c6b1d15",
}


def test_extract_sha_pinned(oracle):
    for name, want in _EXTRACT_SHA.items():
        path = os.path.join(REPO, "scripts", "validate", "fv3_native", name)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if want is None:
            want = str(oracle["dsw4_extract_sha256"])
        assert got == want, (
            f"{name} drifted from the bytes the fixture certifies "
            f"(got sha256 {got}) — regenerate dsw4_duo_oracle_c12.npz")


def _run_chain(inputs):
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo, d_sw4_duo
    from legoesm.core.fv3_native_sw_core import Bounds

    res, ng = int(inputs["res"]), int(inputs["ng"])
    bd = Bounds.single_tile(res, ng)
    m_a = res + 2 * ng
    gs = {k: np.array(inputs[k]) for k in inputs.files
          if k not in ("res", "ng", "dt")}
    gs.update(bounded_domain=False, grid_type=0,
              sw_corner=True, se_corner=True, ne_corner=True,
              nw_corner=True, da_min=float(inputs["da_min"]),
              da_min_c=float(inputs["da_min_c"]))
    s1 = d_sw1_duo(inputs["delp"], inputs["pt"], inputs["w"],
                   inputs["uc"], inputs["vc"],
                   np.zeros((res + 1, res)), np.zeros((res, res + 1)),
                   np.zeros((res + 1, m_a)), np.zeros((m_a, res + 1)),
                   gs, bd, res + 1, res + 1, dt=float(inputs["dt"]),
                   hord_tr=8, hord_vt=6, hord_tm=6, hord_dp=6,
                   nord_v=1, nord_t=0, damp_v=0.2, damp_t=0.0)
    ke0 = np.full((m_a + 1, m_a + 1), SENTINEL)
    return d_sw4_duo(inputs["u"], inputs["v"], s1["ut"], s1["vt"], ke0,
                     gs, bd, res + 1, res + 1, dt=float(inputs["dt"]))


def test_dsw4_duo_bit_exact(inputs, oracle):
    out = _run_chain(inputs)
    got = np.asarray(out["ke"], dtype=np.float64)
    want = np.asarray(oracle["ke"], dtype=np.float64)
    assert got.shape == want.shape
    nd = int((got.view(np.uint64) != want.view(np.uint64)).sum())
    assert nd == 0, f"ke: {nd}/{want.size} words differ bitwise"


def test_exactly_four_corners_written(inputs, oracle):
    """Non-vacuity: exactly the 4 corner B-nodes differ from the
    sentinel, all finite and nonzero."""
    res, ng = int(inputs["res"]), int(inputs["ng"])
    ke = np.asarray(oracle["ke"], dtype=np.float64)
    npx = res + 1
    written = np.argwhere(ke != SENTINEL)
    corners = {(1 + ng - 1, 1 + ng - 1), (npx + ng - 1, 1 + ng - 1),
               (npx + ng - 1, npx + ng - 1), (1 + ng - 1, npx + ng - 1)}
    assert {tuple(w) for w in written} == corners
    for i, j in corners:
        assert np.isfinite(ke[i, j]) and ke[i, j] != 0.0


def test_fixture_provenance(oracle):
    lin = str(oracle["input_lineage"])
    assert "COMMITTED" in lin and "d_sw4" in lin
    assert "d_sw4:1390-1472:" in str(oracle["auth_block_sha256"])

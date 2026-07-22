"""Phase-4c — duo d_sw5 SINGLE-FACE RAW-KEE chain oracle.

Certifies ``d_sw5_duo`` (the symmetryclean sw_core.F90:1474-1869
divergence-damping + KE + vorticity-transport stage, DUO branch) via
the strongest chain yet: the python side runs the CERTIFIED ports
d_sw1 -> d_sw3 -> raw-KEE assembly (the dyn_core inline loops on the
UNEXCHANGED single-face arrays) -> d_sw4 -> d_sw5 and every
d_sw5 output token must be bit-exact (uint64) against the verbatim
Fortran chain on the COMMITTED inputs.  d_sw2 is skipped on both sides
(no d_sw5-lane effect at hydrostatic/damp_w=0).  SCOPE (codex
dsw456-r1 P1): the kee here is the RAW single-face assembly — the
authoritative dyn_core BGRID_NE-exchanges and 0.5-averages
ubb/vbbtemp edges first (dyn_core.F90:968-1020; needs neighbor faces,
not an identity on a real cube); that averaging is excluded six-face
infrastructure, so this is a translation certificate of the stage
composition, not the integrated pipeline.

Sentinel contracts (extract intent shims, defined semantics): ptc
unwritten on nord=1; ub/vb untouched at d_con=0; delpc halo + ke halo
never written — all dumped/compared at 1e30.  uc/vc are CLOBBERED as
divergence-gradient workspaces (auth semantics) and compared over the
regions the n-loop writes plus the untouched remainder.

Fixture ``dsw5_duo_oracle_c12.npz`` from ``scripts/cluster/fv3_native/
dsw5_duo_oracle.sbatch``.
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
    return np.load(os.path.join(FIX, "dsw5_duo_oracle_c12.npz"),
                   allow_pickle=True)


def test_input_hash_enforced(oracle):
    gen = _load_gen("gen_dsw1_duo_oracle.py")
    blob, _res, _ng, _dt = gen.serialize_inputs()
    assert hashlib.sha256(blob).hexdigest() == str(oracle["input_sha256"])


# every extract the dsw5 sbatch compiles (codex dsw456-r1 P2 manifest)
_EXTRACT_SHA = {
    "fv3_dsw5_duo_extract.F90": None,  # pinned via the fixture record
    "fv3_dsw1_duo_extract.F90":
        "b91d91462c24deea842cc2cd9bd7ad5fa90e14d760356824ae27d60a7c6b1d15",
    "fv3_tpcore_duo_extract.F90":
        "71f930b3b3b6291955259e46b7dfcd8bcba5c500206c9b0a5377f24c8f8d99c9",
    "fv3_dsw3_duo_extract.F90":
        "9cf8df3d65383a8de987b7c92acc8a2637cb4b4013470896beba5e9166e94ac2",
    "fv3_dsw4_duo_extract.F90":
        "482d908d1d94a644caf51a2481981254ebacc32c04af6d676d028a90363c8f70",
}


def test_extract_sha_pinned(oracle):
    for name, want in _EXTRACT_SHA.items():
        path = os.path.join(REPO, "scripts", "validate", "fv3_native", name)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if want is None:
            want = str(oracle["dsw5_extract_sha256"])
        assert got == want, (
            f"{name} drifted from the bytes the fixture certifies "
            f"(got sha256 {got}) — regenerate dsw5_duo_oracle_c12.npz")


def _run_chain(inputs, *, nord=1, hord_vt=6, dddmp=0.2):
    from legoesm.core.fv3_native_duo_sw_core import (
        d_sw1_duo,
        d_sw3_duo,
        d_sw4_duo,
        d_sw5_duo,
    )
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
    dt = float(inputs["dt"])

    s1 = d_sw1_duo(inputs["delp"], inputs["pt"], inputs["w"],
                   inputs["uc"], inputs["vc"],
                   np.zeros((res + 1, res)), np.zeros((res, res + 1)),
                   np.zeros((res + 1, m_a)), np.zeros((m_a, res + 1)),
                   gs, bd, res + 1, res + 1, dt=dt,
                   hord_tr=8, hord_vt=6, hord_tm=6, hord_dp=6,
                   nord_v=1, nord_t=0, damp_v=0.2, damp_t=0.0)
    s3 = d_sw3_duo(inputs["u"], inputs["v"], inputs["uc"], inputs["vc"],
                   gs, bd, res + 1, res + 1, dt=dt, hord_mt=6)

    # RAW kee assembly — dyn_core inline loops, unexchanged (see SCOPE)
    ke = np.full((m_a + 1, m_a + 1), SENTINEL)
    ring = slice(ng, ng + res + 1)
    kee = s3["ubbtemp"] * s3["vbbtemp"]
    ke[ring, ring] = 0.5 * (kee + s3["ubb"] * s3["vbb"])

    s4 = d_sw4_duo(inputs["u"], inputs["v"], s1["ut"], s1["vt"], ke,
                   gs, bd, res + 1, res + 1, dt=dt)

    return d_sw5_duo(inputs["delp"], inputs["u"], inputs["v"],
                     inputs["uc"], inputs["vc"], inputs["ua"],
                     inputs["va"], inputs["divg_d_in"],
                     s1["crx_adv"], s1["cry_adv"],
                     s1["xfx_adv"], s1["yfx_adv"],
                     s1["ra_x"], s1["ra_y"], s4["ke"],
                     gs, bd, res + 1, res + 1, dt=dt,
                     hord_vt=hord_vt, nord=nord, dddmp=dddmp,
                     d2_bg=0.0, d4_bg=0.12, d_con=0.0)


@pytest.fixture(scope="module")
def oracle_n2():
    path = os.path.join(FIX, "dsw5_duo_oracle_c12_n2.npz")
    if not os.path.exists(path):
        pytest.skip("nord=2 fixture not generated yet "
                    "(dsw5_duo_oracle.sbatch --n2 lane)")
    return np.load(path, allow_pickle=True)


def test_dsw5_duo_nord2_bit_exact(inputs, oracle_n2):
    """case-8 damping lane: nord=2 (del-6) + hord_vt=8 + dddmp=0,
    every output token BIT-exact vs the verbatim Fortran chain (the
    n-loop generalisation and the hord-8 vorticity transport branch
    certified together)."""
    assert int(oracle_n2["nord"]) == 2 and int(oracle_n2["hord_vt"]) == 8
    out = _run_chain(inputs, nord=2, hord_vt=8, dddmp=0.0)
    for key in ("delpc", "ptc", "wk", "divg_d", "ke", "uc", "ut",
                "vc", "vt", "vortfluxx", "vortfluxy", "ub", "vb"):
        got = np.asarray(out[key], dtype=np.float64)
        want = np.asarray(oracle_n2[key], dtype=np.float64)
        assert got.shape == want.shape, (key, got.shape, want.shape)
        nd = int((got.view(np.uint64) != want.view(np.uint64)).sum())
        assert nd == 0, f"{key}: {nd}/{want.size} words differ bitwise"
    # non-vacuity: nord=2 must genuinely differ from the nord=1 chain
    out1 = _run_chain(inputs)
    assert not np.array_equal(np.asarray(out["ke"]),
                              np.asarray(out1["ke"]))


def test_dsw5_duo_bit_exact(inputs, oracle):
    """Every d_sw5 output token BIT-exact (uint64) vs the verbatim
    Fortran single-face chain, sentinel regions included."""
    out = _run_chain(inputs)
    for key in ("delpc", "ptc", "wk", "divg_d", "ke", "uc", "ut",
                "vc", "vt", "vortfluxx", "vortfluxy", "ub", "vb"):
        got = np.asarray(out[key], dtype=np.float64)
        want = np.asarray(oracle[key], dtype=np.float64)
        assert got.shape == want.shape, (key, got.shape, want.shape)
        nd = int((got.view(np.uint64) != want.view(np.uint64)).sum())
        assert nd == 0, f"{key}: {nd}/{want.size} words differ bitwise"


def test_outputs_discriminated(inputs, oracle):
    """Non-vacuity: the damping increment reached ke — recompute the
    raw kee through the certified d_sw3 port and assert the oracle ke
    DIFFERS on interior B-nodes (beyond the 4 d_sw4 corners); divg_d
    changed from the input; wk/vortflux finite+nonzero; ptc/ub/vb
    all-sentinel."""
    from legoesm.core.fv3_native_duo_sw_core import d_sw3_duo
    from legoesm.core.fv3_native_sw_core import Bounds

    res, ng = int(inputs["res"]), int(inputs["ng"])
    ring = slice(ng, ng + res + 1)
    ke = np.asarray(oracle["ke"], dtype=np.float64)
    assert np.isfinite(ke[ring, ring]).all()
    bd = Bounds.single_tile(res, ng)
    gs = {k: np.array(inputs[k]) for k in inputs.files
          if k not in ("res", "ng", "dt")}
    gs.update(bounded_domain=False, grid_type=0,
              sw_corner=True, se_corner=True, ne_corner=True,
              nw_corner=True, da_min=float(inputs["da_min"]),
              da_min_c=float(inputs["da_min_c"]))
    s3 = d_sw3_duo(inputs["u"], inputs["v"], inputs["uc"], inputs["vc"],
                   gs, bd, res + 1, res + 1,
                   dt=float(inputs["dt"]), hord_mt=6)
    raw_kee = 0.5 * (s3["ubbtemp"] * s3["vbbtemp"]
                     + s3["ubb"] * s3["vbb"])
    interior = ke[ring, ring][1:-1, 1:-1]      # excludes d_sw4 corners
    raw_int = raw_kee[1:-1, 1:-1]
    assert (interior != raw_int).any(), (
        "d_sw5 ke increment did not move interior B-nodes off raw kee")
    dd = np.asarray(oracle["divg_d"], dtype=np.float64)
    assert not np.array_equal(dd[ring, ring],
                              np.asarray(inputs["divg_d_in"])[ring, ring])
    for k in ("wk", "vortfluxx", "vortfluxy"):
        a = np.asarray(oracle[k], dtype=np.float64)
        assert np.isfinite(a).all(), k
        assert np.abs(a).max() > 0.0, k
    for k in ("ptc", "ub", "vb"):
        assert (np.asarray(oracle[k], dtype=np.float64) == SENTINEL).all(), k


def test_fixture_provenance(oracle):
    lin = str(oracle["input_lineage"])
    assert "COMMITTED" in lin and "d_sw5" in lin and "d_sw2 skipped" in lin
    auth = str(oracle["auth_block_sha256"])
    for tag in ("params:36-63:", "a2b_ord4:a2b_edge:50-330:",
                "d_sw5:1474-1869:", "smag_corner:2451-2537:"):
        assert tag in auth, tag

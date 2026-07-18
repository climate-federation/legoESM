"""Phase-4c — duo d_sw2 per-stage oracle (chained after certified d_sw1).

Certifies ``d_sw2_duo`` (the symmetryclean sw_core.F90:1000-1199 d_sw2
post-averaging UPDATE stage) bit-exact (uint64) against the verbatim
Fortran chain d_sw1 -> d_sw2 on the COMMITTED phase-4b inputs.  On the
oracle lane (hydrostatic, inline_q=F, no SW_DYNAMICS/USE_COND) d_sw2
reduces to: heat_source zeroing + the delp/pt else-arm update from
allflux slots 1 (delp fx/fy) and 4 (pt gx/gy):

    pt   = pt*delp + div(g)*rarea
    delp = delp    + div(f)*rarea
    pt   = pt / delp

ptc (INTENT(OUT)) and dw (INTENT(OUT), non-hydro only) are NEVER written
on this lane — the driver initialises them to 1e30 and the dump proves
they stay sentinel (scratch semantics, mirrored by the port).

The single-tile chain is well-posed because dyn_core's inter-panel flux
AVERAGING between d_sw1 and d_sw2 is the identity when both stages run
on the same tile's raw fluxes (the averaging analog is separate 6-face
infrastructure); delp/pt reach d_sw2 unmutated by duo d_sw1
(copy_corners early-returns, so fv_tp_2d never rewrites corner ghosts).

Fixture ``dsw2_duo_oracle_c12.npz`` from ``scripts/cluster/fv3_native/
dsw2_duo_oracle.sbatch``; input hash enforced against the committed
``dswcore_input.npz`` exactly as the d_sw1 oracle does.
"""

import hashlib
import importlib.util
import os

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
REPO = os.path.join(os.path.dirname(__file__), "..", "..")


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
    return np.load(os.path.join(FIX, "dsw2_duo_oracle_c12.npz"),
                   allow_pickle=True)


def test_input_hash_enforced(oracle):
    """input_sha256 must equal the committed input npz's canonical
    serialisation — via the SHARED d_sw1 serializer (single writer)."""
    gen = _load_gen("gen_dsw1_duo_oracle.py")
    blob, _res, _ng, _dt = gen.serialize_inputs()
    assert hashlib.sha256(blob).hexdigest() == str(oracle["input_sha256"])


# Drift guard: the fixture certifies THESE extract bytes (chain =
# tpcore + d_sw1 + d_sw2 extracts).  Intended deviations from the
# authoritative blocks are the intent shims documented in each extract
# header (ut/vt in d_sw1; dw/ptc in d_sw2).  Any edit must regenerate
# the fixture via the sbatch and re-pin.
_EXTRACT_SHA = {
    "fv3_dsw1_duo_extract.F90":
        "b91d91462c24deea842cc2cd9bd7ad5fa90e14d760356824ae27d60a7c6b1d15",
    "fv3_dsw2_duo_extract.F90":
        "c34a3d638f4e6870201540782e13c129d66ea07b4f3ce55201b103e2a7a41ef4",
    "fv3_tpcore_duo_extract.F90":
        "71f930b3b3b6291955259e46b7dfcd8bcba5c500206c9b0a5377f24c8f8d99c9",
}


def test_extract_sha_pinned():
    for name, want in _EXTRACT_SHA.items():
        path = os.path.join(REPO, "scripts", "validate", "fv3_native", name)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert got == want, (
            f"{name} drifted from the bytes the fixture certifies "
            f"(got sha256 {got}) — regenerate dsw2_duo_oracle_c12.npz "
            f"via the sbatch and re-pin")


def _run_chain(inputs):
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo, d_sw2_duo
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
    s2 = d_sw2_duo(s1["delp"], s1["pt"], s1["allflux_x"],
                   s1["allflux_y"], gs, bd)
    return s2


def test_dsw2_duo_bit_exact(inputs, oracle):
    """delp/pt AFTER d_sw2 (FULL data domain: compute update + untouched
    halo strips), the zeroed heat_source, and the sentinel-untouched
    ptc/dw are BIT-exact (uint64) vs the Fortran d_sw1->d_sw2 chain."""
    out = _run_chain(inputs)
    arrs = {"delp2": out["delp"], "pt2": out["pt"],
            "heat": out["heat_source"], "ptc2": out["ptc"],
            "dw": out["dw"]}
    for key, got in arrs.items():
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(oracle[key], dtype=np.float64)
        assert got.shape == want.shape, (key, got.shape, want.shape)
        nd = int((got.view(np.uint64) != want.view(np.uint64)).sum())
        assert nd == 0, f"{key}: {nd}/{want.size} words differ bitwise"


def test_update_is_discriminated(inputs, oracle):
    """Non-vacuity: the compute-domain delp/pt actually CHANGED from the
    inputs (a port that returns its inputs unmodified must fail), and
    heat_source is exactly zero (the zeroing branch fired)."""
    ng = int(inputs["ng"])
    res = int(inputs["res"])
    sl = slice(ng, ng + res)
    assert not np.array_equal(np.asarray(oracle["delp2"])[sl, sl],
                              np.asarray(inputs["delp"])[sl, sl])
    assert not np.array_equal(np.asarray(oracle["pt2"])[sl, sl],
                              np.asarray(inputs["pt"])[sl, sl])
    assert (np.asarray(oracle["heat"]) == 0.0).all()
    assert (np.asarray(oracle["ptc2"]) == 1.0e30).all()
    assert (np.asarray(oracle["dw"]) == 1.0e30).all()


def test_fixture_provenance(oracle):
    lin = str(oracle["input_lineage"])
    assert "COMMITTED" in lin and "d_sw1 -> d_sw2" in lin

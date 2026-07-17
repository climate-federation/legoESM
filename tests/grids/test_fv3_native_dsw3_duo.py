"""Phase-4c — duo d_sw3 per-stage oracle (chained after certified d_sw1).

Certifies ``d_sw3_duo`` (the symmetryclean sw_core.F90:1201-1388 d_sw3
KE-flux stage, DUO branch) bit-exact (uint64) against the verbatim
Fortran chain d_sw1 -> d_sw3 on the COMMITTED phase-4b inputs.  The duo
lane: unclamped B-grid ranges + the INTERIOR contravariant vb/ub
formulas everywhere (the vt/ut edge + corner extrapolations live in the
skipped non-duo else, so d_sw1's ut/vt are never read — every input is
fully defined, no workspace sentinel involved), then ytp_v/xtp_u with
``bounded_domain=.false.`` passed as a LITERAL (upstream quirk) and the
symmetryclean ``dg%is_initialized`` gates active (port: duogrid=True on
the certified plain ytp_v/xtp_u — range gate + the WMP smt5/smt6
edge-fix blocks the symmetryclean tree REMOVED).

Outputs: ubbtemp/vbbtemp (post-ytp_v ub + pre-xtp_u vb) and ubb/vbb
(final) on the B-grid compute ring (is:ie+1, js:je+1) — the arrays
dyn_core carries to d_sw5's KE assembly.

Fixture ``dsw3_duo_oracle_c12.npz`` from ``scripts/cluster/fv3_native/
dsw3_duo_oracle.sbatch``; input hash enforced against the committed
``dswcore_input.npz`` via the SHARED d_sw1 serializer.
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
    return np.load(os.path.join(FIX, "dsw3_duo_oracle_c12.npz"),
                   allow_pickle=True)


def test_input_hash_enforced(oracle):
    gen = _load_gen("gen_dsw1_duo_oracle.py")
    blob, _res, _ng, _dt = gen.serialize_inputs()
    assert hashlib.sha256(blob).hexdigest() == str(oracle["input_sha256"])


# Drift guard: the fixture certifies THESE extract bytes (the d_sw3
# extract bundles verbatim d_sw3 + the symmetryclean xtp_u/ytp_v).
_EXTRACT_SHA = {
    "fv3_dsw3_duo_extract.F90": None,  # pinned by the sbatch run below
    "fv3_dsw1_duo_extract.F90":
        "b91d91462c24deea842cc2cd9bd7ad5fa90e14d760356824ae27d60a7c6b1d15",
    "fv3_tpcore_duo_extract.F90":
        "71f930b3b3b6291955259e46b7dfcd8bcba5c500206c9b0a5377f24c8f8d99c9",
}


def test_extract_sha_pinned(oracle):
    for name, want in _EXTRACT_SHA.items():
        path = os.path.join(REPO, "scripts", "validate", "fv3_native", name)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if want is None:
            want = str(oracle["dsw3_extract_sha256"])
        assert got == want, (
            f"{name} drifted from the bytes the fixture certifies "
            f"(got sha256 {got}) — regenerate dsw3_duo_oracle_c12.npz "
            f"via the sbatch and re-pin")


def _run_chain(inputs):
    from legoesm.core.fv3_native_duo_sw_core import d_sw3_duo
    from legoesm.core.fv3_native_sw_core import Bounds

    res, ng = int(inputs["res"]), int(inputs["ng"])
    bd = Bounds.single_tile(res, ng)
    gs = {k: np.array(inputs[k]) for k in inputs.files
          if k not in ("res", "ng", "dt")}
    gs.update(bounded_domain=False, grid_type=0,
              sw_corner=True, se_corner=True, ne_corner=True,
              nw_corner=True, da_min=float(inputs["da_min"]),
              da_min_c=float(inputs["da_min_c"]))
    return d_sw3_duo(inputs["u"], inputs["v"], inputs["uc"], inputs["vc"],
                     gs, bd, res + 1, res + 1, dt=float(inputs["dt"]),
                     hord_mt=6)


def test_dsw3_duo_bit_exact(inputs, oracle):
    """ubbtemp/vbbtemp/ubb/vbb BIT-exact (uint64) vs the verbatim
    Fortran d_sw3 duo stage over the full B-grid compute ring."""
    out = _run_chain(inputs)
    for key in ("ubbtemp", "vbbtemp", "ubb", "vbb"):
        got = np.asarray(out[key], dtype=np.float64)
        want = np.asarray(oracle[key], dtype=np.float64)
        assert got.shape == want.shape, (key, got.shape, want.shape)
        nd = int((got.view(np.uint64) != want.view(np.uint64)).sum())
        assert nd == 0, f"{key}: {nd}/{want.size} words differ bitwise"


def test_outputs_discriminated(oracle):
    """Non-vacuity: all four outputs finite, nonzero, and pairwise
    distinct (ubbtemp==ubb would mean xtp_u's vb overwrite or the ub
    rebuild was skipped; vbbtemp==vbb likewise)."""
    a = {k: np.asarray(oracle[k], dtype=np.float64)
         for k in ("ubbtemp", "vbbtemp", "ubb", "vbb")}
    for k, arr in a.items():
        assert np.isfinite(arr).all(), k
        assert np.abs(arr).max() > 0.0, k
    assert not np.array_equal(a["ubbtemp"], a["ubb"])
    assert not np.array_equal(a["vbbtemp"], a["vbb"])


def test_fixture_provenance(oracle):
    lin = str(oracle["input_lineage"])
    assert "COMMITTED" in lin and "d_sw3" in lin

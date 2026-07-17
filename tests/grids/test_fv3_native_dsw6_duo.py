"""Phase-4c — duo d_sw6 FULL-CHAIN oracle (the complete d_sw pipeline).

Certifies ``d_sw6_duo`` (the symmetryclean sw_core.F90:1871-2006 final
circulation-form wind update + damp_v del6 vorticity damping) via the
complete single-tile duo chain: the python side runs the CERTIFIED
ports d_sw1 -> d_sw3 -> kee -> d_sw4 -> d_sw5 -> d_sw6 and the final
winds must be bit-exact (uint64) against the verbatim Fortran chain on
the COMMITTED inputs.  With this stage, every d_sw sub-stage of the
symmetryclean duo pipeline is certified end-to-end on one tile
(d_sw2's delp/pt update certified separately; the two mpp averaging
sites are the single-tile identity).

Sentinels: ub/vb untouched at d_con=0; heat_source stays 1e30 (d_sw2,
which zeroes it in the real pipeline, is not in this chain).  ut/vt
compared post-del6 (CLOBBERED as the damping flux outputs).

Fixture ``dsw6_duo_oracle_c12.npz`` from ``scripts/cluster/fv3_native/
dsw6_duo_oracle.sbatch``.
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
    return np.load(os.path.join(FIX, "dsw6_duo_oracle_c12.npz"),
                   allow_pickle=True)


def test_input_hash_enforced(oracle):
    gen = _load_gen("gen_dsw1_duo_oracle.py")
    blob, _res, _ng, _dt = gen.serialize_inputs()
    assert hashlib.sha256(blob).hexdigest() == str(oracle["input_sha256"])


_EXTRACT_SHA = {
    "fv3_dsw6_duo_extract.F90": None,  # pinned via the fixture record
    "fv3_dsw1_duo_extract.F90":
        "b91d91462c24deea842cc2cd9bd7ad5fa90e14d760356824ae27d60a7c6b1d15",
}


def test_extract_sha_pinned(oracle):
    for name, want in _EXTRACT_SHA.items():
        path = os.path.join(REPO, "scripts", "validate", "fv3_native", name)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if want is None:
            want = str(oracle["dsw6_extract_sha256"])
        assert got == want, (
            f"{name} drifted from the bytes the fixture certifies "
            f"(got sha256 {got}) — regenerate dsw6_duo_oracle_c12.npz")


def _run_chain(inputs):
    from legoesm.core.fv3_native_duo_sw_core import (
        d_sw1_duo,
        d_sw3_duo,
        d_sw4_duo,
        d_sw5_duo,
        d_sw6_duo,
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
    ke = np.full((m_a + 1, m_a + 1), SENTINEL)
    ring = slice(ng, ng + res + 1)
    kee = s3["ubbtemp"] * s3["vbbtemp"]
    ke[ring, ring] = 0.5 * (kee + s3["ubb"] * s3["vbb"])
    s4 = d_sw4_duo(inputs["u"], inputs["v"], s1["ut"], s1["vt"], ke,
                   gs, bd, res + 1, res + 1, dt=dt)
    s5 = d_sw5_duo(inputs["delp"], inputs["u"], inputs["v"],
                   inputs["uc"], inputs["vc"], inputs["ua"],
                   inputs["va"], inputs["divg_d_in"],
                   s1["crx_adv"], s1["cry_adv"],
                   s1["xfx_adv"], s1["yfx_adv"],
                   s1["ra_x"], s1["ra_y"], s4["ke"],
                   gs, bd, res + 1, res + 1, dt=dt,
                   hord_vt=6, nord=1, dddmp=0.2, d2_bg=0.0,
                   d4_bg=0.12, d_con=0.0)
    return d_sw6_duo(inputs["u"], inputs["v"], s5["ut"], s5["vt"],
                     s5["ke"], s5["wk"], s5["vortfluxx"],
                     s5["vortfluxy"], gs, bd, res + 1, res + 1,
                     nord_v=1, damp_v=0.2, d_con=0.0)


def test_dsw6_duo_bit_exact(inputs, oracle):
    """Final u/v (full domain: compute update + input halo strips), the
    del6-clobbered ut/vt, and the sentinel ub/vb/heat_source — all
    BIT-exact (uint64) vs the verbatim Fortran complete chain."""
    out = _run_chain(inputs)
    keymap = {"u": "u", "v": "v", "ut": "ut", "vt": "vt",
              "ub": "ub", "vb": "vb", "heat": "heat_source"}
    for okey, pkey in keymap.items():
        got = np.asarray(out[pkey], dtype=np.float64)
        want = np.asarray(oracle[okey], dtype=np.float64)
        assert got.shape == want.shape, (okey, got.shape, want.shape)
        nd = int((got.view(np.uint64) != want.view(np.uint64)).sum())
        assert nd == 0, f"{okey}: {nd}/{want.size} words differ bitwise"


def test_outputs_discriminated(inputs, oracle):
    """Non-vacuity: compute-domain u/v CHANGED from the inputs; the
    del6 diffusive add is live (u would differ if the final += vt were
    dropped — vt nonzero on the compute rows); ub/vb/heat sentinel."""
    ng = int(inputs["ng"])
    res = int(inputs["res"])
    su = (slice(ng, ng + res), slice(ng, ng + res + 1))
    sv = (slice(ng, ng + res + 1), slice(ng, ng + res))
    assert not np.array_equal(np.asarray(oracle["u"])[su],
                              np.asarray(inputs["u"])[su])
    assert not np.array_equal(np.asarray(oracle["v"])[sv],
                              np.asarray(inputs["v"])[sv])
    vt = np.asarray(oracle["vt"], dtype=np.float64)
    assert np.abs(vt[su]).max() > 0.0
    for k in ("ub", "vb", "heat"):
        assert (np.asarray(oracle[k], dtype=np.float64) == SENTINEL).all(), k


def test_fixture_provenance(oracle):
    lin = str(oracle["input_lineage"])
    assert "COMMITTED" in lin and "d_sw6" in lin
    auth = str(oracle["auth_block_sha256"])
    for tag in ("d_sw6:1871-2006:", "del6_vt_flux:2008-2121:"):
        assert tag in auth, tag

"""Brick ``geopk_pgrad`` — multilevel hydrostatic pressure-chain oracle.

WHAT IS CERTIFIED.  The python port in
``legoesm.core.fv3_native_pgrad`` reproduces, BIT FOR BIT (uint64 words,
zero tolerance, sentinel regions included), the verbatim Fortran chain

    geopk(CG=.true. , computehalo=.false.)   dyn_core.F90:2660-2790 @ :533
      -> p_grad_c                            dyn_core.F90:2073-2132 @ :629
      -> geopk(CG=.false., computehalo=.true.)                      @ :1401
        -> one_grad_p                        dyn_core.F90:2347-2480 @ :1531

plus ``a2b_edge.F90:50-330`` (a2b_ord4, reused from the CERTIFIED
``a2b_edge_duo_mod``) and ``a2b_edge.F90:332-453`` (a2b_ord2, linked but
DEAD on this lane), on the authoritative Zenodo 8327578 *symmetryclean*
sources, at C12 with km=2 AND km=3.

SCOPE / EXCLUSIONS.  This is a ROUTINE-TRANSLATION CERTIFICATE on ONE
FACE.  Every inter-stage and cross-face quantity (delpc/ptc/uc/vc, the
D-grid delp/pt/u/v, hs, divg2) arrives as EXPLICIT serialized input, so
the gate certifies NEITHER the six-face exchange cadence NOR the
``ext_scalar`` schedule — those stay with the existing system gates.
Build lane: NO ``-DSW_DYNAMICS``, NO ``-DUSE_COND`` (dry), hydrostatic
only, ``beta <= 0``, ``a2b_ord = 4``, duogrid, ``d_ext > 0``.
``ptop``/``akap``/``cp_air`` are HEADER values fed identically to both
sides (UNCERTAIN U1/U6: FMS ``constants_mod`` is absent from the Zenodo
tree, so they are TEST values, NOT production-fidelity claims).

SENTINEL CONTRACTS.  ``1e30`` marks a never-written OUTPUT region and is
DUMPED AND COMPARED, so the write window is itself certified (geopk's
pk/gz/pe/peln/pkz are ``intent(OUT)`` upstream and only partially
written; the extract's documented intent shims make the round-trip
defined).  ``-9.e9`` marks a poisoned INPUT: the driver re-runs
``p_grad_c`` with ``delpc = -9.e9`` and ``one_grad_p`` with
``delp = -9.e9`` from restored pristine state, and the ``*_DPPOISON``
dumps must be BITWISE identical to the clean ones — that is what proves
the hydrostatic branch never reads them.

FIXTURES are produced by ``scripts/cluster/fv3_native/
geopk_pgrad_oracle.sbatch`` (NEVER hand-written) into
``tests/grids/fixtures/geopk_pgrad_oracle_c12_km{2,3}.npz``.

The transcendental probe ``test_a_transcendental_agreement_probe`` is
defined and RUNS FIRST on purpose: ``geopk`` is the first brick in this
campaign with ``log``/``exp`` in a bit-exact gate, and a
gfortran-vs-NumPy libm disagreement is an ENVIRONMENT result that must
be attributed before any bit-exact comparison below is read as a port
defect.
"""

import hashlib
import importlib.util
import os

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
REPO = os.path.join(os.path.dirname(__file__), "..", "..")
VALID = os.path.join(REPO, "scripts", "validate", "fv3_native")

SENTINEL = 1.0e30
POISON = -9.0e9
KMS = (2, 3)

# Every Fortran file the geopk_pgrad sbatch compiles.  The brick's OWN
# extract/driver/shim are pinned via the FIXTURE records (they cannot go
# stale that way); the SHARED, pre-existing files are pinned here.
_EXTRACT_SHA = {
    "fv3_swcore_shim.F90":
        "6a8df780cdf64600d2e592e78853e8516253d89992f3f62f3fa71a2b53003e50",
    "fv3_tpcore_duo_extract.F90":
        "71f930b3b3b6291955259e46b7dfcd8bcba5c500206c9b0a5377f24c8f8d99c9",
    "fv3_dsw5_duo_extract.F90":
        "27448ad8dc97cb7b1cfbe80910d680b005850b2c001f69709a7f2af289c01b86",
}

# Finer-grained per-subroutine block hashes INSIDE this brick's extract,
# so an unrelated edit to the file (a comment, another block) does not
# spuriously fail while a body edit still does.
_EXTRACT_BLOCK_SHA = {
    "geopk":
        "05c7accebe9ef13e56c994b7a49be695975c8acd519a30fbb892fa8b9b946c8d",
    "p_grad_c":
        "5e21f4b1ef8a5640aff643193d9c1ae36a48508dd17aef172f840ead11b9e353",
    "one_grad_p":
        "09c2082f6ca2e2fb75d65ce8908b6ac8470cce1833f6ae5da02ffe8bf77f01fb",
    "a2b_ord2":
        "077db6d0bf2073553d93f36e356faa727b47de334b832040108a665340a1b0af",
}

# tokens produced by the CHAIN (everything but the standalone probe)
_CHAIN_TOKENS = (
    "pkc_c", "gz_c", "pe_c", "peln_c", "pkz_c",
    "uc_pgc", "vc_pgc", "uc_pgc_dppoison", "vc_pgc_dppoison",
    "pkc_d", "gz_d", "pe_d", "peln_d", "pkz_d",
    "u_ogp", "v_ogp", "pk_ogp", "gz_ogp",
    "u_ogp_dppoison", "v_ogp_dppoison",
)

_PROBE_HINT = (
    "if test_a_transcendental_agreement_probe also FAILED, this is an "
    "ENVIRONMENT result (gfortran libm vs NumPy), not a port defect")


def _load_gen():
    path = os.path.join(VALID, "gen_geopk_pgrad_oracle.py")
    spec = importlib.util.spec_from_file_location(
        "gen_geopk_pgrad_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(km):
    path = os.path.join(FIX, f"geopk_pgrad_oracle_c12_km{km}.npz")
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not generated yet — run "
                    "scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch")
    return np.load(path, allow_pickle=True)


def _fields(oracle):
    """The serialized INPUT set, rebuilt from the fixture's STORED arrays."""
    return {k[3:]: np.array(oracle[k]) for k in oracle.files
            if k.startswith("in_")}


_PORT_CACHE = {}


def _port(km):
    """Run the port on the fixture's stored inputs (memoised per km)."""
    if km not in _PORT_CACHE:
        oracle = _load(km)
        gen = _load_gen()
        flds = _fields(oracle)
        _PORT_CACHE[km] = (oracle, gen, flds, gen.run_port(flds, km))
    return _PORT_CACHE[km]


def _nd(got, want):
    got = np.ascontiguousarray(got, dtype=np.float64)
    want = np.ascontiguousarray(want, dtype=np.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    return int((got.view(np.uint64) != want.view(np.uint64)).sum())


def _bit_exact(oracle, out, keys):
    for key in keys:
        nd = _nd(out[key], oracle[key])
        assert nd == 0, (f"{key}: {nd}/{np.asarray(oracle[key]).size} words "
                         f"differ bitwise — {_PROBE_HINT}")


def _box(oracle, extra=0):
    """python slice for the Fortran window ``is-extra .. ie+extra``."""
    ng = int(oracle["ng"])
    res = int(oracle["res"])
    lo = 1 - ng
    return slice(1 - extra - lo, res + extra - lo + 1)


# =====================================================================
# 5.4  the transcendental trap — RUNS FIRST, attributes libm divergence
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_a_transcendental_agreement_probe(km):
    oracle = _load(km)
    akap = float(oracle["akap"])
    probe = np.ascontiguousarray(np.asarray(oracle["in_logexp_probe"],
                                            dtype=np.float64))
    want = np.ascontiguousarray(oracle["logexp_out"], dtype=np.float64)
    got = np.ascontiguousarray(np.exp(akap * np.log(probe)))
    nd = _nd(got, want)
    assert nd == 0, (
        f"{nd}/{want.size} exp(akap*log(p)) words differ: libm divergence "
        "between gfortran and NumPy on this host — this is an ENVIRONMENT "
        "result, NOT a port defect; the pk/peln/gz comparisons below "
        "cannot be bit-exact and must be re-graded (see the spec 5.4 "
        "fallback: serialize PK/PELN as INPUTS and downgrade geopk's "
        "transcendental core to a 1-ULP gate — a recorded, justified "
        "downgrade, never a silent one)")


@pytest.mark.parametrize("km", KMS)
def test_ptk_uses_power_not_exp_log(km):
    """dyn_core makes ``ptk = ptop**akap`` (:248/:2382) but
    ``pk = exp(akap*log(p))`` (:2746).  Those are different operations;
    the port must mirror each at its own site."""
    oracle = _load(km)
    ptop = float(oracle["ptop"])
    akap = float(oracle["akap"])
    ng, res = int(oracle["ng"]), int(oracle["res"])
    b = slice(ng, ng + res + 1)          # B box [is, ie+1] x [js, je+1]
    pk_ogp = np.asarray(oracle["pk_ogp"], dtype=np.float64)
    top_pow = float(ptop) ** float(akap)
    assert (pk_ogp[b, b, 0] == top_pow).all(), (
        "one_grad_p's top_value seed is not ptop**akap on the B box")
    top_explog = float(np.exp(akap * np.log(ptop)))
    if top_explog != top_pow:
        assert not (pk_ogp[b, b, 0] == top_explog).any(), (
            "the seed matched exp(akap*log(ptop)) instead of ptop**akap")


# =====================================================================
# 4.4  provenance: input hash, extract drift, lineage
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_input_hash_enforced_and_tamper(km):
    oracle = _load(km)
    gen = _load_gen()
    flds = _fields(oracle)
    stored = str(oracle["input_sha256"])
    assert len(stored) == 64 and all(c in "0123456789abcdef" for c in stored)
    blob = gen.serialize_geopk_pgrad_inputs(flds, int(oracle["res"]),
                                            int(oracle["ng"]), km)
    got = hashlib.sha256(blob).hexdigest()
    assert got == stored, (
        f"stored input_sha256 {stored} != hash of the stored arrays {got}")
    tampered = dict(flds)
    tampered["delp"] = flds["delp"].copy()
    tampered["delp"].flat[0] = np.nextafter(tampered["delp"].flat[0], np.inf)
    bad = hashlib.sha256(gen.serialize_geopk_pgrad_inputs(
        tampered, int(oracle["res"]), int(oracle["ng"]), km)).hexdigest()
    assert bad != stored, "one-ULP tamper did not change input_sha256"


@pytest.mark.parametrize("km", KMS)
def test_extract_sha_manifest(km):
    oracle = _load(km)
    for name, want in _EXTRACT_SHA.items():
        path = os.path.join(VALID, name)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert got == want, (
            f"{name} drifted from the bytes the fixture certifies (got "
            f"{got}) — regenerate geopk_pgrad_oracle_c12_km{km}.npz")
    for name, key in (("fv3_geopk_pgrad_extract.F90",
                       "geopk_pgrad_extract_sha256"),
                      ("fv3_geopk_pgrad_driver.F90", "driver_sha256"),
                      ("fv3_geopk_pgrad_shim.F90", "shim_sha256")):
        path = os.path.join(VALID, name)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert got == str(oracle[key]), (
            f"{name} drifted from the bytes the fixture certifies (got "
            f"{got}) — regenerate geopk_pgrad_oracle_c12_km{km}.npz")


def test_extract_block_sha_pinned():
    """Per-subroutine block hashes inside this brick's extract, so an
    edit to a body goes red while an unrelated comment does not."""
    path = os.path.join(VALID, "fv3_geopk_pgrad_extract.F90")
    txt = open(path, encoding="utf-8").read().split("\n")
    for name, want in _EXTRACT_BLOCK_SHA.items():
        i0 = next(i for i, ln in enumerate(txt)
                  if ln.strip().startswith(f"subroutine {name}("))
        i1 = next(i for i, ln in enumerate(txt)
                  if ln.strip() == f"end subroutine {name}")
        block = ("\n".join(txt[i0:i1 + 1]) + "\n").encode()
        assert hashlib.sha256(block).hexdigest() == want, name


@pytest.mark.parametrize("km", KMS)
def test_fixture_provenance(km):
    oracle = _load(km)
    lin = str(oracle["input_lineage"])
    for kw in ("one face", "hydrostatic", "no USE_COND", "no SW_DYNAMICS",
               "TRANSLATION CERTIFICATE", "a2b_ord=4"):
        assert kw in lin, kw
    auth = str(oracle["auth_block_sha256"])
    for tag in ("geopk:dyn_core.F90:2660-2790:",
                "p_grad_c:dyn_core.F90:2073-2132:",
                "one_grad_p:dyn_core.F90:2347-2480:",
                "a2b_ord4:a2b_edge.F90:50-330:",
                "a2b_ord2:a2b_edge.F90:332-453:",
                "a2b_params:a2b_edge.F90:33-43:"):
        assert tag in auth, tag
    # the DEAD-input justifications must actually be recorded
    gen = _load_gen()
    for reason in set(gen.DEAD_INPUTS.values()):
        assert reason in lin, reason[:40]
    assert int(oracle["km"]) == km


# =====================================================================
# 4.4-3  the four bit-exact stage gates
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_geopk_c_bit_exact(km):
    oracle, _gen, _f, out = _port(km)
    _bit_exact(oracle, out,
               ("pkc_c", "gz_c", "pe_c", "peln_c", "pkz_c"))


@pytest.mark.parametrize("km", KMS)
def test_pgrad_c_bit_exact(km):
    oracle, _gen, _f, out = _port(km)
    _bit_exact(oracle, out, ("uc_pgc", "vc_pgc"))


@pytest.mark.parametrize("km", KMS)
def test_geopk_d_bit_exact(km):
    oracle, _gen, _f, out = _port(km)
    _bit_exact(oracle, out,
               ("pkc_d", "gz_d", "pe_d", "peln_d", "pkz_d"))


@pytest.mark.parametrize("km", KMS)
def test_one_grad_p_bit_exact(km):
    oracle, _gen, _f, out = _port(km)
    _bit_exact(oracle, out, ("u_ogp", "v_ogp", "pk_ogp", "gz_ogp"))


# =====================================================================
# 5.1  the killer — a level-independent result that passes trivially
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_levels_are_distinct(km):
    oracle = _load(km)
    ng, res = int(oracle["ng"]), int(oracle["res"])
    m_a = res + 2 * ng
    inner = slice(ng, ng + res)
    for key in ("pkc_c", "gz_c", "pkc_d", "gz_d", "pkz_d", "uc_pgc",
                "vc_pgc", "u_ogp", "v_ogp"):
        a = np.asarray(oracle[key], dtype=np.float64)
        # compute-domain-shaped tokens (pkz) are already unhaloed
        sl0 = inner if a.shape[0] >= m_a else slice(None)
        sl1 = inner if a.shape[1] >= m_a else slice(None)
        nlev = a.shape[-1]
        assert nlev >= 2, key
        for k1 in range(nlev):
            for k2 in range(k1 + 1, nlev):
                s1 = a[sl0, sl1, k1]
                s2 = a[sl0, sl1, k2]
                assert not np.array_equal(s1, s2), (key, k1, k2)
                scale = max(float(np.abs(s1).max()), 1.0e-300)
                rel = float(np.abs(s1 - s2).max()) / scale
                assert rel > 1.0e-3, (
                    f"{key}: levels {k1}/{k2} agree to {rel:.3e} relative — "
                    "a level-broadcast bug would survive this fixture")


@pytest.mark.parametrize("km", KMS)
def test_pressure_column_is_accumulated(km):
    oracle = _load(km)
    ptop, akap = float(oracle["ptop"]), float(oracle["akap"])
    pk = np.asarray(oracle["pkc_d"], dtype=np.float64)
    delp = np.asarray(oracle["in_delp"], dtype=np.float64)
    # the D pass runs computehalo=.true. => the FULL data domain
    assert (np.diff(pk, axis=-1) > 0.0).all(), (
        "pkc_d is not strictly increasing top->surface")
    assert (pk >= ptop ** akap).all()
    assert (pk[:, :, 0] == ptop ** akap).all()
    single = np.exp(akap * np.log(ptop + delp[:, :, km - 1]))
    assert not np.allclose(pk[:, :, km], single, rtol=1e-9, atol=0.0), (
        "the surface interface equals a SINGLE-layer pressure — the "
        "running p1d accumulator was lost")
    assert not np.array_equal(pk[:, :, km], pk[:, :, km - 1])


@pytest.mark.parametrize("km", KMS)
def test_gz_integrates_upward_from_hs(km):
    oracle = _load(km)
    gz = np.asarray(oracle["gz_d"], dtype=np.float64)
    hs = np.asarray(oracle["in_hs"], dtype=np.float64)
    assert _nd(gz[:, :, km], hs) == 0, (
        "gz(km+1) != hs bitwise — the SURFACE seed is wrong")
    assert (np.diff(gz, axis=-1) < 0.0).all(), (
        "gz does not decrease downward — the bottom-up recursion was "
        "flipped (seeded at the top instead of at hs)")


def test_km2_km3_differ():
    """The whole point of the brick: km=3 must not be a relabelled km=2."""
    o2, o3 = _load(2), _load(3)
    ng, res = int(o2["ng"]), int(o2["res"])
    inner = slice(ng, ng + res)
    # the k=1 interface is ptop**akap by construction on BOTH
    assert _nd(np.asarray(o2["pkc_d"])[..., 0],
               np.asarray(o3["pkc_d"])[..., 0]) == 0
    for key, k in (("pkc_d", 1), ("gz_d", 0), ("uc_pgc", 0), ("u_ogp", 0)):
        a2 = np.asarray(o2[key], dtype=np.float64)[inner, inner, k]
        a3 = np.asarray(o3[key], dtype=np.float64)[inner, inner, k]
        scale = max(float(np.abs(a2).max()), 1.0e-300)
        rel = float(np.abs(a2 - a3).max()) / scale
        assert rel > 1.0e-6, (
            f"{key}[...,{k}] agrees to {rel:.3e} between km=2 and km=3 — "
            "the km=3 profile has degenerated into a refinement of km=2")


@pytest.mark.parametrize("km", KMS)
def test_pkz_between_interfaces(km):
    """pkz is the ONLY quantity that consumes peln, so this is also the
    only check that fails if peln was written in the wrong (i,k,j)
    index order."""
    oracle = _load(km)
    ng, res = int(oracle["ng"]), int(oracle["res"])
    c = slice(ng, ng + res)
    pk = np.asarray(oracle["pkc_d"], dtype=np.float64)[c, c, :]
    pkz = np.asarray(oracle["pkz_d"], dtype=np.float64)
    assert pkz.shape == (res, res, km)
    assert (pkz > pk[:, :, :-1]).all()
    assert (pkz < pk[:, :, 1:]).all()


# =====================================================================
# 5.2  sentinel-fed dead paths reported as certified
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_hydrostatic_branch_ignores_delpc(km):
    """The Fortran was re-run with delpc = -9.e9 from restored pristine
    uc/vc; the result must be BITWISE identical."""
    oracle = _load(km)
    for a, b in (("uc_pgc", "uc_pgc_dppoison"), ("vc_pgc", "vc_pgc_dppoison")):
        assert _nd(oracle[b], oracle[a]) == 0, (
            f"{b} differs from {a}: p_grad_c READ delpc on the "
            "hydrostatic branch")
    uc = np.asarray(oracle["uc_pgc"], dtype=np.float64)
    assert np.isfinite(uc).all()
    assert float(np.abs(uc).max()) < 1.0e8, (
        "UC_PGC is out of band — a -9.e9 slot leaked into the update")


@pytest.mark.parametrize("km", KMS)
def test_hydrostatic_branch_ignores_delp_in_ogp(km):
    oracle = _load(km)
    for a, b in (("u_ogp", "u_ogp_dppoison"), ("v_ogp", "v_ogp_dppoison")):
        assert _nd(oracle[b], oracle[a]) == 0, (
            f"{b} differs from {a}: one_grad_p READ delp on the "
            "hydrostatic branch")


@pytest.mark.parametrize("km", KMS)
def test_q_con_unreferenced(km):
    """The compiled lane has no -DUSE_COND path: the fixture matches the
    port's ``use_cond=False`` result bit-exactly, and ``use_cond=True``
    RAISES rather than silently running the dry formulas."""
    from legoesm.core.fv3_native_pgrad import geopk

    oracle, _gen, flds, out = _port(km)
    assert _nd(out["pkc_d"], oracle["pkc_d"]) == 0
    assert float(np.abs(np.asarray(flds["q_con"])).max()) == 0.0
    from legoesm.core.fv3_native_sw_core import Bounds

    bd = Bounds.single_tile(int(oracle["res"]), int(oracle["ng"]))
    with pytest.raises(ValueError, match="USE_COND"):
        geopk(flds["delp"], flds["pt"], flds["hs"], bd, km=km,
              ptop=float(oracle["ptop"]), akap=float(oracle["akap"]),
              cp_air=float(oracle["cp_air"]), cg=False, duogrid=True,
              computehalo=True, npx=int(oracle["npx"]),
              npy=int(oracle["npy"]), a2b_ord=4, q_con=flds["q_con"],
              use_cond=True)


# =====================================================================
# 5.3  unread inputs / wrong read window silently tolerated
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_rdxc_last_row_never_read(km):
    """The driver allocates gs%rdxc at the DUMMY shape
    (isd:ied+1, jsd:jed+1) (dyn_core:2084) while only (m_b, m_a) is
    serialized, so row j=jed+1 stays -9.e9.  A read of it produces
    ~1e9+; UC_PGC staying in band proves the uc loop never goes past
    j=je (UNCERTAIN U3: fv_arrays.F90 is absent, so the PRODUCTION
    allocation of rdxc is unverified — this pins the read window, not
    the allocation)."""
    oracle = _load(km)
    res, ng = int(oracle["res"]), int(oracle["ng"])
    rdxc = np.asarray(oracle["in_rdxc"], dtype=np.float64)
    assert rdxc.shape == (res + 2 * ng + 1, res + 2 * ng)
    assert (rdxc != POISON).all(), "the serialized rdxc carries the poison"
    uc = np.asarray(oracle["uc_pgc"], dtype=np.float64)
    assert np.isfinite(uc).all()
    assert float(np.abs(uc).max()) < 1.0e8


@pytest.mark.parametrize("km", KMS)
def test_geopk_cg_write_box_exact(km):
    """CG=.true. writes EXACTLY [is-1,ie+1] x [js-1,je+1]: 1e30 outside,
    finite inside.  A port using the D-grid 2-halo width, or applying
    the computehalo extension on the C call, dies here."""
    oracle = _load(km)
    m_a = int(oracle["res"]) + 2 * int(oracle["ng"])
    box = _box(oracle, extra=1)
    outside = np.ones((m_a, m_a), dtype=bool)
    outside[box, box] = False
    for key in ("pkc_c", "gz_c"):
        a = np.asarray(oracle[key], dtype=np.float64)
        assert (a[outside] == SENTINEL).all(), f"{key}: wrote outside the box"
        inb = a[box, box]
        assert np.isfinite(inb).all() and (inb != SENTINEL).all(), key


@pytest.mark.parametrize("km", KMS)
def test_geopk_d_write_box_exact(km):
    """CG=.false. + computehalo=.true. on the duo lane reaches the FULL
    data domain (is==1 and ie==npx-1 at C12 single-tile), so PKC_D/GZ_D
    carry NO sentinel at all."""
    oracle = _load(km)
    for key in ("pkc_d", "gz_d"):
        a = np.asarray(oracle[key], dtype=np.float64)
        assert np.isfinite(a).all(), key
        assert not (a == SENTINEL).any(), (
            f"{key}: the computehalo extension did not reach the data domain")


@pytest.mark.parametrize("km", KMS)
def test_pkz_sentinel_on_cg_pass(km):
    """pkz is entirely unwritten when CG=.true. (guard at :2781).  A
    port that computes pkz unconditionally dies here.  peln/pe, by
    contrast, ARE written on the C pass (their guards carry no CG
    test) — assert that too, so the sentinel claim stays precise."""
    oracle = _load(km)
    assert (np.asarray(oracle["pkz_c"], dtype=np.float64) == SENTINEL).all()
    for key in ("peln_c", "pe_c"):
        a = np.asarray(oracle[key], dtype=np.float64)
        assert np.isfinite(a).all() and not (a == SENTINEL).any(), key


@pytest.mark.parametrize("km", KMS)
def test_ogp_replace_semantics(km):
    """a2b_ord4(..., replace=.true.) at :2399/:2409 overwrites pk/gz on
    the B box [is,ie+1] x [js,je+1] and NOWHERE else."""
    oracle = _load(km)
    res, ng = int(oracle["res"]), int(oracle["ng"])
    ptop, akap = float(oracle["ptop"]), float(oracle["akap"])
    m_a = res + 2 * ng
    b = slice(ng, ng + res + 1)
    outside = np.ones((m_a, m_a), dtype=bool)
    outside[b, b] = False
    # pk level 1 is the top_value seed, NOT an a2b output (:2397 starts
    # at k=2), so it is compared separately.
    pk_pre = np.asarray(oracle["pkc_d"], dtype=np.float64)
    pk_post = np.asarray(oracle["pk_ogp"], dtype=np.float64)
    gz_pre = np.asarray(oracle["gz_d"], dtype=np.float64)
    gz_post = np.asarray(oracle["gz_ogp"], dtype=np.float64)
    for k in range(1, km + 1):
        assert not np.array_equal(pk_post[b, b, k], pk_pre[b, b, k]), k
    for k in range(0, km + 1):
        assert not np.array_equal(gz_post[b, b, k], gz_pre[b, b, k]), k
        assert _nd(gz_post[outside][..., k], gz_pre[outside][..., k]) == 0, k
        assert _nd(pk_post[outside][..., k], pk_pre[outside][..., k]) == 0, k
    assert (pk_post[b, b, 0] == ptop ** akap).all()
    assert _nd(pk_post[outside][..., 0], pk_pre[outside][..., 0]) == 0


@pytest.mark.parametrize("km", KMS)
def test_every_serialized_input_is_consumed(km):
    """Perturb each serialized input by a relative 1e-6 and re-run the
    PORT: at least one CHAIN token must change, or the input must be a
    DECLARED dead input whose justification is recorded in the fixture's
    input_lineage.  Two-sided: a declared-dead input that DOES matter is
    also a failure."""
    oracle, gen, flds, base = _port(km)
    dead = gen.DEAD_INPUTS
    for _tok, key, _origin in gen.INPUT_FIELDS:
        pert = {k: np.array(v, dtype=np.float64, copy=True)
                for k, v in flds.items()}
        a = pert[key]
        if a.ndim == 0:
            base_v = float(a)
            pert[key] = np.array(base_v * (1.0 + 1.0e-6) if base_v
                                 else 1.0e-6)
        else:
            nz = np.abs(a) > 0.0
            pert[key] = np.where(nz, a * (1.0 + 1.0e-6), 1.0e-6)
        out = gen.run_port(pert, km)
        moved = any(_nd(out[t], base[t]) != 0 for t in _CHAIN_TOKENS)
        if key in dead:
            assert not moved, (
                f"{key} is declared DEAD but perturbing it moved the chain "
                "— the DEAD_INPUTS claim (and the fixture lineage) is wrong")
        else:
            assert moved, (
                f"{key} is serialized and hashed but changes NO chain token "
                "— either the port is not reading it or it belongs in "
                "DEAD_INPUTS with a justification")


# =====================================================================
# 5.5  accumulation-order and grouping drift
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_accumulation_order_is_top_down(km):
    """Make the bit-exact result ATTRIBUTABLE: a bottom-up accumulation
    of the SAME delp must differ bitwise somewhere in the data domain."""
    oracle = _load(km)
    ptop, akap = float(oracle["ptop"]), float(oracle["akap"])
    delp = np.asarray(oracle["in_delp"], dtype=np.float64)
    got = np.ascontiguousarray(
        np.asarray(oracle["pkc_d"], dtype=np.float64)[:, :, km])
    acc = np.zeros(delp.shape[:2])
    for k in range(km, 0, -1):
        acc = acc + delp[:, :, k - 1]
    rev = np.ascontiguousarray(np.exp(akap * np.log(acc + ptop)))
    nd = int((got.view(np.uint64) != rev.view(np.uint64)).sum())
    assert nd > 0, (
        "top-down and bottom-up accumulation agree bitwise at EVERY point "
        "— the fixture's delp is too smooth to discriminate the sum order "
        "(a FIXTURE DEFECT, not a pass)")


# =====================================================================
# 5.6  the gate as a whole
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_outputs_discriminated(km):
    oracle = _load(km)
    ng, res = int(oracle["ng"]), int(oracle["res"])
    # every non-sentinel token finite and nonzero
    for key in ("pkc_d", "gz_d", "pe_d", "peln_d", "pkz_d", "uc_pgc",
                "vc_pgc", "u_ogp", "v_ogp", "pk_ogp", "gz_ogp",
                "logexp_out"):
        a = np.asarray(oracle[key], dtype=np.float64)
        assert np.isfinite(a).all(), key
        assert float(np.abs(a).max()) > 0.0, key
        assert not (a == SENTINEL).any(), key
    # the pressure gradient actually MOVED the winds
    assert not np.array_equal(np.asarray(oracle["uc_pgc"]),
                              np.asarray(oracle["in_uc"]))
    assert not np.array_equal(np.asarray(oracle["vc_pgc"]),
                              np.asarray(oracle["in_vc"]))
    assert not np.array_equal(np.asarray(oracle["u_ogp"]),
                              np.asarray(oracle["in_u"]))
    assert not np.array_equal(np.asarray(oracle["v_ogp"]),
                              np.asarray(oracle["in_v"]))
    # the two geopk calls saw different delp AND different ranges
    assert not np.array_equal(np.asarray(oracle["pkc_c"]),
                              np.asarray(oracle["pkc_d"]))
    # units/rdx sanity band on the WRITTEN window (outside it U_OGP
    # still holds the untouched input)
    u = np.asarray(oracle["u_ogp"], dtype=np.float64)[ng:ng + res,
                                                      ng:ng + res + 1, :]
    assert float(np.abs(u).max()) < 1.0e8, (
        "U_OGP out of band — an rdx/units mistake or a poisoned metric")
    assert float(np.abs(u).max()) > 0.0

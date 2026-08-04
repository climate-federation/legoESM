"""Brick ``geopk_pgrad`` — multilevel hydrostatic pressure-chain oracle.

WHAT IS CERTIFIED.  The python port in
``legoesm.core.fv3_native_pgrad`` reproduces the verbatim Fortran chain

    geopk(CG=.true. , computehalo=.false.)   dyn_core.F90:2660-2790 @ :533
      -> p_grad_c                            dyn_core.F90:2073-2132 @ :629
      -> geopk(CG=.false., computehalo=.true.)                      @ :1401
        -> one_grad_p                        dyn_core.F90:2347-2480 @ :1531

plus ``a2b_edge.F90:50-330`` (a2b_ord4, reused from the CERTIFIED
``a2b_edge_duo_mod``) and ``a2b_edge.F90:332-453`` (a2b_ord2, linked but
DEAD on this lane), on the authoritative Zenodo 8327578 *symmetryclean*
sources, at C12 with km=2 AND km=3.

GRADING (spec 5.4 fallback — MEASURED, RECORDED, never silent).  Job
9320294 (2026-08-04) measured a real gfortran-vs-NumPy libm divergence
on this host: ``exp(akap*log(p))`` disagrees on 1/64 (km=2) and 3/64
(km=3) probe words.  ``geopk``'s transcendental core therefore CANNOT be
graded bit-exact here.  Rather than surrender exactness everywhere, the
divergence is QUARANTINED: ``p_grad_c`` and ``one_grad_p`` are re-run on
the FORTRAN-computed ``pk``/``gz`` (the spec's "serialize PK/PELN as
INPUTS" variant, realised by feeding back the already-dumped
``PKC_C``/``GZ_C``/``PKC_D``/``GZ_D`` tokens), which makes every
downstream token pure ``+ - * /`` and BIT-EXACT.

  BIT-EXACT, nd == 0 required, no tolerance
      ``pe_c``, ``pkz_c``, ``pe_d`` (pure running sums / all-sentinel);
      ``uc_pgc``, ``vc_pgc``, ``u_ogp``, ``v_ogp``, ``pk_ogp``,
      ``gz_ogp`` and their ``*_DPPOISON`` controls (downstream of the
      quarantine); the ``gz_d(km+1) == hs`` seed identity.
  ULP-BOUNDED, declared per-token caps
      ``pkc_c``, ``gz_c``, ``peln_c``, ``pkc_d``, ``gz_d``, ``peln_d``,
      ``pkz_d``, ``logexp_out``.

The bucket is not a blanket amnesty: it UNLOCKS only when the probe
demonstrates divergence (a libm-clean host is required to be bit-exact
everywhere), the caps are derived rather than fitted, the differing-word
count is capped, and ``test_a_transcendental_agreement_probe`` fails if
the divergence ever moves off the value recorded in the fixture or
exceeds the recorded envelope.  ``test_geopk_pk_is_exp_log_of_the_
bit_exact_pe`` closes the argument by showing the port's ``pk`` is
exactly ``exp(akap*log(<the bit-exact Fortran pe>))``, so the only
unproven step inside the quarantine is the host's own ``log``/``exp``.

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
``tests/grids/fixtures/geopk_pgrad_oracle_c12_km{2,3}.npz``.  They ARE
the certificate and MUST be committed (``test_fixtures_are_committed``),
and each is bound to the build+run that produced it by a RUN MANIFEST
that ``--pack`` refuses to proceed without
(``test_run_manifest_binds_the_build``).

TRANSLATION vs FIDELITY (U1).  This gate certifies a TRANSLATION only.
Codex r20 CONFIRMED that keeping ``ptop``/``akap``/``cp_air``
header-settable — and NOT substituting ``constants.kappa`` — is right for
an oracle-matching brick, and listed what a later FIDELITY claim must
ADDITIONALLY pin, recorded verbatim:

    - exact Zenodo archive/file hashes and preprocess defines;
    - FMS ``constants_mod`` source/object hash and resolved ``cp_air``,
      ``R_d``, and real kind;
    - runtime ``ptop``, ``akap``, and thermodynamic configuration;
    - compiler, version, flags, libm/platform, and linked extract
      dependencies;
    - committed fixtures plus the run manifest above.

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
    "this token is in the BIT-EXACT bucket — it is either libm-free "
    "(a pure running sum / an all-sentinel region) or it sits DOWNSTREAM "
    "of the spec-5.4 quarantine (run on the FORTRAN-computed pk/gz), so "
    "a gfortran-vs-NumPy libm divergence CANNOT explain it.  Grade this "
    "as a PORT DEFECT")

# =====================================================================
# GRADING POLICY — the spec 5.4 fallback, RECORDED and JUSTIFIED
# ---------------------------------------------------------------------
# MEASURED on Ginsburg job 9320294 (2026-08-04, gfortran -O2
# -fdefault-real-8 vs NumPy on the same compute node): the host's libm
# disagrees with NumPy on exp(akap*log(p)) for 1/64 (km=2) and 3/64
# (km=3) probe words.  That is an ENVIRONMENT fact, so geopk's
# transcendental CORE cannot be graded bit-exact here.
#
# The fallback QUARANTINES it instead of surrendering exactness:
# ``p_grad_c`` and ``one_grad_p`` are re-run on the FORTRAN-computed
# pk/gz (the spec's "serialize PK/PELN as INPUTS" variant, realised by
# feeding back the already-dumped PKC_C/GZ_C/PKC_D/GZ_D tokens), which
# makes every downstream token pure + - * / and therefore BIT-EXACT.
#
# BIT-EXACT bucket (nd == 0 REQUIRED, no tolerance):
#   pe_c, pkz_c, pe_d          - pure running sums / all-sentinel
#   uc_pgc, vc_pgc             - p_grad_c on FORTRAN pk/gz
#   uc_pgc_dppoison, vc_pgc_dppoison
#   u_ogp, v_ogp, pk_ogp, gz_ogp - one_grad_p on FORTRAN pk/gz
#   gz_d[..., km] == hs        - seed copy (see test_gz_integrates_...)
# ULP-BOUNDED bucket (transcendental core only):
#   pkc_c, gz_c, peln_c, pkc_d, gz_d, peln_d, pkz_d, logexp_out
#
# The ONE transcendental left inside the bit-exact bucket is the single
# scalar ``ptop**akap`` seed (dyn_core:248/:2382); it is a separate libm
# entry point (pow, not exp/log), it AGREED on this host, and
# test_ptk_uses_power_not_exp_log is asserted as an explicit
# PRECONDITION of the downstream bit-exact grade.
# =====================================================================

_BITEXACT_GEOPK_C = ("pe_c", "pkz_c")
_BITEXACT_GEOPK_D = ("pe_d",)
_BITEXACT_PGRAD_C = ("uc_pgc", "vc_pgc")
_BITEXACT_OGP = ("u_ogp", "v_ogp", "pk_ogp", "gz_ogp")
_ULP_GEOPK_C = ("pkc_c", "gz_c", "peln_c")
_ULP_GEOPK_D = ("pkc_d", "gz_d", "peln_d", "pkz_d")

# Per-token ULP caps.  DERIVED, not guessed:
#   peln = log(p)            -> one libm last-bit disagreement    -> 2
#   pk   = exp(akap*log(p))  -> a 1-ULP log error is 1.8e-15 absolute on
#          log(p)~11.5; times akap -> 5.2e-16 in the exponent; exp turns
#          an absolute argument error into the same RELATIVE error
#          -> ~2.3 ULP, plus exp's own <=1 ULP                    -> 4
#   gz   = gz + cp_air*pt*(pk[k+1]-pk[k]) over <=3 levels; cp*pt~2.8e5,
#          ulp(pk~26.5)=3.6e-15 -> ~8e-9 absolute per level, and gz at
#          the thinnest level is ~2e6 (ulp 4.4e-10) -> ~18 ULP    -> 64
#   pkz  = dpk/(akap*dpeln); dpeln for the thinnest layer is ~0.18
#          against peln~4.8, a 27x amplification of the log error -> 256
# HEADROOM: 256 ULP is 5.7e-14 RELATIVE.  A structural port defect of
# the kind this brick guards against (wrong write window, level swap,
# lost p1d accumulator, regrouped expression) moves a value by >=1e-6
# relative = >=4.5e9 ULP, so even the loosest cap keeps ~7 orders of
# magnitude of discrimination.  These are tripwires, not proofs.
_ULP_TOL = {
    "pkc_c": 4.0, "pkc_d": 4.0,
    "peln_c": 2.0, "peln_d": 2.0,
    "gz_c": 64.0, "gz_d": 64.0,
    "pkz_d": 256.0, "logexp_out": 2.0,
}
# secondary tripwire: a libm last-bit effect touches a MINORITY of words;
# a port defect touches essentially all of them
_MAX_DIFF_FRAC = 0.5

# Probe envelope.  A libm disagreement is by construction a LAST-BIT
# disagreement, so >1 ULP is not "environment" and must go red.
_PROBE_ULP_MAX = 1.0
# ~2.7x the measured 3/64 worst case.  A host whose libm is WORSE than
# this is a DIFFERENT environment: re-measure and re-record it, never
# widen this silently.
_PROBE_DIVERGENCE_MAX = 8


def _load_gen():
    path = os.path.join(VALID, "gen_geopk_pgrad_oracle.py")
    spec = importlib.util.spec_from_file_location(
        "gen_geopk_pgrad_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_path(km):
    return os.path.join(FIX, f"geopk_pgrad_oracle_c12_km{km}.npz")


def _load(km):
    """Load a fixture, FAILING LOUDLY if it is absent or stale.

    Deliberately NOT ``pytest.skip`` (codex r20 blocker 1): the fixtures
    ARE the certificate.  A skip turns the whole gate into a silent no-op
    exactly when the certificate is missing — which is the state codex
    found at e2b06d364, where both npz files were untracked.
    """
    path = fixture_path(km)
    assert os.path.exists(path), (
        f"{os.path.basename(path)} is MISSING.  The fixture is the "
        "certificate, not a build artefact: regenerate it with "
        "scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch and "
        f"`git add tests/grids/fixtures/{os.path.basename(path)}`")
    npz = np.load(path, allow_pickle=True)
    for key in ("libm_probe_ndiff", "libm_probe_n", "grading_policy",
                "sumorder_ndiff", "run_manifest", "manifest_schema"):
        assert key in npz.files, (
            f"{os.path.basename(path)} predates the current certificate "
            f"records (missing '{key}') — regenerate with "
            "scripts/cluster/fv3_native/geopk_pgrad_oracle.sbatch")
    return npz


def test_fixtures_are_committed():
    """The commit must carry the certificate (codex r20 blocker 1).

    At e2b06d364 both fixtures existed on disk but were absent from
    ``git ls-files``, so the commit alone proved nothing.  Nothing in
    ``.gitignore`` excludes them — they were simply never added — and
    every sibling brick's fixture (``dsw5_duo_oracle_c12.npz``,
    ``fv3_duogrid_oracle_n2.npz``, …) IS tracked.
    """
    import subprocess

    for km in KMS:
        rel = os.path.relpath(fixture_path(km), os.path.abspath(REPO))
        rc = subprocess.run(
            ["git", "-C", os.path.abspath(REPO), "ls-files",
             "--error-unmatch", rel],
            capture_output=True, text=True).returncode
        assert rc == 0, (
            f"{rel} is NOT tracked by git — the commit carries no "
            f"reproducible certificate.  Run `git add {rel}`")


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


_FALLBACK_CACHE = {}


def _port_fallback(km):
    """Port run with the FORTRAN pk/gz fed into the downstream stages.

    This is the spec-5.4 quarantine: ``p_grad_c``/``one_grad_p`` see the
    same pk/gz the Fortran saw, so their outputs are pure + - * / of
    identical inputs and stay BIT-EXACT even on a libm-divergent host.
    """
    if km not in _FALLBACK_CACHE:
        oracle, gen, flds, _base = _port(km)
        override = {
            "pk_c": np.asarray(oracle["pkc_c"], dtype=np.float64),
            "gz_c": np.asarray(oracle["gz_c"], dtype=np.float64),
            "pk_d": np.asarray(oracle["pkc_d"], dtype=np.float64),
            "gz_d": np.asarray(oracle["gz_d"], dtype=np.float64),
        }
        _FALLBACK_CACHE[km] = gen.run_port(flds, km, pk_gz_override=override)
    return _FALLBACK_CACHE[km]


def _nd(got, want):
    got = np.ascontiguousarray(got, dtype=np.float64)
    want = np.ascontiguousarray(want, dtype=np.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    return int((got.view(np.uint64) != want.view(np.uint64)).sum())


def _ulp_report(got, want):
    """``(n_differing_words, max_ulp, max_relative)``."""
    got = np.ascontiguousarray(got, dtype=np.float64)
    want = np.ascontiguousarray(want, dtype=np.float64)
    assert got.shape == want.shape, (got.shape, want.shape)
    nd = int((got.view(np.uint64) != want.view(np.uint64)).sum())
    d = np.abs(got - want)
    ulp = np.spacing(np.abs(want))
    ulp = np.where(ulp > 0.0, ulp, np.finfo(np.float64).tiny)
    rel = d / np.maximum(np.abs(want), 1e-300)
    return nd, float((d / ulp).max()), float(rel.max())


def _grade_ulp(oracle, out, keys):
    """Grade the TRANSCENDENTAL bucket.

    Two-sided by design: if this host's libm probe measured ZERO
    divergence there is NO fallback licence and the tokens must be
    BIT-EXACT.  The ULP bucket only unlocks on demonstrated evidence.
    """
    probe_nd = int(oracle["libm_probe_ndiff"])
    for key in keys:
        nd, max_ulp, max_rel = _ulp_report(out[key], oracle[key])
        size = int(np.asarray(oracle[key]).size)
        if probe_nd == 0:
            assert nd == 0, (
                f"{key}: {nd}/{size} words differ, but this host's libm "
                f"probe is CLEAN (0/{int(oracle['libm_probe_n'])}) — there "
                "is no environment excuse, this is a PORT DEFECT")
            continue
        assert max_ulp <= _ULP_TOL[key], (
            f"{key}: max {max_ulp:.2f} ULP (cap {_ULP_TOL[key]}), "
            f"max rel {max_rel:.3e}, {nd}/{size} words differ.  The "
            f"libm probe measured {probe_nd}/{int(oracle['libm_probe_n'])} "
            "words at <=1 ULP, which cannot explain a deviation this "
            "large — grade as a PORT DEFECT, not as environment")
        assert nd <= _MAX_DIFF_FRAC * size, (
            f"{key}: {nd}/{size} words differ ({nd / size:.1%}) — a libm "
            "last-bit effect touches a MINORITY of words; this looks "
            "like a port defect")


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
    """MEASURE the host's gfortran-vs-NumPy libm divergence and hold it
    inside the RECORDED envelope.

    This is not a pass/fail on "is the host clean" — job 9320294 proved
    it is not (1/64 at km=2, 3/64 at km=3).  It is the tripwire that
    keeps the spec-5.4 fallback honest:

    * the live count must EQUAL the count recorded in the fixture at
      pack time (a different host, compiler or NumPy goes red instead of
      quietly re-grading);
    * the count must stay within the absolute envelope;
    * every difference must be a LAST-BIT difference (<=1 ULP) — a
      larger gap is not "environment" and must not be excused as one.

    LIMIT OF THE PROBE (codex r20 finding 7, CONFIRMED).  This is an
    ATTRIBUTION instrument, not a proof that geopk's CORE uses
    ``exp(akap*log(p))``: the probe is a standalone loop, so a port could
    compute it with ``exp(akap*log())`` while using ``**`` inside geopk
    and the probe would never notice.  What forecloses that substitution
    is ``test_geopk_pk_is_exp_log_of_the_bit_exact_pe``, which requires
    the port's ``pk`` to equal ``exp(akap*log(<Fortran pe>))`` BITWISE —
    a ``**`` core differs from that by ~2-4 ULP and fails it.  Note the
    exact-``PK``-comparison route is NOT available on this host (``pkc``
    is ULP-graded, and its cap would absorb a ``**``-vs-``exp(log)``
    swap), so that bit-exact re-derivation is the load-bearing check
    here, not a redundancy.
    """
    oracle = _load(km)
    gen = _load_gen()
    nd, max_ulp = gen.libm_probe_report(oracle["in_logexp_probe"],
                                        oracle["logexp_out"])
    n = int(oracle["libm_probe_n"])
    recorded = int(oracle["libm_probe_ndiff"])
    assert nd == recorded, (
        f"live libm divergence {nd}/{n} != the {recorded}/{n} recorded in "
        "the fixture at pack time — the host, the compiler or NumPy "
        "changed.  RE-MEASURE and re-record; do not re-grade silently")
    assert nd <= _PROBE_DIVERGENCE_MAX, (
        f"libm divergence {nd}/{n} exceeds the recorded envelope "
        f"{_PROBE_DIVERGENCE_MAX}/{n} — this is a DIFFERENT environment "
        "from the one the fallback was justified on; re-measure and "
        "re-record the envelope, never widen it silently")
    assert max_ulp <= _PROBE_ULP_MAX, (
        f"libm divergence reaches {max_ulp:.2f} ULP — a libm last-bit "
        "disagreement is <=1 ULP by construction, so this is NOT a "
        "transcendental-rounding effect and must not be excused as one")
    if nd == 0:
        # host is libm-clean: _grade_ulp then demands bit-exact anyway
        assert float(oracle["libm_probe_max_ulp"]) == 0.0


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
def test_run_manifest_binds_the_build(km):
    """The fixture is bound to the BUILD+RUN that produced it (codex r20
    blocker 2).

    Before this, ``--pack`` verified only that the staging serialised to
    the current input text, then stamped hashes of the CURRENT
    extract/shim/driver onto whatever Fortran output was on disk — so a
    changed source, compiler or preprocessor could be packed against
    stale output and the extract-hash certificate proved nothing.  The
    manifest closes that: it records the input and output hashes, the
    hash of EVERY compiled source, the preprocess defines, the
    compiler+version+target+flags and the executable hash, ``--pack``
    REFUSES without it, and this test re-verifies the source hashes
    against the repo as it stands now.
    """
    gen = _load_gen()
    oracle = _load(km)
    man = gen.read_manifest_text(str(oracle["run_manifest"]))
    assert man["schema"] == gen.MANIFEST_SCHEMA
    assert int(man["km"]) == km
    for key in gen.MANIFEST_REQUIRED:
        assert key in man and man[key] != "", key
    # the preprocess lane is part of the certificate
    assert "SW_DYNAMICS" in man["defines"] and "USE_COND" in man["defines"]
    assert man["compiler"] == "gfortran"
    assert man["flags"] and "-fdefault-real-8" in man["flags"]
    # every compiled source must STILL hash to what was built
    for name in gen.MANIFEST_SOURCES:
        got = hashlib.sha256(
            open(os.path.join(VALID, name), "rb").read()).hexdigest()
        assert got == man[f"source.{name}"], (
            f"{name} drifted from the bytes the fixture certifies (repo "
            f"{got}, built {man[f'source.{name}']}) — the committed "
            f"fixture no longer matches the sources; regenerate "
            f"geopk_pgrad_oracle_c12_km{km}.npz")
    # the standalone hash records must agree with the manifest
    assert str(oracle["geopk_pgrad_extract_sha256"]) == \
        man["source.fv3_geopk_pgrad_extract.F90"]
    assert str(oracle["driver_sha256"]) == \
        man["source.fv3_geopk_pgrad_driver.F90"]
    assert str(oracle["shim_sha256"]) == \
        man["source.fv3_geopk_pgrad_shim.F90"]
    assert str(oracle["input_sha256"]) == man["input_sha256"]


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
    # the grading decision and the sum-order evidence are part of the
    # provenance, not of the test's private opinion
    assert str(oracle["grading_policy"]) in lin
    assert "SUM-ORDER (spec 5.5)" in lin
    assert "measured on the PRESSURE" in lin
    if int(oracle["libm_probe_ndiff"]):
        assert "SPEC-5.4 FALLBACK APPLIED" in lin
        assert f"{int(oracle['libm_probe_ndiff'])}/" \
               f"{int(oracle['libm_probe_n'])} probe words differ" in lin


# =====================================================================
# 4.4-3  the four bit-exact stage gates
# =====================================================================

@pytest.mark.parametrize("km", KMS)
def test_geopk_c_libm_free_bit_exact(km):
    """geopk's LIBM-FREE outputs, graded BIT-EXACT (no tolerance).

    ``pe`` is the running pressure accumulator — pure ``+`` — so it
    certifies the whole structural core of geopk (the write windows, the
    level ordering, the running ``p1d`` accumulator and its sum ORDER)
    without touching ``log``/``exp``.  ``pkz_c`` is all-sentinel.
    """
    oracle, _gen, _f, out = _port(km)
    _bit_exact(oracle, out, _BITEXACT_GEOPK_C)


@pytest.mark.parametrize("km", KMS)
def test_geopk_c_transcendental_ulp(km):
    """geopk's TRANSCENDENTAL outputs, ULP-bounded (spec 5.4 fallback).

    ``pkc_c``/``peln_c`` go through ``log``/``exp`` and ``gz_c`` is
    built from ``pkc_c``, so on a libm-divergent host they cannot be
    bit-exact.  Caps are DERIVED (see _ULP_TOL) and the bucket only
    unlocks when the probe DEMONSTRATES divergence.
    """
    oracle, _gen, _f, out = _port(km)
    _grade_ulp(oracle, out, _ULP_GEOPK_C)


@pytest.mark.parametrize("km", KMS)
def test_geopk_d_libm_free_bit_exact(km):
    oracle, _gen, _f, out = _port(km)
    _bit_exact(oracle, out, _BITEXACT_GEOPK_D)


@pytest.mark.parametrize("km", KMS)
def test_geopk_d_transcendental_ulp(km):
    oracle, _gen, _f, out = _port(km)
    _grade_ulp(oracle, out, _ULP_GEOPK_D)


@pytest.mark.parametrize("km", KMS)
def test_geopk_pk_is_exp_log_of_the_bit_exact_pe(km):
    """ATTRIBUTION: the port's ``pk`` is exactly
    ``exp(akap*log(pe))`` of a ``pe`` that is itself certified BIT-EXACT
    against the Fortran.  So the ONLY unproven step inside geopk's
    transcendental core is the host's own ``log``/``exp`` evaluation —
    the quarantine is complete, not hand-waved."""
    oracle, _gen, _f, out = _port(km)
    res, ng = int(oracle["res"]), int(oracle["ng"])
    akap = float(oracle["akap"])
    # pe covers i in is-1..ie+1, j in js-1..je+1 (origin (is-1, 1, js-1));
    # map onto pk's data-domain indices (origin isd = 1-ng)
    sl = slice(ng - 1, ng + res + 1)
    # pe axes are (i, k, j); pk axes are (i, j, k) — pe[:, k, :] is
    # therefore already (i, j), no transpose.  The FORTRAN pe is used
    # (not the port's) so the statement proved is the strong one: the
    # port's pk is exp(akap*log(<the Fortran pressure>)).
    assert _nd(out["pe_c"], oracle["pe_c"]) == 0, (
        "precondition: pe_c must be bit-exact for the quarantine "
        "argument to close")
    pe = np.asarray(oracle["pe_c"], dtype=np.float64)
    pk = np.asarray(out["pkc_c"], dtype=np.float64)[sl, sl, :]
    for k in range(1, int(oracle["km"]) + 1):
        want = np.exp(akap * np.log(pe[:, k, :]))
        assert _nd(pk[:, :, k], want) == 0, (
            f"level {k}: the port's pk is NOT exp(akap*log(pe)) of its own "
            "pe — the transcendental quarantine argument does not hold")


@pytest.mark.parametrize("km", KMS)
def test_pgrad_c_bit_exact_on_fortran_pk(km):
    """p_grad_c graded BIT-EXACT — spec-5.4 fallback.

    The port's ``p_grad_c`` is driven by the FORTRAN-computed
    ``PKC_C``/``GZ_C`` (already serialized in the fixture), so both
    sides evaluate the SAME pure ``+ - * /`` expression on the SAME
    inputs.  No tolerance is granted here: libm is quarantined upstream,
    it is NOT an excuse for the pressure-gradient stage.
    """
    oracle = _load(km)
    out = _port_fallback(km)
    _bit_exact(oracle, out, _BITEXACT_PGRAD_C)
    # the poison controls ride the same path
    _bit_exact(oracle, out, ("uc_pgc_dppoison", "vc_pgc_dppoison"))


@pytest.mark.parametrize("km", KMS)
def test_one_grad_p_bit_exact_on_fortran_pk(km):
    """one_grad_p graded BIT-EXACT — spec-5.4 fallback.

    Driven by the FORTRAN ``PKC_D``/``GZ_D``.  The stage is pure
    ``+ - * /`` (a2b_ord4, the divg2 differences, the wind update) EXCEPT
    for the single scalar ``ptop**akap`` seed at :2382 — a different
    libm entry point (``pow``) that AGREED on this host.  That agreement
    is asserted HERE as an explicit precondition, so the bit-exact claim
    never rests on an unstated assumption.
    """
    oracle = _load(km)
    ptop, akap = float(oracle["ptop"]), float(oracle["akap"])
    ng, res = int(oracle["ng"]), int(oracle["res"])
    b = slice(ng, ng + res + 1)
    assert (np.asarray(oracle["pk_ogp"], dtype=np.float64)[b, b, 0]
            == ptop ** akap).all(), (
        "PRECONDITION FAILED: gfortran's ptk (ptop**akap, dyn_core:248) "
        "does not equal python's ptop**akap on this host, so `pow` "
        "diverges too and one_grad_p cannot be graded bit-exact either — "
        "the fallback must be extended to seed top_value from the "
        "fixture, and that extension must be RECORDED")
    out = _port_fallback(km)
    _bit_exact(oracle, out, _BITEXACT_OGP)
    _bit_exact(oracle, out, ("u_ogp_dppoison", "v_ogp_dppoison"))


@pytest.mark.parametrize("km", KMS)
def test_grading_buckets_cover_every_token(km):
    """No token may quietly fall out of BOTH buckets: every dumped chain
    token is either graded bit-exact or ULP-bounded, and the fixture's
    recorded grading_policy names the fallback."""
    graded = set(_BITEXACT_GEOPK_C + _BITEXACT_GEOPK_D + _BITEXACT_PGRAD_C
                 + _BITEXACT_OGP + _ULP_GEOPK_C + _ULP_GEOPK_D
                 + ("uc_pgc_dppoison", "vc_pgc_dppoison",
                    "u_ogp_dppoison", "v_ogp_dppoison"))
    missing = set(_CHAIN_TOKENS) - graded
    assert not missing, f"ungraded tokens: {sorted(missing)}"
    assert not (set(_ULP_TOL) - graded - {"logexp_out"})
    oracle = _load(km)
    policy = str(oracle["grading_policy"])
    if int(oracle["libm_probe_ndiff"]):
        assert "SPEC-5.4 FALLBACK APPLIED" in policy
        assert "QUARANTINED" in policy
    else:
        assert "NO FALLBACK NEEDED" in policy


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
    # graded on pe_d, the LIBM-FREE geopk output: q_con's absence must be
    # provable without borrowing the transcendental bucket's tolerance
    assert _nd(out["pe_d"], oracle["pe_d"]) == 0
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
def test_poison_was_installed(km):
    """The poison re-runs prove NOTHING unless the poison was installed
    (codex r20 finding 3).

    Correct hydrostatic behaviour makes the poisoned result EQUAL the
    normal result, and the python generator emits the normal arrays for
    the ``*_DPPOISON`` tokens by construction — so a driver that silently
    dropped the ``delpc = POISON`` / ``delp = POISON`` assignment would
    pass every result comparison.  The driver therefore dumps the
    POISONED ARGUMENT ITSELF immediately before each call
    (``DELPC_POISONED`` / ``DELP_POISONED``); this asserts every element
    carries the poison.  Only the two together — poison installed AND
    output unchanged — establish that the hydrostatic branch never reads
    it.
    """
    oracle = _load(km)
    res, ng = int(oracle["res"]), int(oracle["ng"])
    m_a = res + 2 * ng
    for key in ("delpc_poisoned", "delp_poisoned"):
        a = np.asarray(oracle[key], dtype=np.float64)
        assert a.shape == (m_a, m_a, km), (key, a.shape)
        assert (a == POISON).all(), (
            f"{key}: {int((a != POISON).sum())}/{a.size} elements are NOT "
            f"{POISON} — the driver did not install the poison, so the "
            "*_DPPOISON equality proves nothing")


@pytest.mark.parametrize("km", KMS)
def test_canary_proves_absence_of_writes(km):
    """A CONSTANT sentinel proves a VALUE, not the ABSENCE of a write
    (codex r20 finding 4).

    A kernel that rewrote an "unwritten" slot with the same 1e30 constant
    would pass the sentinel tests.  The driver therefore re-runs both
    geopk calls twice more with DISTINCT, position-dependent canaries.
    Then:

    * a genuinely WRITTEN slot is independent of the canary — the two
      canary runs and the 1e30 main run must all agree BITWISE;
    * a genuinely UNWRITTEN slot keeps its own canary — the two runs must
      each equal their own canary field and DIFFER from each other.

    On the D pass (``computehalo=.true.``) the claim is the opposite:
    the two runs must agree EVERYWHERE, which upgrades "no 1e30 survives"
    into "every slot was actually written".
    """
    gen = _load_gen()
    oracle = _load(km)
    res, ng = int(oracle["res"]), int(oracle["ng"])
    m_a = res + 2 * ng
    box = _box(oracle, extra=1)
    outside = np.ones((m_a, m_a), dtype=bool)
    outside[box, box] = False
    lo = 1 - ng
    cell3, cell_org = (m_a, m_a, km + 1), (lo, lo, 1)
    comp3, comp_org = (res, res, km), (1, 1, 1)

    # --- C pass: written box canary-independent, ring canary-carrying
    for base, shape, org in (("pkc_c", cell3, cell_org),
                             ("gz_c", cell3, cell_org)):
        a1 = np.asarray(oracle[f"{base}_can1"], dtype=np.float64)
        a2 = np.asarray(oracle[f"{base}_can2"], dtype=np.float64)
        main = np.asarray(oracle[base], dtype=np.float64)
        assert _nd(a1[box, box], a2[box, box]) == 0, (
            f"{base}: the WRITTEN box depends on the canary — geopk read "
            "uninitialised memory")
        assert _nd(a1[box, box], main[box, box]) == 0, (
            f"{base}: the written box differs from the 1e30 run")
        c1 = gen.canary_field(shape, org, gen.CANARY_SEEDS["1"])
        c2 = gen.canary_field(shape, org, gen.CANARY_SEEDS["2"])
        assert _nd(a1[outside], c1[outside]) == 0, (
            f"{base}: a slot OUTSIDE the CG write box was overwritten — "
            "the 1e30 sentinel test could not have detected this")
        assert _nd(a2[outside], c2[outside]) == 0, base
        assert (a1[outside] != a2[outside]).all(), base
    # pkz is NEVER written on the CG pass: canary survives everywhere
    z1 = np.asarray(oracle["pkz_c_can1"], dtype=np.float64)
    z2 = np.asarray(oracle["pkz_c_can2"], dtype=np.float64)
    assert _nd(z1, gen.canary_field(comp3, comp_org,
                                    gen.CANARY_SEEDS["1"])) == 0
    assert _nd(z2, gen.canary_field(comp3, comp_org,
                                    gen.CANARY_SEEDS["2"])) == 0
    assert (z1 != z2).all()

    # --- D pass: computehalo=.true. must write EVERY slot
    for base in ("pkc_d", "gz_d", "pkz_d"):
        a1 = np.asarray(oracle[f"{base}_can1"], dtype=np.float64)
        a2 = np.asarray(oracle[f"{base}_can2"], dtype=np.float64)
        main = np.asarray(oracle[base], dtype=np.float64)
        assert _nd(a1, a2) == 0, (
            f"{base}: {_nd(a1, a2)} slots still carry their canary — the "
            "computehalo extension did NOT write the full data domain")
        assert _nd(a1, main) == 0, base


@pytest.mark.parametrize("km", KMS)
def test_per_k_influence_matrix(km):
    """Per-level, per-consumer influence structure (codex r20 finding 5).

    "km=2 differs from km=3" and "some token changed" can both pass while
    a level is zeroed or two levels are swapped.  This asserts the exact
    dependency STRUCTURE the column recursion implies, so a level swap or
    a lost accumulator has nowhere to hide:

      delp[L] -> pe[kk]  changes iff kk >= L+1 (interfaces BELOW the
                         layer), unchanged for kk <= L;
      delp[L] -> pkz[k]  changes iff k >= L;
      pt[L]   -> gz[j]   changes iff j <= L (geopotential ABOVE the
                         layer), unchanged for j > L;
      delpc[L]-> uc_pgc  changes.

    The first row is exactly the "dropping the p1d accumulation must fail
    at interfaces above level 1" requirement: a port using ``delp[k-1]``
    instead of the running sum leaves pe[kk] independent of delp[L] for
    every L < kk-1, and that fires here.
    """
    oracle, gen, flds, base = _port(km)

    def _perturb(key, level):
        pert = {k: np.array(v, dtype=np.float64, copy=True)
                for k, v in flds.items()}
        pert[key][:, :, level] = pert[key][:, :, level] * (1.0 + 1.0e-6)
        return gen.run_port(pert, km)

    for lev in range(km):
        out = _perturb("delp", lev)
        for kk in range(km + 1):
            moved = _nd(out["pe_d"][:, kk, :], base["pe_d"][:, kk, :]) != 0
            if kk >= lev + 1:
                assert moved, (
                    f"delp[{lev}] does not reach interface pe[{kk}] — the "
                    "running p1d accumulator is not being carried down "
                    "the column")
            else:
                assert not moved, (
                    f"delp[{lev}] reached interface pe[{kk}] ABOVE it — "
                    "the column recursion is inverted or mis-indexed")
        for k in range(km):
            moved = _nd(out["pkz_d"][:, :, k], base["pkz_d"][:, :, k]) != 0
            assert moved == (k >= lev), (f"delp[{lev}] -> pkz[{k}]", moved)

        out = _perturb("pt", lev)
        for j in range(km + 1):
            moved = _nd(out["gz_d"][:, :, j], base["gz_d"][:, :, j]) != 0
            assert moved == (j <= lev), (
                f"pt[{lev}] -> gz[{j}] influence is wrong: gz integrates "
                "UPWARD from hs, so pt at level L may only move gz at "
                "interfaces at or above L")

        out = _perturb("delpc", lev)
        assert _nd(out["uc_pgc"], base["uc_pgc"]) != 0, (
            f"delpc[{lev}] does not reach the C-grid wind update")


@pytest.mark.parametrize("km", KMS)
def test_fortran_only_tokens_are_all_asserted(km):
    """Bookkeeping: every FORTRAN-only token (no python counterpart) is
    covered by a test above, so nothing is dumped-but-unconstrained."""
    gen = _load_gen()
    oracle = _load(km)
    covered = {"delpc_poisoned", "delp_poisoned"} | {
        f"{a}_can{s}" for s in ("1", "2")
        for a in ("pkc_c", "gz_c", "pkz_c", "pkc_d", "gz_d", "pkz_d")}
    assert set(gen.FORTRAN_ONLY_TOKENS) == covered
    for key in covered:
        assert key in oracle.files, key
        assert np.isfinite(np.asarray(oracle[key], dtype=np.float64)).all()
    # and they are deliberately OUTSIDE the port-comparison bucket
    assert not (set(gen.FORTRAN_ONLY_TOKENS) & set(_CHAIN_TOKENS))


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
    """The sum-ORDER certificate, graded on the PRESSURE.

    INSTRUMENT CHOICE (this is the fix for the job-9320294 self-reported
    "fixture defect", which was really an INSTRUMENT defect): the check
    runs on ``pe``, not on ``pk``.  ``pk = exp(akap*log(p))`` compresses
    a last-bit pressure difference by the factor ``akap`` and washes out
    ~95% of the sum-order signal — measured 2026-08-04 on this very
    column: 96 -> 2 differing cells at km=3/m=14, 64 -> 11 at m=18.
    ``pe`` is also LIBM-FREE, so this stays a bit-exact gate on a host
    where ``pk`` cannot be one.

    Two-sided, per the coordinator's requirement:
      POSITIVE control — a TOP-DOWN reconstruction must reproduce the
        Fortran ``pe`` BITWISE (so a port that reassociated the sum
        fails here);
      NEGATIVE control — the DELIBERATELY REVERSED (bottom-up)
        accumulation of the same layers must be DETECTED, i.e. differ
        from the Fortran ``pe``.

    STRUCTURAL LIMIT, stated rather than tuned away: at km=2 the surface
    interface is a THREE-addend sum (ptop + d1 + d2) and the two orders
    round identically for essentially every column — an 800-profile
    sweep over the layer split found a maximum of 16/324 cells and 0 for
    every profile of interest.  The negative control is therefore
    ASSERTED at km>=3 and REPORTED at km=2; the km=2 reassociation bug
    is instead caught by ``test_pressure_column_is_accumulated``.
    """
    oracle = _load(km)
    ptop = float(oracle["ptop"])
    ng, res = int(oracle["ng"]), int(oracle["res"])
    # pe origin is (is-1, 1, js-1); delp origin is (isd, jsd) = (1-ng, 1-ng)
    sl = slice(ng - 1, ng + res + 1)
    delp = np.asarray(oracle["in_delp"], dtype=np.float64)[sl, sl, :]
    pe_sfc = np.asarray(oracle["pe_d"], dtype=np.float64)[:, km, :]

    top = np.full(delp.shape[:2], ptop)
    for k in range(km):
        top = top + delp[:, :, k]
    assert _nd(top, pe_sfc) == 0, (
        "the TOP-DOWN reconstruction does not reproduce the Fortran pe "
        "bitwise — the port's running p1d accumulator, its sum order or "
        "its window disagrees with dyn_core:2744")

    bot = np.zeros(delp.shape[:2])
    for k in range(km - 1, -1, -1):
        bot = bot + delp[:, :, k]
    bot = bot + ptop
    n_rev = _nd(bot, pe_sfc)
    n_cells = int(pe_sfc.size)
    # drift check: the count the generator recorded (over the FULL data
    # domain, a wider window than pe's) must still be reproducible
    gen = _load_gen()
    recorded = int(oracle["sumorder_ndiff"])
    assert gen.sumorder_ndiff(oracle["in_delp"], ptop) == recorded, (
        "the recorded sum-order discrimination count is stale")
    assert int(oracle["sumorder_ncells"]) >= n_cells
    if km >= 3:
        assert n_rev > 0, (
            f"the reversed accumulation is INDISTINGUISHABLE from the "
            f"correct one at all {n_cells} cells — the column cannot "
            "discriminate sum order, so this gate is vacuous (a FIXTURE "
            "defect; grade the layer thicknesses).  The generator is "
            "supposed to refuse to pack such a column")
        assert recorded > 0, "the fixture recorded a vacuous sum-order gate"
    else:
        # km=2: three addends, structurally non-discriminating (see the
        # docstring).  Recorded, not asserted, and not faked.
        assert n_rev >= 0


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

"""Phase-4c — duo d_sw1 per-stage oracle.

Certifies ``d_sw1_duo`` (the symmetryclean D-grid TRANSPORT stage, DUO
branch) bit-exact (uint64) against the verbatim Fortran d_sw1
(fv3_dsw1_duo_extract.F90 + the symmetryclean tp_core chain) on the
COMMITTED phase-4b inputs: ut/vt (incl. the deliberate 1e30 workspace
sentinel cells — identical init on both sides makes them bit-comparable),
crx/cry/xfx/yfx, ra_x/ra_y, the allflux slots 1 (delp) + 4 (pt), the
xflux/yflux/cx/cy capacitors.

d_sw1 computes FLUXES only (the delp/pt updates happen in d_sw2 after
dyn_core's inter-panel averaging), so this stage plus the certified duo
c_sw covers the acoustic-step transport numerics; the averaging itself is
6-face infrastructure (documented separately).

Fixture ``dsw1_duo_oracle_c12.npz`` from ``scripts/cluster/fv3_native/
dsw1_duo_oracle.sbatch``: the committed ``dswcore_input.npz`` is
serialised (no regeneration — the pack step refuses on drift), the driver
runs with ``fl%duogrid=.true.`` + ``gs%dg%is_initialized=.true.``, and
``input_sha256`` is re-derivable from the committed npz (enforced below).
"""

import hashlib
import importlib.util
import os

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_gen():
    path = os.path.join(REPO, "scripts", "validate", "fv3_native",
                        "gen_dsw1_duo_oracle.py")
    spec = importlib.util.spec_from_file_location("gen_dsw1_duo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def inputs():
    return np.load(os.path.join(FIX, "dswcore_input.npz"),
                   allow_pickle=True)


@pytest.fixture(scope="module")
def oracle():
    return np.load(os.path.join(FIX, "dsw1_duo_oracle_c12.npz"),
                   allow_pickle=True)


def test_input_hash_enforced(oracle):
    """input_sha256 must equal the hash of the COMMITTED input npz's
    canonical serialisation (the exact bytes the driver read)."""
    gen = _load_gen()
    blob, _res, _ng, _dt = gen.serialize_inputs()
    assert hashlib.sha256(blob).hexdigest() == str(oracle["input_sha256"])


# Drift guard: the fixture certifies THESE extract bytes.  The d_sw1
# extract's single intended deviation from authoritative sw_core.F90
# 500-998 is the ut/vt intent(out)->intent(inout) shim (see its header);
# the tp_core extract is verbatim.  Any edit to either file must
# regenerate the fixture (scripts/cluster/fv3_native/dsw1_duo_oracle
# .sbatch) and re-pin here.
_EXTRACT_SHA = {
    "fv3_dsw1_duo_extract.F90":
        "b91d91462c24deea842cc2cd9bd7ad5fa90e14d760356824ae27d60a7c6b1d15",
    "fv3_tpcore_duo_extract.F90":
        "71f930b3b3b6291955259e46b7dfcd8bcba5c500206c9b0a5377f24c8f8d99c9",
}


def test_extract_sha_pinned():
    for name, want in _EXTRACT_SHA.items():
        path = os.path.join(REPO, "scripts", "validate", "fv3_native", name)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert got == want, (
            f"{name} drifted from the bytes the fixture certifies "
            f"(got sha256 {got}) — regenerate dsw1_duo_oracle_c12.npz "
            f"via the sbatch and re-pin")


def _run_port(inputs):
    from legoesm.core.fv3_native_duo_sw_core import d_sw1_duo
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
    return d_sw1_duo(inputs["delp"], inputs["pt"], inputs["w"],
                     inputs["uc"], inputs["vc"],
                     np.zeros((res + 1, res)), np.zeros((res, res + 1)),
                     np.zeros((res + 1, m_a)), np.zeros((m_a, res + 1)),
                     gs, bd, res + 1, res + 1, dt=float(inputs["dt"]),
                     hord_tr=8, hord_vt=6, hord_tm=6, hord_dp=6,
                     nord_v=1, nord_t=0, damp_v=0.2, damp_t=0.0)


def _bit_check(out, oracle, keys):
    def full(a):
        return np.asarray(a, dtype=np.float64)

    arrs = {
        "crx": full(out["crx_adv"]), "xfx": full(out["xfx_adv"]),
        "rax": full(out["ra_x"]), "cry": full(out["cry_adv"]),
        "yfx": full(out["yfx_adv"]), "ray": full(out["ra_y"]),
        "ut": full(out["ut"]), "vt": full(out["vt"]),
        "afx1": full(out["allflux_x"][:, :, 0]),
        "afx4": full(out["allflux_x"][:, :, 3]),
        "afy1": full(out["allflux_y"][:, :, 0]),
        "afy4": full(out["allflux_y"][:, :, 3]),
        "xflux_out": full(out["xflux"]), "yflux_out": full(out["yflux"]),
        "cx_out": full(out["cx"]), "cy_out": full(out["cy"]),
    }
    for key in keys:
        got = arrs[key]
        want = np.asarray(oracle[key], dtype=np.float64)
        assert got.shape == want.shape, (key, got.shape, want.shape)
        nd = int((got.view(np.uint64) != want.view(np.uint64)).sum())
        assert nd == 0, f"{key}: {nd}/{want.size} words differ bitwise"


def test_dsw1_duo_stage_front_bit_exact(inputs, oracle):
    """The d_sw1 stage FRONT — ut/vt (incl. the 1e30 sentinel workspace
    cells), crx/cry, xfx/yfx, ra_x/ra_y, and the cx/cy Courant
    capacitors — is BIT-exact (uint64) vs the verbatim Fortran duo
    stage on the committed inputs."""
    out = _run_port(inputs)
    _bit_check(out, oracle, ("crx", "xfx", "rax", "cry", "yfx", "ray",
                             "ut", "vt", "cx_out", "cy_out"))


def test_dsw1_duo_fluxes_bit_exact(inputs, oracle):
    """The transport OUTPUT — allflux slots 1 (delp) + 4 (pt) and the
    xflux/yflux capacitors — is BIT-exact (uint64) vs the Fortran duo
    stage.  Root-caused 2026-07-16: the port's gsf handed RAW numpy
    arrays to the fv_tp_2d chain (negative Fortran indices silently
    wrapped from-the-end), corrupting the deln_flux damping increment;
    fort-wrapping the gridstruct fields (mirroring the monolithic
    d_sw's gsf) made every word exact."""
    out = _run_port(inputs)
    _bit_check(out, oracle, ("afx1", "afx4", "afy1", "afy4",
                             "xflux_out", "yflux_out"))


def test_oracle_outputs_finite(oracle):
    """Fixture sanity: all dumped flux/metric tokens finite; ut/vt may
    carry the deliberate 1e30 sentinel (finite too — guards NaN dumps)."""
    for key in ("crx", "xfx", "rax", "cry", "yfx", "ray", "ut", "vt",
                "afx1", "afx4", "afy1", "afy4",
                "xflux_out", "yflux_out", "cx_out", "cy_out"):
        assert np.isfinite(np.asarray(oracle[key], dtype=np.float64)).all(), key


def test_fixture_provenance(oracle):
    lin = str(oracle["input_lineage"])
    assert "COMMITTED" in lin and "duogrid" in lin and "1e30" in lin

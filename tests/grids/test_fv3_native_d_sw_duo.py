"""Duo d_sw gates — fail-loud scope + plain-path preservation (phase-4c).

``d_sw`` now carries VERBATIM transcriptions of the symmetryclean sw_core
DUO branches (see the d_sw docstring for the gate map), but
``duogrid=True`` RAISES: unlike the duo c_sw (certified bit-exact — its
duo branches fully define every read cell), the symmetryclean duo D-grid
PIPELINE is not single-tile well-posed —

1. dyn_core performs INTER-PANEL FLUX AVERAGING between its d_sw1/d_sw2
   stages (mpp_get_boundary + 0.5*(own+neighbor) on the delp/temp fluxes;
   dyn_core.F90:853-900, "averaging ... is fundamental");
2. with the Zenodo reference-run flags (duogrid=T, bounded_domain=F, from
   the rundir input.nml) the always-true d_sw1 edge blocks read ut/vt
   panel-edge workspace cells the duo interior loop never writes, and
   dyn_core's utt/vtt are UNINITIALISED stack arrays — undefined /
   compiler-dependent Fortran results at those cells.

Empirically confirmed here before the raise was added: running the gated
duo path on the phase-4b fixture NaN'd the interior delp/pt/u/v exactly
through that ut(0,*)/vt(*,0) -> yfx/ra_y -> transport chain.  Raising is
the dispatch-hardening doctrine (fail loudly, never return undefined
results).  The gates stay as the verbatim base for future per-stage
certification (d_sw1/d_sw3/d_sw5 with fully-specified inputs) + a
legoESM-side inter-panel flux-averaging analog.

``duogrid=False`` (the default) is byte-identical to the phase-4b plain
d_sw — certified by the 22-test oracle in test_fv3_native_dswcore_phase4b.
"""

import os

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")

_CFG = dict(hord_tr=8, hord_mt=6, hord_vt=6, hord_tm=6, hord_dp=6,
            nord=1, nord_v=1, nord_w=0, nord_t=0,
            dddmp=0.2, d2_bg=0.0, d4_bg=0.12,
            damp_v=0.2, damp_w=0.0, damp_t=0.0,
            d_con=0.0, zvir=0.0, kgb=0.0,
            hydrostatic=True, inline_q=False, use_cond=False,
            do_diss_est=False, sphum=1, nq=1, k=1, km=1, lim_fac=1.0)


@pytest.fixture(scope="module")
def inputs():
    return np.load(os.path.join(FIX, "dswcore_input.npz"))


def _make_gs(inp):
    gs = {k: inp[k] for k in inp.files if k not in ("res", "ng", "dt")}
    gs.update(bounded_domain=False, grid_type=0, do_f3d=False,
              prevent_diss_cooling=False, stretched_grid=False,
              sw_corner=True, se_corner=True, ne_corner=True, nw_corner=True,
              da_min=float(inp["da_min"]), da_min_c=float(inp["da_min_c"]))
    return gs


def _call(inp, duogrid):
    from legoesm.core.fv3_native_d_sw import d_sw
    from legoesm.core.fv3_native_sw_core import Bounds

    res, ng = int(inp["res"]), int(inp["ng"])
    bd = Bounds.single_tile(res, ng)
    m_a = res + 2 * ng
    divg_in = np.array(inp["divg_d_in"], dtype=np.float64, copy=True)
    kw = {} if duogrid is None else {"duogrid": duogrid}
    return d_sw(
        delp=inp["delp"], pt=inp["pt"], w=inp["w"], u=inp["u"], v=inp["v"],
        uc=inp["uc"], vc=inp["vc"], ua=inp["ua"], va=inp["va"],
        divg_d=divg_in,
        xflux=np.zeros((res + 1, res)), yflux=np.zeros((res, res + 1)),
        cx=np.zeros((res + 1, m_a)), cy=np.zeros((m_a, res + 1)),
        gs=_make_gs(inp), bd=bd, npx=res + 1, npy=res + 1,
        dt=float(inp["dt"]), **kw, **_CFG)


def test_duogrid_true_raises_not_single_tile_well_posed(inputs):
    """duogrid=True must FAIL LOUDLY (dispatch-hardening): the duo D-grid
    pipeline needs inter-panel flux averaging + defined panel-edge
    workspace that a single-tile call cannot supply."""
    with pytest.raises(NotImplementedError, match="duogrid.*well-posed"):
        _call(inputs, duogrid=True)


def test_duogrid_false_equals_omitted_default(inputs):
    """duogrid=False and the OMITTED default produce byte-identical output
    on every field (codex duo-d_sw P2 — the earlier test passed False
    explicitly and only checked finiteness).  Byte-identity to the
    pre-gate plain d_sw itself is certified by the phase-4b oracle."""
    out_false = _call(inputs, duogrid=False)
    out_default = _call(inputs, duogrid=None)    # argument omitted
    assert set(out_false) == set(out_default)
    for k in out_false:
        a = np.asarray(out_false[k], dtype=np.float64)
        b = np.asarray(out_default[k], dtype=np.float64)
        assert a.shape == b.shape, k
        # NaN-safe byte equality (unwritten slots are NaN on both sides)
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), k

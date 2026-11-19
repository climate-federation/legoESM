"""NEMO 5.0.1 ORCA1 oracle: zdf_mxl N2-integral MLD uses the run's EOS for bn2.

Fixture ``fixtures/orca1_t3/mld_n2_eos.npz`` (+ ``.json`` provenance): 120 wet
columns of the tier-3 MLD record (kt 17281, day-30 restart, 32 ranks), 60 of
them columns where the S-EOS-locked bn2 put the ML base on the wrong level.
ORCA1 runs TEOS-10 (``ln_teos10=.true.``); NEMO's ``rn2b`` takes alpha/beta from
that EOS (eosbn2.F90 rab_3d), so the MLD integral must too.  NEMO's own
20-seed 1-ulp spread on hmlp is 6.9e-16, so hmlp is checked at rel 1e-12.
"""
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (  # noqa: E402
    _nemo_mld,
    _nemo_wpoint_e3w_wmask_n2,
)
from legoesm.ocean.vertical import (  # noqa: E402
    create_partial_cell_coordinate,
    create_z_star_from_thicknesses,
)

FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "mld_n2_eos.npz"


@pytest.fixture(scope="module")
def case():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    d = dict(np.load(FIX))
    zs = create_z_star_from_thicknesses(
        jnp.asarray(d["e3t_1d"]), jnp.asarray(d["gdept_1d"]),
        nemo_e3w_source="depth_difference")
    tmask = d["tmask"].astype(np.float64)
    e3t = d["e3t"] * tmask
    tmpl = create_partial_cell_coordinate(zs, jnp.asarray(e3t.sum(-1)))
    f64 = lambda x: jnp.asarray(np.asarray(x, np.float64))  # noqa: E731
    # NEMO geometry injected (fp64 ladder, NEMO partial cells and tmask).
    zc = tmpl._replace(
        z_full_ref=f64(zs.z_full_ref), z_half_ref=f64(zs.z_half_ref),
        dz_ref=f64(d["e3t_1d"]), dz_half_ref=f64(zs.dz_half_ref),
        t_depth_ref=f64(d["gdept_1d"]), h_partial=f64(e3t),
        bottom_level=jnp.asarray(np.where(d["mbkt"] > 0, d["mbkt"] - 1, -1),
                                 dtype=tmpl.bottom_level.dtype),
        is_active=jnp.asarray(tmask > 0, dtype=tmpl.is_active.dtype))
    yield d, zc, tmask
    set_policy(prev)


def _no_eos_fn(*_a):
    raise AssertionError("eos_fn must not be read on the NEMO bn2 path")


def _mld(case, eos, eos_fn=_no_eos_fn):
    d, zc, tmask = case
    hml, m_base = _nemo_mld(
        "n2_integral", jnp.asarray(d["T"]), jnp.asarray(d["S"]),
        jnp.asarray(tmask[..., 0]), zc, eos_fn, float(d["rho_c"]),
        g=float(d["grav"]), rho_0=float(d["rho0"]),
        active_3d=jnp.asarray(tmask), jacobian=jnp.asarray(1.0 + d["r3t"]),
        eos=eos)
    return np.asarray(hml), np.asarray(m_base) + 2      # NEMO w-level nmln


def test_teos10_mld_matches_nemo(case):
    d = case[0]
    hml, nmln = _mld(case, "nemo_teos10")
    assert np.array_equal(nmln, d["nmln"]), int((nmln != d["nmln"]).sum())
    rel = np.abs(hml - d["hmlp"]) / np.maximum(np.abs(hml), np.abs(d["hmlp"]))
    assert rel.max() <= 1e-12, rel.max()


def test_unthreaded_eos_keeps_legacy_seos(case):
    # eos=None (direct callers) = the old S-EOS branch: still wrong on ORCA1.
    nmln = _mld(case, None)[1]
    assert (nmln != case[0]["nmln"]).any()


def test_non_nemo_eos_takes_adiabatic_branch(case):
    calls = []

    def eos_fn(T, S, p):
        calls.append(1)
        return 1025.0 - 0.2 * T + 0.8 * S + 0.0 * p

    _mld(case, "wright", eos_fn)
    assert calls


def test_slope_n2_nemo_bn2_uses_run_eos(case):
    d, zc, tmask = case
    T, S = jnp.asarray(d["T"]), jnp.asarray(d["S"])
    args = (T, T, S, zc, _no_eos_fn, float(d["rho0"]), float(d["grav"]),
            jnp.asarray(tmask))
    J = jnp.asarray(1.0 + d["r3t"])
    pn2_teos = _nemo_wpoint_e3w_wmask_n2(
        *args, slope_n2="nemo_bn2", jacobian=J, eos="nemo_teos10")[2]
    pn2_seos = _nemo_wpoint_e3w_wmask_n2(
        *args, slope_n2="nemo_bn2", jacobian=J, eos=None)[2]
    # Oracle: NEMO's recorded rn2b at wet interior W interfaces.
    wet = (tmask[..., 1:] * tmask[..., :-1]) > 0
    a = np.asarray(pn2_teos)[..., 1:][wet]
    b = d["rn2b"][..., 1:][wet]
    rel = np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))
    assert rel.max() <= 1e-11, rel.max()                 # measured 2.9e-12
    assert not np.allclose(np.asarray(pn2_teos), np.asarray(pn2_seos))
    with pytest.raises(ValueError, match="nemo_bn2"):
        _nemo_wpoint_e3w_wmask_n2(
            *args, slope_n2="nemo_bn2", jacobian=J, eos="wright")
    assert GMRediConfig().mld_criterion == "rho_c"

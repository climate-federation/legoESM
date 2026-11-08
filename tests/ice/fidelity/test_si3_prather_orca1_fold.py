"""NEMO 5.0.1 ORCA1 oracle for the Prather halo refresh on the F-pivot north fold.

Fixture ``fixtures/orca1_t4/prather_fold_orca1.{npz,json}``: the top band of the global eORCA1 ice_dyn_adv_pra record
(kt 17281, icycle 1, nn_hls 2) - all 364 extended columns, 10 interior rows, the 2 north fold-halo rows and 2 rows below
- for ``v_i`` and ``a_i`` with their five Prather moments, inputs and outputs with NEMO's recorded halos.  At
``jt == icycle`` NEMO refreshes the moments with ``lbc_lnk`` (icedyn_adv_pra.F90:481-520; signs -1 for sx/sy, +1 for
the rest), so the returned north halo rows are fold images of the advected interior.  The port's halo may not be
further from NEMO than the interior it folds (pass rule: halo error <= interior error of the folded rows).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ice.transport import advect_si3_prather_2d, si3_prather_pack_intensives

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).resolve().parent / "fixtures" / "orca1_t4" / "prather_fold_orca1"
_D = dict(np.load(str(_FIX) + ".npz"))
_TR = ("v_i", "a_i")
_M = ("sx", "sy", "sxx", "syy", "sxy")
_H = 2


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    d = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)))


def _run(north_fold, u_scale=1.0):
    area = jnp.asarray(_D["in_e1e2t"])
    wet = jnp.asarray(_D["in_tmask1"] > 0)
    x = jnp.stack([jnp.asarray(_D["in_" + t]) for t in _TR], -1)
    mom = tuple(jnp.stack([jnp.asarray(_D[f"in_{m}_{t}"]) for t in _TR], -1) for m in _M)
    contents, moments, _ = advect_si3_prather_2d(
        si3_prather_pack_intensives(x, area), mom, jnp.asarray(_D["in_u_ice"]) * u_scale,
        jnp.asarray(_D["in_v_ice"]), area, wet, float(_D["rDt_ice"]),
        dx=jnp.asarray(_D["in_e1v"]), dy=jnp.asarray(_D["in_e2u"]), ice_step_index=int(_D["kt"]),
        nn_fsbc=int(_D["nn_fsbc"]), halo_width=_H, ice_volume_index=0, concentration_index=1,
        subcycles=int(_D["icycle"]), north_fold=north_fold)
    return contents, moments


def _scores(moments):
    halo, inner = 0.0, 0.0
    for mi, m in enumerate(_M):
        for ti, t in enumerate(_TR):
            got = np.asarray(moments[mi])[..., ti]
            want = _D[f"out_{m}_{t}"]
            halo = max(halo, _rel(got[_H:-_H, -_H:], want[_H:-_H, -_H:]))       # north fold-halo rows
            inner = max(inner, _rel(got[_H:-_H, -2 * _H:-_H], want[_H:-_H, -2 * _H:-_H]))   # the rows they fold
    return halo, inner


def test_fixture_is_the_orca1_fold_band():
    assert json.loads(Path(str(_FIX) + ".json").read_text())["oracle"].startswith("NEMO 5.0.1")
    assert int(_D["icycle"]) == 1 and _D["in_v_i"].shape[0] == 364
    assert np.any(_D["out_sx_v_i"][:, -_H:] != 0.0)            # the fold halo carries ice moments


def test_f_pivot_refresh_matches_nemo_fold_halos():
    _, moments = _run("orca_f_pivot")
    halo, inner = _scores(moments)
    assert inner <= 1e-9                                       # tier-4 U10: moments A
    assert halo <= inner + 1e-15


def test_periodic_refresh_is_wrong_at_the_fold():
    _, moments = _run("none")
    halo, inner = _scores(moments)
    assert halo > 1e-3 and inner <= 1e-9


def test_unknown_fold_scheme_is_refused():
    with pytest.raises(ValueError, match="north_fold"):
        _run("t_pivot")


def test_fold_refresh_gradient_is_finite():
    g = jax.grad(lambda s: jnp.sum(_run("orca_f_pivot", s)[1][0][:, -_H:, 0]))(jnp.asarray(1.0))
    assert np.isfinite(float(g)) and float(g) != 0.0

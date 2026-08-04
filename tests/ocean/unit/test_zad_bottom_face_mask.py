"""#1226 level-29-onset fix: ZAD bottom/straddling-face mask convention.

``zad_level29_onset_walk.py`` (commit ``b6d0d9877``) found that NEMO's
``dyn_zad`` (``dynzad.F90:86-119``) has NO per-face ``umask``/``vmask`` guard
inside its ``DO_3D(0,0,0,0,1,jpk-2)`` loop: at a u-face whose OWN T-column
bottoms out at level ``k`` (still "interior" to the fixed loop bound), it
unconditionally reads ``ww`` from BOTH T-neighbours at interface ``k+1`` --
including a neighbour whose bottom is one level DEEPER and is therefore still
genuinely wet there.  Masking is deferred entirely to the velocity update
(``dynzdf.F90:121``, ``puu(Kaa) = (puu(Kbb)+rDt*Krhs) * umask(ji,jj,jk)``),
which zeroes the FINAL TENDENCY by the cell's OWN level -- not an AND of
interface neighbours.

legoESM's pre-fix default ("min_rule") instead zeroed the flux ``G`` at ANY
interface bordering an inactive NEIGHBOUR cell -- discarding that genuine
deeper-neighbour ``ww`` and losing ~100% of the active-only dyn_zad row error
at levels 29-34 (measured, ratio 1.000, zero free parameters).

This module tests the NEW ``bottom_face_mask_mode="nemo_faithful"`` option on
:func:`nemo_advective_vertical_momentum_advection` against an INDEPENDENT
2-column pure-NumPy transcription of ``dynzad.F90:86-119`` (not a call into
the function under test), on a synthetic case where column 0 (the u-face's
"west" T-neighbour) bottoms out one level shallower than column 1 ("east").

fp64 + CPU for deterministic bit-identity checks.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.vertical import (
    VALID_ZAD_BOTTOM_FACE_MASK,
    nemo_advective_vertical_momentum_advection,
)


def _dynzad_two_column_reference(u_w, u_e, ww_w, ww_e, e1e2t_w, e1e2t_e, e3u,
                                  bottom_w, bottom_e):
    """Independent literal Python transcription of dynzad.F90:83-119 AT A
    SINGLE u-FACE between two T-columns ("west"=self, "east"=neighbour), with
    NEMO's own bottom-masking semantics applied EXACTLY as NEMO applies them
    elsewhere (not inside this loop -- dynzad.F90 itself has none):

      - ww/u below a column's OWN bottom level are architecturally zero
        (matches dynzdf.F90:121's *umask on every PRIOR step -- the state
        this loop reads was already masked when it was written).
      - dynzad.F90:86-119 itself applies ZERO additional masking: the u-face
        loop runs jk=0..nlev-2 UNCONDITONALLY and reads ww from BOTH T
        columns raw.
      - The ONLY mask applied to THIS routine's output is the downstream
        dynzdf.F90:121 post-hoc "* umask(jk)" on the u-FACE's own bottom
        level (min(bottom_w, bottom_e), since a u-face is wet only where
        BOTH T-neighbours are wet -- domzgr umask construction).

    Returns the per-level tendency (nlev,), EXCLUDING the r1_e1e2u face-area
    factor (isolated the same way test_vertical_momentum_scheme.py's
    _dynzad_reference does).
    """
    nlev = len(u_w)
    # Architectural zeroing below each column's own bottom (state invariant
    # dynzdf.F90:121 maintains across steps -- NOT part of dynzad.F90 itself).
    u_w = np.array(u_w, dtype=np.float64)
    u_e = np.array(u_e, dtype=np.float64)
    ww_w = np.array(ww_w, dtype=np.float64)
    ww_e = np.array(ww_e, dtype=np.float64)
    u_w[bottom_w + 1:] = 0.0
    u_e[bottom_e + 1:] = 0.0
    ww_w[bottom_w + 1:] = 0.0   # ww at T-point, level index = w-interface below cell k
    ww_e[bottom_e + 1:] = 0.0

    bottom_face = min(bottom_w, bottom_e)   # u-face wet only where BOTH are wet

    tend = np.zeros(nlev)
    zWdzU = 0.0                              # dynzad.F90:83 surface=0
    for jk in range(0, nlev - 1):             # Fortran jk=1..jpk-2 (dynzad.F90:86)
        # dynzad.F90:93-97 -- NO mask, raw ww read from BOTH T-neighbours at
        # interface jk+1 (0-indexed: ww array index jk+1).
        zWf = e1e2t_w * ww_w[jk + 1]
        zWfi = e1e2t_e * ww_e[jk + 1]
        zzWfu = zWfi + zWf
        zzWdzU = zzWfu * (u_w[jk] - u_w[jk + 1])   # dynzad.F90:100, self-column u
        tend[jk] = -0.25 / e3u[jk] * (zWdzU + zzWdzU)   # dynzad.F90:104-105
        zWdzU = zzWdzU                            # dynzad.F90:109
    jk = nlev - 1                                  # dynzad.F90:113 jk=jpkm1
    tend[jk] = -0.25 / e3u[jk] * zWdzU              # dynzad.F90:115-116

    # dynzdf.F90:121 -- the ONLY masking dynzad's Krhs ever receives, keyed
    # on the u-FACE's own level jk (min-rule bottom), not per-interface.
    tend[bottom_face + 1:] = 0.0
    return tend


def _straddling_case(nlev=8, bottom_w=4, bottom_e=5, seed=0):
    """West column bottoms one level shallower than east -- the #1226
    straddling geometry (bl_u_face==bottom_w, east neighbour one level
    deeper and genuinely wet there)."""
    rng = np.random.default_rng(seed)
    e3u = rng.uniform(50.0, 200.0, nlev)
    u_w = rng.normal(0.0, 0.3, nlev)
    u_e = rng.normal(0.0, 0.3, nlev)
    ww_w = rng.normal(0.0, 1.0e-4, nlev + 1)
    ww_e = rng.normal(0.0, 1.0e-4, nlev + 1)
    ww_w[0] = 0.0
    ww_e[0] = 0.0
    e1e2t_w = 1.1e9
    e1e2t_e = 0.95e9
    return u_w, u_e, ww_w, ww_e, e1e2t_w, e1e2t_e, e3u, bottom_w, bottom_e


def _run_lego(mode, u_w, u_e, ww_w, ww_e, e1e2t_w, e1e2t_e, e3u,
              bottom_w, bottom_e):
    """Drive nemo_advective_vertical_momentum_advection at the west u-face
    (shape (1,1,nlev), single face) with the given bottom_face_mask_mode."""
    nlev = len(u_w)
    # w_area_half at the u-FACE = e1e2t_w*ww_w + e1e2t_e*ww_e (matches the
    # module's own w_area_half convention: interp_cell_to_uface(area_T*w) is
    # exactly the (unweighted, since interp_cell_to_uface is a plain mean of
    # the TWO already-area-weighted fields *2 folded into the 2.0 in G) sum
    # dynzad.F90:97 computes -- see nemo_advective_vertical_momentum_advection
    # docstring "zzWfu = zWfi + zWf ... = 2*mean(e1e2t*ww) at u-face").
    # Architectural zeroing below each column's OWN bottom (the state
    # invariant dynzdf.F90:121 maintains every step -- both u and ww are
    # already zero there when this routine reads them; NOT part of
    # dynzad.F90 itself, but a precondition the reference also assumes).
    u_w = np.array(u_w, dtype=np.float64)
    ww_w = np.array(ww_w, dtype=np.float64)
    ww_e = np.array(ww_e, dtype=np.float64)
    u_w[bottom_w + 1:] = 0.0
    ww_w[bottom_w + 1:] = 0.0
    ww_e[bottom_e + 1:] = 0.0

    area_w_w = e1e2t_w * ww_w
    area_w_e = e1e2t_e * ww_e
    w_area_half = 0.5 * (area_w_w + area_w_e)   # mean; module doubles internally

    u3 = jnp.asarray(u_w)[None, None, :]
    w3 = jnp.asarray(w_area_half)[None, None, :]
    h3 = jnp.asarray(e3u)[None, None, :]
    face_area = jnp.ones((1, 1, 1))

    bottom_face = min(bottom_w, bottom_e)
    face_active = np.ones(nlev)
    face_active[bottom_face + 1:] = 0.0
    face_active3 = jnp.asarray(face_active)[None, None, :]

    got = nemo_advective_vertical_momentum_advection(
        u3, w3, h3, face_area, face_active=face_active3,
        bottom_face_mask_mode=mode,
    )
    return np.asarray(got)[0, 0]


# ===========================================================================
# 1. PRE-FIX FAILURE: "min_rule" does NOT match dynzad.F90 at the straddling
#    face's own last-wet cell (quote this failing BEFORE trusting the fix).
# ===========================================================================
def test_min_rule_does_not_match_dynzad_at_straddling_face():
    """Non-vacuous pre-fix demonstration: the OLD default masks away the
    genuine deeper-neighbour ww contribution, so it does NOT match the
    independent dynzad.F90 transcription at the straddling face's own
    bottom cell (bottom_w). This is the #1226 defect itself."""
    args = _straddling_case()
    u_w, u_e, ww_w, ww_e, e1e2t_w, e1e2t_e, e3u, bottom_w, bottom_e = args
    ref = _dynzad_two_column_reference(*args)
    got_min_rule = _run_lego("min_rule", *args)

    # At bottom_w (the straddling face's own last-wet cell), min_rule must
    # DIFFER from dynzad's true (unmasked-input) answer -- this is the bug.
    diff_at_bottom = abs(got_min_rule[bottom_w] - ref[bottom_w])
    assert diff_at_bottom > 1e-6, (
        "expected min_rule to mismatch dynzad at the straddling face's own "
        f"bottom cell (bottom_w={bottom_w}); got diff={diff_at_bottom:.3e} "
        "-- pre-fix defect not reproduced, check the test setup")


# ===========================================================================
# 2. POST-FIX: "nemo_faithful" matches the independent transcription exactly
#    at EVERY level, including the straddling bottom cell.
# ===========================================================================
def test_nemo_faithful_matches_independent_transcription():
    args = _straddling_case()
    _, _, _, _, _, _, _, bottom_w, bottom_e = args
    ref = _dynzad_two_column_reference(*args)
    got = _run_lego("nemo_faithful", *args)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("bottom_w,bottom_e,seed", [(4, 5, 0), (3, 6, 1), (5, 4, 2)])
def test_nemo_faithful_matches_independent_transcription_multi(bottom_w, bottom_e, seed):
    """Generalises across straddle depth and direction (east deeper AND
    west deeper) -- dynzad.F90 has no asymmetry between the two neighbours."""
    args = _straddling_case(bottom_w=bottom_w, bottom_e=bottom_e, seed=seed)
    ref = _dynzad_two_column_reference(*args)
    got = _run_lego("nemo_faithful", *args)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


def test_nemo_faithful_zeros_below_own_seafloor():
    """Regardless of mask mode, the cell strictly below the FACE's own
    (min-rule) bottom must be exactly zero -- dynzdf.F90:121 always masks
    there; this is not the part of the convention that changed."""
    args = _straddling_case()
    _, _, _, _, _, _, _, bottom_w, bottom_e = args
    bottom_face = min(bottom_w, bottom_e)
    got = _run_lego("nemo_faithful", *args)
    assert np.all(got[bottom_face + 1:] == 0.0)


# ===========================================================================
# 3. DEFAULT PATH BIT-IDENTITY: omitting bottom_face_mask_mode (or passing
#    "min_rule" explicitly) must be EXACTLY unchanged (max|diff| == 0.0).
# ===========================================================================
def test_default_path_bit_identical_to_pre_fix_min_rule():
    args = _straddling_case()
    got_default = _run_lego("min_rule", *args)
    # Call again with the mode omitted entirely (uses the function default).
    u_w, u_e, ww_w, ww_e, e1e2t_w, e1e2t_e, e3u, bottom_w, bottom_e = args
    area_w_w = e1e2t_w * np.asarray(ww_w)
    area_w_e = e1e2t_e * np.asarray(ww_e)
    w_area_half = 0.5 * (area_w_w + area_w_e)
    u3 = jnp.asarray(u_w)[None, None, :]
    w3 = jnp.asarray(w_area_half)[None, None, :]
    h3 = jnp.asarray(e3u)[None, None, :]
    face_area = jnp.ones((1, 1, 1))
    bottom_face = min(bottom_w, bottom_e)
    face_active = np.ones(len(u_w))
    face_active[bottom_face + 1:] = 0.0
    face_active3 = jnp.asarray(face_active)[None, None, :]
    got_omitted = np.asarray(
        nemo_advective_vertical_momentum_advection(
            u3, w3, h3, face_area, face_active=face_active3))[0, 0]
    max_abs_diff = float(np.max(np.abs(got_omitted - got_default)))
    assert max_abs_diff == 0.0, f"default-path bit-identity broken: max|diff|={max_abs_diff!r}"


# ===========================================================================
# 4. DISPATCH: unknown bottom_face_mask_mode raises.
# ===========================================================================
def test_unknown_bottom_face_mask_mode_raises():
    args = _straddling_case()
    with pytest.raises(ValueError, match="bottom_face_mask_mode must be one of"):
        _run_lego("typo_mode", *args)


def test_dino_config_unknown_zad_bottom_face_mask_raises():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    grid = create_latlon_grid(4, 4)
    z = create_ocean_z_star(n_levels=4, H_max=4000.0)
    with pytest.raises(ValueError, match="zad_bottom_face_mask must be one of"):
        LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(
            zad_bottom_face_mask="typo"))


def test_valid_zad_bottom_face_mask_contents():
    assert VALID_ZAD_BOTTOM_FACE_MASK == frozenset({"min_rule", "nemo_faithful"})


# ===========================================================================
# 5. DINO CARD WIRING: nemo_dino_kamm(+_mlf) select nemo_faithful; the
#    default recipe keeps min_rule (bit-identical).
# ===========================================================================
@pytest.mark.parametrize("recipe", ["nemo_dino_kamm", "nemo_dino_kamm_mlf"])
def test_dino_kamm_cards_select_nemo_faithful(recipe):
    from legoesm.ocean.experiments.dino import dino_config_for_recipe
    cfg = dino_config_for_recipe(recipe)
    assert cfg.zad_bottom_face_mask == "nemo_faithful"


def test_dino_default_card_keeps_min_rule():
    from legoesm.ocean.experiments.dino import dino_config_for_recipe, DINOConfig
    assert DINOConfig().zad_bottom_face_mask == "min_rule"
    assert (dino_config_for_recipe("legoesm_default").zad_bottom_face_mask
            == "min_rule")

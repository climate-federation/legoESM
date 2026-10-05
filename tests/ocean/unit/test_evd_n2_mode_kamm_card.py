"""The DINO/NEMO Kamm cards must feed the EVD trigger NEMO's own ``bn2``.

NEMO's enhanced-vertical-diffusion convective adjustment (``zdfevd.F90:93``
for ``avt`` and ``:119`` for ``avm``) is a **sign test** on ``rn2``/``rn2b``
at ``-1e-12``, and those two arrays are produced by ``bn2``
(``eosbn2.F90:1459-1466``): local ``alpha``/``beta`` evaluated at each cell's
own ``gdept`` (``rab_3d_t``, ``eosbn2.F90:1161-1173``, ``np_seos`` branch),
interpolated to the w-point by the geometric ``zrw`` weight, then differenced
**linearly** in T and S.

legoESM transcribes that exactly as ``n2_mode="nemo_bn2"``.  The Kamm cards
previously selected ``"adiabatic"`` instead — the parcel-displacement N², in
which both interface cells are pushed to the upper cell's pressure and the
**full nonlinear in-situ density** is differenced — on the grounds that the
two are "numerically equivalent (corr 1.0000)".  They are not equivalent for a
sign test: measured on NEMO's own state the parcel form misses 48 of NEMO's
60845 firing interfaces, and one of those 48 carried 99.995% of the run's
largest per-step tracer error.

These tests lock (a) the card selection, and (b) the reason it matters — on a
compensated front (warmer AND saltier below) tuned to sit ~0.15% from the sign
knife-edge, the two modes disagree in SIGN and therefore give *different
convective diffusivities*.  Test (b) is the non-vacuous one: it fails if the
two N² formulas are ever made equivalent, and it fails if the trigger stops
reading the mode.
"""
import jax

# fp64 for the knife-edge N2 comparisons.  NOT the env var: tests/conftest.py
# imports jax before any test module loads, so an os.environ.setdefault here
# never reaches jax's config and the file ran in float32 under a bare pytest
# invocation (3 pre-existing failures).  jax.config.update is honoured
# whenever it runs -- same pattern as test_barotropic_wide_halo.py.
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    NemoSEOSConfig,
    compute_buoyancy_frequency_adiabatic,
    compute_buoyancy_frequency_nemo_bn2,
    compute_hydrostatic_pressure,
    make_eos_fn,
    nemo_bn2_live_ladders,
)
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    create_dino_z_star,
    dino_config_for_recipe,
    dino_lat_lon_grid,
    dino_lat_lon_model_config,
)
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.enhanced_diffusion import (
    convective_K_A_flag,
)
# Reuse the existing independent NumPy transcription of eosbn2.F90 rather than
# writing a second one (pre-impl grep: tests/ocean/unit/test_nemo_bn2.py).
from tests.ocean.unit.test_nemo_bn2 import _numpy_bn2

KAMM_CARDS = ("nemo_dino_kamm", "nemo_dino_kamm_mlf")

# --- the tuned compensated-front column (see the module docstring) ---------
_KTOP = 9              # interface index: upper cell 9 over lower cell 10
_DT_WARM_BELOW = 0.012   # [degC] lower cell warmer  -> destabilising
_DS_SALT_BELOW = 0.0033475  # [psu]  lower cell saltier -> stabilising
_THR = -1e-12          # NEMO zdfevd rn2 threshold (namzdf, the Kamm card)


@pytest.mark.parametrize("recipe", KAMM_CARDS)
def test_kamm_card_evd_trigger_uses_nemo_bn2(recipe):
    """The trigger's N² mode, read off the ASSEMBLED physics config."""
    cfg = dino_config_for_recipe(recipe)
    grid = dino_lat_lon_grid(cfg, n_lon=8)
    _, physics = dino_lat_lon_model_config(grid, cfg)
    ed = physics.convection.enhanced_diffusion
    assert ed.n2_mode == "nemo_bn2", (
        f"{recipe}: the EVD trigger must consume NEMO's bn2 "
        f"(eosbn2.F90:1459-1466), got n2_mode={ed.n2_mode!r}")
    # The rest of the NEMO zdfevd trigger contract, so a card edit that fixes
    # the mode while breaking the threshold or the hard switch still goes red.
    assert ed.smooth_transition is False
    assert ed.n2_threshold == _THR
    # The two OTHER N² consumers are SEPARATE fields and already NEMO-faithful;
    # this pins that the convection flip did not silently move them.
    assert cfg.tke_n2_mode == "nemo_bn2"
    assert cfg.gm_redi_slope_n2 == "nemo_bn2"


def test_non_kamm_dino_cards_keep_their_n2_mode():
    """The flip is card-local: no other DINO recipe's trigger moved."""
    assert dino_config_for_recipe("nemo_paper").convection_n2_mode == "adiabatic"
    assert dino_config_for_recipe("legoesm_default").convection_n2_mode == "insitu"


def _compensated_front_column():
    """A near-neutral column with one compensated front at ``_KTOP``.

    Warmer AND saltier below: the thermal term destabilises, the haline term
    stabilises, and they very nearly cancel — the regime the whole DINO
    convective population lives in (|N²| ~ 1e-10..1e-12, five orders below the
    stratified interior).  ``_DS_SALT_BELOW`` is tuned so the column sits
    between the two formulas' zero crossings, which are 0.149% apart in the
    salinity contrast.
    """
    dcfg = DINOConfig()
    z_coord = create_dino_z_star(dcfg)
    nlev = int(np.asarray(z_coord.dz_ref).shape[0])
    T = np.linspace(14.8, 14.0, nlev)
    S = np.linspace(35.00, 35.02, nlev)
    T[_KTOP + 1:] = T[_KTOP] + _DT_WARM_BELOW
    S[_KTOP + 1:] = S[_KTOP] + _DS_SALT_BELOW
    return z_coord, float(dcfg.H_deep), T, S


def _both_n2(z_coord, H_deep, T, S):
    """(N²_nemo_bn2, N²_adiabatic, gdept, gdepw_int) on that column.

    Wired the way production wires them: the bn2 path takes the live
    ``gdept(Kmm)`` ladders (``eos.nemo_bn2_live_ladders``, as
    ``k_profiles``/``enhanced_diffusion`` do), the adiabatic path takes the
    cell-centre hydrostatic pressure and the model EOS.  ``eta = 0`` so the z*
    stretch is exactly 1 and the two paths differ ONLY in the N² formula.
    """
    eos_fn = make_eos_fn("nemo_seos")
    Tj = jnp.asarray(T)[None, None, :]
    Sj = jnp.asarray(S)[None, None, :]
    eta = jnp.zeros((1, 1))
    J = jnp.ones((1, 1))
    gdept, gdepw = nemo_bn2_live_ladders(z_coord, jnp.zeros(()),
                                         jnp.asarray(H_deep))
    rho_0 = NemoSEOSConfig().rho0
    # Boussinesq depth->pressure for the first density pass, then the true
    # hydrostatic integral for p_cell (the quantity the adiabatic form wants).
    rho = eos_fn(Tj, Sj,
                 jnp.asarray(np.asarray(gdept))[None, None, :] * constants.g * rho_0)
    p_cell = compute_hydrostatic_pressure(rho, eta, z_coord.dz_ref, J, rho_0)
    n2_bn2 = compute_buoyancy_frequency_nemo_bn2(
        Tj, Sj, gdept[None, None, :], gdepw[None, None, :], g=constants.g,
        e3w_source="depth_difference")
    n2_adia = compute_buoyancy_frequency_adiabatic(
        Tj, Sj, p_cell, z_coord.dz_ref, J, eos_fn=eos_fn, rho_ref=rho_0,
        g=constants.g)
    return (np.asarray(n2_bn2)[0, 0], np.asarray(n2_adia)[0, 0],
            np.asarray(gdept), np.asarray(gdepw))


def test_compensated_front_two_n2_modes_disagree_in_sign():
    """At the knife-edge the parcel form has the WRONG SIGN for the trigger."""
    z_coord, H_deep, T, S = _compensated_front_column()
    n2_bn2, n2_adia, _, _ = _both_n2(z_coord, H_deep, T, S)
    b, a = float(n2_bn2[_KTOP]), float(n2_adia[_KTOP])
    assert b < _THR, f"NEMO bn2 must see the front as unstable, got {b:.6e}"
    assert a > _THR, f"parcel form must MISS it, got {a:.6e}"
    # Both sit in the marginal band that correlation cannot resolve — this is
    # the point: a bulk metric on |N²| ~ 1e-9 against a 1e-5 interior is blind.
    assert abs(b) < 1e-8 and abs(a) < 1e-8


def test_compensated_front_flips_the_convective_diffusivity():
    """The TRIGGER, not just N²: K jumps K_bg -> K_conv when the mode is right.

    This is the acceptance test for the card flip.  It exercises the same
    ``convective_K_A_flag`` the explicit and implicit convection paths both
    call, with the Kamm card's own hard-switch settings.
    """
    z_coord, H_deep, T, S = _compensated_front_column()
    eos_fn = make_eos_fn("nemo_seos")
    Tj = jnp.asarray(T)[None, None, :]
    Sj = jnp.asarray(S)[None, None, :]
    J = jnp.ones((1, 1))
    rho_0 = NemoSEOSConfig().rho0
    gdept, gdepw = nemo_bn2_live_ladders(z_coord, jnp.zeros(()),
                                         jnp.asarray(H_deep))
    rho = eos_fn(Tj, Sj,
                 jnp.asarray(np.asarray(gdept))[None, None, :] * constants.g * rho_0)
    p_cell = compute_hydrostatic_pressure(rho, jnp.zeros((1, 1)),
                                          z_coord.dz_ref, J, rho_0)

    def K_for(mode):
        cfg = EnhancedDiffusionConfig(
            K_conv=100.0, K_bg=1e-5, smooth_transition=False,
            n2_threshold=_THR, n2_mode=mode)
        K, _, flag = convective_K_A_flag(
            rho, z_coord.dz_ref, J, cfg, T=Tj, S=Sj, p_cell=p_cell,
            eos_fn=eos_fn, t_depth=gdept[None, None, :],
            w_depth=gdepw[None, None, :],
            e3w_int=jnp.diff(gdept, axis=-1)[None, None, :],
            g=constants.g, rho_ref=rho_0)
        return float(np.asarray(K)[0, 0, _KTOP]), float(np.asarray(flag)[0, 0, _KTOP])

    K_nemo, flag_nemo = K_for("nemo_bn2")
    K_parcel, flag_parcel = K_for("adiabatic")
    assert (K_nemo, flag_nemo) == (100.0, 1.0), (
        f"NEMO bn2 trigger must fire: K={K_nemo}, flag={flag_nemo}")
    assert (K_parcel, flag_parcel) == (1e-5, 0.0), (
        f"parcel trigger must miss: K={K_parcel}, flag={flag_parcel}")


def test_knife_edge_bn2_matches_nemo_fortran_transcription():
    """legoESM's bn2 == an independent loop transcription of eosbn2.F90.

    Anchors the previous test's verdict to the ORACLE rather than to our own
    vectorised kernel: if the transcription drifted, "the parcel form is
    wrong" would be an unsupported claim.
    """
    z_coord, H_deep, T, S = _compensated_front_column()
    n2_bn2, _, gdept, gdepw = _both_n2(z_coord, H_deep, T, S)
    ref = _numpy_bn2(T, S, gdept, gdepw, NemoSEOSConfig(), constants.g)
    assert np.allclose(n2_bn2, ref, rtol=0, atol=1e-18), (
        np.max(np.abs(n2_bn2 - ref)))
    assert ref[_KTOP] < _THR


# ---------------------------------------------------------------------------
# EnhancedDiffusionConfig.n2_eos_form -- WHICH alpha/beta the bn2 trigger uses
# ---------------------------------------------------------------------------
# The EVD trigger's N2 was hard-wired to the simplified EOS: the bn2 call
# omitted eos_form and so always took the "seos" default, even on a card whose
# TKE sibling (TKEConfig.n2_eos_form, threaded in vertical_mixing/_shared.py)
# was already running the TEOS-10 polynomial.  Same axis, same name, same
# default here.


def _evd_n2_column(n2_eos_form, smooth_transition=False):
    """(K, flag) at the front interface with the given alpha/beta selector."""
    z_coord, H_deep, T, S = _compensated_front_column()
    eos_fn = make_eos_fn("nemo_seos")
    Tj = jnp.asarray(T)[None, None, :]
    Sj = jnp.asarray(S)[None, None, :]
    J = jnp.ones((1, 1))
    rho_0 = NemoSEOSConfig().rho0
    gdept, gdepw = nemo_bn2_live_ladders(z_coord, jnp.zeros(()),
                                         jnp.asarray(H_deep))
    rho = eos_fn(Tj, Sj, jnp.asarray(np.asarray(gdept))[None, None, :]
                 * constants.g * rho_0)
    kwargs = {} if n2_eos_form is None else {"n2_eos_form": n2_eos_form}
    cfg = EnhancedDiffusionConfig(
        K_conv=100.0, K_bg=1e-5, smooth_transition=smooth_transition,
        sigmoid_sharpness=1e10, n2_threshold=_THR, n2_mode="nemo_bn2",
        **kwargs)
    K, _, flag = convective_K_A_flag(
        rho, z_coord.dz_ref, J, cfg, T=Tj, S=Sj,
        t_depth=gdept[None, None, :], w_depth=gdepw[None, None, :],
        e3w_int=jnp.diff(gdept, axis=-1)[None, None, :],
        g=constants.g, rho_ref=rho_0)
    return np.asarray(K)[0, 0], np.asarray(flag)[0, 0]


def test_evd_n2_eos_form_default_is_seos_and_bit_identical():
    """Omitting the field is BYTE-identical to selecting 'seos' explicitly."""
    assert EnhancedDiffusionConfig().n2_eos_form == "seos"
    K_default, flag_default = _evd_n2_column(None)
    K_seos, flag_seos = _evd_n2_column("seos")
    assert np.array_equal(K_default, K_seos)
    assert np.array_equal(flag_default, flag_seos)


def test_evd_n2_eos_form_reaches_the_bn2_kernel():
    """The selector is FORWARDED: 'teos10' gives the trigger a different N2.

    Non-vacuity: run the smooth (sigmoid) trigger so the comparison sees the
    continuous N2 rather than a saturated 0/1 flag, and check against the
    kernel called directly with each eos_form.  Before the fix both arms
    returned the 'seos' answer, so the arrays were byte-equal.
    """
    z_coord, H_deep, T, S = _compensated_front_column()
    Tj = jnp.asarray(T)[None, None, :]
    Sj = jnp.asarray(S)[None, None, :]
    gdept, gdepw = nemo_bn2_live_ladders(z_coord, jnp.zeros(()),
                                         jnp.asarray(H_deep))
    n2_seos = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        Tj, Sj, gdept[None, None, :], gdepw[None, None, :],
        g=constants.g, eos_form="seos",
        e3w_source="depth_difference"))[0, 0]
    n2_teos = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        Tj, Sj, gdept[None, None, :], gdepw[None, None, :],
        g=constants.g, eos_form="teos10",
        e3w_source="depth_difference"))[0, 0]
    assert not np.allclose(n2_seos, n2_teos, rtol=1e-3, atol=0), (
        "the two alpha/beta sets agree on this column -- the test below "
        "cannot detect a dropped forward")

    _, flag_seos = _evd_n2_column("seos", smooth_transition=True)
    _, flag_teos = _evd_n2_column("teos10", smooth_transition=True)
    assert not np.array_equal(flag_seos, flag_teos), (
        "convective_K_A_flag ignored n2_eos_form -- the forward to "
        "compute_buoyancy_frequency_nemo_bn2 is missing again")


def test_evd_n2_eos_form_unknown_raises():
    """Dispatch hardening: a typo raises at function entry, not silently."""
    z_coord, H_deep, T, S = _compensated_front_column()
    Tj = jnp.asarray(T)[None, None, :]
    Sj = jnp.asarray(S)[None, None, :]
    J = jnp.ones((1, 1))
    rho_0 = NemoSEOSConfig().rho0
    gdept, gdepw = nemo_bn2_live_ladders(z_coord, jnp.zeros(()),
                                         jnp.asarray(H_deep))
    rho = make_eos_fn("nemo_seos")(
        Tj, Sj,
        jnp.asarray(np.asarray(gdept))[None, None, :] * constants.g * rho_0)
    # Raises even under an n2_mode whose branch never reaches the bn2 kernel.
    for mode in ("nemo_bn2", "insitu"):
        cfg = EnhancedDiffusionConfig(n2_mode=mode, n2_eos_form="bogus")
        with pytest.raises(ValueError, match="n2_eos_form"):
            convective_K_A_flag(
                rho, z_coord.dz_ref, J, cfg, T=Tj, S=Sj,
                t_depth=gdept[None, None, :], w_depth=gdepw[None, None, :],
                g=constants.g, rho_ref=rho_0)


def test_evd_n2_eos_form_accepts_the_eos80_arm_the_orca2_card_selects():
    """ORCA2 runs ``ln_eos80 = .true.``; the trigger must accept and use it.

    Before this arm existed the call RAISED, so "it runs" is already the
    headline.  Non-vacuity follows the TEOS-10 test's shape: the EOS-80 and
    simplified sets must give a different N2 on this column and the smooth
    trigger must then differ, so a dropped forward cannot pass.  EOS-80 and
    TEOS-10 are deliberately NOT compared here -- they are two fits of the
    same seawater and agree far inside this trigger's threshold; the kernel
    test in ``test_nemo_roquet_eos.py`` pins that they are different numbers.
    """
    z_coord, H_deep, T, S = _compensated_front_column()
    Tj = jnp.asarray(T)[None, None, :]
    Sj = jnp.asarray(S)[None, None, :]
    gdept, gdepw = nemo_bn2_live_ladders(z_coord, jnp.zeros(()),
                                         jnp.asarray(H_deep))
    n2_seos = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        Tj, Sj, gdept[None, None, :], gdepw[None, None, :],
        g=constants.g, eos_form="seos",
        e3w_source="depth_difference"))[0, 0]
    n2_eos80 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        Tj, Sj, gdept[None, None, :], gdepw[None, None, :],
        g=constants.g, eos_form="eos80",
        e3w_source="depth_difference"))[0, 0]
    assert not np.allclose(n2_seos, n2_eos80, rtol=1e-3, atol=0), (
        "the EOS-80 and simplified sets agree on this column -- the check "
        "below cannot detect a dropped forward")

    K_eos80, flag_eos80 = _evd_n2_column("eos80", smooth_transition=True)
    assert np.all(np.isfinite(K_eos80)) and np.all(np.isfinite(flag_eos80))
    _, flag_seos = _evd_n2_column("seos", smooth_transition=True)
    assert not np.array_equal(flag_seos, flag_eos80), (
        "convective_K_A_flag ignored the eos80 arm")

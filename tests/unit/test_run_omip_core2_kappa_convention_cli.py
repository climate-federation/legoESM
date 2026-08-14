"""``--tke-kappa-convention`` and the ORCA1 card's NEMO K-from-TKE amplitude.

NEMO's tracer/momentum coefficient is ``avm = MAX(avtb, rn_ediff*zmxlm*en^1/2)``
(``zdftke.F90:150``, and the ``tke_avn`` header at ``:553``) — the square root
carries ``en``, NOT ``2*en``.  The factor 2 lives in the LENGTH,
``zmxlm = SQRT(2*en/rn2)`` (``:651``).  legoESM's ``TKEConfig`` default
``kappa_convention="gaspar_sqrt2e"`` puts ``sqrt(2*e)`` in the amplitude too,
and the ``tke_mxl_choice`` 3/4 branch already builds its length from
``sqrt(2e)/N``, so the default double-counts the ``sqrt(2)`` on exactly the
branch the ORCA1 card selects.  Measured against NEMO's own state (Stage A,
commit 39ce0701c): 4.81x NEMO's ``avt`` with the default, 3.25x with the NEMO
form.  The card now pins ``veros_sqrte``; this module holds that pin down.
"""
from __future__ import annotations

import math

import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_card_pins_the_nemo_tke_diffusion_coefficient():
    """NEMO diffuses `en` with the PLAIN viscosity, not 30x it.

    zdftke.F90:407-409 assembles zzd_up = -0.5*rn_Dt*(avm(k+1)+avm(k)) /
    (e3t*e3w), a face coefficient of mean(avm) — i.e. alpha_tke = 1.
    TKEConfig's default is the Veros/Gaspar 30.0, and its own __param_spec__
    reference says so ("NEMO avm x1 (zdftke); Veros/Gaspar 30").  Measured on
    NEMO's own state and en (job 9405026): with 30 our one-step diffusivity is
    1.846x NEMO's, with 1.0 it is 0.634x, while the ZERO-step value is 0.861x
    either way — the coefficient only acts through the step.
    """
    cfg = _core2().orca1_zdftke_config()
    assert cfg.alpha_tke == 1.0, (
        "orca1_zdftke_config no longer pins NEMO's TKE diffusion coefficient; "
        "the TKEConfig default of 30.0 is the Veros/Gaspar value and diffuses "
        "TKE thirty times too fast against a NEMO oracle.")
    vm = _core2().build_tripole_vmix_config("tke", iwm=None)
    assert vm.tke.alpha_tke == 1.0, "the MPAS/tripole builder lost the pin"


def test_card_pins_the_nemo_amplitude():
    """The ORCA1 card must select NEMO's ``rn_ediff*zmxlm*sqrt(en)`` form."""
    cfg = _core2().orca1_zdftke_config()
    assert cfg.kappa_convention == "veros_sqrte", (
        "orca1_zdftke_config no longer pins the NEMO K amplitude; the "
        "TKEConfig default gaspar_sqrt2e double-counts the sqrt(2) that the "
        "nn_mxl=2/3 buoyancy length sqrt(2e)/N already carries.")


def test_mpas_call_site_inherits_the_pin():
    """--mpas-vmix tke reaches the SAME card, so the fix must reach MPAS too.

    The MPAS branch calls ``build_tripole_vmix_config("tke", iwm=None)`` with
    no overrides; if that ever stopped resolving to the ORCA1 card, the two
    grids would run different closures and the cross-grid comparison would be
    measuring the code, not the physics.
    """
    vm = _core2().build_tripole_vmix_config("tke", iwm=None)
    assert vm.scheme == "tke"
    assert vm.tke.kappa_convention == "veros_sqrte"


@pytest.mark.parametrize("value", ["veros_sqrte", "gaspar_sqrt2e"])
def test_cli_roundtrip_reaches_the_config(value):
    """--tke-kappa-convention parses and lands on the TKEConfig leaf."""
    core2 = _core2()
    p = core2._build_arg_parser()
    a = p.parse_args(["--grid", "tripole", "--tripole-vmix", "tke",
                      "--tke-kappa-convention", value])
    assert a.tke_kappa_convention == value
    vm = core2.build_tripole_vmix_config(
        "tke", iwm=None, tke_kappa_convention=a.tke_kappa_convention)
    assert vm.tke.kappa_convention == value


def test_flag_without_the_tke_closure_raises():
    """Silent-discard footgun: the knob only exists inside the tke branch."""
    core2 = _core2()
    with pytest.raises(SystemExit, match="tke-kappa-convention"):
        core2._validate_tke_card_grid(
            "tripole", tripole_vmix="kpp", tke_kappa_convention="veros_sqrte")
    with pytest.raises(SystemExit, match="tke-kappa-convention"):
        core2._validate_tke_card_grid(
            "mpas", tripole_vmix="tke", tke_kappa_convention="veros_sqrte")
    with pytest.raises(ValueError, match="tke-kappa-convention"):
        core2.build_tripole_vmix_config(
            "kpp", iwm=None, tke_kappa_convention="veros_sqrte")


def test_unknown_convention_raises_not_defaults():
    """Dispatch hardening: a typo must not silently pick an amplitude."""
    with pytest.raises(ValueError, match="kappa_convention"):
        _core2().orca1_zdftke_config(kappa_convention="gaspar")


def test_the_pin_is_load_bearing_sqrt2():
    """NON-VACUITY: the two conventions must actually differ, by sqrt(2).

    A pin that changed no number would be ceremony.  Drive the model's own
    ``compute_K_from_tke`` with both cards on identical inputs, in a regime
    chosen so neither the kappaM floor nor the kappaM_max ceiling binds, and
    require the ratio to be sqrt(2) to within float tolerance.
    """
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing.tke import compute_K_from_tke

    core2 = _core2()
    c_nemo = core2.orca1_zdftke_config()
    c_legacy = core2.orca1_zdftke_config(kappa_convention="gaspar_sqrt2e")
    # One column.  e well above tke_background (1e-6) so gaspar's max() is
    # inert, and l_k modest so K lands between kappaM_min and kappaM_max.
    e = jnp.full((1, 8), 1.0e-3)
    l_k = jnp.full((1, 8), 2.0)
    N2 = jnp.full((1, 8), 1.0e-5)
    sh2 = jnp.full((1, 8), 1.0e-5)
    KM_nemo, KH_nemo = compute_K_from_tke(e, l_k, c_nemo, N2=N2, shear_sq=sh2)
    KM_leg, KH_leg = compute_K_from_tke(e, l_k, c_legacy, N2=N2, shear_sq=sh2)
    ratio = float(jnp.max(KM_leg / KM_nemo))
    assert ratio == pytest.approx(math.sqrt(2.0), rel=1e-6), (
        f"legacy/NEMO amplitude ratio {ratio} is not sqrt(2) — either a floor "
        "or the ceiling is binding in this fixture (so the test proves "
        "nothing), or the amplitude branch changed.")
    # The TRACER coefficient is what sets entrainment, and it runs through the
    # Richardson Prandtl chain and an INDEPENDENT kappaH_min floor -- so the
    # momentum ratio above does not imply it (codex 9400815 #5).  Assert it.
    ratio_H = float(jnp.max(KH_leg / KH_nemo))
    assert ratio_H == pytest.approx(math.sqrt(2.0), rel=1e-6), (
        f"legacy/NEMO TRACER ratio {ratio_H} is not sqrt(2); the Prandtl "
        "chain or the kappaH_min floor is intercepting the amplitude.")
    # And the fixture must sit strictly inside every clamp, else the checks
    # above could pass for the wrong reason.
    assert float(jnp.min(KM_nemo)) > c_nemo.kappaM_min
    assert float(jnp.max(KM_leg)) < c_nemo.kappaM_max
    assert float(jnp.min(KH_nemo)) > c_nemo.kappaH_min


def test_the_nn_mxl_length_really_carries_the_sqrt2():
    """The OTHER half of the double-count argument, which the ratio test
    ASSUMES rather than shows (codex 9400815 #5: the fixture above supplies
    ``l_k`` directly, so it cannot prove the length already has the sqrt(2)).

    Drive the model's own ``compute_mixing_lengths`` on the choice-3 branch in
    a regime where neither the |dl/dz|<=e3t sweeps nor the surface anchor can
    bind — huge cells, huge anchor — so the returned length must be the raw
    buoyancy length.  NEMO's is ``SQRT(2*en/rn2)`` (zdftke.F90:651); if ours
    were ``sqrt(en)/N`` instead, the amplitude's sqrt(2) would NOT be a
    double-count and the pin would be wrong.
    """
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing.tke import compute_mixing_lengths

    cfg = _core2().orca1_zdftke_config()
    assert cfg.tke_mxl_choice in (3, 4), (
        "the card no longer selects a NEMO nn_mxl branch; this test targets "
        "the branch whose length is claimed to carry the sqrt(2).")
    nlev = 6
    e_val, n2_val = 1.0e-3, 1.0e-5
    e = jnp.full((1, nlev - 1), e_val)
    N2 = jnp.full((1, nlev - 1), n2_val)
    dz_half = jnp.full(nlev - 1, 1.0e6)          # sweeps cannot bind
    dz_cell = jnp.full((1, nlev), 1.0e6)
    anchor = jnp.full((1,), 1.0e6)               # surface seed cannot bind
    l_k, _ = compute_mixing_lengths(e, N2, dz_half, cfg, signed_n2=False,
                                    dz_cell=dz_cell, boundary_cap=None,
                                    l_surface_anchor=anchor)
    expected_nemo = math.sqrt(2.0 * e_val) / math.sqrt(n2_val)
    got = float(jnp.max(l_k))
    assert got == pytest.approx(expected_nemo, rel=1e-9), (
        f"buoyancy length {got} != NEMO's sqrt(2*en/rn2) = {expected_nemo}; "
        "if it equals sqrt(en)/N instead, the amplitude sqrt(2) is NOT a "
        "double-count and the card pin must be reverted.")
    # Stated as the ratio the whole argument rests on.
    assert got / (math.sqrt(e_val) / math.sqrt(n2_val)) == pytest.approx(
        math.sqrt(2.0), rel=1e-9)

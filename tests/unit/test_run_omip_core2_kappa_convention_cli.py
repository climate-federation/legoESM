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
    K_nemo, _ = compute_K_from_tke(e, l_k, c_nemo, N2=N2, shear_sq=sh2)
    K_leg, _ = compute_K_from_tke(e, l_k, c_legacy, N2=N2, shear_sq=sh2)
    ratio = float(jnp.max(K_leg / K_nemo))
    assert ratio == pytest.approx(math.sqrt(2.0), rel=1e-6), (
        f"legacy/NEMO amplitude ratio {ratio} is not sqrt(2) — either a floor "
        "or the ceiling is binding in this fixture (so the test proves "
        "nothing), or the amplitude branch changed.")
    # And the fixture must sit strictly inside the clamps, else the check above
    # could pass for the wrong reason.
    assert float(jnp.min(K_nemo)) > c_nemo.kappaM_min
    assert float(jnp.max(K_leg)) < c_nemo.kappaM_max

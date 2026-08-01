"""Direct tests for the gSAM-deck -> SCM bridge.

The bridge is what makes the LES-vs-SCM turbulence comparison controlled, so
the two conversions that are easy to get backwards (absolute-T vs
potential-T tendencies, and the surface heat flux NOT being a theta flux on
the SCM side) are pinned here against the deck values they came from.

Requires the cached gSAM decks under ``data/les_cases/``; skipped when absent.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from legoesm.atmosphere.forcing.sam_case_forcing import (  # noqa: E402
    read_sam_lsf,
    read_sam_sfc,
    read_sam_snd,
    resolve_sam_case_dir,
)
from legoesm.atmosphere.forcing.scm.sam_case_scm import (  # noqa: E402
    SAM_SCM_CASES,
    load_sam_scm_case,
    surface_kinematic_moisture_flux,
    surface_kinematic_temperature_flux,
)

from legoesm import constants  # noqa: E402


def _deck_available(case: str) -> bool:
    try:
        resolve_sam_case_dir(SAM_SCM_CASES[case].gsam_dir)
    except Exception:
        return False
    return True


requires_bomex = pytest.mark.skipif(
    not _deck_available("bomex"), reason="BOMEX gSAM deck not cached"
)


# --- dispatch hardening -----------------------------------------------------

def test_unknown_case_raises():
    with pytest.raises(ValueError, match="Unknown SAM SCM case"):
        load_sam_scm_case("not_a_case")


@pytest.mark.parametrize("bad", [{"nlev": 1}, {"sigma_top": 0.0},
                                 {"sigma_top": 1.0}, {"sigma_top": -0.1}])
@requires_bomex
def test_invalid_grid_args_raise(bad):
    with pytest.raises(ValueError):
        load_sam_scm_case("bomex", **bad)


# --- flux conversions: the trap this module exists to document --------------

def test_surface_temperature_flux_has_no_exner_factor():
    """w_th_s is consumed as an ABSOLUTE-T flux, so SHF/(rho*c_pd), no Exner.

    Guards against copying the LES conversion, which carries an extra
    /Exner_sfc because a height-coordinate LES prognoses theta.
    """
    shf, rho = 8.0, 1.15
    got = surface_kinematic_temperature_flux(shf, rho)
    assert got == pytest.approx(shf / (rho * constants.c_pd), rel=1e-12)
    # and it is NOT the theta-flux form
    exner_sfc = (1.015e5 / constants.p_ref) ** constants.kappa
    theta_flux = shf / (rho * constants.c_pd * exner_sfc)
    assert not np.isclose(got, theta_flux, rtol=1e-6)


def test_surface_moisture_flux_conversion():
    lhf, rho = 130.0, 1.15
    assert surface_kinematic_moisture_flux(lhf, rho) == pytest.approx(
        lhf / (rho * constants.L_v), rel=1e-12)


@pytest.mark.parametrize("rho", [0.0, -1.0, float("nan")])
def test_flux_conversions_reject_bad_density(rho):
    with pytest.raises(ValueError):
        surface_kinematic_temperature_flux(10.0, rho)
    with pytest.raises(ValueError):
        surface_kinematic_moisture_flux(10.0, rho)


# --- column construction ----------------------------------------------------

@requires_bomex
def test_bomex_column_is_top_to_bottom_and_physical():
    case = load_sam_scm_case("bomex", nlev=48)

    assert case.z_full.shape == (48,) and case.p_full.shape == (48,)
    # top-to-bottom: height decreases, pressure increases
    assert np.all(np.diff(case.z_full) < 0.0)
    assert np.all(np.diff(case.p_full) > 0.0)
    assert case.z_full[-1] < case.z_full[0]
    # p_full[-1] is the lowest FULL level, which sits above the surface, so it
    # is strictly below p_s (sigma_full[-1] < sigma_half[-1] == 1). Asserting
    # equality here was the test being wrong, not the column.
    assert case.p_full[-1] < case.p_s
    assert case.p_full[-1] > 0.985 * case.p_s      # within one layer of ground
    assert case.z_full[-1] > 0.0

    assert np.all(np.isfinite(case.T_profile))
    assert np.all((case.T_profile > 150.0) & (case.T_profile < 340.0))
    assert np.all(case.q_v_profile >= -1e-12)
    assert case.q_v_profile[-1] > 1.0e-3          # moist trade-wind sub-cloud
    assert np.all(np.isfinite(case.u_profile))
    assert np.all(np.isfinite(case.v_profile))
    assert 1.0e5 < case.p_s < 1.05e5


@requires_bomex
def test_les_mask_selects_the_les_domain_only():
    case = load_sam_scm_case("bomex", nlev=64)
    mask = case.les_mask()
    assert mask.dtype == bool and mask.shape == (64,)
    assert mask.sum() >= 4, "too few SCM levels inside the LES domain"
    assert np.all(case.z_full[mask] <= case.les_domain_top_m)
    assert np.all(case.z_full[~mask] > case.les_domain_top_m)
    # the mask is contiguous at the bottom of a top-to-bottom column
    assert mask[-1] and not mask[0]


@requires_bomex
def test_temperature_is_theta_times_exner_of_the_scm_pressure():
    """IC temperature must be consistent with the SCM's OWN sigma pressure."""
    case = load_sam_scm_case("bomex", nlev=40)
    exner = (case.p_full / constants.p_ref) ** constants.kappa
    theta = case.T_profile / exner
    # theta must increase upward => decrease with index reversed... in a
    # top-to-bottom column theta DEcreases with index (warmer aloft).
    assert theta[0] > theta[-1], "stratosphere must be warmer in theta"
    assert np.all(np.isfinite(theta))


# --- forcing ----------------------------------------------------------------

@requires_bomex
def test_bomex_uses_prescribed_fluxes_matching_the_deck():
    case = load_sam_scm_case("bomex", nlev=32)
    assert case.forcing.prescribe == "fluxes"
    assert case.forcing.w_th_s is not None and case.forcing.w_qv_s is not None

    sfc = read_sam_sfc(f"{case.case_dir}/sfc")
    shf0, lhf0 = float(sfc.shf[0]), float(sfc.lhf[0])
    assert shf0 != 0.0 and lhf0 != 0.0, "BOMEX deck should prescribe fluxes"

    got_T = float(case.forcing.w_th_s(0.0))
    got_q = float(case.forcing.w_qv_s(0.0))
    assert got_T == pytest.approx(shf0 / (case.rho_sfc * constants.c_pd), rel=1e-9)
    assert got_q == pytest.approx(lhf0 / (case.rho_sfc * constants.L_v), rel=1e-9)
    assert got_T > 0.0 and got_q > 0.0        # upward = surface heats/moistens


@requires_bomex
def test_theta_adv_round_trips_to_the_deck_absolute_T_tendency():
    """theta_adv * Exner must reproduce the deck's dT/dt column."""
    case = load_sam_scm_case("bomex", nlev=32)
    from legoesm.atmosphere.forcing.sam_case_forcing import (
        interp_forcing_to_levels,
    )
    lsf = read_sam_lsf(f"{case.case_dir}/lsf")
    deck = interp_forcing_to_levels(lsf, case.z_full, day=float(lsf.days[0]))

    exner = (case.p_full / constants.p_ref) ** constants.kappa
    theta_adv = np.asarray(case.forcing.theta_adv(0.0))
    np.testing.assert_allclose(theta_adv * exner, deck["T_adv"], rtol=1e-9,
                               atol=1e-18)
    # and it is genuinely different from the raw deck value (Exner != 1)
    assert not np.allclose(theta_adv, deck["T_adv"], rtol=1e-6)


@requires_bomex
def test_subsidence_is_negative_and_positive_up_convention():
    """BOMEX has large-scale SUBSIDENCE; SCMForcing.subsidence_w is positive-up,
    so the profile must be negative in the subsiding layer."""
    case = load_sam_scm_case("bomex", nlev=48)
    w = np.asarray(case.forcing.subsidence_w(0.0))
    assert w.shape == (48,)
    inside = case.les_mask()
    assert np.min(w[inside]) < 0.0, "expected subsidence (w_ls < 0)"


@requires_bomex
def test_forcing_callables_are_jax_traceable():
    """The callables run inside the traced physics step: no NumPy host reads."""
    case = load_sam_scm_case("bomex", nlev=24)
    for fn in (case.forcing.subsidence_w, case.forcing.theta_adv,
               case.forcing.qv_adv, case.forcing.u_geo, case.forcing.v_geo):
        assert fn is not None
        out = jax.jit(fn)(0.0)
        assert out.shape == (24,)
        assert np.all(np.isfinite(np.asarray(out)))
    for fn in (case.forcing.w_th_s, case.forcing.w_qv_s):
        if fn is not None:
            assert np.isfinite(float(jax.jit(fn)(0.0)))


@requires_bomex
def test_coriolis_matches_latitude_and_can_be_disabled():
    case = load_sam_scm_case("bomex", nlev=16)
    expect = 2.0 * constants.Omega * np.sin(np.deg2rad(case.latitude_deg))
    assert case.forcing.f_c == pytest.approx(expect, rel=1e-12)
    assert load_sam_scm_case("bomex", nlev=16, coriolis=False).forcing.f_c == 0.0


@requires_bomex
def test_deck_sounding_drives_the_column_not_a_hardcoded_profile():
    """Sanity: the loaded column must actually track the deck sounding."""
    case = load_sam_scm_case("bomex", nlev=64)
    snd = read_sam_snd(f"{case.case_dir}/snd")
    inside = case.les_mask()
    exner = (case.p_full / constants.p_ref) ** constants.kappa
    theta = case.T_profile / exner
    deck_theta = np.interp(case.z_full[inside], snd.z, snd.theta)
    np.testing.assert_allclose(theta[inside], deck_theta, rtol=2e-3)


@requires_bomex
def test_repeat_loads_are_identical():
    """Every turbulence scheme must be handed a byte-identical column."""
    a = load_sam_scm_case("bomex", nlev=40)
    b = load_sam_scm_case("bomex", nlev=40)
    for name in ("T_profile", "q_v_profile", "u_profile", "v_profile",
                 "z_full", "p_full"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.p_s == b.p_s and a.rho_sfc == b.rho_sfc


@pytest.mark.parametrize("case_name", sorted(SAM_SCM_CASES))
def test_every_registered_case_loads(case_name):
    if not _deck_available(case_name):
        pytest.skip(f"{case_name} deck not cached")
    case = load_sam_scm_case(case_name, nlev=32)
    assert case.name == case_name
    assert np.all(np.isfinite(case.T_profile))
    assert case.forcing.prescribe in ("fluxes", "T_s")
    assert case.les_mask().sum() >= 3

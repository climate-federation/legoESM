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
from legoesm.atmosphere.physics._shared import exner_function  # noqa: E402


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
    exner_sfc = float(exner_function(1.015e5))
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
def test_column_top_is_the_les_domain_top_not_the_sounding_top():
    """Levels above the LES lid would still set the SCM's upper boundary and
    change the gradients feeding the scored levels, so masking them from the
    score is not enough — the column must not extend there at all."""
    case = load_sam_scm_case("bomex", nlev=64)
    snd_top = 4000.0            # BOMEX sounding reaches 4 km
    assert case.les_domain_top_m == 3000.0
    assert case.z_full[0] <= case.les_domain_top_m + 1.0
    assert case.z_full[0] < snd_top - 500.0, "column still spans the sounding"
    mask = case.les_mask()
    assert mask.dtype == bool and mask.shape == (64,)
    # Scoring stops at the SPONGE BASE, above which the LES relaxes toward a
    # reference; the column itself still spans the full domain.
    assert case.les_score_top_m == pytest.approx(
        case.spec.les_sponge_frac * case.les_domain_top_m)
    assert 0 < mask.sum() < 64, "sponge region must be excluded from scoring"
    assert np.all(case.z_full[mask] <= case.les_score_top_m)


@requires_bomex
def test_les_mask_excludes_levels_above_an_explicit_taller_column():
    """With an explicit taller column the mask must still exclude the top."""
    case = load_sam_scm_case("bomex", nlev=64, sigma_top=0.645)
    mask = case.les_mask()
    assert np.all(case.z_full[mask] <= case.les_score_top_m)
    assert np.all(case.z_full[~mask] > case.les_score_top_m)
    assert mask[-1] and not mask[0]


@requires_bomex
def test_temperature_is_theta_times_exner_of_the_scm_pressure():
    """IC temperature must be consistent with the SCM's OWN sigma pressure."""
    case = load_sam_scm_case("bomex", nlev=40)
    exner = np.asarray(exner_function(case.p_full))
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

    exner = np.asarray(exner_function(case.p_full))
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
def test_coriolis_matches_the_les_driver_not_the_latitude():
    """The cases HARDCODE f. Deriving it from latitude gave DYCOMS
    2*Omega*sin(31.5) = 7.62e-5 against the driver's 3.76e-5, a 2.03x error
    the tuner would have charged to turbulence parameters."""
    case = load_sam_scm_case("bomex", nlev=16)
    assert case.forcing.f_c == pytest.approx(case.spec.les_f_c, rel=1e-12)
    derived = 2.0 * constants.Omega * np.sin(np.deg2rad(case.latitude_deg))
    dyc = SAM_SCM_CASES["dycoms"]
    dyc_derived = 2.0 * constants.Omega * np.sin(np.deg2rad(31.5))
    assert not np.isclose(dyc_derived, dyc.les_f_c, rtol=0.1), (
        "latitude-derived f must differ from the driver's, or this guard is "
        "vacuous")
    del derived
    assert load_sam_scm_case("bomex", nlev=16, coriolis=False).forcing.f_c == 0.0


@requires_bomex
def test_deck_sounding_drives_the_column_not_a_hardcoded_profile():
    """Sanity: the loaded column must actually track the deck sounding."""
    case = load_sam_scm_case("bomex", nlev=64)
    snd = read_sam_snd(f"{case.case_dir}/snd")
    inside = case.les_mask()
    exner = np.asarray(exner_function(case.p_full))
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


# --- adversarial-review fixes ----------------------------------------------

@requires_bomex
def test_height_mapping_uses_virtual_potential_temperature():
    """A dry mapping displaces heights by ~0.608*q_v (~1% in BOMEX's moist
    sub-cloud layer) against an LES whose reference state is built from
    theta_v. Compare the shipped mapping to a deliberately dry one."""
    import jax.numpy as jnp
    from legoesm.grids.vertical import create_stretched_height_coordinate
    from legoesm.atmosphere.physics._shared import exner_to_pressure
    from legoesm.atmosphere.forcing.scm import sam_case_scm as mod

    case = load_sam_scm_case("bomex", nlev=48)
    snd = read_sam_snd(f"{case.case_dir}/snd")
    z_top = float(np.max(np.asarray(snd.z)))
    z_snd = jnp.asarray(np.asarray(snd.z, dtype=np.float64))
    theta_dry = jnp.asarray(np.asarray(snd.theta, dtype=np.float64))
    hc = create_stretched_height_coordinate(
        mod._AUX_LEVELS, H=z_top, dz_sfc=0.5 * z_top / mod._AUX_LEVELS,
        theta_ref_fn=lambda z: jnp.interp(z, z_snd, theta_dry),
        p_sfc=case.p_s,
    )
    # same staggering as the shipped mapping (interfaces)
    p_dry = np.asarray(exner_to_pressure(
        getattr(hc, "exner_ref_half", hc.exner_ref)), dtype=np.float64)
    z_aux, p_moist = mod._deck_pressure_profile(snd, case.p_s)
    # the two mappings must differ measurably where the air is moist
    rel = np.abs(p_moist - p_dry) / p_dry
    assert rel.max() > 1.0e-4, "virtual correction had no effect"
    assert rel.max() < 5.0e-2, "virtual correction implausibly large"


@requires_bomex
def test_rho_sfc_uses_virtual_temperature():
    """A dry rho is ~1% high in a moist sub-cloud layer, which would make BOTH
    prescribed kinematic fluxes ~1% too small from the first step."""
    from legoesm import constants
    case = load_sam_scm_case("bomex", nlev=48)
    dry = case.p_s / (constants.R_d * float(case.T_profile[-1]))
    assert case.rho_sfc < dry, "rho_sfc is not the virtual (moist) density"
    assert abs(case.rho_sfc - dry) / dry > 1.0e-3


def test_surface_mode_is_declared_per_case_not_inferred_from_the_deck():
    """RICO's deck carries H=15/LE=115 W/m2 but its LES IGNORES them and uses
    interactive bulk fluxes over a fixed SST, so a flux-magnitude heuristic
    would give RICO the wrong boundary condition."""
    assert SAM_SCM_CASES["rico"].surface_mode == "T_s"
    assert SAM_SCM_CASES["bomex"].surface_mode == "fluxes"
    assert SAM_SCM_CASES["dycoms"].surface_mode == "fluxes"


@pytest.mark.skipif(not _deck_available("rico"), reason="RICO deck not cached")
def test_rico_gets_interactive_surface_despite_nonzero_deck_fluxes():
    sfc = read_sam_sfc(f"{resolve_sam_case_dir('RICO')}/sfc")
    assert float(sfc.shf[0]) != 0.0 and float(sfc.lhf[0]) != 0.0
    case = load_sam_scm_case("rico", nlev=32)
    assert case.forcing.prescribe == "T_s"
    assert case.forcing.T_s is not None
    assert case.forcing.w_th_s is None and case.forcing.w_qv_s is None


@requires_bomex
def test_constant_surface_series_collapses_to_a_constant_callable():
    """Keeps a prescribed T_s off the traced path, and is cheaper."""
    import jax
    case = load_sam_scm_case("bomex", nlev=24)
    fn = case.forcing.w_th_s
    a, b = float(fn(0.0)), float(fn(1.0e6))
    assert a == b, "constant deck series must not vary in time"
    # a constant closure traces without touching jnp.interp
    assert np.isfinite(float(jax.jit(fn)(0.0)))


@pytest.mark.skipif(not _deck_available("rico"), reason="RICO deck not cached")
def test_rico_prescribed_T_s_is_constant_and_traceable():
    """The earlier constant-series test exercised BOMEX's w_th_s, not a
    prescribed T_s, so it did not cover the traced-T_s hazard at all.

    inject_prescribed_T_sfc_into_phys_state validates T_s by materialising it
    with numpy.asarray; inside a scanned rollout a traced T_s would raise. The
    constant collapse is what keeps it concrete.
    """
    import jax
    case = load_sam_scm_case("rico", nlev=24)
    assert case.forcing.prescribe == "T_s"
    fn = case.forcing.T_s
    a, b = float(fn(0.0)), float(fn(9.9e5))
    assert a == b, "RICO SST series must collapse to a constant"
    assert 280.0 < a < 310.0
    # concrete under jit: no tracer reaches the host-side validator
    assert np.isfinite(float(jax.jit(fn)(0.0)))
    import numpy as _np
    assert _np.asarray(fn(0.0)).shape == ()


@pytest.mark.skipif(not _deck_available("rico"), reason="RICO deck not cached")
def test_rico_carries_the_les_bulk_exchange_coefficients():
    """Otherwise closures are ranked on compensating a 32-37% flux error."""
    spec = SAM_SCM_CASES["rico"]
    assert spec.bulk_ch == pytest.approx(0.001094)
    assert spec.bulk_ce == pytest.approx(0.001133)
    assert SAM_SCM_CASES["bomex"].bulk_ch is None   # prescribed-flux case


@requires_bomex
def test_rho_sfc_does_not_depend_on_nlev():
    """rho_sfc sets the W/m2 -> kinematic flux conversion; if it followed the
    lowest SCM level, changing resolution would change the surface forcing
    even though the LES deck is unchanged."""
    a = load_sam_scm_case("bomex", nlev=24).rho_sfc
    b = load_sam_scm_case("bomex", nlev=96).rho_sfc
    assert a == pytest.approx(b, rel=1e-12), (a, b)


# --- DYCOMS-II RF02 ---------------------------------------------------------

requires_rf02 = pytest.mark.skipif(
    not _deck_available("rf02"), reason="DYCOMS_RF02 gSAM deck not cached"
)


@requires_rf02
def test_rf02_prescribes_the_deck_surface_fluxes():
    """SHF 16 / LHF 93 W/m^2, which are RF01's 15 / 115 in neither term."""
    case = load_sam_scm_case("rf02", nlev=32)
    assert case.forcing.prescribe == "fluxes"
    sfc = read_sam_sfc(f"{case.case_dir}/sfc")
    assert float(sfc.shf[0]) == pytest.approx(16.0)
    assert float(sfc.lhf[0]) == pytest.approx(93.0)
    w_th = float(case.forcing.w_th_s(0.0))
    w_qv = float(case.forcing.w_qv_s(0.0))
    assert w_th == pytest.approx(
        surface_kinematic_temperature_flux(16.0, case.rho_sfc), rel=1e-9)
    assert w_qv == pytest.approx(
        surface_kinematic_moisture_flux(93.0, case.rho_sfc), rel=1e-9)


@requires_rf02
def test_rf02_geostrophic_wind_is_sheared_not_uniform():
    """The RF02 spec's u_g = 3.0 + 4.3 z_km, v_g = -9.0 + 5.6 z_km. RF01's is
    a uniform (7, -5.5), so a uniform-wind assumption is a real mis-forcing
    here and not a cosmetic one."""
    case = load_sam_scm_case("rf02", nlev=48)
    z = np.asarray(case.z_full)
    u_g = np.asarray(case.forcing.u_geo(0.0))
    v_g = np.asarray(case.forcing.v_geo(0.0))
    inside = case.les_mask()
    np.testing.assert_allclose(u_g[inside], 3.0 + 4.3 * z[inside] / 1000.0,
                               atol=2e-3)
    np.testing.assert_allclose(v_g[inside], -9.0 + 5.6 * z[inside] / 1000.0,
                               atol=2e-3)
    assert np.ptp(u_g[inside]) > 1.0, "a uniform profile would pass vacuously"


@requires_rf02
def test_rf02_column_carries_the_deck_inversion_at_795_m():
    """theta_l steps 288.300 -> 296.710 K between 795 and 800 m in the deck."""
    case = load_sam_scm_case("rf02", nlev=96)
    z = np.asarray(case.z_full)
    theta = np.asarray(case.T_profile) / np.asarray(
        exner_function(case.p_full))
    below = (z > 100.0) & (z < 700.0)
    above = (z > 900.0) & (z < 1400.0)
    assert theta[below].max() - theta[below].min() < 0.5, "mixed layer"
    assert theta[above].min() - theta[below].max() > 6.0, "inversion jump"


@requires_rf02
def test_rf02_coriolis_is_the_latitude_value_unlike_rf01():
    """RF01's deck hardcodes fcor = 0.376e-4; RF02's does not and sets
    latitude0 = 31.5, so the two flights rotate at different rates."""
    case = load_sam_scm_case("rf02", nlev=16)
    derived = 2.0 * constants.Omega * np.sin(np.deg2rad(31.5))
    assert case.forcing.f_c == pytest.approx(derived, rel=1e-12)
    assert not np.isclose(case.forcing.f_c, SAM_SCM_CASES["dycoms"].les_f_c,
                          rtol=0.05)

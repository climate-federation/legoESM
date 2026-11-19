"""Specified-Nc droplet PSD must pair an IN-CLOUD number with IN-CLOUD water.

``compute_cloud_properties`` builds the M2005/SAM liquid effective radius from
the gamma PSD, ``reffc = (PGAM+3)/(2*LAMC)`` with
``LAMC ~ (N_c/q_c)^(1/3)`` (module_mp_graupel.f90:1691).  That ratio is only
meaningful if numerator and denominator live on the same footing:

* the prognostic ``N_c`` tracer is a GRID-MEAN number density and the ``q_c``
  reaching the optics is a grid-mean mixing ratio, so their ratio is already the
  in-cloud ratio and needs no cloud fraction;
* ``CloudConfig.Nc_default`` is NOT a grid mean.  It is SAM's specified
  IN-CLOUD droplet concentration (``dopredictNc=.false.``), exact in a CRM
  because a cell there is either fully cloudy or fully clear.  Pairing it with a
  grid-mean ``q_c`` inflates LAMC by ``cf^(-1/3)``, shrinking ``reffc`` by
  ``cf^(1/3)`` and inflating the in-cloud optical depth by ``cf^(-1/3)``.  The
  error is 1 for an overcast layer and unbounded as the layer breaks up, so it
  falls hardest on the subsidence regimes that should be the DARK end of the
  shortwave contrast.

The scheme therefore reconstructs the in-cloud condensate ``q_c/cf`` in the
SPECIFIED-Nc branch only, using the same ``_INHOM_CF_FLOOR`` the in-cloud water
path uses so both agree on what "in-cloud" means.

These tests pin the RELATIONSHIP, not recorded numbers:

* an overcast layer and a broken layer carrying the SAME in-cloud water get the
  same droplet radius and the same in-cloud optical depth (the reconstruction
  identity; this is what the pre-fix code violated by ``cf^(1/3)``);
* the specified-Nc answer equals the LIVE-tracer answer fed the reconstructed
  in-cloud water, so the two branches agree on the physics;
* a live prognostic tracer is untouched by the cloud fraction (no-op guard);
* the ice radius never moves, since ``N_i`` is a live grid-mean tracer already
  consistent with the grid-mean ``q_i``;
* ``n_cloud=None`` still returns the config constant;
* vanishing cover stays finite, positive and bounded by the LAMMIN/LAMMAX clip,
  in the forward pass and under ``jax.grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import pytest

from legoesm import constants
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    _CLOUD_R_EFF_MAX_M,
    _INHOM_CF_FLOOR,
    _TAU_GEOMETRIC_COEFF,
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig

NCOL, NLEV = 1, 6

# ``resolved`` keeps ``q_c`` EXACTLY the supplied ``q_cloud`` (the sub-grid
# condensate floor is gated to the diagnostic-fraction schemes), and the CLUBB
# level gate is disabled so ``cloud_fraction_override`` lands verbatim.  Both are
# needed to state the reconstruction identity as an equality rather than a
# tolerance on a floor-perturbed field.
CFG = CloudConfig(scheme="resolved", clubb_cf_override_p_min_pa=0.0,
                  clubb_cf_override_ramp_pa=0.0)

# SAM's specified-Nc Morrison leaves the N_c tracer slot at 0; the scheme keys
# the specified branch off ``n_cloud <= 1``, which no physical #/m^3 can hit.
SPECIFIED = 0.0
LIVE = 3.0e8            # 300 /cm^3, a polluted-continental live tracer


def _column(q_c, cf, n_cloud, q_i=0.0, n_ice=None, cfg=CFG):
    """One warm column at fixed p/T; ``q_c``, ``cf``, ``n_cloud`` are scalars."""
    sh = np.full((NCOL, NLEV), 1.0)
    kw = dict(
        T=jnp.full((NCOL, NLEV), 285.0),
        p_full=jnp.asarray(sh * 9.0e4),
        q_v=jnp.full((NCOL, NLEV), 6.0e-3),
        dp=jnp.asarray(sh * 1.0e4),
        q_cloud=jnp.asarray(sh * q_c),
        q_ice=jnp.asarray(sh * q_i),
        n_cloud=jnp.asarray(sh * n_cloud),
        cloud_fraction_override=jnp.asarray(sh * cf),
    )
    if n_ice is not None:
        kw["n_ice"] = jnp.asarray(sh * n_ice)
    return compute_cloud_properties(config=cfg, **kw)


def _reff(**kw):
    return float(_column(**kw).r_eff_liq[0, 0])


def _tau_incloud(props):
    """In-cloud optical depth, the same form the inhomogeneity optics uses."""
    cf = np.clip(np.asarray(props.cloud_fraction), _INHOM_CF_FLOOR, 1.0)
    lwp = np.asarray(props.lwp)
    r = np.asarray(props.r_eff_liq)
    return float((_TAU_GEOMETRIC_COEFF * (lwp / cf)
                  / (constants.rho_water * r))[0, 0])


# ------------------------------------------------- the reconstruction identity

@pytest.mark.parametrize("cf", [0.05, 0.15, 0.3, 0.5, 0.75, 0.9])
def test_broken_and_overcast_layers_with_equal_incloud_water_agree(cf):
    """The defining identity, and the one the pre-fix code broke.

    A layer of cover ``cf`` holding grid-mean ``cf*W`` holds exactly the same
    IN-CLOUD water ``W`` as an overcast layer holding ``W``.  Under a SPECIFIED
    in-cloud Nc the droplet PSD in the cloudy part is therefore identical, so
    both the effective radius and the in-cloud optical depth must match.  The
    old grid-mean pairing made the broken layer's radius smaller by
    ``cf^(1/3)`` and its in-cloud tau larger by ``cf^(-1/3)``.
    """
    W = 4.0e-4                                    # in-cloud LWC [kg/kg]
    broken = _column(q_c=cf * W, cf=cf, n_cloud=SPECIFIED)
    overcast = _column(q_c=W, cf=1.0, n_cloud=SPECIFIED)
    assert float(broken.r_eff_liq[0, 0]) == pytest.approx(
        float(overcast.r_eff_liq[0, 0]), rel=1e-12)
    assert _tau_incloud(broken) == pytest.approx(_tau_incloud(overcast), rel=1e-12)


def test_specified_nc_matches_the_live_tracer_on_the_reconstructed_water():
    """The two branches must describe the SAME PSD.

    Feeding the live-tracer branch ``Nc_default`` together with the in-cloud
    water ``q_c/cf`` in an overcast layer is the specified-Nc physics written
    out by hand; the specified branch must reproduce it exactly.
    """
    q_c = 3.0e-4
    for cf in (0.1, 0.35, 0.6, 1.0):
        got = _reff(q_c=q_c, cf=cf, n_cloud=SPECIFIED)
        hand = _reff(q_c=q_c / max(cf, _INHOM_CF_FLOOR), cf=1.0,
                     n_cloud=CFG.Nc_default)
        assert got == pytest.approx(hand, rel=1e-12), f"cf={cf}"


def test_radius_grows_as_the_layer_breaks_up():
    """Sign and exponent: at fixed GRID-MEAN water, less cover concentrates the
    same condensate into fewer droplets' worth of volume, so ``reffc`` rises as
    ``cf^(-1/3)``.  The pre-fix radius was independent of ``cf`` altogether,
    which this ordering assertion alone already rejects.
    """
    q_c = 3.0e-4
    cfs = np.array([1.0, 0.75, 0.5, 0.25, 0.1])
    r = np.array([_reff(q_c=q_c, cf=float(c), n_cloud=SPECIFIED) for c in cfs])
    assert np.all(np.diff(r) > 0.0), f"radius must rise as cover falls: {r}"
    np.testing.assert_allclose(r / r[0], cfs ** (-1.0 / 3.0), rtol=1e-10)


def test_incloud_tau_no_longer_grows_without_limit_as_cover_falls():
    """The radiative consequence: at fixed grid-mean water the in-cloud optical
    depth must scale as ``(q_c/cf)^(2/3)``, not ``(q_c/cf) * q_c^(-1/3)``.  The
    ratio of the two is ``cf^(-1/3)``, the spurious brightening of broken cloud.
    """
    q_c = 3.0e-4
    cfs = np.array([1.0, 0.5, 0.2, 0.05])
    tau = np.array([_tau_incloud(_column(q_c=q_c, cf=float(c), n_cloud=SPECIFIED))
                    for c in cfs])
    np.testing.assert_allclose(tau / tau[0], cfs ** (-2.0 / 3.0), rtol=1e-10)


# ------------------------------------------------------------- the no-op guards

@pytest.mark.parametrize("n_cloud", [1.0e7, 1.0e8, LIVE])
def test_live_tracer_radius_is_independent_of_cloud_fraction(n_cloud):
    """``N_c`` from microphysics is a GRID-MEAN density paired with a grid-mean
    ``q_c``: the ratio is already in-cloud, so dividing by ``cf`` would be a
    DOUBLE correction.  The radius must not move with the cover at all.
    """
    q_c = 3.0e-4
    ref = _reff(q_c=q_c, cf=1.0, n_cloud=n_cloud)
    for cf in (0.02, 0.2, 0.5, 0.999):
        assert _reff(q_c=q_c, cf=cf, n_cloud=n_cloud) == ref, f"cf={cf}"


def test_overcast_specified_nc_is_exactly_the_grid_mean_answer():
    """cf=1 divides by 1: the reconstruction must be an exact identity there,
    not a floating-point approximation, so overcast columns are unperturbed.
    """
    for q_c in (1.0e-5, 3.0e-4, 2.0e-3):
        assert (_reff(q_c=q_c, cf=1.0, n_cloud=SPECIFIED)
                == _reff(q_c=q_c, cf=1.0, n_cloud=CFG.Nc_default))


def test_ice_radius_never_sees_the_liquid_reconstruction():
    """``N_i`` is a live per-mass tracer paired with the grid-mean ``q_i`` and is
    therefore already consistent; the liquid fix must not touch it.  Toggling the
    droplet branch and the cover must leave ``r_eff_ice`` bitwise unchanged.
    """
    kw = dict(q_c=3.0e-4, q_i=5.0e-5, n_ice=3.0e4)
    ref = np.asarray(_column(cf=1.0, n_cloud=SPECIFIED, **kw).r_eff_ice)
    for cf in (0.02, 0.3, 0.8, 1.0):
        for nc in (SPECIFIED, 1.0, LIVE):
            got = np.asarray(_column(cf=cf, n_cloud=nc, **kw).r_eff_ice)
            np.testing.assert_array_equal(got, ref)


def test_absent_number_still_falls_back_to_the_config_constant():
    """``n_cloud=None`` (single-moment microphysics) must keep returning
    ``config.r_eff_liq`` at every cover, with no PSD evaluated at all.
    """
    cfg = CloudConfig(scheme="resolved", r_eff_liq=13.0e-6,
                      clubb_cf_override_p_min_pa=0.0,
                      clubb_cf_override_ramp_pa=0.0)
    sh = np.full((NCOL, NLEV), 1.0)
    for cf in (0.0, 0.05, 0.5, 1.0):
        props = compute_cloud_properties(
            config=cfg,
            T=jnp.full((NCOL, NLEV), 285.0),
            p_full=jnp.asarray(sh * 9.0e4),
            q_v=jnp.full((NCOL, NLEV), 6.0e-3),
            dp=jnp.asarray(sh * 1.0e4),
            q_cloud=jnp.asarray(sh * 3.0e-4),
            cloud_fraction_override=jnp.asarray(sh * cf),
        )
        np.testing.assert_array_equal(
            np.asarray(props.r_eff_liq), np.full((NCOL, NLEV), 13.0e-6))


# ------------------------------------------------------------ the vanishing end

@pytest.mark.parametrize("q_c", [0.0, 1.0e-20, 3.0e-4, 5.0e-2])
def test_vanishing_cover_is_finite_positive_and_clipped(q_c):
    """``q_c/cf`` is unbounded as cf -> 0, so the result must be held by the
    ``_INHOM_CF_FLOOR`` and the SAM LAMMIN/LAMMAX clip: never NaN, never inf,
    never negative, and never past the maximum radius the clip admits.
    """
    for cf in (0.0, 1.0e-30, 1.0e-12, 1.0e-6, _INHOM_CF_FLOOR, 1.0e-2):
        props = _column(q_c=q_c, cf=cf, n_cloud=SPECIFIED)
        r = np.asarray(props.r_eff_liq)
        assert np.all(np.isfinite(r)), f"cf={cf}, q_c={q_c}: {r}"
        assert np.all(r > 0.0), f"cf={cf}, q_c={q_c}: {r}"
        assert np.all(r <= _CLOUD_R_EFF_MAX_M), f"cf={cf}, q_c={q_c}: {r}"
        for f in ("cloud_fraction", "lwp", "iwp", "r_eff_ice"):
            assert np.all(np.isfinite(np.asarray(getattr(props, f)))), f


def test_radius_saturates_rather_than_diverging_below_the_floor():
    """Below ``_INHOM_CF_FLOOR`` the reconstruction is frozen, so the radius must
    PLATEAU at one value instead of running away with 1/cf.
    """
    q_c = 3.0e-4
    at_floor = _reff(q_c=q_c, cf=_INHOM_CF_FLOOR, n_cloud=SPECIFIED)
    for cf in (0.0, 1.0e-30, 1.0e-9, 0.5 * _INHOM_CF_FLOOR):
        assert _reff(q_c=q_c, cf=cf, n_cloud=SPECIFIED) == at_floor


def test_gradient_through_the_reconstruction_is_finite():
    """The division runs inside ``jax.grad`` during calibration; the cover and
    the condensate must both give finite gradients, including at cf=0.
    """
    sh = np.full((NCOL, NLEV), 1.0)
    base = dict(
        T=jnp.full((NCOL, NLEV), 285.0),
        p_full=jnp.asarray(sh * 9.0e4),
        q_v=jnp.full((NCOL, NLEV), 6.0e-3),
        dp=jnp.asarray(sh * 1.0e4),
        n_cloud=jnp.zeros((NCOL, NLEV)),
    )

    def loss_cf(cf):
        p = compute_cloud_properties(
            config=CFG, q_cloud=jnp.asarray(sh * 3.0e-4),
            cloud_fraction_override=cf, **base)
        return jnp.sum(p.r_eff_liq ** 2) + jnp.sum(p.lwp ** 2)

    def loss_qc(q_c):
        p = compute_cloud_properties(
            config=CFG, q_cloud=q_c,
            cloud_fraction_override=jnp.asarray(sh * 0.4), **base)
        return jnp.sum(p.r_eff_liq ** 2) + jnp.sum(p.lwp ** 2)

    for cf0 in (0.0, 1.0e-12, _INHOM_CF_FLOOR, 0.4, 1.0):
        g = jax.grad(loss_cf)(jnp.full((NCOL, NLEV), cf0))
        assert np.all(np.isfinite(np.asarray(g))), f"cf={cf0}: {g}"
    for q0 in (0.0, 1.0e-20, 3.0e-4):
        g = jax.grad(loss_qc)(jnp.full((NCOL, NLEV), q0))
        assert np.all(np.isfinite(np.asarray(g))), f"q_c={q0}: {g}"

"""Matched-kernel override for the SCM-RCE convection intercomparison.

Ranking convection schemes against a CRM is confounded if the schemes do not
share a vertical transport kernel: Bechtold / EDMF / Kain-Fritsch ship the
conservative ``"implicit_flux"`` mass-flux solve while Tiedtke / Emanuel /
Zhang-McFarlane / mass_flux ship the leaky ``"advective"`` one, so part of any
score gap measures the KERNEL rather than the scheme.
``run_scm_rce_campaign.apply_subsidence_solve_override`` is the single shared
selector that puts the whole mass-flux family on one solve; these are fast unit
checks of it (no CRM reference data or SCM integration required).
"""
from __future__ import annotations

import pytest

import scripts.run.run_scm_rce_campaign as campaign

# Schemes that DO own a mass-flux compensating-subsidence kernel, and so must
# accept the override.
MASS_FLUX_FAMILY = (
    "mass_flux", "edmf", "zhang_mcfarlane", "kain_fritsch",
    "tiedtke", "bechtold",
)
# Schemes the override CANNOT reach, for two different structural reasons --
# both must be REPORTED, never silently skipped:
#   * sbm / dca / kuo -- adjustment / Kuo-type closures with no
#     compensating-subsidence mass-flux kernel at all;
#   * emanuel -- its SHIPPED path (use_genuine_mixing=True) is a
#     buoyancy-sorting mixing matrix that never calls the shared kernel.
NON_MASS_FLUX = ("sbm", "dca", "kuo", "emanuel")

ALL_SCHEMES = MASS_FLUX_FAMILY + NON_MASS_FLUX


def _cfg(scheme):
    return campaign.make_physics_config(convection=scheme)


def test_campaign_sweep_covers_exactly_these_ten_schemes():
    """Guards the arm definitions against a scheme being added/renamed."""
    assert set(campaign.SCHEME_SWEEPS["convection"]) == set(ALL_SCHEMES)
    assert len(ALL_SCHEMES) == 10


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_as_shipped_is_a_true_no_op(scheme):
    cfg = _cfg(scheme)
    out, status = campaign.apply_subsidence_solve_override(cfg, "as_shipped")
    assert status == "as_shipped"
    assert out == cfg, f"{scheme}: as_shipped must not touch the config"


@pytest.mark.parametrize("scheme", MASS_FLUX_FAMILY)
def test_implicit_flux_is_applied_to_the_mass_flux_family(scheme):
    cfg = _cfg(scheme)
    out, status = campaign.apply_subsidence_solve_override(cfg, "implicit_flux")
    assert status == f"forced:{scheme}=implicit_flux"
    _component, active, subcfg = campaign._active_subconfig(out, "convection")
    assert active == scheme
    assert subcfg.subsidence_solve == "implicit_flux"


@pytest.mark.parametrize("scheme", NON_MASS_FLUX)
def test_non_mass_flux_schemes_report_not_applicable(scheme):
    """Never silently skipped -- the arm's status string must say why.

    ``emanuel`` is in this list ON PURPOSE: it IS a mass-flux-style scheme by
    name, but its production path bypasses the shared kernel, so claiming it
    was kernel-matched would be false.
    """
    cfg = _cfg(scheme)
    out, status = campaign.apply_subsidence_solve_override(cfg, "implicit_flux")
    assert status == f"not_applicable:{scheme}"
    assert out == cfg


@pytest.mark.parametrize("scheme", MASS_FLUX_FAMILY)
def test_advective_mode_is_also_selectable(scheme):
    """The symmetric arm: force the LEAKY solve on everyone (isolates the
    kernel's contribution in the other direction)."""
    cfg = _cfg(scheme)
    out, status = campaign.apply_subsidence_solve_override(cfg, "advective")
    assert status == f"forced:{scheme}=advective"
    _c, _s, subcfg = campaign._active_subconfig(out, "convection")
    assert subcfg.subsidence_solve == "advective"


def test_unknown_mode_raises():
    """Dispatch-hardening: a typo must not silently produce the as-shipped arm
    and get reported as the matched-kernel one."""
    with pytest.raises(ValueError, match="unknown mode"):
        campaign.apply_subsidence_solve_override(_cfg("tiedtke"), "implicitflux")


def test_override_is_non_vacuous_for_the_shipped_advective_schemes():
    """The four schemes threaded in this campaign genuinely START advective, so
    forcing implicit_flux is a real change (not a no-op that would make the
    primary/secondary tables identical)."""
    for scheme in ("tiedtke", "zhang_mcfarlane", "mass_flux"):
        cfg = _cfg(scheme)
        _c, _s, before = campaign._active_subconfig(cfg, "convection")
        assert before.subsidence_solve == "advective", (
            f"{scheme} no longer ships 'advective' -- the primary vs secondary "
            "kernel arms would collapse; update the campaign documentation"
        )
        after, _status = campaign.apply_subsidence_solve_override(
            cfg, "implicit_flux")
        _c2, _s2, after_sub = campaign._active_subconfig(after, "convection")
        assert after_sub.subsidence_solve == "implicit_flux"


def test_already_implicit_schemes_are_unchanged_by_the_primary_arm():
    """Bechtold/EDMF already default implicit_flux, so the PRIMARY arm must be
    a no-op for them -- i.e. their primary and secondary numbers should agree,
    which is a useful internal consistency check on the two tables."""
    for scheme in ("bechtold", "edmf"):
        cfg = _cfg(scheme)
        _c, _s, before = campaign._active_subconfig(cfg, "convection")
        assert before.subsidence_solve == "implicit_flux"
        after, status = campaign.apply_subsidence_solve_override(
            cfg, "implicit_flux")
        assert status == f"forced:{scheme}=implicit_flux"
        assert after == cfg, f"{scheme}: already implicit_flux, expected no-op"
